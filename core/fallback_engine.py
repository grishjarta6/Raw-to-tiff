"""
Fallback-движок: интерпретирует essence как 2D Bayer-массив.

Работает с:
  - .mxf → читает essence через mxf_parser
  - .tif/.png → читает как изображение

Пробует разные режимы упаковки (12-бит × 4 формулы, 16-бит LE/BE, 8-бит)
и разные известные разрешения, автоматически определяет паттерн Bayer
по корреляции G1↔G2.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np


PATTERNS = {
    "GBRG": ("G1", "B",  "R",  "G2"),
    "GRBG": ("G1", "R",  "B",  "G2"),
    "BGGR": ("B",  "G2", "G1", "R"),
    "RGGB": ("R",  "G1", "G2", "B"),
}

KNOWN_RES = [
    (3424, 2202), (3200, 1800), (3168, 1776),
    (2880, 2160), (2880, 1620),
    (4448, 3096), (4096, 2304), (4608, 3164), (6560, 3100),
    (4096, 2160), (3840, 2160), (1920, 1080),
]

MAX_HEADER = 512


# ===========================================================================
# 12-бит распаковка: 3 байта → 2 пикселя
# ===========================================================================

def _unpack_12(data: bytes, kind: str) -> np.ndarray:
    n = len(data) // 3
    d = np.frombuffer(data[: n * 3], dtype=np.uint8)
    b0 = d[0::3].astype(np.uint16)
    b1 = d[1::3].astype(np.uint16)
    b2 = d[2::3].astype(np.uint16)

    if kind == "arri_alt":
        p1 = (b0 << 4) | (b2 >> 4)
        p2 = ((b2 & 0x0F) << 8) | b1
    elif kind == "std":
        p1 = (b0 << 4) | (b1 >> 4)
        p2 = ((b1 & 0x0F) << 8) | b2
    elif kind == "big_endian":
        p1 = (b0 << 4) | (b2 >> 4)
        p2 = (b1 << 4) | (b2 & 0x0F)
    elif kind == "little_endian":
        p1 = ((b1 & 0x0F) << 8) | b0
        p2 = ((b1 >> 4) << 8) | b2
    else:
        raise ValueError(kind)

    out = np.empty(len(b0) * 2, dtype=np.uint16)
    out[0::2] = p1
    out[1::2] = p2
    return out


# ===========================================================================
# Оценка качества
# ===========================================================================

def _spatial_corr(x: np.ndarray) -> float:
    x = x.astype(np.float32)
    if x.shape[0] < 2 or x.shape[1] < 2:
        return 0.0
    hx = float(np.corrcoef(x[:, :-1].flatten(), x[:, 1:].flatten())[0, 1])
    hy = float(np.corrcoef(x[:-1, :].flatten(), x[1:, :].flatten())[0, 1])
    return (hx + hy) / 2


def _split_bayer(arr: np.ndarray, pattern: str) -> dict:
    tl, tr, bl, br = PATTERNS[pattern]
    return {
        tl: arr[0::2, 0::2],
        tr: arr[0::2, 1::2],
        bl: arr[1::2, 0::2],
        br: arr[1::2, 1::2],
    }


def _score(chans: dict) -> dict | None:
    stds = {k: float(v.std()) for k, v in chans.items()}
    mn, mx = min(stds.values()), max(stds.values())
    if mn < 20.0 or mx < 1.0 or mn / mx < 0.05:
        return None

    g1 = chans["G1"].flatten().astype(np.float32)
    g2 = chans["G2"].flatten().astype(np.float32)
    b  = chans["B"].flatten().astype(np.float32)
    r  = chans["R"].flatten().astype(np.float32)

    corr_g  = float(np.corrcoef(g1, g2)[0, 1])
    corr_br = float(np.corrcoef(b, r)[0, 1])
    sp = (_spatial_corr(chans["B"]) + _spatial_corr(chans["G1"]) +
          _spatial_corr(chans["G2"]) + _spatial_corr(chans["R"])) / 4

    return {
        "corr_g1g2": corr_g,
        "corr_br":   corr_br,
        "sp_avg":    sp,
        "stds":      stds,
        "min_std":   mn,
        "max_std":   mx,
        "score":     corr_g * 1000.0 + sp * 10.0 + mn / 100.0,
    }


# ===========================================================================
# Ядро: пробуем все интерпретации essence
# ===========================================================================

def _candidate_from_pixels(pixels: np.ndarray, W: int, H: int,
                           mode: str, header: int,
                           verbose: bool) -> list[dict]:
    """Пробует все паттерны на 2D-массиве из pixels."""
    if pixels.size < W * H:
        return []
    try:
        img = pixels[: W * H].reshape(H, W)
    except Exception:
        return []

    out = []
    for pattern in PATTERNS:
        chans = _split_bayer(img, pattern)
        sc = _score(chans)
        if sc is None:
            continue
        out.append({
            "mode":   mode,
            "header": header,
            "W":      W,
            "H":      H,
            "pattern": pattern,
            "img":    img,
            "chans":  chans,
            **sc,
        })
    return out


def _try_essence(raw: bytes, verbose: bool = True) -> list[dict]:
    """
    Перебирает (header, mode, W, H) так, чтобы

        (len(raw) - header) точно соответствовал W*H пикселям.

    Для каждого варианта пробует 4 паттерна Bayer.
    """
    n = len(raw)
    cands: list[dict] = []

    for W, H in KNOWN_RES:
        total_px = W * H

        # --- 12-бит (3 байта → 2 пикселя)
        if (total_px * 3) % 2 == 0:
            bytes_12 = total_px * 3 // 2
            header = n - bytes_12
            if 0 <= header < MAX_HEADER:
                data = raw[header : header + bytes_12]
                for mode in ("arri_alt", "std", "big_endian", "little_endian"):
                    try:
                        px = _unpack_12(data, mode)
                    except Exception:
                        continue
                    cands += _candidate_from_pixels(
                        px, W, H, f"12_{mode}", header, verbose)

        # --- 16-бит LE
        bytes_16 = total_px * 2
        header = n - bytes_16
        if 0 <= header < MAX_HEADER:
            data = raw[header : header + bytes_16]
            try:
                px_le = np.frombuffer(data, dtype="<u2").copy()
                cands += _candidate_from_pixels(
                    px_le, W, H, "16_le", header, verbose)
            except Exception:
                pass
            try:
                px_be = np.frombuffer(data, dtype=">u2").astype(np.uint16)
                cands += _candidate_from_pixels(
                    px_be, W, H, "16_be", header, verbose)
            except Exception:
                pass

    return cands


# ===========================================================================
# Сохранение
# ===========================================================================

def _save(chans: dict, out_dir: Path, stem: str,
          save_rgb: bool = True) -> None:
    import tifffile

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / stem

    for name, img in chans.items():
        tifffile.imwrite(str(base) + f"_{name}.tiff", img)

    if save_rgb:
        R  = chans["R"].astype(np.float32)
        B  = chans["B"].astype(np.float32)
        G1 = chans["G1"].astype(np.float32)
        G2 = chans["G2"].astype(np.float32)
        G  = 0.5 * (G1 + G2)

        def norm(x):
            p = np.percentile(x, 99.0)
            return np.clip(x / max(p, 1), 0, 1)

        rgb = np.stack([norm(R), norm(G), norm(B)], axis=-1)
        tifffile.imwrite(str(base) + "_RGB.tiff",
                         (rgb * 255).astype(np.uint8))


# ===========================================================================
# Точка входа
# ===========================================================================

def _read_image_file(path: Path) -> tuple[np.ndarray | None, str]:
    """Для .tif/.png/.jpg — обычное чтение."""
    try:
        import tifffile
        return tifffile.imread(str(path)), "tifffile"
    except Exception:
        pass
    try:
        import cv2
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is not None:
            return arr, "cv2"
    except Exception:
        pass
    try:
        from PIL import Image
        return np.array(Image.open(str(path))), "Pillow"
    except Exception:
        pass
    return None, ""


def try_process(filepath: Path, out_dir: Path | None = None,
                save_rgb: bool = True) -> bool:
    """
    Fallback: 
      - .tif/.png → читает как изображение
      - .mxf/.ari → читает essence и интерпретирует как 2D Bayer
    """
    print(f"   🔄 Fallback-движок")

    raw_essence = None
    source_kind = ""

    # --- .mxf → essence
    if filepath.suffix.lower() in (".mxf", ".ari"):
        try:
            from .mxf_parser import parse as mxf_parse
            result = mxf_parse(filepath)
            pkt = result["essence"][0]
            with open(filepath, "rb") as f:
                f.seek(pkt["value_start"])
                raw_essence = f.read(pkt["length"])
            source_kind = "mxf"
            print(f"   ✅ Essence: {len(raw_essence):,} байт, "
                  f"кадров {len(result['essence'])}")
        except Exception as e:
            print(f"   ❌ Не удалось прочитать essence: {e}")
            return False

    # --- .tif/.png → изображение
    if raw_essence is None:
        arr, lib = _read_image_file(filepath)
        if arr is None:
            print(f"   ❌ Не удалось прочитать {filepath.name} "
                  f"ни как изображение, ни как MXF.")
            return False
        print(f"   ✅ Изображение ({lib}): shape={arr.shape}, "
              f"dtype={arr.dtype}")

        if arr.ndim != 2:
            print(f"   ❌ Ожидается 2D (Bayer), получено {arr.ndim}D.")
            return False
        if arr.dtype != np.uint16:
            arr = arr.astype(np.uint16)

        best = None
        for pattern in PATTERNS:
            chans = _split_bayer(arr, pattern)
            sc = _score(chans)
            if sc is None:
                continue
            if best is None or sc["score"] > best["score"]:
                best = {"pattern": pattern, "chans": chans, **sc}

        if best is None:
            print(f"   ❌ Паттерн Bayer не определился.")
            return False

        print(f"   🏆 pattern={best['pattern']} "
              f"corr(G1,G2)={best['corr_g1g2']:+.4f}")

        stem = filepath.stem
        if out_dir is None:
            out_dir = filepath.parent / f"{stem}_channels"
        _save(best["chans"], out_dir, stem, save_rgb)

        print(f"   ✅ Сохранено: {out_dir}")
        for name in ("B", "G1", "G2", "R"):
            img = best["chans"][name]
            print(f"      {name:<3}: {img.shape[1]}×{img.shape[0]}  "
                  f"std={img.std():.0f}  mean={img.mean():.0f}")
        return True

    # --- MXF essence → пробуем интерпретации
    print(f"   🔍 Пробую интерпретации essence...")
    cands = _try_essence(raw_essence, verbose=True)

    if not cands:
        print(f"   ❌ Не удалось интерпретировать essence "
              f"ни как 12-бит, ни как 16-бит.")
        return False

    cands.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n   🏆 Топ-10 интерпретаций:")
    print(f"      {'#':>3}  {'mode':<18} {'WxH':<12} {'hdr':>4}  "
          f"{'pattern':<6}  {'corr_G':>8}  {'sp':>7}  "
          f"{'min_std':>8}  {'score':>9}")
    print("      " + "-" * 100)
    for i, c in enumerate(cands[:10], 1):
        marker = " ★" if i == 1 else "  "
        print(f"      {i:>3}  {c['mode']:<18} {c['W']}×{c['H']:<8} "
              f"{c['header']:>4}  {c['pattern']:<6}  "
              f"{c['corr_g1g2']:>+8.4f}  {c['sp_avg']:>+7.3f}  "
              f"{c['min_std']:>8.0f}  {c['score']:>9.2f}{marker}")

    best = cands[0]
    if best["corr_g1g2"] < 0.5:
        print(f"\n   ⚠ corr_G={best['corr_g1g2']:+.3f} < 0.5")
        print(f"   ⚠ Уверенности нет. Возможные причины:")
        print(f"      - HDE (сжатый ARRIRAW) → нужен ARRI SDK")
        print(f"      - нестандартное разрешение (нет в KNOWN_RES)")

    stem = filepath.stem
    if out_dir is None:
        out_dir = filepath.parent / f"{stem}_channels"
    _save(best["chans"], out_dir, stem, save_rgb)

    print(f"\n   ✅ Сохранено: {out_dir}")
    for name in ("B", "G1", "G2", "R"):
        img = best["chans"][name]
        print(f"      {name:<3}: {img.shape[1]}×{img.shape[0]}  "
              f"std={img.std():.0f}  mean={img.mean():.0f}")
    return True