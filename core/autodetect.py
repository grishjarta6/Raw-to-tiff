"""
Автоопределение параметров распаковки ARRIRAW.

Метрика: corr(G1, G2). У правильного конфига > 0.9.

Возвращает словарь:
    header, i1, i2, assembly, pattern,
    unpack (человекочитаемая строка),
    corr_g1g2, corr_br, sp_avg, min_std, max_std, stds, score

Декодер сам решит, использовать decode_frame_pair или decode_frame,
по наличию i1/i2.
"""

import numpy as np

from .decoder import (ASSEMBLIES, PATTERNS, SIDE_FORMULAS,
                      split_channels, unpack_pair)


MIN_STD_ABS = 20.0
MIN_STD_RATIO = 0.05
GOOD_CORR = 0.85
WARN_CORR = 0.50


def _find_idx(expr: str) -> int:
    for i, (name, _) in enumerate(SIDE_FORMULAS):
        if name == expr:
            return i
    raise KeyError(expr)


KNOWN_PAIRS = {
    "arri_alt":     (_find_idx("(b0<<4)|(b2>>4)"), _find_idx("((b2&0x0F)<<8)|b1")),
    "std":          (_find_idx("(b0<<4)|(b1>>4)"), _find_idx("((b1&0x0F)<<8)|b2")),
    "big_endian":   (_find_idx("(b0<<4)|(b2>>4)"), _find_idx("(b2<<4)|(b1&0x0F)")),
    "little_endian":(_find_idx("((b1&0x0F)<<8)|b0"), _find_idx("((b1>>4)<<8)|b2")),
}


def _spatial_corr(img: np.ndarray) -> float:
    x = img.astype(np.float32)
    if x.shape[0] < 2 or x.shape[1] < 2:
        return 0.0
    hx = float(np.corrcoef(x[:, :-1].flatten(), x[:, 1:].flatten())[0, 1])
    hy = float(np.corrcoef(x[:-1, :].flatten(), x[1:, :].flatten())[0, 1])
    return (hx + hy) / 2


def _score(chans: dict) -> dict | None:
    stds = {k: float(v.std()) for k, v in chans.items()}
    mn, mx = min(stds.values()), max(stds.values())
    if mn < MIN_STD_ABS or mx < 1.0 or mn / mx < MIN_STD_RATIO:
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


def _viable_headers(essence_size, W, H, max_header=512):
    chunk = (W * H // 2) * 3 // 2
    needed = 2 * chunk
    return [h for h in range(max_header) if essence_size - h == needed]


def _eval_pair(raw, hdr, i1, i2, assembly, W, H, sample_H):
    """Возвращает лучший pattern-кандидат или None."""
    remaining = len(raw) - hdr
    if remaining <= 0:
        return None
    chunk_bytes = remaining // 2
    try:
        c1 = unpack_pair(raw[hdr : hdr + chunk_bytes], i1, i2)
        c2 = unpack_pair(raw[hdr + chunk_bytes : hdr + 2 * chunk_bytes], i1, i2)
        frame = ASSEMBLIES[assembly](c1, c2, W, H)
    except Exception:
        return None

    sample = frame[:sample_H, :]
    best = None
    for pattern in PATTERNS:
        chans = split_channels(sample, pattern)
        sc = _score(chans)
        if sc is None:
            continue
        cand = {
            "header":   hdr,
            "i1":       i1,
            "i2":       i2,
            "assembly": assembly,
            "pattern":  pattern,
            **sc,
        }
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


def _label(i1: int, i2: int) -> str:
    return f"{SIDE_FORMULAS[i1][0]} + {SIDE_FORMULAS[i2][0]}"


def _finalize(c: dict) -> dict:
    """Дописывает человекочитаемый unpack. Не трогает i1/i2."""
    i1, i2 = c.get("i1"), c.get("i2")
    if i1 is None or i2 is None:
        # не pair-декодер, оставляем как есть
        return c
    canon = None
    for name, (a, b) in KNOWN_PAIRS.items():
        if (i1, i2) == (a, b):
            canon = name
            break
    c["unpack"] = canon or _label(i1, i2)
    return c


def autodetect(filepath, pkt, W, H, sample_rows=300, verbose=False):
    essence_size = pkt["length"]
    sample_H = min(sample_rows, H)

    viable = _viable_headers(essence_size, W, H)
    print(f"   🔍 Жизнеспособные header для {W}×{H}: {viable}")
    if not viable:
        print(f"   ⚠ Ни один header в [0, 512) не подходит.")
        print(f"   ⚠ essence_size={essence_size:,}")
        return None

    with open(filepath, "rb") as f:
        f.seek(pkt["value_start"])
        raw = f.read(essence_size)

    # ---- Шаг 1: известные пары на top_bottom
    print(f"\n   🔍 Шаг 1: известные пары формул (top_bottom)...")

    best_known = None
    for name, (i1, i2) in KNOWN_PAIRS.items():
        for hdr in viable:
            cand = _eval_pair(raw, hdr, i1, i2, "top_bottom",
                              W, H, sample_H)
            if cand is None:
                continue
            if best_known is None or cand["score"] > best_known["score"]:
                best_known = cand
                print(f"      ✅ {name:<14} hdr={hdr} "
                      f"pattern={cand['pattern']:<4} "
                      f"corr_G={cand['corr_g1g2']:+.4f}")

    if best_known and best_known["corr_g1g2"] > GOOD_CORR:
        print(f"   ✅ corr_G > {GOOD_CORR} — используем "
              f"{_finalize(best_known)['unpack']}")
        return _finalize(best_known)

    # ---- Шаг 2: полный перебор пар (i1, i2) × assembly
    print(f"\n   🔍 Шаг 2: полный перебор 24×23 пар формул...")

    candidates = []
    n_pairs = 0
    for i1 in range(len(SIDE_FORMULAS)):
        for i2 in range(len(SIDE_FORMULAS)):
            if i1 == i2:
                continue
            n_pairs += 1
            for hdr in viable:
                for asm in ASSEMBLIES:
                    cand = _eval_pair(raw, hdr, i1, i2, asm,
                                      W, H, sample_H)
                    if cand is None:
                        continue
                    candidates.append(cand)

    print(f"      Проверено пар: {n_pairs} × "
          f"{len(viable)} hdr × {len(ASSEMBLIES)} asm = "
          f"{n_pairs * len(viable) * len(ASSEMBLIES)}")

    if not candidates:
        if best_known:
            return _finalize(best_known)
        return None

    candidates.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n   🏆 Топ-10 кандидатов:")
    print(f"      {'#':>3}  {'corr_G':>8}  {'corr_BR':>8}  "
          f"{'sp':>7}  {'min_std':>8}  {'hdr':>4}  "
          f"{'assembly':<18}  {'pattern':<6}  formulas")
    print("      " + "-" * 116)
    for i, c in enumerate(candidates[:10], 1):
        marker = " ★" if i == 1 else "  "
        print(f"      {i:>3}  {c['corr_g1g2']:>+8.4f}  {c['corr_br']:>+8.4f}  "
              f"{c['sp_avg']:>+7.3f}  {c['min_std']:>8.0f}  "
              f"{c['header']:>4}  {c['assembly']:<18}  "
              f"{c['pattern']:<6}  {_label(c['i1'], c['i2'])}{marker}")

    best = candidates[0]
    if best["corr_g1g2"] < WARN_CORR:
        print(f"\n   ⚠ corr_G = {best['corr_g1g2']:+.3f} < {WARN_CORR}")
        print(f"   ⚠ Уверенности нет:")
        print(f"      - файл HDE/сжат → нужен ARRI SDK")
        print(f"      - повреждён / тестовый")
        print(f"      - нестандартный формат")
        if best_known:
            print(f"   ℹ Откат на известную пару: "
                  f"{_finalize(best_known)['unpack']}")
            return _finalize(best_known)
        return _finalize(best)

    return _finalize(best)