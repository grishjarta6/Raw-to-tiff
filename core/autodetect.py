"""
Автоопределение параметров распаковки ARRIRAW.

Стратегия:
  1. Проверяем known-good (header=76 + arri_alt + top_bottom) для всех
     pattern. Если лучший corr(G1,G2) > 0.9 — сразу возвращаем.
  2. Иначе полный перебор по viable header'ам.

Фильтр каналов — ОТНОСИТЕЛЬНЫЙ: min_std > MIN_STD_RATIO * max_std.
Это позволяет обрабатывать сцены с преобладающим цветом
(например, тёплые кадры, где синий канал имеет малый std).
"""

import numpy as np

from .decoder import UNPACK_FUNCS, ASSEMBLIES, PATTERNS, split_channels


MIN_STD_ABS = 20.0       # абсолютный минимум (защита от полного нуля)
MIN_STD_RATIO = 0.05     # min_std > 0.05 * max_std — все каналы «живые»
GOOD_CORR = 0.85


def _spatial_corr(img: np.ndarray) -> float:
    x = img.astype(np.float32)
    if x.shape[0] < 2 or x.shape[1] < 2:
        return 0.0
    hx = float(np.corrcoef(x[:, :-1].flatten(), x[:, 1:].flatten())[0, 1])
    hy = float(np.corrcoef(x[:-1, :].flatten(), x[1:, :].flatten())[0, 1])
    return (hx + hy) / 2


def _score_channels(chans: dict) -> dict | None:
    stds = {name: float(img.std()) for name, img in chans.items()}
    min_std = min(stds.values())
    max_std = max(stds.values())

    if min_std < MIN_STD_ABS:
        return None
    if max_std < 1.0 or min_std / max_std < MIN_STD_RATIO:
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
        "score":     corr_g * 1000.0 + sp * 10.0 + min_std / 100.0,
        "corr_g1g2": corr_g,
        "corr_br":   corr_br,
        "sp_avg":    sp,
        "stds":      stds,
        "min_std":   min_std,
        "max_std":   max_std,
        "mean_R":    float(r.mean()),
        "mean_B":    float(b.mean()),
    }


def _viable_headers(essence_size: int, W: int, H: int,
                    max_header: int = 512) -> list[int]:
    """
    Все header'ы в [0, max_header), при которых

        (essence_size - header) == 2 * ((W*H/2) * 3/2)

    Для 3424×2202 / essence 11 309 548 → [76]
    Для 3424×2202 / essence 11 309 472 → [0]
    Для 4448×3096 / essence 20 656 552 → [40]
    """
    chunk_pixels = (W * H) // 2
    chunk_bytes = chunk_pixels * 3 // 2
    needed = 2 * chunk_bytes
    return [h for h in range(0, max_header) if essence_size - h == needed]


def _evaluate_one(raw, essence_size, hdr, unpack_mode, assembly,
                  W, H, sample_H, debug: bool = False) -> list[dict]:
    remaining = essence_size - hdr
    if remaining <= 0:
        return []
    chunk_bytes = remaining // 2

    try:
        c1 = UNPACK_FUNCS[unpack_mode](raw[hdr : hdr + chunk_bytes])
        c2 = UNPACK_FUNCS[unpack_mode](
            raw[hdr + chunk_bytes : hdr + 2 * chunk_bytes])
    except Exception as e:
        if debug:
            print(f"      ⚠ unpack {unpack_mode}: {e}")
        return []

    try:
        frame = ASSEMBLIES[assembly](c1, c2, W, H)
    except Exception as e:
        if debug:
            print(f"      ⚠ assembly {assembly}: {e}")
        return []

    sample = frame[:sample_H, :]
    out = []
    for pattern in PATTERNS:
        chans = split_channels(sample, pattern)
        sc = _score_channels(chans)
        if sc is None:
            if debug:
                stds = {k: float(v.std()) for k, v in chans.items()}
                print(f"      ⚠ {unpack_mode}/{assembly}/{pattern}: "
                      f"stds={ {k: round(v,1) for k,v in stds.items()} }")
            continue
        out.append({
            "header":   hdr,
            "unpack":   unpack_mode,
            "assembly": assembly,
            "pattern":  pattern,
            **sc,
        })
    return out


def autodetect(filepath, pkt, W: int, H: int,
               sample_rows: int = 500,
               verbose: bool = False) -> dict | None:
    essence_size = pkt["length"]
    sample_H = min(sample_rows, H)

    print(f"\n   🔧 Доступные unpack: {list(UNPACK_FUNCS)}")

    viable = _viable_headers(essence_size, W, H)
    print(f"   🔍 Жизнеспособные header для {W}×{H}: {viable}")
    if not viable:
        print(f"   ⚠ Ни один header в [0, 512) не подходит для {W}×{H}.")
        print(f"   ⚠ essence_size={essence_size:,}")
        print(f"   ⚠ Задайте разрешение вручную ([p] в меню).")
        return None

    with open(filepath, "rb") as f:
        f.seek(pkt["value_start"])
        raw = f.read(essence_size)

    # ===============================================================
    # ШАГ 1. Known-good
    # ===============================================================
    print(f"\n   🔍 Шаг 1: arri_alt + top_bottom (known-good)...")

    known_good: list[dict] = []
    for hdr in viable:
        cands = _evaluate_one(raw, essence_size, hdr,
                              "arri_alt", "top_bottom",
                              W, H, sample_H, debug=True)
        known_good.extend(cands)

    if known_good:
        known_good.sort(key=lambda c: c["corr_g1g2"], reverse=True)
        print(f"   ✅ Получено {len(known_good)} кандидатов:")
        for c in known_good:
            print(f"      pattern={c['pattern']:<6}  "
                  f"corr_G={c['corr_g1g2']:+.4f}  "
                  f"corr_BR={c['corr_br']:+.4f}  "
                  f"sp={c['sp_avg']:+.4f}  "
                  f"min_std={c['min_std']:.0f}")

        best = known_good[0]
        if best["corr_g1g2"] > GOOD_CORR:
            print(f"   ✅ corr_G = {best['corr_g1g2']:+.4f} > "
                  f"{GOOD_CORR} — используем {best['pattern']}")
            return best
        print(f"   ⚠ corr_G = {best['corr_g1g2']:+.4f} < {GOOD_CORR}, "
              f"продолжаю поиск")
    else:
        print(f"   ⚠ known-good не дал кандидатов")

    # ===============================================================
    # ШАГ 2. Полный перебор
    # ===============================================================
    print(f"\n   🔍 Шаг 2: полный перебор...")

    candidates: list[dict] = []
    for hdr in viable:
        for unpack_mode in UNPACK_FUNCS:
            for assembly in ASSEMBLIES:
                candidates.extend(
                    _evaluate_one(raw, essence_size, hdr, unpack_mode,
                                  assembly, W, H, sample_H))

    if not candidates:
        if known_good:
            return known_good[0]
        return None

    candidates.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n   🏆 Топ-5 кандидатов:")
    print(f"      {'#':>3}  {'hdr':>4}  {'unpack':<15} "
          f"{'assembly':<18} {'pattern':<6}  "
          f"{'corr_G':>8}  {'corr_BR':>8}  {'sp':>7}  "
          f"{'min_std':>8}  {'max_std':>8}")
    print("      " + "-" * 110)
    for i, c in enumerate(candidates[:5], 1):
        marker = " ★" if i == 1 else "  "
        print(f"      {i:>3}  {c['header']:>4}  {c['unpack']:<15} "
              f"{c['assembly']:<18} {c['pattern']:<6}  "
              f"{c['corr_g1g2']:>+8.4f}  {c['corr_br']:>+8.4f}  "
              f"{c['sp_avg']:>+7.3f}  {c['min_std']:>8.0f}  "
              f"{c['max_std']:>8.0f}{marker}")

        best = candidates[0]
    if best["corr_g1g2"] < 0.5:
        print(f"\n   ⚠ Даже лучший кандидат corr(G1,G2) = "
              f"{best['corr_g1g2']:+.3f} < 0.5")
        print(f"   ⚠ Параметры распаковки подобраны неуверенно.")
        print(f"   ⚠ Возможные причины:")
        print(f"      - файл в формате HDE (сжат) — нужен ARRI SDK")
        print(f"      - пустой / тестовый кадр")
        print(f"      - разрешение определено неверно (попробуйте [p])")
        if known_good:
            print(f"   ℹ Возвращаю known-good (может быть пустой кадр).")
            return known_good[0]
        print(f"   ℹ Возвращаю лучший найденный (возможен мусор).")
    return best
