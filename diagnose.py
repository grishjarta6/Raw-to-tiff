"""
Диагностика: прогоняет все (unpack × assembly × pattern) на одном кадре
и печатает полную таблицу. Полезно, если автоопределение ошибается.

Запуск:
    uv run python diagnose.py
"""

from pathlib import Path
import numpy as np

from core.mxf_parser import parse as mxf_parse
from core.decoder import (UNPACK_FUNCS, ASSEMBLIES, PATTERNS,
                          split_channels)


MXF_FILE = "A003C007_191110_R3MG.mxf"
W, H = 3424, 2202
SAMPLE_ROWS = 500


def spatial_corr(img):
    x = img.astype(np.float32)
    if x.shape[0] < 2 or x.shape[1] < 2:
        return 0.0
    hx = float(np.corrcoef(x[:, :-1].flatten(), x[:, 1:].flatten())[0, 1])
    hy = float(np.corrcoef(x[:-1, :].flatten(), x[1:, :].flatten())[0, 1])
    return (hx + hy) / 2


def main():
    print(f"UNPACK_FUNCS: {list(UNPACK_FUNCS)}")
    print(f"ASSEMBLIES:   {list(ASSEMBLIES)}")
    print(f"PATTERNS:     {list(PATTERNS)}")
    print()

    path = Path(MXF_FILE)
    if not path.exists():
        print(f"❌ Файл не найден: {path}")
        return

    result = mxf_parse(path)
    pkt = result["essence"][0]

    print(f"Essence size: {pkt['length']:,}")
    print(f"Ожидаемый размер для {W}×{H}: "
          f"76 + 2 × {(W*H//2)*3//2:,} = "
          f"{76 + 2*(W*H//2)*3//2:,}")
    print()

    with open(path, "rb") as f:
        f.seek(pkt["value_start"])
        raw = f.read(pkt["length"])

    sample_H = min(SAMPLE_ROWS, H)
    hdr = 76
    chunk_bytes = (pkt["length"] - hdr) // 2

    print(f"{'unpack':<15} {'assembly':<18} {'pattern':<6}  "
          f"{'corr_G':>8}  {'corr_BR':>8}  {'sp':>7}  "
          f"{'min_std':>8}  {'score':>9}")
    print("-" * 100)

    rows = []
    for unpack_mode in UNPACK_FUNCS:
        try:
            c1 = UNPACK_FUNCS[unpack_mode](
                raw[hdr : hdr + chunk_bytes])
            c2 = UNPACK_FUNCS[unpack_mode](
                raw[hdr + chunk_bytes : hdr + 2 * chunk_bytes])
        except Exception as e:
            print(f"❌ {unpack_mode}: {e}")
            continue

        for assembly in ASSEMBLIES:
            try:
                frame = ASSEMBLIES[assembly](c1, c2, W, H)
            except Exception as e:
                continue

            sample = frame[:sample_H, :]

            for pattern in PATTERNS:
                chans = split_channels(sample, pattern)
                stds = {n: float(v.std()) for n, v in chans.items()}
                min_std = min(stds.values())
                max_std = max(stds.values())
                if min_std < 20 or max_std < 1 or min_std / max_std < 0.05:
                    continue

                g1 = chans["G1"].flatten().astype(np.float32)
                g2 = chans["G2"].flatten().astype(np.float32)
                b  = chans["B"].flatten().astype(np.float32)
                r  = chans["R"].flatten().astype(np.float32)

                corr_g  = float(np.corrcoef(g1, g2)[0, 1])
                corr_br = float(np.corrcoef(b, r)[0, 1])
                sp = (spatial_corr(chans["B"]) +
                      spatial_corr(chans["G1"]) +
                      spatial_corr(chans["G2"]) +
                      spatial_corr(chans["R"])) / 4

                score = corr_g * 1000.0 + sp * 10.0 + min_std / 100.0
                rows.append({
                    "unpack": unpack_mode, "assembly": assembly,
                    "pattern": pattern,
                    "corr_g": corr_g, "corr_br": corr_br,
                    "sp": sp, "min_std": min_std, "score": score,
                })

    rows.sort(key=lambda x: x["score"], reverse=True)

    for r in rows[:30]:
        print(f"{r['unpack']:<15} {r['assembly']:<18} "
              f"{r['pattern']:<6}  "
              f"{r['corr_g']:>+8.4f}  {r['corr_br']:>+8.4f}  "
              f"{r['sp']:>+7.3f}  {r['min_std']:>8.0f}  "
              f"{r['score']:>9.2f}")

    print()
    if rows:
        best = rows[0]
        print(f"🏆 ЛУЧШЕЕ: {best['unpack']} + {best['assembly']} + "
              f"{best['pattern']}")
        print(f"   corr_G={best['corr_g']:+.4f}, "
              f"sp={best['sp']:+.4f}, min_std={best['min_std']:.0f}")


if __name__ == "__main__":
    main()