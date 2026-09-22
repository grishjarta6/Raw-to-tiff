"""Вывод статистики по каналам и параметрам распаковки."""

import numpy as np


def channel_stats(img: np.ndarray) -> dict:
    return {
        "min":  int(img.min()),
        "max":  int(img.max()),
        "mean": float(img.mean()),
        "std":  float(img.std()),
    }


def print_detection(det: dict) -> None:
    print(f"\n   📊 Параметры распаковки (автоопределение):")
    print(f"      header:       {det['header']} байт")
    print(f"      unpack:       {det['unpack']}")
    print(f"      assembly:     {det['assembly']}")
    print(f"      pattern:      {det['pattern']}")
    print(f"      score:        {det['score']:.3f}")
    print(f"      corr(G1,G2):  {det['corr_g1g2']:+.4f}")
    print(f"      corr(B,R):    {det['corr_br']:+.4f}")
    print(f"      sp_avg:       {det['sp_avg']:+.4f}")
    s = det["stds"]
    print(f"      std:  B={s['B']:.0f}  G1={s['G1']:.0f}  "
          f"G2={s['G2']:.0f}  R={s['R']:.0f}")


def print_channels_short(chans: dict) -> str:
    parts = []
    for name in ("B", "G1", "G2", "R"):
        s = chans[name].std()
        parts.append(f"{name}(std={s:.0f})")
    return "  ".join(parts)


def print_channels_full(chans: dict) -> None:
    for name in ("B", "G1", "G2", "R"):
        s = channel_stats(chans[name])
        print(f"      {name:<3}: min={s['min']:>5}  max={s['max']:>5}  "
              f"mean={s['mean']:>8.1f}  std={s['std']:>7.1f}")