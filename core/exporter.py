"""Сохранение каналов кадра в TIFF."""

from pathlib import Path

import numpy as np
import tifffile


def build_rgb(chans: dict) -> np.ndarray:
    R  = chans["R"].astype(np.float32)
    B  = chans["B"].astype(np.float32)
    G1 = chans["G1"].astype(np.float32)
    G2 = chans["G2"].astype(np.float32)
    G  = 0.5 * (G1 + G2)

    def norm(x):
        p = np.percentile(x, 99.0)
        return np.clip(x / max(p, 1), 0, 1)

    rgb = np.stack([norm(R), norm(G), norm(B)], axis=-1)
    return (rgb * 255).astype(np.uint8)


def save_frame(chans: dict, out_dir: Path, frame_idx: int,
               pattern: str, save_rgb: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"frame{frame_idx:04d}_{pattern}"

    for name, img in chans.items():
        tifffile.imwrite(str(base) + f"_{name}.tiff", img)

    if save_rgb:
        rgb = build_rgb(chans)
        tifffile.imwrite(str(base) + "_RGB.tiff", rgb)