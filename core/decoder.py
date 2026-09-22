"""
Распаковка 12-бит ARRIRAW и сборка кадра.

Каждый essence-пакет = [header][chunk1][chunk2], где chunk1/chunk2 —
две половины кадра (top/bottom, left/right, чёт/нечет строк или столбцов).
"""

import numpy as np


# ===========================================================================
# 12-бит распаковка: 3 байта → 2 пикселя
# ===========================================================================

def _unpack_arri_alt(raw: bytes) -> np.ndarray:
    """ARRIRAW ALEXA Mini (найдено эмпирически)."""
    n = len(raw) // 3
    d = np.frombuffer(raw[:n * 3], dtype=np.uint8)
    b0 = d[0::3].astype(np.uint16)
    b1 = d[1::3].astype(np.uint16)
    b2 = d[2::3].astype(np.uint16)
    p1 = (b0 << 4) | (b2 >> 4)
    p2 = ((b2 & 0x0F) << 8) | b1
    out = np.empty(len(b0) * 2, dtype=np.uint16)
    out[0::2] = p1
    out[1::2] = p2
    return out


def _unpack_std(raw: bytes) -> np.ndarray:
    n = len(raw) // 3
    d = np.frombuffer(raw[:n * 3], dtype=np.uint8)
    b0 = d[0::3].astype(np.uint16)
    b1 = d[1::3].astype(np.uint16)
    b2 = d[2::3].astype(np.uint16)
    p1 = (b0 << 4) | (b1 >> 4)
    p2 = ((b1 & 0x0F) << 8) | b2
    out = np.empty(len(b0) * 2, dtype=np.uint16)
    out[0::2] = p1
    out[1::2] = p2
    return out


def _unpack_big_endian(raw: bytes) -> np.ndarray:
    n = len(raw) // 3
    d = np.frombuffer(raw[:n * 3], dtype=np.uint8)
    b0 = d[0::3].astype(np.uint16)
    b1 = d[1::3].astype(np.uint16)
    b2 = d[2::3].astype(np.uint16)
    p1 = (b0 << 4) | (b2 >> 4)
    p2 = (b1 << 4) | (b2 & 0x0F)
    out = np.empty(len(b0) * 2, dtype=np.uint16)
    out[0::2] = p1
    out[1::2] = p2
    return out


def _unpack_little_endian(raw: bytes) -> np.ndarray:
    n = len(raw) // 3
    d = np.frombuffer(raw[:n * 3], dtype=np.uint8)
    b0 = d[0::3].astype(np.uint16)
    b1 = d[1::3].astype(np.uint16)
    b2 = d[2::3].astype(np.uint16)
    p1 = ((b1 & 0x0F) << 8) | b0
    p2 = ((b1 >> 4) << 8) | b2
    out = np.empty(len(b0) * 2, dtype=np.uint16)
    out[0::2] = p1
    out[1::2] = p2
    return out


UNPACK_FUNCS = {
    "arri_alt":      _unpack_arri_alt,
    "std":           _unpack_std,
    "big_endian":    _unpack_big_endian,
    "little_endian": _unpack_little_endian,
}


# ===========================================================================
# Сборка двух чанков в кадр
# ===========================================================================

def _assemble_top_bottom(c1, c2, W, H):
    HH = H // 2
    n = HH * W
    frame = np.empty((H, W), dtype=np.uint16)
    frame[:HH, :] = c1[:n].reshape(HH, W)
    frame[HH:, :] = c2[:n].reshape(HH, W)
    return frame


def _assemble_left_right(c1, c2, W, H):
    HW = W // 2
    n = H * HW
    frame = np.empty((H, W), dtype=np.uint16)
    frame[:, :HW] = c1[:n].reshape(H, HW)
    frame[:, HW:] = c2[:n].reshape(H, HW)
    return frame


def _assemble_even_odd_rows(c1, c2, W, H):
    HH = H // 2
    n = HH * W
    frame = np.empty((H, W), dtype=np.uint16)
    frame[0::2, :] = c1[:n].reshape(HH, W)
    frame[1::2, :] = c2[:n].reshape(HH, W)
    return frame


def _assemble_even_odd_columns(c1, c2, W, H):
    HW = W // 2
    n = H * HW
    frame = np.empty((H, W), dtype=np.uint16)
    frame[:, 0::2] = c1[:n].reshape(H, HW)
    frame[:, 1::2] = c2[:n].reshape(H, HW)
    return frame


ASSEMBLIES = {
    "top_bottom":       _assemble_top_bottom,
    "left_right":       _assemble_left_right,
    "even_odd_rows":    _assemble_even_odd_rows,
    "even_odd_columns": _assemble_even_odd_columns,
}


# ===========================================================================
# Паттерны Bayer
# ===========================================================================

PATTERNS = {
    "GBRG": ("G1", "B",  "R",  "G2"),   # ALEXA Mini
    "GRBG": ("G1", "R",  "B",  "G2"),
    "BGGR": ("B",  "G2", "G1", "R"),
    "RGGB": ("R",  "G1", "G2", "B"),
}


def split_channels(frame: np.ndarray, pattern: str) -> dict:
    tl, tr, bl, br = PATTERNS[pattern]
    return {
        tl: frame[0::2, 0::2],
        tr: frame[0::2, 1::2],
        bl: frame[1::2, 0::2],
        br: frame[1::2, 1::2],
    }


# ===========================================================================
# Декод одного кадра
# ===========================================================================

def decode_frame(raw: bytes, header: int, unpack_mode: str,
                 assembly: str, pattern: str, W: int, H: int) -> np.ndarray:
    total = len(raw) - header
    if total <= 0:
        raise ValueError("header больше данных")
    chunk_bytes = total // 2

    c1 = UNPACK_FUNCS[unpack_mode](raw[header : header + chunk_bytes])
    c2 = UNPACK_FUNCS[unpack_mode](
        raw[header + chunk_bytes : header + 2 * chunk_bytes])

    return ASSEMBLIES[assembly](c1, c2, W, H)


# ===========================================================================
# Автоопределение разрешения по размеру essence
# ===========================================================================

# Известные разрешения ARRI (W, H, name)
KNOWN_RESOLUTIONS = [
    (3424, 2202, "ALEXA Mini 3.4K Open Gate"),
    (2880, 2160, "ALEXA Mini 4:3 2.8K"),
    (2880, 1620, "ALEXA Mini 16:9 2.8K"),
    (3200, 1800, "ALEXA Mini 3.2K 16:9"),
    (4448, 3096, "ALEXA 65 4.5K"),
    (4096, 2304, "ALEXA LF 4K 16:9"),
    (3840, 2160, "UHD 4K"),
    (1920, 1080, "HD 1080p"),
]


def detect_frame_size(essence_size: int,
                      header: int = 76) -> tuple[int, int, str] | None:
    """
    Определяет (W, H, имя) по размеру essence.

    Модель: essence_size = header + 2 * chunk_bytes,
           chunk_bytes   = (W*H/2) * 3/2,
           значит W*H = essence_size_avail * 2 / 3.
    """
    avail = essence_size - header
    if avail <= 0 or avail % 3 != 0:
        return None
    total_pixels = (avail * 2) // 3

    for w, h, name in KNOWN_RESOLUTIONS:
        if w * h == total_pixels:
            return (w, h, name)
    return None