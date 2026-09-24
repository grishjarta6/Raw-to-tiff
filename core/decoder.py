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
# Автоопределение разрешения и header по размеру essence
# ===========================================================================

KNOWN_RESOLUTIONS = [
    # ALEXA Mini
    (3424, 2202, "ALEXA Mini 3.4K Open Gate"),
    (3200, 1800, "ALEXA Mini 3.2K 16:9"),
    (3168, 1776, "ALEXA Mini 3.2K"),
    (2880, 2160, "ALEXA Mini 4:3 2.8K"),
    (2880, 1620, "ALEXA Mini 16:9 2.8K"),
    # ALEXA LF / 65
    (4448, 3096, "ALEXA LF 4.5K Open Gate"),
    (4096, 2304, "ALEXA LF 4K 16:9"),
    (4608, 3164, "ALEXA 65 4.6K"),
    (6560, 3100, "ALEXA 65 6.5K"),
    # Универсальные
    (4096, 2160, "DCI 4K"),
    (3840, 2160, "UHD 4K"),
    (1920, 1080, "HD 1080p"),
]


def detect_resolution(essence_size: int,
                      max_header: int = 512) -> list[tuple[int, int, int, str]]:
    """
    Возвращает список (W, H, header, name), для которых

        essence_size == header + 2 * (W*H/2) * 3/2

    Отсортирован по header (меньший — раньше).
    """
    out = []
    for w, h, name in KNOWN_RESOLUTIONS:
        chunk_pixels = (w * h) // 2
        chunk_bytes = chunk_pixels * 3 // 2
        needed = 2 * chunk_bytes
        header = essence_size - needed
        if 0 <= header < max_header:
            out.append((w, h, header, name))
    out.sort(key=lambda c: c[2])
    return out


def detect_frame_size(essence_size: int,
                      header: int | None = None) -> tuple[int, int, str] | None:
    """
    Backward-compat: (W, H, name) или None.

    Если header задан — используется как ограничение (ищет только
    с этим header). Если None — перебирает все [0, 512).
    """
    cands = detect_resolution(essence_size, max_header=512)
    if header is not None:
        cands = [c for c in cands if c[2] == header]
    if not cands:
        return None
    w, h, _, name = cands[0]
    return (w, h, name)

# ===========================================================================
# 24 side-формулы 12-бит: независимые формулы для p1 и p2.
# Автоопределение перебирает пары (p1, p2).
# ===========================================================================

def _make_side_formulas():
    names = ["b0", "b1", "b2"]
    out = []
    for xi, x in enumerate(names):
        for yi, y in enumerate(names):
            if xi == yi:
                continue
            out.append((
                f"({x}<<4)|({y}>>4)",
                lambda b0, b1, b2, X=xi, Y=yi:
                    ((b0, b1, b2)[X] << 4) | ((b0, b1, b2)[Y] >> 4),
            ))
            out.append((
                f"({x}<<4)|({y}&0x0F)",
                lambda b0, b1, b2, X=xi, Y=yi:
                    ((b0, b1, b2)[X] << 4) | ((b0, b1, b2)[Y] & 0x0F),
            ))
            out.append((
                f"(({x}&0x0F)<<8)|{y}",
                lambda b0, b1, b2, X=xi, Y=yi:
                    (((b0, b1, b2)[X] & 0x0F) << 8) | (b0, b1, b2)[Y],
            ))
            out.append((
                f"(({x}>>4)<<8)|{y}",
                lambda b0, b1, b2, X=xi, Y=yi:
                    (((b0, b1, b2)[X] >> 4) << 8) | (b0, b1, b2)[Y],
            ))
    return out


SIDE_FORMULAS = _make_side_formulas()  # список (name, fn) — 24 штуки


def unpack_pair(raw: bytes, i1: int, i2: int) -> np.ndarray:
    """12-бит распаковка с независимыми формулами для p1 и p2."""
    n = len(raw) // 3
    d = np.frombuffer(raw[:n * 3], dtype=np.uint8)
    b0 = d[0::3].astype(np.uint16)
    b1 = d[1::3].astype(np.uint16)
    b2 = d[2::3].astype(np.uint16)
    p1 = SIDE_FORMULAS[i1][1](b0, b1, b2)
    p2 = SIDE_FORMULAS[i2][1](b0, b1, b2)
    out = np.empty(len(b0) * 2, dtype=np.uint16)
    out[0::2] = p1
    out[1::2] = p2
    return out

# ===========================================================================
# Независимый декодер по ПАРЕ индексов (i1, i2) — для автоопределения
# ===========================================================================

def decode_frame_pair(raw: bytes, header: int, i1: int, i2: int,
                      assembly: str, pattern: str,
                      W: int, H: int) -> np.ndarray:
    """
    Декодирует кадр, используя явные формулы SIDE_FORMULAS[i1] и [i2].
    Не зависит от UNPACK_FUNCS. Используется, когда autodetect нашёл
    произвольную пару, не совпадающую с известной.
    """
    total = len(raw) - header
    if total <= 0:
        raise ValueError("header больше данных")
    chunk_bytes = total // 2

    c1 = unpack_pair(raw[header : header + chunk_bytes], i1, i2)
    c2 = unpack_pair(raw[header + chunk_bytes : header + 2 * chunk_bytes],
                     i1, i2)
    return ASSEMBLIES[assembly](c1, c2, W, H)


def make_decoder(det: dict):
    """
    Возвращает функцию decode(raw, W, H) -> np.ndarray.
    Если в det есть i1/i2 — используется decode_frame_pair,
    иначе — decode_frame с канонической функцией по имени.
    """
    if det.get("i1") is not None and det.get("i2") is not None:
        i1 = det["i1"]; i2 = det["i2"]
        header = det["header"]; assembly = det["assembly"]; pattern = det["pattern"]
        def _decode(raw, W, H):
            return decode_frame_pair(raw, header, i1, i2, assembly, pattern, W, H)
        return _decode
    # fallback на канонические функции
    name = det["unpack"]
    if name not in UNPACK_FUNCS:
        raise ValueError(f"Неизвестная распаковка: {name}. "
                         f"Доступны: {list(UNPACK_FUNCS)}")
    header = det["header"]; assembly = det["assembly"]; pattern = det["pattern"]
    def _decode(raw, W, H):
        return decode_frame(raw, header, name, assembly, pattern, W, H)
    return _decode