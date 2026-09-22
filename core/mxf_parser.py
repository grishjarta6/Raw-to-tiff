"""
Парсер MXF для ARRIRAW (SMPTE RDD 54).

Essence key ARRI: 060e2b34 0102010d 0f010301 01010100
"""

from __future__ import annotations
from pathlib import Path


MXF_UL_PREFIX = b"\x06\x0e\x2b\x34"

ESSENCE_KEY_PREFIX  = bytes.fromhex("060e2b34010201010d010301")
ARRIRAW_ESSENCE_KEY = bytes.fromhex("060e2b340102010d0f01030101010100")
ESSENCE_CONTAINER_PREFIX = bytes.fromhex("060e2b340401010d0401")

HEADER_PARTITION = bytes.fromhex("060e2b34020501010d0102010102")
BODY_PARTITION   = bytes.fromhex("060e2b34020501010d0102010103")
FOOTER_PARTITION = bytes.fromhex("060e2b34020501010d0102010104")
PRIMER_PACK      = bytes.fromhex("060e2b34020501010d0102010105")
INDEX_TABLE      = bytes.fromhex("060e2b34020501010d0102010110")
RANDOM_INDEX     = bytes.fromhex("060e2b34020501010d0102010111")


def read_ber_length(data: bytes, offset: int) -> tuple[int, int]:
    if offset >= len(data):
        raise EOFError("EOF при чтении BER length")
    b0 = data[offset]
    if (b0 & 0x80) == 0:
        return b0, 1
    n = b0 & 0x7F
    if n == 0 or n > 8:
        raise ValueError(f"Некорректная BER-длина: n={n}")
    if offset + 1 + n > len(data):
        raise EOFError("EOF внутри BER length")
    length = 0
    for i in range(n):
        length = (length << 8) | data[offset + 1 + i]
    return length, 1 + n


def identify_key(key: bytes) -> str:
    if len(key) < 16:
        return "truncated"
    if not key.startswith(MXF_UL_PREFIX):
        return "not_mxf"
    if key == ARRIRAW_ESSENCE_KEY:
        return "essence_arriraw"
    if key[:12] == ESSENCE_KEY_PREFIX:
        return "essence"
    if key[:10] == ESSENCE_CONTAINER_PREFIX:
        return "essence_container"
    if key[:14] == HEADER_PARTITION:  return "header_partition"
    if key[:14] == BODY_PARTITION:    return "body_partition"
    if key[:14] == FOOTER_PARTITION:  return "footer_partition"
    if key[:14] == PRIMER_PACK:       return "primer_pack"
    if key[:14] == INDEX_TABLE:       return "index_table"
    if key[:14] == RANDOM_INDEX:      return "random_index"
    return "unknown"


def _scan_range(data, start, end, packets, depth=0,
                verbose=False, max_packets=20000):
    pos = start
    search_from = start
    while pos < end and len(packets) < max_packets:
        if data[pos:pos + 4] != MXF_UL_PREFIX:
            idx = data.find(MXF_UL_PREFIX, search_from)
            if idx < 0 or idx >= end:
                break
            pos = idx
            search_from = pos + 1
        if pos + 16 > end:
            break
        key = data[pos:pos + 16]
        try:
            length, len_size = read_ber_length(data, pos + 16)
        except (EOFError, ValueError):
            pos += 1
            search_from = pos
            continue
        value_start = pos + 16 + len_size
        value_end = value_start + length
        if value_end > end:
            length = max(0, end - value_start)
        key_type = identify_key(key)
        pkt = {
            "offset": pos, "key": key.hex(), "type": key_type,
            "length": length, "value_start": value_start,
            "value_end": value_end, "depth": depth,
        }
        packets.append(pkt)
        if verbose:
            indent = "  " * depth
            print(f"{indent}[{len(packets):>4}] @ {pos:>10}  "
                  f"{key_type:<20}  len={length:>12}")

        if key_type == "body_partition" and length > 0:
            _scan_range(data, value_start, min(value_end, end), packets,
                        depth=depth + 1, verbose=verbose,
                        max_packets=max_packets)

        pos = value_start + length if length > 0 else pos + 1
        if pos <= value_end - length:
            pos += 1
        search_from = pos
    return None


def scan_klv_packets(filepath: Path, max_scan=2 * 1024 * 1024 * 1024,
                     verbose=False, max_packets=20000) -> list[dict]:
    file_size = filepath.stat().st_size
    scan_limit = min(file_size, max_scan)
    with open(filepath, "rb") as f:
        data = f.read(scan_limit)
    packets: list[dict] = []
    _scan_range(data, 0, len(data), packets, 0, verbose, max_packets)
    return packets


def parse(filepath: str | Path, verbose: bool = False) -> dict:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(filepath)
    packets = scan_klv_packets(filepath, verbose=verbose, max_packets=20000)
    if not packets:
        raise RuntimeError("KLV-пакеты не найдены. Файл не MXF?")
    essence = [p for p in packets
               if p["type"] in ("essence", "essence_arriraw",
                                "essence_container")]
    if not essence:
        types_seen = sorted({p["type"] for p in packets})
        raise RuntimeError(f"Essence не найден. Типы: {types_seen}")
    return {
        "header_offset": essence[0]["value_start"],
        "essence_key":   essence[0]["key"],
        "packet_count":  len(packets),
        "frame_count":   len(essence),
        "frame_size":    essence[0]["length"],
        "packets":       packets,
        "essence":       essence,
    }