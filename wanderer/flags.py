"""SPHEREx FLAGS extension: bit names are read from each file's MSKNnnnn/MSKMnnnn header cards.

Day-2 lesson: SOURCE ("pixel mapped to a known source") and OUTLIER often cover the asteroid
itself, so they are informational. Only real defects are masked.
"""
from __future__ import annotations

DEFECTS = (
    "TRANSIENT", "OVERFLOW", "SUR_ERROR", "NONFUNC", "MISSING_DATA", "HOT", "COLD", "NONLINEAR",
    "PERSIST", "CROSSTALK", "GHOST", "GHOST_FPA", "GHOST_EXT", "STREAK", "BLOOM", "SNOWBALL", "HALO",
)
INFORMATIONAL = ("DICHROIC", "FULLSAMPLE", "PHANMISS", "OUTLIER", "SOURCE")

# Values as documented in DR1 headers; used when a header lacks the MSK cards.
DR1_BITS = {
    "TRANSIENT": 1, "OVERFLOW": 2, "SUR_ERROR": 4, "NONFUNC": 64, "DICHROIC": 128, "MISSING_DATA": 512,
    "HOT": 1024, "COLD": 2048, "FULLSAMPLE": 4096, "PHANMISS": 16384, "NONLINEAR": 32768,
    "PERSIST": 131072, "OUTLIER": 524288, "CROSSTALK": 1048576, "SOURCE": 2097152, "GHOST": 4194304,
    "GHOST_FPA": 8388608, "GHOST_EXT": 16777216, "STREAK": 33554432, "BLOOM": 67108864,
    "SNOWBALL": 134217728, "HALO": 268435456,
}


def flag_bits(header) -> dict[str, int]:
    """Map flag name -> bit value from a FLAGS header; falls back to the DR1 table."""
    bits = {header[k]: int(header["MSKM" + k[4:]]) for k in header if k.startswith("MSKN") and "MSKM" + k[4:] in header}
    return bits or dict(DR1_BITS)


def mask_value(bits: dict[str, int], names=DEFECTS) -> int:
    v = 0
    for n in names:
        v |= bits.get(n, 0)
    return v


def describe(value: int, bits: dict[str, int]) -> list[str]:
    return [n for n, b in bits.items() if value & b]
