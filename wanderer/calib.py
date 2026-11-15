"""Per-pixel calibration lookups (science wavelength, bandwidth, pixel solid angle).

Uses the stand-alone Spectral WCS product (CWAVE, CBAND), which the archive recommends for
science, and the solid-angle pixel map. Handles are cached per thread and per detector, so a
lookup costs one small range read instead of reopening a 32 MB file.
"""
from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

from . import config, s3

_local = threading.local()


def _handle(product: str, det: int):
    cache = _local.__dict__.setdefault("handles", {})
    if (product, det) not in cache:
        cache[(product, det)] = s3.open_fits(s3.calibration_key(product, det))
    return cache[(product, det)]


def wavelength(det: int, x: float, y: float) -> tuple[float, float]:
    """Central wavelength and bandwidth (um) at 0-based pixel (x, y)."""
    h = _handle("spectral_wcs", det)
    xi, yi = int(round(x)), int(round(y))
    return float(h["CWAVE"].section[yi, xi]), float(h["CBAND"].section[yi, xi])


@lru_cache(maxsize=6)
def wavelength_map(det: int):
    """Full CWAVE and CBAND arrays for a detector (2 x 16 MB, read once per process)."""
    with s3.open_fits(s3.calibration_key("spectral_wcs", det), block_size=2**22) as h:
        return np.asarray(h["CWAVE"].data, np.float32), np.asarray(h["CBAND"].data, np.float32)


def solid_angle_arcsec2(det: int, x: float, y: float) -> float:
    h = _handle("solid_angle_pixel_map", det)
    hdu = h[1] if len(h) > 1 and h[1].header.get("NAXIS", 0) == 2 else h[0]
    return float(hdu.section[int(round(y)), int(round(x))])


def in_bounds(x: float, y: float, margin: int) -> bool:
    n = config.IMAGE_SIZE
    return margin <= round(x) < n - margin and margin <= round(y) < n - margin
