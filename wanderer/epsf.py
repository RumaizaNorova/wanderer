"""Effective PSF (ePSF) from the EPSF extension: 33x33 samples, oversampled 5x, one per detector zone.

An ePSF is already pixel-integrated, so a native-pixel model is the ePSF sampled at native pixel
centres times oversampling**2 (the oversampled array sums to 1).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

OVERSAMPLE = 5
SIZE = 33
CENTRE = (SIZE - 1) / 2


class EPSFSet:
    def __init__(self, table):
        self.xc = np.asarray(table["XCENTER"], float)
        self.yc = np.asarray(table["YCENTER"], float)
        self.cwave = np.asarray(table["CWAVE"], float)
        self.psf = np.asarray(table["EPSF"], float).reshape(-1, SIZE, SIZE)

    def zone(self, x: float, y: float) -> int:
        return int(np.argmin(np.hypot(self.xc - x, self.yc - y)))

    def model(self, zone: int, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Fraction of source flux in native pixels at offsets (dx, dy) from the source centre."""
        coords = np.stack([np.asarray(dy) * OVERSAMPLE + CENTRE, np.asarray(dx) * OVERSAMPLE + CENTRE])
        return OVERSAMPLE**2 * map_coordinates(self.psf[zone], coords, order=1, mode="constant", cval=0.0)


def gaussian_epsf(sigma_native: float = 0.6) -> np.ndarray:
    """Synthetic ePSF for tests: a pixel-integrated-like Gaussian normalised to sum 1."""
    g = (np.arange(SIZE) - CENTRE) / OVERSAMPLE
    yy, xx = np.meshgrid(g, g, indexing="ij")
    p = np.exp(-(xx**2 + yy**2) / (2 * sigma_native**2))
    return p / p.sum()
