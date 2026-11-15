"""Diagnostic: subtract the DR1 All-Sky Cube (static sky) from an exposure cutout.

Day-1 result: residual rings at every star (the cube is 12.9"/px vs 6.15"/px), so image
subtraction against the cube is NOT used for detection. Kept as a reproducible diagnostic.
"""
from __future__ import annotations

import numpy as np
from astropy.wcs import WCS
from astropy_healpix import HEALPix
from reproject import reproject_adaptive

from . import config, flags, s3

CUBE_PREFIX = "dr1/hips5_cube/hips_20261026/Norder5"


def cube_key(tile: int) -> str:
    return f"{CUBE_PREFIX}/Dir{tile // 10000 * 10000}/Npix{tile}_cube.fits"


def subtract_cube(key: str, x: float, y: float, lam_um: float, half: int = 60) -> dict:
    n = config.IMAGE_SIZE
    xi = int(min(max(round(x), half), n - 1 - half))
    yi = int(min(max(round(y), half), n - 1 - half))
    sl = (slice(yi - half, yi + half + 1), slice(xi - half, xi + half + 1))
    with s3.open_fits(key) as h:
        img = h["IMAGE"].section[sl] - h["ZODI"].section[sl]
        var = h["VARIANCE"].section[sl]
        flg = h["FLAGS"].section[sl] if hasattr(h["FLAGS"], "section") else h["FLAGS"].data[sl]
        bad = flags.mask_value(flags.flag_bits(h["FLAGS"].header))
        w = WCS(h["IMAGE"].header, naxis=2)[sl]
    tile = int(HEALPix(nside=32, order="nested", frame="icrs").skycoord_to_healpix(w.pixel_to_world(half, half)))
    with s3.open_fits(cube_key(tile), block_size=2**20) as c:
        ch = c["SPECTRAL_CHANNELS"].data
        k = int(np.nanargmin(np.abs(np.asarray(ch["WAVELENGTH"], float) - lam_um)))
        wc = WCS(c["IMAGE"].header, fobj=c).celestial
        plane = c["IMAGE"].section[k, :, :]
    ref, _ = reproject_adaptive((plane, wc), w, shape_out=img.shape, conserve_flux=False)
    good = ((flg & bad) == 0) & np.isfinite(img) & np.isfinite(ref)
    gain = float(np.sum(img[good] * ref[good]) / np.sum(ref[good] ** 2))
    sig = (img - gain * ref) / np.sqrt(var)
    stars = good & (ref > np.nanpercentile(ref[good], 99))
    tx, ty = half + (round(x) - xi), half + (round(y) - yi)
    return {
        "tile": tile, "channel_um": float(ch["WAVELENGTH"][k]), "gain": gain,
        "scatter_sigma_all": float(1.4826 * np.nanmedian(np.abs(sig[good]))),
        "scatter_sigma_stars": float(1.4826 * np.nanmedian(np.abs(sig[stars]))),
        "target_peak_sigma": float(np.nanmax(sig[ty - 2:ty + 3, tx - 2:tx + 3])),
        "image": img, "reference": gain * ref, "significance": sig, "target_xy": (tx, ty),
    }
