"""C2: per-exposure source catalog and astrometric check against Gaia DR3.

Reads one full exposure (IMAGE, ZODI, VARIANCE, FLAGS: ~50 MB via range reads), subtracts zodi
and a smooth background, finds point sources, measures centroids and aperture fluxes, attaches
per-source wavelength, and compares positions with proper-motion-corrected Gaia stars.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip
from astropy.wcs import WCS, FITSFixedWarning
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder

from . import calib, flags, gaia, s3

log = logging.getLogger(__name__)
warnings.simplefilter("ignore", FITSFixedWarning)

FWHM_PX = 7.5 / 6.15
APERTURE_PX = 2.0


def source_catalog(key: str, det: int, nsigma: float = 5.0) -> tuple[pd.DataFrame, WCS, float]:
    with s3.open_fits(key, block_size=2**22) as h:
        img = h["IMAGE"].data - h["ZODI"].data
        var = h["VARIANCE"].data
        flg = h["FLAGS"].data
        bad = flags.mask_value(flags.flag_bits(h["FLAGS"].header))
        wcs = WCS(h["IMAGE"].header, naxis=2)
        mjd = float(h["IMAGE"].header["MJD-AVG"])
    mask = ~np.isfinite(img) | ~np.isfinite(var) | (var <= 0) | ((flg & bad) != 0)
    bkg = Background2D(img, (64, 64), filter_size=(3, 3), mask=mask, sigma_clip=SigmaClip(3.0),
                       bkg_estimator=MedianBackground())
    data = np.where(mask, 0.0, img - bkg.background)
    noise = float(np.nanmedian(np.sqrt(var[~mask])))
    found = DAOStarFinder(fwhm=FWHM_PX, threshold=nsigma * noise, exclude_border=True)(data, mask=mask)
    if found is None:
        return pd.DataFrame(), wcs, mjd
    xy = np.c_[found["xcentroid"], found["ycentroid"]]
    ap = CircularAperture(xy, r=APERTURE_PX)
    phot = aperture_photometry(data, ap, error=np.sqrt(np.where(mask, np.inf, var)), mask=mask)
    ra, dec = wcs.all_pix2world(xy, 0).T
    cat = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "ra": ra, "dec": dec,
                        "flux": np.asarray(phot["aperture_sum"]), "flux_err": np.asarray(phot["aperture_sum_err"]),
                        "sharpness": np.asarray(found["sharpness"]), "roundness": np.asarray(found["roundness1"])})
    cat["snr"] = cat.flux / cat.flux_err
    cwave, cband = calib.wavelength_map(det)
    xi, yi = np.rint(xy[:, 0]).astype(int), np.rint(xy[:, 1]).astype(int)
    cat["lam_um"], cat["bw_um"] = cwave[yi, xi], cband[yi, xi]
    log.info("%s: %d sources above %.0f sigma", key.split("/")[-1], len(cat), nsigma)
    return cat, wcs, mjd


def astrometry_check(cat: pd.DataFrame, wcs: WCS, mjd: float, gmax: float = 15.0, match_arcsec: float = 4.0) -> dict:
    """Median offset between detected sources and Gaia stars (proper motion applied)."""
    ra0, dec0 = wcs.all_pix2world([[1019.5, 1019.5]], 0)[0]
    stars = gaia.cone(ra0, dec0, 2.5, gmax)
    sra, sdec = gaia.at_epoch(stars, mjd)
    g = SkyCoord(sra * u.deg, sdec * u.deg)
    good = cat[(cat.snr > 20) & (cat.sharpness.between(0.2, 1.0))]
    c = SkyCoord(good.ra.values * u.deg, good.dec.values * u.deg)
    idx, sep, _ = c.match_to_catalog_sky(g)
    m = sep.arcsec < match_arcsec
    dra = (good.ra.values[m] - sra[idx[m]]) * np.cos(np.radians(sdec[idx[m]])) * 3600
    ddec = (good.dec.values[m] - sdec[idx[m]]) * 3600
    return {"gaia_stars": int(len(stars)), "sources_used": int(len(good)), "matched": int(m.sum()),
            "median_offset_arcsec": float(np.median(sep.arcsec[m])) if m.any() else None,
            "median_dra_arcsec": float(np.median(dra)) if m.any() else None,
            "median_ddec_arcsec": float(np.median(ddec)) if m.any() else None,
            "pass_under_1arcsec": bool(m.any() and np.median(sep.arcsec[m]) < 1.0)}
