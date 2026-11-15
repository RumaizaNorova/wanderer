"""C3 (known objects): forced ePSF photometry at predicted positions.

The fit solves flux + local background in closed form for every trial sub-pixel offset, all
offsets at once (coarse 0.25 px grid, then a 0.05 px refinement), instead of one least-squares
call per offset. Each exposure is opened once and only a 15x15 cutout is read.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from . import calib, config, flags, gaia, s3
from .epsf import EPSFSet

log = logging.getLogger(__name__)

CUT = 7   # cutout half-size (15x15)
BOX = 4   # fit box half-size (9x9)
SEARCH = 1.5  # max offset from the prediction (px)
ARCSEC2_TO_SR = (np.pi / 180 / 3600) ** 2


@dataclass
class Fit:
    flux: float       # in image units (MJy/sr) integrated over the PSF
    flux_err: float
    background: float
    dx: float
    dy: float
    chi2: float
    npix: int

    @property
    def chi2_dof(self) -> float:
        return self.chi2 / max(self.npix - 2, 1)


def _solve(D, W, M):
    """Weighted fit D ~ a*M + b for many models M (n_models, n_pix). Returns a, b, var_a, chi2."""
    Sw, Sd = W.sum(), (W * D).sum()
    Sm, Smm, Smd = (W * M).sum(1), (W * M * M).sum(1), (W * M * D).sum(1)
    det = Smm * Sw - Sm**2
    a = (Smd * Sw - Sm * Sd) / det
    b = (Smm * Sd - Sm * Smd) / det
    chi2 = (W * (D - a[:, None] * M - b[:, None]) ** 2).sum(1)
    return a, b, Sw / det, chi2


def _solve_with_stars(D, W, M, E):
    """Weighted fit D ~ a*M + sum_k s_k*E_k + b for many target models M (n, npix) and fixed star
    models E (k, npix). Returns a, b, var_a, chi2 per target model."""
    n, npix = M.shape
    X = np.concatenate([M[:, :, None], np.broadcast_to(E.T[None], (n, npix, len(E))), np.ones((n, npix, 1))], 2)
    XtW = X.transpose(0, 2, 1) * W[None, None, :]
    A = XtW @ X + 1e-12 * np.eye(X.shape[2])
    coef = np.linalg.solve(A, (XtW @ D)[..., None])[..., 0]
    chi2 = (W * (D - (X @ coef[..., None])[..., 0]) ** 2).sum(1)
    cov00 = np.linalg.inv(A)[:, 0, 0]
    return coef[:, 0], coef[:, -1], cov00, chi2


def fit_point_source(img, var, good, psf: EPSFSet, zone: int, x0: float, y0: float, stars_xy=()) -> Fit:
    """Fit a point source near (x0, y0), given in cutout pixel coordinates.

    `stars_xy`: cutout positions of known static stars to model at the same time (fixed position,
    free flux), so their light is not counted as the target's.
    """
    cy, cx = int(round(y0)), int(round(x0))
    yy, xx = np.mgrid[cy - BOX:cy + BOX + 1, cx - BOX:cx + BOX + 1]
    sel = good[yy, xx]
    D, W = img[yy, xx][sel], 1.0 / var[yy, xx][sel]
    px, py = xx[sel].astype(float), yy[sel].astype(float)
    E = np.array([psf.model(zone, px - sx, py - sy) for sx, sy in stars_xy]).reshape(-1, len(D))
    E = E[E.sum(1) > 1e-3]  # drop stars that put no light in the fit box

    def evaluate(offsets):
        dx, dy = offsets[:, 0:1], offsets[:, 1:2]
        M = psf.model(zone, px[None, :] - (x0 + dx), py[None, :] - (y0 + dy))
        return _solve_with_stars(D, W, M, E) if len(E) else _solve(D, W, M)

    g = np.arange(-SEARCH, SEARCH + 1e-9, 0.25)
    coarse = np.array([(a, b) for a in g for b in g])
    a, b, va, chi2 = evaluate(coarse)
    best = coarse[np.argmin(chi2)]
    f = np.arange(-0.25, 0.25 + 1e-9, 0.05)
    fine = np.clip(np.array([(best[0] + a_, best[1] + b_) for a_ in f for b_ in f]), -SEARCH, SEARCH)
    a, b, va, chi2 = evaluate(fine)
    i = int(np.argmin(chi2))
    return Fit(float(a[i]), float(np.sqrt(va[i])), float(b[i]), float(fine[i, 0]), float(fine[i, 1]),
               float(chi2[i]), int(sel.sum()) - len(E))


BLEND_PX = 1.0  # a star closer than this cannot be separated from the target


def measure(sighting: dict, stars: pd.DataFrame | None = None) -> dict | None:
    """Forced photometry of one sighting; returns a result row or None if the source is unusable."""
    x, y, det = sighting["x"], sighting["y"], sighting["det"]
    if not calib.in_bounds(x, y, CUT):
        return None
    xi, yi = int(round(x)), int(round(y))
    sl = (slice(yi - CUT, yi + CUT + 1), slice(xi - CUT, xi + CUT + 1))
    with s3.open_fits(sighting["key"]) as h:
        img = h["IMAGE"].section[sl] - h["ZODI"].section[sl]
        var = h["VARIANCE"].section[sl]
        flg = h["FLAGS"].section[sl] if hasattr(h["FLAGS"], "section") else h["FLAGS"].data[sl]
        bits = flags.flag_bits(h["FLAGS"].header)
        psf = EPSFSet(h["EPSF"].data)
        wcs = WCS(h["IMAGE"].header, naxis=2)
    good = np.isfinite(img) & np.isfinite(var) & (var > 0) & ((flg & flags.mask_value(bits)) == 0)
    core = (slice(CUT - 2, CUT + 3),) * 2
    if good[core].sum() < 15:
        return {**_ids(sighting), "status": "masked"}
    stars_xy, nearest = [], np.inf
    if stars is not None and len(stars):
        sra, sdec = gaia.at_epoch(stars, sighting["mjd"])
        sx, sy = wcs.all_world2pix(np.c_[sra, sdec], 0).T
        sx, sy = sx - xi + CUT, sy - yi + CUT
        near = (np.abs(sx - CUT) <= BOX + 3) & (np.abs(sy - CUT) <= BOX + 3)
        stars_xy = list(zip(sx[near], sy[near]))
        if near.any():
            nearest = float(np.min(np.hypot(sx[near] - (x - xi + CUT), sy[near] - (y - yi + CUT))))
    blended = nearest < BLEND_PX
    fit = fit_point_source(img, var, good, psf, psf.zone(x, y), x - xi + CUT, y - yi + CUT,
                           [] if blended else stars_xy)
    xf, yf = x + fit.dx, y + fit.dy
    lam, bw = calib.wavelength(det, xf, yf)
    omega = calib.solid_angle_arcsec2(det, xf, yf)
    to_mjy = omega * ARCSEC2_TO_SR * 1e9  # MJy/sr * sr -> MJy -> mJy
    return {**_ids(sighting), "status": "ok", "lam_um": lam, "bw_um": bw,
            "flux_mJy": fit.flux * to_mjy, "err_mJy": fit.flux_err * to_mjy, "snr": fit.flux / fit.flux_err,
            "chi2_dof": fit.chi2_dof, "dx_pix": fit.dx, "dy_pix": fit.dy, "omega_arcsec2": omega,
            "star_overlap": float(((flg[core] & bits.get("SOURCE", 0)) > 0).mean()),
            "n_stars_modelled": 0 if blended else len(stars_xy), "nearest_star_px": nearest, "blended": blended}


def _ids(s: dict) -> dict:
    return {k: s[k] for k in ("key", "obsid", "det", "mjd", "block", "x", "y")}


def stars_along_track(sightings: pd.DataFrame, gmax: float = 19.0) -> pd.DataFrame:
    """One Gaia query covering the whole track (plus the fit-box margin)."""
    c = SkyCoord(sightings.ra_pred.values * u.deg, sightings.dec_pred.values * u.deg)
    mid = SkyCoord(np.median(sightings.ra_pred) * u.deg, np.median(sightings.dec_pred) * u.deg)
    radius = float(c.separation(mid).deg.max()) + (BOX + 4) * config.PIXEL_SCALE_ARCSEC / 3600
    return gaia.cone(mid.ra.deg, mid.dec.deg, radius, gmax)


def forced_photometry(sightings: pd.DataFrame, workers: int = 8, model_stars: bool = True) -> pd.DataFrame:
    rows = sightings.to_dict("records")
    stars = None
    if model_stars:
        try:
            stars = stars_along_track(sightings)
            log.info("modelling background stars: %d Gaia DR3 sources (G<19) along the track", len(stars))
        except RuntimeError as e:
            log.warning("no Gaia stars (%s); fitting without star modelling", e)
    with ThreadPoolExecutor(workers) as ex:
        out = [r for r in ex.map(lambda s: _safe_measure(s, stars), rows) if r]
    df = pd.DataFrame(out).sort_values(["mjd", "det"]).reset_index(drop=True)
    log.info("photometry: %d ok, %d masked", (df.status == "ok").sum(), (df.status == "masked").sum())
    return df


def _safe_measure(s: dict, stars=None) -> dict | None:
    try:
        return measure(s, stars)
    except Exception as e:  # noqa: BLE001 - report and continue
        log.warning("photometry failed for %s D%s: %s", s["obsid"], s["det"], e)
        return {**_ids(s), "status": f"error: {e}"}
