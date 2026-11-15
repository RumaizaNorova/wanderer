"""C5: photometry -> reflectance spectrum, rotation-free colours, rotation scatter."""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.io import fits

from . import config, ephemeris, s3

NORM_WINDOW = (0.95, 1.15)   # um, reflectance normalised to 1 here
THERMAL_FROM_UM = 3.5        # beyond this, thermal emission adds flux for main-belt asteroids
RELIABLE = {"snr": 5.0, "chi2_dof": 4.0}


def solar_spectrum() -> tuple[np.ndarray, np.ndarray]:
    """Solar F_nu shape on 0.6-5.2 um: CALSPEC to 2.6 um, 5778 K blackbody tail scaled at 2.5-2.65 um."""
    path = config.REFERENCE_DIR / "sun_reference_stis_002.fits"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(s3.http_get(config.SOLAR_SPECTRUM_URL).content)
    d = fits.getdata(path)
    wl = d["WAVELENGTH"] / 1e4
    fnu = d["FLUX"] * d["WAVELENGTH"] ** 2
    m = (wl > 2.5) & (wl < 2.65)
    k = np.median(fnu[m] / _planck_nu(wl[m]))
    grid = np.linspace(0.6, 5.2, 4600)
    return grid, np.where(grid < 2.6, np.interp(grid, wl, fnu), k * _planck_nu(grid))


def _planck_nu(wl_um, t=5778.0):
    return wl_um**-3 / np.expm1(14387.77 / (wl_um * t))


def solar_in_band(lam, bw, sun=None) -> np.ndarray:
    grid, f = sun or solar_spectrum()
    return np.array([f[np.abs(grid - l) <= b / 2].mean() for l, b in zip(lam, bw)])


def reflectance(phot: pd.DataFrame, track: pd.DataFrame) -> pd.DataFrame:
    """Geometry-corrected, Sun-divided reflectance normalised at NORM_WINDOW."""
    p = phot[phot.status == "ok"].copy()
    p = pd.concat([p.reset_index(drop=True), ephemeris.geometry_at(track, p.mjd.values)], axis=1)
    g = (p.r_au * p.delta_au) ** 2 / np.mean(p.r_au * p.delta_au) ** 2
    sun = solar_in_band(p.lam_um, p.bw_um)
    p["refl"], p["refl_err"] = p.flux_mJy * g / sun, p.err_mJy * g / sun
    p["reliable"] = (p.snr > RELIABLE["snr"]) & (p.chi2_dof < RELIABLE["chi2_dof"]) & (p.lam_um < THERMAL_FROM_UM)
    if "blended" in p:
        p["reliable"] &= ~p.blended.fillna(False).astype(bool)
    w = p[(p.lam_um > NORM_WINDOW[0]) & (p.lam_um < NORM_WINDOW[1]) & (p.snr > RELIABLE["snr"])]
    if w.empty:
        raise ValueError("no measurements in the normalisation window")
    norm = w.refl.median()
    p["refl"] /= norm
    p["refl_err"] /= norm
    return p.sort_values("lam_um").reset_index(drop=True)


def simultaneous_colours(refl: pd.DataFrame, max_chi2: float = 5.0) -> pd.DataFrame:
    """Long/short reflectance ratio from detector pairs exposed at the same instant (rotation cancels)."""
    q = refl[(refl.snr > RELIABLE["snr"]) & (refl.chi2_dof < max_chi2)]
    short = q[q.det.isin([a for a, _ in config.DETECTOR_PAIRS])].set_index("obsid")
    long_ = q[q.det.isin([b for _, b in config.DETECTOR_PAIRS])].set_index("obsid")
    cols = ["det", "lam_um", "refl", "refl_err", "star_overlap", "block"]
    j = short[cols].join(long_[cols[:-1]], lsuffix="_s", rsuffix="_l", how="inner")
    j["ratio"] = j.refl_l / j.refl_s
    j["ratio_err"] = j.ratio * np.hypot(j.refl_err_l / j.refl_l, j.refl_err_s / j.refl_s)
    return j.reset_index().sort_values("lam_um_l").reset_index(drop=True)


def rotation_scatter(refl: pd.DataFrame, dlam: float = 0.03) -> dict:
    """Magnitude differences between same-detector, same-wavelength sightings in different slew blocks."""
    q = refl[refl.reliable]
    diffs = []
    for _, g in q.groupby("det"):
        r = g.to_dict("records")
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                if r[i]["block"] != r[j]["block"] and abs(r[i]["lam_um"] - r[j]["lam_um"]) < dlam:
                    diffs.append(2.5 * np.log10(r[i]["refl"] / r[j]["refl"]))
    d = np.array(diffs)
    return {"pairs": len(d), "scatter_mag": float(np.std(d)) if len(d) else None,
            "max_abs_mag": float(np.max(np.abs(d))) if len(d) else None}


def summary(refl: pd.DataFrame, colours: pd.DataFrame) -> dict:
    vis = refl[refl.reliable & (refl.lam_um < 1.8)]
    mid = colours[colours.lam_um_l < THERMAL_FROM_UM]
    hot = refl[refl.lam_um > 3.7]
    return {
        "measurements": int(len(refl)),
        "lambda_range_um": [float(refl.lam_um.min()), float(refl.lam_um.max())],
        "snr_range": [float(refl.snr.min()), float(refl.snr.max())],
        "refl_0p75_1p8_range": [float(vis.refl.min()), float(vis.refl.max())] if len(vis) else None,
        "colour_ratio_2p4_3p4_median": float(mid.ratio.median()) if len(mid) else None,
        "colour_ratio_by_block": mid.groupby("block").ratio.median().round(3).to_dict() if len(mid) else {},
        "refl_beyond_3p7_median": float(hot.refl.median()) if len(hot) else None,
        "position_offset_px_median": [float(refl.dx_pix.median()), float(refl.dy_pix.median())],
        "rotation": rotation_scatter(refl),
    }
