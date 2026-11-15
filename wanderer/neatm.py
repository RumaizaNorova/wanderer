"""C5: NEATM thermal fit (Harris 1998) for diameter and geometric albedo.

Thermal flux = measured flux minus the reflected continuum. The continuum is extrapolated from
the 2.4-3.4 um reflectance (thermal emission is negligible there for main-belt objects), so
thermal = flux * (1 - R_cont / R). Diameter and albedo are tied through H:
    D[km] = 1329 / sqrt(pv) * 10**(-H/5)
Free parameter: pv (hence D); eta fixed (default 1.0, the usual value for main-belt asteroids
when only short thermal wavelengths are available) and also reported on a small grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

H_PLANCK, C, K_B, SIGMA = 6.62607015e-34, 2.99792458e8, 1.380649e-23, 5.670374419e-8
S0 = 1361.0         # W m^-2 at 1 au
AU_M = 1.495978707e11
EMISSIVITY = 0.9
G_SLOPE = 0.15      # used for the phase integral q = 0.290 + 0.684 G
THERMAL_FROM_UM = 3.5

_th, _ph = np.meshgrid(np.linspace(-np.pi / 2, np.pi / 2, 91), np.linspace(-np.pi / 2, np.pi / 2, 181), indexing="ij")


def diameter_km(h: float, pv: float) -> float:
    return 1329.0 / np.sqrt(pv) * 10 ** (-h / 5)


def thermal_flux_mjy(lam_um, d_km, pv, r_au, delta_au, alpha_deg, eta=1.0):
    """NEATM thermal flux density (mJy) at each wavelength for one geometry."""
    q = 0.290 + 0.684 * G_SLOPE
    a_bond = q * pv
    tss = ((1 - a_bond) * S0 / (eta * EMISSIVITY * SIGMA * r_au**2)) ** 0.25
    alpha = np.radians(alpha_deg)
    # surface grid: latitude th, longitude ph from the subsolar point; visible if cos(ph - alpha) > 0
    cosz = np.clip(np.cos(_th) * np.cos(_ph), 0, None)
    temp = tss * cosz**0.25
    view = np.cos(_th) ** 2 * np.clip(np.cos(_ph - alpha), 0, None)  # projected area weight (incl. cos th Jacobian)
    dA = (_th[1, 0] - _th[0, 0]) * (_ph[0, 1] - _ph[0, 0])
    out = []
    for lam in np.atleast_1d(lam_um):
        nu = C / (lam * 1e-6)
        with np.errstate(over="ignore"):
            b = 2 * H_PLANCK * nu**3 / C**2 / np.expm1(H_PLANCK * nu / (K_B * np.where(temp > 0, temp, 1e-3)))
        integral = np.sum(b * view) * dA
        f = EMISSIVITY * (d_km * 1e3) ** 2 / (4 * (delta_au * AU_M) ** 2) * integral  # W m^-2 Hz^-1
        out.append(f * 1e29)  # -> mJy
    return np.array(out)


def thermal_excess(refl: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Thermal flux points (mJy) beyond 3.5 um and the reflected continuum level used."""
    cont = refl[(refl.lam_um > 2.4) & (refl.lam_um < 3.4) & (refl.snr > 5)].refl.median()
    t = refl[(refl.lam_um > THERMAL_FROM_UM) & (refl.snr > 5)].copy()
    t["thermal_mJy"] = t.flux_mJy * (1 - cont / t.refl)
    t["thermal_err_mJy"] = np.hypot(t.err_mJy, t.flux_mJy * 0.05 * cont / t.refl)  # 5% continuum uncertainty
    return t, float(cont)


def fit(refl: pd.DataFrame, h_mag: float, eta: float = 1.0) -> dict:
    t, cont = thermal_excess(refl)
    t = t[t.thermal_mJy > 0]
    if len(t) < 2:
        return {"status": "not enough thermal points", "n_points": int(len(t))}

    def chi2(log_pv):
        pv = 10**log_pv
        d = diameter_km(h_mag, pv)
        model = np.array([thermal_flux_mjy(r.lam_um, d, pv, r.r_au, r.delta_au, r.phase_deg, eta)[0] for r in t.itertuples()])
        return float(np.sum(((t.thermal_mJy - model) / t.thermal_err_mJy) ** 2))

    best = minimize_scalar(chi2, bounds=(np.log10(0.01), np.log10(0.8)), method="bounded")
    pv = 10**best.x
    # 1-sigma from delta chi2 = 1 (scan)
    grid = np.linspace(best.x - 0.5, best.x + 0.5, 101)
    c = np.array([chi2(g) for g in grid])
    ok = grid[c <= best.fun + max(1.0, best.fun / max(len(t) - 1, 1))]
    pv_lo, pv_hi = 10 ** ok.min(), 10 ** ok.max()
    eta_scan = {}
    for e in (0.8, 1.0, 1.2):
        b = minimize_scalar(lambda lp: _chi2_eta(t, h_mag, lp, e), bounds=(np.log10(0.01), np.log10(0.8)), method="bounded")
        eta_scan[e] = {"diameter_km": round(diameter_km(h_mag, 10**b.x), 1), "albedo": round(10**b.x, 4)}
    return {"status": "ok", "n_points": int(len(t)), "eta": eta, "continuum_refl": cont,
            "albedo": float(pv), "albedo_range": [float(pv_lo), float(pv_hi)],
            "diameter_km": float(diameter_km(h_mag, pv)),
            "diameter_range_km": [float(diameter_km(h_mag, pv_hi)), float(diameter_km(h_mag, pv_lo))],
            "chi2_red": float(best.fun / max(len(t) - 1, 1)), "eta_sensitivity": eta_scan}


def _chi2_eta(t, h_mag, log_pv, eta):
    pv = 10**log_pv
    d = diameter_km(h_mag, pv)
    model = np.array([thermal_flux_mjy(r.lam_um, d, pv, r.r_au, r.delta_au, r.phase_deg, eta)[0] for r in t.itertuples()])
    return float(np.sum(((t.thermal_mJy - model) / t.thermal_err_mJy) ** 2))
