"""C5: Bus-DeMeo class fit for SPHEREx reflectance spectra (0.75-2.45 um overlap).

Templates: Bus-DeMeo mean spectra (DeMeo et al. 2009), MIT copy
http://smass.mit.edu/_documents/busdemeo-meanspectra.xlsx, cached in data/reference/.
Each template is rescaled to the data by weighted least squares; chi2 uses the measurement error
plus the class's own spread (sigma) in quadrature, so broad classes are not unfairly favoured.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import config, s3

TEMPLATE_URL = "http://smass.mit.edu/_documents/busdemeo-meanspectra.xlsx"
FIT_RANGE_UM = (0.75, 2.45)
COMPLEXES = {
    "S": ["S", "Sa", "Sq", "Sr", "Sv", "Q", "K", "L", "A", "R", "O", "V"],
    "C": ["B", "C", "Cb", "Cg", "Cgh", "Ch"],
    "X": ["X", "Xc", "Xe", "Xk", "Xn", "T"],
    "D": ["D"],
}


@lru_cache(maxsize=1)
def templates() -> pd.DataFrame:
    """Long table: class, wavelength_um, mean, sigma."""
    path = config.REFERENCE_DIR / "busdemeo-meanspectra.xlsx"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(s3.http_get(TEMPLATE_URL).content)
    raw = pd.read_excel(path, header=None)
    head = raw.iloc[1].tolist()
    body = raw.iloc[2:].apply(pd.to_numeric, errors="coerce")
    wl = body.iloc[:, 0].values
    rows = []
    for j, name in enumerate(head):
        if isinstance(name, str) and name.endswith("_Mean"):
            cls = name[:-5]
            sigma = body.iloc[:, j + 1].values.astype(float)
            single = bool(np.all(sigma < 0))  # one-object classes (Cg, O, R) store -0.999: no spread
            rows.append(pd.DataFrame({"class": cls, "wavelength_um": wl, "mean": body.iloc[:, j].values,
                                      "sigma": np.where(sigma < 0, np.nan, sigma), "single_object": single}))
    t = pd.concat(rows, ignore_index=True).dropna(subset=["mean"])
    typical = t.groupby("class").sigma.median().median()
    t["sigma"] = t.sigma.fillna(typical)  # a single-object class gets a typical class spread
    return t


def complex_of(cls: str) -> str:
    return next((k for k, v in COMPLEXES.items() if cls in v), "?")


def fit(lam, refl, err) -> pd.DataFrame:
    """Rank all classes by reduced chi2 against reflectance points (any normalisation)."""
    lam, refl, err = map(np.asarray, (lam, refl, err))
    m = (lam >= FIT_RANGE_UM[0]) & (lam <= FIT_RANGE_UM[1]) & np.isfinite(refl) & (err > 0)
    lam, refl, err = lam[m], refl[m], err[m]
    out = []
    for cls, t in templates().groupby("class"):
        mu = np.interp(lam, t.wavelength_um, t["mean"])
        sd = np.interp(lam, t.wavelength_um, t.sigma)
        k = np.sum(refl * mu / err**2) / np.sum(mu**2 / err**2)  # scale template to the data
        s2 = err**2 + (k * sd) ** 2
        chi2 = float(np.sum((refl - k * mu) ** 2 / s2))
        out.append({"class": cls, "complex": complex_of(cls), "chi2_red": chi2 / max(len(lam) - 1, 1), "scale": k,
                    "single_object_class": bool(t.single_object.iloc[0])})
    res = pd.DataFrame(out).sort_values("chi2_red").reset_index(drop=True)
    res.attrs["n_points"] = int(len(lam))
    res.attrs["lambda_max_um"] = float(lam.max()) if len(lam) else None
    return res


def slope_per_um(lam, refl, lo=0.85, hi=1.7) -> float:
    """Spectral slope (fraction per um) of normalised reflectance between lo and hi."""
    lam, refl = np.asarray(lam), np.asarray(refl)
    m = (lam >= lo) & (lam <= hi)
    return float(np.polyfit(lam[m], refl[m] / np.median(refl[m]), 1)[0]) if m.sum() >= 3 else float("nan")
