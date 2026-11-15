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


BLOCK_FIT_MAX_UM = 2.6     # templates end at 2.45 um; extended flat to 2.6 to use the bottom of detector 4
MIN_LEVER_UM = 0.8         # a block must span this much wavelength to constrain colour
MIN_SHAPE_POINTS = 6


def _template_at(t: pd.DataFrame, lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.interp(lam, t.wavelength_um, t["mean"]), np.interp(lam, t.wavelength_um, t.sigma)  # flat beyond ends


def coverage(df: pd.DataFrame) -> dict:
    """How much rotation-free shape information the points carry (per slew block)."""
    d = df[(df.lam_um >= FIT_RANGE_UM[0]) & (df.lam_um <= BLOCK_FIT_MAX_UM)]
    per_block = d.groupby("block").lam_um.agg(["count", "min", "max"])
    per_block["span"] = per_block["max"] - per_block["min"]
    useful = per_block[per_block["count"] >= 2]
    return {"points": int(len(d)), "blocks": int(len(per_block)), "shape_points": int(useful["count"].sum()),
            "max_block_span_um": float(useful.span.max()) if len(useful) else 0.0,
            "total_span_um": float(d.lam_um.max() - d.lam_um.min()) if len(d) else 0.0,
            "detectors": sorted(int(x) for x in d.det.unique()) if "det" in d else []}


def sufficient(cov: dict) -> bool:
    return cov["max_block_span_um"] >= MIN_LEVER_UM and cov["shape_points"] >= MIN_SHAPE_POINTS


def fit_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Rotation-robust class fit: every slew block gets its own free scale.

    Points inside one block were taken within ~6 minutes, at one rotation phase, so their
    relative fluxes carry true spectral shape; offsets *between* blocks are rotation and are
    absorbed by the per-block scales. Only blocks with >= 2 points contribute.
    """
    d = df[(df.lam_um >= FIT_RANGE_UM[0]) & (df.lam_um <= BLOCK_FIT_MAX_UM) & (df.refl_err > 0)]
    d = d[d.groupby("block").lam_um.transform("count") >= 2]
    out = []
    for cls, t in templates().groupby("class"):
        chi2, n = 0.0, 0
        for _, b in d.groupby("block"):
            mu, sd = _template_at(t, b.lam_um.values)
            r, e = b.refl.values, b.refl_err.values
            c = np.sum(r * mu / e**2) / np.sum(mu**2 / e**2)
            s2 = e**2 + (c * sd * 0.5) ** 2  # half the class spread: shape within a block varies less than across members
            c = np.sum(r * mu / s2) / np.sum(mu**2 / s2)
            chi2 += float(np.sum((r - c * mu) ** 2 / s2))
            n += len(b) - 1
        out.append({"class": cls, "complex": complex_of(cls), "chi2_red": chi2 / max(n, 1),
                    "single_object_class": bool(t.single_object.iloc[0])})
    res = pd.DataFrame(out).sort_values("chi2_red").reset_index(drop=True)
    res.attrs.update(coverage(df))
    return res


# --- Group classification from what one planning period measures robustly --------------------
# Boundaries come from the literature and the Bus-DeMeo templates, NOT from the validation set:
#   albedo 0.10 separates dark (C, P/D) from bright (S, M/E) asteroids (e.g. DeMeo & Carry 2013);
#   rotation-free colour R(2.4-3.4um)/R(0.8-1.1um): highest C-complex template 1.24, T 1.29, D 1.59 -> 1.27;
#   among bright objects, Xe/Xc/Xn templates sit at 1.05-1.10 and S at 1.24-1.32 -> 1.15.
ALBEDO_SPLIT = 0.10
DARK_COLOUR_SPLIT = 1.27
BRIGHT_COLOUR_SPLIT = 1.15
GROUP_TO_COMPLEX = {"C": "C", "P/D": "X", "S": "S", "E/M": "X"}


def template_colours() -> pd.DataFrame:
    rows = []
    for c, g in templates().groupby("class"):
        short = g[(g.wavelength_um >= 0.8) & (g.wavelength_um <= 1.1)]["mean"].mean()
        long_ = g[g.wavelength_um >= 2.35]["mean"].mean()
        rows.append({"class": c, "complex": complex_of(c), "colour": long_ / short, "single": bool(g.single_object.iloc[0])})
    return pd.DataFrame(rows).sort_values("colour").reset_index(drop=True)


def group(colour: float | None, albedo: float | None) -> str | None:
    """Albedo + rotation-free colour -> C, P/D, S or E/M (None when either is missing)."""
    if colour is None or albedo is None or not np.isfinite(colour) or not np.isfinite(albedo):
        return None
    if albedo < ALBEDO_SPLIT:
        return "C" if colour < DARK_COLOUR_SPLIT else "P/D"
    return "E/M" if colour < BRIGHT_COLOUR_SPLIT else "S"


def slope_per_um(lam, refl, lo=0.85, hi=1.7) -> float:
    """Spectral slope (fraction per um) of normalised reflectance between lo and hi."""
    lam, refl = np.asarray(lam), np.asarray(refl)
    m = (lam >= lo) & (lam <= hi)
    return float(np.polyfit(lam[m], refl[m] / np.median(refl[m]), 1)[0]) if m.sum() >= 3 else float("nan")
