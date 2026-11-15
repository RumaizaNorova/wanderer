"""Survey geometry and revisit cadence inside one planning period."""
from __future__ import annotations

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord

from . import config
from .index import corners


def pair_separation(index: pd.DataFrame, a: int, b: int) -> np.ndarray:
    """Centre separation (deg) between detectors a and b for the same observation."""
    da, db = index[index.det == a].set_index("obsid"), index[index.det == b].set_index("obsid")
    both = da.index.intersection(db.index)
    ca = SkyCoord(da.loc[both, "ra"].values * u.deg, da.loc[both, "dec"].values * u.deg)
    cb = SkyCoord(db.loc[both, "ra"].values * u.deg, db.loc[both, "dec"].values * u.deg)
    return ca.separation(cb).deg


def field_side_deg(index: pd.DataFrame) -> float:
    c = corners(index.iloc[0])
    return SkyCoord(c[0, 0] * u.deg, c[0, 1] * u.deg).separation(SkyCoord(c[1, 0] * u.deg, c[1, 1] * u.deg)).deg


def revisit_stats(index: pd.DataFrame, det: int = 1) -> dict:
    """Looks per field and time gaps between overlapping looks, using field centres."""
    d = index[index.det == det].reset_index(drop=True)
    side = field_side_deg(d)
    cen = SkyCoord(d.ra.values * u.deg, d.dec.values * u.deg)
    i1, i2, _, _ = cen.search_around_sky(cen, (side / 2) * u.deg)
    counts = np.bincount(i1, minlength=len(d))
    gaps = []
    for i in range(len(d)):
        t = np.sort(d.mjd.values[i2[i1 == i]])
        gaps.extend(np.diff(t) * 24 * 60)
    gaps = np.array(gaps)
    beta = np.abs(cen.barycentrictrueecliptic.lat.deg)
    return {
        "exposures": len(d),
        "span_hours": float((d.mjd.max() - d.mjd.min()) * 24),
        "field_side_deg": float(side),
        "exptime_s": float(d.exptime_s.median()),
        "psf_fwhm_arcsec": float(d.psf_fwhm_arcsec.median()),
        "looks_per_field_median": float(np.median(counts)),
        "looks_per_field_p10_p90": np.percentile(counts, [10, 90]).tolist(),
        "gap_minutes_p10_p50_p90": np.percentile(gaps, [10, 50, 90]).round(1).tolist() if len(gaps) else [],
        "fields_within_10deg_of_ecliptic": int((beta < 10).sum()),
        "pair_separation_deg": {f"D{a}/D{b}": float(np.median(pair_separation(index, a, b)))
                                for a, b in config.DETECTOR_PAIRS
                                if {a, b} <= set(index.det.unique())},
    }
