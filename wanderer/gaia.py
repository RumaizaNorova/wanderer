"""Gaia DR3 stars for astrometry checks and background-star modelling.

Mirrors are tried in order; from this network ESA and CDS time out but GAVO (Heidelberg) answers.
Results are cached on disk per (position, radius, magnitude limit).
"""
from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd
import pyvo

from . import config

log = logging.getLogger(__name__)

MIRRORS = (
    ("https://dc.zah.uni-heidelberg.de/tap", "gaia.dr3lite", "source_id", "ra", "dec", "pmra", "pmdec", "phot_g_mean_mag"),
    ("https://gea.esac.esa.int/tap-server/tap", "gaiadr3.gaia_source", "source_id", "ra", "dec", "pmra", "pmdec", "phot_g_mean_mag"),
    ("https://tapvizier.cds.unistra.fr/TAPVizieR/tap", '"I/355/gaiadr3"', "Source", "RA_ICRS", "DE_ICRS", "pmRA", "pmDE", "Gmag"),
)
GAIA_EPOCH_MJD = 57388.0  # J2016.0
CACHE = config.DATA_DIR / "cache" / "gaia"


def cone(ra: float, dec: float, radius_deg: float, gmax: float = 18.0) -> pd.DataFrame:
    """Gaia DR3 sources (source_id, ra, dec, pmra, pmdec, g) within a cone, from cache or a mirror."""
    tag = hashlib.sha1(f"{ra:.4f},{dec:.4f},{radius_deg:.3f},{gmax:.1f}".encode()).hexdigest()[:16]
    path = CACHE / f"{tag}.csv"
    if path.exists():
        return pd.read_csv(path)
    last = None
    for url, table, sid, a, d, pa, pd_, g in MIRRORS:
        q = (f"SELECT {sid} AS source_id, {a} AS ra, {d} AS dec, {pa} AS pmra, {pd_} AS pmdec, {g} AS g "
             f"FROM {table} WHERE 1=CONTAINS(POINT('ICRS',{a},{d}), CIRCLE('ICRS',{ra},{dec},{radius_deg})) "
             f"AND {g} < {gmax}")
        try:
            df = pyvo.dal.TAPService(url).run_sync(q, timeout=120, maxrec=2_000_000).to_table().to_pandas()
            CACHE.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            return df
        except Exception as e:  # noqa: BLE001 - try the next mirror
            log.info("Gaia mirror %s failed: %s", url.split("/")[2], str(e)[:100])
            last = e
    raise RuntimeError(f"all Gaia mirrors failed: {last}")


def at_epoch(stars: pd.DataFrame, mjd: float) -> tuple[np.ndarray, np.ndarray]:
    """Positions propagated with proper motion from J2016.0 to `mjd` (linear, fine over ~10 yr)."""
    dt_yr = (mjd - GAIA_EPOCH_MJD) / 365.25
    pmra = np.nan_to_num(stars.pmra.values) / 3.6e6  # mas/yr -> deg/yr (pmra already includes cos(dec))
    pmdec = np.nan_to_num(stars.pmdec.values) / 3.6e6
    dec = stars.dec.values + pmdec * dt_yr
    ra = stars.ra.values + pmra * dt_yr / np.cos(np.radians(dec))
    return ra % 360, dec
