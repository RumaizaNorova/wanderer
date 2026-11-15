"""Find every exposure that contains a known object and where it falls on the detector."""
from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS, FITSFixedWarning

from . import calib, config, ephemeris, s3

log = logging.getLogger(__name__)
warnings.simplefilter("ignore", FITSFixedWarning)


def find_sightings(target: str, index: pd.DataFrame, margin: int = 8, workers: int = config.WORKERS,
                   spacecraft_frame: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sightings, track). A sighting = one exposure with the predicted position on the detector."""
    track = ephemeris.horizons_track(target, index.mjd.min() - 0.05, index.mjd.max() + 0.05)
    ra, dec = ephemeris.position_at(track, index.mjd.values)
    pred = SkyCoord(ra * u.deg, dec * u.deg)
    centre = SkyCoord(index.ra.values * u.deg, index.dec.values * u.deg)
    near = pred.separation(centre).deg < 2.6  # field half-diagonal ~2.45 deg
    cand = index[near].assign(pra=ra[near], pdec=dec[near])
    log.info("%s: %d candidate exposures near the predicted track", target, len(cand))

    def locate(row) -> dict | None:
        x, y = WCS(s3.read_header(row.key, hdu=1), naxis=2).all_world2pix([[row.pra, row.pdec]], 0)[0]
        if not calib.in_bounds(x, y, margin):
            return None
        return {"key": row.key, "obsid": row.obsid, "det": int(row.det), "mjd": row.mjd,
                "ra_pred": row.pra, "dec_pred": row.pdec, "x": float(x), "y": float(y)}

    with ThreadPoolExecutor(workers) as ex:
        found = [r for r in ex.map(locate, cand.itertuples()) if r]
    if found and spacecraft_frame:
        # refine: positions as seen from SPHEREx at each exact exposure time
        times = sorted({r["mjd"] for r in found})
        exact = ephemeris.positions_at_times(target, times).set_index(pd.Index(times))
        refined = cand.assign(pra=exact.ra.reindex(cand.mjd).values, pdec=exact.dec.reindex(cand.mjd).values)
        refined = refined[refined.key.isin({r["key"] for r in found})]
        with ThreadPoolExecutor(workers) as ex:
            found = [r for r in ex.map(locate, refined.itertuples()) if r]
    s = pd.DataFrame(found).sort_values(["mjd", "det"]).reset_index(drop=True)
    s["frame"] = "spherex" if spacecraft_frame else "geocentric"
    if len(s):
        s["block"] = s.obsid.str.rsplit("_", n=1).str[0]  # one large slew: ~6 min, same rotation phase
    return s, track
