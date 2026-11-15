"""Validation targets: known asteroids inside an indexed period that have published taxonomy and size."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord

from . import config, ephemeris

log = logging.getLogger(__name__)


def ecliptic_fields(index: pd.DataFrame, det: int = 1, max_beta: float = 15.0, n: int = 40) -> pd.DataFrame:
    d = index[index.det == det].copy()
    beta = SkyCoord(d.ra.values * u.deg, d.dec.values * u.deg).barycentrictrueecliptic.lat.deg
    d = d[np.abs(beta) < max_beta].sort_values("mjd")
    return d.iloc[np.linspace(0, len(d) - 1, min(n, len(d))).astype(int)]


def candidates(index: pd.DataFrame, n_fields: int = 40, vmag: float = 16.5, workers: int = 6) -> pd.DataFrame:
    """Known objects brighter than `vmag` in a spread of ecliptic fields (JPL SB Identification)."""
    fields = ecliptic_fields(index, n=n_fields)

    def q(r):
        try:
            d = ephemeris.sb_ident(r.ra, r.dec, r.mjd, vmag_lim=vmag)
            return d.assign(obsid=r.obsid, field_mjd=r.mjd)
        except Exception as e:  # noqa: BLE001
            log.info("sb_ident failed for %s: %s", r.obsid, e)
            return pd.DataFrame()

    with ThreadPoolExecutor(workers) as ex:
        res = pd.concat(list(ex.map(q, fields.itertuples())), ignore_index=True)
    if res.empty:
        return res
    res["vmag"] = pd.to_numeric(res.vmag, errors="coerce")
    res["number"] = res.name.str.extract(r"^(\d+)\s")[0]
    return res.dropna(subset=["number"]).drop_duplicates("number").sort_values("vmag").reset_index(drop=True)


def with_reference(cands: pd.DataFrame, workers: int = 6) -> pd.DataFrame:
    with ThreadPoolExecutor(workers) as ex:
        refs = list(ex.map(ephemeris.ssodnet_reference, cands.number))
    out = pd.concat([cands.reset_index(drop=True), pd.DataFrame(refs)], axis=1)
    out["complex"] = out.taxonomy_class.map(lambda c: _complex(c) if isinstance(c, str) else None)
    return out


# Tholen-scheme letters that SsODNet may return, mapped to Bus-DeMeo complexes
_THOLEN = {"M": "X", "P": "X", "E": "X", "F": "C", "G": "C", "B": "C", "D": "D", "T": "X"}


def _complex(c: str) -> str:
    from .taxonomy import complex_of
    k = complex_of(c)
    return k if k != "?" else _THOLEN.get(c[:1], c[:1])


def select(ref: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Pick up to n objects with a published class, spreading across complexes, brightest first."""
    ok = ref.dropna(subset=["taxonomy_class"]).sort_values("vmag")
    picks, per = [], max(2, n // max(ok.complex.nunique(), 1))
    for _, g in ok.groupby("complex"):
        picks.append(g.head(per))
    out = pd.concat(picks).sort_values("vmag")
    if len(out) < n:
        out = pd.concat([out, ok[~ok.number.isin(out.number)].head(n - len(out))])
    return out.head(n).reset_index(drop=True)
