"""Validation targets: known asteroids inside an indexed period that have published taxonomy and size."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord

from . import config, ephemeris  # noqa: F401  (ephemeris used for H in group_report)

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


def score(targets: pd.DataFrame, period: str) -> pd.DataFrame:
    """Published vs measured taxonomy for each target that has a summary."""
    import json
    rows = []
    for t in targets.itertuples():
        path = config.RESULTS_DIR / str(t.number) / period / "summary.json"
        row = {"number": t.number, "name": t.name, "vmag": t.vmag, "published_class": t.taxonomy_class,
               "published_complex": t.complex, "published_diameter_km": t.diameter_km, "published_albedo": t.albedo}
        if path.exists():
            s = json.loads(path.read_text())
            tax = s.get("taxonomy", {})
            ranking = tax.get("complex_ranking", {})
            row.update({"measurements": s.get("measurements"), "points_fit": tax.get("points"),
                        "snr_median": None, "our_class": tax.get("best_class"),
                        "our_complex": next(iter(ranking), None), "chi2_red": tax.get("best_chi2_red"),
                        "slope_per_um": tax.get("slope_0p85_1p7_per_um"),
                        "colour_2p4_3p4": s.get("colour_ratio_2p4_3p4_median"),
                        "thermal_refl_3p7": s.get("refl_beyond_3p7_median"),
                        "neatm_diameter_km": (s.get("neatm") or {}).get("diameter_km"),
                        "neatm_albedo": (s.get("neatm") or {}).get("albedo"),
                        "neatm_chi2_red": (s.get("neatm") or {}).get("chi2_red")})
        rows.append(row)
    r = pd.DataFrame(rows)
    if "our_complex" in r:
        r["complex_match"] = r.our_complex == r.published_complex
        r["class_match"] = r.our_class == r.published_class
    return r


def group_report(targets: pd.DataFrame, period: str, etas=np.round(np.arange(0.6, 1.81, 0.05), 2)) -> pd.DataFrame:
    """Leave-one-out validation of the albedo + colour group classifier.

    For each object, eta is chosen to minimise the median |log(D/D_published)| over the OTHER
    objects, then that eta gives this object's albedo. The object's own size never tunes its grade.
    """
    from . import neatm, reflectance, taxonomy
    objs = []
    for t in targets.itertuples():
        d = config.RESULTS_DIR / str(t.number) / period
        if not (d / "reflectance.csv").exists():
            continue
        refl = pd.read_csv(d / "reflectance.csv")
        col = reflectance.simultaneous_colours(refl)
        col = col[col.lam_um_l < 3.4]
        h = float(ephemeris.physical_parameters(str(t.number))["H"])
        sizes = {}
        for e in etas:
            f = neatm.fit(refl, h, eta=float(e))
            if f.get("status") == "ok":
                sizes[float(e)] = (f["diameter_km"], f["albedo"])
        objs.append({"number": str(t.number), "name": t.name, "published_class": t.taxonomy_class,
                     "published_complex": t.complex, "published_diameter_km": t.diameter_km, "published_albedo": t.albedo,
                     "colour": float(col.ratio.median()) if len(col) else None, "colour_pairs": int(len(col)), "sizes": sizes})
    rows = []
    for o in objs:
        others = [p for p in objs if p is not o and p["sizes"] and np.isfinite(p["published_diameter_km"])]
        def loss(e):
            v = [abs(np.log(p["sizes"][e][0] / p["published_diameter_km"])) for p in others if e in p["sizes"]]
            return np.median(v) if v else np.inf
        eta = min(etas, key=loss) if others else 1.0
        d_km, alb = o["sizes"].get(float(eta), (None, None))
        g = taxonomy.group(o["colour"], alb)
        rows.append({k: v for k, v in o.items() if k != "sizes"} | {
            "eta_loo": float(eta), "diameter_km": d_km, "albedo": alb, "group": g,
            "group_complex": taxonomy.GROUP_TO_COMPLEX.get(g) if g else None,
            "d_ratio": (d_km / o["published_diameter_km"]) if d_km else None})
    r = pd.DataFrame(rows)
    r["complex_match"] = r.group_complex == r.published_complex
    return r


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
