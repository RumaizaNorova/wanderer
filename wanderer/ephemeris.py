"""Known-object positions: JPL Horizons tracks and JPL Small-Body Identification."""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from . import s3

SB_IDENT_URL = "https://ssd-api.jpl.nasa.gov/sb_ident.api"
SBDB_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"
# Geocentric is used for coarse tracks and SB Identification. Final positions are computed as seen
# from SPHEREx itself (Horizons spacecraft -163182): the low-Earth-orbit parallax is 2-4.4" (0.3-0.7 px)
# for a main-belt asteroid and oscillates with the ~95 min orbit, so it must be evaluated at each
# exposure time, never interpolated from a coarse track.
OBSERVER = "500"
SPHEREX = "500@-163182"


def horizons_track(target: str, mjd_start: float, mjd_end: float, step: str = "30m",
                   location: str = OBSERVER) -> pd.DataFrame:
    """Ephemeris on a regular grid; interpolate with `position_at` / `geometry_at`."""
    eph = Horizons(id=target, location=location, epochs={
        "start": f"JD{mjd_start + 2400000.5:.5f}", "stop": f"JD{mjd_end + 2400000.5:.5f}", "step": step,
    }).ephemerides()
    return pd.DataFrame({
        "mjd": np.asarray(eph["datetime_jd"], float) - 2400000.5,
        "ra": np.asarray(eph["RA"], float), "dec": np.asarray(eph["DEC"], float),
        "r_au": np.asarray(eph["r"], float), "delta_au": np.asarray(eph["delta"], float),
        "phase_deg": np.asarray(eph["alpha"], float), "vmag": np.asarray(eph["V"], float),
        "ra_rate": np.asarray(eph["RA_rate"], float), "dec_rate": np.asarray(eph["DEC_rate"], float),
    })


def positions_at_times(target: str, mjd, location: str = SPHEREX, chunk: int = 40) -> pd.DataFrame:
    """Exact-time ephemeris (no interpolation), queried in chunks to keep Horizons URLs short."""
    mjd = np.asarray(mjd, float)
    parts = []
    for i in range(0, len(mjd), chunk):
        jd = list(mjd[i:i + chunk] + 2400000.5)
        e = Horizons(id=target, location=location, epochs=jd).ephemerides()
        parts.append(pd.DataFrame({"mjd": np.asarray(e["datetime_jd"], float) - 2400000.5,
                                   "ra": np.asarray(e["RA"], float), "dec": np.asarray(e["DEC"], float),
                                   "r_au": np.asarray(e["r"], float), "delta_au": np.asarray(e["delta"], float),
                                   "phase_deg": np.asarray(e["alpha"], float)}))
    out = pd.concat(parts, ignore_index=True)
    # Horizons returns rows sorted by time; map back to the requested order
    order = np.argsort(np.argsort(mjd))
    return out.sort_values("mjd").iloc[order].reset_index(drop=True)


def position_at(track: pd.DataFrame, mjd) -> tuple[np.ndarray, np.ndarray]:
    """RA, Dec (deg) at the given times; RA interpolated through the 0/360 wrap."""
    ra = np.degrees(np.interp(mjd, track.mjd, np.unwrap(np.radians(track.ra)))) % 360
    return ra, np.interp(mjd, track.mjd, track.dec)


def geometry_at(track: pd.DataFrame, mjd) -> pd.DataFrame:
    return pd.DataFrame({c: np.interp(mjd, track.mjd, track[c]) for c in ("r_au", "delta_au", "phase_deg")})


def _ra_sexagesimal(ra_deg: float) -> str:
    total = round((ra_deg % 360) / 15 * 3600, 2) % 86400
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}-{int(m):02d}-{s:05.2f}"


def _dec_sexagesimal(dec_deg: float) -> str:
    sign = "M" if dec_deg < 0 else ""
    total = round(abs(dec_deg) * 3600, 1)
    d, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{sign}{int(d):02d}-{int(m):02d}-{s:04.1f}"


def sb_ident(ra_deg: float, dec_deg: float, mjd: float, half_width_deg: float = 1.75,
             vmag_lim: float = 17.0) -> pd.DataFrame:
    """Known small bodies inside a square field at a given time."""
    t = Time(mjd, format="mjd", scale="utc").strftime("%Y-%m-%d_%H:%M:%S")
    params = {
        "obs-time": t, "mpc-code": OBSERVER,
        "fov-ra-center": _ra_sexagesimal(ra_deg), "fov-dec-center": _dec_sexagesimal(dec_deg),
        "fov-ra-hwidth": half_width_deg, "fov-dec-hwidth": half_width_deg,
        "two-pass": "true", "suppress-first-pass": "true", "vmag-lim": vmag_lim,
    }
    js = s3.http_get(SB_IDENT_URL, params=params, timeout=180).json()
    rows = js.get("data_second_pass") or []
    fields = js.get("fields_second") or []
    df = pd.DataFrame(rows, columns=fields)
    return df.rename(columns={"Object name": "name", "Visual magnitude (V)": "vmag"})


SSODNET_URL = "https://ssp.imcce.fr/webservices/ssodnet/api/ssocard/{}"


def ssodnet_reference(target: str) -> dict:
    """Published taxonomy, diameter and albedo from IMCCE SsODNet (best-estimate 'ssoCard')."""
    try:
        name = physical_parameters(target)["name"].split("(")[0].strip().split(" ", 1)[-1]
        p = s3.http_get(SSODNET_URL.format(name.replace(" ", "_"))).json()["parameters"]["physical"]
    except Exception as e:  # noqa: BLE001 - reference data is optional
        return {"error": str(e)[:120]}
    tax = p.get("taxonomy") or {}
    return {"taxonomy_class": tax.get("class"), "taxonomy_scheme": tax.get("scheme"),
            "taxonomy_waverange": tax.get("waverange"),
            "diameter_km": (p.get("diameter") or {}).get("value"), "albedo": (p.get("albedo") or {}).get("value")}


def physical_parameters(target: str) -> dict:
    """Diameter, albedo, rotation period, H and orbit size from JPL SBDB."""
    js = s3.http_get(SBDB_URL, params={"sstr": target, "phys-par": "1"}).json()
    out = {"name": js["object"]["fullname"]}
    for p in js.get("phys_par", []):
        out[p["name"]] = p["value"]
    out.update({e["name"]: e["value"] for e in js["orbit"]["elements"] if e["name"] in ("a", "e", "i")})
    return out
