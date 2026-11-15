"""C1 store: exposures in a local DuckDB file with fast sky-position lookup.

DuckDB needs no server; the schema is plain SQL so it ports to Postgres (+q3c) in the cloud.
Lookup = a declination band + unit-vector dot product (cheap, uses zone maps), then an exact
point-in-footprint test on the four corners in a tangent-plane projection.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import config

DB_PATH = config.DATA_DIR / "wanderer.duckdb"
FIELD_RADIUS_DEG = 2.6  # > half-diagonal of a 3.47 deg field

SCHEMA = """
CREATE TABLE IF NOT EXISTS exposures (
    key VARCHAR PRIMARY KEY, release VARCHAR, period VARCHAR, obsid VARCHAR, det INTEGER,
    mjd DOUBLE, mjd_beg DOUBLE, mjd_end DOUBLE, exptime_s DOUBLE,
    ra DOUBLE, dec DOUBLE, cx DOUBLE, cy DOUBLE, cz DOUBLE,
    corners VARCHAR, psf_fwhm_arcsec DOUBLE, processing VARCHAR
);
CREATE TABLE IF NOT EXISTS periods (
    release VARCHAR, period VARCHAR, n_exposures INTEGER, detectors VARCHAR,
    mjd_min DOUBLE, mjd_max DOUBLE, loaded_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (release, period)
);
"""


def connect(path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(SCHEMA)
    return con


def _unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)


def load_period(con, index: pd.DataFrame, period: str, release: str = config.RELEASE) -> int:
    df = index.copy()
    df["release"], df["period"] = release, period
    df["cx"], df["cy"], df["cz"] = _unit(df.ra.values, df.dec.values)
    for col in ("mjd_beg", "mjd_end", "processing"):
        if col not in df:
            df[col] = None
    if "psf_fwhm_arcsec" not in df and "psf_fwhm" in df:
        df["psf_fwhm_arcsec"] = df["psf_fwhm"]
    cols = ["key", "release", "period", "obsid", "det", "mjd", "mjd_beg", "mjd_end", "exptime_s",
            "ra", "dec", "cx", "cy", "cz", "corners", "psf_fwhm_arcsec", "processing"]
    con.register("incoming", df[cols])
    con.execute(f"INSERT OR REPLACE INTO exposures SELECT {', '.join(cols)} FROM incoming")
    con.unregister("incoming")
    con.execute("""INSERT OR REPLACE INTO periods (release, period, n_exposures, detectors, mjd_min, mjd_max)
                   SELECT ?, ?, count(*), string_agg(DISTINCT det::VARCHAR, ',' ORDER BY det::VARCHAR),
                          min(mjd), max(mjd) FROM exposures WHERE release = ? AND period = ?""",
                [release, period, release, period])
    return len(df)


def _inside(ra0, dec0, ra_c, dec_c, corners_str) -> np.ndarray:
    """Is (ra0, dec0) inside each exposure's quadrilateral? Gnomonic projection about each centre."""
    out = np.zeros(len(corners_str), bool)
    for i, (rc, dc, cs) in enumerate(zip(ra_c, dec_c, corners_str)):
        pts = np.array([[float(v) for v in c.split(",")] for c in cs.split(";")] + [[ra0, dec0]])
        a, d, a0, d0 = np.radians(pts[:, 0]), np.radians(pts[:, 1]), np.radians(rc), np.radians(dc)
        cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
        x = np.cos(d) * np.sin(a - a0) / cosc
        y = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / cosc
        q, p = np.stack([x[:4], y[:4]], 1), np.array([x[4], y[4]])
        e = np.roll(q, -1, 0) - q
        cross = e[:, 0] * (p[1] - q[:, 1]) - e[:, 1] * (p[0] - q[:, 0])
        out[i] = np.all(cross >= 0) or np.all(cross <= 0)
    return out


def exposures_covering(con, ra: float, dec: float, mjd_min: float | None = None, mjd_max: float | None = None,
                       detectors=None, release: str = config.RELEASE) -> pd.DataFrame:
    """Every indexed exposure whose footprint contains (ra, dec), optionally within a time window."""
    x, y, z = _unit(ra, dec)
    sql = ["SELECT * FROM exposures WHERE release = ? AND dec BETWEEN ? AND ?",
           "AND cx * ? + cy * ? + cz * ? > ?"]
    args = [release, dec - FIELD_RADIUS_DEG, dec + FIELD_RADIUS_DEG, x, y, z, np.cos(np.radians(FIELD_RADIUS_DEG))]
    if mjd_min is not None:
        sql.append("AND mjd >= ?"); args.append(mjd_min)
    if mjd_max is not None:
        sql.append("AND mjd <= ?"); args.append(mjd_max)
    if detectors:
        sql.append(f"AND det IN ({', '.join('?' * len(detectors))})"); args += list(detectors)
    cand = con.execute(" ".join(sql) + " ORDER BY mjd, det", args).df()
    if cand.empty:
        return cand
    return cand[_inside(ra, dec, cand.ra.values, cand.dec.values, cand.corners.values)].reset_index(drop=True)


def periods(con) -> pd.DataFrame:
    return con.execute("SELECT * FROM periods ORDER BY release, period").df()
