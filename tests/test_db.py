"""Offline tests for the exposure store (synthetic footprints, temporary DuckDB file)."""
import time

import numpy as np
import pandas as pd

from wanderer import db


def _square(ra, dec, half=1.7):
    """Approximate 3.4 deg field corners around (ra, dec), in the index's 'ra,dec;...' format."""
    c = np.cos(np.radians(dec))
    pts = [(ra - half / c, dec - half), (ra + half / c, dec - half), (ra + half / c, dec + half), (ra - half / c, dec + half)]
    return ";".join(f"{a:.5f},{d:.5f}" for a, d in pts)


def _index(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ra, dec = rng.uniform(0, 360, n), np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    return pd.DataFrame({
        "key": [f"k{i}" for i in range(n)], "obsid": [f"o{i}" for i in range(n)], "det": rng.integers(1, 7, n),
        "mjd": rng.uniform(60796, 60800, n), "exptime_s": 115.1, "ra": ra, "dec": dec,
        "corners": [_square(a, d) for a, d in zip(ra, dec)], "psf_fwhm_arcsec": 7.5,
    })


def test_load_and_cone_lookup(tmp_path):
    idx = _index()
    with db.connect(tmp_path / "t.duckdb") as con:
        assert db.load_period(con, idx, "TEST") == len(idx)
        assert db.load_period(con, idx, "TEST") == len(idx)  # idempotent: replaces, no duplicates
        assert con.execute("SELECT count(*) FROM exposures").fetchone()[0] == len(idx)
        target = idx.iloc[123]
        t = time.perf_counter()
        hits = db.exposures_covering(con, target.ra, target.dec)
        assert time.perf_counter() - t < 1.0  # plan acceptance test: under 1 second
        assert "k123" in set(hits.key)
        far = db.exposures_covering(con, (target.ra + 180) % 360, -target.dec)
        assert "k123" not in set(far.key)


def test_point_in_footprint_edges():
    corners = _square(100.0, 20.0)
    assert db._inside(100.0, 20.0, [100.0], [20.0], [corners])[0]
    assert db._inside(100.0, 21.6, [100.0], [20.0], [corners])[0]
    assert not db._inside(100.0, 21.9, [100.0], [20.0], [corners])[0]


def test_time_and_detector_filters(tmp_path):
    idx = _index(500, seed=3)
    with db.connect(tmp_path / "t.duckdb") as con:
        db.load_period(con, idx, "TEST")
        r = idx.iloc[7]
        hits = db.exposures_covering(con, r.ra, r.dec, mjd_min=r.mjd - 1e-6, mjd_max=r.mjd + 1e-6, detectors=[int(r.det)])
        assert list(hits.key) == ["k7"]
