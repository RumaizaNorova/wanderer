"""C1: exposure index built from spectral-image headers (time, detector, footprint)."""
from __future__ import annotations

import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.wcs import WCS, FITSFixedWarning

from . import config, s3

log = logging.getLogger(__name__)
warnings.simplefilter("ignore", FITSFixedWarning)

# centre + four corners, 0-based pixel coordinates
_PROBE = np.array([[1019.5, 1019.5], [0, 0], [2039, 0], [2039, 2039], [0, 2039]])


def list_periods(release: str = config.RELEASE) -> list[str]:
    _, prefixes = s3.list_keys(f"{release}/level2/", "/")
    return [p.rstrip("/").split("/")[-1] for p in prefixes]


def list_exposure_keys(period: str, detectors=config.DETECTORS, release: str = config.RELEASE) -> list[str]:
    """Spectral-image keys of one planning period. Keeps the main 'l2b-' processing, not 'l2b_retry-'."""
    keys, _ = s3.list_keys(f"{release}/level2/{period}/")
    dets = {str(d) for d in detectors}
    return [k for k, _ in keys if "/l2b-" in k and k.split("/")[4] in dets and k.endswith(".fits")]


def exposure_record(key: str) -> dict:
    h = s3.read_header(key, hdu=1)
    ra, dec = WCS(h, naxis=2).all_pix2world(_PROBE, 0).T
    return {
        "key": key,
        "obsid": h["OBSID"],
        "det": int(h["DETECTOR"]),
        "mjd": float(h["MJD-AVG"]),
        "mjd_beg": float(h["MJD-BEG"]),
        "mjd_end": float(h["MJD-END"]),
        "exptime_s": round((h["MJD-END"] - h["MJD-BEG"]) * 86400, 1),
        "ra": float(ra[0]),
        "dec": float(dec[0]),
        "corners": ";".join(f"{a:.5f},{d:.5f}" for a, d in zip(ra[1:], dec[1:])),
        "psf_fwhm_arcsec": h.get("PSF_FWHM"),
        "processing": key.split("/")[3],
    }


def index_path(period: str, release: str = config.RELEASE) -> Path:
    return config.INDEX_DIR / f"{release}_{period}.csv"


def build_index(period: str, detectors=config.DETECTORS, release: str = config.RELEASE,
                workers: int = config.WORKERS) -> pd.DataFrame:
    """Index a planning period; detectors already in the saved index are skipped, new ones appended."""
    path = index_path(period, release)
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["key"])
    have = set(old["key"])
    keys = [k for k in list_exposure_keys(period, detectors, release) if k not in have]
    log.info("%s: %d new exposures to index (%d already indexed)", period, len(keys), len(have))
    rows, failed = [], []
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for n, (k, rec) in enumerate(zip(keys, ex.map(_safe_record, keys)), 1):
            (rows if rec else failed).append(rec or k)
            if n % 500 == 0 or n == len(keys):
                rate = n / max(time.time() - t0, 1e-9)
                log.info("  %d/%d headers (%.0f/s, ~%.0f s left)", n, len(keys), rate, (len(keys) - n) / rate)
    if failed:
        log.warning("%d headers could not be read, e.g. %s", len(failed), failed[0])
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if rows else old
    df = df.sort_values(["mjd", "det"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _safe_record(key: str) -> dict | None:
    try:
        return exposure_record(key)
    except Exception as e:  # noqa: BLE001 - one bad header must not stop a 10,000-file index
        log.debug("header failed for %s: %s", key, e)
        return None


def load_index(period: str, release: str = config.RELEASE) -> pd.DataFrame:
    path = index_path(period, release)
    if not path.exists():
        raise FileNotFoundError(f"no index for {period}; run: wanderer index {period}")
    return pd.read_csv(path)


def corners(row) -> np.ndarray:
    return np.array([[float(v) for v in c.split(",")] for c in row["corners"].split(";")])
