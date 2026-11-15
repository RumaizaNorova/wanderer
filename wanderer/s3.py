"""Anonymous access to the public SPHEREx bucket: listing, header reads, and remote FITS handles.

Nothing here downloads whole files. Headers come from HTTP range requests and pixels from
fsspec-backed FITS sections, so a 70 MB spectral image costs a few hundred KB per cutout.
"""
from __future__ import annotations

import re
import threading
import time
from functools import lru_cache

import requests
from astropy.io import fits

from . import config

_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def http_get(url: str, *, params=None, headers=None, timeout: float = config.HTTP_TIMEOUT_S) -> requests.Response:
    """GET with retries on connection errors and 5xx responses."""
    last = None
    for attempt in range(config.HTTP_RETRIES):
        try:
            r = _session().get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code < 500 and r.status_code != 429:
                r.raise_for_status()
                return r
            last = requests.HTTPError(f"{r.status_code} for {r.url}")
            if r.status_code == 429:  # rate limited (JPL APIs): honour Retry-After, else back off longer
                time.sleep(float(r.headers.get("Retry-After", 5 * 2**attempt)))
                continue
        except (requests.ConnectionError, requests.Timeout) as e:
            last = e
        time.sleep(1.5 * 2**attempt)
    raise last  # type: ignore[misc]


def url(key: str) -> str:
    return config.BUCKET_URL + key


_KEY_RE = re.compile(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", re.S)
_PREFIX_RE = re.compile(r"<Prefix>([^<]+)</Prefix>")
_TOKEN_RE = re.compile(r"<NextContinuationToken>([^<]+)</NextContinuationToken>")


def parse_listing(xml: str) -> tuple[list[tuple[str, int]], list[str], str | None]:
    """Parse one ListObjectsV2 page into (keys with sizes, common prefixes, continuation token)."""
    keys = [(k, int(s)) for k, s in _KEY_RE.findall(xml)]
    prefixes = _PREFIX_RE.findall(xml)[1:]  # the first <Prefix> echoes the request
    m = _TOKEN_RE.search(xml)
    return keys, prefixes, (m.group(1) if m else None)


def list_keys(prefix: str, delimiter: str | None = None) -> tuple[list[tuple[str, int]], list[str]]:
    """All keys (with sizes) and sub-prefixes under a prefix, following pagination."""
    keys: list[tuple[str, int]] = []
    prefixes: list[str] = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        k, p, token = parse_listing(http_get(config.BUCKET_URL, params=params).text)
        keys += k
        prefixes += p
        if not token:
            return keys, prefixes


def _header_end(raw: bytes, start: int) -> int | None:
    """Byte offset just past the 2880-byte block holding the END card of the header at `start`."""
    for off in range(start, len(raw) - 79, 80):
        if raw[off:off + 8] == b"END     ":
            block_end = off + 80
            return start + ((block_end - start + 2879) // 2880) * 2880
    return None


def read_header(key: str, hdu: int = 1, chunk: int = 2880 * 20) -> fits.Header:
    """Read the header of extension `hdu` with range requests only (default: the IMAGE extension)."""
    raw = b""
    pos = 0
    for _ in range(hdu + 1):
        while True:
            end = _header_end(raw, pos)
            if end is not None:
                break
            raw += http_get(url(key), headers={"Range": f"bytes={len(raw)}-{len(raw) + chunk - 1}"}).content
        start, pos = pos, end
    return fits.Header.fromstring(raw[start:pos].decode("ascii", "replace"))


def open_fits(key: str, block_size: int = config.FSSPEC_BLOCK) -> fits.HDUList:
    """Remote FITS handle; use `.section[...]` on image HDUs to read only the pixels you need."""
    return fits.open(url(key), use_fsspec=True, fsspec_kwargs={"block_size": block_size, "cache_type": "blockcache"})


@lru_cache(maxsize=None)
def calibration_key(product: str, detector: int, release: str = config.RELEASE) -> str:
    """Key of a per-detector calibration file, e.g. product='spectral_wcs' or 'solid_angle_pixel_map'."""
    keys, _ = list_keys(f"{release}/{product}/")
    hits = sorted(k for k, _ in keys if f"/{detector}/" in k and k.endswith(".fits"))
    if not hits:
        raise FileNotFoundError(f"no {product} file for detector {detector} in {release}")
    return hits[-1]  # latest version
