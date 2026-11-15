"""Project-wide settings. Override the data folder with the WANDERER_DATA environment variable."""
from __future__ import annotations

import os
from pathlib import Path

BUCKET_URL = "https://nasa-irsa-spherex.s3.us-east-1.amazonaws.com/"
RELEASE = "dr1"  # DR1 + QR3 is the uniformly processed set as of Nov 2026; QR2 is retired

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("WANDERER_DATA", ROOT / "data"))
INDEX_DIR = DATA_DIR / "index"          # exposure indexes (regenerable, git-ignored)
REFERENCE_DIR = DATA_DIR / "reference"  # downloaded reference files (git-ignored)
RESULTS_DIR = DATA_DIR / "results"      # per-target tables and figures (committed)

DETECTORS = (1, 2, 3, 4, 5, 6)
# Detector pairs behind the dichroic that see the same sky at the same instant.
# D1/D4 verified on 2025W18_2B (centres 0.02 deg apart); D2/D5 and D3/D6 follow the same design.
DETECTOR_PAIRS = ((1, 4), (2, 5), (3, 6))
IMAGE_SIZE = 2040
PIXEL_SCALE_ARCSEC = 6.15

# Network behaviour
HTTP_TIMEOUT_S = 60
HTTP_RETRIES = 4
WORKERS = 16
FSSPEC_BLOCK = 2**16  # small blocks: we read headers and cutouts, never whole files

SOLAR_SPECTRUM_URL = (
    "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/sun_reference_stis_002.fits"
)
