# WANDERER

Find moving objects in NASA's SPHEREx all-sky survey and measure what they are made of. Built for the NASA Space Apps challenge "Planet X and SPHEREx".

Everything reads public SPHEREx Data Release 1 straight from the AWS bucket with HTTP range requests. A 70 MB spectral image costs a few hundred KB per cutout, so nothing large is downloaded.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
```

## Usage

```powershell
.venv\Scripts\wanderer periods                                 # list DR1 planning periods
.venv\Scripts\wanderer index 2025W18_2B                        # index one period (~11,000 exposures, ~9 min) + load into DuckDB
.venv\Scripts\wanderer covering 162.211 10.773 --det 1,4       # exposures containing a sky position (~20 ms)
.venv\Scripts\wanderer cadence 2025W18_2B                      # field size, revisit cadence, detector pairs
.venv\Scripts\wanderer catalog 2025W18_2B --obsid 2025W18_2B_0266_1 --det 1   # source catalog + Gaia astrometry check
.venv\Scripts\wanderer field-objects 2025W18_2B --obsid 2025W18_2B_0266_1   # known asteroids in a field
.venv\Scripts\wanderer run 1633 --period 2025W18_2B            # sightings -> photometry -> spectrum -> figures
.venv\Scripts\wanderer cube-test 1633 --period 2025W18_2B      # static-sky subtraction diagnostic
```

Results land in `data/results/<target>/<period>/`: `sightings.csv`, `photometry.csv`, `reflectance.csv`, `colours.csv`, `summary.json`, and the figures `spectrum.png`, `sightings.png`, `moves.png`.

## Layout

| Module | Plan component | What it does |
| --- | --- | --- |
| `wanderer/s3.py` | C1 | Bucket listing, header range reads, remote FITS handles, retries |
| `wanderer/index.py` | C1 | Exposure index per planning period (time, detector, footprint) |
| `wanderer/db.py` | C1 | DuckDB store with sky-position lookup (ports to Postgres later) |
| `wanderer/catalog.py` | C2 | Per-exposure source catalog and Gaia astrometric check |
| `wanderer/gaia.py` | C2 | Gaia DR3 via TAP mirrors (GAVO, ESA, CDS), cached |
| `wanderer/cadence.py` | C1 | Field geometry, revisit cadence, detector pairing |
| `wanderer/flags.py` | C2 | Flag bits read from each file; defect-only mask |
| `wanderer/calib.py` | C2 | Science wavelength, bandwidth and pixel solid angle per pixel |
| `wanderer/epsf.py` | C2 | Effective PSF per detector zone |
| `wanderer/cube.py` | C3 | All-Sky Cube subtraction (diagnostic only) |
| `wanderer/ephemeris.py` | C4 | JPL Horizons tracks, Small-Body Identification, SBDB physical data |
| `wanderer/sightings.py` | C4 | Every exposure containing a known object, with detector position |
| `wanderer/photometry.py` | C3/C5 | Forced ePSF photometry at predicted positions |
| `wanderer/reflectance.py` | C5 | Reflectance spectrum, rotation-free colours, rotation scatter |
| `wanderer/plots.py` | C6 | Figures |
| `wanderer/cli.py` | | Command line |

`data/index/` and `data/reference/` are regenerated automatically and are git-ignored.

## Key findings so far

- Detectors 1/4 see the same sky at the same instant (likewise 2/5 and 3/6), so each pointing gives two wavelengths per source. Their ratio cancels asteroid rotation.
- Fields are about 3.5 deg across and are revisited about 2 minutes and about 3 hours apart, with a median of 8 looks per field per 3.5-day period.
- Subtracting the All-Sky Cube leaves a ring at every star, so unknown objects will be found with catalog-level change detection instead.
- 1633 Chimay: 65 sightings over 0.75-4.53 um, signal-to-noise 10-44. The spectrum is flat to 1.7 um, gently red to 3.3 um, with thermal emission beyond 3.7 um, consistent with its 0.08 albedo.

## Data credit

This work makes use of data products from the Spectro-Photometer for the History of the Universe, Epoch of Reionization and Ices Explorer (SPHEREx), a joint project of the Jet Propulsion Laboratory and the California Institute of Technology, funded by NASA. SPHEREx DR1/QR3 data from IRSA. Ephemerides from JPL Horizons and the JPL Small-Body Identification API.
