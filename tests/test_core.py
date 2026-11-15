"""Offline tests: no network. Run with `pytest`; network tests are marked and skipped by default."""
import numpy as np
import pytest
from astropy.io import fits

from wanderer import ephemeris, flags, photometry, reflectance, s3
from wanderer.epsf import EPSFSet, gaussian_epsf


def _epsf_table(psf):
    return {"XCENTER": [1019.5], "YCENTER": [1019.5], "CWAVE": [1.0], "EPSF": [psf.ravel()]}


# --- ePSF -------------------------------------------------------------------------------------
@pytest.mark.parametrize("dx,dy", [(0, 0), (0.3, -0.2), (0.5, 0.5)])
def test_epsf_native_model_conserves_flux(dx, dy):
    e = EPSFSet(_epsf_table(gaussian_epsf(0.6)))
    yy, xx = np.mgrid[-6:7, -6:7]
    assert e.model(0, xx - dx, yy - dy).sum() == pytest.approx(1.0, abs=0.02)


def test_epsf_zone_is_nearest():
    t = {"XCENTER": [100, 1000], "YCENTER": [100, 1000], "CWAVE": [1, 2], "EPSF": [np.zeros(1089)] * 2}
    assert EPSFSet(t).zone(900, 950) == 1


# --- photometry -------------------------------------------------------------------------------
def test_forced_fit_recovers_flux_offset_and_background():
    rng = np.random.default_rng(1)
    e = EPSFSet(_epsf_table(gaussian_epsf(0.6)))
    n, c = 15, 7
    yy, xx = np.mgrid[0:n, 0:n]
    true = dict(flux=500.0, bkg=3.0, dx=0.4, dy=-0.3)
    img = true["flux"] * e.model(0, xx - (c + true["dx"]), yy - (c + true["dy"])) + true["bkg"]
    var = np.full_like(img, 0.25)
    img = img + rng.normal(0, 0.5, img.shape)
    fit = photometry.fit_point_source(img, var, np.ones_like(img, bool), e, 0, c, c)
    assert fit.flux == pytest.approx(true["flux"], rel=0.02)
    assert fit.background == pytest.approx(true["bkg"], abs=0.2)
    assert (fit.dx, fit.dy) == pytest.approx((true["dx"], true["dy"]), abs=0.06)
    assert 0.5 < fit.chi2_dof < 2.0


def test_star_modelling_removes_blend_bias():
    """A star 2.5 px away biases a single-source fit; modelling it recovers the target flux."""
    rng = np.random.default_rng(4)
    e = EPSFSet(_epsf_table(gaussian_epsf(0.9)))
    n, c = 15, 7
    yy, xx = np.mgrid[0:n, 0:n]
    star = (c + 2.5, c + 0.5)
    img = 400 * e.model(0, xx - c, yy - c) + 900 * e.model(0, xx - star[0], yy - star[1]) + 2.0
    img = img + rng.normal(0, 0.3, img.shape)
    var, good = np.full_like(img, 0.09), np.ones_like(img, bool)
    naive = photometry.fit_point_source(img, var, good, e, 0, c, c)
    modelled = photometry.fit_point_source(img, var, good, e, 0, c, c, stars_xy=[star])
    assert abs(naive.flux - 400) > 20 or naive.chi2_dof > 5  # biased or a bad fit
    assert modelled.flux == pytest.approx(400, rel=0.03)
    assert modelled.chi2_dof < 2.0


def test_solve_with_stars_reduces_to_solve_without_stars_columns():
    rng = np.random.default_rng(5)
    M = rng.random((4, 60)); E = rng.random((1, 60)); W = np.full(60, 2.0)
    D = 3 * M[2] + 0.0 * E[0] + 1.0
    a, b, _, chi2 = photometry._solve_with_stars(D, W, M, E)
    assert a[2] == pytest.approx(3, rel=1e-6) and b[2] == pytest.approx(1, rel=1e-6) and chi2[2] < 1e-10


def test_solve_matches_lstsq():
    rng = np.random.default_rng(2)
    M = rng.random((3, 50)); D = 2 * M[1] + 0.5 + rng.normal(0, 0.01, 50); W = np.ones(50)
    a, b, _, _ = photometry._solve(D, W, M)
    ref = np.linalg.lstsq(np.stack([M[1], np.ones(50)], 1), D, rcond=None)[0]
    assert (a[1], b[1]) == pytest.approx(tuple(ref), rel=1e-6)


# --- flags ------------------------------------------------------------------------------------
def test_flag_bits_from_header_and_mask():
    h = fits.Header()
    h["MSKN0001"], h["MSKM0001"] = "OVERFLOW", 2
    h["MSKN0021"], h["MSKM0021"] = "SOURCE", 2097152
    bits = flags.flag_bits(h)
    assert bits == {"OVERFLOW": 2, "SOURCE": 2097152}
    assert flags.mask_value(bits) == 2  # SOURCE is informational, never masked


def test_flag_fallback_table_masks_no_informational_bits():
    m = flags.mask_value(flags.flag_bits(fits.Header()))
    for name in flags.INFORMATIONAL:
        assert m & flags.DR1_BITS[name] == 0


# --- S3 listing and header parsing ------------------------------------------------------------
def test_parse_listing_with_pagination_token():
    xml = ("<ListBucketResult><Prefix>dr1/</Prefix><Contents><Key>dr1/a.fits</Key><Size>10</Size></Contents>"
           "<CommonPrefixes><Prefix>dr1/level2/</Prefix></CommonPrefixes>"
           "<NextContinuationToken>tok</NextContinuationToken></ListBucketResult>")
    assert s3.parse_listing(xml) == ([("dr1/a.fits", 10)], ["dr1/level2/"], "tok")


def test_header_end_finds_block_boundary():
    h = fits.Header(); h["OBSID"] = "x"
    raw = fits.PrimaryHDU(header=h).header.tostring().encode()
    assert s3._header_end(raw, 0) == 2880


# --- ephemeris formatting ---------------------------------------------------------------------
def test_sexagesimal_formats_for_sb_ident():
    assert ephemeris._ra_sexagesimal(162.211) == "10-48-50.64"
    assert ephemeris._dec_sexagesimal(-12.5) == "M12-30-00.0"
    assert ephemeris._dec_sexagesimal(10.773) == "10-46-22.8"
    assert ephemeris._ra_sexagesimal(359.99999) == "00-00-00.00"  # rounds up and wraps, never 24h
    assert ephemeris._ra_sexagesimal(-15.0) == "23-00-00.00"


def test_horizons_queries_are_small_body(monkeypatch):
    """Regression: a bare '6' made Horizons return Saturn instead of asteroid (6) Hebe."""
    seen = {}

    class Fake:
        def __init__(self, **kw):
            seen.update(kw)

        def ephemerides(self):
            return "eph"

    monkeypatch.setattr(ephemeris, "Horizons", Fake)
    assert ephemeris._horizons("6", "500", 2460798.5) == "eph"
    assert seen["id_type"] == "smallbody" and seen["id"] == "6"


# --- reflectance ------------------------------------------------------------------------------
def test_sun_divided_by_sun_is_flat():
    grid = np.linspace(0.6, 5.2, 4600)
    sun = (grid, grid**-1.5)
    lam, bw = np.array([0.8, 1.6, 3.0]), np.array([0.02, 0.04, 0.08])
    s = reflectance.solar_in_band(lam, bw, sun)
    assert np.allclose(s / s, 1.0)
    assert s[0] > s[1] > s[2]
