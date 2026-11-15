import numpy as np
import pytest

from wanderer import neatm


def test_flux_scales_with_diameter_squared():
    f1 = neatm.thermal_flux_mjy(4.4, 10.0, 0.1, 2.5, 1.8, 20.0)
    f2 = neatm.thermal_flux_mjy(4.4, 20.0, 0.1, 2.5, 1.8, 20.0)
    assert f2[0] / f1[0] == pytest.approx(4.0, rel=1e-6)


def test_closer_to_sun_is_hotter_and_brighter():
    far = neatm.thermal_flux_mjy(4.4, 10.0, 0.1, 3.0, 2.0, 15.0)[0]
    near = neatm.thermal_flux_mjy(4.4, 10.0, 0.1, 2.0, 2.0, 15.0)[0]
    assert near > 5 * far


def test_higher_phase_angle_shows_less_dayside():
    lo = neatm.thermal_flux_mjy(4.4, 10.0, 0.1, 2.5, 1.8, 5.0)[0]
    hi = neatm.thermal_flux_mjy(4.4, 10.0, 0.1, 2.5, 1.8, 40.0)[0]
    assert hi < lo


def test_diameter_from_h_and_albedo():
    # D = 1329/sqrt(pv) * 10^(-H/5): 1 km at pv=0.25 has H = 17.12
    assert neatm.diameter_km(17.12, 0.25) == pytest.approx(1.0, rel=0.01)


def test_fit_recovers_synthetic_albedo():
    import pandas as pd
    h, pv, geo = 11.0, 0.07, dict(r_au=2.8, delta_au=2.2, phase_deg=19.0)
    d = neatm.diameter_km(h, pv)
    lam = np.array([3.7, 3.8, 3.9, 4.4, 4.5])
    thermal = neatm.thermal_flux_mjy(lam, d, pv, geo["r_au"], geo["delta_au"], geo["phase_deg"])
    refl_flux = np.full_like(lam, 1.0)  # reflected part, mJy
    rows = [dict(lam_um=l, flux_mJy=refl_flux[i] + thermal[i], err_mJy=0.02, snr=50,
                 refl=1.2 * (refl_flux[i] + thermal[i]) / refl_flux[i], **geo) for i, l in enumerate(lam)]
    rows += [dict(lam_um=l, flux_mJy=1.0, err_mJy=0.02, snr=50, refl=1.2, **geo) for l in (2.5, 2.8, 3.1)]
    res = neatm.fit(pd.DataFrame(rows), h)
    assert res["albedo"] == pytest.approx(pv, rel=0.05)
