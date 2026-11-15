"""Command line: `wanderer <command>` (or `python -m wanderer <command>`).

    wanderer periods                       list DR1 planning periods
    wanderer index 2025W18_2B              index a planning period (all detectors)
    wanderer covering 162.2 10.8           exposures containing a sky position (needs an index)
    wanderer cadence 2025W18_2B            geometry and revisit cadence
    wanderer field-objects 2025W18_2B --obsid 2025W18_2B_0266_1
    wanderer run 1633 --period 2025W18_2B  sightings -> photometry -> spectrum -> figures
    wanderer cube-test 1633 --period 2025W18_2B
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from . import cadence, config, ephemeris, index, photometry, plots, reflectance, sightings


def _results_dir(target: str, period: str) -> Path:
    d = config.RESULTS_DIR / target.replace(" ", "_") / period
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float))


def cmd_periods(a):
    p = index.list_periods(a.release)
    print(f"{len(p)} planning periods: {p[0]} .. {p[-1]}")


def cmd_index(a):
    t = time.time()
    df = index.build_index(a.period, [int(d) for d in a.detectors.split(",")], a.release, a.workers)
    print(f"{len(df)} exposures indexed -> {index.index_path(a.period, a.release)} ({time.time() - t:.0f} s)")
    print(df.det.value_counts().sort_index().to_string())
    from . import db
    with db.connect() as con:
        n = db.load_period(con, df, a.period, a.release)
    print(f"loaded {n} exposures into {db.DB_PATH}")


def cmd_db_load(a):
    from . import db
    with db.connect() as con:
        n = db.load_period(con, index.load_index(a.period, a.release), a.period, a.release)
        print(f"loaded {n} exposures for {a.period}")
        print(db.periods(con).to_string(index=False))


def cmd_covering(a):
    from . import db
    dets = [int(d) for d in a.det.split(",")] if a.det else None
    with db.connect() as con:
        t = time.perf_counter()
        df = db.exposures_covering(con, a.ra, a.dec, a.mjd_min, a.mjd_max, dets, a.release)
        ms = (time.perf_counter() - t) * 1000
    print(f"{len(df)} exposures contain RA {a.ra}, Dec {a.dec} ({ms:.0f} ms)")
    if len(df):
        print(df[["obsid", "det", "mjd", "period"]].to_string(index=False))


def cmd_cadence(a):
    print(json.dumps(cadence.revisit_stats(index.load_index(a.period, a.release), a.det), indent=2))


def cmd_field_objects(a):
    idx = index.load_index(a.period, a.release)
    row = idx[(idx.obsid == a.obsid) & (idx.det == a.det)].iloc[0]
    df = ephemeris.sb_ident(row.ra, row.dec, row.mjd, vmag_lim=a.vmag)
    print(f"{a.obsid} D{a.det}: {len(df)} known objects brighter than V {a.vmag}")
    print(df.to_string(index=False))


def cmd_run(a):
    out = _results_dir(a.target, a.period)
    idx = index.load_index(a.period, a.release)
    t = time.time()
    s, track = sightings.find_sightings(a.target, idx, workers=a.workers)
    s.to_csv(out / "sightings.csv", index=False)
    track.to_csv(out / "track.csv", index=False)
    print(f"sightings: {len(s)} on detectors {sorted(s.det.unique())} ({time.time() - t:.0f} s)")
    t = time.time()
    phot = photometry.forced_photometry(s)
    phot.to_csv(out / "photometry.csv", index=False)
    print(f"photometry: {(phot.status == 'ok').sum()} measured, {(phot.status != 'ok').sum()} skipped ({time.time() - t:.0f} s)")
    _spectrum(a, out, phot, track)
    if not a.no_cutouts:
        plots.sightings_grid(s, a.target, out / "sightings.png")
        plots.moves_proof(s, a.target, out / "moves.png")
    print(f"results -> {out}")


def cmd_spectrum(a):
    """Rebuild spectrum, colours, taxonomy and summary from saved photometry (no downloads)."""
    out = _results_dir(a.target, a.period)
    _spectrum(a, out, pd.read_csv(out / "photometry.csv"), pd.read_csv(out / "track.csv"))


def _spectrum(a, out, phot, track):
    refl = reflectance.reflectance(phot, track)
    refl.to_csv(out / "reflectance.csv", index=False)
    col = reflectance.simultaneous_colours(refl)
    col.to_csv(out / "colours.csv", index=False)
    from . import taxonomy
    rel = refl[refl.reliable]
    tax = taxonomy.fit(rel.lam_um, rel.refl, rel.refl_err)
    tax.to_csv(out / "taxonomy.csv", index=False)
    multi = tax[~tax.single_object_class]
    summ = {"target": a.target, "period": a.period, "release": a.release,
            "physical_jpl": ephemeris.physical_parameters(a.target),
            "reference_ssodnet": ephemeris.ssodnet_reference(a.target),
            **reflectance.summary(refl, col),
            "taxonomy": {"best_class": multi["class"].iloc[0], "best_chi2_red": float(multi.chi2_red.iloc[0]),
                         "complex_ranking": tax.groupby("complex").chi2_red.min().sort_values().round(2).to_dict(),
                         "top3": multi.head(3)[["class", "chi2_red"]].round(2).to_dict("records"),
                         "points": tax.attrs["n_points"], "slope_0p85_1p7_per_um": taxonomy.slope_per_um(rel.lam_um, rel.refl)}}
    summ["photometry"] = {"chi2_dof_median": float(phot.chi2_dof.median()),
                          "blended": int(phot.get("blended", pd.Series(dtype=bool)).fillna(False).sum()),
                          "stars_modelled_total": int(phot.get("n_stars_modelled", pd.Series(dtype=float)).fillna(0).sum())}
    _dump(summ, out / "summary.json")
    plots.spectrum(refl, a.target, a.period, out / "spectrum.png")
    print(json.dumps(summ, indent=2, default=float))


def cmd_catalog(a):
    from . import catalog
    idx = index.load_index(a.period, a.release)
    row = idx[(idx.obsid == a.obsid) & (idx.det == a.det)].iloc[0]
    t = time.time()
    cat, wcs, mjd = catalog.source_catalog(row.key, a.det)
    out = config.RESULTS_DIR / "catalogs" / a.period
    out.mkdir(parents=True, exist_ok=True)
    cat.to_csv(out / f"{a.obsid}_D{a.det}.csv", index=False)
    print(f"{len(cat)} sources ({time.time() - t:.0f} s) -> {out / f'{a.obsid}_D{a.det}.csv'}")
    check = catalog.astrometry_check(cat, wcs, mjd)
    _dump(check, out / f"{a.obsid}_D{a.det}_astrometry.json")
    print(json.dumps(check, indent=2))


def cmd_targets(a):
    from . import validation
    idx = index.load_index(a.period, a.release)
    cands = validation.candidates(idx, a.fields, a.vmag)
    print(f"{len(cands)} numbered asteroids brighter than V {a.vmag} in {a.fields} ecliptic fields")
    ref = validation.with_reference(cands)
    out = config.RESULTS_DIR / "validation" / a.period
    out.mkdir(parents=True, exist_ok=True)
    ref.to_csv(out / "candidates.csv", index=False)
    sel = validation.select(ref, a.n)
    sel.to_csv(out / "targets.csv", index=False)
    print(f"{ref.taxonomy_class.notna().sum()} have a published class; selected {len(sel)}:")
    print(sel[["number", "name", "vmag", "taxonomy_class", "complex", "diameter_km", "albedo"]].to_string(index=False))


def cmd_cube_test(a):
    out = _results_dir(a.target, a.period)
    phot = pd.read_csv(out / "photometry.csv")
    r = phot[(phot.status == "ok") & (phot.det == a.det)].sort_values("snr", ascending=False).iloc[0]
    from . import cube
    res = cube.subtract_cube(r.key, r.x, r.y, r.lam_um)
    plots.cube_residuals(res, out / "cube_residuals.png")
    stats = {k: v for k, v in res.items() if k not in ("image", "reference", "significance", "target_xy")}
    _dump(stats, out / "cube_test.json")
    print(json.dumps(stats, indent=2))


def main(argv=None):
    def common(defaults: bool) -> argparse.ArgumentParser:
        # Global options work before or after the subcommand; SUPPRESS keeps a later copy from
        # overwriting a value given earlier on the line.
        c = argparse.ArgumentParser(add_help=False)
        d = (lambda v: v) if defaults else (lambda v: argparse.SUPPRESS)
        c.add_argument("-v", "--verbose", action="store_true", default=d(False))
        c.add_argument("--release", default=d(config.RELEASE))
        c.add_argument("--workers", type=int, default=d(config.WORKERS))
        return c

    p = argparse.ArgumentParser(prog="wanderer", description=__doc__, parents=[common(True)],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub_common = common(False)
    sub = p.add_subparsers(dest="cmd", required=True)
    _add = sub.add_parser
    sub.add_parser = lambda name, **kw: _add(name, parents=[sub_common], **kw)  # type: ignore[method-assign]
    sub.add_parser("periods").set_defaults(fn=cmd_periods)
    q = sub.add_parser("index"); q.add_argument("period"); q.add_argument("--detectors", default="1,2,3,4,5,6"); q.set_defaults(fn=cmd_index)
    q = sub.add_parser("db-load"); q.add_argument("period"); q.set_defaults(fn=cmd_db_load)
    q = sub.add_parser("covering"); q.add_argument("ra", type=float); q.add_argument("dec", type=float)
    q.add_argument("--mjd-min", type=float); q.add_argument("--mjd-max", type=float)
    q.add_argument("--det", help="comma-separated detectors"); q.set_defaults(fn=cmd_covering)
    q = sub.add_parser("cadence"); q.add_argument("period"); q.add_argument("--det", type=int, default=1); q.set_defaults(fn=cmd_cadence)
    q = sub.add_parser("field-objects"); q.add_argument("period"); q.add_argument("--obsid", required=True)
    q.add_argument("--det", type=int, default=1); q.add_argument("--vmag", type=float, default=17.0); q.set_defaults(fn=cmd_field_objects)
    q = sub.add_parser("run"); q.add_argument("target"); q.add_argument("--period", required=True)
    q.add_argument("--no-cutouts", action="store_true", help="skip the cutout figures (faster)"); q.set_defaults(fn=cmd_run)
    q = sub.add_parser("catalog"); q.add_argument("period"); q.add_argument("--obsid", required=True)
    q.add_argument("--det", type=int, default=1); q.set_defaults(fn=cmd_catalog)
    q = sub.add_parser("targets", help="validation targets with published taxonomy"); q.add_argument("period")
    q.add_argument("--fields", type=int, default=40); q.add_argument("--vmag", type=float, default=16.5)
    q.add_argument("-n", type=int, default=20); q.set_defaults(fn=cmd_targets)
    q = sub.add_parser("spectrum", help="rebuild spectrum/taxonomy from saved photometry")
    q.add_argument("target"); q.add_argument("--period", required=True); q.set_defaults(fn=cmd_spectrum)
    q = sub.add_parser("cube-test"); q.add_argument("target"); q.add_argument("--period", required=True)
    q.add_argument("--det", type=int, default=1); q.set_defaults(fn=cmd_cube_test)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("urllib3", "fsspec", "asyncio", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    a.fn(a)


if __name__ == "__main__":
    main()
