"""Figures for the plan doc, the pitch and the web app."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

from . import s3  # noqa: E402
from .reflectance import THERMAL_FROM_UM  # noqa: E402

DET_COLOURS = {1: "#1f77b4", 2: "#2ca02c", 3: "#9467bd", 4: "#d62728", 5: "#ff7f0e", 6: "#8c564b"}


def _cutout_at_sky(key: str, ra: float, dec: float, r: int = 15) -> np.ndarray:
    with s3.open_fits(key) as h:
        x, y = WCS(h["IMAGE"].header, naxis=2).all_world2pix([[ra, dec]], 0)[0]
        xi, yi = int(round(x)), int(round(y))
        if not (r <= xi < 2040 - r and r <= yi < 2040 - r):
            return np.full((2 * r + 1, 2 * r + 1), np.nan)
        sl = (slice(yi - r, yi + r + 1), slice(xi - r, xi + r + 1))
        return h["IMAGE"].section[sl] - h["ZODI"].section[sl]


def _show(ax, v, title, r=15):
    lo, hi = np.nanpercentile(v, [5, 99.5]) if np.isfinite(v).any() else (0, 1)
    ax.imshow(v, origin="lower", cmap="gray", vmin=lo, vmax=hi)
    ax.add_patch(plt.Circle((r, r), 3.5, fill=False, color="r", lw=1))
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def sightings_grid(sightings: pd.DataFrame, target: str, out: Path, ncols: int = 8, limit: int = 32) -> None:
    s = sightings.head(limit)
    rows = int(np.ceil(len(s) / ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=(1.5 * ncols, 1.7 * rows))
    t0 = s.mjd.min()
    for ax, r in zip(np.ravel(axes), s.itertuples()):
        _show(ax, _cutout_at_sky(r.key, r.ra_pred, r.dec_pred), f"D{r.det} +{(r.mjd - t0) * 24:.1f} h")
    for ax in np.ravel(axes)[len(s):]:
        ax.axis("off")
    fig.suptitle(f"{target}: predicted position (red) in SPHEREx DR1 exposures", fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close(fig)


def moves_proof(sightings: pd.DataFrame, target: str, out: Path, pairs: int = 3) -> None:
    """Left: target at time A. Right: same sky in a later exposure on the same detector."""
    d = sightings[sightings.det == sightings.det.mode()[0]].sort_values("mjd").reset_index(drop=True)
    blocks = d.block.unique()
    chosen = [(d[d.block == blocks[i]].iloc[0], d[d.block == blocks[(i + len(blocks) // 2) % len(blocks)]].iloc[0])
              for i in range(min(pairs, len(blocks)))]
    fig, axes = plt.subplots(len(chosen), 2, figsize=(4.2, 2.1 * len(chosen)), squeeze=False)
    for k, (a, b) in enumerate(chosen):
        _show(axes[k, 0], _cutout_at_sky(a.key, a.ra_pred, a.dec_pred), f"MJD {a.mjd:.3f}: {target} here")
        _show(axes[k, 1], _cutout_at_sky(b.key, a.ra_pred, a.dec_pred), f"MJD {b.mjd:.3f}: same sky")
    fig.suptitle(f"{target} in SPHEREx DR1: the source leaves, stars stay", fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close(fig)


def spectrum(refl: pd.DataFrame, target: str, period: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for d, g in refl.groupby("det"):
        ok, bad = g[g.reliable], g[~g.reliable]
        ax.errorbar(ok.lam_um, ok.refl, ok.refl_err, fmt="o", ms=4, color=DET_COLOURS[d], label=f"D{d}")
        ax.errorbar(bad.lam_um, bad.refl, bad.refl_err, fmt="x", ms=4, color=DET_COLOURS[d], alpha=0.4)
    ax.axhline(1, color="gray", lw=0.8, ls="--")
    top = max(2.0, float(np.nanmax(refl.refl + refl.refl_err)) * 1.08)
    ax.set_ylim(0, top)
    ax.axvspan(2.7, 3.1, color="steelblue", alpha=0.08)
    ax.text(2.72, top * 0.95, "3 µm\nhydration", fontsize=7, va="top")
    ax.axvspan(THERMAL_FROM_UM, 5.1, color="orange", alpha=0.07)
    ax.text(THERMAL_FROM_UM + 0.05, top * 0.04, "thermal emission\nadds flux here", fontsize=7)
    ax.set_xlabel("wavelength (µm)")
    ax.set_ylabel("reflectance (normalised at 0.95–1.15 µm)")
    ax.set_title(f"{target}: SPHEREx DR1 reflectance spectrum, planning period {period} (x = flagged)", fontsize=9)
    ax.legend(fontsize=7, ncol=6)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close(fig)


def cube_residuals(res: dict, out: Path) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.3))
    for a, key, title in zip(ax, ("image", "reference", "significance"),
                             ("SPHEREx exposure (zodi removed)", "All-Sky Cube, same channel", "difference / noise (sigma)")):
        v = res[key]
        lo, hi = (-5, 5) if key == "significance" else np.nanpercentile(v, [5, 99.5])
        a.imshow(v, origin="lower", cmap="gray", vmin=lo, vmax=hi)
        a.set_title(title, fontsize=8)
        a.axis("off")
    ax[2].add_patch(plt.Circle(res["target_xy"], 5, fill=False, color="r"))
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close(fig)
