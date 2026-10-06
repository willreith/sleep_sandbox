"""Stitch the per-chunk scoring results into one multi-day hypnogram on an absolute UTC axis.

Usage: python stitch_chunks.py --scoring DIR [--out PNG] [--cols 7200]
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

parser = argparse.ArgumentParser()
parser.add_argument("--scoring", type=Path, required=True, help="dir holding the chunk* folders")
parser.add_argument("--out", type=Path, default=None)
parser.add_argument("--cols", type=int, default=7200, help="columns per 24 h row")
args = parser.parse_args()

COL = {"REM": "red", "NREM": "#0022ee", "wake": "#83008f", "nodata": "0.65"}
LANES = [("wake", "wake"), ("REM", "rem"), ("NREM", "nrem")]   # top to bottom
LANE_H, LANE_GAP = 0.22, 0.10
TOP = (1 - (3 * LANE_H + 2 * LANE_GAP)) / 2
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans"],
                     "font.size": 30, "axes.titlesize": 30, "axes.labelsize": 34,
                     "xtick.labelsize": 28, "ytick.labelsize": 28,
                     "xtick.color": "black", "xtick.labelcolor": "black",
                     "figure.titlesize": 40, "axes.linewidth": 1.6})

chunks = sorted(args.scoring.glob("chunk*"))
res = [dict(np.load(d / "result.npz")) for d in chunks]
summ = [json.loads((d / "summary.json").read_text()) for d in chunks]
prov = [yaml.safe_load((d / "result.yml").read_text())["provenance"] for d in chunks]

rec_start = datetime.fromisoformat(prov[0]["rec_start"])
ta = np.concatenate([r["times_abs"] for r in res])
step = float(np.median(np.diff(ta)))
assert np.allclose(np.diff(ta), step), "times_abs is not contiguous across chunks"

dt_col = 24 * 3600 / args.cols                      # 12 s at the default 7200
XMIN = -2.0                                         # chunk00 starts before its row's midday

fig, all_ax = plt.subplots(len(chunks) + 1, 1, figsize=(28, 2.05 * len(chunks) + 2.2), dpi=320,
                           sharex=True, gridspec_kw={"height_ratios": [0.36] + [1] * len(chunks)})
cyc, axes = all_ax[0], all_ax[1:]

hh = np.linspace(XMIN, 24, 4000)                     # 19-06 light, 07-18 dark, 1 h twilight ramps
hr = (hh + 12) % 24
lum = np.select([(hr >= 19) | (hr < 6), (hr >= 7) & (hr < 18)], [2, 0], default=1)
cyc.imshow(lum[None], extent=(XMIN, 24, 0, 1), aspect="auto", vmin=-0.5, vmax=2.5,
           interpolation="nearest",
           cmap=matplotlib.colors.ListedColormap(["#151515", "#8a8574", "#fff4cf"]))
for xt, lb, c in ((3.0, "dark", "0.85"), (12.5, "light", "0.15"), (21.5, "dark", "0.85")):
    cyc.text(xt, 0.5, lb, ha="center", va="center", color=c, fontsize=17, fontweight="bold")
cyc.text(-0.055, 0.5, "light cycle", transform=cyc.transAxes, ha="center", va="center",
         color="black", fontsize=24)
cyc.set(xlim=(XMIN, 24), ylim=(0, 1), yticks=[])
cyc.tick_params(bottom=False)
for sp in cyc.spines.values():
    sp.set(color="0.75", linewidth=1.0)
for i, (ax, d, r, s_, p) in enumerate(zip(np.atleast_1d(axes), chunks, res, summ, prov)):
    t0 = datetime.fromisoformat(p["chunk"]["t_start"])
    origin = t0.replace(hour=12, minute=0, second=0, microsecond=0)
    x = (r["times_abs"] - (origin - rec_start).total_seconds()) / 3600

    col = ((x - x[0]) * 3600 / dt_col).astype(int)   # own extent, so no blank pad in short rows
    ncol = col[-1] + 1
    x1 = x[0] + dt_col / 3600 * ncol

    nod = np.zeros(ncol, bool)
    np.logical_or.at(nod, col, r["nodata"])
    ax.imshow(np.where(nod, 1.0, np.nan)[None], extent=(x[0], x1, TOP, 1 - TOP), aspect="auto",
              interpolation="nearest", cmap=matplotlib.colors.ListedColormap([COL["nodata"]]),
              vmin=0, vmax=1, zorder=0)

    f = s_["state_fractions_of_valid"]
    for k, (st, key) in enumerate(LANES):
        y0 = 1 - TOP - LANE_H - k * (LANE_H + LANE_GAP)
        on = np.zeros(ncol, bool)
        np.logical_or.at(on, col, r[key])
        if st == "REM":                              # sparse: lines, never resampled away
            ax.vlines(x[0] + dt_col / 3600 * (np.flatnonzero(on) + 0.5), y0, y0 + LANE_H,
                      color=COL[st], lw=1.2, zorder=2)
        else:
            rgba = np.zeros((1, ncol, 4))
            rgba[0, on] = to_rgba(COL[st])
            ax.imshow(rgba, extent=(x[0], x1, y0, y0 + LANE_H), aspect="auto",
                      interpolation="nearest", zorder=2)
        ax.text(1.022, y0 + LANE_H / 2, f"{st} {100 * f[key]:.1f}%", transform=ax.transAxes,
                ha="left", va="center", color=COL[st], fontsize=23, fontweight="bold")

    ax.text(-0.052, 0.68, f"Day {i}", transform=ax.transAxes, ha="center", va="center",
            fontsize=32)
    ax.text(-0.052, 0.26, f"no data {100 * r['nodata'].mean():.0f}%", transform=ax.transAxes,
            ha="center", va="center", color="black", fontsize=22)
    ax.set(xlim=(XMIN, 24), ylim=(0, 1), yticks=[])
    ax.tick_params(length=6, width=1.4, bottom=ax is axes[-1])
    for sp in ax.spines.values():
        sp.set_visible(False)

ticks = np.arange(0, 25, 3)
axes[-1].set(xticks=ticks, xticklabels=[f"{(12 + t) % 24:02d}:00" for t in ticks],
             xlabel="UTC time of day")
fig.suptitle("Pilot04")
fig.tight_layout(rect=(0.075, 0, 0.88, 0.95), h_pad=1.6)
out = args.out or args.scoring.parent / "hypnogram_full.png"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}  ({args.cols} columns per 24 h, {dt_col:.1f} s each)")
