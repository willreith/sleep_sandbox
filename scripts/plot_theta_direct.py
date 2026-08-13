"""Theta thresholding on ~nrem & ~mov directly (quiet wake + REM), vs the unconditioned
distribution, for all 2 probes x 2 variants x 3 conventions. Own-channel theta, EMG motion.

The ~mov pool is shown faintly for context only: it is 85-92% NREM, so its trough marks the
NREM/non-NREM boundary rather than REM vs quiet wake. conditioned_theta_thresh no longer consults
it, so the direct threshold below is the production one by construction.

Usage: python plot_theta_direct.py --seg seg5-148
"""

import warnings
import argparse
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from sleep_sandbox.analysis import smooth_norm, find_thresh

repo_root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument("--seg", required=True,
                     help="segment range, e.g. 'seg5-148'; reads/writes under data/derivatives/{seg}/")
args = parser.parse_args()
deriv_base = repo_root / "data/derivatives" / args.seg

with open(repo_root / "config/sleep_scoring.yml") as f:
    cfg = yaml.safe_load(f)

step_s, win_s = cfg["spectrogram"]["step_s"], cfg["smoothing"]["window_s"]
bs, bm = cfg["bimodal_threshold"]["startbins"], cfg["bimodal_threshold"]["maxbins"]
method, grid_n = cfg["threshold"]["method"], cfg["threshold"]["kde_grid_n"]
CONVS = list(cfg["theta"]["conventions"])
COMBOS = [(p, v) for p in ["ProbeA", "ProbeB"] for v in ["lfp_cmr", "lfp_nocmr"]]

fig, axes = plt.subplots(len(COMBOS), len(CONVS), figsize=(16, 15))

for row, (probe, variant) in enumerate(COMBOS):
    d = deriv_base / probe / variant
    result = dict(np.load(d / "result.npz"))
    extras = dict(np.load(d / "result_extras.npz"))
    sw_thresh = yaml.safe_load(open(d / "result.yml"))["result_scalars"]["sw_thresh"]
    sw, mo = result["sw_metric"], result["motion_metric"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        motion_thresh = find_thresh(mo, method, bs, bm, grid_n)
    nrem = sw > sw_thresh
    mov = (sw < sw_thresh) & (mo > motion_thresh)
    pool = ~nrem & ~mov                      # quiet wake + REM -- what REM must be separated within

    for col, conv in enumerate(CONVS):
        ax = axes[row][col]
        th = smooth_norm(extras[f"theta_own_ratio_{conv}"], step_s=step_s, win_s=win_s)
        ch = int(extras[f"theta_own_channel_{conv}"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t_uncond = find_thresh(th, method, bs, bm, grid_n)
            t_direct = find_thresh(th[pool], method, bs, bm, grid_n)

        ax.hist(th, bins=np.linspace(0, 1, 80), density=True, alpha=0.25, color="tab:blue",
                label=f"all epochs (n={th.size})")
        ax.hist(th[~mov], bins=np.linspace(0, 1, 80), density=True, alpha=0.18, color="tab:orange",
                label=f"~mov (n={(~mov).sum()}, {100*nrem[~mov].mean():.0f}% NREM)")
        ax.hist(th[pool], bins=np.linspace(0, 1, 40), density=True, alpha=0.5, color="tab:red",
                label=f"~nrem&~mov (n={pool.sum()})")

        # The KDE find_thresh actually sees on the direct pool -- shows why it splits or doesn't.
        grid = np.linspace(th[pool].min(), th[pool].max(), grid_n)
        ax.plot(grid, gaussian_kde(th[pool])(grid), color="darkred", lw=1.4, label="KDE(~nrem&~mov)")

        for t, c, lbl in ((t_uncond, "tab:blue", "uncond"), (t_direct, "darkred", "direct")):
            if np.isfinite(t):
                ax.axvline(t, color=c, ls="--", lw=1.8, label=f"{lbl} thr = {t:.3f}")
            else:
                ax.plot([], [], ls="--", color=c, label=f"{lbl} thr = NaN")

        ax.set_title(f"{probe}/{variant.replace('lfp_', '')} — {conv} (ch{ch})", fontsize=9)
        ax.legend(fontsize=6.5)
        ax.set_xlim(0, 1)
        if col == 0:
            ax.set_ylabel("density")
        if row == len(COMBOS) - 1:
            ax.set_xlabel("theta metric [0,1]")

fig.suptitle("Theta threshold on ~nrem & ~mov (quiet wake + REM) directly, vs unconditioned — "
             "own-channel, EMG motion", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.98])
out = deriv_base / "theta_direct_pool_comparison.png"
fig.savefig(out, dpi=200)
print("wrote", out)
