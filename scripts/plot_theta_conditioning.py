"""Conditioned vs unconditioned theta distributions + thresholds, all 2 probes x 2 variants x 3
conventions in one 4x3 figure. Own-channel theta, EMG motion.

This is the evidence behind config theta.movement_conditioned. Each panel overlays the full
distribution (blue), the ~mov subset (orange) and the ~nrem & ~mov pool (red outline), with all
three thresholds. The orange pool is shifted left of blue in all 12 panels: conditioning removes
locomotor theta, which is the high mode. Orange is labelled unused because
conditioned_theta_thresh now always takes the red pool -- ~mov is 85-92% NREM, so its trough marks
the NREM/non-NREM boundary rather than REM vs quiet wake. See docs/threshold_comparison.md §5.

Writes data/derivatives/{seg}/theta_conditioning_comparison.png.
Usage: python plot_theta_conditioning.py --seg seg5-148
"""

import warnings
import argparse
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sleep_sandbox.analysis import smooth_norm, find_thresh, conditioned_theta_thresh

repo_root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument("--seg", required=True,
                     help="segment range, e.g. 'seg5-148'; reads/writes under data/derivatives/{seg}/")
args = parser.parse_args()
deriv_base = repo_root / "data/derivatives" / args.seg

with open(repo_root / "config/sleep_scoring.yml") as f:
    cfg = yaml.safe_load(f)

step_s = cfg["spectrogram"]["step_s"]
win_s = cfg["smoothing"]["window_s"]
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

    for col, conv in enumerate(CONVS):
        ax = axes[row][col]
        th = smooth_norm(extras[f"theta_own_ratio_{conv}"], step_s=step_s, win_s=win_s)
        ch = int(extras[f"theta_own_channel_{conv}"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t_uncond, _ = conditioned_theta_thresh(th, sw, mo, sw_thresh, motion_thresh, bs, bm,
                                                   method, grid_n, conditioned=False)
            t_cond, _ = conditioned_theta_thresh(th, sw, mo, sw_thresh, motion_thresh, bs, bm,
                                                 method, grid_n, conditioned=True)
            t_mov = find_thresh(th[~mov], method, bs, bm, grid_n)   # buzcode's primary, now unused

        bins = np.linspace(0, 1, 80)
        ax.hist(th, bins=bins, density=True, alpha=0.45, color="tab:blue",
                label=f"unconditioned (n={th.size})")
        ax.hist(th[~mov], bins=bins, density=True, alpha=0.45, color="tab:orange",
                label=f"~mov, unused (n={(~mov).sum()}, {100*nrem[~mov].mean():.0f}% NREM)")
        ax.hist(th[~nrem & ~mov], bins=bins, density=True, histtype="step", lw=1.2,
                color="tab:red", label=f"~nrem&~mov (n={(~nrem & ~mov).sum()})")

        for t, c, lbl in ((t_uncond, "tab:blue", "uncond"), (t_mov, "tab:orange", "~mov"),
                          (t_cond, "tab:red", "~nrem&~mov")):
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

fig.suptitle("Theta metric: unconditioned (all epochs) vs movement-conditioned (~mov), "
             "with KDE thresholds — own-channel, EMG motion", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.98])
out = deriv_base / "theta_conditioning_comparison.png"
fig.savefig(out, dpi=200)
print("wrote", out)
