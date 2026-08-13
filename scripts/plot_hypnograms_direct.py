"""Hypnograms for all 3 theta conventions x 4 combos, with the theta threshold set on the
~nrem & ~mov pool (quiet wake + REM). Own-channel theta, EMG motion.

Uses classify(theta_conditioned=True) and asserts its threshold equals find_thresh on the direct
pool. Since conditioned_theta_thresh now always takes the ~nrem & ~mov trough rather than falling
back to it, the two agree by construction and the assert guards that invariant rather than an
empirical coincidence.

Usage: python plot_hypnograms_direct.py --seg seg5-148
"""

import warnings
import argparse
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sleep_sandbox.analysis import smooth_norm, find_thresh, state_intervals, bout_durations
from sleep_sandbox.scoring import classify

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
dur = cfg["duration_criteria"]
CONVS = list(cfg["theta"]["conventions"])
COMBOS = [(p, v) for p in ["ProbeA", "ProbeB"] for v in ["lfp_cmr", "lfp_nocmr"]]
STATES = ["nrem", "rem", "wake"]
COLOR = {"nrem": "tab:blue", "rem": "tab:red", "wake": "tab:green"}

rows = [(p, v, c) for (p, v) in COMBOS for c in CONVS]
fig, axes = plt.subplots(len(rows), 1, figsize=(16, 1.55 * len(rows)), sharex=True)

for ax, (probe, variant, conv) in zip(axes, rows):
    d = deriv_base / probe / variant
    result = dict(np.load(d / "result.npz"))
    extras = dict(np.load(d / "result_extras.npz"))
    sw_thresh = yaml.safe_load(open(d / "result.yml"))["result_scalars"]["sw_thresh"]
    times, sw, mo = result["times"], result["sw_metric"], result["motion_metric"]
    dt = float(np.median(np.diff(times)))

    th = smooth_norm(extras[f"theta_own_ratio_{conv}"], step_s=step_s, win_s=win_s)
    ch = int(extras[f"theta_own_channel_{conv}"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        motion_thresh = find_thresh(mo, method, bs, bm, grid_n)
        pool = ~(sw > sw_thresh) & ~((sw < sw_thresh) & (mo > motion_thresh))
        t_direct = find_thresh(th[pool], method, bs, bm, grid_n)
        s = classify(sw, th, mo, sw_thresh, bs, bm, method, grid_n, dt=dt,
                     merge_shorter_than_s=dur["merge_shorter_than_s"],
                     min_state_s=dur["min_state_s"],
                     microarousal_max_s=dur["microarousal_max_s"], theta_conditioned=True)

    got = s["th_thresh"]
    assert (np.isnan(got) and np.isnan(t_direct)) or np.isclose(got, t_direct), \
        f"{probe}/{variant}/{conv}: classify gave {got}, direct pool gives {t_direct}"

    for i, st in enumerate(STATES):
        ax.broken_barh(state_intervals(s[st], times, dt), (i - 0.4, 0.8), facecolors=COLOR[st])
    rb = bout_durations(s["rem"], dt)
    med = f"{np.median(rb):.0f}s" if len(rb) else "--"
    ax.set_yticks(range(len(STATES)), STATES, fontsize=7)
    ax.set_ylim(-0.5, len(STATES) - 0.5)
    ax.set_ylabel(f"{probe[-1]}/{variant.replace('lfp_', '')}\n{conv} ch{ch}", fontsize=7.5)
    ax.set_title(f"th_thresh={'NaN' if np.isnan(got) else f'{got:.3f}'}  "
                 f"REM={s['rem'].mean():.4f} ({len(rb)} bouts, median {med})  "
                 f"pool n={pool.sum()}", fontsize=7.5, loc="left", pad=2)

axes[-1].set_xlabel("time (s)")
fig.suptitle("Hypnograms by theta convention — theta threshold set on ~nrem & ~mov "
             "(quiet wake + REM), EMG motion, own-channel", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out = deriv_base / "hypnograms_theta_direct_pool.png"
fig.savefig(out, dpi=200)
print("wrote", out)
