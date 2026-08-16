"""Condensed hypnograms for the go-forward ProbeB set: one colour-coded bar per configuration,
stacked so states line up vertically for direct comparison.

Configurations (docs/sleep_classification_algorithm.md, "Note on referencing"):
  lfp_cmr   + watson
  lfp_cmr   + shin
  lfp_nocmr + shin     CMR vs noCMR control

Unlike plot_hypnograms_direct.py (one lane per state), each row here is a single lane in which
colour encodes the state -- nrem/rem/wake partition every epoch, so the lane is fully covered and
the three rows can be read against each other at a glance.

Each configuration is drawn twice: the top block without Watson's 20 s packet minimum and the
bottom block with it (duration_criteria.min_state_s), so the effect of the criterion can be read
off vertically. Bout counts for both are printed to stdout.

Writes data/derivatives/hypnograms_probeb_condensed.png.
Usage: python plot_hypnograms_probeb.py
"""

import warnings
import argparse
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sleep_sandbox.analysis import smooth_norm, state_intervals, bout_durations
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
theta_cond = cfg["theta"]["movement_conditioned"]

CONFIGS = [("lfp_cmr", "watson"), ("lfp_cmr", "shin"), ("lfp_nocmr", "shin")]
STATES = ["nrem", "rem", "wake"]
COLOR = {"nrem": "tab:blue", "rem": "tab:red", "wake": "tab:green"}

min_state_s = dur["min_state_s"]
BLOCKS = [("no 20 s min", None), (f"{min_state_s:g} s min", min_state_s)]
n_lanes = len(BLOCKS) * len(CONFIGS)

fig, ax = plt.subplots(figsize=(16, 6.4))
lane_labels, rows = [], []

for i, (variant, conv) in enumerate(CONFIGS):
    d = deriv_base / "ProbeB" / variant
    result = dict(np.load(d / "result.npz"))
    extras = dict(np.load(d / "result_extras.npz"))
    sw_thresh = yaml.safe_load(open(d / "result.yml"))["result_scalars"]["sw_thresh"]
    times, sw, mo = result["times"], result["sw_metric"], result["motion_metric"]
    dt = float(np.median(np.diff(times)))

    th = smooth_norm(extras[f"theta_own_ratio_{conv}"], step_s=step_s, win_s=win_s)
    ch = int(extras[f"theta_own_channel_{conv}"])

    for b, (block, min_s) in enumerate(BLOCKS):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            s = classify(sw, th, mo, sw_thresh, bs, bm, method, grid_n, dt=dt,
                         merge_shorter_than_s=dur["merge_shorter_than_s"], min_state_s=min_s,
                         microarousal_max_s=dur["microarousal_max_s"], theta_conditioned=theta_cond)

        row = b * len(CONFIGS) + i
        y = n_lanes - 1 - row                     # first config of the first block on top
        for st in STATES:
            ax.broken_barh(state_intervals(s[st], times, dt), (y - 0.38, 0.76), facecolors=COLOR[st])

        rb, nb = bout_durations(s["rem"], dt), bout_durations(s["nrem"], dt)
        ax.text(times[-1] + 250, y, f"REM {s['rem'].mean()*100:.2f}%  ({len(rb)} bouts, "
                f"median {np.median(rb):.0f}s)" if len(rb) else "REM none", va="center", fontsize=7.5)
        lane_labels.append((row, f"{variant.replace('lfp_', '')} / {conv}\n{block}"))
        rows.append((variant, conv, block, s, rb, nb))

ax.axhline(len(CONFIGS) - 0.5, color="0.3", lw=1.0, ls="--")
ax.set_yticks([n_lanes - 1 - r for r, _ in lane_labels], [lab for _, lab in lane_labels], fontsize=8)
ax.set_ylim(-0.6, n_lanes - 0.4)
ax.set_xlabel("time (s)")
ax.set_xlim(times[0] - 200, times[-1] + 6000)
ax.legend(handles=[Patch(facecolor=COLOR[s], label=s) for s in STATES],
          loc="upper right", ncol=3, fontsize=8, framealpha=0.9)
ax.set_title(f"ProbeB hypnograms — go-forward set, without vs with Watson's {min_state_s:g} s "
             "packet minimum", fontsize=11)
fig.tight_layout()
out = deriv_base / "hypnograms_probeb_condensed.png"
fig.savefig(out, dpi=200)
print("wrote", out)

print(f"\n{'config':<18} {'block':<12} {'REM n':>6} {'REM %':>7} {'REM med':>8} "
      f"{'NREM n':>7} {'NREM %':>7} {'NREM med':>9} {'MA %':>6}")
for variant, conv, block, s, rb, nb in sorted(rows, key=lambda r: (r[0], r[1])):
    print(f"{variant.replace('lfp_', '') + '/' + conv:<18} {block:<12} {len(rb):>6} "
          f"{s['rem'].mean()*100:>6.2f}% {np.median(rb) if len(rb) else np.nan:>7.0f}s "
          f"{len(nb):>7} {s['nrem'].mean()*100:>6.2f}% {np.median(nb) if len(nb) else np.nan:>8.0f}s "
          f"{s['ma'].mean()*100:>5.2f}%")
