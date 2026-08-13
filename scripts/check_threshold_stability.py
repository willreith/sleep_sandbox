"""Gate for the 22 h scoring: do the KDE thresholds move when estimated on the whole recording
versus on only its first 7.47 h (the span the short segment covers)?

The metrics are held fixed at the long recording's throughout -- same dip-test channel, same PC1,
same global min-max -- and only the threshold estimation window varies. A threshold read off the
short run's own result.yml is NOT comparable to the long run's: both live on a [0,1] scale whose
anchors are that recording's own smoothed min/max, and PC1 is refit per recording, so the same
number denotes different raw values. Re-estimating on a prefix of the long run's metrics removes
that confound, leaving only the question asked.

Reports per combo: the two threshold triplets, Cohen's kappa and per-state Dice between the two
classifications of the same 22 h metrics, and how much of the long recording's normalised range
the prefix spans. Both thresholds sit on the long recording's scale by construction, so the span
is not a normalisation check -- it is why the KDE moves at all, the prefix being a truncated view
of the distribution the full-window estimate sees.

Usage: python check_threshold_stability.py [--long seg5-148] [--short seg5-52]
                                           [--probe ProbeA|ProbeB] [--variant lfp_cmr|lfp_nocmr]
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

from sleep_sandbox.analysis import (find_thresh, conditioned_theta_thresh,
                                    state_codes, cohens_kappa, state_intervals)
from sleep_sandbox.scoring import classify

repo_root = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--long", default="seg5-148", help="segment scored over the full recording")
parser.add_argument("--short", default="seg5-52", help="segment whose time span defines the prefix")
parser.add_argument("--probe", choices=["ProbeA", "ProbeB"])
parser.add_argument("--variant", choices=["lfp_cmr", "lfp_nocmr"])
args = parser.parse_args()

cfg = yaml.safe_load(open(repo_root / "config/sleep_scoring.yml"))
bs, bm = cfg["bimodal_threshold"]["startbins"], cfg["bimodal_threshold"]["maxbins"]
method, grid_n = cfg["threshold"]["method"], cfg["threshold"]["kde_grid_n"]
dur = cfg["duration_criteria"]
theta_cond = cfg["theta"]["movement_conditioned"]

probes = [args.probe] if args.probe else ["ProbeA", "ProbeB"]
variants = [args.variant] if args.variant else ["lfp_cmr", "lfp_nocmr"]
STATES = ["nrem", "rem", "wake"]
COLOR = {"nrem": "tab:blue", "rem": "tab:red", "wake": "tab:green"}
WINDOWS = ["full 22 h", "first 7.5 h"]

rows = []
for probe in probes:
    for variant in variants:
        dl = repo_root / "data/derivatives" / args.long / probe / variant
        ds = repo_root / "data/derivatives" / args.short / probe / variant
        r = dict(np.load(dl / "result.npz"))
        sw_full = yaml.safe_load(open(dl / "result.yml"))["result_scalars"]["sw_thresh"]
        t, sw, th, mo = r["times"], r["sw_metric"], r["theta_metric"], r["motion_metric"]
        dt = float(np.median(np.diff(t)))

        # Prefix = the long run's own epochs covering the short segment's span. The two runs are
        # built from the same blocks (5-52 is a strict prefix of 5-148), so this is the same data,
        # not an approximation.
        t_short = np.load(ds / "result.npz")["times"]
        n = int(np.searchsorted(t, t_short[-1], side="right"))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sw_pre = find_thresh(sw[:n], method, bs, bm, grid_n, label="slow_wave|prefix")
            mo_pre = find_thresh(mo[:n], method, bs, bm, grid_n, label="motion|prefix")
            th_pre, _ = conditioned_theta_thresh(th[:n], sw[:n], mo[:n], sw_pre, mo_pre,
                                                 bs, bm, method, grid_n, theta_cond)

            common = dict(dt=dt, merge_shorter_than_s=dur["merge_shorter_than_s"],
                          min_state_s=dur["min_state_s"],
                          microarousal_max_s=dur["microarousal_max_s"], theta_conditioned=theta_cond)
            s_full = classify(sw, th, mo, sw_full, bs, bm, method, grid_n, **common)
            s_pre = classify(sw, th, mo, sw_pre, bs, bm, method, grid_n,
                             motion_thresh=mo_pre, th_thresh=th_pre, **common)

        # Both thresholds are already on the long recording's normalised scale, so there is no
        # renormalisation to measure -- what moves the KDE is that the prefix distribution it sees
        # is truncated relative to the full one. Quantify that as the share of the recording lying
        # outside anything the prefix contained.
        outside = float(((sw < sw[:n].min()) | (sw > sw[:n].max())).mean())

        codes = [state_codes({k: s[k] for k in STATES}, STATES) for s in (s_full, s_pre)]
        kappa = cohens_kappa(codes[0], codes[1], len(STATES))
        dice = {k: (2 * (s_full[k] & s_pre[k]).sum() / (s_full[k].sum() + s_pre[k].sum())
                    if s_full[k].sum() + s_pre[k].sum() else np.nan) for k in STATES}

        rows.append(dict(probe=probe, variant=variant, n=n, kappa=kappa, dice=dice, outside=outside,
                         span=(float(sw[:n].min()), float(sw[:n].max())),
                         thr=[(sw_full, s_full["motion_thresh"], s_full["th_thresh"]),
                              (sw_pre, mo_pre, th_pre)],
                         states=[s_full, s_pre], times=t))

n_lanes = 2 * len(rows)
fig, ax = plt.subplots(figsize=(16, 1.05 * n_lanes + 1.6))
labels = []
for i, row in enumerate(rows):
    for w, s in enumerate(row["states"]):
        lane = 2 * i + w
        y = n_lanes - 1 - lane
        for st in STATES:
            ax.broken_barh(state_intervals(s[st], row["times"], 1.0), (y - 0.38, 0.76),
                           facecolors=COLOR[st])
        ax.text(row["times"][-1] + 600, y, f"NREM {s['nrem'].mean()*100:.1f}%  "
                f"REM {s['rem'].mean()*100:.2f}%", va="center", fontsize=7.5)
        labels.append((lane, f"{row['probe'][-1]}/{row['variant'].replace('lfp_', '')}\n"
                             f"{WINDOWS[w]}"))
    if i:
        ax.axhline(n_lanes - 2 * i + 0.5, color="0.3", lw=1.0, ls="--")

ax.axvline(rows[0]["times"][rows[0]["n"] - 1], color="k", lw=1.0, ls=":")
ax.set_yticks([n_lanes - 1 - l for l, _ in labels], [lab for _, lab in labels], fontsize=8)
ax.set_ylim(-0.6, n_lanes - 0.4)
ax.set_xlim(rows[0]["times"][0] - 400, rows[0]["times"][-1] + 14000)
ax.set_xlabel("time (s)")
ax.legend(handles=[Patch(facecolor=COLOR[s], label=s) for s in STATES],
          loc="upper right", ncol=3, fontsize=8, framealpha=0.9)
ax.set_title(f"{args.long} scored with thresholds estimated on the full recording vs on its first "
             f"{args.short} span (dotted line)", fontsize=11)
fig.tight_layout()
out = repo_root / "data/derivatives" / args.long / "threshold_stability.png"
fig.savefig(out, dpi=200)
print("wrote", out)

print(f"\n{'combo':<16} {'window':<12} {'sw':>7} {'motion':>7} {'theta':>7}")
for row in rows:
    for w, (a, b, c) in enumerate(row["thr"]):
        print(f"{row['probe'][-1] + '/' + row['variant'].replace('lfp_', ''):<16} "
              f"{WINDOWS[w]:<12} {a:>7.4f} {b:>7.4f} {c:>7.4f}")

print(f"\n{'combo':<16} {'kappa':>7} {'dice nrem':>10} {'dice rem':>9} {'dice wake':>10} "
      f"{'prefix span':>18} {'outside':>8}")
for row in rows:
    lo, hi = row["span"]
    print(f"{row['probe'][-1] + '/' + row['variant'].replace('lfp_', ''):<16} {row['kappa']:>7.4f} "
          f"{row['dice']['nrem']:>10.4f} {row['dice']['rem']:>9.4f} {row['dice']['wake']:>10.4f} "
          f"{'[' + format(lo, '.3f') + ', ' + format(hi, '.3f') + ']':>18} {row['outside']:>7.2%}")
print("\ndice = nan where neither classification found that state (ProbeA finds no REM at all).")
