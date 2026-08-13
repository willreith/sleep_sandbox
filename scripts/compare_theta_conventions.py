"""Score each saved recording with three theta ratio conventions and compare the resulting state
sequences. Motion is fixed to the EMG proxy (result['motion_metric'], already smoothed) across all
three, so the theta convention is the only thing that varies:

  watson      5-10 Hz / 2-16 Hz    (Watson et al. 2016)
  shin        6-12 Hz / 1-4 Hz     (Shin et al. 2026)
  shin_mod    5-10 Hz / 1-4 Hz     (local variant)

Each convention uses its *own* dip-selected channel (theta_own_channel_<conv> /
theta_own_ratio_<conv> in result_extras.npz), not score_recording's shared peakTH channel -- so
this isolates the combined effect of channel choice + band definition per convention, letting
channel selection adapt if e.g. a PFC probe's peakTH channel just doesn't carry theta at all.
This is a different (and for this comparison, more appropriate) choice than score_recording's
production path, which shares one peakTH channel across all conventions -- see
docs/sleep_classification_algorithm.md.

Requires run_scoring.py and plot_scoring_figures.py to have been run (reads result.npz +
result_extras.npz; no raw IMU needed, unlike compare_motion_sources.py). Writes
theta_comparison_{diagnostics.json,confusion.png,distributions.png,hypnogram.png} into each
{out_base}/{probe}/{variant}/theta_convention_comparison/.

Usage: python compare_theta_conventions.py [--out-base data/derivatives]
"""

import json
import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from diptest import diptest

from sleep_sandbox.analysis import (
    smooth_norm, state_codes, confusion, cohens_kappa, state_intervals, bout_durations,
)
from sleep_sandbox.scoring import classify

repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")

parser = argparse.ArgumentParser()
parser.add_argument("--seg", required=True,
                     help="segment range, e.g. 'seg5-148'; reads/writes under {out_base}/{seg}/")
parser.add_argument("--out-base", type=Path, default=repo_root / "data" / "derivatives")
args = parser.parse_args()

with open(repo_root / "config/sleep_scoring.yml") as f:
    scoring_config = yaml.safe_load(f)

step_s = scoring_config["spectrogram"]["step_s"]
smooth_win_s = scoring_config["smoothing"]["window_s"]
bt_startbins = scoring_config["bimodal_threshold"]["startbins"]
bt_maxbins = scoring_config["bimodal_threshold"]["maxbins"]
thresh_cfg = scoring_config["threshold"]
dur_cfg = scoring_config["duration_criteria"]

STATES = ["nrem", "rem", "wake"]
N_STATES = len(STATES)
CONVENTIONS = list(scoring_config["theta"]["conventions"])   # ["watson", "shin", "shin_mod"]

for probe in ["ProbeA", "ProbeB"]:
    for variant in ["lfp_cmr", "lfp_nocmr"]:
        out_dir = args.out_base / args.seg / probe / variant
        fig_dir = out_dir / "theta_convention_comparison"
        fig_dir.mkdir(parents=True, exist_ok=True)
        print(f"--- {probe} {variant} ---", flush=True)

        result = dict(np.load(out_dir / "result.npz"))
        extras = dict(np.load(out_dir / "result_extras.npz"))
        sidecar = yaml.safe_load(open(out_dir / "result.yml"))
        times = result["times"]
        sw_metric = result["sw_metric"]
        motion_metric = result["motion_metric"]          # EMG, already smoothed -- fixed across conventions
        sw_thresh = sidecar["result_scalars"]["sw_thresh"]
        dt = float(np.median(np.diff(times)))

        theta = {conv: smooth_norm(extras[f"theta_own_ratio_{conv}"], step_s=step_s, win_s=smooth_win_s)
                 for conv in CONVENTIONS}
        channel = {conv: int(extras[f"theta_own_channel_{conv}"]) for conv in CONVENTIONS}

        scored = {conv: classify(sw_metric, theta[conv], motion_metric, sw_thresh, bt_startbins,
                                 bt_maxbins, thresh_cfg["method"], thresh_cfg["kde_grid_n"], dt=dt,
                                 merge_shorter_than_s=dur_cfg["merge_shorter_than_s"],
                                 min_state_s=dur_cfg["min_state_s"],
                                 microarousal_max_s=dur_cfg["microarousal_max_s"],
                                 theta_conditioned=scoring_config["theta"]["movement_conditioned"])
                  for conv in CONVENTIONS}
        codes = {conv: state_codes(s, STATES) for conv, s in scored.items()}

        diagnostics = {
            "n_epochs": int(len(times)), "dt_s": dt, "sw_thresh": float(sw_thresh),
            "per_convention": {}, "pairwise": {}, "theta_metric_correlations": {},
        }

        for conv in CONVENTIONS:
            s, m = scored[conv], theta[conv]
            dip, dip_p = diptest(m)
            rem_bouts = bout_durations(s["rem"], dt)
            diagnostics["per_convention"][conv] = {
                "channel": channel[conv], "th_thresh": float(s["th_thresh"]),
                "frac": {st: float(s[st].mean()) for st in STATES},
                "frac_qwake": float(s["qwake"].mean()),
                # Bimodality of the theta *ratio itself* -- bimodal_thresh's histogram-trough
                # split is only meaningful if the distribution has two modes at all. On a probe
                # where theta isn't well expressed (e.g. PFC), the ratio may be unimodal on every
                # channel, in which case the trough threshold is an arbitrary cut, not a real
                # narrowband-theta/no-theta boundary.
                "theta_dip": float(dip), "theta_dip_p": float(dip_p),
                "rem_bouts": {"n": int(len(rem_bouts)),
                              "median_s": float(np.median(rem_bouts)) if len(rem_bouts) else None,
                              "frac_under_30s": float((rem_bouts < 30).mean()) if len(rem_bouts) else None},
            }

        for a, b in combinations(CONVENTIONS, 2):
            cm = confusion(codes[a], codes[b], N_STATES)
            key = f"{a}__vs__{b}"
            diagnostics["pairwise"][key] = {
                "same_channel": channel[a] == channel[b],
                "confusion_rows_are_first": cm.tolist(), "states": STATES,
                "raw_agreement": float(np.trace(cm) / cm.sum()),
                "cohens_kappa": float(cohens_kappa(codes[a], codes[b], N_STATES)),
                # Per-state Dice: symmetric overlap, no convention treated as ground truth.
                "dice": {st: float(2 * (scored[a][st] & scored[b][st]).sum() /
                                   max(scored[a][st].sum() + scored[b][st].sum(), 1)) for st in STATES},
            }
            diagnostics["theta_metric_correlations"][key] = float(np.corrcoef(theta[a], theta[b])[0, 1])

        with open(fig_dir / "theta_comparison_diagnostics.json", "w") as f:
            json.dump(diagnostics, f, indent=2)

        # Confusion matrices, all pairs.
        pairs = list(combinations(CONVENTIONS, 2))
        fig, axes = plt.subplots(1, len(pairs), figsize=(4.3 * len(pairs), 4))
        for ax, (a, b) in zip(np.atleast_1d(axes), pairs):
            cm = confusion(codes[a], codes[b], N_STATES)
            cmn = cm / cm.sum()
            ax.imshow(cmn, cmap="Blues", vmin=0, vmax=cmn.max())
            for i in range(N_STATES):
                for j in range(N_STATES):
                    ax.text(j, i, f"{cmn[i, j]:.3f}", ha="center", va="center",
                            color="white" if cmn[i, j] > cmn.max() / 2 else "black", fontsize=9)
            ax.set_xticks(range(N_STATES), STATES)
            ax.set_yticks(range(N_STATES), STATES)
            ax.set_ylabel(a); ax.set_xlabel(b)
            ax.set_title(f"kappa={diagnostics['pairwise'][f'{a}__vs__{b}']['cohens_kappa']:.3f}", fontsize=10)
        fig.suptitle(f"{probe} {variant}: theta-convention state confusion (fraction of all epochs)")
        fig.tight_layout()
        fig.savefig(fig_dir / "theta_comparison_confusion.png", dpi=150)
        plt.close(fig)

        # Theta metric distributions + threshold, and traces over time.
        fig, axes = plt.subplots(2, len(CONVENTIONS), figsize=(4 * len(CONVENTIONS), 7))
        for ax, conv in zip(axes[0], CONVENTIONS):
            d = diagnostics["per_convention"][conv]
            ax.hist(theta[conv], bins=60, density=True, alpha=0.6)
            ax.axvline(d["th_thresh"], color="r", ls="--")
            ax.set_title(f"{conv} (ch{channel[conv]})\ndip={d['theta_dip']:.4f} p={d['theta_dip_p']:.3f}",
                         fontsize=9)
            ax.set_xlabel("theta metric [0,1]")
        axes[0][0].set_ylabel("density")
        for ax, conv in zip(axes[1], CONVENTIONS):
            ax.plot(times, theta[conv], lw=0.3)
            ax.axhline(diagnostics["per_convention"][conv]["th_thresh"], color="r", ls="--", lw=0.8)
            ax.set_xlabel("time (s)")
        axes[1][0].set_ylabel("theta metric [0,1]")
        fig.suptitle(f"{probe} {variant}: theta conventions (own-channel)")
        fig.tight_layout()
        fig.savefig(fig_dir / "theta_comparison_distributions.png", dpi=150)
        plt.close(fig)

        # Hypnograms: per-state horizontal-bar lanes, one row per convention.
        state_color = {"nrem": "tab:blue", "rem": "tab:red", "wake": "tab:green"}
        fig, axes = plt.subplots(len(CONVENTIONS), 1, figsize=(14, 2.0 * len(CONVENTIONS)), sharex=True)
        for ax, conv in zip(axes, CONVENTIONS):
            for i, st in enumerate(STATES):
                bars = state_intervals(scored[conv][st], times, dt)
                ax.broken_barh(bars, (i - 0.4, 0.8), facecolors=state_color[st])
            ax.set_yticks(range(N_STATES), STATES)
            ax.set_ylabel(f"{conv}\n(ch{channel[conv]})", fontsize=9)
            ax.set_ylim(-0.5, N_STATES - 0.5)
        axes[-1].set_xlabel("time (s)")
        fig.suptitle(f"{probe} {variant}: hypnograms by theta convention (own channel, EMG motion)")
        fig.tight_layout()
        fig.savefig(fig_dir / "theta_comparison_hypnogram.png", dpi=150)
        plt.close(fig)

        print(json.dumps({conv: {"channel": channel[conv], **diagnostics["per_convention"][conv]["frac"]}
                           for conv in CONVENTIONS}, indent=2), flush=True)
        print({k: round(d["cohens_kappa"], 3) for k, d in diagnostics["pairwise"].items()}, flush=True)
        print(f"--- {probe} {variant} done -> {fig_dir} ---", flush=True)

print("ALL THETA COMPARISONS COMPLETE")
