"""Score each saved recording four ways -- one per motion signal -- and compare the resulting
state sequences. All four share the same sw_metric / theta_metric / sw_thresh (loaded from
run_scoring.py's result.npz, since the motion signal doesn't enter slow-wave or theta), so the
motion source is the only thing that varies:

  A       EMG proxy (cross-shank high-frequency correlation) -- buzcode SleepScoreMaster default
  B1      |accel| bandpassed 0.1-1 Hz -> abs -> bin-mean     -- buzcode bz_getIntanAccel recipe
  B2      IMU translational speed                             -- no buzcode precedent
  B3      per-bin variance of |accel|                         -- no buzcode precedent

Requires run_scoring.py and plot_scoring_figures.py to have been run (reads result.npz +
result_extras.npz). B1 is the only one needing raw IMU, since it must filter at the native IMU
rate before binning. Writes comparison_{diagnostics.json,confusion.png,motion.png,hypnogram.png}
into each {out_base}/{probe}/{variant}/motion_sources/.

Usage: python compare_motion_sources.py [--out-base data/derivatives]
"""

import os
import re
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

from sleep_sandbox.io import align_bno055_to_lfp
from sleep_sandbox.preprocessing import imu_kinematics
from sleep_sandbox.analysis import (
    smooth_norm, accel_motion_buzcode, state_codes, confusion, cohens_kappa,
    state_intervals, bout_durations,
)
from sleep_sandbox.scoring import classify

repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")

parser = argparse.ArgumentParser()
parser.add_argument("--seg", required=True,
                     help="preprocessed segment range, e.g. 'seg5-148'; selects both the "
                          "$PREPRO_OUTPUT_DIR/{probe}_{seg} inputs and the {out_base}/{seg} outputs")
parser.add_argument("--out-base", type=Path, default=repo_root / "data" / "derivatives")
args = parser.parse_args()

with open(repo_root / "config/sleep_scoring.yml") as f:
    scoring_config = yaml.safe_load(f)

raw_dir = Path(os.environ["PREPRO_RAW_DIR"])
prepro_base = Path(os.environ["PREPRO_OUTPUT_DIR"])
deriv_dirs = {p: prepro_base / f"{p}_{args.seg}" for p in ("ProbeA", "ProbeB")}

step_s = scoring_config["spectrogram"]["step_s"]
smooth_win_s = scoring_config["smoothing"]["window_s"]
bt_startbins = scoring_config["bimodal_threshold"]["startbins"]
bt_maxbins = scoring_config["bimodal_threshold"]["maxbins"]
thresh_cfg = scoring_config["threshold"]
dur_cfg = scoring_config["duration_criteria"]
accel_cfg = scoring_config["imu"]["accel_motion"]

STATES = ["nrem", "rem", "wake"]
VERSIONS = ["A_emg", "B1_accel_bp", "B2_speed", "B3_accel_var"]
N_STATES = len(STATES)


for probe, deriv_dir in deriv_dirs.items():
    seg_start, seg_end = (int(n) for n in re.search(r"_seg(\d+)-(\d+)$", deriv_dir.name).groups())

    print(f"=== {probe}: loading + aligning IMU (blocks {seg_start}-{seg_end}) ===", flush=True)
    imu_data, imu_t, imu_valid, imu_edges = align_bno055_to_lfp(
        raw_dir, blocks=range(seg_start, seg_end + 1), probe=probe)
    imu_kin_cfg = scoring_config["imu"]["kinematics"]
    _, _, imu_accel = imu_kinematics(
        imu_data, imu_t, imu_edges, hp_fc=imu_kin_cfg["highpass_fc_hz"], hp_order=imu_kin_cfg["highpass_order"])

    for variant in ["lfp_cmr", "lfp_nocmr"]:
        out_dir = args.out_base / args.seg / probe / variant
        fig_dir = out_dir / "motion_sources"
        fig_dir.mkdir(parents=True, exist_ok=True)
        print(f"--- {probe} {variant} ---", flush=True)

        result = dict(np.load(out_dir / "result.npz"))
        extras = dict(np.load(out_dir / "result_extras.npz"))
        sidecar = yaml.safe_load(open(out_dir / "result.yml"))
        times = result["times"]
        sw_metric, theta_metric = result["sw_metric"], result["theta_metric"]
        sw_thresh = sidecar["result_scalars"]["sw_thresh"]
        dt = float(np.median(np.diff(times)))

        # B1 filters at the native IMU rate (~100 Hz) before binning -- the 1 s grid can't
        # represent a 1 Hz corner, so it can't be built from the already-binned imu_accel_b.
        tv = imu_t[imu_valid]
        accel_bp = accel_motion_buzcode(
            imu_accel[imu_valid], tv, times,
            low_hz=accel_cfg["low_hz"], high_hz=accel_cfg["high_hz"], order=accel_cfg["order"])

        raw_motion = {
            "A_emg": extras["emg_b"],
            "B1_accel_bp": accel_bp,
            "B2_speed": result["imu_speed"],
            "B3_accel_var": extras["imu_accel_var_b"],
        }
        # Empty IMU bins are NaN; smooth_norm/bimodal_thresh need finite input. Carry the last
        # finite value forward (bins are 1 s, gaps are sparse) so an occasional dropout doesn't
        # register as a spurious zero-motion epoch.
        motion = {}
        for name, sig in raw_motion.items():
            x = np.asarray(sig, dtype=float)
            bad = ~np.isfinite(x)
            if bad.any():
                x = x.copy()
                x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), x[~bad])
            motion[name] = smooth_norm(x, step_s=step_s, win_s=smooth_win_s)

        scored = {v: classify(sw_metric, theta_metric, motion[v], sw_thresh, bt_startbins, bt_maxbins,
                              thresh_cfg["method"], thresh_cfg["kde_grid_n"], dt=dt,
                              merge_shorter_than_s=dur_cfg["merge_shorter_than_s"],
                              min_state_s=dur_cfg["min_state_s"],
                              microarousal_max_s=dur_cfg["microarousal_max_s"],
                              theta_conditioned=scoring_config["theta"]["movement_conditioned"])
                  for v in VERSIONS}
        codes = {v: state_codes(s, STATES) for v, s in scored.items()}

        diagnostics = {
            "n_epochs": int(len(times)), "dt_s": dt, "sw_thresh": float(sw_thresh),
            "per_version": {}, "pairwise": {},
            "motion_metric_correlations": {},
        }

        for v in VERSIONS:
            s, m = scored[v], motion[v]
            dip, dip_p = diptest(m)
            rem_bouts, nrem_bouts = bout_durations(s["rem"], dt), bout_durations(s["nrem"], dt)
            diagnostics["per_version"][v] = {
                "motion_thresh": float(s["motion_thresh"]), "th_thresh": float(s["th_thresh"]),
                "frac": {st: float(s[st].mean()) for st in STATES},
                "frac_qwake": float(s["qwake"].mean()), "frac_mov": float(s["mov"].mean()),
                # Bimodality of the motion metric: bimodal_thresh's histogram-trough split is only
                # meaningful if the distribution actually has two modes. A high dip p-value means
                # the threshold is an arbitrary cut through a unimodal blob.
                "motion_dip": float(dip), "motion_dip_p": float(dip_p),
                "rem_bouts": {"n": int(len(rem_bouts)),
                              "median_s": float(np.median(rem_bouts)) if len(rem_bouts) else None,
                              "max_s": float(rem_bouts.max()) if len(rem_bouts) else None,
                              "frac_under_30s": float((rem_bouts < 30).mean()) if len(rem_bouts) else None},
                "nrem_bouts": {"n": int(len(nrem_bouts)),
                               "median_s": float(np.median(nrem_bouts)) if len(nrem_bouts) else None},
            }

        for a, b in combinations(VERSIONS, 2):
            cm = confusion(codes[a], codes[b], N_STATES)
            key = f"{a}__vs__{b}"
            diagnostics["pairwise"][key] = {
                "confusion_rows_are_first": cm.tolist(), "states": STATES,
                "raw_agreement": float(np.trace(cm) / cm.sum()),
                "cohens_kappa": float(cohens_kappa(codes[a], codes[b], N_STATES)),
                # Per-state Dice: symmetric overlap, so no version is treated as ground truth.
                "dice": {st: float(2 * (scored[a][st] & scored[b][st]).sum() /
                                   max(scored[a][st].sum() + scored[b][st].sum(), 1)) for st in STATES},
            }
            # Continuous-metric correlation: distinguishes "different signal" from "same signal,
            # different threshold". Near-1 r with low kappa means the threshold is the problem.
            diagnostics["motion_metric_correlations"][key] = float(np.corrcoef(motion[a], motion[b])[0, 1])

        with open(fig_dir / "comparison_diagnostics.json", "w") as f:
            json.dump(diagnostics, f, indent=2)

        # Confusion matrices, all pairs.
        pairs = list(combinations(VERSIONS, 2))
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        for ax, (a, b) in zip(axes.ravel(), pairs):
            cm = confusion(codes[a], codes[b], N_STATES)
            cmn = cm / cm.sum()
            ax.imshow(cmn, cmap="Blues", vmin=0, vmax=cmn.max())
            for i in range(len(STATES)):
                for j in range(len(STATES)):
                    ax.text(j, i, f"{cmn[i, j]:.3f}", ha="center", va="center",
                            color="white" if cmn[i, j] > cmn.max() / 2 else "black", fontsize=9)
            ax.set_xticks(range(len(STATES)), STATES)
            ax.set_yticks(range(len(STATES)), STATES)
            ax.set_ylabel(a); ax.set_xlabel(b)
            ax.set_title(f"kappa={diagnostics['pairwise'][f'{a}__vs__{b}']['cohens_kappa']:.3f}", fontsize=10)
        fig.suptitle(f"{probe} {variant}: state confusion (fraction of all epochs)")
        fig.tight_layout()
        fig.savefig(fig_dir / "comparison_confusion.png", dpi=150)
        plt.close(fig)

        # Motion metric distributions + thresholds, and pairwise scatter.
        fig, axes = plt.subplots(2, len(VERSIONS), figsize=(4 * len(VERSIONS), 7))
        for ax, v in zip(axes[0], VERSIONS):
            d = diagnostics["per_version"][v]
            ax.hist(motion[v], bins=60, density=True, alpha=0.6)
            ax.axvline(d["motion_thresh"], color="r", ls="--")
            ax.set_title(f"{v}\ndip={d['motion_dip']:.4f} p={d['motion_dip_p']:.3f}", fontsize=9)
            ax.set_xlabel("motion metric [0,1]")
        axes[0][0].set_ylabel("density")
        for ax, v in zip(axes[1], VERSIONS):
            ax.plot(times, motion[v], lw=0.3)
            ax.axhline(diagnostics["per_version"][v]["motion_thresh"], color="r", ls="--", lw=0.8)
            ax.set_xlabel("time (s)")
        axes[1][0].set_ylabel("motion metric [0,1]")
        fig.suptitle(f"{probe} {variant}: motion signals")
        fig.tight_layout()
        fig.savefig(fig_dir / "comparison_motion.png", dpi=150)
        plt.close(fig)

        # Hypnograms: the actual deliverable -- state sequence per version, shared time axis.
        # Each state gets its own horizontal lane (broken_barh); a bar's presence/absence in a
        # lane is what state transitions look like, rather than a step trace jumping between
        # y-levels -- easier to read bout duration/count directly off the plot.
        state_color = {"nrem": "tab:blue", "rem": "tab:red", "wake": "tab:green"}
        fig, axes = plt.subplots(len(VERSIONS), 1, figsize=(14, 2.0 * len(VERSIONS)), sharex=True)
        for ax, v in zip(axes, VERSIONS):
            for i, st in enumerate(STATES):
                bars = state_intervals(scored[v][st], times, dt)
                ax.broken_barh(bars, (i - 0.4, 0.8), facecolors=state_color[st])
            ax.set_yticks(range(len(STATES)), STATES)
            ax.set_ylabel(v, fontsize=9)
            ax.set_ylim(-0.5, len(STATES) - 0.5)
        axes[-1].set_xlabel("time (s)")
        fig.suptitle(f"{probe} {variant}: hypnograms by motion source")
        fig.tight_layout()
        fig.savefig(fig_dir / "comparison_hypnogram.png", dpi=150)
        plt.close(fig)

        print(json.dumps({v: diagnostics["per_version"][v]["frac"] for v in VERSIONS}, indent=2), flush=True)
        print({k: round(d["cohens_kappa"], 3) for k, d in diagnostics["pairwise"].items()}, flush=True)
        print(f"--- {probe} {variant} done -> {fig_dir} ---", flush=True)

print("ALL COMPARISONS COMPLETE")
