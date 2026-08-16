"""Recompute thresholds and state masks for existing scoring results, in place.

score_recording's metrics (sw_metric/theta_metric/motion_metric) and both channel selections depend
only on the raw recordings, not on the threshold method or the duration criteria, so they are reused
from the saved result.npz instead of being recomputed -- a full run_scoring.py rerun would recompute
every spectrogram and channel selection to land on bit-identical metrics. Only sw_thresh and
classify() are redone, driven by config/sleep_scoring.yml as it stands now.

result.npz and result.yml are backed up to .bak before being overwritten. The sidecar's 'sources'
block is carried over unchanged; its 'scoring_config' is replaced with the current config. That
replacement is required, not cosmetic: plot_scoring_figures.py reads scoring_config back out of the
sidecar rather than from the repo config, so a stale sidecar makes plot_notebook_figures raise
KeyError on the new 'threshold' block.

Usage: python reclassify.py [--out-base data/derivatives]
"""

import shutil
import argparse
from pathlib import Path

import numpy as np
import yaml

from sleep_sandbox.analysis import find_thresh
from sleep_sandbox.scoring import classify, STATES

repo_root = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--seg", required=True,
                     help="segment range, e.g. 'seg5-148'; reclassifies under {out_base}/{seg}/")
parser.add_argument("--out-base", type=Path, default=repo_root / "data" / "derivatives")
args = parser.parse_args()

with open(repo_root / "config/sleep_scoring.yml") as f:
    scoring_config = yaml.safe_load(f)

step_s = scoring_config["spectrogram"]["step_s"]
bt_startbins = scoring_config["bimodal_threshold"]["startbins"]
bt_maxbins = scoring_config["bimodal_threshold"]["maxbins"]
thresh_cfg = scoring_config["threshold"]
dur_cfg = scoring_config["duration_criteria"]

for probe in ["ProbeA", "ProbeB"]:
    for variant in ["lfp_cmr", "lfp_nocmr"]:
        out_dir = args.out_base / args.seg / probe / variant
        npz_path, yml_path = out_dir / "result.npz", out_dir / "result.yml"
        print(f"--- {probe} {variant} ---", flush=True)

        result = dict(np.load(npz_path))
        with open(yml_path) as f:
            sidecar = yaml.safe_load(f)
        old_scalars = sidecar["result_scalars"]
        # Results saved before wake/qwake/ma were added hold only nrem/rem/rem_cand/mov.
        old_frac = {st: float(result[st].mean()) for st in STATES if st in result}

        shutil.copy2(npz_path, npz_path.with_suffix(".npz.bak"))
        shutil.copy2(yml_path, yml_path.with_suffix(".yml.bak"))

        sw_thresh = find_thresh(result["sw_metric"], thresh_cfg["method"], bt_startbins, bt_maxbins,
                                thresh_cfg["kde_grid_n"], thresh_cfg["min_prominence_frac"],
                                label=f"slow_wave[{probe}/{variant}]")
        states = classify(
            result["sw_metric"], result["theta_metric"], result["motion_metric"], sw_thresh,
            bt_startbins, bt_maxbins, thresh_cfg["method"], thresh_cfg["kde_grid_n"], dt=step_s,
            merge_shorter_than_s=dur_cfg["merge_shorter_than_s"],
            min_state_s=dur_cfg["min_state_s"],
            microarousal_max_s=dur_cfg["microarousal_max_s"],
            theta_conditioned=scoring_config["theta"]["movement_conditioned"],
            min_prominence_frac=thresh_cfg["min_prominence_frac"])

        result.update({k: v for k, v in states.items() if isinstance(v, np.ndarray)})
        np.savez(npz_path, **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})

        scalars = dict(old_scalars)
        scalars["sw_thresh"] = float(sw_thresh)
        scalars.update({k: float(v) for k, v in states.items() if not isinstance(v, np.ndarray)})
        sidecar["scoring_config"] = scoring_config
        sidecar["result_scalars"] = scalars
        with open(yml_path, "w") as f:
            yaml.safe_dump(sidecar, f, sort_keys=False)

        print(f"  sw_thresh   {old_scalars['sw_thresh']:.4g} -> {sw_thresh:.4g}")
        print(f"  motion_thr  {old_scalars['motion_thresh']:.4g} -> {states['motion_thresh']:.4g}")
        print(f"  th_thresh   {old_scalars['th_thresh']:.4g} -> {states['th_thresh']:.4g}")
        for st in STATES:
            old = f"{old_frac[st]:.4f}" if st in old_frac else " (new)"
            print(f"  {st:5s}       {old} -> {float(states[st].mean()):.4f}")
        print(f"  ma          (new) {float(states['ma'].mean()):.4f}")

print("ALL RECLASSIFIED (originals at result.npz.bak / result.yml.bak)")
