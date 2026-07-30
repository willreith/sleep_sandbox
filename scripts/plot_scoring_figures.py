"""Generate the notebook-style diagnostic figures (sleep_sandbox.scoring.plot_notebook_figures)
for existing sleep-scoring results under data/derivatives/{Probe}/{variant}/result.npz. Reloads the
lfp_cmr/lfp_nocmr + emg derivatives and aligned IMU (same pattern as run_scoring.py) to recompute the
extra raw signals plot_notebook_figures needs (notebook_extras: per-convention theta channels,
spectrogram, raw EMG, IMU angular/accel); these are cached to result_extras.npz next to each
result.npz so a rerun skips the recompute. Requires run_scoring.py to have already been run for the
combo(s) being plotted.

Usage: python plot_scoring_figures.py [--out-base data/derivatives] [--probe ProbeA|ProbeB] [--variant lfp_cmr|lfp_nocmr]
Omitting --probe/--variant runs all 4 combos in one invocation.
"""

import os
import re
import argparse
from pathlib import Path

import yaml
import numpy as np
from dotenv import load_dotenv
from probeinterface import read_probeinterface

from sleep_sandbox.io import load_preprocessed, align_bno055_to_lfp, find_amplifier_files
from sleep_sandbox.preprocessing import imu_kinematics
from sleep_sandbox.scoring import notebook_extras, plot_notebook_figures

repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")

parser = argparse.ArgumentParser()
parser.add_argument("--out-base", type=Path, default=repo_root / "data" / "derivatives",
                     help="base dir; reads/writes under {out_base}/{probe}/{variant}/")
parser.add_argument("--probe", choices=["ProbeA", "ProbeB"], help="run only this probe (default: both)")
parser.add_argument("--variant", choices=["lfp_cmr", "lfp_nocmr"],
                     help="run only this LFP variant (default: both)")
args = parser.parse_args()
out_base = args.out_base

raw_dir = Path(os.environ["PREPRO_RAW_DIR"])
probes = {
    "ProbeA": Path(os.environ["PROBEA_DERIV_DIR"]),
    "ProbeB": Path(os.environ["PROBEB_DERIV_DIR"]),
}
if args.probe:
    probes = {args.probe: probes[args.probe]}
variants = [args.variant] if args.variant else ["lfp_cmr", "lfp_nocmr"]


def find_folder(deriv_dir, stream):
    matches = sorted(deriv_dir.glob(f"{stream}__*"))
    assert len(matches) == 1, f"expected exactly one '{stream}' folder in {deriv_dir}, got {matches}"
    return matches[0]


for probe, deriv_dir in probes.items():
    print(f"=== {probe} ===", flush=True)

    # Same segment-range restriction as run_scoring.py (encoded in the derivative folder name).
    seg_start, seg_end = (int(n) for n in re.search(r"_seg(\d+)-(\d+)$", deriv_dir.name).groups())
    imu_blocks = range(seg_start, seg_end + 1)

    print(f"loading + aligning IMU (blocks {seg_start}-{seg_end})...", flush=True)
    imu_data, imu_t, imu_valid, imu_edges = align_bno055_to_lfp(raw_dir, blocks=imu_blocks, probe=probe)

    print("loading emg derivative...", flush=True)
    recording_emg = load_preprocessed(deriv_dir, "emg")
    _, probe_config_path = find_amplifier_files(raw_dir.parent, probe)
    probe_obj = read_probeinterface(probe_config_path).probes[0]
    active_probe = probe_obj.get_slice(probe_obj.device_channel_indices != -1)
    recording_emg = recording_emg.set_probe(active_probe)

    for variant in variants:
        print(f"--- {probe} {variant} ---", flush=True)
        out_dir = out_base / probe / variant
        result_path, yml_path = out_dir / "result.npz", out_dir / "result.yml"
        assert result_path.exists() and yml_path.exists(), (
            f"missing {result_path} / {yml_path} -- run run_scoring.py for {probe}/{variant} first")

        with open(yml_path) as f:
            sidecar = yaml.safe_load(f)
        scoring_config = sidecar["scoring_config"]
        result = dict(np.load(result_path))
        result.update(sidecar["result_scalars"])

        print(f"loading {variant} derivative...", flush=True)
        recording_lfp = load_preprocessed(deriv_dir, variant)

        imu_kin_cfg = scoring_config["imu"]["kinematics"]
        imu_speed, imu_ang, imu_accel = imu_kinematics(
            imu_data, imu_t, imu_edges, hp_fc=imu_kin_cfg["highpass_fc_hz"], hp_order=imu_kin_cfg["highpass_order"])

        print("computing notebook extras (cached to result_extras.npz)...", flush=True)
        extras = notebook_extras(
            recording_lfp, recording_emg, scoring_config, result["times"],
            imu_t, imu_valid, imu_ang, imu_accel, result,
            cache_path=out_dir / "result_extras.npz")

        print(f"plotting -> {out_dir}...", flush=True)
        plot_notebook_figures(result, extras, scoring_config, out_dir=out_dir)
        print(f"--- {probe} {variant} done ---", flush=True)

print("ALL PLOTS COMPLETE")
