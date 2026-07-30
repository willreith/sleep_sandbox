"""Run the sleep-scoring pipeline (score_recording/summarize_run/plot_summary/save_run) on the
lfp_cmr and lfp_nocmr derivatives for ProbeA and ProbeB, each paired with that probe's emg
derivative and aligned IMU. Writes outputs into {out_base}/{probe}/{variant}/ for each run.

Usage: python run_scoring.py [--out-base data/derivatives] [--probe ProbeA|ProbeB] [--variant lfp_cmr|lfp_nocmr]
Omitting --probe/--variant runs all 4 combos in one invocation; passing both runs just that one
combo (for launching the 4 in parallel, e.g. via separate srun calls).
"""

import os
import re
import json
import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv
from probeinterface import read_probeinterface

from sleep_sandbox.io import load_preprocessed, align_bno055_to_lfp, find_amplifier_files
from sleep_sandbox.preprocessing import imu_kinematics
from sleep_sandbox.scoring import score_recording, summarize_run, plot_summary, save_run

repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")

parser = argparse.ArgumentParser()
parser.add_argument("--out-base", type=Path, default=repo_root / "data" / "derivatives",
                     help="base dir; outputs land under {out_base}/{probe}/{variant}/")
parser.add_argument("--probe", choices=["ProbeA", "ProbeB"], help="run only this probe (default: both)")
parser.add_argument("--variant", choices=["lfp_cmr", "lfp_nocmr"],
                     help="run only this LFP variant (default: both)")
args = parser.parse_args()
out_base = args.out_base

with open(repo_root / "config/sleep_scoring.yml") as f:
    scoring_config = yaml.safe_load(f)

raw_dir = Path(os.environ["PREPRO_RAW_DIR"])   # IMU (Bno055) lives alongside the raw ephys data
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

    # Restrict to the exact segment range these derivatives were built from (encoded in the
    # folder name, e.g. 'ProbeA_seg5-52') -- blocks=None would scan every Bno055 block in the raw
    # session, including ones far outside this recording (and, on this session, a corrupt one).
    seg_start, seg_end = (int(n) for n in re.search(r"_seg(\d+)-(\d+)$", deriv_dir.name).groups())
    imu_blocks = range(seg_start, seg_end + 1)

    print(f"loading + aligning IMU (blocks {seg_start}-{seg_end})...", flush=True)
    imu_data, imu_t, imu_valid, imu_edges = align_bno055_to_lfp(raw_dir, blocks=imu_blocks, probe=probe)
    imu_kin_cfg = scoring_config["imu"]["kinematics"]
    imu_speed, _, _ = imu_kinematics(
        imu_data, imu_t, imu_edges, hp_fc=imu_kin_cfg["highpass_fc_hz"], hp_order=imu_kin_cfg["highpass_order"])

    print("loading emg derivative...", flush=True)
    emg_folder = find_folder(deriv_dir, "emg")
    recording_emg = load_preprocessed(deriv_dir, "emg")

    # Derivatives are saved without a probe attached (SI drops it on save); reattach it here so
    # recording_emg.get_probes()[0].shank_ids works in score_recording (mirrors ripple.py's
    # si.load(path).set_probe(active) pattern).
    _, probe_config_path = find_amplifier_files(raw_dir.parent, probe)
    probe_obj = read_probeinterface(probe_config_path).probes[0]
    active_probe = probe_obj.get_slice(probe_obj.device_channel_indices != -1)
    recording_emg = recording_emg.set_probe(active_probe)

    for variant in variants:
        print(f"--- {probe} {variant} ---", flush=True)
        print(f"loading {variant} derivative...", flush=True)
        lfp_folder = find_folder(deriv_dir, variant)
        recording_lfp = load_preprocessed(deriv_dir, variant)

        print("running score_recording...", flush=True)
        result = score_recording(
            recording_lfp, recording_emg, scoring_config,
            imu_t=imu_t, imu_valid=imu_valid, imu_speed=imu_speed)

        summary = summarize_run(result)
        print(json.dumps(summary, indent=2))

        out_dir = out_base / probe / variant
        out_dir.mkdir(parents=True, exist_ok=True)

        # plot_summary writes distributions.png/correlations.png (+ imu_wake_xcorr.png, since IMU
        # data is passed above) into out_dir.
        print(f"plotting -> {out_dir}...", flush=True)
        plot_summary(result, out_dir=out_dir)
        print(f"saving -> {out_dir}...", flush=True)
        save_run(result, scoring_config, out_dir / "result.npz",
                 source_dirs={variant: lfp_folder, "emg": emg_folder})
        print(f"--- {probe} {variant} done ---", flush=True)

print("ALL RUNS COMPLETE")
