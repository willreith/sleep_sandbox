"""EMG proxy on the pre-downsampled recording, per midday chunk or in one pass, for chunked scoring.

emg_from_lfp has no fitted parameters and its 1 s windows are independent, and chunk boundaries fall
on whole seconds, so the per-chunk outputs together equal a whole-recording run exactly.

Usage: python compute_emg.py [--chunk N] [--probe ProbeB] [--shanks 0 1 2 3] [--experiment NAME]
--chunk N computes one midday_chunks chunk (output emg_chunkNN.*); without it, the whole recording (emg.*).
--experiment and --zarr-pattern point it at another session's zarrs; the defaults are abcEphys01's.
"""

import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

from sleep_sandbox.io import load_session_timeline, midday_chunks, load_input_recording
from sleep_sandbox.analysis import make_emg_pairs, emg_from_lfp

repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")

parser = argparse.ArgumentParser()
parser.add_argument("--emg-dir", type=Path, default=Path(os.environ["PREPRO_EMG_DIR"]),
                    help="dir of per-shank EMG zarrs; its name (e.g. emg_pp_no_cmr) goes into the output path")
parser.add_argument("--chunk", type=int, default=None, help="midday_chunks index; omit for the whole recording")
parser.add_argument("--probe", default="ProbeB")
parser.add_argument("--shanks", type=int, nargs="+", default=[0, 1, 2, 3])
parser.add_argument("--csv", type=Path, default=None,
                    help="ephys-paths CSV (default: derived from --experiment and --probe)")
parser.add_argument("--experiment", default="abcEphys01", help="names the --csv and --out-base defaults")
parser.add_argument("--zarr-pattern", default="emg_pp_{probe}_shank_{shank}.zarr",
                    help="EMG zarr filename in --emg-dir")
parser.add_argument("--out-base", type=Path, default=None)
args = parser.parse_args()

out_base = args.out_base or repo_root / "data" / "derivatives" / args.experiment
csv_path = args.csv or args.emg_dir.parent / f"{args.experiment}_{args.probe[-1].upper()}_ephys_paths.csv"
zarr_paths = [args.emg_dir / args.zarr_pattern.format(probe=args.probe, shank=s) for s in args.shanks]

with open(repo_root / "config/sleep_scoring.yml") as f:
    emg_cfg = yaml.safe_load(f)["emg"]

rec_start, _ = load_session_timeline(csv_path)
rec = load_input_recording(zarr_paths)
groups = np.asarray(rec.get_property("group"))
pairs, combos = make_emg_pairs(groups, n_pairs=emg_cfg["n_pairs"], min_shank_dist=emg_cfg["min_shank_dist"],
                               seed=emg_cfg["seed"])
print(f"{rec.get_num_channels()} ch, {rec.get_num_frames()} frames; {len(pairs)} pairs over shank combos "
      f"{[tuple(map(int, c)) for c in combos]}, {len(np.unique(pairs))} channels", flush=True)

fs = rec.get_sampling_frequency()
start_frame, stem, chunk_meta = 0, "emg", {}
if args.chunk is not None:
    c = midday_chunks(rec_start, rec.get_num_frames(), fs)[args.chunk]
    rec = rec.frame_slice(start_frame=c["start_frame"], end_frame=c["end_frame"])
    start_frame, stem = c["start_frame"], f"emg_chunk{args.chunk:02d}"
    chunk_meta = {"chunk": c["label"], "t_start": str(c["t_start"]), "t_end": str(c["t_end"]),
                  "start_frame": int(c["start_frame"]), "end_frame": int(c["end_frame"])}

tic = time.time()
emg, times = emg_from_lfp(rec, pairs, win_s=emg_cfg["window_s"])
times = times + start_frame / fs   # seconds from recording start, whatever the chunk
runtime_s = time.time() - tic

nan_frac = float(np.isnan(emg).mean())
# NaN tracks the zero-filled blanks, whose share differs per experiment: 0.3% on abcEphys01 (chunks 0,
# 11, 13, 14, 15), 11.5% on abcEphysPilot04, where one 14.25 h blank leaves chunk02 56.6% empty. Well
# above that experiment's share means a flat channel is NaN-ing every window it is paired in.
print(f"{len(emg)} windows in {runtime_s / 3600:.2f} h; NaN fraction {nan_frac:.4%} "
      f"({nan_frac * len(emg) * emg_cfg['window_s'] / 3600:.2f} h)", flush=True)

out = out_base / args.emg_dir.name / f"{args.probe}_shanks{''.join(map(str, args.shanks))}"
out.mkdir(parents=True, exist_ok=True)
np.savez(out / f"{stem}.npz", emg=emg, times=times, pairs=pairs, pair_channel_ids=np.asarray(rec.channel_ids)[pairs],
         win_s=emg_cfg["window_s"])
(out / f"{stem}.json").write_text(json.dumps({
    **chunk_meta,
    "zarr_paths": [str(p) for p in zarr_paths], "rec_start": str(rec_start),
    "emg_config": emg_cfg, "combos": [list(map(int, c)) for c in combos],
    "n_windows": len(emg), "nan_fraction": nan_frac, "runtime_s": runtime_s,
}, indent=2))
print(f"wrote {out}/", flush=True)
