"""Cache one full-chunk, single-channel spectrogram per (chunk, channel) unit.

Same grid, dtype path and file as select_global_basis.py's cached_spec (pc1_basis/spec_cache/
chunk{N}_ch{C}.npz with spec, freqs; bins centred at chunk start + window/2 + k * step), so files it
already wrote are skipped. Unit u is chunk u // len(channels), channel channels[u % len(channels)].

Usage: python cache_chunk_spectrograms.py --unit N --lfp-dir .../lfp_pp_pilot04 --csv ..._B_ephys_paths.csv
                                          [--channels 80 204] [--experiment abcEphysPilot04]
"""

import time
import argparse
from pathlib import Path

import numpy as np
import yaml

from sleep_sandbox.io import load_session_timeline, midday_chunks, load_input_recording
from sleep_sandbox.analysis import log_spectrogram

repo_root = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--unit", type=int, required=True)
parser.add_argument("--lfp-dir", type=Path, required=True, help="dir of per-shank LFP zarrs")
parser.add_argument("--csv", type=Path, required=True, help="ephys-paths CSV (recording start)")
parser.add_argument("--channels", type=int, nargs="+", default=[80, 204], help="aggregate channel indices")
parser.add_argument("--experiment", default="abcEphysPilot04")
parser.add_argument("--probe", default="ProbeB")
parser.add_argument("--shanks", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--out-base", type=Path, default=None)
args = parser.parse_args()


def zarr_path(shank):
    hits = sorted(args.lfp_dir.glob(f"*_{args.probe}_shank{shank}.zarr")) + \
           sorted(args.lfp_dir.glob(f"*_{args.probe}_shank_{shank}.zarr"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{len(hits)} zarrs in {args.lfp_dir} match {args.probe} shank {shank}")
    return hits[0]


with open(repo_root / "config/sleep_scoring.yml") as f:
    sc = yaml.safe_load(f)["spectrogram"]
skw = {k: sc[k] for k in ("window_s", "step_s", "freq_min", "freq_max", "n_freq_bins")}

shank_tag = f"{args.probe}_shanks{''.join(map(str, args.shanks))}"
cache = ((args.out_base or repo_root / "data" / "derivatives" / args.experiment) / args.lfp_dir.name
         / shank_tag / "pc1_basis" / "spec_cache")
rec = load_input_recording([zarr_path(s) for s in args.shanks])
fs = rec.get_sampling_frequency()
rec_start, _ = load_session_timeline(args.csv)
chunks = midday_chunks(rec_start, rec.get_num_frames(), fs)
ci, ch = args.unit // len(args.channels), args.channels[args.unit % len(args.channels)]
c = chunks[ci]
out = cache / f"chunk{ci}_ch{ch}.npz"
if out.exists():
    raise SystemExit(f"{out} exists, skipping")

# Read in 300 s slabs: one get_traces over 24 h peaked at 48+ GB on the 196-channel shank.
tic = time.time()
n = c["end_frame"] - c["start_frame"]
slab = int(300 * fs)
trace = np.empty(n, dtype=rec.get_dtype())
for s0 in range(0, n, slab):
    s1 = min(s0 + slab, n)
    trace[s0:s1] = rec.get_traces(start_frame=c["start_frame"] + s0, end_frame=c["start_frame"] + s1,
                                  channel_ids=[rec.channel_ids[ch]])[:, 0]
print(f"{c['label']} ch{ch}: read {n / fs / 3600:.2f} h in {time.time() - tic:.0f} s", flush=True)
spec, freqs, _ = log_spectrogram(trace, fs, **skw)
cache.mkdir(parents=True, exist_ok=True)
np.savez(out, spec=spec, freqs=freqs)
print(f"wrote {out} {spec.shape} in {time.time() - tic:.0f} s", flush=True)
