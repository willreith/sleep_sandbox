"""Pick the ripple detection channel on each shank, and measure how stable that choice is.

Usage: python select_ripple_channel.py --seg seg5-148 [--probe ProbeB] [--out-base data/derivatives]

Band power per channel is averaged over a random sample of NREM windows -- a PSD over the whole
22 h recording would need ~152 GB of reads. Three scores are computed (see ripple.channel_scores);
channel_selection.method picks which one selects. Writes to {out_base}/{seg}/{probe}/ripples/
channel_selection/, a sibling of the sleep-classification variant dirs rather than a child of one.

The sweep varies the number of windows and the seed to answer 'how much data does a stable choice
need'. Draws are NESTED -- the n=10 sample is a prefix of n=50 and so on -- so differences across n
are more data, never different data. Consistency is reported as the spread in um of the selected
depth across seeds: two contacts 15 um apart are the same anatomical choice, so an exact-match rate
would understate agreement.

Requires run_scoring.py to have run for this seg/probe: the NREM mask comes from its result.npz.
"""

import os
import argparse
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from probeinterface import read_probeinterface

from sleep_sandbox.io import load_preprocessed, find_amplifier_files
from sleep_sandbox.ripple import compute_psd, shank_index, channel_scores, select_channels
from plot_ripple_channel import METHODS, make_figures

repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")

parser = argparse.ArgumentParser()
parser.add_argument("--seg", required=True, help="preprocessed segment range, e.g. 'seg5-148'")
parser.add_argument("--probe", default="ProbeB", choices=["ProbeA", "ProbeB"])
parser.add_argument("--out-base", type=Path, default=repo_root / "data" / "derivatives")
args = parser.parse_args()

with open(repo_root / "config/ripple.yml") as f:
    cfg = yaml.safe_load(f)
variant = cfg["recording"]["variant"]
cs = cfg["channel_selection"]
passband, delta_band = cfg["band"]["passband"], cs["delta_band"]

sweep_n = sorted(cs["sweep"]["n_windows"])
seeds = cs["sweep"]["seeds"]
assert cs["n_windows"] in sweep_n and cs["seed"] in seeds, \
    "production (n_windows, seed) must appear in the sweep so it is not recomputed"

deriv_dir = Path(os.environ["PREPRO_OUTPUT_DIR"]) / f"{args.probe}_{args.seg}"
states_path = args.out_base / args.seg / args.probe / variant / "result.npz"
out_dir = args.out_base / args.seg / args.probe / "ripples" / "channel_selection"
out_dir.mkdir(parents=True, exist_ok=True)
print(f"=== {args.probe} {variant} {args.seg} ===", flush=True)

rec = load_preprocessed(deriv_dir, variant)
_, probe_config_path = find_amplifier_files(Path(os.environ["PREPRO_RAW_DIR"]).parent, args.probe)
probe_obj = read_probeinterface(probe_config_path).probes[0]
rec = rec.set_probe(probe_obj.get_slice(probe_obj.device_channel_indices != -1))
fs = rec.get_sampling_frequency()
n_samples = rec.get_num_frames()
locs = rec.get_channel_locations()
shank = shank_index(locs)
shanks = np.unique(shank)
print(f"{rec.get_num_channels()} ch on {len(shanks)} shanks, {n_samples / fs / 3600:.2f} h", flush=True)

result = np.load(states_path)
win = int(cs["window_s"] * fs)
nrem_t = result["times"][result["nrem"]]
nrem_t = nrem_t[nrem_t * fs + win < n_samples]
print(f"NREM windows available: {nrem_t.size} ({result['nrem'].mean():.1%} of epochs)", flush=True)
assert nrem_t.size >= max(sweep_n), f"only {nrem_t.size} NREM windows, need {max(sweep_n)}"

# --- sweep: accumulate each seed's PSD once, snapshotting at each sweep size ---
snapshots = {}   # (seed, n) -> mean psd
for seed in seeds:
    picks = np.random.default_rng(seed).choice(nrem_t, size=max(sweep_n), replace=False)
    psd_sum = None
    for i, t0 in enumerate(picks, start=1):
        s0 = int(t0 * fs)
        traces = rec.get_traces(start_frame=s0, end_frame=s0 + win).astype(np.float32)
        freqs, psd = compute_psd(traces, fs, cs["psd_nperseg_s"])
        psd_sum = psd if psd_sum is None else psd_sum + psd
        if i in sweep_n:
            snapshots[(seed, i)] = psd_sum / i
    print(f"  seed {seed}: {max(sweep_n)} windows read", flush=True)

records = {k: [] for k in ("seed", "n_windows", "method", "shank", "channel", "depth_um")}
for (seed, n), mean_psd in snapshots.items():
    sc = channel_scores(freqs, mean_psd, locs, passband, delta_band, cs["smooth_um"])
    for method in METHODS:
        for s, ch in select_channels(sc, locs, method).items():
            for key, val in zip(records, (seed, n, method, s, ch, locs[ch, 1])):
                records[key].append(val)
sweep_rec = {k: np.array(v) for k, v in records.items()}
np.savez(out_dir / "window_sweep.npz", **sweep_rec)

# --- production selection ---
prod_psd = snapshots[(cs["seed"], cs["n_windows"])]
scores = channel_scores(freqs, prod_psd, locs, passband, delta_band, cs["smooth_um"])
selected = select_channels(scores, locs, cs["method"])
np.savez(out_dir / "profiles.npz", freqs=freqs, psd=prod_psd, locations=locs, shank=shank,
         **{k: v for k, v in scores.items()})

print(f"\n--- selection ({cs['method']}, n={cs['n_windows']}, seed={cs['seed']}) ---", flush=True)
print(f"{'shank':>5} {'raw':>6} {'smoothed':>9} {'delta_ratio':>12}   depth(um) of each pick")
agreement = {}
for s in shanks:
    picks = {m: select_channels(scores, locs, m)[int(s)] for m in METHODS}
    agreement[int(s)] = picks
    print(f"{s:>5} {picks['raw']:>6} {picks['smoothed']:>9} {picks['delta_ratio']:>12}   "
          + " ".join(f"{locs[picks[m], 1]:.0f}" for m in METHODS))
for s, ch in selected.items():
    print(f"  shank {s}: ch {ch} at {locs[ch, 1]:.0f} um, spikiness {scores['spikiness'][ch]:.2f}"
          + ("  <-- HIGH, likely a bad contact" if scores["spikiness"][ch] > 2 else ""), flush=True)

# --- consistency: spread in um of the selected depth across seeds ---
print("\n--- selection consistency across seeds (depth range, um) ---", flush=True)
consistency = {}
for method in METHODS:
    consistency[method] = {}
    for s in shanks:
        row = []
        for n in sweep_n:
            m = ((sweep_rec["method"] == method) & (sweep_rec["shank"] == s)
                 & (sweep_rec["n_windows"] == n))
            d = sweep_rec["depth_um"][m]
            vals, counts = np.unique(d, return_counts=True)
            row.append({"n_windows": int(n), "depth_range_um": float(d.max() - d.min()),
                        "n_unique_channels": int(np.unique(sweep_rec["channel"][m]).size),
                        "modal_depth_um": float(vals[counts.argmax()])})
        consistency[method][int(s)] = row
        print(f"  {method:>11} shank {s}: "
              + "  ".join(f"n={r['n_windows']}:{r['depth_range_um']:.0f}um"
                          f"/{r['n_unique_channels']}ch" for r in row), flush=True)

with open(out_dir / "channel_selection.yml", "w") as f:
    yaml.safe_dump({
        "ripple_config": cfg,
        "sources": {"lfp": str(deriv_dir), "variant": variant, "states": str(states_path)},
        "method": cs["method"],
        "selected": {int(s): {"channel": int(ch), "depth_um": float(locs[ch, 1]),
                              "spikiness": float(scores["spikiness"][ch])}
                     for s, ch in selected.items()},
        "method_agreement": {s: {m: int(c) for m, c in p.items()} for s, p in agreement.items()},
        "consistency_depth_range_um": consistency,
    }, f, sort_keys=False)

# --- figures ---
# Plotting lives in plot_ripple_channel.py so the figures can be regenerated from the arrays above
# without repeating the window reads, with one copy of the code either way.
make_figures(out_dir, f"{args.probe} {args.seg}")

print(f"\nsaved -> {out_dir}", flush=True)
