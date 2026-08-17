"""Detect ripples on the NREM epochs of a preprocessed LFP derivative and write an event table.

Usage: python run_ripples.py --seg seg5-148 [--probe ProbeB] [--variant lfp_cmr]
                             [--channel N] [--out-base data/derivatives]

Requires run_scoring.py to have already run: the NREM mask and the baseline pool come from
{out_base}/{seg}/{probe}/{variant}/result.npz, so --seg/--probe/--variant must name a combo that
has been scored. Detection is always restricted to NREM; baseline.pool in config/ripple.yml only
controls which samples the z-score mean/SD is estimated on.

EVERY candidate event is written with its corroborating-neighbour count, so sweeping
neighbours.n_required is a filter on the saved table rather than a rerun.
"""

import os
import argparse
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from probeinterface import read_probeinterface

from sleep_sandbox.io import load_preprocessed, find_amplifier_files
from sleep_sandbox.ripple import (compute_psd, band_power, shank_index, neighbour_channels,
                                  bandpass_envelope, zscore_envelope, epoch_mask_to_samples,
                                  detect_events, count_corroborating)

repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")

parser = argparse.ArgumentParser()
parser.add_argument("--seg", required=True,
                    help="preprocessed segment range, e.g. 'seg5-148'; selects both the "
                         "$PREPRO_OUTPUT_DIR/{probe}_{seg} input and the {out_base}/{seg} output")
parser.add_argument("--probe", default="ProbeB", choices=["ProbeA", "ProbeB"])
parser.add_argument("--variant", choices=["lfp_cmr", "lfp_nocmr"],
                    help="default: recording.variant from config/ripple.yml")
parser.add_argument("--channel", type=int,
                    help="detection channel index; default is the highest ripple-band power channel")
parser.add_argument("--out-base", type=Path, default=repo_root / "data" / "derivatives")
args = parser.parse_args()

with open(repo_root / "config/ripple.yml") as f:
    cfg = yaml.safe_load(f)

variant = args.variant or cfg["recording"]["variant"]
passband = cfg["band"]["passband"]
order = cfg["band"]["order"]
det, nbc, cs = cfg["detection"], cfg["neighbours"], cfg["channel_selection"]

deriv_dir = Path(os.environ["PREPRO_OUTPUT_DIR"]) / f"{args.probe}_{args.seg}"
out_dir = args.out_base / args.seg / args.probe / variant
print(f"=== {args.probe} {variant} {args.seg} ===", flush=True)

rec = load_preprocessed(deriv_dir, variant)
# Derivatives are saved without a probe attached, but neighbour_channels needs channel locations.
# Same discovery run_scoring.py uses, so the geometry matches the one the derivatives were built
# with rather than being pinned separately in .env and free to drift.
_, probe_config_path = find_amplifier_files(Path(os.environ["PREPRO_RAW_DIR"]).parent, args.probe)
probe_obj = read_probeinterface(probe_config_path).probes[0]
rec = rec.set_probe(probe_obj.get_slice(probe_obj.device_channel_indices != -1))
fs = rec.get_sampling_frequency()
n_samples = rec.get_num_frames()
locs = rec.get_channel_locations()
print(f"{rec.get_num_channels()} ch, {n_samples / fs / 3600:.2f} h at {fs:.0f} Hz", flush=True)

# --- state masks from the scoring run ---
result = np.load(out_dir / "result.npz")
pool = cfg["baseline"]["pool"]
pool_epochs = np.ones_like(result["nrem"]) if pool == "all" else result[pool]
nrem_mask = epoch_mask_to_samples(result["times"], result["nrem"], n_samples, fs)
baseline_mask = (nrem_mask if pool == "nrem"
                 else epoch_mask_to_samples(result["times"], pool_epochs, n_samples, fs))
nrem_s = nrem_mask.sum() / fs
print(f"NREM {nrem_s / 3600:.2f} h ({nrem_mask.mean():.1%}); z-score baseline pool '{pool}' "
      f"({baseline_mask.mean():.1%} of samples)", flush=True)

# --- detection channel: band power averaged over a sample of NREM windows ---
if args.channel is None:
    rng = np.random.default_rng(cs["seed"])
    win = int(cs["window_s"] * fs)
    nrem_t = result["times"][result["nrem"]]
    picks = rng.choice(nrem_t[nrem_t * fs + win < n_samples],
                       size=min(cs["n_windows"], nrem_t.size), replace=False)
    psd_sum = None
    for t0 in picks:
        s0 = int(t0 * fs)
        traces = rec.get_traces(start_frame=s0, end_frame=s0 + win).astype(np.float32)
        freqs, psd = compute_psd(traces, fs, cs["psd_nperseg_s"])
        psd_sum = psd if psd_sum is None else psd_sum + psd
    power = band_power(freqs, psd_sum / len(picks), passband)
    channel = int(np.argmax(power))
    shank = shank_index(locs)
    print(f"band power over {len(picks)} NREM windows -- peak channel per shank:", flush=True)
    for s in np.unique(shank):
        sel = np.flatnonzero(shank == s)
        best = sel[np.argmax(power[sel])]
        print(f"  shank {s}: ch {best} (depth {locs[best, 1]:.0f} um) power {power[best]:.4g}")
else:
    channel = args.channel
    power = None
neighbours = neighbour_channels(locs, channel)
print(f"detection channel {channel} at {tuple(locs[channel])}; neighbours "
      f"{ {k: v for k, v in neighbours.items()} }", flush=True)

# --- envelopes: one channel at a time, each is ~1e8 samples ---
def envelope_z(ch):
    trace = rec.get_traces(channel_ids=[rec.channel_ids[ch]])[:, 0].astype(np.float64)
    _, env = bandpass_envelope(trace, fs, passband, order, det["envelope_smooth_s"])
    return zscore_envelope(env, baseline_mask)


print("detecting candidates...", flush=True)
z = envelope_z(channel)
events = detect_events(z, fs, det["boundary_sd"], det["peak_sd"], det["min_duration_s"],
                       det["max_duration_s"], restrict=nrem_mask,
                       min_inter_event_s=det["min_inter_event_s"])
del z
print(f"  {events['start'].size} candidates", flush=True)

# Neighbours are detected unrestricted: the NREM test belongs to the candidate, and a neighbour's
# own peak can fall just the other side of a (several-second-fuzzy) state boundary.
neighbour_events = []
for name, ch in neighbours.items():
    zn = envelope_z(ch)
    ev = detect_events(zn, fs, nbc["boundary_sd"], nbc["peak_sd"], nbc["min_duration_s"],
                       nbc["max_duration_s"])
    del zn
    neighbour_events.append(ev)
    print(f"  neighbour {name} (ch {ch}): {ev['start'].size} events", flush=True)

counts = count_corroborating(events, neighbour_events, n_samples)
events["n_corroborating"] = counts
for key in ("start", "end", "peak"):
    events[f"{key}_s"] = events[key] / fs

# --- summary ---
sweep = {k: int((counts >= k).sum()) for k in range(len(neighbours) + 1)}
keep = counts >= nbc["n_required"]
kept = {k: v[keep] for k, v in events.items()}
iei = np.diff(kept["peak_s"])
summary = {
    "n_candidates": int(events["start"].size),
    "n_required": nbc["n_required"],
    "n_kept": int(keep.sum()),
    "nrem_hours": round(float(nrem_s / 3600), 3),
    "rate_per_min_nrem": round(float(keep.sum() / (nrem_s / 60)), 4) if nrem_s else None,
    "duration_ms_median": round(float(np.median(kept["duration_s"]) * 1000), 1) if keep.any() else None,
    "duration_ms_iqr": [round(float(q * 1000), 1) for q in np.percentile(kept["duration_s"], [25, 75])]
                       if keep.any() else None,
    "peak_z_median": round(float(np.median(kept["peak_z"])), 3) if keep.any() else None,
    "iei_s_median": round(float(np.median(iei)), 3) if iei.size else None,
    "n_corroborating_sweep": sweep,
}
print(yaml.safe_dump(summary, sort_keys=False), flush=True)

np.savez(out_dir / "ripples.npz", **events)
with open(out_dir / "ripples.yml", "w") as f:
    yaml.safe_dump({
        "ripple_config": cfg,
        "sources": {"lfp": str(deriv_dir), "variant": variant, "states": str(out_dir / "result.npz")},
        "channel": channel,
        "channel_depth_um": float(locs[channel, 1]),
        "neighbours": {k: int(v) for k, v in neighbours.items()},
        "summary": summary,
    }, f, sort_keys=False)
print(f"saved -> {out_dir / 'ripples.npz'} (all candidates, filter on n_corroborating)")
