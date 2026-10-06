"""Pick the ripple detection channel on each shank of a chunked zarr recording, once across all chunks.

Usage: python select_ripple_channel_chunked.py --lfp-dir .../lfp_pp_pilot04 --reference cmr
                                               --csv .../abcEphysPilot04_B_ephys_paths.csv
                                               [--experiment abcEphysPilot04] [--probe ProbeB] [--shanks 0 1 2]

The chunked counterpart of select_ripple_channel.py: same three scores, same nested window sweep,
same figures (plot_ripple_channel.make_figures). NREM windows are drawn from every chunk's scoring
result, so one channel per shank serves the whole recording, as in chunked sleep scoring.

--reference selects the arm: 'none' reads the zarrs as stored, 'cmr' median-references each shank
against itself on read (the upstream abcEphys01 CMR arm is per shank too). Referencing can move the
depth profile, so each arm gets its own selection under ripples/{reference}/channel_selection/.

Requires run_scoring_chunked.py to have run for every chunk: the NREM bins come from its
scoring/{chunk}/result.npz (times_abs, nrem; nrem already excludes nodata).
"""

import json
import argparse
from pathlib import Path

import numpy as np
import yaml
import spikeinterface.preprocessing as spre

from sleep_sandbox.io import load_input_recording
from sleep_sandbox.ripple import compute_psd, shank_index, channel_scores, select_channels
from plot_ripple_channel import METHODS, make_figures

repo_root = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--lfp-dir", type=Path, required=True, help="dir of per-shank LFP zarrs")
parser.add_argument("--reference", required=True, choices=["none", "cmr"])
parser.add_argument("--csv", type=Path, required=True, help="ephys-paths CSV (recording start)")
parser.add_argument("--experiment", default="abcEphysPilot04")
parser.add_argument("--probe", default="ProbeB")
parser.add_argument("--shanks", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--scoring", default="scoring", help="scoring dir name whose NREM masks are used")
parser.add_argument("--out-base", type=Path, default=None)
args = parser.parse_args()

with open(repo_root / "config/ripple.yml") as f:
    cfg = yaml.safe_load(f)
cs = cfg["channel_selection"]
passband, delta_band = cfg["band"]["passband"], cs["delta_band"]
sweep_n = sorted(cs["sweep"]["n_windows"])
seeds = cs["sweep"]["seeds"]
assert cs["n_windows"] in sweep_n and cs["seed"] in seeds, \
    "production (n_windows, seed) must appear in the sweep so it is not recomputed"

shank_tag = f"{args.probe}_shanks{''.join(map(str, args.shanks))}"
base = (args.out_base or repo_root / "data" / "derivatives" / args.experiment) / args.lfp_dir.name / shank_tag
scoring_dir = base / args.scoring
out_dir = base / "ripples" / args.reference / "channel_selection"
out_dir.mkdir(parents=True, exist_ok=True)


def zarr_path(shank):
    hits = sorted(args.lfp_dir.glob(f"*_{args.probe}_shank{shank}.zarr")) + \
           sorted(args.lfp_dir.glob(f"*_{args.probe}_shank_{shank}.zarr"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{len(hits)} zarrs in {args.lfp_dir} match {args.probe} shank {shank}")
    return hits[0]


zarr_paths = [zarr_path(s) for s in args.shanks]
rec = load_input_recording(zarr_paths)
if args.reference == "cmr":
    group = rec.get_property("group")
    rec = spre.common_reference(rec, reference="global", operator="median",
                                groups=[list(rec.channel_ids[group == g]) for g in np.unique(group)])
# SI 0.103.2 does not restore the probe from these files; contact i of each file is its channel i.
locs = np.concatenate([np.array(json.loads((p / ".zattrs").read_text())
                                ["probegroup"]["probes"][0]["contact_positions"]) for p in zarr_paths])
assert len(locs) == rec.get_num_channels()
fs = rec.get_sampling_frequency()
n_samples = rec.get_num_frames()
shank = shank_index(locs)
shanks = np.unique(shank)
print(f"=== {args.probe} {args.lfp_dir.name} reference={args.reference} ===", flush=True)
print(f"{rec.get_num_channels()} ch on {len(shanks)} shanks, {n_samples / fs / 3600:.2f} h", flush=True)

# --- NREM bins from every chunk, on absolute recording time ---
results = sorted(scoring_dir.glob("chunk*/result.npz"))
assert results, f"no scoring results under {scoring_dir}"
nrem_t = np.concatenate([np.load(r)["times_abs"][np.load(r)["nrem"]] for r in results])
win = int(cs["window_s"] * fs)
nrem_t = nrem_t[nrem_t * fs + win < n_samples]
print(f"NREM windows available: {nrem_t.size} across {len(results)} chunks", flush=True)
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
        "sources": {"lfp": [str(p) for p in zarr_paths], "reference": args.reference,
                    "scoring": str(scoring_dir), "chunks": [r.parent.name for r in results]},
        "method": cs["method"],
        "selected": {int(s): {"channel": int(ch), "depth_um": float(locs[ch, 1]),
                              "spikiness": float(scores["spikiness"][ch])}
                     for s, ch in selected.items()},
        "method_agreement": {s: {m: int(c) for m, c in p.items()} for s, p in agreement.items()},
        "consistency_depth_range_um": consistency,
    }, f, sort_keys=False)

make_figures(out_dir, f"{args.probe} {args.experiment} reference={args.reference}")
print(f"\nsaved -> {out_dir}", flush=True)
