"""Detect ripples on the NREM epochs of a preprocessed LFP derivative and write an event table.

Usage: python run_ripples.py --seg seg5-148 [--probe ProbeB] [--out-base data/derivatives]

Requires two earlier runs for this seg/probe: run_scoring.py (the NREM mask and the z-score
baseline pool come from its result.npz) and select_ripple_channel.py (the per-shank detection
channels come from its channel_selection.yml). Detection is always restricted to NREM;
baseline.pool in config/ripple.yml only controls which samples the mean/SD is estimated on.

Detection runs on every shank, each with its own three neighbours, into one flat event table with
`shank` and `channel` columns. Shanks off the CA1 pyramidal layer are NOT skipped -- ProbeA is PFC
and is the negative control, so what an unconverged selection detects is the point. Use the
`spikiness` recorded per shank in events.yml to filter afterwards.

Output goes to {seg}/{probe}/ripples/events/{run_id}/, where run_id is an 8-hex digest of the
parameters that determine the event set (see `run_params` below), so a parameter sweep accumulates
side-by-side directories and a rerun of the same parameters overwrites in place. events/runs.yml
indexes them. neighbours.n_required is deliberately NOT hashed: every candidate is written with its
corroborating-neighbour count, so sweeping n_required is a filter on the saved table rather than a
rerun, and hashing it would split byte-identical outputs across directories.
"""

import os
import time
import argparse
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from probeinterface import read_probeinterface

from sleep_sandbox.io import load_preprocessed, find_amplifier_files
from sleep_sandbox.ripple import (neighbour_channels, bandpass_envelope, zscore_envelope,
                                  epoch_mask_to_samples, detect_events, count_corroborating,
                                  event_spectral_stats)

repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")

parser = argparse.ArgumentParser()
parser.add_argument("--seg", required=True,
                    help="preprocessed segment range, e.g. 'seg5-148'; selects both the "
                         "$PREPRO_OUTPUT_DIR/{probe}_{seg} input and the {out_base}/{seg} output")
parser.add_argument("--probe", default="ProbeB", choices=["ProbeA", "ProbeB"])
parser.add_argument("--out-base", type=Path, default=repo_root / "data" / "derivatives")
args = parser.parse_args()

with open(repo_root / "config/ripple.yml") as f:
    cfg = yaml.safe_load(f)

variant = cfg["recording"]["variant"]
passband, order = cfg["band"]["passband"], cfg["band"]["order"]
det, nbc, cs, spec = cfg["detection"], cfg["neighbours"], cfg["channel_selection"], cfg["spectrum"]
pool = cfg["baseline"]["pool"]

deriv_dir = Path(os.environ["PREPRO_OUTPUT_DIR"]) / f"{args.probe}_{args.seg}"
states_path = args.out_base / args.seg / args.probe / variant / "result.npz"
sel_path = args.out_base / args.seg / args.probe / "ripples" / "channel_selection" / "channel_selection.yml"
print(f"=== {args.probe} {variant} {args.seg} ===", flush=True)

with open(sel_path) as f:
    sel = yaml.safe_load(f)
# A selection built against a different recording would silently put the channels of one run in the
# table of another.
assert sel["sources"]["states"] == str(states_path), \
    f"channel_selection.yml was built against {sel['sources']['states']}, not {states_path}"
selected = {int(s): int(d["channel"]) for s, d in sel["selected"].items()}
spikiness = {int(s): float(d["spikiness"]) for s, d in sel["selected"].items()}
print(f"channels from {sel['method']} selection: "
      + ", ".join(f"shank {s}: ch {c}" for s, c in sorted(selected.items())), flush=True)

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
result = np.load(states_path)
pool_epochs = np.ones_like(result["nrem"]) if pool == "all" else result[pool]
nrem_mask = epoch_mask_to_samples(result["times"], result["nrem"], n_samples, fs)
baseline_mask = (nrem_mask if pool == "nrem"
                 else epoch_mask_to_samples(result["times"], pool_epochs, n_samples, fs))
nrem_s = nrem_mask.sum() / fs
print(f"NREM {nrem_s / 3600:.2f} h ({nrem_mask.mean():.1%}); z-score baseline pool '{pool}' "
      f"({baseline_mask.mean():.1%} of samples)", flush=True)

# --- run identity ---
run_params = {
    "seg": args.seg,
    "probe": args.probe,
    "variant": variant,
    "band": cfg["band"],
    "baseline_pool": pool,
    "detection": det,
    # n_required excluded: a post-hoc filter on n_corroborating, not a property of the run
    "neighbours": {k: v for k, v in nbc.items() if k != "n_required"},
    "channel_selection": {k: cs[k] for k in ("method", "n_windows", "seed", "smooth_um")},
    # the channels themselves, not just the recipe: catches a selection rerun that changed the
    # answer without config/ripple.yml changing
    "channels": {int(s): int(c) for s, c in selected.items()},
}
run_id = hashlib.sha256(yaml.safe_dump(run_params, sort_keys=True).encode()).hexdigest()[:8]
events_root = args.out_base / args.seg / args.probe / "ripples" / "events"
out_dir = events_root / run_id
if out_dir.exists():
    print(f"run {run_id} exists -- same parameters, replacing its contents", flush=True)
out_dir.mkdir(parents=True, exist_ok=True)
print(f"run_id {run_id} -> {out_dir}", flush=True)


# --- traces: every channel we will need, read in one chunked pass ---
# The derivative is sample-interleaved, so pulling a single channel over the whole recording still
# touches every page of it -- reading per channel would be one full pass over the 83G file each
# time. So read every channel we need together, but do it in time chunks rather than one call:
# one call fancy-indexes a memmap spanning the whole file, and under a SLURM memory cgroup the
# page cache that pulls in is charged to the job, so it hits the cap and thrashes instead of
# finishing (measured: 3 h without completing the read, MaxRSS pinned at the 24G limit). Chunked,
# each slab is contiguous, its page cache is reclaimable, and a full pass takes ~2.3 min.
shank_nb = {s: neighbour_channels(locs, selected[s]) for s in sorted(selected)}
read_ch = sorted({c for s in shank_nb for c in (selected[s], *shank_nb[s].values())})
col = {c: i for i, c in enumerate(read_ch)}
chunk = int(300 * fs)
traces = np.empty((n_samples, len(read_ch)), dtype=rec.get_dtype())
print(f"reading {len(read_ch)} channels in {-(-n_samples // chunk)} chunks "
      f"({traces.nbytes / 1e9:.2f} GB held as {traces.dtype})...", flush=True)
t_read = time.time()
ids = [rec.channel_ids[c] for c in read_ch]
for i, s0 in enumerate(range(0, n_samples, chunk)):
    s1 = min(s0 + chunk, n_samples)
    traces[s0:s1] = rec.get_traces(start_frame=s0, end_frame=s1, channel_ids=ids)
    if i % 50 == 0:
        print(f"  {s1 / n_samples:.0%} ({time.time() - t_read:.0f}s)", flush=True)
print(f"  read in {time.time() - t_read:.0f}s", flush=True)


def envelope_z(ch):
    _, env = bandpass_envelope(traces[:, col[ch]].astype(np.float64), fs, passband, order,
                               det["envelope_smooth_s"])
    return zscore_envelope(env, baseline_mask)


def stats(ev, iei=True):
    n = ev["start"].size
    out = {"n": int(n),
           "rate_per_min_nrem": round(float(n / (nrem_s / 60)), 4) if nrem_s else None}
    if n:
        out["duration_ms_median"] = round(float(np.median(ev["duration_s"]) * 1000), 1)
        out["duration_ms_iqr"] = [round(float(q * 1000), 1)
                                  for q in np.percentile(ev["duration_s"], [25, 75])]
        out["peak_z_median"] = round(float(np.median(ev["peak_z"])), 3)
        out["peak_hz_median"] = round(float(np.median(ev["peak_hz"])), 1)
        out["frac_peak_hz_in_band"] = round(float(np.mean((ev["peak_hz"] >= passband[0])
                                                          & (ev["peak_hz"] <= passband[1]))), 3)
        out["prominence_median"] = round(float(np.median(ev["prominence"])), 2)
        out["frac_prominence_gt2"] = round(float(np.mean(ev["prominence"] > 2)), 3)
        if iei and n > 1:
            out["iei_s_median"] = round(float(np.median(np.diff(ev["peak"]) / fs)), 3)
    return out


tables, per_shank, kept_by_shank = [], {}, {}
for s in sorted(selected):
    ch = selected[s]
    nb = shank_nb[s]
    print(f"\n--- shank {s}: ch {ch} at {locs[ch, 1]:.0f} um, neighbours "
          f"{ {k: v for k, v in nb.items()} } ---", flush=True)

    z = envelope_z(ch)
    ev = detect_events(z, fs, det["boundary_sd"], det["peak_sd"], det["min_duration_s"],
                       det["max_duration_s"], restrict=nrem_mask,
                       min_inter_event_s=det["min_inter_event_s"])
    del z
    print(f"  {ev['start'].size} candidates", flush=True)

    # Neighbours are detected unrestricted: the NREM test belongs to the candidate, and a
    # neighbour's own peak can fall just the other side of a (several-second-fuzzy) state boundary.
    neighbour_events = []
    for name, nch in nb.items():
        zn = envelope_z(nch)
        nev = detect_events(zn, fs, nbc["boundary_sd"], nbc["peak_sd"], nbc["min_duration_s"],
                            nbc["max_duration_s"])
        del zn
        neighbour_events.append(nev)
        print(f"  neighbour {name} (ch {nch}): {nev['start'].size} events", flush=True)

    ev["n_corroborating"] = count_corroborating(ev, neighbour_events, n_samples)
    # Wideband, not the bandpassed trace -- see event_peak_frequency. An event whose argmax sits at
    # the search-band floor has no spectral peak at all, which is the transient signature.
    ev.update(event_spectral_stats(traces[:, col[ch]].astype(np.float64), ev["peak"], fs,
                                   spec["window_s"], spec["nfft"], spec["search_band"],
                                   spec["background_band"]))
    n = ev["start"].size
    ev["shank"] = np.full(n, s, dtype=int)
    ev["channel"] = np.full(n, ch, dtype=int)

    sweep = {k: int((ev["n_corroborating"] >= k).sum()) for k in range(len(nb) + 1)}
    keep = ev["n_corroborating"] >= nbc["n_required"]
    kept_by_shank[s] = {k: v[keep] for k, v in ev.items()}
    per_shank[s] = {
        "channel": ch,
        "depth_um": float(locs[ch, 1]),
        "spikiness": spikiness[s],
        "neighbours": {k: int(v) for k, v in nb.items()},
        "n_candidates": int(n),
        "n_corroborating_sweep": sweep,
        "kept": stats(kept_by_shank[s]),
    }
    print(f"  sweep {sweep}; kept {int(keep.sum())} at n_required={nbc['n_required']}", flush=True)
    tables.append(ev)

events = {k: np.concatenate([t[k] for t in tables]) for k in tables[0]}
for key in ("start", "end", "peak"):
    events[f"{key}_s"] = events[key] / fs

# --- cross-shank coincidence of kept events ---
# Asymmetric by construction: rows are the fraction of THAT shank's events with an overlapping
# event on the column's shank, and the two shanks have different event counts. A ripple field is
# local to a shank, so high off-diagonal values are the signature of something volume-conducted or
# shared by reference rather than of a common ripple.
coincidence = {}
for a in sorted(kept_by_shank):
    row = {}
    for b in sorted(kept_by_shank):
        if a == b or kept_by_shank[a]["start"].size == 0:
            continue
        frac = count_corroborating(kept_by_shank[a], [kept_by_shank[b]], n_samples).mean()
        row[int(b)] = round(float(frac), 4)
    coincidence[int(a)] = row
print("\n--- cross-shank coincidence (fraction of row's events seen on column) ---", flush=True)
for a, row in coincidence.items():
    print(f"  shank {a}: " + "  ".join(f"{b}:{v:.3f}" for b, v in row.items()), flush=True)

# IEI is omitted from the combined stats: the table is shanks concatenated, so a difference of
# consecutive peaks crosses shank boundaries and means nothing.
kept_all = {k: v[events["n_corroborating"] >= nbc["n_required"]] for k, v in events.items()}
summary = {
    "n_candidates": int(events["start"].size),
    "n_required": nbc["n_required"],
    "nrem_hours": round(float(nrem_s / 3600), 3),
    "kept": stats(kept_all, iei=False),
}
print("\n" + yaml.safe_dump(summary, sort_keys=False), flush=True)

np.savez(out_dir / "events.npz", **events)
with open(out_dir / "events.yml", "w") as f:
    yaml.safe_dump({
        "run_id": run_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "run_params": run_params,
        "ripple_config": cfg,
        "sources": {"lfp": str(deriv_dir), "states": str(states_path),
                    "channel_selection": str(sel_path)},
        "per_shank": per_shank,
        "cross_shank_coincidence": coincidence,
        "summary": summary,
    }, f, sort_keys=False)

runs_path = events_root / "runs.yml"
runs = yaml.safe_load(runs_path.read_text()) if runs_path.exists() else {}
runs[run_id] = {
    "created": datetime.now().isoformat(timespec="seconds"),
    "run_params": run_params,
    "n_candidates": summary["n_candidates"],
    "n_required": nbc["n_required"],
    "n_kept": summary["kept"]["n"],
    "rate_per_min_nrem": {s: v["kept"]["rate_per_min_nrem"] for s, v in per_shank.items()},
}
with open(runs_path, "w") as f:
    yaml.safe_dump(runs, f, sort_keys=False)
print(f"saved -> {out_dir / 'events.npz'} (all candidates, filter on n_corroborating)")
print(f"indexed -> {runs_path} ({len(runs)} runs)")
