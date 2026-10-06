"""Detect ripples on the NREM epochs of one midday chunk of a chunked zarr recording.

Usage: python run_ripples_chunked.py --chunk N --lfp-dir .../lfp_pp_pilot04 --reference cmr
                                     --csv .../abcEphysPilot04_B_ephys_paths.csv
                                     [--merge-s 0 0.03] [--pad-s 10] [--experiment abcEphysPilot04]

The chunked counterpart of run_ripples.py, with the same detection, neighbour corroboration,
spectral annotation, cross-shank coincidence and summary. Differences:
- Input is the per-shank 1250 Hz zarrs. --reference none reads them as stored; cmr median-references
  each shank against itself on read.
- Channels come from select_ripple_channel_chunked.py for the same reference arm (one selection for
  the whole recording). The NREM mask and the z-score baseline come from this chunk's scoring
  result, so the baseline is re-estimated per chunk.
- --pad-s of context is filtered either side of the chunk so filter edges fall outside it; a
  candidate's peak must lie inside the chunk and inside NREM.
- Every --merge-s value is detected from the same envelopes in one pass, each as its own run: the
  merge window is part of the run_id hash, the chunk is not, so all chunks of one configuration
  share events/{run_id}/ and each writes its own {chunk label}/ below it.

Output: {base}/ripples/{reference}/events/{run_id}/{chunk label}/events.{npz,yml}, plus
events/{run_id}/run.yml (the run's parameters; identical from every chunk, written atomically
because the array tasks run concurrently). start/end/peak are absolute recording frames and
*_s absolute recording seconds, the same clock as the scoring result's times_abs.
"""

import os
import json
import time
import argparse
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
import spikeinterface.preprocessing as spre

from sleep_sandbox.io import load_session_timeline, midday_chunks, load_input_recording
from sleep_sandbox.ripple import (neighbour_channels, bandpass_envelope, zscore_envelope,
                                  epoch_mask_to_samples, detect_events, count_corroborating,
                                  event_spectral_stats)

repo_root = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--chunk", type=int, required=True, help="midday_chunks index")
parser.add_argument("--lfp-dir", type=Path, required=True, help="dir of per-shank LFP zarrs")
parser.add_argument("--reference", required=True, choices=["none", "cmr"])
parser.add_argument("--csv", type=Path, required=True, help="ephys-paths CSV (recording start)")
parser.add_argument("--experiment", default="abcEphysPilot04")
parser.add_argument("--probe", default="ProbeB")
parser.add_argument("--shanks", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--scoring", default="scoring", help="scoring dir name whose NREM masks are used")
parser.add_argument("--merge-s", type=float, nargs="+", default=None,
                    help="min_inter_event_s values to detect with (default: config/ripple.yml's)")
parser.add_argument("--pad-s", type=float, default=10.0, help="context filtered either side of the chunk")
parser.add_argument("--out-base", type=Path, default=None)
args = parser.parse_args()

with open(repo_root / "config/ripple.yml") as f:
    cfg = yaml.safe_load(f)
passband, order = cfg["band"]["passband"], cfg["band"]["order"]
det, nbc, cs, spec = cfg["detection"], cfg["neighbours"], cfg["channel_selection"], cfg["spectrum"]
pool = cfg["baseline"]["pool"]
merges = args.merge_s if args.merge_s is not None else [det["min_inter_event_s"]]

shank_tag = f"{args.probe}_shanks{''.join(map(str, args.shanks))}"
base = (args.out_base or repo_root / "data" / "derivatives" / args.experiment) / args.lfp_dir.name / shank_tag
scoring_dir = base / args.scoring
sel_path = base / "ripples" / args.reference / "channel_selection" / "channel_selection.yml"
events_root = base / "ripples" / args.reference / "events"

with open(sel_path) as f:
    sel = yaml.safe_load(f)
# A selection built on another arm or another scoring run would put one run's channels in another's table.
assert sel["sources"]["reference"] == args.reference and sel["sources"]["scoring"] == str(scoring_dir), \
    f"{sel_path} was built for reference={sel['sources']['reference']}, scoring={sel['sources']['scoring']}"
selected = {int(s): int(d["channel"]) for s, d in sel["selected"].items()}
spikiness = {int(s): float(d["spikiness"]) for s, d in sel["selected"].items()}


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
fs = rec.get_sampling_frequency()

rec_start, _ = load_session_timeline(args.csv)
c = midday_chunks(rec_start, rec.get_num_frames(), fs)[args.chunk]
pad = int(args.pad_s * fs)
a0, a1 = max(0, c["start_frame"] - pad), min(rec.get_num_frames(), c["end_frame"] + pad)
n_samples = a1 - a0
print(f"=== {args.probe} {args.lfp_dir.name} reference={args.reference} {c['label']} ===", flush=True)
print(f"{c['t_start']} -> {c['t_end']}, frames {a0}-{a1} ({n_samples / fs / 3600:.2f} h with padding); "
      "channels from selection: " + ", ".join(f"shank {s}: ch {ch}" for s, ch in sorted(selected.items())),
      flush=True)

# --- state masks from this chunk's scoring run, onto the padded slice's sample grid ---
# epoch_mask_to_samples extends the edge epochs into the padding by nearest-neighbour, so the padding
# is cleared explicitly: it belongs to the neighbouring chunks, whose own runs detect there.
states_path = scoring_dir / c["label"] / "result.npz"
result = np.load(states_path)
times = result["times_abs"] - a0 / fs
in_chunk = np.zeros(n_samples, dtype=bool)
in_chunk[c["start_frame"] - a0:c["end_frame"] - a0] = True
pool_epochs = ~result["nodata"] if pool == "all" else result[pool]
nrem_mask = epoch_mask_to_samples(times, result["nrem"], n_samples, fs) & in_chunk
baseline_mask = (nrem_mask if pool == "nrem"
                 else epoch_mask_to_samples(times, pool_epochs, n_samples, fs) & in_chunk)
nrem_s = nrem_mask.sum() / fs
print(f"NREM {nrem_s / 3600:.2f} h ({nrem_mask[in_chunk].mean():.1%} of the chunk); z-score baseline "
      f"pool '{pool}' ({baseline_mask[in_chunk].mean():.1%})", flush=True)

# --- traces: every channel we will need, read in time slabs (see run_ripples.py) ---
shank_nb = {s: neighbour_channels(locs, selected[s]) for s in sorted(selected)}
read_ch = sorted({ch for s in shank_nb for ch in (selected[s], *shank_nb[s].values())})
col = {ch: i for i, ch in enumerate(read_ch)}
ids = [rec.channel_ids[ch] for ch in read_ch]
slab = int(300 * fs)
traces = np.empty((n_samples, len(read_ch)), dtype=rec.get_dtype())
print(f"reading {len(read_ch)} channels in {-(-n_samples // slab)} slabs "
      f"({traces.nbytes / 1e9:.2f} GB held as {traces.dtype})...", flush=True)
t_read = time.time()
for i, s0 in enumerate(range(0, n_samples, slab)):
    s1 = min(s0 + slab, n_samples)
    traces[s0:s1] = rec.get_traces(start_frame=a0 + s0, end_frame=a0 + s1, channel_ids=ids)
    if i % 50 == 0:
        print(f"  {s1 / n_samples:.0%} ({time.time() - t_read:.0f}s)", flush=True)
print(f"  read in {time.time() - t_read:.0f}s", flush=True)


def envelope_z(ch):
    _, env = bandpass_envelope(traces[:, col[ch]].astype(np.float64), fs, passband, order,
                               det["envelope_smooth_s"])
    return zscore_envelope(env, baseline_mask).astype(np.float32)


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


# Per shank, the envelopes and the neighbours' events are computed once; only candidate detection
# (where the merge window acts) and what depends on it are repeated per merge value.
tables = {m: [] for m in merges}
per_shank = {m: {} for m in merges}
kept_by_shank = {m: {} for m in merges}
for s in sorted(selected):
    ch = selected[s]
    nb = shank_nb[s]
    print(f"\n--- shank {s}: ch {ch} at {locs[ch, 1]:.0f} um, neighbours "
          f"{ {k: v for k, v in nb.items()} } ---", flush=True)
    z = envelope_z(ch)

    # Neighbours are detected unrestricted, as in run_ripples.py, and without the merge window.
    neighbour_events = []
    for name, nch in nb.items():
        zn = envelope_z(nch)
        nev = detect_events(zn, fs, nbc["boundary_sd"], nbc["peak_sd"], nbc["min_duration_s"],
                            nbc["max_duration_s"])
        del zn
        neighbour_events.append(nev)
        print(f"  neighbour {name} (ch {nch}): {nev['start'].size} events", flush=True)

    wide = traces[:, col[ch]].astype(np.float64)
    for m in merges:
        ev = detect_events(z, fs, det["boundary_sd"], det["peak_sd"], det["min_duration_s"],
                           det["max_duration_s"], restrict=nrem_mask, min_inter_event_s=m)
        ev["n_corroborating"] = count_corroborating(ev, neighbour_events, n_samples)
        ev.update(event_spectral_stats(wide, ev["peak"], fs, spec["window_s"], spec["nfft"],
                                       spec["search_band"], spec["background_band"]))
        n = ev["start"].size
        ev["shank"] = np.full(n, s, dtype=int)
        ev["channel"] = np.full(n, ch, dtype=int)
        sweep = {k: int((ev["n_corroborating"] >= k).sum()) for k in range(len(nb) + 1)}
        keep = ev["n_corroborating"] >= nbc["n_required"]
        kept_by_shank[m][s] = {k: v[keep] for k, v in ev.items()}
        per_shank[m][s] = {
            "channel": ch,
            "depth_um": float(locs[ch, 1]),
            "spikiness": spikiness[s],
            "neighbours": {k: int(v) for k, v in nb.items()},
            "n_candidates": int(n),
            "n_corroborating_sweep": sweep,
            "kept": stats(kept_by_shank[m][s]),
        }
        tables[m].append(ev)
        print(f"  merge {m * 1000:g} ms: {n} candidates, sweep {sweep}", flush=True)
    del z, wide

for m in merges:
    run_params = {
        "experiment": args.experiment,
        "lfp": args.lfp_dir.name,
        "probe": args.probe,
        "shanks": args.shanks,
        "reference": args.reference,
        "scoring": args.scoring,
        "band": cfg["band"],
        "baseline_pool": pool,
        "detection": {**det, "min_inter_event_s": m},
        # n_required excluded: a post-hoc filter on n_corroborating, not a property of the run
        "neighbours": {k: v for k, v in nbc.items() if k != "n_required"},
        "channel_selection": {k: cs[k] for k in ("method", "n_windows", "seed", "smooth_um")},
        "channels": {int(s): int(ch) for s, ch in selected.items()},
        "pad_s": args.pad_s,
    }
    run_id = hashlib.sha256(yaml.safe_dump(run_params, sort_keys=True).encode()).hexdigest()[:8]
    out_dir = events_root / run_id / c["label"]
    out_dir.mkdir(parents=True, exist_ok=True)

    events = {k: np.concatenate([t[k] for t in tables[m]]) for k in tables[m][0]}

    # cross-shank coincidence of kept events; see run_ripples.py for why it is asymmetric
    coincidence = {}
    for a in sorted(kept_by_shank[m]):
        row = {}
        for b in sorted(kept_by_shank[m]):
            if a == b or kept_by_shank[m][a]["start"].size == 0:
                continue
            frac = count_corroborating(kept_by_shank[m][a], [kept_by_shank[m][b]], n_samples).mean()
            row[int(b)] = round(float(frac), 4)
        coincidence[int(a)] = row

    kept_all = {k: v[events["n_corroborating"] >= nbc["n_required"]] for k, v in events.items()}
    summary = {
        "n_candidates": int(events["start"].size),
        "n_required": nbc["n_required"],
        "nrem_hours": round(float(nrem_s / 3600), 3),
        "kept": stats(kept_all, iei=False),
    }
    print(f"\n=== merge {m * 1000:g} ms -> run {run_id} ===\n"
          f"cross-shank coincidence: {coincidence}\n" + yaml.safe_dump(summary, sort_keys=False), flush=True)

    # slice-relative samples -> absolute recording frames and seconds (the scoring result's clock)
    for key in ("start", "end", "peak"):
        events[key] = events[key] + a0
        events[f"{key}_s"] = events[key] / fs
    np.savez(out_dir / "events.npz", **events)
    with open(out_dir / "events.yml", "w") as f:
        yaml.safe_dump({
            "run_id": run_id,
            "created": datetime.now().isoformat(timespec="seconds"),
            "chunk": {"label": c["label"], "t_start": str(c["t_start"]), "t_end": str(c["t_end"]),
                      "start_frame": int(c["start_frame"]), "end_frame": int(c["end_frame"])},
            "filtered_frames": [int(a0), int(a1)],
            "fs": float(fs),
            "run_params": run_params,
            "ripple_config": cfg,
            "sources": {"lfp": [str(p) for p in zarr_paths], "states": str(states_path),
                        "channel_selection": str(sel_path)},
            "per_shank": per_shank[m],
            "cross_shank_coincidence": coincidence,
            "summary": summary,
        }, f, sort_keys=False)
    tmp = events_root / run_id / f".run.yml.{os.getpid()}"
    tmp.write_text(yaml.safe_dump({"run_id": run_id, "run_params": run_params}, sort_keys=False))
    os.replace(tmp, events_root / run_id / "run.yml")
    print(f"saved -> {out_dir}", flush=True)
