"""Channel selection on the pre-downsampled 1250 Hz zarr input, under three subsampling variants.

Variants: A = 6 x 15 min per chunk spread across all 16 chunks; B1/B2 = one full gap-free chunk
each (5 and 10). Running all three shows whether the pick depends on how the recording is sampled.
Per-shank best channels and their summary statistics are saved so a shank that finishes later can
be compared against these on identical measures.

Usage: python select_global_basis.py --variant A --scan-shank N [--stride 4] [--shanks 0 1 2 3]
       python select_global_basis.py --variant A --merge [--shanks 0 1 2 3]
       python select_global_basis.py --spec-unit K | --fit-basis [--variants A B1 B2]
"""

import os
import json
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spikeinterface.full as si
from diptest import diptest

from sleep_sandbox.io import load_session_timeline, midday_chunks, load_input_recording
from sleep_sandbox.analysis import (
    broadband_pc1, band_power,
    log_spectrogram, theta_ratio, smooth_norm, find_thresh, fit_pc1_basis, project_pc1, cohens_kappa,
)

repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")

parser = argparse.ArgumentParser()
parser.add_argument("--lfp-dir", type=Path, default=Path(os.environ["PREPRO_LFP_DIR"]),
                    help="dir of per-shank LFP zarrs; its name (e.g. lfp_pp_with_cmr) is the file prefix")
parser.add_argument("--probe", default="ProbeB")
parser.add_argument("--shanks", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--csv", type=Path, default=None, help="ephys-paths CSV (default: derived from --probe)")
parser.add_argument("--variant", choices=["A", "B1", "B2"])
parser.add_argument("--scan-shank", type=int, default=None,
                    help="scan one shank of --shanks, candidates counted within the shank")
parser.add_argument("--merge", action="store_true", help="combine the per-shank scans into the variant summary")
parser.add_argument("--variants", nargs="+", choices=["A", "B1", "B2"], default=["A", "B1", "B2"],
                    help="summaries whose picks --spec-unit/--fit-basis use")
parser.add_argument("--fit-basis", action="store_true",
                    help="skip the scan; fit PC1 bases on the channels in the --variants summaries and evaluate them")
parser.add_argument("--spec-unit", type=int, default=None,
                    help="compute and cache one spectrogram unit (index into spec_units()) and exit")
parser.add_argument("--stride", type=int, default=4)
parser.add_argument("--experiment", default="abcEphys01", help="names the --csv and --out-base defaults")
parser.add_argument("--b1-chunk", type=int, default=5, help="the chunk variant B1 fits on")
parser.add_argument("--b2-chunk", type=int, default=10, help="the chunk variant B2 fits on")
parser.add_argument("--eval-chunks", type=int, nargs="+", default=None,
                    help="basis evaluation chunks (default: chunk 1 and the B1/B2 chunks)")
parser.add_argument("--a-windows-per-chunk", type=int, default=6, help="variant A windows per chunk")
parser.add_argument("--a-window-min", type=float, default=15, help="variant A window length, minutes")
parser.add_argument("--a-spacing-h", type=float, default=4, help="variant A window spacing, hours")
parser.add_argument("--out-base", type=Path, default=None)
args = parser.parse_args()
if not args.fit_basis and args.spec_unit is None and (args.variant is None or (args.scan_shank is None) == (not args.merge)):
    parser.error("give --variant with exactly one of --scan-shank/--merge, or --fit-basis, or --spec-unit")

out_base = args.out_base or repo_root / "data" / "derivatives" / args.experiment
csv_path = args.csv or args.lfp_dir.parent / f"{args.experiment}_{args.probe[-1].upper()}_ephys_paths.csv"
B_CHUNK = {"B1": args.b1_chunk, "B2": args.b2_chunk}
# Upstream names differ by arm: lfp_pp_with_cmr_ProbeB_shank0.zarr (plus pp_rec_ProbeB_shank_3.zarr there)
# vs lfp_pp_rec_ProbeB_shank_0.zarr in lfp_pp_no_cmr. Accept either separator, but only one match.
def zarr_path(shank):
    hits = sorted(args.lfp_dir.glob(f"*_{args.probe}_shank{shank}.zarr")) + \
           sorted(args.lfp_dir.glob(f"*_{args.probe}_shank_{shank}.zarr"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{len(hits)} zarrs in {args.lfp_dir} match {args.probe} shank {shank}")
    return hits[0]


zarr_paths = [zarr_path(s) for s in args.shanks]

with open(repo_root / "config/sleep_scoring.yml") as f:
    cfg = yaml.safe_load(f)
spec_cfg = cfg["spectrogram"]
skw = dict(window_s=spec_cfg["window_s"], step_s=spec_cfg["step_s"], freq_min=spec_cfg["freq_min"],
           freq_max=spec_cfg["freq_max"], n_freq_bins=spec_cfg["n_freq_bins"])
pc_index = cfg["slow_wave"]["pc_index"]
orient = cfg["slow_wave"]["orientation_freq_hz"]
peak_band = cfg["theta"]["channel_peak_band"]
thresh_cfg = cfg["threshold"]

# --- subsampling variants ---------------------------------------------------
# A spreads its windows over each chunk's full span rather than clustering them at one time of day:
# a fixed offset would sample a single circadian phase and under-represent the NREM-heavy hours.

def variant_windows(chunks, variant, fs, n_per_chunk=None, window_min=None, spacing_h=None):
    """A: n_per_chunk windows of window_min at spacing_h intervals from each chunk's start.

    n_per_chunk * spacing_h should tile 24 h exactly, so the windows fall on evenly spaced times of
    day and every circadian phase is represented. Windows running past a chunk's end are dropped
    (only reachable for a chunk shorter than n_per_chunk * spacing_h)."""
    n_per_chunk = args.a_windows_per_chunk if n_per_chunk is None else n_per_chunk
    window_min = args.a_window_min if window_min is None else window_min
    spacing_h = args.a_spacing_h if spacing_h is None else spacing_h
    if variant != "A":
        c = chunks[B_CHUNK[variant]]
        return [(c["start_frame"], c["end_frame"])]
    w = int(window_min * 60 * fs)
    out = []
    for c in chunks:
        for k in range(n_per_chunk):
            s = c["start_frame"] + int(k * spacing_h * 3600 * fs)
            if s + w <= c["end_frame"]:
                out.append((s, s + w))
    return out


def _thresh(x, label):
    return find_thresh(x, thresh_cfg["method"], cfg["bimodal_threshold"]["startbins"],
                       cfg["bimodal_threshold"]["maxbins"], thresh_cfg["kde_grid_n"],
                       thresh_cfg["min_prominence_frac"], label=label)


def channel_stats(rec, ch, fs):
    """(stats, series, times) for one channel, all from a single spectrogram pass.

    series carries the per-bin PC1 and the smoothed/[0,1] sw_metric that is actually thresholded,
    so the threshold can be checked against the distribution it was drawn from."""
    trace = rec.get_traces(channel_ids=[rec.channel_ids[ch]]).squeeze()
    pc1, loading, spec, freqs, times = broadband_pc1(trace, fs, pc=pc_index, orientation_freq_hz=orient, **skw)
    real = ~np.isnan(pc1)
    dip, pval = diptest(pc1[real])
    sw = smooth_norm(pc1, step_s=skw["step_s"], win_s=cfg["smoothing"]["window_s"])
    thr = _thresh(sw, f"sw:ch{ch}")
    stats = {
        "channel": int(ch),
        "dip": float(dip), "dip_pvalue": float(pval),
        # zspec rows are unit-variance, so total variance is n_freq_bins and this is exact
        "pc1_variance_explained": float(np.nanvar(pc1) / skw["n_freq_bins"]),
        "peak_theta_ratio": float(band_power(spec, freqs, tuple(peak_band["theta"])).mean()
                                  / band_power(spec, freqs, tuple(peak_band["denom"])).mean()),
        "sw_thresh": float(thr),
        "frac_above_thresh": float(np.nanmean(sw > thr)),
        "sw_metric": {"mean": float(np.nanmean(sw)), "std": float(np.nanstd(sw)),
                      "median": float(np.nanmedian(sw)), "q25": float(np.nanquantile(sw, 0.25)),
                      "q75": float(np.nanquantile(sw, 0.75))},
        "n_bins": int(len(pc1)), "n_valid_bins": int(real.sum()),
        "loading": loading.tolist(),
    }
    return stats, {"pc1": pc1, "sw_metric": sw}, times


def theta_series(rec, ch, fs):
    """(series, thresh) for the peakTH-selected channel: the raw ratio and the smoothed/[0,1]
    theta_metric that classify() thresholds, on the concordant convention."""
    trace = rec.get_traces(channel_ids=[rec.channel_ids[ch]]).squeeze()
    spec, freqs, _ = log_spectrogram(trace, fs, **skw)
    ratio = theta_ratio(spec, freqs, cfg["theta"]["concordant_convention"])
    tm = smooth_norm(ratio, step_s=skw["step_s"], win_s=cfg["smoothing"]["window_s"])
    return {"theta_ratio": ratio, "theta_metric": tm}, _thresh(tm, f"theta:ch{ch}")


# --- PC1 basis fit (--fit-basis) --------------------------------------------
# One basis per variant per shank's dip-best channel, each projected onto the gap-free evaluation
# chunks and compared with a refit on that chunk -- the per-chunk behaviour a global basis replaces.
EVAL_CHUNKS = args.eval_chunks or sorted({1, args.b1_chunk, args.b2_chunk})
IN_SAMPLE = {("B1", args.b1_chunk), ("B2", args.b2_chunk)}   # these cells must equal the refit exactly


# B1's fit window is exactly chunk 5 and B2's is chunk 10, so keying the cache by source rather
# than by variant leaves 12 distinct spectrograms (4 sources x 3 channels) instead of 18.
SPEC_SOURCES = ["A"] + [f"chunk{c}" for c in EVAL_CHUNKS]
BASIS_SOURCE = {"A": "A", "B1": f"chunk{args.b1_chunk}", "B2": f"chunk{args.b2_chunk}"}


def picked_channels(sel_dir, variants):
    picks = {v: [s["channel"] for s in json.loads((sel_dir / f"{v}_summary.json").read_text())["per_shank"].values()]
             for v in variants}
    if len({tuple(p) for p in picks.values()}) != 1:
        raise ValueError(f"variants picked different channels {picks}; the comparison assumes one set")
    return picks[variants[0]]


def spec_units(chans):
    return [(src, ch) for src in SPEC_SOURCES for ch in chans]


def source_rec(source, rec, chunks):
    if source == "A":
        return si.concatenate_recordings(
            [rec.frame_slice(start_frame=a, end_frame=b) for a, b in variant_windows(chunks, "A", fs)])
    c = chunks[int(source.removeprefix("chunk"))]
    return rec.frame_slice(start_frame=c["start_frame"], end_frame=c["end_frame"])


def cached_spec(source, ch, rec, chunks, cache):
    """One 24 h single-channel spectrogram, read from disk if a --spec-unit task already wrote it."""
    f = cache / f"{source}_ch{ch}.npz"
    if f.exists():
        d = np.load(f)
        return d["spec"], d["freqs"]
    trace = source_rec(source, rec, chunks).get_traces(channel_ids=[rec.channel_ids[ch]]).squeeze()
    spec, freqs, _ = log_spectrogram(trace, fs, **skw)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(f, spec=spec, freqs=freqs)
    return spec, freqs


def nrem_mask(pc1, label):
    sw = smooth_norm(pc1, step_s=skw["step_s"], win_s=cfg["smoothing"]["window_s"])
    thr = _thresh(sw, label)
    return sw > thr, thr


def fit_bases(rec, chunks, sel_dir, variants):
    chans = picked_channels(sel_dir, variants)
    cache = sel_dir.parent / "pc1_basis" / "spec_cache"

    bases = {}
    for v in variants:
        for ch in chans:
            spec, freqs = cached_spec(BASIS_SOURCE[v], ch, rec, chunks, cache)
            bases[v, ch] = fit_pc1_basis(spec, freqs, pc=pc_index, orientation_freq_hz=orient)
            print(f"fit {v} ch{ch}: var explained {bases[v, ch]['var_explained']:.3f}", flush=True)

    comp = []
    for ch in chans:
        for i, j in combinations(variants, 2):
            bi, bj = bases[i, ch], bases[j, ch]
            # mu drift only offsets PC1 by a constant, which smooth_norm's min-max removes, so it
            # cannot move a threshold; sd drift reweights frequencies and can.
            comp.append({"channel": ch, "pair": f"{i}-{j}",
                         "loading_cos": float(bi["loading"] @ bj["loading"]),   # PCA components are unit-norm
                         "mu_drift_sd": float(np.max(np.abs(bi["mu"] - bj["mu"]) / ((bi["sd"] + bj["sd"]) / 2))),
                         "sd_drift_rel": float(np.max(np.abs(bi["sd"] - bj["sd"]) / ((bi["sd"] + bj["sd"]) / 2)))})

    rows = []
    for k in EVAL_CHUNKS:
        for ch in chans:
            spec, freqs = cached_spec(f"chunk{k}", ch, rec, chunks, cache)
            ref = fit_pc1_basis(spec, freqs, pc=pc_index, orientation_freq_hz=orient)
            valid = spec.sum(axis=0) > 0
            # every direction is scored on the chunk's own z-scoring, so var explained compares direction
            # only (identical for a variant's global and middle rows)
            zloc = (np.log10(spec[:, valid]) - ref["mu"][:, None]) / ref["sd"][:, None]
            pc_ref = project_pc1(spec, ref)
            nrem_ref, _ = nrem_mask(pc_ref, f"sw:ch{ch}:chunk{k}:refit")
            dip_ref = diptest(pc_ref[valid])[0]
            arms = [*((v, m, bases[v, ch]) for m in ("global", "middle") for v in variants), ("refit", "refit", ref)]
            for v, m, b in arms:
                pc = pc_ref if m == "refit" else project_pc1(spec, b, m)
                nrem, thr = nrem_mask(pc, f"sw:ch{ch}:chunk{k}:{v}:{m}")
                found = bool(np.isfinite(thr))
                rows.append({
                    "channel": ch, "eval_chunk": k, "basis": v, "mode": m,
                    "in_sample": v == "refit" or (v, k) in IN_SAMPLE,
                    "loading_cos_to_refit": float(b["loading"] @ ref["loading"]),
                    "var_explained": float(np.var(b["loading"] @ zloc) / zloc.shape[0]),
                    "var_explained_ratio": float(np.var(b["loading"] @ zloc) / np.var(ref["loading"] @ zloc)),
                    "r_to_refit": float(np.corrcoef(pc[valid], pc_ref[valid])[0, 1]),
                    "dip_ratio": float(diptest(pc[valid])[0] / dip_ref),
                    "thresh_found": found,
                    "nrem_frac": float(nrem[valid].mean()) if found else np.nan,
                    "nrem_frac_refit": float(nrem_ref[valid].mean()),
                    "kappa_to_refit": float(cohens_kappa(nrem[valid].astype(int), nrem_ref[valid].astype(int), 2))
                                      if found else np.nan,
                })
            print(f"evaluated chunk {k} ch{ch}", flush=True)

    out = sel_dir.parent / "pc1_basis"
    out.mkdir(exist_ok=True)
    for v in variants:
        np.savez(out / f"{v}_basis.npz", channels=np.array(chans), freqs=bases[v, chans[0]]["freqs"],
                 windows=np.array(variant_windows(chunks, v, fs)),
                 **{f"ch{ch}_{key}": bases[v, ch][key] for ch in chans for key in ("loading", "mu", "sd", "var_explained")})
    comp, rows = pd.DataFrame(comp), pd.DataFrame(rows)
    rows.to_csv(out / "basis_evaluation.csv", index=False)
    comp.to_csv(out / "variant_comparison.csv", index=False)
    print(comp.round(3).to_string(index=False))
    print(rows.round(3).to_string(index=False))

    fig, ax = plt.subplots(3, len(chans), figsize=(4.5 * len(chans), 8), dpi=150, sharex=True, squeeze=False)
    for j, ch in enumerate(chans):
        for v in variants:
            for i, key in enumerate(("loading", "mu", "sd")):
                ax[i][j].semilogx(bases[v, ch]["freqs"], bases[v, ch][key], label=v)
        ax[0][j].set_title(f"ch{ch} (shank {groups[ch]})")
        ax[2][j].set_xlabel("frequency (Hz)")
    ax[0][0].set_ylabel("PC1 loading")
    ax[1][0].set_ylabel("z-score centre (mean log10 power)")
    ax[2][0].set_ylabel("z-score scale (sd log10 power)")
    ax[0][0].legend()
    fig.suptitle(f"{args.probe} shanks {args.shanks}: PC1 bases by sampling variant")
    fig.tight_layout()
    fig.savefig(out / "loadings.png")

    # Projected-basis arms against the per-chunk refit they would replace; in-sample cells are hollow.
    metrics = [("kappa_to_refit", "NREM kappa to refit"), ("r_to_refit", "PC1 r to refit"),
               ("var_explained_ratio", "var explained / refit")]
    fig, ax = plt.subplots(len(metrics), len(chans), figsize=(4.5 * len(chans), 8), dpi=150, sharex=True,
                           squeeze=False)
    for j, ch in enumerate(chans):
        for (v, m), g in rows[(rows["channel"] == ch) & (rows["mode"] != "refit")].groupby(["basis", "mode"]):
            for i, (key, name) in enumerate(metrics):
                line, = ax[i][j].plot(g["eval_chunk"], g[key], lw=0.8, ls="-" if m == "global" else "--",
                                      label=f"{v} {m}")
                ax[i][j].scatter(g["eval_chunk"], g[key], s=18, color=line.get_color(),
                                 facecolors=["none" if s else line.get_color() for s in g["in_sample"]])
                ax[i][0].set_ylabel(name)
        ax[0][j].set_title(f"ch{ch} (shank {groups[ch]})")
        ax[-1][j].set_xlabel("evaluation chunk")
        ax[-1][j].set_xticks(EVAL_CHUNKS)
    ax[0][0].legend(fontsize=7)
    fig.suptitle(f"{args.probe} shanks {args.shanks}: basis {variants} vs per-chunk refit")
    fig.tight_layout()
    fig.savefig(out / "basis_evaluation.png")
    print(f"\nwrote {out}/", flush=True)


# --- run --------------------------------------------------------------------
rec_start, _ = load_session_timeline(csv_path)
rec = load_input_recording(zarr_paths)
fs = rec.get_sampling_frequency()
chunks = midday_chunks(rec_start, rec.get_num_frames(), fs)
groups = np.asarray(rec.get_property("group"))

out_dir = (out_base / args.lfp_dir.name / f"{args.probe}_shanks{''.join(map(str, args.shanks))}"
           / "channel_selection")
out_dir.mkdir(parents=True, exist_ok=True)

if args.spec_unit is not None:
    src, ch = spec_units(picked_channels(out_dir, args.variants))[args.spec_unit]
    cached_spec(src, ch, rec, chunks, out_dir.parent / "pc1_basis" / "spec_cache")
    print(f"cached {src}_ch{ch}", flush=True)
    raise SystemExit

if args.fit_basis:
    fit_bases(rec, chunks, out_dir, args.variants)
    raise SystemExit

windows = variant_windows(chunks, args.variant, fs)
total_h = sum(b - a for a, b in windows) / fs / 3600

# --- per-shank scan (--scan-shank) ------------------------------------------
# Loading one shank alone counts candidates from its own first contact, so a pick does not depend on
# which other shanks are loaded, and holds one shank's traces in memory instead of the probe's.
if args.scan_shank is not None:
    sh = args.scan_shank
    offset = int(np.flatnonzero(groups == sh)[0])   # aggregate index of the shank's first channel
    srec = load_input_recording([zarr_path(sh)])
    parts = [srec.frame_slice(start_frame=a, end_frame=b) for a, b in windows]
    vrec = parts[0] if len(parts) == 1 else si.concatenate_recordings(parts)
    # device_channel_indices are 0..n-1 in every file, so contact i is channel i
    pos = np.array(json.loads((zarr_path(sh) / ".zattrs").read_text())
                   ["probegroup"]["probes"][0]["contact_positions"])
    n = srec.get_num_channels()
    cand = np.arange(0, n, args.stride)
    print(f"variant {args.variant}: {len(windows)} window(s), {total_h:.2f} h; shank {sh}, {n} ch, "
          f"{len(cand)} candidates (stride {args.stride})", flush=True)

    # Same statistics as select_channel_by_dip and select_theta_channel_peak, from one spectrogram per
    # candidate instead of two, keeping the PC1 variance explained and loading peak those discard.
    dip, var_exp, load_peak_hz, peak_th = (np.full(n, np.nan) for _ in range(4))
    traces = vrec.get_traces(channel_ids=vrec.channel_ids[cand])
    for j, ci in enumerate(cand):
        pc1, loading, spec, freqs, _ = broadband_pc1(traces[:, j], fs, pc=pc_index, orientation_freq_hz=orient, **skw)
        dip[ci] = diptest(pc1[~np.isnan(pc1)])[0]
        var_exp[ci] = np.nanvar(pc1) / skw["n_freq_bins"]
        load_peak_hz[ci] = freqs[np.argmax(loading)]
        peak_th[ci] = (band_power(spec, freqs, tuple(peak_band["theta"])).mean()
                       / band_power(spec, freqs, tuple(peak_band["denom"])).mean())
        print(f"  ch{ci}: dip {dip[ci]:.4f}, PC1 var {var_exp[ci]:.3f}, peakTH {peak_th[ci]:.3f}", flush=True)
    del traces

    best, th_local = int(np.nanargmax(dip)), int(np.nanargmax(peak_th))
    print(f"shank {sh}: best-by-dip ch{best} (y {pos[best, 1]:.0f} um), best-by-peakTH ch{th_local} "
          f"(y {pos[th_local, 1]:.0f} um); stats...", flush=True)
    s, ser, times = channel_stats(vrec, best, fs)
    tser, th_thresh = theta_series(vrec, th_local, fs)
    s.update({"channel": offset + best, "channel_id": int(srec.channel_ids[best]), "y_um": float(pos[best, 1]),
              "stride": args.stride, "n_candidates": int(len(cand)),
              "peak_theta_best_channel": offset + th_local, "peak_theta_best_y_um": float(pos[th_local, 1]),
              "th_thresh": float(th_thresh)})
    (out_dir / f"{args.variant}_shank{sh}.json").write_text(json.dumps(s, indent=2))
    np.savez(out_dir / f"{args.variant}_shank{sh}_curves.npz", channel_idx=offset + np.arange(n),
             channel_ids=np.asarray(srec.channel_ids), x_um=pos[:, 0], y_um=pos[:, 1], candidate_idx=offset + cand,
             dip_stats=dip, pc1_var_explained=var_exp, loading_peak_hz=load_peak_hz, peak_stats=peak_th,
             times=times, **ser, **tser)
    print(f"wrote {out_dir}/{args.variant}_shank{sh}*", flush=True)
    raise SystemExit

# --- merge (--merge) --------------------------------------------------------
per_shank = {sh: json.loads((out_dir / f"{args.variant}_shank{sh}.json").read_text()) for sh in args.shanks}
curves = {sh: dict(np.load(out_dir / f"{args.variant}_shank{sh}_curves.npz")) for sh in args.shanks}
# args.shanks order is the aggregate's channel order, so concatenating gives aggregate-indexed arrays
cat = {k: np.concatenate([curves[sh][k] for sh in args.shanks])
       for k in ("channel_idx", "channel_ids", "x_um", "y_um", "candidate_idx", "dip_stats", "pc1_var_explained",
                 "loading_peak_hz", "peak_stats")}
sw_best, th_best = int(np.nanargmax(cat["dip_stats"])), int(np.nanargmax(cat["peak_stats"]))

summary = {
    "variant": args.variant, "stride": per_shank[args.shanks[0]]["stride"], "probe": args.probe,
    "shanks": args.shanks, "zarr_paths": [str(p) for p in zarr_paths], "rec_start": str(rec_start),
    "n_windows": len(windows), "total_hours": total_h,
    "n_channels": int(rec.get_num_channels()), "n_candidates": int(len(cat["candidate_idx"])),
    "sw_channel": sw_best, "theta_channel": th_best,
    "sw_channel_shank": int(groups[sw_best]), "theta_channel_shank": int(groups[th_best]),
    "sw_channel_id": int(cat["channel_ids"][sw_best]), "theta_channel_id": int(cat["channel_ids"][th_best]),
    "per_shank": per_shank,
}
(out_dir / f"{args.variant}_summary.json").write_text(json.dumps(summary, indent=2))
np.savez(out_dir / f"{args.variant}_curves.npz", groups=groups, windows=np.array(windows), **cat)

# Depth profiles: where along each shank the slow-wave PC1 is most bimodal (the pick), where PC1 explains
# the most variance, and where the theta ratio peaks.
stats = [("dip_stats", "Hartigan dip (PC1)"), ("pc1_var_explained", "PC1 variance explained"),
         ("loading_peak_hz", "PC1 loading peak (Hz)"), ("peak_stats", "peakTH theta ratio")]
fig, ax = plt.subplots(len(stats), len(args.shanks), figsize=(3.6 * len(args.shanks), 12), dpi=150,
                       sharey=True, squeeze=False)
for j, sh in enumerate(args.shanks):
    c = curves[sh]
    idx = c["candidate_idx"] - c["channel_idx"][0]
    idx = idx[np.argsort(c["y_um"][idx], kind="stable")]   # channel order does not follow depth
    marks = [(int(np.nanargmax(c["dip_stats"])), "r", "--", "max dip (pick)"),
             (int(np.nanargmax(c["pc1_var_explained"])), "orange", ":", "max PC1 var"),
             (int(np.nanargmax(c["peak_stats"])), "b", "--", "max peakTH (pick)")]
    for i, (key, name) in enumerate(stats):
        a = ax[i][j]
        a.plot(c[key][idx], c["y_um"][idx], "o-", lw=0.8, ms=2.5, color="k")
        for k, color, ls, label in marks:
            a.axhline(c["y_um"][k], color=color, ls=ls, lw=1, label=f"{label}: ch{c['channel_idx'][k]}")
        a.set_xlabel(name)
    ax[0][j].set_title(f"shank {sh}")
    ax[0][j].legend(fontsize=6)
for i in range(len(stats)):
    ax[i][0].set_ylabel("contact y (um)")
fig.suptitle(f"{args.probe} {args.lfp_dir.name} - variant {args.variant} ({total_h:.1f} h, stride {summary['stride']})")
fig.tight_layout()
fig.savefig(out_dir / f"{args.variant}_depth_profiles.png")

# Distributions the thresholds were drawn from, per shank, for visual validation.
fig, ax = plt.subplots(2, len(args.shanks), figsize=(4.5 * len(args.shanks), 6), dpi=150, squeeze=False)
for j, sh in enumerate(args.shanks):
    s = per_shank[sh]
    for i, (key, thr, name) in enumerate([
            ("sw_metric", s["sw_thresh"], f"sw_metric  ch{s['channel']}"),
            ("theta_metric", s["th_thresh"], f"theta_metric  ch{s['peak_theta_best_channel']}")]):
        x = curves[sh][key]
        a = ax[i][j]
        a.hist(x[~np.isnan(x)], bins=80, density=True, alpha=0.6)
        if np.isfinite(thr):
            a.axvline(thr, color="r", ls="--", label=f"thresh = {thr:.3f}")
            a.legend(fontsize=8)
        else:
            a.set_title("threshold = NaN (unimodal)", color="r", fontsize=8)
        a.set_xlabel(f"shank {sh}  {name}")
    ax[0][j].set_ylabel("density")
fig.suptitle(f"{args.probe} shanks {args.shanks} - variant {args.variant}: thresholded metric distributions")
fig.tight_layout()
fig.savefig(out_dir / f"{args.variant}_metric_histograms.png")

print(f"\nwrote {out_dir}/{args.variant}_*.{{json,npz,png}}")
print(json.dumps({k: v for k, v in summary.items() if k != "per_shank"}, indent=2))
