"""Exploratory: theta KDE on the REM decision population (non-NREM, non-moving bins) for every channel.

Usage: python explore_pool_kde.py --lfp-dir DIR --make-group --variant A                          # group on the A windows
       python explore_pool_kde.py --lfp-dir DIR --shank N [--chunk 1 | --variant A] [--stride 1]   # theta at group bins
       python explore_pool_kde.py --lfp-dir DIR --plot [--chunk 1 | --variant A] [--stride 1]      # KDE pages, summary, CSV
The group always comes from no-CMR slow wave and EMG (decided 2026-09-21): a chunk's scoring result, or the
variant A windows (15 min every 4 h in every chunk) with thresholds fitted once across all of them.
"""

import os
import csv
import json
import argparse
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from scipy import signal
from scipy.stats import gaussian_kde
from diptest import diptest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sleep_sandbox.io import load_input_recording, load_session_timeline, midday_chunks
from sleep_sandbox.analysis import log_spectrogram, theta_ratio, project_pc1, smooth_norm, find_thresh

repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")

parser = argparse.ArgumentParser()
parser.add_argument("--lfp-dir", type=Path, required=True, help="dir of per-shank LFP zarrs (the theta arm)")
parser.add_argument("--shank", type=int, default=None, help="compute theta at the group's bins for this shank")
parser.add_argument("--plot", action="store_true", help="KDE pages, summary and CSV from the per-shank outputs")
parser.add_argument("--make-group", action="store_true", help="build the group on the --variant windows")
parser.add_argument("--chunk", type=int, default=1, help="group from this chunk's no-CMR scoring (unused with --variant)")
parser.add_argument("--variant", choices=["A"], default=None, help="group on this channel-selection variant's windows")
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--probe", default="ProbeB")
parser.add_argument("--shanks", type=int, nargs="+", default=[0, 1, 2, 3])
parser.add_argument("--emg-dir", type=Path, default=Path(os.environ["PREPRO_EMG_DIR"]),
                    help="EMG zarr dir; its name locates the compute_emg.py outputs")
parser.add_argument("--out-base", type=Path, default=repo_root / "data" / "derivatives" / "abcEphys01")
args = parser.parse_args()
if [args.shank is not None, args.plot, args.make_group].count(True) != 1 or (args.make_group and args.variant is None):
    parser.error("give exactly one of --shank/--plot/--make-group; --make-group needs --variant")

shank_tag = f"{args.probe}_shanks{''.join(map(str, args.shanks))}"
sw_dir = args.out_base / "lfp_pp_no_cmr" / shank_tag          # the group's source, for both arms
source = f"chunk{args.chunk:02d}" if args.variant is None else args.variant
out_dir = args.out_base / args.lfp_dir.name / shank_tag / f"pool_kde_{source}"
out_dir.mkdir(parents=True, exist_ok=True)
group_file = sw_dir / f"pool_kde_{source}" / f"group_{source}.npz"
with open(repo_root / "config/sleep_scoring.yml") as f:
    cfg = yaml.safe_load(f)
skw = {k: cfg["spectrogram"][k] for k in ("window_s", "step_s", "freq_min", "freq_max", "n_freq_bins")}
tc, bt = cfg["threshold"], cfg["bimodal_threshold"]
floor = tc["min_prominence_frac"]


def thresh(x, label):
    return find_thresh(x, tc["method"], bt["startbins"], bt["maxbins"], tc["kde_grid_n"], floor, label=label)


def zarr_path(shank):
    hits = sorted(args.lfp_dir.glob(f"*_{args.probe}_shank{shank}.zarr")) + \
           sorted(args.lfp_dir.glob(f"*_{args.probe}_shank_{shank}.zarr"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{len(hits)} zarrs in {args.lfp_dir} match {args.probe} shank {shank}")
    return hits[0]


# --- group on the variant windows (--make-group) ------------------------------
if args.make_group:
    rec = load_input_recording([zarr_path(args.shanks[0])])
    fs = rec.get_sampling_frequency()
    rec_start, _ = load_session_timeline(args.lfp_dir.parent / f"abcEphys01_{args.probe[-1].upper()}_ephys_paths.csv")
    chunks = midday_chunks(rec_start, rec.get_num_frames(), fs)
    # As select_global_basis.variant_windows("A"): 6 x 15 min per chunk, 4 h apart from each chunk's start.
    w = int(15 * 60 * fs)
    starts = np.array([s for c in chunks for s in (c["start_frame"] + int(k * 4 * 3600 * fs) for k in range(6))
                       if s + w <= c["end_frame"]])
    sw_ch = json.loads((sw_dir / "channel_selection" / "A_summary.json").read_text())["sw_channel"]
    b = np.load(sw_dir / "pc1_basis" / "A_basis.npz")
    spec = np.load(sw_dir / "pc1_basis" / "spec_cache" / f"A_ch{sw_ch}.npz")["spec"]    # the windows, concatenated in order
    pc1 = project_pc1(spec, {"loading": b[f"ch{sw_ch}_loading"], "mu": b[f"ch{sw_ch}_mu"], "sd": b[f"ch{sw_ch}_sd"]})
    t_cat = skw["window_s"] / 2 + np.arange(spec.shape[1]) * skw["step_s"]
    win_s = w / fs
    wi = (t_cat // win_s).astype(int)
    off = t_cat - wi * win_s
    t_abs = starts[wi] / fs + off
    # A bin whose spectrogram window crosses a seam mixes two windows. Dropping those leaves 10 bins between
    # windows, more than the smoothing half-window, so smoothing also stays within each window.
    seam = (off < skw["window_s"] / 2) | (off > win_s - skw["window_s"] / 2)
    edge = int(round(cfg["missing_data"]["gap_margin_s"] / skw["step_s"]))
    emg = [np.load(f) for f in sorted((args.out_base / args.emg_dir.name / shank_tag).glob("emg_chunk*.npz"))]
    emg_b = np.interp(t_abs, np.concatenate([e["times"] for e in emg]), np.concatenate([e["emg"] for e in emg]))
    nodata = (np.convolve(np.isnan(pc1), np.ones(2 * edge + 1), mode="same") > 0) | np.isnan(emg_b) | seam
    sw = smooth_norm(np.where(nodata, np.nan, pc1), step_s=skw["step_s"], win_s=cfg["smoothing"]["window_s"])
    mo = smooth_norm(np.where(nodata, np.nan, emg_b), step_s=skw["step_s"], win_s=cfg["smoothing"]["window_s"])
    sw_thr, mo_thr = thresh(sw, "sw"), thresh(mo, "motion")
    group = ~(sw > sw_thr) & ~((sw < sw_thr) & (mo > mo_thr)) & ~nodata    # scoring's rule, per-chunk fits replaced by one
    np.savez(group_file, times_abs=t_abs[group], all_times_abs=t_abs, window=wi, sw_metric=sw, motion_metric=mo,
             nodata=nodata, sw_thresh=sw_thr, motion_thresh=mo_thr)

    valid = ~nodata
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5), dpi=130, gridspec_kw={"width_ratios": [1, 1, 2]})
    for a, (x, t, name) in zip(ax, [(sw, sw_thr, f"sw_metric (no-CMR ch{sw_ch}, A basis)"), (mo, mo_thr, "motion_metric (EMG)")]):
        a.hist(x[valid], bins=100, density=True, color="0.6")
        a.axvline(t, color="k", ls="--", label=f"thresh = {t:.3f}")
        a.set(xlabel=name, ylabel="density (valid bins)")
        a.legend(fontsize=8)
    ax[2].bar(np.arange(len(starts)), np.bincount(wi[group], minlength=len(starts)), color="tab:red", width=0.8)
    ax[2].set(xlabel="window (6 per chunk, 4 h apart from each chunk's start)", ylabel="group bins (of ~890 per window)")
    for k in range(0, len(starts), 6):
        ax[2].axvline(k - 0.5, color="0.8", lw=0.8)
    fig.suptitle(f"variant A group: {int(group.sum())} non-NREM, non-moving bins of {int(valid.sum())} valid "
                 f"({len(starts)} windows x 15 min, {int(nodata.sum())} bins nodata incl. seams)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(group_file.with_name(f"group_{source}.png"))
    plt.close(fig)
    print(f"wrote {group_file} ({int(group.sum())} group bins); sw_thresh {sw_thr:.3f}, motion_thresh {mo_thr:.3f}", flush=True)
    raise SystemExit

if args.variant is None:
    res_dir = next((sw_dir / "scoring").glob(f"chunk{args.chunk:02d}_*"))
    r = np.load(res_dir / "result.npz")
    sw_thresh = json.loads((res_dir / "summary.json").read_text())["sw_thresh"]
    gt = r["times_abs"][~(r["sw_metric"] > sw_thresh) & ~r["mov"] & ~r["nodata"]]    # conditioned_theta_thresh's keep
else:
    gt = np.load(group_file)["times_abs"]

# --- theta at the group's bins, one shank (--shank) -----------------------------
if args.shank is not None:
    rec = load_input_recording([zarr_path(args.shank)])
    fs = rec.get_sampling_frequency()
    offset = sum(load_input_recording([zarr_path(s)]).get_num_channels() for s in args.shanks[:args.shanks.index(args.shank)])
    pos = np.array(json.loads((zarr_path(args.shank) / ".zattrs").read_text())["probegroup"]["probes"][0]["contact_positions"])
    cand = np.arange(0, rec.get_num_channels(), args.stride)
    ids = rec.channel_ids[cand]

    # A group bin's smoothed value needs the smoothing half-window of bins either side, and each bin needs half a
    # spectrogram window of signal either side; everything else is skipped. Bin centres sit on whole multiples of
    # step from recording start (chunk and window starts are whole seconds), so bins are indexed on that grid.
    step, half, hw = skw["step_s"], int(round(cfg["smoothing"]["window_s"] / skw["step_s"])) // 2, skw["window_s"] / 2
    gk = np.round(gt / step).astype(np.int64)
    need = np.unique((gk[:, None] + np.arange(-half, half + 1)).ravel())
    need = need[(need * step >= hw) & (need * step <= rec.get_num_frames() / fs - hw)]   # recording ends
    segs = np.split(need, np.flatnonzero(np.diff(need) > 1) + 1)
    print(f"{args.lfp_dir.name} shank {args.shank}, {source}: {len(cand)} channels, {len(gt)} group bins, {len(segs)} segments, "
          f"{len(need) / 3600:.2f} h of bins", flush=True)

    conv = cfg["theta"]["concordant_convention"]
    kern = np.ones(2 * half + 1)
    ratio, sm = np.full((len(cand), len(need)), np.nan), np.full((len(cand), len(need)), np.nan)
    p0 = 0
    for k, seg in enumerate(segs):
        traces = rec.get_traces(start_frame=int(round((seg[0] * step - hw) * fs)),
                                end_frame=int(round((seg[-1] * step + hw) * fs)), channel_ids=ids)
        sl, p0 = slice(p0, p0 + len(seg)), p0 + len(seg)
        norm = np.convolve(np.ones(len(seg)), kern, mode="same")
        for j in range(len(cand)):
            spec, freqs, _ = log_spectrogram(traces[:, j], fs, **skw)
            ratio[j, sl] = theta_ratio(spec, freqs, conv)
            # smooth_norm's moving average (shrinking only at segment ends, which no group bin reaches) without its
            # min-max, which needs the whole day; an affine rescale leaves KDE shape, prominences and dip unchanged.
            sm[j, sl] = np.convolve(ratio[j, sl], kern, mode="same") / norm
        if k % 25 == 0:
            print(f"  segment {k}/{len(segs)}", flush=True)

    gi = np.searchsorted(need, gk)
    np.savez(out_dir / f"pool_theta_shank{args.shank}_stride{args.stride}.npz",
             channel=offset + cand, channel_id=np.asarray(ids), shank=np.full(len(cand), args.shank),
             x_um=pos[cand, 0], y_um=pos[cand, 1], group_times_abs=gt,
             theta_smoothed=sm[:, gi].astype(np.float32), theta_raw=ratio[:, gi].astype(np.float32))
    print(f"wrote {out_dir}/pool_theta_shank{args.shank}_stride{args.stride}.npz", flush=True)
    raise SystemExit

# --- plot (--plot) -----------------------------------------------------------
d = [np.load(out_dir / f"pool_theta_shank{s}_stride{args.stride}.npz") for s in args.shanks]
cat = {k: np.concatenate([x[k] for x in d]) for k in ("channel", "channel_id", "shank", "x_um", "y_um", "theta_smoothed", "theta_raw")}
rows, kdes = [], []
for j in range(len(cat["channel"])):
    v = cat["theta_smoothed"][j]
    grid = np.linspace(v.min(), v.max(), tc["kde_grid_n"])
    dens = gaussian_kde(v)(grid)
    pk, pr = signal.find_peaks(dens, prominence=0)
    prom = pr["prominences"] / dens.max()
    p2, valley = 0.0, np.nan
    if len(prom) > 1:     # kde_thresh's trough between the two most prominent peaks, here without the floor
        p2 = np.sort(prom)[-2]
        lo, hi = np.sort(pk[np.argsort(prom)[-2:]])
        valley = grid[lo + np.argmin(dens[lo:hi + 1])]
    rows.append({"channel": int(cat["channel"][j]), "channel_id": str(cat["channel_id"][j]), "shank": int(cat["shank"][j]),
                 "x_um": float(cat["x_um"][j]), "y_um": float(cat["y_um"][j]), "prominence2": float(p2),
                 "threshold": float(valley) if p2 >= floor else np.nan, "valley": float(valley),
                 "frac_below_valley": float(np.mean(v < valley)) if np.isfinite(valley) else np.nan,
                 "dip_smoothed": float(diptest(v)[0]), "dip_lograw": float(diptest(np.log10(cat["theta_raw"][j]))[0])})
    kdes.append((grid, dens, pk, prom))

order = np.argsort([-row["prominence2"] for row in rows], kind="stable")
with open(out_dir / f"prominence_stride{args.stride}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["rank", *rows[0]])
    w.writeheader()
    for rank, j in enumerate(order, 1):
        w.writerow({"rank": rank, **rows[j]})

per_page = 25
n_pages = -(-len(order) // per_page)
for p in range(n_pages):
    fig, ax = plt.subplots(5, 5, figsize=(20, 16), dpi=110)
    for a, rank in zip(ax.ravel(), range(p * per_page, min((p + 1) * per_page, len(order)))):
        j = order[rank]; row = rows[j]; grid, dens, pk, prom = kdes[j]
        a.hist(cat["theta_smoothed"][j], bins=60, density=True, color="tab:red", alpha=0.5)
        a.plot(grid, dens, color="k", lw=1.5)
        for q, pq in zip(pk, prom):
            a.plot(grid[q], dens[q], "v", color="k" if pq >= floor else "0.6", ms=6)
        if len(pk) > 1:
            q2 = pk[np.argsort(prom)[-2]]
            a.plot(grid[q2], dens[q2], "v", mfc="none", mec="tab:blue", mew=1.5, ms=11)
        if np.isfinite(row["threshold"]):
            a.axvline(row["threshold"], color="k", ls="--", lw=1)
        a.set_title(f"#{rank + 1} ch{row['channel']} sh{row['shank']} y{row['y_um']:.0f}: 2nd mode {row['prominence2']:.3f}", fontsize=9)
        a.tick_params(labelsize=7)
    for a in ax.ravel()[min(per_page, len(order) - p * per_page):]:
        a.axis("off")
    fig.suptitle(f"{args.lfp_dir.name} {source}: smoothed Watson theta ratio (not 0-1 normalised) on the REM decision "
                 f"population (n={len(gt)} bins), sorted by 2nd-mode prominence / max density; page {p + 1}/{n_pages}\n"
                 f"triangles = KDE peaks (black = over the {floor} floor), blue ring = the 2nd mode, dashed = threshold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / f"pool_kde_stride{args.stride}_page{p + 1:02d}.png")
    plt.close(fig)

# Summary: 2nd-mode prominence against the share of the group below the valley, one point per channel.
p2 = np.array([row["prominence2"] for row in rows]); below = np.array([row["frac_below_valley"] for row in rows])
ch, sh = cat["channel"], cat["shank"]
ok = np.isfinite(below)
fig, a = plt.subplots(figsize=(9, 6.5), dpi=140)
for s, col in zip(args.shanks, ["tab:blue", "tab:orange", "tab:green", "tab:purple"]):
    for over, m in ((True, ok & (sh == s) & (p2 >= floor)), (False, ok & (sh == s) & (p2 < floor))):
        a.scatter(below[m], p2[m], s=22, color=col if over else "none", edgecolors=col, lw=1, label=f"shank {s}" if over else None)
a.axhline(floor, color="0.3", ls="--", lw=1)
a.text(1.0, floor, f"floor {floor}", va="bottom", ha="right", fontsize=8, color="0.3")
for i, j in enumerate(order[:5]):
    a.annotate(f"ch{ch[j]}", (below[j], p2[j]), xytext=(5, 3 if i % 2 == 0 else -9), textcoords="offset points", fontsize=8)
a.set(xlim=(0, 1), xlabel="proportion of group bins below the threshold\n(valley between the two most prominent KDE peaks)",
      ylabel="2nd-mode prominence / max density",
      title=f"{args.lfp_dir.name} {source}, stride {args.stride}: {int(ok.sum())} channels with >= 2 KDE peaks "
            f"({int((ok & (p2 >= floor)).sum())} over the floor), {int((~ok).sum())} with one peak (not shown)")
a.legend(fontsize=8, title="filled = over the floor", title_fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(out_dir / f"prominence_vs_below_stride{args.stride}.png")
plt.close(fig)
print(f"wrote {n_pages} pages, prominence_vs_below_stride{args.stride}.png and prominence_stride{args.stride}.csv to {out_dir}/",
      flush=True)
