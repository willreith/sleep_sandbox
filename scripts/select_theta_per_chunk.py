"""Per-chunk theta channel: the largest dip of the smoothed theta metric on each chunk's REM decision population.

Usage: python select_theta_per_chunk.py --lfp-dir DIR --chunk N --shank S [--stride 1] [--pool all]   # theta at the pool's bins
       python select_theta_per_chunk.py --lfp-dir DIR --pick [--stride 1] [--all-stride 2]      # theta_picks json and figures
Each chunk's group (non-NREM, non-moving, valid bins) comes from its result in --group-scoring; it does not depend
on the theta channel. The pick is the largest Hartigan dip among channels whose group KDE yields a threshold under
scoring's rule (decided 2026-09-21). A chunk with no such channel is flagged no_threshold_channel; scoring then sets
its REM threshold on all valid bins, so its channel is the largest all-bins dip among channels whose all-bins KDE
yields a threshold (--pool all scans, decided 2026-09-22). If none does either (no_threshold_all_bins), it takes the
largest group dip overall and scoring finds no theta threshold.
"""

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

from sleep_sandbox.io import load_input_recording
from sleep_sandbox.analysis import log_spectrogram, theta_ratio

repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")

parser = argparse.ArgumentParser()
parser.add_argument("--lfp-dir", type=Path, required=True, help="dir of per-shank LFP zarrs (the theta arm)")
parser.add_argument("--chunk", type=int, default=None, help="chunk index (position among --group-scoring's chunk folders)")
parser.add_argument("--shank", type=int, default=None, help="compute theta at the chunk's group bins for this shank")
parser.add_argument("--pick", action="store_true", help="pick per chunk from all scan outputs; write json and figures")
parser.add_argument("--group-scoring", default="scoring_th-lfp_pp_no_cmr-ch289",
                    help="no-CMR scoring folder whose per-chunk results give each chunk's group")
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--pool", choices=["group", "all"], default="group",
                    help="scan bins: the group, or all valid bins (the population of scoring's theta fallback)")
parser.add_argument("--all-stride", type=int, default=2, help="--pick: stride of the --pool all scans")
parser.add_argument("--probe", default="ProbeB")
parser.add_argument("--shanks", type=int, nargs="+", default=[0, 1, 2, 3])
parser.add_argument("--out-base", type=Path, default=repo_root / "data" / "derivatives" / "abcEphys01")
args = parser.parse_args()
if args.pick == (args.chunk is not None and args.shank is not None):
    parser.error("give --chunk and --shank, or --pick")

shank_tag = f"{args.probe}_shanks{''.join(map(str, args.shanks))}"
group_dir = args.out_base / "lfp_pp_no_cmr" / shank_tag / args.group_scoring     # the group's source, for any theta arm
chunk_dirs = sorted(p for p in group_dir.iterdir() if p.is_dir() and p.name.startswith("chunk"))
out_dir = args.out_base / args.lfp_dir.name / shank_tag / "theta_per_chunk"
with open(repo_root / "config/sleep_scoring.yml") as f:
    cfg = yaml.safe_load(f)
skw = {k: cfg["spectrogram"][k] for k in ("window_s", "step_s", "freq_min", "freq_max", "n_freq_bins")}
floor = cfg["threshold"]["min_prominence_frac"]


def zarr_path(shank):
    hits = sorted(args.lfp_dir.glob(f"*_{args.probe}_shank{shank}.zarr")) + \
           sorted(args.lfp_dir.glob(f"*_{args.probe}_shank_{shank}.zarr"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{len(hits)} zarrs in {args.lfp_dir} match {args.probe} shank {shank}")
    return hits[0]


# --- theta at one chunk's pool bins, one shank (--chunk --shank) -------------------
if not args.pick:
    cd = chunk_dirs[args.chunk]
    r = np.load(cd / "result.npz")
    sw_thresh = json.loads((cd / "summary.json").read_text())["sw_thresh"]
    # group: conditioned_theta_thresh's keep; all: every bin classify's fallback thresholds on
    keep = ~r["nodata"] if args.pool == "all" else ~(r["sw_metric"] > sw_thresh) & ~r["mov"] & ~r["nodata"]
    gt = r["times_abs"][keep]

    rec = load_input_recording([zarr_path(args.shank)])
    fs = rec.get_sampling_frequency()
    offset = sum(load_input_recording([zarr_path(s)]).get_num_channels() for s in args.shanks[:args.shanks.index(args.shank)])
    pos = np.array(json.loads((zarr_path(args.shank) / ".zattrs").read_text())["probegroup"]["probes"][0]["contact_positions"])
    cand = np.arange(0, rec.get_num_channels(), args.stride)
    ids = rec.channel_ids[cand]

    # A group bin's smoothed value needs the smoothing half-window of bins either side, and each bin needs half a
    # spectrogram window of signal either side; the rest of the chunk is skipped. Bin centres sit on whole multiples
    # of step from recording start (chunk starts are whole seconds), so bins are indexed on that grid.
    step, half, hw = skw["step_s"], int(round(cfg["smoothing"]["window_s"] / skw["step_s"])) // 2, skw["window_s"] / 2
    gk = np.round(gt / step).astype(np.int64)
    need = np.unique((gk[:, None] + np.arange(-half, half + 1)).ravel())
    need = need[(need * step >= hw) & (need * step <= rec.get_num_frames() / fs - hw)]   # recording ends
    segs = np.split(need, np.flatnonzero(np.diff(need) > 1) + 1)
    # scoring blanks nodata bins before smoothing (gaps plus their margin, which still has signal)
    nod = np.isin(need, np.round(r["times_abs"][r["nodata"]] / step).astype(np.int64))
    print(f"{args.lfp_dir.name} {cd.name} shank {args.shank}: {len(cand)} channels, {len(gt)} {args.pool} bins, "
          f"{len(segs)} segments, {len(need) / 3600:.2f} h of bins", flush=True)

    conv = cfg["theta"]["concordant_convention"]
    kern = np.ones(2 * half + 1)
    blk_n = int(round(3600 / step))   # a 24 h segment (--pool all) in one read would need ~10 GB of traces
    sm = np.full((len(cand), len(need)), np.nan)
    p0 = 0
    for k, seg in enumerate(segs):
        raw = np.empty((len(cand), len(seg)))
        # Each bin depends only on its own spectrogram window, so blocks give the same bins as one read.
        for b0 in range(0, len(seg), blk_n):
            blk = seg[b0:b0 + blk_n]
            traces = rec.get_traces(start_frame=int(round((blk[0] * step - hw) * fs)),
                                    end_frame=int(round((blk[-1] * step + hw) * fs)), channel_ids=ids)
            for j in range(len(cand)):
                spec, freqs, _ = log_spectrogram(traces[:, j], fs, **skw)
                raw[j, b0:b0 + len(blk)] = theta_ratio(spec, freqs, conv)
        sl, p0 = slice(p0, p0 + len(seg)), p0 + len(seg)
        raw[:, nod[sl]] = np.nan
        real = ~np.isnan(raw)
        # smooth_norm's NaN-skipping moving average (shrinking only at segment ends, which no pool bin reaches) without
        # its min-max, which needs the whole chunk; an affine rescale leaves KDE shape, prominences and dip unchanged.
        for j in range(len(cand)):
            sm[j, sl] = (np.convolve(np.where(real[j], raw[j], 0.0), kern, mode="same")
                         / np.convolve(real[j].astype(float), kern, mode="same"))
        if k % 25 == 0:
            print(f"  segment {k}/{len(segs)}", flush=True)

    (out_dir / cd.name).mkdir(parents=True, exist_ok=True)
    out = out_dir / cd.name / f"theta{'_all' if args.pool == 'all' else ''}_shank{args.shank}_stride{args.stride}.npz"
    np.savez(out, channel=offset + cand, channel_id=np.asarray(ids), shank=np.full(len(cand), args.shank),
             y_um=pos[cand, 1], **{f"{args.pool}_times_abs": gt},
             theta_smoothed=sm[:, np.searchsorted(need, gk)].astype(np.float32))
    print(f"wrote {out}", flush=True)
    raise SystemExit


# --- pick per chunk (--pick) ----------------------------------------------------
def kde_stats(v):
    """(2nd-mode prominence / max density, valley between the two most prominent peaks, grid, density), as kde_thresh."""
    grid = np.linspace(v.min(), v.max(), cfg["threshold"]["kde_grid_n"])
    dens = gaussian_kde(v)(grid)
    pk, pr = signal.find_peaks(dens, prominence=0)
    prom = pr["prominences"] / dens.max()
    p2, valley = 0.0, np.nan
    if len(prom) > 1:
        p2 = np.sort(prom)[-2]
        lo, hi = np.sort(pk[np.argsort(prom)[-2:]])
        valley = grid[lo + np.argmin(dens[lo:hi + 1])]
    return p2, valley, grid, dens


picks, per = {}, {}
for cd in chunk_dirs:
    d = [np.load(out_dir / cd.name / f"theta_shank{s}_stride{args.stride}.npz") for s in args.shanks]
    cat = {k: np.concatenate([x[k] for x in d]) for k in ("channel", "channel_id", "shank", "y_um", "theta_smoothed")}
    n = len(cat["channel"])
    p2, valley, dip = np.zeros(n), np.full(n, np.nan), np.zeros(n)
    for j, v in enumerate(cat["theta_smoothed"]):
        p2[j], valley[j], _, _ = kde_stats(v)
        dip[j] = diptest(v)[0]
    below = np.array([np.mean(v < t) if np.isfinite(t) else np.nan for v, t in zip(cat["theta_smoothed"], valley)])
    eligible = p2 >= floor
    no_thr = not eligible.any()
    cands = np.flatnonzero(eligible) if not no_thr else np.arange(n)
    ranked = cands[np.argsort(-dip[cands])]
    j, r2 = ranked[0], (ranked[1] if len(ranked) > 1 else None)
    rank_ch, rank_dip = cat["channel"], dip
    fb, fbd, picked_on, frac_above = None, None, "group" if not no_thr else "none", 1 - below[j]
    if no_thr:
        # Scoring's fallback thresholds on all valid bins, so the channel is ranked on those.
        da = [np.load(out_dir / cd.name / f"theta_all_shank{s}_stride{args.all_stride}.npz") for s in args.shanks]
        ca = {k: np.concatenate([x[k] for x in da]) for k in ("channel", "theta_smoothed")}
        na = len(ca["channel"])
        p2a, valleya, dipa = np.zeros(na), np.full(na, np.nan), np.zeros(na)
        for i, v in enumerate(ca["theta_smoothed"]):
            p2a[i], valleya[i], _, _ = kde_stats(v)
            dipa[i] = diptest(v)[0]
        elig_a = p2a >= floor
        if elig_a.any():
            ra = np.flatnonzero(elig_a)[np.argsort(-dipa[elig_a])]
            i, r2 = ra[0], (ra[1] if len(ra) > 1 else None)
            j = np.flatnonzero(cat["channel"] == ca["channel"][i])[0]
            rank_ch, rank_dip, picked_on = ca["channel"], dipa, "all_bins"
            frac_above = np.mean(cat["theta_smoothed"][j] > valleya[i])
            fb = {"stride": args.all_stride, "dip_all": float(dipa[i]), "prominence2_all": float(p2a[i]),
                  "valley_all": float(valleya[i]), "n_eligible_all": int(elig_a.sum()), "n_channels_all": int(na),
                  "n_all_bins": int(ca["theta_smoothed"].shape[1])}
            fbd = {"v": ca["theta_smoothed"][i], "n_eligible": int(elig_a.sum()), "n_channels": int(na)}
    picks[cd.name] = {
        "channel": int(cat["channel"][j]), "channel_id": str(cat["channel_id"][j]), "shank": int(cat["shank"][j]),
        "y_um": float(cat["y_um"][j]), "picked_on": picked_on, "dip": float(dip[j]), "prominence2": float(p2[j]),
        "valley_smoothed_ratio": float(valley[j]), "frac_group_above_valley": float(frac_above),
        "runner_up": None if r2 is None else {"channel": int(rank_ch[r2]), "dip": float(rank_dip[r2])},
        "margin": None if r2 is None else float(rank_dip[j if fb is None else i] - rank_dip[r2]),
        "n_eligible": int(eligible.sum()), "n_channels": int(n), "n_group_bins": int(cat["theta_smoothed"].shape[1]),
        "no_threshold_channel": bool(no_thr), "fallback": fb, "no_threshold_all_bins": bool(no_thr and fb is None)}
    per[cd.name] = {"cat": cat, "p2": p2, "below": below, "dip": dip, "valley": valley, "pick": j, "fb": fbd}
    print(f"{cd.name}: ch{picks[cd.name]['channel']} (shank {picks[cd.name]['shank']}, y {picks[cd.name]['y_um']:.0f}), "
          f"dip {dip[j]:.4f}, prominence {p2[j]:.3f}, {int(eligible.sum())} of {n} channels eligible", flush=True)
    if fb:
        print(f"{'!' * 100}\n!!! {cd.name}: NO CHANNEL'S non-NREM, non-moving theta yields a threshold. Picked "
              f"ch{picks[cd.name]['channel']} by the largest ALL-BINS dip ({fb['dip_all']:.4f}) among "
              f"{fb['n_eligible_all']} of {na} channels whose all-bins theta yields a threshold;\n!!! scoring will set "
              f"its REM threshold on ALL valid bins (fallback), with {frac_above:.0%} of the group above it.\n"
              f"{'!' * 100}", flush=True)
    elif no_thr:
        print(f"{'!' * 100}\n!!! {cd.name}: NO CHANNEL'S theta yields a threshold on non-NREM, non-moving bins OR on all "
              f"valid bins. Picked the largest group dip overall (ch{picks[cd.name]['channel']});\n!!! scoring will find "
              f"no theta threshold: REM = 0.\n{'!' * 100}", flush=True)

no_thr_chunks = [k for k, v in picks.items() if v["no_threshold_channel"]]
no_all_chunks = [k for k, v in picks.items() if v["no_threshold_all_bins"]]
out = out_dir / f"theta_picks_stride{args.stride}.json"
out.write_text(json.dumps({
    "rule": "largest dip of the smoothed theta metric on non-NREM, non-moving bins, among channels whose group KDE "
            "yields a threshold (picked_on group); else (no_threshold_channel) the largest dip on all valid bins among "
            f"channels whose all-bins KDE yields a threshold, stride {args.all_stride} (picked_on all_bins; scoring's "
            "fallback thresholds there); else the largest group dip overall (no_threshold_all_bins, REM = 0)",
    "theta_arm": args.lfp_dir.name, "group_scoring": str(group_dir), "stride": args.stride, "floor": floor,
    "no_threshold_chunks": no_thr_chunks, "no_threshold_all_bins_chunks": no_all_chunks, "chunks": picks}, indent=2))
print(f"wrote {out}", flush=True)
if no_thr_chunks:
    print(f"{'!' * 100}\n!!! {len(no_thr_chunks)} of {len(picks)} chunks have NO channel with a group threshold "
          f"(picked on all bins): {', '.join(no_thr_chunks)}\n!!! of these, no all-bins threshold either (REM = 0): "
          f"{', '.join(no_all_chunks) or 'none'}\n{'!' * 100}", flush=True)

# --- figures ---------------------------------------------------------------------
labels = list(per)
nr, nc = -(-len(labels) // 4), 4
colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]


def flag(a, label, text):
    a.set_title(text, fontsize=9, color="tab:red" if picks[label]["no_threshold_channel"] else "k")


# Prominence against share below the valley, one panel per chunk.
fig, ax = plt.subplots(nr, nc, figsize=(20, 4.3 * nr), dpi=110, sharex=True, sharey=True, squeeze=False)
for a, label in zip(ax.ravel(), labels):
    q = per[label]; sh = q["cat"]["shank"]; ok = np.isfinite(q["below"])
    for s, col in zip(args.shanks, colors):
        for over, m in ((True, ok & (sh == s) & (q["p2"] >= floor)), (False, ok & (sh == s) & (q["p2"] < floor))):
            a.scatter(q["below"][m], q["p2"][m], s=12, color=col if over else "none", edgecolors=col, lw=0.8,
                      label=f"shank {s}" if over else None)
    j = q["pick"]
    a.plot(q["below"][j], q["p2"][j], "*", color="k", ms=14, mfc="none", mew=1.5)
    a.annotate(f"ch{q['cat']['channel'][j]}", (q["below"][j], q["p2"][j]), xytext=(6, 4), textcoords="offset points", fontsize=8)
    a.axhline(floor, color="0.3", ls="--", lw=0.8)
    flag(a, label, f"{label}: pick ch{picks[label]['channel']}, dip {picks[label]['dip']:.4f}"
                   + ("\nNO GROUP THRESHOLD: picked on all valid bins (see picks_kde)" if picks[label]["fallback"]
                      else "\nNO THRESHOLD ON GROUP OR ALL BINS" if picks[label]["no_threshold_channel"] else ""))
for a in ax.ravel()[len(labels):]:
    a.axis("off")
for a in ax[-1]:
    a.set_xlabel("proportion of group below the valley")
for a in ax[:, 0]:
    a.set_ylabel("2nd-mode prominence / max density")
ax[0, 0].legend(fontsize=7, title="filled = over the floor", title_fontsize=7, loc="upper right")
fig.suptitle(f"{args.lfp_dir.name}, stride {args.stride}: per chunk, one point per channel; star = pick "
             f"(largest dip among filled points; red titles: picked on all valid bins instead), dashed = floor {floor}",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(out_dir / f"prominence_vs_below_per_chunk_stride{args.stride}.png")
plt.close(fig)

# Dip and prominence across chunks x channels, channels ordered by shank then depth.
c0 = per[labels[0]]["cat"]
order = np.lexsort((c0["y_um"], c0["shank"]))
fig, ax = plt.subplots(2, 1, figsize=(18, 10), dpi=120, sharex=True)
for a, key, name in zip(ax, ("dip", "p2"), ("dip (smoothed metric)", "2nd-mode prominence / max density")):
    im = a.imshow(np.stack([per[k][key][order] for k in labels]), aspect="auto", cmap="Blues", interpolation="none")
    fig.colorbar(im, ax=a, label=name, pad=0.01)
    a.plot([np.flatnonzero(order == per[k]["pick"])[0] for k in labels], np.arange(len(labels)), "x", color="tab:red", ms=7,
           label="pick")
    for b in np.flatnonzero(np.diff(c0["shank"][order])) + 0.5:
        a.axvline(b, color="k", lw=1)
    a.set_yticks(np.arange(len(labels)), labels, fontsize=7)
    a.legend(fontsize=8, loc="upper right")
starts = np.r_[0, np.flatnonzero(np.diff(c0["shank"][order])) + 1, len(order)]
ax[1].set_xticks((starts[:-1] + starts[1:]) / 2, [f"shank {s} (deep to superficial by y)" for s in c0["shank"][order][starts[:-1]]])
fig.suptitle(f"{args.lfp_dir.name}, stride {args.stride}: theta on non-NREM, non-moving bins, per chunk and channel")
fig.tight_layout()
fig.savefig(out_dir / f"dip_prominence_heatmap_stride{args.stride}.png")
plt.close(fig)

# The picks across chunks.
x = np.arange(len(labels))
P = [picks[k] for k in labels]
y, shk, nt = (np.array([p[k] for p in P]) for k in ("y_um", "shank", "no_threshold_channel"))
fig, ax = plt.subplots(3, 1, figsize=(12, 10), dpi=120, sharex=True)
for s, col in zip(args.shanks, colors):
    ax[0].scatter(x[(shk == s) & ~nt], y[(shk == s) & ~nt], color=col, s=40, label=f"shank {s}")
    ax[0].scatter(x[(shk == s) & nt], y[(shk == s) & nt], color="none", edgecolors=col, s=40)
for xi, p in zip(x, P):
    ax[0].annotate(f"ch{p['channel']}", (xi, p["y_um"]), xytext=(4, 4), textcoords="offset points", fontsize=7,
                   color="tab:red" if p["no_threshold_channel"] else "k")
ax[0].set(ylabel="pick depth y (um)",
          title="pick per chunk (hollow, red label = no channel with a group threshold: picked on all valid bins)")
ax[0].legend(fontsize=8, ncol=4)
ax[1].plot(x, [p["fallback"]["dip_all"] if p["fallback"] else p["dip"] for p in P], "o-", color="k", label="pick")
ax[1].plot(x, [p["runner_up"]["dip"] if p["runner_up"] else np.nan for p in P], "o--", color="0.6", label="runner-up")
ax[1].set(ylabel="dip of the ranking used\n(all valid bins on red chunks)")
ax[1].legend(fontsize=8)
ax[2].plot(x, [p["frac_group_above_valley"] for p in P], "o-", color="tab:red")
ax[2].set(ylabel="share of group above the threshold\n(REM candidates, before duration rules)", ylim=(0, 1))
ax[2].set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
fig.tight_layout()
fig.savefig(out_dir / f"picks_per_chunk_stride{args.stride}.png")
plt.close(fig)

# The picked channel's group distribution, one panel per chunk.
fig, ax = plt.subplots(nr, nc, figsize=(20, 4 * nr), dpi=110, squeeze=False)
for a, label in zip(ax.ravel(), labels):
    q = per[label]; v = q["cat"]["theta_smoothed"][q["pick"]]; p = picks[label]
    p2, valley, grid, dens = kde_stats(v)
    a.hist(v, bins=60, density=True, color="tab:red", alpha=0.5, label="non-NREM, non-moving")
    a.plot(grid, dens, color="k", lw=1.5)
    if np.isfinite(valley) and p2 >= floor:
        a.axvline(valley, color="k", ls="--", lw=1)
    text = f"{label}: ch{p['channel']}, 2nd mode {p2:.3f}"
    if q["fb"] is not None:
        p2a, _, ga, dena = kde_stats(q["fb"]["v"])
        a.hist(q["fb"]["v"], bins=60, density=True, color="0.5", alpha=0.4, label="all valid bins")
        a.plot(ga, dena, color="0.3", lw=1.5)
        a.axvline(p["fallback"]["valley_all"], color="k", ls="--", lw=1)
        a.legend(fontsize=7, loc="upper right")
        text = (f"{label}: ch{p['channel']} FALLBACK, largest all-bins dip ({p['fallback']['dip_all']:.4f}) among\n"
                f"{q['fb']['n_eligible']}/{q['fb']['n_channels']} channels with an all-bins threshold; all-bins 2nd mode "
                f"{p2a:.3f}, {p['frac_group_above_valley']:.0%} of group above")
    elif p["no_threshold_channel"]:
        text += " - NO THRESHOLD ON GROUP OR ALL BINS"
    flag(a, label, text)
    a.tick_params(labelsize=7)
for a in ax.ravel()[len(labels):]:
    a.axis("off")
fig.suptitle(f"{args.lfp_dir.name}: the pick's smoothed Watson theta ratio on each chunk's non-NREM, non-moving bins "
             f"(red); grey = all valid bins, where no channel's group yields a threshold; dashed = the threshold scoring "
             f"uses", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(out_dir / f"picks_kde_stride{args.stride}.png")
plt.close(fig)
print(f"wrote figures to {out_dir}/", flush=True)
