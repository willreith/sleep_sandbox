"""Theta prominence over the 1/f baseline, per channel and state group, on each chunk of the pre-downsampled zarrs.

Usage: python theta_prominence_scan.py --lfp-dir DIR --chunk N --ref-theta     # the --ref-channel high-theta mask
       python theta_prominence_scan.py --lfp-dir DIR --chunk N --shank S       # mean spectra and prominences
       python theta_prominence_scan.py --lfp-dir DIR --plot                    # figures
Groups come from the chunk's scoring result, so they are the same for every channel: NREM, non-NREM moving, non-NREM
still, still with high theta from that chunk's scoring theta channel, and still with high theta from --ref-channel
(its own threshold, non-NREM non-moving bins with scoring's all-bins fallback). Prominence is the residual of log10
mean power over a log-log fit on 1-50 Hz excluding 4-12 Hz, taken over 5-10 and 6-12 Hz (mean and max).
"""

import json
import argparse
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sleep_sandbox.io import load_input_recording
from sleep_sandbox.analysis import log_spectrogram, theta_ratio, kde_thresh, confusion, cohens_kappa

repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")

parser = argparse.ArgumentParser()
parser.add_argument("--lfp-dir", type=Path, required=True, help="dir of per-shank LFP zarrs")
parser.add_argument("--chunk", type=int, default=None, help="chunk index (position among --scoring's chunk folders)")
parser.add_argument("--shank", type=int, default=None)
parser.add_argument("--ref-theta", action="store_true", help="compute --ref-channel's high-theta mask for --chunk")
parser.add_argument("--chan-theta", type=int, default=None, help="compute this channel's three theta conventions for --chunk")
parser.add_argument("--chan-plot", type=int, default=None, help="--chan-theta figures for this channel")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--scoring", default="scoring_th-lfp_pp_no_cmr-perchunk", help="scoring folder giving the groups")
parser.add_argument("--ref-channel", type=int, default=289, help="aggregate channel for the fifth group's theta")
parser.add_argument("--top", type=int, default=96, help="channels kept in the distribution and heatmap figures")
parser.add_argument("--probe", default="ProbeB")
parser.add_argument("--shanks", type=int, nargs="+", default=[0, 1, 2, 3])
parser.add_argument("--out-base", type=Path, default=repo_root / "data" / "derivatives" / "abcEphys01")
args = parser.parse_args()

shank_tag = f"{args.probe}_shanks{''.join(map(str, args.shanks))}"
scoring_dir = args.out_base / args.lfp_dir.name / shank_tag / args.scoring
chunk_dirs = sorted(p for p in scoring_dir.iterdir() if p.is_dir() and p.name.startswith("chunk"))
out_dir = args.out_base / args.lfp_dir.name / shank_tag / "theta_prominence"
with open(repo_root / "config/sleep_scoring.yml") as f:
    cfg = yaml.safe_load(f)
skw = {k: cfg["spectrogram"][k] for k in ("window_s", "step_s", "freq_min", "freq_max", "n_freq_bins")}
conv_names = cfg["theta"]["report_conventions"]
step, hw = skw["step_s"], skw["window_s"] / 2
half = int(round(cfg["smoothing"]["window_s"] / step)) // 2
kern = np.ones(2 * half + 1)
BANDS = {"5-10 Hz": (5, 10), "6-12 Hz": (6, 12)}


def zarr_path(shank):
    hits = sorted(args.lfp_dir.glob(f"*_{args.probe}_shank{shank}.zarr")) + \
           sorted(args.lfp_dir.glob(f"*_{args.probe}_shank_{shank}.zarr"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{len(hits)} zarrs in {args.lfp_dir} match {args.probe} shank {shank}")
    return hits[0]


def shank_offsets():
    n = [load_input_recording([zarr_path(s)]).get_num_channels() for s in args.shanks]
    return dict(zip(args.shanks, np.cumsum([0] + n[:-1])))


def chunk_spectra(rec, ids, ks):
    """Power spectra on the 1 s grid at grid indices ks, read in 1 h blocks (a 24 h read would need ~10 GB)."""
    fs = rec.get_sampling_frequency()
    out = np.empty((len(ids), skw["n_freq_bins"], len(ks)), np.float32)
    blk = int(round(3600 / step))
    for b0 in range(0, len(ks), blk):
        k = ks[b0:b0 + blk]
        tr = rec.get_traces(start_frame=int(round((k[0] * step - hw) * fs)),
                            end_frame=int(round((k[-1] * step + hw) * fs)), channel_ids=ids)
        for j in range(len(ids)):
            spec, freqs, _ = log_spectrogram(tr[:, j], fs, **skw)
            out[j, :, b0:b0 + len(k)] = spec
        print(f"  block {b0 // blk + 1}/{-(-len(ks) // blk)}", flush=True)
    return out, freqs


def smooth(x, nodata):
    """smooth_norm's NaN-skipping moving average; its min-max follows once the margin bins are dropped."""
    x = np.where(nodata, np.nan, x)
    real = ~np.isnan(x)
    sm = np.convolve(np.where(real, x, 0.0), kern, mode="same") / np.convolve(real.astype(float), kern, mode="same")
    sm[~real] = np.nan
    return sm


def norm01(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


def chunk_grid(cd):
    """(result, summary, grid indices of the chunk's bins, grid indices including the smoothing margin)."""
    r = np.load(cd / "result.npz")
    s = json.loads((cd / "summary.json").read_text())
    k = np.round(r["times_abs"] / step).astype(np.int64)
    return r, s, k, np.arange(k[0] - half, k[-1] + half + 1)


# --- the --ref-channel high-theta mask for one chunk (--chunk --ref-theta) ---------
if args.ref_theta:
    cd = chunk_dirs[args.chunk]
    r, s, k, kk = chunk_grid(cd)
    shank = [sh for sh, off in shank_offsets().items() if off <= args.ref_channel][-1]
    rec = load_input_recording([zarr_path(shank)])
    local = args.ref_channel - shank_offsets()[shank]
    kk = kk[(kk * step >= hw) & (kk * step <= rec.get_num_frames() / rec.get_sampling_frequency() - hw)]
    print(f"{cd.name}: ch{args.ref_channel} = shank {shank} channel {local}, {len(kk)} bins", flush=True)
    spec, freqs = chunk_spectra(rec, rec.channel_ids[[local]], kk)
    nodata = np.zeros(len(kk), bool)
    nodata[np.searchsorted(kk, k[r["nodata"]])] = True
    th = norm01(smooth(theta_ratio(spec[0], freqs, cfg["theta"]["concordant_convention"]), nodata)[np.searchsorted(kk, k)])

    valid = ~r["nodata"]
    pool = valid & ~(r["sw_metric"] > s["sw_thresh"]) & ~r["mov"]
    thr, source = kde_thresh(th[pool], label=f"ch{args.ref_channel}|~nrem&~mov"), "group"
    if not np.isfinite(thr):
        thr, source = kde_thresh(th[valid], label=f"ch{args.ref_channel}|all (fallback)"), "all_bins"
    if not np.isfinite(thr):
        source = "none"
    mask = valid & ~r["mov"] & (th > thr)
    (out_dir / cd.name).mkdir(parents=True, exist_ok=True)
    out = out_dir / cd.name / f"ref_theta_ch{args.ref_channel}.npz"
    np.savez(out, theta=th.astype(np.float32), thresh=thr, source=source, mask=mask)
    print(f"{cd.name}: ch{args.ref_channel} threshold {thr:.3f} from {source}, {mask.sum()} bins in the group\n"
          f"wrote {out}", flush=True)
    raise SystemExit


# --- one channel's three theta conventions for one chunk (--chunk --chan-theta) ----
def theta_populations(r, s):
    """The three populations the theta distributions and thresholds are taken over."""
    valid, nrem = ~r["nodata"], r["sw_metric"] > s["sw_thresh"]
    return {"all valid bins": valid, "non-NREM": valid & ~nrem, "non-NREM, still": valid & ~nrem & ~r["mov"]}


if args.chan_theta is not None:
    cd = chunk_dirs[args.chunk]
    r, s, k, kk = chunk_grid(cd)
    shank = [sh for sh, off in shank_offsets().items() if off <= args.chan_theta][-1]
    rec = load_input_recording([zarr_path(shank)])
    local = args.chan_theta - shank_offsets()[shank]
    kk = kk[(kk * step >= hw) & (kk * step <= rec.get_num_frames() / rec.get_sampling_frequency() - hw)]
    print(f"{cd.name}: ch{args.chan_theta} = shank {shank} channel {local}, {len(kk)} bins", flush=True)
    spec, freqs = chunk_spectra(rec, rec.channel_ids[[local]], kk)
    nodata = np.zeros(len(kk), bool)
    inside = np.searchsorted(kk, k)
    nodata[inside[r["nodata"]]] = True
    th = np.stack([norm01(smooth(theta_ratio(spec[0], freqs, c), nodata)[inside]) for c in conv_names])

    pops = theta_populations(r, s)
    thr = np.array([[kde_thresh(t[m], label=f"ch{args.chan_theta}|{c}|{p}") for p, m in pops.items()]
                    for c, t in zip(conv_names, th)])
    (out_dir / cd.name).mkdir(parents=True, exist_ok=True)
    out = out_dir / cd.name / f"chan_theta_ch{args.chan_theta}.npz"
    np.savez(out, theta=th.astype(np.float32), thresh=thr, conventions=np.array(conv_names),
             populations=np.array(list(pops)), pop_mask=np.stack(list(pops.values())))
    for ci, c in enumerate(conv_names):
        print(f"  {c}: " + ", ".join(f"{p} {thr[ci, pi]:.3f}" if np.isfinite(thr[ci, pi]) else f"{p} none"
                                     for pi, p in enumerate(pops)), flush=True)
    print(f"wrote {out}", flush=True)
    raise SystemExit


# --- mean spectra and prominences for one chunk and shank (--chunk --shank) --------
def group_masks(cd, r, s):
    """The five state groups, all independent of the channel being measured."""
    valid = ~r["nodata"]
    nrem = r["sw_metric"] > s["sw_thresh"]
    ref = np.load(out_dir / cd.name / f"ref_theta_ch{args.ref_channel}.npz")
    return {"NREM": valid & nrem,
            "non-NREM, moving": valid & ~nrem & r["mov"],
            "non-NREM, still": valid & ~nrem & ~r["mov"],
            f"still, high theta (scoring ch{s['theta_channel']})": valid & ~r["mov"] & (r["theta_metric"] > s["th_thresh"]),
            f"still, high theta (ch{args.ref_channel}, {ref['source']})": ref["mask"]}, valid


if not args.plot and args.chan_plot is None:
    cd = chunk_dirs[args.chunk]
    r, s, k, kk = chunk_grid(cd)
    rec = load_input_recording([zarr_path(args.shank)])
    kk = kk[(kk * step >= hw) & (kk * step <= rec.get_num_frames() / rec.get_sampling_frequency() - hw)]
    masks, valid = group_masks(cd, r, s)
    pos = np.array(json.loads((zarr_path(args.shank) / ".zattrs").read_text())["probegroup"]["probes"][0]["contact_positions"])
    print(f"{args.lfp_dir.name} {cd.name} shank {args.shank}: {rec.get_num_channels()} channels, {len(kk)} bins, "
          f"groups " + ", ".join(f"{n}={m.sum()}" for n, m in masks.items()), flush=True)

    spec, freqs = chunk_spectra(rec, rec.channel_ids, kk)
    inside = np.searchsorted(kk, k)
    nodata = np.zeros(len(kk), bool)
    nodata[inside[r["nodata"]]] = True
    nch, ngr = rec.get_num_channels(), len(masks)

    mean_spec = np.full((nch, ngr, skw["n_freq_bins"]), np.nan, np.float32)
    for gi, m in enumerate(masks.values()):
        if m.any():
            mean_spec[:, gi] = spec[:, :, inside[m]].mean(axis=2)

    lf, fit = np.log10(freqs), (freqs >= 1) & (freqs <= 50) & ~((freqs >= 4) & (freqs <= 12))
    prom_mean = np.full((nch, ngr, len(BANDS)), np.nan)
    prom_max = np.full((nch, ngr, len(BANDS)), np.nan)
    for j in range(nch):
        for gi in range(ngr):
            y = np.log10(mean_spec[j, gi])
            if not np.isfinite(y).all():
                continue
            res = y - np.polyval(np.polyfit(lf[fit], y[fit], 1), lf)
            for bi, (lo, hi) in enumerate(BANDS.values()):
                b = (freqs >= lo) & (freqs <= hi)
                prom_mean[j, gi, bi], prom_max[j, gi, bi] = res[b].mean(), res[b].max()

    theta = np.stack([[norm01(smooth(theta_ratio(spec[j], freqs, c), nodata)[inside]) for j in range(nch)]
                      for c in conv_names]).astype(np.float32) if args.chunk == 1 else np.zeros(0, np.float32)
    out = out_dir / cd.name / f"prominence_shank{args.shank}.npz"
    np.savez(out, freqs=freqs, mean_spec=mean_spec, prom_mean=prom_mean, prom_max=prom_max, theta_smoothed=theta,
             channel=shank_offsets()[args.shank] + np.arange(nch), shank=np.full(nch, args.shank), y_um=pos[:, 1],
             groups=np.array(list(masks)), group_n=np.array([m.sum() for m in masks.values()]),
             bands=np.array(list(BANDS)), conventions=np.array(conv_names))
    print(f"wrote {out}", flush=True)
    raise SystemExit


# --- one channel's theta distributions and Watson/Shin agreement (--chan-plot) -----
if args.chan_plot is not None:
    ch = args.chan_plot
    labels = [cd.name for cd in chunk_dirs]
    per = {lab: np.load(out_dir / lab / f"chan_theta_ch{ch}.npz") for lab in labels}
    conv = list(per[labels[0]]["conventions"])
    pops = list(per[labels[0]]["populations"])
    PCOL = {"all valid bins": "0.5", "non-NREM": "tab:orange", "non-NREM, still": "tab:purple"}
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 13, "axes.labelsize": 12,
                         "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10})

    fig, ax = plt.subplots(len(labels), len(conv), figsize=(7 * len(conv), 3.6 * len(labels)), dpi=120)
    for row, lab in zip(ax, labels):
        d = per[lab]
        for a, ci in zip(row, range(len(conv))):
            for pi, pop in enumerate(pops):
                x = d["theta"][ci][d["pop_mask"][pi]]
                x = x[~np.isnan(x)]
                a.hist(x, bins=100, range=(0, 1), density=True, color=PCOL[pop], alpha=0.35)
                g = np.linspace(x.min(), x.max(), 512)
                t = d["thresh"][ci, pi]
                a.plot(g, gaussian_kde(x)(g), color=PCOL[pop],
                       label=f"{pop} (n={len(x)}), thresh {t:.3f}" if np.isfinite(t) else f"{pop} (n={len(x)}), none")
                if np.isfinite(t):
                    a.axvline(t, color=PCOL[pop], ls="--", lw=1.2)
            a.set(xlabel=f"{conv[ci]}, min-maxed", xlim=(0, 1), title=f"{lab}, {conv[ci]}")
            a.legend(loc="upper right")
    fig.suptitle(f"ch{ch}, {args.lfp_dir.name}: theta distribution per population and convention, each chunk. "
                 f"Thresholds are kde_thresh on that population alone.", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_dir / f"theta_ch{ch}_populations_per_chunk.png")
    plt.close(fig)
    print(f"wrote theta_ch{ch}_populations_per_chunk.png", flush=True)

    wi, si, pi = conv.index("watson"), conv.index("shin"), pops.index("non-NREM, still")
    have = [lab for lab in labels if np.isfinite(per[lab]["thresh"][[wi, si], pi]).all()]
    nc = 4
    fig, ax = plt.subplots(-(-len(have) // nc), nc, figsize=(4.6 * nc, 4.6 * -(-len(have) // nc)), dpi=120)
    for a, lab in zip(ax.ravel(), have):
        d = per[lab]
        m = d["pop_mask"][pi] & ~np.isnan(d["theta"][wi]) & ~np.isnan(d["theta"][si])
        w = (d["theta"][wi][m] > d["thresh"][wi, pi]).astype(int)
        s_ = (d["theta"][si][m] > d["thresh"][si, pi]).astype(int)
        cm = confusion(w, s_, 2)
        a.imshow(cm / cm.sum(), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{cm[i, j]}\n{100 * cm[i, j] / cm.sum():.1f}%", ha="center", va="center",
                       color="white" if cm[i, j] / cm.sum() > 0.5 else "black", fontsize=12)
        a.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["shin below", "shin above"],
              yticklabels=["watson below", "watson above"], xlabel="", ylabel="")
        a.set_title(f"{lab}\nagreement {100 * np.trace(cm) / cm.sum():.1f}%, "
                    f"kappa {cohens_kappa(w, s_, 2):.3f}", fontsize=12)
    for a in ax.ravel()[len(have):]:
        a.axis("off")
    fig.suptitle(f"ch{ch}: do the Watson and Shin thresholds call the same bins high-theta? "
                 f"non-NREM, still bins only; each convention uses its own threshold on that population. "
                 f"{len(have)} of {len(labels)} chunks have both.", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / f"theta_ch{ch}_watson_shin_agreement.png")
    plt.close(fig)
    print(f"wrote theta_ch{ch}_watson_shin_agreement.png", flush=True)
    raise SystemExit


# --- figures (--plot) --------------------------------------------------------------
per = {cd.name: [np.load(out_dir / cd.name / f"prominence_shank{s}.npz", allow_pickle=False) for s in args.shanks]
       for cd in chunk_dirs}
labels = list(per)
cat = {k: np.concatenate([x[k] for x in per[labels[1]]]) for k in ("channel", "shank", "y_um", "mean_spec",
                                                                   "prom_mean", "prom_max")}
freqs = per[labels[1]][0]["freqs"]
GROUPS = [list(per[lab][0]["groups"]) for lab in labels]          # group 4 and 5 name their defining channel per chunk
GCOL = ["tab:blue", "tab:orange", "tab:red", "tab:purple", "tab:green"]
order = np.argsort(-cat["prom_mean"][:, 2, 0])                    # 5-10 Hz, non-NREM still, on chunk01
top = order[:args.top]
theta1 = np.concatenate([x["theta_smoothed"] for x in per[labels[1]]], axis=1)   # (convention, channel, bin)
r1, s1, _, _ = chunk_grid(chunk_dirs[1])
POP = {"all valid bins": ~r1["nodata"],
       "non-NREM": ~r1["nodata"] & ~(r1["sw_metric"] > s1["sw_thresh"]),
       "non-NREM, still": ~r1["nodata"] & ~(r1["sw_metric"] > s1["sw_thresh"]) & ~r1["mov"]}
PCOL = {"all valid bins": "0.5", "non-NREM": "tab:orange", "non-NREM, still": "tab:purple"}
conv_names = list(per[labels[1]][0]["conventions"])
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9,
                     "ytick.labelsize": 9, "legend.fontsize": 8})


def spectra_panel(a, j, title):
    for gi, (name, col) in enumerate(zip(GROUPS[1], GCOL)):
        a.semilogx(freqs, np.log10(cat["mean_spec"][j, gi]), color=col, lw=1.2, label=name)
    a.axvspan(5, 10, color="0.85", zorder=0)
    a.set_title(title, fontsize=9)


# 1. every channel's mean spectra on chunk01, ordered by prominence.
nr = -(-len(order) // 16)
fig, ax = plt.subplots(nr, 16, figsize=(56, 3.2 * nr), dpi=100, sharex=True)
for a, j in zip(ax.ravel(), order):
    spectra_panel(a, j, f"ch{cat['channel'][j]} (sh{cat['shank'][j]}, y{cat['y_um'][j]:.0f})\n"
                        f"5-10 Hz prom {cat['prom_mean'][j, 2, 0]:.3f}")
for a in ax.ravel()[len(order):]:
    a.axis("off")
for a in ax[-1]:
    a.set_xlabel("Hz (grey = 5-10 Hz)")
for a in ax[:, 0]:
    a.set_ylabel("log10 mean power")
ax[0, 0].legend(fontsize=7)
fig.suptitle(f"{labels[1]}, {args.lfp_dir.name}: mean spectrum per state group, all {len(order)} channels, ordered by "
             f"5-10 Hz prominence over the 1/f fit on non-NREM, still bins. The high-theta groups are defined by the "
             f"chunk's scoring theta channel and by ch{args.ref_channel}, NOT by the channel in the panel.", fontsize=16)
fig.tight_layout(rect=(0, 0, 1, 0.985))
fig.savefig(out_dir / "spectra_all_channels_chunk01.png")
plt.close(fig)
print("wrote spectra_all_channels_chunk01.png", flush=True)

# 2. the top channels' spectrum and theta distributions on chunk01, in pages.
page = 24
for p0 in range(0, len(top), page):
    sel = top[p0:p0 + page]
    fig, ax = plt.subplots(len(sel), 4, figsize=(22, 3.4 * len(sel)), dpi=100)
    for row, j in zip(ax, sel):
        spectra_panel(row[0], j, f"ch{cat['channel'][j]} (sh{cat['shank'][j]}, y{cat['y_um'][j]:.0f}), "
                                 f"5-10 Hz prom {cat['prom_mean'][j, 2, 0]:.3f} / 6-12 Hz {cat['prom_mean'][j, 2, 1]:.3f}")
        row[0].set(xlabel="Hz (grey = 5-10 Hz)", ylabel="log10 mean power")
        for a, c in zip(row[1:], conv_names):
            v = theta1[conv_names.index(c), j]
            for pop, m in POP.items():
                x = v[m]
                a.hist(x, bins=100, range=(0, 1), density=True, color=PCOL[pop], alpha=0.35)
                g = np.linspace(np.nanmin(x), np.nanmax(x), 512)
                t = kde_thresh(x, label=f"ch{cat['channel'][j]} {c}|{pop}")
                a.plot(g, gaussian_kde(x[~np.isnan(x)])(g), color=PCOL[pop],
                       label=f"{pop}, thresh {t:.3f}" if np.isfinite(t) else f"{pop}, none")
                if np.isfinite(t):
                    a.axvline(t, color=PCOL[pop], ls="--", lw=1)
            a.set(xlabel=f"{c}, min-maxed", xlim=(0, 1), title=f"ch{cat['channel'][j]}, {c}")
            a.legend(loc="upper right")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(f"{labels[1]}: spectra and theta distributions, channels {p0 + 1}-{p0 + len(sel)} of "
                 f"{len(top)} by 5-10 Hz prominence on non-NREM, still bins", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.012 * 24 / len(sel)))
    fig.savefig(out_dir / f"theta_distributions_top{args.top}_chunk01_p{p0 // page + 1}.png")
    plt.close(fig)
    print(f"wrote theta_distributions_top{args.top}_chunk01_p{p0 // page + 1}.png", flush=True)

# 3. prominence per chunk and channel, one panel per group and band.
P = {k: np.stack([np.concatenate([x[k] for x in per[lab]]) for lab in labels]) for k in ("prom_mean", "prom_max")}
ngr = P["prom_mean"].shape[2]
fig, ax = plt.subplots(len(BANDS), ngr, figsize=(6 * ngr, 5.5 * len(BANDS)), dpi=120, sharex=True, sharey=True)
vmax = np.nanpercentile(P["prom_mean"][:, top], 99)
for bi, band in enumerate(BANDS):
    for gi in range(ngr):
        a = ax[bi, gi]
        im = a.imshow(P["prom_mean"][:, top, gi, bi], aspect="auto", cmap="magma", vmin=0, vmax=vmax,
                      interpolation="none")
        a.set_yticks(np.arange(len(labels)), [f"{lab}  [{GROUPS[c][gi].split('(')[-1].rstrip(')')}]"
                                              if gi >= 3 else lab for c, lab in enumerate(labels)], fontsize=7)
        a.set_title(f"{GROUPS[1][gi].split(' (')[0]}, {band}", fontsize=11)
        if gi == ngr - 1:
            fig.colorbar(im, ax=a, label="mean prominence over the 1/f fit", pad=0.01)
for a in ax[-1]:
    a.set_xlabel(f"top {len(top)} channels, ordered by chunk01 5-10 Hz prominence (non-NREM, still)")
fig.suptitle(f"{args.lfp_dir.name}: theta prominence over the 1/f baseline per chunk and channel. The two high-theta "
             f"groups take their theta from the chunk's scoring channel and from ch{args.ref_channel} (named per row), "
             f"not from the column's channel.", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out_dir / f"prominence_heatmap_chunks_top{args.top}.png")
plt.close(fig)
print(f"wrote prominence_heatmap_chunks_top{args.top}.png to {out_dir}/", flush=True)
