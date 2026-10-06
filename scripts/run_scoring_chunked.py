"""Score one midday chunk of the pre-downsampled 16-day recording with fixed channels and PC1 basis.

Channels come from the channel-selection summary and the slow-wave PC1 is projected onto that
selection's basis ('global'); thresholds are fitted per chunk. EMG is the precomputed per-chunk
emg_from_lfp output. Writes result.npz/.yml and qc/ diagnostics to
{out_base}/{lfp_dir.name}/{probe}_shanks{...}/scoring/{chunk label}/.

Usage: python run_scoring_chunked.py --chunk N --lfp-dir .../lfp_pp_no_cmr [--variant A] [--pad-s 60]
                                     [--theta-lfp-dir .../lfp_pp_with_cmr] [--theta-channel N | --theta-picks JSON]
A theta override writes to scoring_th-{theta arm}-ch{N}/ (or -perchunk/ with --theta-picks) instead of scoring/.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sleep_sandbox.io import load_session_timeline, midday_chunks, load_input_recording
from sleep_sandbox.analysis import run_bounds, state_intervals
from sleep_sandbox.scoring import score_recording, summarize_run, plot_summary, save_run

repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")

parser = argparse.ArgumentParser()
parser.add_argument("--chunk", type=int, required=True, help="midday_chunks index")
parser.add_argument("--lfp-dir", type=Path, required=True, help="dir of per-shank LFP zarrs")
parser.add_argument("--probe", default="ProbeB")
parser.add_argument("--shanks", type=int, nargs="+", default=[0, 1, 2, 3])
parser.add_argument("--variant", default="A", help="channel-selection variant whose picks and basis are used")
parser.add_argument("--pad-s", type=float, default=60.0, help="context scored either side of the chunk, then trimmed")
parser.add_argument("--theta-lfp-dir", type=Path, default=None, help="LFP zarr dir for theta (default: --lfp-dir)")
parser.add_argument("--theta-channel", type=int, default=None,
                    help="aggregate theta channel index (default: the theta arm's channel-selection pick)")
parser.add_argument("--theta-picks", type=Path, default=None,
                    help="theta_picks json from select_theta_per_chunk.py; this chunk's pick overrides --theta-channel")
parser.add_argument("--emg-dir", type=Path, default=Path(os.environ["PREPRO_EMG_DIR"]),
                    help="EMG zarr dir; its name locates the compute_emg.py outputs")
parser.add_argument("--csv", type=Path, default=None, help="ephys-paths CSV (default: derived from --probe)")
parser.add_argument("--experiment", default="abcEphys01", help="names the --csv and --out-base defaults")
parser.add_argument("--theta-pool", choices=["group", "non_nrem"], default="group",
                    help="bins the theta threshold is taken on: non-NREM non-moving, or all non-NREM")
parser.add_argument("--emg-shanks", type=int, nargs="+", default=None,
                    help="shanks the EMG was computed over, when they differ from --shanks")
parser.add_argument("--out-base", type=Path, default=None)
args = parser.parse_args()

shank_tag = f"{args.probe}_shanks{''.join(map(str, args.shanks))}"
emg_tag = f"{args.probe}_shanks{''.join(map(str, args.emg_shanks or args.shanks))}"
out_base = args.out_base or repo_root / "data" / "derivatives" / args.experiment
csv_path = args.csv or args.lfp_dir.parent / f"{args.experiment}_{args.probe[-1].upper()}_ephys_paths.csv"
sel_dir = out_base / args.lfp_dir.name / shank_tag / "channel_selection"
basis_path = out_base / args.lfp_dir.name / shank_tag / "pc1_basis" / f"{args.variant}_basis.npz"
emg_res_dir = out_base / args.emg_dir.name / emg_tag


# Same naming rule as select_global_basis.py: either separator, exactly one match.
def zarr_path(lfp_dir, shank):
    hits = sorted(lfp_dir.glob(f"*_{args.probe}_shank{shank}.zarr")) + \
           sorted(lfp_dir.glob(f"*_{args.probe}_shank_{shank}.zarr"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{len(hits)} zarrs in {lfp_dir} match {args.probe} shank {shank}")
    return hits[0]


th_dir = args.theta_lfp_dir or args.lfp_dir
zarr_paths = [zarr_path(args.lfp_dir, s) for s in args.shanks]
th_zarr_paths = [zarr_path(th_dir, s) for s in args.shanks]
with open(repo_root / "config/sleep_scoring.yml") as f:
    cfg = yaml.safe_load(f)

sw_ch = json.loads((sel_dir / f"{args.variant}_summary.json").read_text())["sw_channel"]
th_sel = out_base / th_dir.name / shank_tag / "channel_selection" / f"{args.variant}_summary.json"
b = np.load(basis_path)
basis = {"loading": b[f"ch{sw_ch}_loading"], "mu": b[f"ch{sw_ch}_mu"], "sd": b[f"ch{sw_ch}_sd"]}

rec_start, _ = load_session_timeline(csv_path)
rec = load_input_recording(zarr_paths)
fs = rec.get_sampling_frequency()
c = midday_chunks(rec_start, rec.get_num_frames(), fs)[args.chunk]
th_pick = json.loads(args.theta_picks.read_text())["chunks"][c["label"]] if args.theta_picks else None
th_ch = th_pick["channel"] if th_pick else (args.theta_channel if args.theta_channel is not None
                                            else json.loads(th_sel.read_text())["theta_channel"])
if th_pick:
    scoring_name = f"scoring_th-{th_dir.name}-perchunk"
elif args.theta_lfp_dir is not None or args.theta_channel is not None:
    scoring_name = f"scoring_th-{th_dir.name}-ch{th_ch}"
else:
    scoring_name = "scoring"
if args.theta_pool != "group":
    scoring_name += f"-thpool_{args.theta_pool}"
if th_pick and th_pick["no_threshold_channel"]:
    for stream in (sys.stdout, sys.stderr):
        print(f"{'!' * 100}\n!!! {c['label']}: NO CHANNEL'S non-NREM, non-moving theta yields a threshold; theta ch{th_ch} "
              f"is the largest dip overall and the REM threshold will come from the all-bins fallback\n{'!' * 100}",
              file=stream, flush=True)
pad = int(args.pad_s * fs)
a0, a1 = max(0, c["start_frame"] - pad), min(rec.get_num_frames(), c["end_frame"] + pad)
srec = rec.frame_slice(start_frame=a0, end_frame=a1)
# Both arms are cut from the same upstream recording (1,740,608,750 frames each), so one slice fits both.
th_srec = None if th_dir == args.lfp_dir else load_input_recording(th_zarr_paths).frame_slice(start_frame=a0, end_frame=a1)
print(f"{c['label']}: {c['t_start']} -> {c['t_end']}, scored frames {a0}-{a1} ({(a1 - a0) / fs / 3600:.2f} h); "
      f"sw ch{sw_ch} ({args.lfp_dir.name}), theta ch{th_ch} ({th_dir.name}), basis {basis_path.name}", flush=True)

# EMG files are on absolute recording time; loading all of them gives the padding real EMG too.
emg_files = sorted(emg_res_dir.glob("emg_chunk*.npz"))
emg = [np.load(f) for f in emg_files]
emg_score = np.concatenate([e["emg"] for e in emg])
emg_times = np.concatenate([e["times"] for e in emg]) - a0 / fs

result = score_recording(srec, None, cfg, emg=(emg_score, emg_times),
                         sw_channel=sw_ch, th_channel=th_ch, sw_basis=basis, th_recording=th_srec,
                         theta_exclude_mov=(args.theta_pool == "group"))

# Trim the context: keep bins whose centre lies inside the chunk, on absolute recording time.
t_abs = result["times"] + a0 / fs
keep = (t_abs >= c["start_frame"] / fs) & (t_abs < c["end_frame"] / fs)
n_bins = len(result["times"])
result = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == n_bins else v) for k, v in result.items()}
result["times_abs"] = t_abs[keep]

out_dir = out_base / args.lfp_dir.name / shank_tag / scoring_name / c["label"]
qc = out_dir / "qc"
qc.mkdir(parents=True, exist_ok=True)

valid = ~result["nodata"]
fracs = {s: float(result[s][valid].mean()) for s in ("nrem", "rem", "wake", "qwake", "ma")}
run_summary = {"chunk": c["label"], "n_bins": int(len(valid)), "nodata_s": int((~valid).sum()),
               "state_fractions_of_valid": fracs,
               **{k: float(result[k]) for k in ("sw_thresh", "motion_thresh", "th_thresh")},
               "th_thresh_source": result["th_thresh_source"], "theta_channel": int(th_ch),
               **({"theta_pick": th_pick} if th_pick else {}),
               **summarize_run(result)}
print(json.dumps(run_summary, indent=2), flush=True)
(out_dir / "summary.json").write_text(json.dumps(run_summary, indent=2))

save_run(result, cfg, out_dir / "result.npz", source_dirs={}, provenance={
    "zarr_paths": [str(p) for p in zarr_paths], "rec_start": str(rec_start),
    "chunk": {"label": c["label"], "t_start": str(c["t_start"]), "t_end": str(c["t_end"]),
              "start_frame": int(c["start_frame"]), "end_frame": int(c["end_frame"])},
    "scored_frames": [int(a0), int(a1)], "pad_s": args.pad_s,
    "channel_selection": str(sel_dir / f"{args.variant}_summary.json"), "basis": str(basis_path),
    "basis_mode": "global", "emg_files": [str(f) for f in emg_files],
    "sw_channel": int(sw_ch), "theta_channel": int(th_ch), "theta_zarr_paths": [str(p) for p in th_zarr_paths],
    "theta_channel_source": (str(args.theta_picks) if th_pick else "--theta-channel" if args.theta_channel is not None
                             else str(th_sel))})
plot_summary(result, out_dir=qc)

# --- diagnostics --------------------------------------------------------------
STATE_COLORS = {"nrem": "tab:blue", "rem": "tab:red", "wake": "tab:orange", "nodata": "0.6"}
t_h = (result["times_abs"] - c["start_frame"] / fs) / 3600
title = f"{args.probe} {args.lfp_dir.name} sw ch{sw_ch}, theta {th_dir.name} ch{th_ch}, {c['label']} ({c['t_start']} UTC +h)"
sw, th, mo = result["sw_metric"], result["theta_metric"], result["motion_metric"]
thr = {"sw": result["sw_thresh"], "mo": result["motion_thresh"], "th": result["th_thresh"]}
nrem = sw > thr["sw"]
# The REM decision population: not NREM, not moving (mov = low SW & high motion), as in conditioned_theta_thresh.
rem_pool = valid & ~nrem & ~result["mov"]


def hist(ax, x, t, name, pool, bg=None):
    if bg is not None:
        ax.hist(bg[~np.isnan(bg)], bins=100, density=True, color="0.8", label="all valid bins")
    ax.hist(x[~np.isnan(x)], bins=100, density=True, alpha=0.7, label=pool)
    if np.isfinite(t):
        ax.axvline(t, color="k", ls="--", label=f"thresh = {t:.3f}")
    else:
        ax.set_title("threshold = NaN (unimodal)", color="r")
    ax.set_xlabel(name)
    ax.legend(fontsize=7)


fig, ax = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
hist(ax[0], sw[valid], thr["sw"], "sw_metric", f"all valid bins (n={valid.sum()})")
hist(ax[1], mo[valid & ~nrem], thr["mo"], "motion_metric", f"non-NREM (n={(valid & ~nrem).sum()})")
hist(ax[2], th[rem_pool], thr["th"], "theta_metric", f"non-NREM, non-moving (n={rem_pool.sum()})", bg=th[valid])
if result["th_thresh_source"] == "all_bins":
    ax[2].set_title("threshold from all valid bins (fallback: non-NREM, non-moving had one mode)", color="tab:orange", fontsize=9)
ax[0].set_ylabel("density")
fig.suptitle(f"{title}: threshold distributions")
fig.tight_layout()
fig.savefig(qc / "thresholds.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 5), dpi=150)
for s in ("nrem", "wake", "rem"):
    m = result[s]
    ax[0].scatter(mo[m], sw[m], s=1, alpha=0.2, color=STATE_COLORS[s], label=s, rasterized=True)
    if s != "nrem":
        ax[1].scatter(mo[m], th[m], s=1, alpha=0.2, color=STATE_COLORS[s], label=s, rasterized=True)
ax[0].axhline(thr["sw"], color="k", ls="--")
ax[0].axvline(thr["mo"], color="k", ls=":")
ax[0].set(xlabel="motion_metric", ylabel="sw_metric", title="all valid bins")
ax[1].axhline(thr["th"], color="k", ls="--")
ax[1].axvline(thr["mo"], color="k", ls=":")
ax[1].set(xlabel="motion_metric", ylabel="theta_metric", title="non-NREM bins")
for a in ax:
    a.legend(markerscale=10, fontsize=8)
fig.suptitle(f"{title}: state space (final states)")
fig.tight_layout()
fig.savefig(qc / "state_space.png")
plt.close(fig)


def hypno_axes(axs, sel):
    for a, (x, t, name) in zip(axs, [(sw, thr["sw"], "sw_metric"), (th, thr["th"], "theta_metric"),
                                     (mo, thr["mo"], "motion_metric")]):
        a.plot(t_h[sel], x[sel], lw=0.3, color="k")
        a.axhline(t, color="r", ls="--", lw=0.8)
        a.set_ylabel(name)
    dt_h = cfg["spectrogram"]["step_s"] / 3600
    for i, s in enumerate(("nrem", "rem", "wake", "nodata")):
        ivs = [(t_h[sel][st], d / 3600) for st, d in state_intervals(result[s][sel], np.arange(sel.sum()), 1)]
        axs[3].broken_barh([(t0, max(d, dt_h)) for t0, d in ivs], (i, 0.8), color=STATE_COLORS[s])
    axs[3].set_yticks(np.arange(4) + 0.4, ["nrem", "rem", "wake", "nodata"])
    axs[3].set_xlabel("hours from chunk start")


# Zoom: 1 h centred on the longest REM bout, or the chunk's first hour if there is none.
codes = np.where(result["nodata"], 3, np.argmax(np.stack([result["nrem"], result["rem"], result["wake"]]), axis=0))
starts, stops, vals = run_bounds(codes)
rem_runs = vals == 1
if rem_runs.any():
    i = np.argmax(np.where(rem_runs, stops - starts, -1))
    mid = t_h[(starts[i] + stops[i] - 1) // 2]
else:
    mid = 0.5
zoom = (t_h >= mid - 0.5) & (t_h < mid + 0.5)

fig, ax = plt.subplots(8, 1, figsize=(16, 16), dpi=150, gridspec_kw={"height_ratios": [1, 1, 1, 0.7] * 2})
hypno_axes(ax[:4], np.ones(len(t_h), dtype=bool))
hypno_axes(ax[4:], zoom)
ax[4].set_title(f"zoom: 1 h around {'longest REM bout' if rem_runs.any() else 'chunk start (no REM)'}")
fig.suptitle(f"{title}: metrics and hypnogram")
fig.tight_layout()
fig.savefig(qc / "hypnogram.png")
plt.close(fig)

dur = cfg["duration_criteria"]
fig, ax = plt.subplots(1, 4, figsize=(18, 4), dpi=150)
for a, (code, s) in zip(ax, enumerate(("nrem", "rem", "wake"))):
    d = (stops - starts)[vals == code] * cfg["spectrogram"]["step_s"]
    if len(d):
        a.hist(d, bins=np.logspace(0, np.log10(d.max() + 1), 40), color=STATE_COLORS[s])
    a.set_xscale("log")
    a.axvline(dur["min_state_s"], color="k", ls="--", label=f"min_state_s = {dur['min_state_s']}")
    a.axvline(dur["microarousal_max_s"], color="k", ls=":", label=f"microarousal_max_s = {dur['microarousal_max_s']}")
    a.set(xlabel=f"{s} bout duration (s)", title=f"{s}: {len(d)} bouts, median {np.median(d) if len(d) else np.nan:.0f} s")
    a.legend(fontsize=7)
ax[0].set_ylabel("bouts")
names = ["nrem", "rem", "wake", "qwake", "ma", "nodata"]
vals_frac = [result[s].mean() for s in names]
ax[3].bar(names, vals_frac, color=[STATE_COLORS.get(s, "0.4") for s in names])
ax[3].set(ylabel="fraction of chunk bins", title="state fractions (qwake, ma are subsets of wake)")
fig.suptitle(f"{title}: bouts and fractions")
fig.tight_layout()
fig.savefig(qc / "bouts.png")
plt.close(fig)

print(f"wrote {out_dir}/", flush=True)
