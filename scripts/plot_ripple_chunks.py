"""Ripple properties across chunks (days), for every reference arm x merge window of a chunked run.

Usage: python plot_ripple_chunks.py --lfp-dir .../lfp_pp_pilot04 [--experiment abcEphysPilot04]
                                    [--probe ProbeB] [--shanks 0 1 2]

Reads every run under ripples/{none,cmr}/events/*/ written by run_ripples_chunked.py and writes to
ripples/comparison/:
- across_chunks.png / .csv: per shank and chunk, NREM time, rate, duration, fraction at the duration
  floor, spectral peak in band, prominence, burst fraction and the fraction kept at k = 3, one line
  per configuration (colour = reference arm, line style = merge window). A trend across days is a
  drift question; a gap between lines is a parameter question.
- reference_agreement.png / .csv: per shank and chunk, the fraction of one arm's events with an
  overlapping event on the other arm. The arms can pick different channels, so this is agreement in
  time on the same shank, not channel identity.
Events are counted at the n_required the run was configured with, as in plot_ripple_events.py.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sleep_sandbox.ripple import count_corroborating

repo_root = Path(__file__).resolve().parent.parent
BURST_S = 0.2                                   # same definition as plot_ripple_events.py
REF_COLOUR = {"none": "C0", "cmr": "C3"}
MERGE_LS = ["-", "--", ":", "-."]
MERGE_MARKER = ["o", "s", "^", "D"]    # as well as line style: a lone chunk draws no line
sns.set_context("talk")

parser = argparse.ArgumentParser()
parser.add_argument("--lfp-dir", type=Path, required=True, help="dir of per-shank LFP zarrs (only its name is used)")
parser.add_argument("--experiment", default="abcEphysPilot04")
parser.add_argument("--probe", default="ProbeB")
parser.add_argument("--shanks", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--out-base", type=Path, default=None)
args = parser.parse_args()

shank_tag = f"{args.probe}_shanks{''.join(map(str, args.shanks))}"
base = (args.out_base or repo_root / "data" / "derivatives" / args.experiment) / args.lfp_dir.name / shank_tag
out_dir = base / "ripples" / "comparison"
out_dir.mkdir(parents=True, exist_ok=True)

# --- load every (reference, run, chunk) ---
runs = {}   # (reference, merge_ms) -> {chunk label: (events, meta)}
for ref in REF_COLOUR:
    for run_yml in sorted((base / "ripples" / ref / "events").glob("*/run.yml")):
        merge_ms = round(yaml.safe_load(run_yml.read_text())["run_params"]["detection"]["min_inter_event_s"] * 1000, 3)
        assert (ref, merge_ms) not in runs, \
            f"two runs for reference={ref}, merge={merge_ms} ms under {run_yml.parent.parent}; move the stale one"
        runs[(ref, merge_ms)] = {d.name: (dict(np.load(d / "events.npz")), yaml.safe_load((d / "events.yml").read_text()))
                                 for d in sorted(run_yml.parent.glob("chunk*")) if (d / "events.npz").exists()}
        print(f"{ref:>4} merge {merge_ms:g} ms: run {run_yml.parent.name}, {len(runs[(ref, merge_ms)])} chunks", flush=True)
assert runs, f"no runs under {base / 'ripples'}"
merges = sorted({m for _, m in runs})
chunks = sorted({c for r in runs.values() for c in r})

rows = []
for (ref, merge_ms), by_chunk in runs.items():
    for label, (ev, meta) in by_chunk.items():
        fs = meta["fs"]
        det = meta["run_params"]["detection"]
        passband = meta["run_params"]["band"]["passband"]
        n_min_dur = np.ceil(det["min_duration_s"] * fs - 1e-6)
        nrem_min = meta["summary"]["nrem_hours"] * 60
        keep = ev["n_corroborating"] >= meta["summary"]["n_required"]
        for s, info in meta["per_shank"].items():
            m = keep & (ev["shank"] == int(s))
            d, h, c = ev["duration_s"][m], ev["peak_hz"][m], ev["n_corroborating"][ev["shank"] == int(s)]
            iei = np.diff(np.sort(ev["peak_s"][m]))
            rows.append({
                "reference": ref, "merge_ms": merge_ms, "run_id": meta["run_id"], "chunk": label,
                "shank": int(s), "channel": info["channel"], "depth_um": info["depth_um"],
                "nrem_h": nrem_min / 60, "n": int(m.sum()),
                "rate_per_min_nrem": m.sum() / nrem_min if nrem_min else np.nan,
                "duration_ms_median": np.median(d) * 1000 if d.size else np.nan,
                "frac_at_min_duration": np.mean(np.round(d * fs) == n_min_dur) if d.size else np.nan,
                "frac_peak_hz_in_band": np.mean((h >= passband[0]) & (h <= passband[1])) if h.size else np.nan,
                "prominence_median": np.median(ev["prominence"][m]) if m.any() else np.nan,
                "burst_fraction": np.mean(iei < BURST_S) if iei.size else np.nan,
                "frac_kept_k3": np.mean(c >= 3) if c.size and len(info["neighbours"]) >= 3 else np.nan,
            })
df = pd.DataFrame(rows).sort_values(["reference", "merge_ms", "shank", "chunk"])
df.to_csv(out_dir / "across_chunks.csv", index=False)

print("\n--- detection channel per shank, by reference arm ---")
print(df.groupby(["reference", "shank"])[["channel", "depth_um"]].first().to_string(), flush=True)

# --- figure: properties across chunks ---
METRICS = [("nrem_h", "NREM (h)"), ("rate_per_min_nrem", "ripples / min NREM"),
           ("duration_ms_median", "median duration (ms)"), ("frac_at_min_duration", "fraction at\nduration floor"),
           ("frac_peak_hz_in_band", "fraction peak\nin band"), ("prominence_median", "median prominence\n(x 1/f)"),
           ("burst_fraction", f"burst fraction\n(IEI < {BURST_S:g} s)"), ("frac_kept_k3", "fraction kept\nat k = 3")]
x = np.arange(len(chunks))
xlabels = [c.split("_", 1)[1][5:] for c in chunks]   # MM-DD
shanks = sorted(df["shank"].unique())
fig, axes = plt.subplots(len(METRICS), len(shanks), figsize=(6.5 * len(shanks), 3.0 * len(METRICS)),
                         sharex=True, squeeze=False)
for j, s in enumerate(shanks):
    for i, (key, label) in enumerate(METRICS):
        ax = axes[i, j]
        for (ref, merge_ms), g in df[df["shank"] == s].groupby(["reference", "merge_ms"]):
            y = g.set_index("chunk").reindex(chunks)[key]
            k = merges.index(merge_ms)
            ax.plot(x, y, MERGE_MARKER[k % len(MERGE_MARKER)], ls=MERGE_LS[k % len(MERGE_LS)], ms=6,
                    color=REF_COLOUR[ref], label=f"{ref}, merge {merge_ms:g} ms")
        if j == 0:
            ax.set_ylabel(label)
        if i == 0:
            chs = df[df["shank"] == s].groupby("reference")["channel"].first()
            ax.set_title(f"shank {s}  (" + ", ".join(f"{r}: ch{c}" for r, c in chs.items()) + ")")
        ax.grid(alpha=0.3)
for ax in axes[-1]:
    ax.set_xticks(x, xlabels, rotation=45)
    ax.set_xlabel("chunk (start date)")
handles, labels = axes[1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
fig.suptitle(f"{args.experiment} {shank_tag}: ripple properties across chunks")
fig.tight_layout(rect=[0, 0.025, 1, 0.985])
fig.savefig(out_dir / "across_chunks.png", dpi=150)
plt.close(fig)

# --- reference agreement: fraction of one arm's events overlapping the other arm's, same shank ---
agree = []
if all(("none", m) in runs and ("cmr", m) in runs for m in merges):
    for merge_ms in merges:
        for label in chunks:
            if label not in runs[("none", merge_ms)] or label not in runs[("cmr", merge_ms)]:
                continue
            (ev_n, meta_n), (ev_c, _) = runs[("none", merge_ms)][label], runs[("cmr", merge_ms)][label]
            f0, f1 = meta_n["filtered_frames"]
            n_req = meta_n["summary"]["n_required"]
            for s in shanks:
                sub = {}
                for ref, ev in (("none", ev_n), ("cmr", ev_c)):
                    m = (ev["shank"] == s) & (ev["n_corroborating"] >= n_req)
                    sub[ref] = {"start": ev["start"][m] - f0, "end": ev["end"][m] - f0}
                for a, b in (("cmr", "none"), ("none", "cmr")):
                    hit = count_corroborating(sub[a], [sub[b]], f1 - f0)
                    agree.append({"merge_ms": merge_ms, "chunk": label, "shank": s, "events_of": a,
                                  "seen_on": b, "n": int(hit.size),
                                  "frac_overlapping": hit.mean() if hit.size else np.nan})
    ag = pd.DataFrame(agree)
    ag.to_csv(out_dir / "reference_agreement.csv", index=False)

    fig, axes = plt.subplots(1, len(shanks), figsize=(6.5 * len(shanks), 5.5), sharey=True, squeeze=False)
    for j, s in enumerate(shanks):
        ax = axes[0, j]
        for (a, merge_ms), g in ag[ag["shank"] == s].groupby(["events_of", "merge_ms"]):
            y = g.set_index("chunk").reindex(chunks)["frac_overlapping"]
            k = merges.index(merge_ms)
            ax.plot(x, y, MERGE_MARKER[k % len(MERGE_MARKER)], ls=MERGE_LS[k % len(MERGE_LS)], ms=6, color=REF_COLOUR[a],
                    label=f"{a} events seen on {'none' if a == 'cmr' else 'cmr'}, merge {merge_ms:g} ms")
        ax.set_xticks(x, xlabels, rotation=45)
        ax.set_xlabel("chunk (start date)")
        ax.set_title(f"shank {s}")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("fraction of events overlapping\nan event on the other arm")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"{args.experiment} {shank_tag}: agreement between reference arms")
    fig.tight_layout(rect=[0, 0.2, 1, 0.94])
    fig.savefig(out_dir / "reference_agreement.png", dpi=150)
    plt.close(fig)
else:
    print("\nreference agreement skipped: not every merge window has both arms", flush=True)

print(f"\nsaved -> {out_dir}", flush=True)
