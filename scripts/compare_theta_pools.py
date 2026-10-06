"""Compare two scoring folders that differ only in the theta-threshold pool, and plot 3D state spaces.

Usage: python compare_theta_pools.py --a SCORING_A --b SCORING_B [--out-base ...] [--state-space]
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

repo_root = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(repo_root))
from sleep_sandbox.analysis import confusion, cohens_kappa

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=Path, required=True, help="dir holding both scoring folders")
parser.add_argument("--a", default="scoring", help="reference scoring folder")
parser.add_argument("--b", default="scoring-thpool_non_nrem", help="comparison scoring folder")
parser.add_argument("--state-space", action="store_true", help="also write the 3D state-space grids")
args = parser.parse_args()

STATES = ["NREM", "REM", "wake", "nodata"]
COL = {"NREM": "#0033ff", "REM": "red", "wake": "black", "nodata": "0.8"}
SS_COL = {"wake": "black", "NREM": "#0033ff", "REM": "red"}
SS_ALPHA = {"wake": 0.45, "NREM": 0.35, "REM": 0.45}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans"],
                     "font.size": 22, "axes.titlesize": 22, "axes.labelsize": 22,
                     "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18,
                     "figure.titlesize": 28, "axes.linewidth": 1.6, "lines.linewidth": 2.6})


def load(folder):
    ds = sorted((args.base / folder).glob("chunk*"))
    return {d.name: (np.load(d / "result.npz"), json.loads((d / "summary.json").read_text())) for d in ds}


def codes(r):
    c = np.full(len(r["nrem"]), 2, int)          # wake
    c[r["nrem"]] = 0
    c[r["rem"]] = 1
    c[r["nodata"]] = 3
    return c


A, B = load(args.a), load(args.b)
labels = [k for k in A if k in B]
if not labels:
    raise SystemExit(f"no chunks shared between {args.a} and {args.b}")

# --- 1. per-chunk comparison: theta distributions, confusion, hypnograms -------------
fig, ax = plt.subplots(len(labels), 3, figsize=(25, 5.4 * len(labels)), dpi=120)
rows = []
for row, lab in zip(np.atleast_2d(ax), labels):
    ra, sa = A[lab]
    rb, sb = B[lab]
    th, valid = ra["theta_metric"], ~ra["nodata"]
    nrem = ra["sw_metric"] > sa["sw_thresh"]
    pools = {f"{args.a}: non-NREM, still": valid & ~nrem & ~ra["mov"],
             f"{args.b}: all non-NREM": valid & ~nrem}
    a0 = row[0]
    for (name, m), c in zip(pools.items(), ["tab:purple", "tab:orange"]):
        x = th[m][~np.isnan(th[m])]
        a0.hist(x, bins=100, range=(0, 1), density=True, color=c, alpha=0.35)
        g = np.linspace(x.min(), x.max(), 512)
        a0.plot(g, gaussian_kde(x)(g), color=c, label=f"{name} (n={len(x)})")
    for t, c, n in ((sa["th_thresh"], "tab:purple", args.a), (sb["th_thresh"], "tab:orange", args.b)):
        if t is not None and np.isfinite(t):
            a0.axvline(t, color=c, ls="--", lw=1.6, label=f"{n} thresh {t:.3f}")
    a0.set(xlabel="theta, min-maxed", xlim=(0, 1), title=f"{lab}: theta pools and thresholds")
    a0.legend(loc="upper right", fontsize=13)

    ca, cb = codes(ra), codes(rb)
    cm = confusion(ca, cb, 4)
    k = cohens_kappa(ca[(ca < 3) & (cb < 3)], cb[(ca < 3) & (cb < 3)], 3)
    a1 = row[1]
    a1.imshow(cm / cm.sum(), cmap="Blues", vmin=0, vmax=0.6)
    for i in range(4):
        for j in range(4):
            a1.text(j, i, f"{cm[i, j]}\n{100 * cm[i, j] / cm.sum():.1f}%", ha="center", va="center",
                    fontsize=14, color="white" if cm[i, j] / cm.sum() > 0.3 else "black")
    a1.set(xticks=range(4), yticks=range(4), xticklabels=STATES, yticklabels=STATES,
           xlabel=args.b, ylabel=args.a,
           title=f"agreement {100 * np.trace(cm) / cm.sum():.1f}%, kappa {k:.3f} (states only)")
    a1.tick_params(labelsize=16)

    a2 = row[2]
    t_h = ra["times"] / 3600
    for yi, (c, nm) in enumerate(((ca, args.a), (cb, args.b))):
        for si, st in enumerate(STATES):
            m = c == si
            a2.fill_between(t_h, yi, yi + 0.9, where=m, color=COL[st], step="mid", lw=0)
    a2.set(yticks=[0.45, 1.45], yticklabels=[args.a, args.b], xlabel="hours into chunk",
           title=f"{lab}: hypnograms")
    rows.append({"chunk": lab, "kappa": float(k), "agree": float(np.trace(cm) / cm.sum()),
                 "th_a": sa["th_thresh"], "th_b": sb["th_thresh"],
                 "rem_a": float(ra["rem"].mean()), "rem_b": float(rb["rem"].mean()),
                 "nrem_a": float(ra["nrem"].mean()), "nrem_b": float(rb["nrem"].mean())})

fig.suptitle(f"theta-threshold pool: {args.a} (non-NREM, still) vs {args.b} (all non-NREM)")
fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=3.0, w_pad=2.5)
fig.savefig(args.base / "theta_pool_comparison.png")
plt.close(fig)
print(f"wrote {args.base}/theta_pool_comparison.png")

# --- 2. summary across chunks --------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(23, 6.5), dpi=120)
x = np.arange(len(rows))
ax[0].bar(x - 0.2, [r["th_a"] for r in rows], 0.4, color="tab:purple", label=args.a)
ax[0].bar(x + 0.2, [r["th_b"] for r in rows], 0.4, color="tab:orange", label=args.b)
ax[0].set(xticks=x, xticklabels=[r["chunk"][:7] for r in rows], ylabel="th_thresh", title="theta threshold")
ax[0].legend()
for i, k in enumerate(("rem", "nrem")):
    ax[i + 1].bar(x - 0.2, [100 * r[f"{k}_a"] for r in rows], 0.4, color="tab:purple", label=args.a)
    ax[i + 1].bar(x + 0.2, [100 * r[f"{k}_b"] for r in rows], 0.4, color="tab:orange", label=args.b)
    ax[i + 1].set(xticks=x, xticklabels=[r["chunk"][:7] for r in rows], ylabel=f"{k.upper()} %",
                  title=f"{k.upper()} fraction")
    ax[i + 1].legend()
fig.suptitle("theta-threshold pool: per-chunk thresholds and state fractions  |  "
             + "  ".join(f"{r['chunk'][:7]} k={r['kappa']:.2f}" for r in rows), fontsize=20)
fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2.5)
fig.savefig(args.base / "theta_pool_summary.png")
plt.close(fig)
print(f"wrote {args.base}/theta_pool_summary.png")
(args.base / "theta_pool_comparison.json").write_text(json.dumps(rows, indent=2))

# --- 3. 3D state space per scoring ---------------------------------------------------
if args.state_space:
    for name, S in ((args.a, A), (args.b, B)):
        n = len(labels)
        fig = plt.figure(figsize=(7.5 * min(n, 3), 7.0 * -(-n // 3)), dpi=300)
        for i, lab in enumerate(labels):
            r, _ = S[lab]
            a = fig.add_subplot(-(-n // 3), min(n, 3), i + 1, projection="3d")
            keep = ~r["nodata"]
            sel = np.zeros(len(keep), bool)
            sel[::4] = True
            sel = (sel | r["rem"]) & keep
            masks = {"wake": r["wake"], "NREM": r["nrem"], "REM": r["rem"]}
            for st, m in masks.items():
                q = m & sel
                a.scatter(r["theta_metric"][q], r["sw_metric"][q], r["motion_metric"][q],
                          s=6, c=SS_COL[st], alpha=SS_ALPHA[st], lw=0, label=st)
            for j, (st, m) in enumerate(masks.items()):     # fractions over valid bins, not the subsample
                a.text2D(0.01, 0.99 - 0.075 * j, f"{st}  {100 * (m & keep).sum() / keep.sum():.1f}%",
                         transform=a.transAxes, ha="left", va="top", color=SS_COL[st],
                         fontsize=21, fontweight="bold")
            for axis in (a.xaxis, a.yaxis, a.zaxis):
                axis._axinfo["grid"].update(color="0.88", linewidth=0.5)
            a.set(xlabel="Narrowband theta", ylabel="PC1 loading", zlabel="EMG estimate", title=lab,
                  xticks=[0, 0.5, 1], yticks=[0, 0.5, 1], zticks=[0, 0.5, 1])
            a.tick_params(labelsize=18, pad=4)
            for ax_, lp in ((a.xaxis, 14), (a.yaxis, 14), (a.zaxis, 10)):
                ax_.labelpad = lp
            a.set_ylim(1, 0)
            a.set_box_aspect((4, 4, 4.5))
            a.view_init(elev=20, azim=-45)
        fig.suptitle(f"{name}: state space")
        fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=2.5, w_pad=2.0)
        out = args.base / f"state_space_3d_{name}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")
