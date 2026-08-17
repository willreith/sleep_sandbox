"""Diagnostic figures for the ripple detection-channel selection, from saved derivatives.

Usage: python plot_ripple_channel.py --seg seg5-148 [--probe ProbeB] [--out-base data/derivatives]
Omitting --probe plots both.

Everything the figures need is already in select_ripple_channel.py's profiles.npz, window_sweep.npz
and channel_selection.yml, so replotting never touches the recording -- the 5 seeds x 200 window
reads behind those files are not repeated. select_ripple_channel.py calls make_figures at the end of
its own run, so there is one copy of the plotting code and the figures always match the arrays that
were saved next to them.
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sleep_sandbox.ripple import select_channels

METHODS = ["raw", "smoothed", "delta_ratio"]
GAP_UM = 100.0   # depth stretches longer than this with no contacts are compressed on the y axis


def make_figures(out_dir, title):
    """Write the four channel-selection figures into out_dir from the arrays saved there."""
    out_dir = Path(out_dir)
    sel_yml = yaml.safe_load((out_dir / "channel_selection.yml").read_text())
    cfg = sel_yml["ripple_config"]
    cs = cfg["channel_selection"]
    passband, delta_band = cfg["band"]["passband"], cs["delta_band"]
    sweep_n, seeds = sorted(cs["sweep"]["n_windows"]), cs["sweep"]["seeds"]

    pr = np.load(out_dir / "profiles.npz")
    sweep_rec = np.load(out_dir / "window_sweep.npz")
    freqs, prod_psd, locs, shank = pr["freqs"], pr["psd"], pr["locations"], pr["shank"]
    shanks = np.unique(shank)
    scores = {k: pr[k] for k in ("raw", "smoothed", "delta", "delta_ratio", "spikiness")}
    selected = {int(s): int(d["channel"]) for s, d in sel_yml["selected"].items()}

    # Depth axis, shared across every panel so profiles are comparable shank to shank. Real depth is
    # piecewise-compressed: a stretch longer than GAP_UM with no contacts on any shank is capped at
    # GAP_UM, and the axis resumes at full scale as soon as contacts do. Inside a populated block the
    # mapping is the identity, so local scale is untouched -- a 50 um feature is the same size on
    # every panel -- and no contact is ever dropped, unlike clipping the axis to a percentile.
    # Compressed stretches are shaded so a break is never silent.
    depths = np.unique(locs[:, 1])
    dy = np.diff(depths)
    axis_um = np.r_[depths[0], depths[0] + np.cumsum(np.minimum(dy, GAP_UM))]
    breaks = depths[:-1][dy > GAP_UM]
    ylim = (axis_um[0] - 20, axis_um[-1] + 20)

    def depth_to_axis(y):
        return np.interp(y, depths, axis_um)

    # Ticks. Every block edge -- the first and last contact depth of each contiguous run -- is
    # always labelled, because those are exactly the isolated and extreme contacts a compressed
    # axis otherwise leaves unplaceable (an isolated contact 1000 um out gets a 100 um-wide slot,
    # and a regular 250 um grid never lands on it). Interior ticks fill in on a 250 um grid, but
    # only where they clear the labels already placed, so nothing collides.
    edges = np.unique(np.r_[depths[0], depths[-1],
                            depths[:-1][dy > GAP_UM], depths[1:][dy > GAP_UM]])
    interior = np.arange(np.ceil(depths[0] / 250) * 250, depths[-1] + 250, 250)
    interior = interior[np.min(np.abs(interior[:, None] - depths[None, :]), axis=1) <= GAP_UM / 2]
    ticks = list(edges)
    for t in interior:
        v = np.interp(t, depths, axis_um)
        if np.abs(v - np.interp(ticks, depths, axis_um)).min() >= 1.5 * GAP_UM:
            ticks.append(float(t))
    ticks = np.unique(ticks)

    def depth_axis(ax):
        for b in breaks:
            ax.axhspan(depth_to_axis(b), depth_to_axis(b) + GAP_UM, color="0.92", zorder=0)
        ax.set_ylim(ylim)
        ax.set_yticks(depth_to_axis(ticks))
        # Small font: block edges either side of a compressed stretch are only GAP_UM apart on the
        # axis however far apart they are in the tissue, so their labels need to fit in that slot.
        ax.set_yticklabels([f"{v:.0f}" for v in ticks], fontsize=7)

    def mark(ax, ch, color="red"):
        ax.axhline(depth_to_axis(locs[ch, 1]), color=color, ls="--", lw=0.8)

    fig, axes = plt.subplots(len(shanks), 3, figsize=(13, 3.2 * len(shanks)), sharey=True,
                             squeeze=False)
    for r, s in enumerate(shanks):
        sel = np.flatnonzero(shank == s)
        y = depth_to_axis(locs[sel, 1])
        for c, (key, label) in enumerate([("raw", f"{passband[0]}-{passband[1]} Hz power"),
                                          ("smoothed", f"depth-smoothed (+/-{cs['smooth_um']:.0f} um)"),
                                          ("delta_ratio", f"ripple / delta {delta_band}")]):
            ax = axes[r, c]
            ax.plot(scores[key][sel], y, "o", ms=3, alpha=0.75)
            if key == "smoothed":
                ax.plot(scores["raw"][sel], y, "o", ms=2, color="grey", alpha=0.4, label="raw")
                ax.legend(fontsize=6)
            mark(ax, select_channels(scores, locs, key)[int(s)])
            depth_axis(ax)
            ax.set_xlabel(label, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"shank {s}\ndepth (um)")
    fig.suptitle(f"{title} — channel scores by depth (dashed = that score's pick); shaded bands are "
                 f"depth stretches >{GAP_UM:.0f} um with no contacts, compressed")
    fig.tight_layout()
    fig.savefig(out_dir / "depth_profiles.png", dpi=150)
    plt.close(fig)

    fmask = (freqs >= 0.25) & (freqs <= 300)
    fig, ax = plt.subplots(figsize=(8, 5))
    for s, ch in selected.items():
        ax.semilogy(freqs[fmask], prod_psd[fmask, ch], lw=0.9,
                    label=f"shank {s} — ch {ch} ({locs[ch, 1]:.0f} um)")
    ax.axvspan(*passband, color="grey", alpha=0.2)
    ax.axvspan(*delta_band, color="steelblue", alpha=0.15)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (NREM mean)")
    ax.set_title(f"{title} — selected channel per shank; a real ripple channel shows a bump in the "
                 "shaded band")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "psd_selected.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(shanks), len(METHODS),
                             figsize=(4 * len(METHODS), 2.8 * len(shanks)),
                             sharex=True, sharey="row", squeeze=False)
    for r, s in enumerate(shanks):
        for c, method in enumerate(METHODS):
            ax = axes[r, c]
            for seed in seeds:
                m = ((sweep_rec["method"] == method) & (sweep_rec["shank"] == s)
                     & (sweep_rec["seed"] == seed))
                o = np.argsort(sweep_rec["n_windows"][m])
                ax.plot(sweep_rec["n_windows"][m][o], sweep_rec["depth_um"][m][o], "o-", ms=4,
                        lw=0.7, alpha=0.7, label=f"seed {seed}" if r == 0 and c == 0 else None)
            ax.set_xscale("log")
            ax.set_xticks(sweep_n)
            ax.set_xticklabels(sweep_n)
            if r == 0:
                ax.set_title(method, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"shank {s}\nselected depth (um)")
            if r == len(shanks) - 1:
                ax.set_xlabel("n windows (10 s each)")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f"{title} — selection stability: flat lines = converged")
    fig.tight_layout()
    fig.savefig(out_dir / "window_sweep.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(shanks), figsize=(3.2 * len(shanks), 5), sharey=True,
                             squeeze=False)
    for c, s in enumerate(shanks):
        sel = np.flatnonzero(shank == s)
        ax = axes[0, c]
        ax.plot(scores["spikiness"][sel], depth_to_axis(locs[sel, 1]), "o", ms=3, alpha=0.75)
        ax.axvline(1.0, color="grey", lw=0.6)
        ax.axvline(2.0, color="red", ls="--", lw=0.8)
        mark(ax, selected[int(s)])
        depth_axis(ax)
        ax.set_xlabel("raw / smoothed")
        ax.set_title(f"shank {s}", fontsize=9)
        if c == 0:
            ax.set_ylabel("depth (um)")
    fig.suptitle(f"{title} — spikiness: isolated peaks (>2, red) are single-contact artefacts, "
                 "not ripple fields")
    fig.tight_layout()
    fig.savefig(out_dir / "spikiness.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg", required=True, help="preprocessed segment range, e.g. 'seg5-148'")
    parser.add_argument("--probe", choices=["ProbeA", "ProbeB"], help="default: both")
    parser.add_argument("--out-base", type=Path, default=repo_root / "data" / "derivatives")
    args = parser.parse_args()

    for probe in ([args.probe] if args.probe else ["ProbeA", "ProbeB"]):
        out_dir = args.out_base / args.seg / probe / "ripples" / "channel_selection"
        make_figures(out_dir, f"{probe} {args.seg}")
        print(f"replotted -> {out_dir}", flush=True)
