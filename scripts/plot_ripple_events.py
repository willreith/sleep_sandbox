"""Summary statistics and diagnostic figures for detected ripple events.

Usage: python plot_ripple_events.py --seg seg5-148 [--run-id-a HASH] [--run-id-b HASH]
                                    [--out-base data/derivatives]
       python plot_ripple_events.py --events-dir .../ripples/{reference}/events/{run_id}/{chunk}
                                    --out-dir .../ripples/{reference}/summary/{run_id}/{chunk}

Reads both probes' events.npz plus the sleep-scoring result.npz and writes figures and a stats.yml
into {out_base}/{seg}/ripples_summary/. Cross-probe by design: ProbeA is PFC and carries no ripple
field, so it is the empirical false-positive floor and is only informative drawn on the same axes as
ProbeB -- it is plotted faint throughout, ProbeB solid. Because the product spans both probes it
cannot live under either probe's run directory; sources.yml records which run_id each probe
contributed. Defaults to the most recently created run per probe.

--events-dir reads one run_ripples_chunked.py output instead: a single probe, its states from the
scoring result named in events.yml, times on the absolute recording clock (times_abs). With no
ProbeA there is no control, so the specificity block is left out.
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROBES = ["ProbeA", "ProbeB"]
STYLE = {"ProbeA": dict(alpha=0.30, ls="--", lw=1.0),      # negative control, deliberately faint
         "ProbeB": dict(alpha=0.95, ls="-", lw=1.6)}
BURST_S = 0.2          # IEI below this counts as within-burst (ripple doublets/trains)
LAG_MS = 100           # half-width of the cross-shank lag histogram


def load_run(out_base, seg, probe, run_id):
    root = out_base / seg / probe / "ripples" / "events"
    runs = yaml.safe_load((root / "runs.yml").read_text())
    rid = run_id or max(runs, key=lambda k: runs[k]["created"])
    ev = dict(np.load(root / rid / "events.npz"))
    meta = yaml.safe_load((root / rid / "events.yml").read_text())
    return rid, ev, meta


def bouts(mask, times):
    """(start, end) times of each contiguous True run in a per-epoch mask."""
    d = np.diff(mask.astype(np.int8))
    s = np.flatnonzero(d == 1) + 1
    e = np.flatnonzero(d == -1) + 1
    if mask[0]:
        s = np.r_[0, s]
    if mask[-1]:
        e = np.r_[e, mask.size]
    return times[s], times[e - 1]


def nearest_lag(a, b):
    """Signed time to the nearest element of b for each element of a (b - a). Both sorted."""
    if b.size == 0:
        return np.array([])
    if b.size == 1:
        return b[0] - a
    i = np.searchsorted(b, a).clip(1, b.size - 1)
    left, right = b[i - 1], b[i]
    return np.where(a - left <= right - a, left - a, right - a)


parser = argparse.ArgumentParser()
parser.add_argument("--seg", help="preprocessed segment range, e.g. 'seg5-148'")
parser.add_argument("--run-id-a", help="ProbeA run_id; default is its most recent run")
parser.add_argument("--run-id-b", help="ProbeB run_id; default is its most recent run")
parser.add_argument("--out-base", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "derivatives")
parser.add_argument("--events-dir", type=Path, help="one chunk of a run_ripples_chunked.py run")
parser.add_argument("--out-dir", type=Path, help="with --events-dir: where figures and stats go")
args = parser.parse_args()
assert (args.seg is None) != (args.events_dir is None), "give exactly one of --seg or --events-dir"

E, META, RID, STATE, EVENTS_DIR = {}, {}, {}, {}, {}
if args.events_dir:
    out_dir = args.out_dir
    meta = yaml.safe_load((args.events_dir / "events.yml").read_text())
    PROBES = [meta["run_params"]["probe"]]
    p = PROBES[0]
    RID[p], E[p], META[p], EVENTS_DIR[p] = (meta["run_id"], dict(np.load(args.events_dir / "events.npz")),
                                            meta, args.events_dir)
    r = np.load(meta["sources"]["states"])
    # nrem already excludes nodata; times_abs is the clock the chunked events are on
    STATE[p] = {"times": r["times_abs"], "nrem": r["nrem"]}
else:
    out_dir = args.out_base / args.seg / "ripples_summary"
    for probe, rid_arg in zip(PROBES, [args.run_id_a, args.run_id_b]):
        RID[probe], E[probe], META[probe] = load_run(args.out_base, args.seg, probe, rid_arg)
        EVENTS_DIR[probe] = args.out_base / args.seg / probe / "ripples" / "events" / RID[probe]
        variant = META[probe]["run_params"]["variant"]
        STATE[probe] = np.load(args.out_base / args.seg / probe / variant / "result.npz")
out_dir.mkdir(parents=True, exist_ok=True)
for p in PROBES:
    print(f"{p}: run {RID[p]}, {E[p]['start'].size} events", flush=True)
REF = PROBES[-1]   # the probe whose run supplies band/detection parameters

HAS_HZ = all("peak_hz" in E[p] for p in PROBES)
HAS_PROM = all("prominence" in E[p] for p in PROBES)
passband = META[REF]["run_params"]["band"]["passband"]
det = META[REF]["run_params"]["detection"]
fs = float(E[REF]["peak"][-1] / E[REF]["peak_s"][-1])
# Durations are whole samples, so the floor is the first sample count at/above min_duration_s (38
# samples = 30.4 ms at 1250 Hz, not 30 ms); comparing seconds against 0.030 never matched.
n_min_dur = np.ceil(det["min_duration_s"] * fs - 1e-6)
n_max_dur = np.floor(det["max_duration_s"] * fs + 1e-6)
shanks = {p: np.unique(E[p]["shank"]) for p in PROBES}
# Separate colour families, not just separate alpha: sharing the tab10 cycle across probes makes
# "orange" mean shank 1 on both, so a reader has to decode linestyle to know which probe they are
# looking at. ProbeB takes the colours, the ProbeA control takes greys.
colour = {"ProbeB": {int(s): f"C{i}" for i, s in enumerate(shanks.get("ProbeB", []))},
          "ProbeA": {int(s): str(0.25 + 0.15 * i) for i, s in enumerate(shanks.get("ProbeA", []))}}
SERIES = [(p, int(s)) for p in PROBES for s in shanks[p]]
handles = [plt.Line2D([], [], color=colour[p][int(s)], label=f"{p} sh{int(s)}", **STYLE[p])
           for p in PROBES for s in shanks[p]]


def legend(fig):
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.01))


def ridgeline(ax, edges, values_of, smooth=3):
    """One offset density curve per (probe, shank), bottom to top. Seven step-histograms on shared
    axes overlap into illegibility; offsetting them keeps every distribution and its shape while
    removing the overlap entirely. Each row is scaled to its own peak, so rows show shape, not
    relative count -- rates live in figure 1."""
    centres = (edges[:-1] + edges[1:]) / 2
    ticks, labels = [], []
    for i, (probe, sh) in enumerate(SERIES):
        v = values_of(probe, sh)
        h = np.histogram(v, edges, density=True)[0]
        if smooth > 1:
            h = np.convolve(h, np.ones(smooth) / smooth, mode="same")
        if h.max():
            h = h / h.max()
        col = colour[probe][sh]
        ax.fill_between(centres, i, i + h, color=col, alpha=0.45 * STYLE[probe]["alpha"] + 0.1, lw=0)
        ax.plot(centres, i + h, color=col, lw=STYLE[probe]["lw"], alpha=STYLE[probe]["alpha"],
                ls=STYLE[probe]["ls"])
        ticks.append(i)
        labels.append(f"{probe[-1]} sh{sh}")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(-0.15, len(SERIES) + 0.15)


def per_shank(probe):
    for s in shanks[probe]:
        yield int(s), E[probe]["shank"] == s


# NREM-time coordinate: cumulative NREM seconds at each event's peak. Rate against this axis has a
# constant denominator, so a dip is fewer ripples per unit sleep rather than simply less sleep.
nrem_time = {}
for p in PROBES:
    t, nrem = STATE[p]["times"], STATE[p]["nrem"]
    dt = float(np.median(np.diff(t)))
    cum = np.cumsum(nrem) * dt
    nrem_time[p] = np.interp(E[p]["peak_s"], t, cum)

stats = {"sources": {p: {"run_id": RID[p], "n_events": int(E[p]["start"].size)} for p in PROBES},
         "per_probe": {}}

# =============================== figure 1: rates ===============================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
ax = axes[0]
width = 0.35
for k, p in enumerate(PROBES):
    nrem_min = STATE[p]["nrem"].sum() * float(np.median(np.diff(STATE[p]["times"]))) / 60
    for s, m in per_shank(p):
        ax.bar(s + (k - 0.5) * width, m.sum() / nrem_min, width,
               color=colour[p][s], alpha=STYLE[p]["alpha"],
               edgecolor="k", lw=0.5)
ax.set_xlabel("shank")
ax.set_ylabel("ripples / min NREM")
ax.set_title("Rate per shank (grey = ProbeA, PFC control)", fontsize=9)

ax = axes[1]
bin_min = 10
for p in PROBES:
    for s, m in per_shank(p):
        v = nrem_time[p][m] / 60
        edges = np.arange(0, v.max() + bin_min, bin_min) if v.size else np.array([0, bin_min])
        cnt, _ = np.histogram(v, edges)
        ax.plot(edges[:-1] + bin_min / 2, cnt / bin_min, color=colour[p][s], **STYLE[p])
ax.set_xlabel(f"cumulative NREM time (min)")
ax.set_ylabel("ripples / min NREM")
ax.set_title(f"Rate through sleep ({bin_min} min NREM bins)", fontsize=9)

ax = axes[2]
for p in PROBES:
    t, nrem = STATE[p]["times"], STATE[p]["nrem"]
    dt = float(np.median(np.diff(t)))
    hours = np.arange(np.floor(t.min() / 3600), t.max() / 3600 + 1)
    nrem_per_h = np.histogram(t[nrem] / 3600, hours)[0] * dt / 60
    for s, m in per_shank(p):
        cnt = np.histogram(E[p]["peak_s"][m] / 3600, hours)[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            ax.plot(hours[:-1] + 0.5, np.where(nrem_per_h > 1, cnt / nrem_per_h, np.nan),
                    color=colour[p][s], **STYLE[p])
ax_t = ax.twinx()
ax_t.fill_between(hours[:-1] + 0.5, nrem_per_h, color="grey", alpha=0.15, step="mid")
ax_t.set_ylabel("NREM min / h (grey)", fontsize=8)
ax.set_xlabel("recording hour")
ax.set_ylabel("ripples / min NREM")
ax.set_title("Rate by wall-clock hour, normalised by NREM in that hour", fontsize=9)
legend(fig)
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(out_dir / "rates.png", dpi=200)
plt.close(fig)

# ========================= figure 2: event properties =========================
ncol = 3 + int(HAS_HZ) + int(HAS_PROM)
fig, axes = plt.subplots(1, ncol, figsize=(4.1 * ncol, 4.8))
ax = axes[0]
# x clipped to 150 ms: the longest event anywhere is 131 ms, so the range out to the 200 ms bound
# is empty and only compresses the part that has data. The bound is annotated instead of drawn.
ridgeline(ax, np.arange(det["min_duration_s"] * 1000, 152, 2),
          lambda pr, sh: E[pr]["duration_s"][E[pr]["shank"] == sh] * 1000)
ax.axvline(det["min_duration_s"] * 1000, color="red", ls=":", lw=1)
ax.set_xlabel("duration (ms)")
ax.set_title(f"Duration — red = {det['min_duration_s'] * 1000:.0f} ms floor "
             f"(ceiling {det['max_duration_s'] * 1000:.0f} ms, never reached)", fontsize=8)

ax = axes[1]
# Survival function, not a histogram: S(z) = fraction of events exceeding z, from sorted values and
# their ranks. No bins, so the tail stays a smooth curve where a histogram degenerates into isolated
# one-event bars (and, on a log axis, gaps where a bin holds zero). Each curve ends at that shank's
# largest event, making maxima directly comparable, and all curves descend monotonically from 1 so
# they fan out instead of tangling.
for p in PROBES:
    for s, m in per_shank(p):
        v = np.sort(E[p]["peak_z"][m])
        ax.plot(v, 1 - np.arange(v.size) / v.size, drawstyle="steps-post",
                color=colour[p][s], **STYLE[p])
ax.set_yscale("log")
# x linear to 20, log above: every ProbeB curve lives inside z=5-20 and was squeezed into the left
# quarter by ProbeA's tail out to 75. Compressing the tail rather than cutting it gives the bulk
# room while still showing where the outlier curves end.
ax.set_xscale("symlog", linthresh=20, linscale=2.0)
ax.set_xticks([5, 10, 15, 20, 30, 50, 75])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.axvline(20, color="grey", ls=":", lw=0.7)
ax.set_xlim(4.5, 80)
ax.set_xlabel("peak z")
ax.set_ylabel("fraction of events exceeding")
ax.set_title("Amplitude survival (log above z=20) — each curve\nends at that shank's largest event",
             fontsize=9)

ax = axes[2]
# Raw points dropped to a faint backdrop; the binned means carry the trend. Bins with fewer than
# MIN_PER_BIN events are dropped rather than plotted as noise.
MIN_PER_BIN = 8
dur_bins = np.arange(30, 145, 10)
for p in PROBES:
    for s, m in per_shank(p):
        d, z = E[p]["duration_s"][m] * 1000, E[p]["peak_z"][m]
        ax.plot(d, z, ".", ms=1.5, color=colour[p][s], alpha=STYLE[p]["alpha"] * 0.18)
        idx = np.digitize(d, dur_bins) - 1
        cx, cy = [], []
        for b in range(len(dur_bins) - 1):
            sel = idx == b
            if sel.sum() >= MIN_PER_BIN:
                cx.append((dur_bins[b] + dur_bins[b + 1]) / 2)
                cy.append(z[sel].mean())
        ax.plot(cx, cy, "o-", ms=7, color=colour[p][s], alpha=STYLE[p]["alpha"],
                lw=STYLE[p]["lw"], mec="k", mew=0.5, zorder=3)
ax.set_xlabel("duration (ms)")
ax.set_ylabel("peak z")
# Linear to 20, logarithmic above: the artefact tail runs to peak_z 75 while the bulk sits under 15,
# so a linear axis spends three quarters of its height on a handful of points. symlog compresses
# that tail rather than clipping it -- every event is still drawn, above the dotted line.
ax.set_yscale("symlog", linthresh=20, linscale=1.2)
ax.set_yticks([5, 10, 15, 20, 30, 50, 75])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.axhline(20, color="grey", ls=":", lw=0.7)
ax.set_title("Duration vs amplitude (log above z=20)", fontsize=9)

if HAS_HZ:
    ax = axes[3]
    ridgeline(ax, np.arange(100, 262, 4),
              lambda pr, sh: E[pr]["peak_hz"][E[pr]["shank"] == sh])
    ax.axvspan(*passband, color="grey", alpha=0.15, zorder=0)
    ax.set_xlabel("peak frequency (Hz)")
    ax.set_title("Spectral peak — a pile at 100 Hz is 1/f, i.e. no peak", fontsize=9)

if HAS_PROM:
    ax = axes[3 + int(HAS_HZ)]
    for p in PROBES:
        for s, m in per_shank(p):
            ax.hist(E[p]["prominence"][m], np.logspace(-0.3, 2.5, 45), histtype="step",
                    density=True, color=colour[p][s], alpha=STYLE[p]["alpha"],
                    ls=STYLE[p]["ls"], lw=STYLE[p]["lw"])
    ax.axvline(1.0, color="red", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("peak prominence (x 1/f background)")
    ax.set_title("Prominence — 1 (red) is a pure 1/f decay,\nno oscillation at all", fontsize=9)
legend(fig)
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(out_dir / "event_properties.png", dpi=200)
plt.close(fig)

# ======================= figure 3: temporal structure =======================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
edges = np.logspace(-2, 3, 60)
for p in PROBES:
    for s, m in per_shank(p):
        iei = np.diff(np.sort(E[p]["peak_s"][m]))
        if iei.size:
            ax.hist(iei, edges, histtype="step", density=True, color=colour[p][s],
                    alpha=STYLE[p]["alpha"], ls=STYLE[p]["ls"], lw=STYLE[p]["lw"])
ax.axvline(BURST_S, color="red", ls=":", lw=1)
ax.set_xscale("log")
ax.set_xlabel("inter-event interval (s)")
ax.set_ylabel("density")
ax.set_title(f"IEI — mass left of {BURST_S:g}s (red) is burst structure", fontsize=9)

# Latency from NREM-bout onset, corrected for availability: without dividing by how much NREM time
# actually reaches each latency, long latencies look rare purely because few bouts last that long.
ax = axes[1]
lat_edges = np.arange(0, 600, 30)
for p in PROBES:
    bs, be = bouts(STATE[p]["nrem"], STATE[p]["times"])
    dur = be - bs
    avail = np.array([np.clip(dur - lo, 0, 30).sum() for lo in lat_edges[:-1]]) / 60
    for s, m in per_shank(p):
        t = E[p]["peak_s"][m]
        i = np.searchsorted(bs, t, side="right") - 1
        lat = t - bs[i.clip(0)]
        cnt = np.histogram(lat, lat_edges)[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            ax.plot(lat_edges[:-1] + 15, np.where(avail > 0.5, cnt / avail, np.nan),
                    color=colour[p][s], **STYLE[p])
ax.set_xlabel("time since NREM bout onset (s)")
ax.set_ylabel("ripples / min available")
ax.set_title("Rate vs bout latency (availability-corrected)", fontsize=9)
legend(fig)
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(out_dir / "temporal.png", dpi=200)
plt.close(fig)

# ========================= figure 4: corroboration =========================
ncol = 3 if HAS_HZ else 2
fig, axes = plt.subplots(1, ncol, figsize=(4.6 * ncol, 4.2))
ks = np.arange(0, 4)
ax = axes[0]
for p in PROBES:
    for s, m in per_shank(p):
        c = E[p]["n_corroborating"][m]
        kmax = len(META[p]["per_shank"][s]["neighbours"])
        frac = [(c >= k).sum() / c.size if k <= kmax else np.nan for k in ks]
        ax.plot(ks, frac, "o", ms=4, color=colour[p][s], **STYLE[p])
ax.set_xticks(ks)
ax.set_xlabel("n_required (neighbours corroborating)")
ax.set_ylabel("fraction of candidates retained")
ax.set_title("Neighbour sweep", fontsize=9)

ax = axes[1]
for p in PROBES:
    for s, m in per_shank(p):
        c, z = E[p]["n_corroborating"][m], E[p]["peak_z"][m]
        vals = [np.median(z[c == k]) if (c == k).any() else np.nan for k in ks]
        ax.plot(ks, vals, "o", ms=4, color=colour[p][s], **STYLE[p])
ax.set_xticks(ks)
ax.set_xlabel("n_corroborating")
ax.set_ylabel("median peak z")
ax.set_title("Amplitude vs corroboration — rising = criterion tracks a real field", fontsize=9)

if HAS_HZ:
    ax = axes[2]
    for p in PROBES:
        for s, m in per_shank(p):
            c, h = E[p]["n_corroborating"][m], E[p]["peak_hz"][m]
            inb = (h >= passband[0]) & (h <= passband[1])
            vals = [inb[c == k].mean() if (c == k).any() else np.nan for k in ks]
            ax.plot(ks, vals, "o", ms=4, color=colour[p][s], **STYLE[p])
    ax.set_xticks(ks)
    ax.set_xlabel("n_corroborating")
    ax.set_ylabel(f"fraction with peak in {passband[0]}-{passband[1]} Hz")
    ax.set_title("Spectral quality vs corroboration", fontsize=9)
legend(fig)
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(out_dir / "corroboration.png", dpi=200)
plt.close(fig)

# ========================= figure 5: cross-shank =========================
fig, axes = plt.subplots(1, len(PROBES) + 1, figsize=(5 * (len(PROBES) + 1), 4.2))
for k, p in enumerate(PROBES):
    ax = axes[k]
    sh = shanks[p]
    M = np.full((len(sh), len(sh)), np.nan)
    for i, a in enumerate(sh):
        pa = np.sort(E[p]["peak_s"][E[p]["shank"] == a])
        for j, b in enumerate(sh):
            if a == b or pa.size == 0:
                continue
            pb = np.sort(E[p]["peak_s"][E[p]["shank"] == b])
            M[i, j] = np.mean(np.abs(nearest_lag(pa, pb)) <= LAG_MS / 1000)
    im = ax.imshow(M, vmin=0, vmax=max(0.05, np.nanmax(M)), cmap="viridis")
    ax.set_xticks(range(len(sh)), [str(int(s)) for s in sh])
    ax.set_yticks(range(len(sh)), [str(int(s)) for s in sh])
    for i in range(len(sh)):
        for j in range(len(sh)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="w" if M[i, j] < 0.6 * np.nanmax(M) else "k")
    ax.set_xlabel("also detected on shank")
    ax.set_ylabel("events on shank")
    ax.set_title(f"{p} — coincidence within {LAG_MS} ms", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

ax = axes[-1]
for p in PROBES:
    sh = shanks[p]
    for i, a in enumerate(sh):
        pa = np.sort(E[p]["peak_s"][E[p]["shank"] == a])
        for b in sh[i + 1:]:
            pb = np.sort(E[p]["peak_s"][E[p]["shank"] == b])
            lag = nearest_lag(pa, pb) * 1000
            lag = lag[np.abs(lag) <= LAG_MS]
            if lag.size > 20:
                ax.hist(lag, np.arange(-LAG_MS, LAG_MS + 4, 4), histtype="step", density=True,
                        alpha=STYLE[p]["alpha"], ls=STYLE[p]["ls"], lw=STYLE[p]["lw"],
                        label=f"{p[-1]} {a}-{b}")
ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("lag to nearest event on other shank (ms)")
ax.set_ylabel("density")
ax.set_title("Lag — tight peak at 0 = one shared field;\nflat = independent coincidence", fontsize=9)
ax.legend(fontsize=6)
legend(fig)
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(out_dir / "cross_shank.png", dpi=200)
plt.close(fig)

# ============================== stats.yml ==============================
for p in PROBES:
    dt = float(np.median(np.diff(STATE[p]["times"])))
    nrem_min = STATE[p]["nrem"].sum() * dt / 60
    bs, be = bouts(STATE[p]["nrem"], STATE[p]["times"])
    block = {"run_id": RID[p], "nrem_min": round(float(nrem_min), 1),
             "n_nrem_bouts": int(bs.size),
             "nrem_bout_median_s": round(float(np.median(be - bs)), 1), "shanks": {}}
    for s, m in per_shank(p):
        d, z, c = E[p]["duration_s"][m], E[p]["peak_z"][m], E[p]["n_corroborating"][m]
        iei = np.diff(np.sort(E[p]["peak_s"][m]))
        hourly = np.histogram(E[p]["peak_s"][m] / 3600,
                              np.arange(np.floor(STATE[p]["times"].min() / 3600),
                                        STATE[p]["times"].max() / 3600 + 1))[0]
        row = {
            "n": int(m.sum()),
            "rate_per_min_nrem": round(float(m.sum() / nrem_min), 4),
            "duration_ms_median": round(float(np.median(d) * 1000), 1),
            "frac_at_min_duration": round(float(np.mean(np.round(d * fs) == n_min_dur)), 3),
            "frac_at_max_duration": round(float(np.mean(np.round(d * fs) == n_max_dur)), 3),
            "peak_z_median": round(float(np.median(z)), 2),
            "peak_z_max": round(float(z.max()), 1),
            "amplitude_outlier_ratio": round(float(z.max() / np.median(z)), 2),
            "iei_s_median": round(float(np.median(iei)), 2) if iei.size else None,
            "burst_fraction": round(float(np.mean(iei < BURST_S)), 3) if iei.size else None,
            "hourly_rate_cv": round(float(hourly.std() / hourly.mean()), 3) if hourly.mean() else None,
            "corroboration_sweep": {int(k): int((c >= k).sum()) for k in ks},
        }
        if HAS_HZ:
            h = E[p]["peak_hz"][m]
            row["peak_hz_median"] = round(float(np.median(h)), 1)
            row["frac_peak_hz_in_band"] = round(float(np.mean((h >= passband[0])
                                                             & (h <= passband[1]))), 3)
            row["frac_peak_hz_at_floor"] = round(float(np.mean(h <= h.min() + 5)), 3)
        if HAS_PROM:
            pr = E[p]["prominence"][m]
            row["prominence_median"] = round(float(np.median(pr)), 2)
            row["frac_prominence_gt2"] = round(float(np.mean(pr > 2)), 3)
        block["shanks"][s] = row
    stats["per_probe"][p] = block

if {"ProbeA", "ProbeB"} <= set(PROBES):
    best_b = max(stats["per_probe"]["ProbeB"]["shanks"].values(), key=lambda r: r["rate_per_min_nrem"])
    worst_a = max(stats["per_probe"]["ProbeA"]["shanks"].values(), key=lambda r: r["rate_per_min_nrem"])
    stats["specificity"] = {
        "note": "ProbeA is PFC and should carry no ripples, so its rate is an empirical false-positive "
                "floor for the current parameters.",
        "probeB_best_shank_rate": best_b["rate_per_min_nrem"],
        "probeA_worst_shank_rate": worst_a["rate_per_min_nrem"],
        "ratio": round(best_b["rate_per_min_nrem"] / worst_a["rate_per_min_nrem"], 2),
    }
with open(out_dir / "stats.yml", "w") as f:
    yaml.safe_dump(stats, f, sort_keys=False)
with open(out_dir / "sources.yml", "w") as f:
    yaml.safe_dump({p: {"run_id": RID[p], "events": str(EVENTS_DIR[p])} for p in PROBES},
                   f, sort_keys=False)
print(f"\nsaved -> {out_dir}", flush=True)
if "specificity" in stats:
    print(yaml.safe_dump(stats["specificity"], sort_keys=False))
