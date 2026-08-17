"""Pure analysis functions for the ripple explorer (mirrors notebooks/ripple_detection.ipynb)."""

import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import welch, butter, hilbert, sosfiltfilt


def load_recording(path, probe_config_path, config):
    """Load an LFP recording, auto-detecting a preprocessed SI folder vs a raw .bin.

    For a raw .bin the LFP chain (bandpass -> global-median CMR -> resample) from `config` is
    applied lazily. Sets the active probe so get_channel_locations() works."""
    from pathlib import Path
    import spikeinterface.full as si
    import spikeinterface.extractors as se
    import spikeinterface.preprocessing as sp
    from probeinterface import read_probeinterface

    path = Path(path)
    probe = read_probeinterface(probe_config_path).probes[0]
    active = probe.get_slice(probe.device_channel_indices != -1)

    if path.is_dir():
        rec = si.load(path).set_probe(active)
    else:
        rec = se.read_binary(
            path,
            config["recording"]["sample_rate"],
            config["recording"]["dtype"],
            config["recording"]["n_channels"],
        ).set_probe(active)
        lfp = config["lfp"]
        rec = sp.bandpass_filter(rec, freq_min=lfp["bandpass_filter"]["freq_min"],
                                 freq_max=lfp["bandpass_filter"]["freq_max"])
        rec = sp.common_reference(rec, reference=lfp["common_reference"]["reference"],
                                  operator=lfp["common_reference"]["operator"])
        rec = sp.resample(rec, resample_rate=lfp["resample_rate"])
    return rec


def compute_psd(traces, fs, nperseg_seconds=2.0):
    """Welch PSD per channel. traces: (n_samples, n_channels). Returns (freqs, psd)."""
    return welch(traces, fs=fs, nperseg=int(nperseg_seconds * fs), axis=0)


def band_power(freqs, psd, passband):
    """Integrate PSD over passband per channel. Returns (n_channels,)."""
    mask = (freqs >= passband[0]) & (freqs <= passband[1])
    return np.trapezoid(psd[mask], freqs[mask], axis=0)


def shank_index(locations, pitch=250.0):
    """Shank index per channel from x positions (NeuropixelsV2 shank pitch = 250 um)."""
    xs = locations[:, 0]
    return np.round((xs - xs.min()) / pitch).astype(int)


def pick_channels_per_shank(power, shank, n_per_shank=1):
    """Per shank, pick n_per_shank channels spanning highest->lowest power. {shank: indices}."""
    picks = {}
    for s in np.unique(shank):
        sel = np.where(shank == s)[0]
        ranked = sel[np.argsort(power[sel])[::-1]]
        picks[s] = ranked[np.linspace(0, len(ranked) - 1, n_per_shank).astype(int)]
    return picks


def neighbour_channels(locations, channel):
    """The above/below/lateral neighbours of `channel` on its own shank: the next contact up and
    the next down its own column, plus the nearest contact in the other column. Neighbours that
    don't exist (channel at a shank end, single-column probe) are omitted from the dict."""
    shank = shank_index(locations)
    x0, y0 = locations[channel]
    same = np.flatnonzero(shank == shank[channel])
    col = same[locations[same, 0] == x0]
    other = same[locations[same, 0] != x0]
    up = col[locations[col, 1] > y0]
    down = col[locations[col, 1] < y0]
    out = {}
    if up.size:
        out["above"] = int(up[np.argmin(locations[up, 1])])
    if down.size:
        out["below"] = int(down[np.argmax(locations[down, 1])])
    if other.size:
        out["lateral"] = int(other[np.argmin(np.abs(locations[other, 1] - y0))])
    return out


def bandpass_envelope(trace, fs, passband, order=4, smooth_s=0.0):
    """Zero-phase bandpass (Butterworth SOS, sosfiltfilt) + Hilbert envelope, optionally
    moving-average smoothed. Returns (filtered, envelope)."""
    sos = butter(order, passband, btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, trace)
    # hilbert FFTs at len(trace); a 22 h recording is ~1e8 samples of arbitrary length, which drops
    # scipy into Bluestein's algorithm. Zero-pad to the next fast length and truncate back -- the
    # only cost is a padding edge effect in the final samples.
    envelope = np.abs(hilbert(filtered, next_fast_len(filtered.size)))[:filtered.size]
    if smooth_s > 0:
        n = max(int(smooth_s * fs), 1)
        envelope = np.convolve(envelope, np.ones(n) / n, mode="same")
    return filtered, envelope


def zscore_envelope(envelope, baseline_mask=None):
    """Z-score `envelope` by its mean/SD over `baseline_mask` samples (None = all samples)."""
    ref = envelope if baseline_mask is None else envelope[baseline_mask]
    return (envelope - ref.mean()) / ref.std()


def epoch_mask_to_samples(times, mask, n_samples, fs):
    """Expand a per-epoch boolean mask (on score_recording's `times` grid) onto the LFP sample
    grid by nearest epoch centre. Epoch boundaries are only accurate to a few seconds -- `times`
    are 10 s STFT frame centres and the metrics behind `mask` carry a 15 s moving average."""
    t = np.arange(n_samples) / fs
    i = np.searchsorted(times, t).clip(1, len(times) - 1)
    nearest = np.where(t - times[i - 1] <= times[i] - t, i - 1, i)
    return mask[nearest]


def detect_events(z, fs, boundary_sd, peak_sd, min_duration_s, max_duration_s,
                  restrict=None, min_inter_event_s=0.0):
    """Runs of `z` above boundary_sd, merged if separated by less than min_inter_event_s, then
    kept if duration is in [min, max] and peak exceeds peak_sd. restrict: boolean sample mask an
    event's peak must fall inside (None = whole recording); tested on the peak rather than the
    extent so events straddling a state boundary keep their true duration. Order is load-bearing:
    merging after the duration filter would be useless, the fragments are already gone.
    Returns a dict of arrays: start/end/peak (samples), peak_z, duration_s."""
    above = z > boundary_sd
    d = np.diff(above.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if above[0]:
        starts = np.r_[0, starts]
    if above[-1]:
        ends = np.r_[ends, above.size]
    if starts.size == 0:
        empty = np.array([], dtype=int)
        return {"start": empty, "end": empty, "peak": empty,
                "peak_z": np.array([]), "duration_s": np.array([])}

    if min_inter_event_s > 0 and starts.size > 1:
        # keep marks each merged group's first run; the last run of a group is the one before it
        keep = np.r_[True, (starts[1:] - ends[:-1]) > min_inter_event_s * fs]
        starts, ends = starts[keep], ends[np.r_[keep[1:], True]]

    peak = np.array([s + np.argmax(z[s:e]) for s, e in zip(starts, ends)])
    duration_s = (ends - starts) / fs
    ok = ((duration_s >= min_duration_s) & (duration_s <= max_duration_s) & (z[peak] > peak_sd))
    if restrict is not None:
        ok &= restrict[peak]
    return {"start": starts[ok], "end": ends[ok], "peak": peak[ok],
            "peak_z": z[peak][ok], "duration_s": duration_s[ok]}


def count_corroborating(events, neighbour_events, n_samples):
    """Per event in `events`, how many of the `neighbour_events` event dicts have an event
    overlapping it in time. Neighbours are detected separately, so their criteria are independent
    of the ones that produced `events`."""
    counts = np.zeros(events["start"].size, dtype=int)
    for nb in neighbour_events:
        m = np.zeros(n_samples, dtype=bool)
        for s, e in zip(nb["start"], nb["end"]):
            m[s:e] = True
        counts += np.fromiter((m[s:e].any() for s, e in zip(events["start"], events["end"])),
                              dtype=int, count=counts.size)
    return counts


def peri_event_trace(trace, event, fs, window_s):
    """Trace +/- window_s around an event (start, end) in samples, clipped to bounds."""
    pad = int(window_s * fs)
    s, e = event
    lo = max(int(s) - pad, 0)
    hi = min(int(e) + pad, len(trace))
    return np.arange(lo, hi) / fs, trace[lo:hi]
