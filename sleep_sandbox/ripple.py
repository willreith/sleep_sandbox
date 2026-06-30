"""Pure analysis functions for the ripple explorer (mirrors notebooks/ripple_detection.ipynb)."""

import numpy as np
from scipy.signal import welch, butter, hilbert, sosfilt


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


def bandpass_envelope(trace, fs, passband, order=4):
    """Bandpass (Butterworth SOS) + Hilbert envelope. Returns (filtered, envelope)."""
    sos = butter(order, passband, btype="bandpass", fs=fs, output="sos")
    filtered = sosfilt(sos, trace)
    return filtered, np.abs(hilbert(filtered))


def detect_events(envelope, fs, boundary_sd, peak_sd, min_duration_s, max_duration_s):
    """Z-score envelope; keep runs above boundary_sd whose peak exceeds peak_sd and whose
    duration is in [min, max]. Returns (events (n,2) samples, z-scored envelope)."""
    z = (envelope - envelope.mean()) / envelope.std()
    above = z > boundary_sd
    runs = []
    start = None
    for i, a in enumerate(above):
        if a and start is None:
            start = i
        elif not a and start is not None:
            runs.append((start, i))
            start = None
    min_d = int(min_duration_s * fs)
    max_d = int(max_duration_s * fs)
    events = [(s, e) for s, e in runs
              if min_d <= (e - s) <= max_d and z[s:e].max() > peak_sd]
    return np.array(events).reshape(-1, 2), z


def peri_event_trace(trace, event, fs, window_s):
    """Trace +/- window_s around an event (start, end) in samples, clipped to bounds."""
    pad = int(window_s * fs)
    s, e = event
    lo = max(int(s) - pad, 0)
    hi = min(int(e) + pad, len(trace))
    return np.arange(lo, hi) / fs, trace[lo:hi]
