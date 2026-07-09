"""Core sleep-scoring analysis: shared log-spectrogram front-end, the broadband slow-wave
PC1 metric, narrowband theta ratios, most-bimodal channel selection, and grid reductions.
Operates on 1250 Hz LFP; buzcode SleepScoreLFP / Watson et al. 2016 conventions."""

import numpy as np
import scipy.signal as signal

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from diptest import diptest


def log_spectrogram(trace, fs, freqs=None):
    """STFT (10 s Hann, 1 s step) with power interpolated onto a log-spaced 1-100 Hz grid.
    Returns (spec[freq, time], freqs, times). Shared front-end for the slow-wave and theta metrics."""
    if freqs is None:
        freqs = np.logspace(0, 2, 100)  # 1-100 Hz, log-spaced
    nperseg = int(10 * fs)            # 10 s window
    noverlap = nperseg - int(1 * fs)  # 1 s step
    f, times, spec_lin = signal.spectrogram(trace, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
    spec = np.stack([np.interp(freqs, f, spec_lin[:, i]) for i in range(spec_lin.shape[1])], axis=1)
    return spec, freqs, times


def broadband_pc1(trace, fs, pc=1):
    """Steps 1-4 on one channel: STFT -> log-freq power -> log10 -> per-freq z-score -> PCA -> oriented PC1."""
    pc_index = pc - 1
    spec, freqs, times = log_spectrogram(trace, fs)

    logspec = np.log10(spec)
    zspec = (logspec - logspec.mean(axis=1, keepdims=True)) / logspec.std(axis=1, keepdims=True)

    pca = PCA()
    scores = pca.fit_transform(zspec.T)
    pc1, loading = scores[:, pc_index], pca.components_[pc_index]
    if loading[freqs < 20].mean() < 0:  # orient: high score = more slow wave
        pc1, loading = -pc1, -loading
    return pc1, loading, spec, freqs, times


def select_channel_by_dip(rec, fs, stride=16, manual_channel=None):
    """Scan every Nth channel, score PC1 bimodality with Hartigan's dip; return best channel + dip stats."""
    n_ch = rec.get_num_channels()
    candidate_idx = np.arange(0, n_ch, stride)
    dip_stats = np.full(n_ch, np.nan)
    for ci in candidate_idx:
        trace = rec.get_traces(channel_ids=[rec.channel_ids[ci]]).squeeze()
        dip_stats[ci] = diptest(broadband_pc1(trace, fs)[0])[0]
    best_channel = manual_channel if manual_channel is not None else int(np.nanargmax(dip_stats))
    return best_channel, dip_stats, candidate_idx


def pc1_threshold(pc1):
    """Step 5: threshold between the two PC1 modes at the 2-component GMM posterior crossover."""
    gmm = GaussianMixture(n_components=2, random_state=0).fit(pc1.reshape(-1, 1))
    hi_comp = np.argsort(gmm.means_.ravel())[1]            # higher-mean component = NREM
    lo_mean, hi_mean = np.sort(gmm.means_.ravel())
    grid = np.linspace(lo_mean, hi_mean, 1000)
    post_hi = gmm.predict_proba(grid.reshape(-1, 1))[:, hi_comp]
    return grid[np.argmin(np.abs(post_hi - 0.5))]          # posterior crossover (decision boundary)


# --- Narrowband theta --------------------------------------------------------
THETA_BANDS = {
    "watson":    {"theta": (5, 10), "denom": (2, 16)},   # Watson et al. 2016: theta 5-10 / broadband 2-16 Hz
    "shin":      {"theta": (6, 12), "denom": (1, 4)},    # Shin et al. 2026: theta 6-12 / delta 1-4 Hz
    "shin_mod":  {"theta": (5, 10), "denom": (1, 4)}     # Default: theta 5-10 / broadband 1-4 Hz
}


def band_power(spec, freqs, band):
    """Integrate log-spaced power spectrum over [lo, hi] per time bin (trapezoid on the freq grid)."""
    lo, hi = band
    m = (freqs >= lo) & (freqs <= hi)
    return np.trapezoid(spec[m], freqs[m], axis=0)


def theta_ratio(spec, freqs, convention):
    """theta-band / denominator-band power per time bin, for the given convention."""
    b = THETA_BANDS[convention]
    return band_power(spec, freqs, b["theta"]) / band_power(spec, freqs, b["denom"])


def select_theta_channel(rec, fs, convention, stride=16, manual_channel=None):
    """buzcode-style theta-channel pick: the channel whose theta-ratio distribution separates
    best into two modes (REM vs rest). We quantify that separation exactly as for the slow-wave
    channel -- Hartigan's dip test -- on the log10 theta ratio (ratios are right-skewed, so log
    first), and take the max-dip channel. Mirrors select_channel_by_dip."""
    n_ch = rec.get_num_channels()
    candidate_idx = np.arange(0, n_ch, stride)
    dip_stats = np.full(n_ch, np.nan)
    for ci in candidate_idx:
        trace = rec.get_traces(channel_ids=[rec.channel_ids[ci]]).squeeze()
        spec, freqs, _ = log_spectrogram(trace, fs)
        dip_stats[ci] = diptest(np.log10(theta_ratio(spec, freqs, convention)))[0]
    best = manual_channel if manual_channel is not None else int(np.nanargmax(dip_stats))
    return best, dip_stats, candidate_idx


# --- Small reductions --------------------------------------------------------
def zscore(x):
    return (x - x.mean()) / x.std()


def bin_max(t, x, grid):
    """Reduce samples to the nearest grid bin, taking the max per bin (empty bins -> NaN).
    Drops samples outside [grid[0], grid[-1]] so out-of-range tails don't pollute the edge bins."""
    keep = (t >= grid[0]) & (t <= grid[-1])
    t, x = t[keep], x[keep]
    idx = np.searchsorted(grid, t).clip(0, len(grid) - 1)      # nearest grid bin per sample
    out = np.full(len(grid), -np.inf)
    np.maximum.at(out, idx, x)
    out[np.isneginf(out)] = np.nan                             # bins with no sample
    return out
