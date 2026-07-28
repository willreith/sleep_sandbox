"""Core sleep-scoring analysis: shared log-spectrogram front-end, the broadband slow-wave
PC1 metric, narrowband theta ratios, most-bimodal channel selection, and grid reductions.
Operates on 1250 Hz LFP; buzcode SleepScoreLFP / Watson et al. 2016 conventions."""

import numpy as np
import scipy.signal as signal

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from diptest import diptest


def log_spectrogram(trace, fs, freqs=None, window_s=10.0, step_s=1.0,
                     freq_min=1.0, freq_max=100.0, n_freq_bins=100):
    """STFT (window_s Hann, step_s step) with power interpolated onto a log-spaced
    freq_min-freq_max grid (n_freq_bins). Returns (spec[freq, time], freqs, times). Shared
    front-end for the slow-wave and theta metrics. Pass freqs explicitly to reuse an existing
    grid instead of building one from freq_min/freq_max/n_freq_bins."""
    if freqs is None:
        freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freq_bins)
    nperseg = int(window_s * fs)
    noverlap = nperseg - int(step_s * fs)
    f, times, spec_lin = signal.spectrogram(trace, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
    spec = np.stack([np.interp(freqs, f, spec_lin[:, i]) for i in range(spec_lin.shape[1])], axis=1)
    return spec, freqs, times


def broadband_pc1(trace, fs, pc=1, orientation_freq_hz=20.0, **spectrogram_kwargs):
    """Steps 1-4 on one channel: STFT -> log-freq power -> log10 -> per-freq z-score -> PCA -> oriented PC1.
    orientation_freq_hz: sign-flip PC if the mean loading below this freq is negative.
    spectrogram_kwargs: forwarded to log_spectrogram (window_s, step_s, freq_min, freq_max, n_freq_bins, freqs)."""
    pc_index = pc - 1
    spec, freqs, times = log_spectrogram(trace, fs, **spectrogram_kwargs)

    logspec = np.log10(spec)
    zspec = (logspec - logspec.mean(axis=1, keepdims=True)) / logspec.std(axis=1, keepdims=True)

    pca = PCA()
    scores = pca.fit_transform(zspec.T)
    pc1, loading = scores[:, pc_index], pca.components_[pc_index]
    if loading[freqs < orientation_freq_hz].mean() < 0:  # orient: high score = more slow wave
        pc1, loading = -pc1, -loading
    return pc1, loading, spec, freqs, times


def select_channel_by_dip(rec, fs, stride=16, manual_channel=None, **pc1_kwargs):
    """Scan every Nth channel, score PC1 bimodality with Hartigan's dip; return best channel + dip stats.
    pc1_kwargs: forwarded to broadband_pc1 (pc, orientation_freq_hz, spectrogram params)."""
    n_ch = rec.get_num_channels()
    candidate_idx = np.arange(0, n_ch, stride)
    dip_stats = np.full(n_ch, np.nan)
    for ci in candidate_idx:
        trace = rec.get_traces(channel_ids=[rec.channel_ids[ci]]).squeeze()
        dip_stats[ci] = diptest(broadband_pc1(trace, fs, **pc1_kwargs)[0])[0]
    best_channel = manual_channel if manual_channel is not None else int(np.nanargmax(dip_stats))
    return best_channel, dip_stats, candidate_idx


def smooth_norm(x, step_s=1.0, win_s=15.0):
    """buzcode metric prep before thresholding: centred moving average over win_s seconds, then
    min-max to [0, 1]. Mirrors smooth(metric, smoothfact/specdt) (smoothfact=15) + bz_NormToRange
    ([0 1]) in ClusterStates_GetMetrics. step_s is the metric bin width (1 s on the sw_times grid);
    edges use a shrinking window (local mean), approximating MATLAB smooth."""
    n = max(1, int(round(win_s / step_s)))
    if n % 2 == 0:
        n += 1                                            # MATLAB smooth uses an odd span
    kernel = np.ones(n)
    sm = np.convolve(x, kernel, mode="same") / np.convolve(np.ones_like(x), kernel, mode="same")
    return (sm - sm.min()) / (sm.max() - sm.min())


def bimodal_thresh(x, startbins=12, maxbins=25):
    """buzcode bz_BimodalThresh: the coarsest histogram (bins from startbins up to maxbins) that
    resolves two peaks, then the deepest trough between them; returns that bin centre, or NaN if no
    two-peak split is found. Peaks are taken on the zero-padded histogram (so edge bins can be modes),
    first two by location -- as in bz_BimodalThresh.m / the Motion branch. (The SW/theta inline copies
    instead take the two tallest, findpeaks ...,'SortStr','descend'; kept simple here.)"""
    for numbins in range(startbins, maxbins + 1):
        hist, edges = np.histogram(x, bins=numbins)
        peaks, _ = signal.find_peaks(np.concatenate(([0], hist, [0])))
        peaks = peaks - 1                                 # undo left pad -> indices into hist
        if len(peaks) >= 2:
            lo, hi = peaks[0], peaks[1]
            centers = (edges[:-1] + edges[1:]) / 2
            return centers[lo + np.argmin(hist[lo:hi + 1])]   # deepest valley between the two modes
    return np.nan


def pc1_threshold(pc1, step_s=1.0):
    """Step 5 (buzcode-concordant): between-mode threshold as the histogram trough of the 15 s-smoothed,
    [0, 1]-normalised PC1. NOTE the threshold is in smoothed/normalised units -- compare it against
    smooth_norm(pc1), not raw pc1. See pc1_threshold_gmm for the earlier GMM variant."""
    return bimodal_thresh(smooth_norm(pc1, step_s))


def pc1_threshold_gmm(pc1):
    """Earlier variant (non-buzcode): threshold at the 2-component GMM posterior crossover."""
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


def select_theta_channel(rec, fs, convention, stride=16, manual_channel=None, **spectrogram_kwargs):
    """buzcode-style theta-channel pick: the channel whose theta-ratio distribution separates
    best into two modes (REM vs rest). We quantify that separation exactly as for the slow-wave
    channel -- Hartigan's dip test -- on the log10 theta ratio (ratios are right-skewed, so log
    first), and take the max-dip channel. Mirrors select_channel_by_dip.
    spectrogram_kwargs: forwarded to log_spectrogram (window_s, step_s, freq_min, freq_max, n_freq_bins)."""
    n_ch = rec.get_num_channels()
    candidate_idx = np.arange(0, n_ch, stride)
    dip_stats = np.full(n_ch, np.nan)
    for ci in candidate_idx:
        trace = rec.get_traces(channel_ids=[rec.channel_ids[ci]]).squeeze()
        spec, freqs, _ = log_spectrogram(trace, fs, **spectrogram_kwargs)
        dip_stats[ci] = diptest(np.log10(theta_ratio(spec, freqs, convention)))[0]
    best = manual_channel if manual_channel is not None else int(np.nanargmax(dip_stats))
    return best, dip_stats, candidate_idx


def select_theta_channel_peak(rec, fs, theta=(5, 10), denom=(2, 20), stride=16, manual_channel=None,
                              **spectrogram_kwargs):
    """buzcode PickSWTHChannel theta pick (the actual buzcode default): the channel with the highest
    mean theta-band power ratio, NOT the most bimodal. Mirrors peakTH =
    sum(meanspec(thfreqs))/sum(meanspec(:)) -- theta 5-10 Hz over broadband 2-20 Hz on the time-mean
    spectrum. (Contrast select_theta_channel, which picks max dip -- a deliberate deviation.)
    spectrogram_kwargs: forwarded to log_spectrogram (window_s, step_s, freq_min, freq_max, n_freq_bins)."""
    n_ch = rec.get_num_channels()
    candidate_idx = np.arange(0, n_ch, stride)
    peak_stats = np.full(n_ch, np.nan)
    for ci in candidate_idx:
        trace = rec.get_traces(channel_ids=[rec.channel_ids[ci]]).squeeze()
        spec, freqs, _ = log_spectrogram(trace, fs, **spectrogram_kwargs)
        peak_stats[ci] = band_power(spec, freqs, theta).mean() / band_power(spec, freqs, denom).mean()
    best = manual_channel if manual_channel is not None else int(np.nanargmax(peak_stats))
    return best, peak_stats, candidate_idx


def conditioned_theta_thresh(theta_metric, sw_metric, motion_metric, sw_thresh, motion_thresh,
                             startbins=12, maxbins=25):
    """buzcode movement-conditioned theta threshold (ClusterStates_GetMetrics): the theta trough is
    taken only on non-moving epochs. MOVtimes = low SW & high motion; the theta dip is found on
    ~MOVtimes, and if that isn't bimodal, retried also excluding NREM (~NREMtimes & ~MOVtimes).
    All inputs are the smoothed/[0,1] metrics on the same grid. Returns (th_thresh, movtimes)."""
    nrem = sw_metric > sw_thresh
    mov = (sw_metric < sw_thresh) & (motion_metric > motion_thresh)
    th = bimodal_thresh(theta_metric[~mov], startbins, maxbins)
    if np.isnan(th):
        th = bimodal_thresh(theta_metric[~nrem & ~mov], startbins, maxbins)
    return th, mov


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
