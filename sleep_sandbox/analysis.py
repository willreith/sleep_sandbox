"""Core sleep-scoring analysis: shared log-spectrogram front-end, the broadband slow-wave
PC1 metric, narrowband theta ratios, most-bimodal channel selection, and grid reductions.
Operates on 1250 Hz LFP; buzcode SleepScoreLFP / Watson et al. 2016 conventions."""

import warnings

import numpy as np
import scipy.signal as signal

from scipy.stats import gaussian_kde
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


def broadband_pc1(trace, fs, pc=1, orientation_freq_hz=20.0, mode="refit", basis=None, **spectrogram_kwargs):
    """Steps 1-4 on one channel: STFT -> log-freq power -> log10 -> per-freq z-score -> PCA -> oriented PC1.
    orientation_freq_hz: sign-flip PC if the mean loading below this freq is negative.
    mode: 'refit' fits loading/mu/sd here; 'global' and 'middle' project onto basis (see project_pc1).
    spectrogram_kwargs: forwarded to log_spectrogram (window_s, step_s, freq_min, freq_max, n_freq_bins, freqs)."""
    pc_index = pc - 1
    spec, freqs, times = log_spectrogram(trace, fs, **spectrogram_kwargs)
    if mode != "refit":
        if basis is None:
            raise ValueError(f"mode={mode!r} needs a basis from fit_pc1_basis")
        return project_pc1(spec, basis, mode), basis["loading"], spec, freqs, times

    # Zero-power bins are recording gaps the upstream loader zero-filled. log10(0) = -inf, and a
    # single -inf takes the per-frequency z-score and the PCA with it -- poisoning the whole
    # recording, not just the gap -- so fit on the real bins and mark the rest NaN.
    valid = spec.sum(axis=0) > 0
    logspec = np.log10(spec[:, valid])
    zspec = (logspec - logspec.mean(axis=1, keepdims=True)) / logspec.std(axis=1, keepdims=True)

    pca = PCA()
    scores = pca.fit_transform(zspec.T)
    loading = pca.components_[pc_index]
    pc1 = np.full(len(times), np.nan)
    pc1[valid] = scores[:, pc_index]
    if loading[freqs < orientation_freq_hz].mean() < 0:  # orient: high score = more slow wave
        pc1, loading = -pc1, -loading
    return pc1, loading, spec, freqs, times


def fit_pc1_basis(spec, freqs, pc=1, orientation_freq_hz=20.0):
    """broadband_pc1's fit as a reusable basis: per-freq log10 mean/std and the oriented loading, on non-gap bins."""
    valid = spec.sum(axis=0) > 0
    logspec = np.log10(spec[:, valid])
    mu, sd = logspec.mean(axis=1), logspec.std(axis=1)
    pca = PCA().fit(((logspec - mu[:, None]) / sd[:, None]).T)
    loading = pca.components_[pc - 1]
    if loading[freqs < orientation_freq_hz].mean() < 0:
        loading = -loading
    return {"loading": loading, "mu": mu, "sd": sd, "freqs": freqs,
            "var_explained": float(pca.explained_variance_ratio_[pc - 1])}


def project_pc1(spec, basis, mode="global"):
    """PC1 per bin on basis['loading']: 'global' z-scores with basis mu/sd, 'middle' with spec's own; gaps NaN."""
    valid = spec.sum(axis=0) > 0
    logspec = np.log10(spec[:, valid])
    if mode == "global":
        mu, sd = basis["mu"], basis["sd"]
    elif mode == "middle":
        mu, sd = logspec.mean(axis=1), logspec.std(axis=1)
    else:
        raise ValueError(f"unknown PC1 projection mode {mode!r}")
    pc1 = np.full(spec.shape[1], np.nan)
    pc1[valid] = basis["loading"] @ ((logspec - mu[:, None]) / sd[:, None])
    return pc1


def select_channel_by_dip(rec, fs, stride=16, manual_channel=None, **pc1_kwargs):
    """Scan every Nth channel, score PC1 bimodality with Hartigan's dip; return best channel + dip stats.
    pc1_kwargs: forwarded to broadband_pc1 (pc, orientation_freq_hz, spectrogram params)."""
    n_ch = rec.get_num_channels()
    candidate_idx = np.arange(0, n_ch, stride)
    dip_stats = np.full(n_ch, np.nan)
    traces = rec.get_traces(channel_ids=rec.channel_ids[candidate_idx])  # one pass over the file
    for j, ci in enumerate(candidate_idx):
        pc1 = broadband_pc1(traces[:, j], fs, **pc1_kwargs)[0]
        dip_stats[ci] = diptest(pc1[~np.isnan(pc1)])[0]    # gaps carry no bimodality information
    best_channel = manual_channel if manual_channel is not None else int(np.nanargmax(dip_stats))
    return best_channel, dip_stats, candidate_idx


def smooth_norm(x, step_s=1.0, win_s=15.0):
    """buzcode metric prep before thresholding: centred moving average over win_s seconds, then
    min-max to [0, 1]. Mirrors smooth(metric, smoothfact/specdt) (smoothfact=15) + bz_NormToRange
    ([0 1]) in ClusterStates_GetMetrics. step_s is the metric bin width (1 s on the sw_times grid);
    edges use a shrinking window (local mean), approximating MATLAB smooth.

    NaN (a recording gap) is skipped rather than propagated: the denominator counts only real
    samples, so a window straddling a gap averages just those -- the same shrinking-window rule the
    edges already used. Gaps stay NaN in the output (they are not interpolated across), and the
    [0, 1] range is taken over the real samples."""
    n = max(1, int(round(win_s / step_s)))
    if n % 2 == 0:
        n += 1                                            # MATLAB smooth uses an odd span
    kernel = np.ones(n)
    real = ~np.isnan(x)
    sm = np.convolve(np.where(real, x, 0.0), kernel, mode="same") / np.convolve(real.astype(float), kernel, mode="same")
    sm[~real] = np.nan
    lo, hi = np.nanmin(sm), np.nanmax(sm)
    return (sm - lo) / (hi - lo)


def bimodal_thresh(x, startbins=12, maxbins=25):
    """buzcode bz_BimodalThresh: the coarsest histogram (bins from startbins up to maxbins) that
    resolves two peaks, then the deepest trough between them; returns that bin centre, or NaN if no
    two-peak split is found. Peaks are taken on the zero-padded histogram (so edge bins can be modes),
    first two by location -- as in bz_BimodalThresh.m / the Motion branch. (The SW/theta inline copies
    instead take the two tallest, findpeaks ...,'SortStr','descend'; kept simple here.)

    NaN (recording gaps) is dropped: np.histogram would otherwise fail on the implicit range."""
    x = x[~np.isnan(x)]
    for numbins in range(startbins, maxbins + 1):
        hist, edges = np.histogram(x, bins=numbins)
        peaks, _ = signal.find_peaks(np.concatenate(([0], hist, [0])))
        peaks = peaks - 1                                 # undo left pad -> indices into hist
        if len(peaks) >= 2:
            lo, hi = peaks[0], peaks[1]
            centers = (edges[:-1] + edges[1:]) / 2
            return centers[lo + np.argmin(hist[lo:hi + 1])]   # deepest valley between the two modes
    return np.nan


def kde_thresh(x, grid_n=512, min_prominence_frac=0.03, label=""):
    """Between-mode threshold as the deepest trough of a Gaussian KDE (Scott's rule) between its two
    most prominent peaks.

    min_prominence_frac: a peak must rise this fraction of the maximum density above its
    surroundings to count as a mode. Required, not cosmetic -- with no floor, floating-point ripple
    in the near-zero tails registers as a "mode" and a unimodal Gaussian yields a threshold out in
    its own tail. This is the KDE analogue of bimodal_thresh's zero-padded edge bin being taken as
    a second mode (docs/threshold_comparison.md §2a).

    Returns NaN, with a RuntimeWarning naming the metric, when fewer than two modes clear the floor.
    That is the honest answer for a unimodal metric -- and it is a real outcome here, not a corner
    case: several of the series x combo cases in docs/threshold_comparison.md, concentrated on
    ProbeA theta, which the dip test independently calls unimodal (p >= 0.99). NaN propagates:
    comparisons against it are all-False, so classify() yields an empty mask rather than raising.

    NaN (recording gaps) is dropped: gaussian_kde returns an all-NaN density otherwise."""
    x = x[~np.isnan(x)]
    grid = np.linspace(x.min(), x.max(), grid_n)
    dens = gaussian_kde(x)(grid)
    peaks, props = signal.find_peaks(dens, prominence=min_prominence_frac * dens.max())
    if len(peaks) < 2:
        warnings.warn(f"kde_thresh({label or 'unnamed'}): density has {len(peaks)} mode(s), no "
                      "two-mode split -> NaN threshold; the dependent state mask will be empty",
                      RuntimeWarning, stacklevel=2)
        return np.nan
    lo, hi = np.sort(peaks[np.argsort(props["prominences"])[-2:]])
    return grid[lo + np.argmin(dens[lo:hi + 1])]


def find_thresh(x, method="histogram", startbins=12, maxbins=25, grid_n=512,
                min_prominence_frac=0.03, label=""):
    """Dispatch to the configured between-mode threshold method (config threshold.method).
    'histogram' is buzcode's bz_BimodalThresh; 'kde' is the KDE trough, which agrees with it where
    it works and repairs it where it fails -- see docs/threshold_comparison.md."""
    if method == "kde":
        return kde_thresh(x, grid_n=grid_n, min_prominence_frac=min_prominence_frac, label=label)
    if method == "histogram":
        return bimodal_thresh(x, startbins, maxbins)
    raise ValueError(f"unknown threshold method: {method!r} (expected 'kde' or 'histogram')")


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
        lr = np.log10(theta_ratio(spec, freqs, convention))   # 0/0 -> NaN in zero-power gap bins
        dip_stats[ci] = diptest(lr[~np.isnan(lr)])[0]
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
    traces = rec.get_traces(channel_ids=rec.channel_ids[candidate_idx])  # one pass over the file
    for j, ci in enumerate(candidate_idx):
        spec, freqs, _ = log_spectrogram(traces[:, j], fs, **spectrogram_kwargs)
        peak_stats[ci] = band_power(spec, freqs, theta).mean() / band_power(spec, freqs, denom).mean()
    best = manual_channel if manual_channel is not None else int(np.nanargmax(peak_stats))
    return best, peak_stats, candidate_idx


def conditioned_theta_thresh(theta_metric, sw_metric, motion_metric, sw_thresh, motion_thresh,
                             startbins=12, maxbins=25, method="histogram", grid_n=512,
                             conditioned=True, min_prominence_frac=0.03, exclude_mov=True):
    """Movement-conditioned theta threshold, buzcode ClusterStates_GetMetrics with one deviation:
    the trough is always taken on ~NREMtimes & ~MOVtimes, where buzcode takes it on ~MOVtimes and
    only excludes NREM as a fallback. MOVtimes = low SW & high motion. All inputs are the
    smoothed/[0,1] metrics on the same grid. Returns (th_thresh, movtimes).

    The fallback order is wrong for this dataset. ~mov is 85-92% NREM, so its trough marks the
    NREM/non-NREM boundary rather than the REM/quiet-wake one the theta threshold exists to place,
    and on ProbeB it clears the prominence floor just well enough (0.024) never to fall back --
    silently putting REM detection on the wrong split. Excluding NREM leaves the wake+REM
    population, where the same split carries ~10x the mode prominence (0.22 on ProbeB/lfp_cmr over
    22 h). That subset is smaller, which is what the longer recording buys back.

    conditioned=False takes the trough on the full distribution instead; movtimes is returned either
    way, since downstream plots and result.npz report it regardless of how the threshold was set.

    exclude_mov=False keeps the movement bins in the pool, i.e. the trough is taken on all non-NREM
    bins. Those carry running theta, the strongest theta in the recording, so the trough can land
    between quiet wake and run theta rather than on the REM/quiet-wake split this threshold exists
    to place."""
    mov = (sw_metric < sw_thresh) & (motion_metric > motion_thresh)
    if not conditioned:
        return find_thresh(theta_metric, method, startbins, maxbins, grid_n,
                           min_prominence_frac, label="theta|all"), mov
    keep = ~(sw_metric > sw_thresh) & ~mov if exclude_mov else ~(sw_metric > sw_thresh)
    label = "theta|~nrem&~mov" if exclude_mov else "theta|~nrem"
    return find_thresh(theta_metric[keep], method, startbins, maxbins, grid_n,
                       min_prominence_frac, label=label), mov


# --- Intracranial EMG proxy ---------------------------------------------------
def make_emg_pairs(shank_ids, n_pairs=100, min_shank_dist=2, seed=0):
    """n_pairs random channel pairs whose shanks are >= min_shank_dist apart. Pairs are spread
    round-robin across every qualifying shank-combo, drawing distinct channels (a shank's pool is
    reshuffled + refilled only once exhausted) so as many electrodes as possible are covered."""
    rng = np.random.default_rng(seed)
    shanks = np.asarray(shank_ids, dtype=int)
    uniq = np.unique(shanks)
    ch_by_shank = {s: np.where(shanks == s)[0] for s in uniq}
    combos = [(a, b) for i, a in enumerate(uniq) for b in uniq[i + 1:] if b - a >= min_shank_dist]
    queues = {s: [] for s in uniq}
    def draw(s):
        if not queues[s]:
            queues[s] = list(rng.permutation(ch_by_shank[s]))
        return int(queues[s].pop())
    pairs = np.array([draw(s) for k in range(n_pairs) for s in combos[k % len(combos)]]).reshape(-1, 2)
    return pairs, combos


def emg_from_lfp(rec, pairs, win_s=0.5, chunk_s=600):
    """Watson EMG score: mean per-window Pearson r across channel pairs, on non-overlapping win_s
    windows of the raw 300-600 Hz trace. Returns (emg, times) at the window rate (1/win_s).
    chunk_s: seconds of data held in memory at a time, bounding peak memory to
    chunk_s * n_channels_used regardless of total recording length."""
    fs = rec.get_sampling_frequency()
    win = int(round(win_s * fs))
    n_win = rec.get_num_frames() // win
    ch_ids = rec.channel_ids
    used = np.unique(pairs)
    col = {c: j for j, c in enumerate(used)}

    win_per_chunk = max(1, int(chunk_s / win_s))
    r = np.zeros(n_win)
    for start_win in range(0, n_win, win_per_chunk):
        end_win = min(start_win + win_per_chunk, n_win)
        traces = rec.get_traces(channel_ids=ch_ids[used],
                                 start_frame=start_win * win, end_frame=end_win * win).astype(np.float32)
        n_w = end_win - start_win
        for a, b in pairs:
            A = traces[:, col[a]].reshape(n_w, win)
            B = traces[:, col[b]].reshape(n_w, win)
            A = A - A.mean(1, keepdims=True)
            B = B - B.mean(1, keepdims=True)
            r[start_win:end_win] += (A * B).sum(1) / np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1))
    emg = r / len(pairs)
    times = (np.arange(n_win) * win + win / 2) / fs
    return emg, times


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


def bin_var(t, x, grid):
    """Per-grid-bin variance of x; NaN for empty bins. Nearest-bin assignment, like bin_max (but no
    out-of-range drop -- callers pass already-valid samples). NB: no buzcode precedent -- buzcode
    has no accelerometer variance measure anywhere (see docs/ANALYSIS_FRAMEWORK.md); this is a
    local tone-like candidate."""
    idx = np.searchsorted(grid, t).clip(0, len(grid) - 1)
    c = np.bincount(idx, minlength=len(grid)).astype(float)
    s = np.bincount(idx, weights=x, minlength=len(grid))
    s2 = np.bincount(idx, weights=x * x, minlength=len(grid))
    c[c == 0] = np.nan
    return s2 / c - (s / c) ** 2


def bin_mean(t, x, grid):
    """Per-grid-bin mean of x; NaN for empty bins. Matches buzcode bz_getIntanAccel's final step
    (bin-average the filtered motion magnitude), unlike bin_max which takes the per-bin peak."""
    idx = np.searchsorted(grid, t).clip(0, len(grid) - 1)
    c = np.bincount(idx, minlength=len(grid)).astype(float)
    s = np.bincount(idx, weights=x, minlength=len(grid))
    c[c == 0] = np.nan
    return s / c


def accel_motion_buzcode(accel, t, grid, low_hz=0.1, high_hz=1.0, order=2):
    """buzcode bz_getIntanAccel's motion proxy applied to an acceleration magnitude: bandpass
    (low_hz-high_hz, Butterworth, filtfilt) -> abs -> per-bin mean onto grid. buzcode bandpasses the
    vector-norm of raw 3-axis accelerometer voltage; here `accel` is already a magnitude
    (imu_kinematics' |LinearAcceleration|, gravity removed)."""
    fs = 1.0 / np.median(np.diff(t))
    b, a = signal.butter(order, [low_hz / (fs / 2), high_hz / (fs / 2)], btype="band")
    return bin_mean(t, np.abs(signal.filtfilt(b, a, accel)), grid)


# --- Classification-comparison diagnostics --------------------------------------
def state_codes(states, state_names):
    """states: dict of boolean masks, one per name in state_names, partitioning every epoch
    exactly once -> one integer code per epoch (index into state_names)."""
    codes = np.full(len(states[state_names[0]]), -1, dtype=int)
    for i, name in enumerate(state_names):
        codes[states[name]] = i
    assert (codes >= 0).all(), "states do not partition all epochs"
    return codes


def confusion(a, b, n_states):
    """a, b: integer state-code arrays (state_codes) over the same epochs -> n_states x n_states
    confusion matrix, rows indexed by a, columns by b."""
    return np.bincount(a * n_states + b, minlength=n_states ** 2).reshape(n_states, n_states)


def cohens_kappa(a, b, n_states):
    """Chance-corrected agreement between two state-code sequences over the same epochs. Raw
    agreement (trace/n) is inflated whenever one state dominates; kappa subtracts the agreement
    expected from each sequence's own marginal frequencies before rescaling."""
    cm = confusion(a, b, n_states)
    n = cm.sum()
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def state_intervals(mask, t, dt):
    """Contiguous True-runs of a boolean state mask as (start_time, duration) pairs, for
    ax.broken_barh. dt is the epoch width (grid spacing), so a single-epoch run still renders as
    a dt-wide bar rather than a zero-width one."""
    padded = np.concatenate(([0], mask.astype(int), [0]))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(t[starts], (ends - starts) * dt))


def bout_durations(mask, dt):
    """Contiguous-run lengths (seconds) of a boolean state mask."""
    return np.array([d for _, d in state_intervals(mask, np.zeros(len(mask)), dt)])


# --- Watson duration criteria -------------------------------------------------
# Watson et al. 2016 supplementary defines each period type by a minimum duration as well as by its
# metric thresholds. Implemented so far: the short-run merge below and microarousals. The 20 s
# minimum for nonREM packets and REM is NOT implemented -- deliberately deferred, so a bout that
# clears the thresholds is currently kept regardless of length. Watson's episode-level definitions
# (REM/nonREM episodes tolerating <=40 s interruptions, SLEEP, WAKE, WAKE-SLEEP cycle) are also not
# implemented; our nrem/rem are packets, not episodes.
def run_bounds(codes):
    """Contiguous runs of an integer code array as (start_idx, stop_idx, value) arrays."""
    change = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(codes)]))
    return starts, stops, codes[starts]


def merge_short_states(codes, dt, max_s):
    """Runs lasting <= max_s take the preceding run's state (de-flicker). Single left-to-right pass,
    reading the already-updated predecessor, so consecutive short runs cascade into one state rather
    than each inheriting the original labels. The first run has no predecessor and is left as-is."""
    out = codes.copy()
    starts, stops, _ = run_bounds(codes)
    for i in range(1, len(starts)):
        if (stops[i] - starts[i]) * dt <= max_s:
            out[starts[i]:stops[i]] = out[starts[i] - 1]
    return out


def drop_short_packets(codes, dt, min_s, wake_code):
    """Watson: a nonREM packet is 20+ s of nonREM and REM is 20+ s of REM, so shorter NREM/REM runs
    do not qualify and fall to the residual class, wake. Single pass over the pre-pass run bounds,
    so a demoted REM and a short NREM beside it are both caught without cascading."""
    out = codes.copy()
    starts, stops, vals = run_bounds(codes)
    for i in range(len(starts)):
        if vals[i] != wake_code and (stops[i] - starts[i]) * dt < min_s:
            out[starts[i]:stops[i]] = wake_code
    return out


def label_microarousals(codes, dt, wake_code, nrem_code, max_s):
    """Watson microarousal: a WAKE run (i.e. neither NREM nor REM) shorter than max_s, flanked by
    NREM on both sides. Returned as a boolean mask -- MA is a subset of wake, not a fourth state, so
    the nrem/rem/wake partition is unchanged (mirrors how qwake is reported)."""
    ma = np.zeros(len(codes), dtype=bool)
    starts, stops, vals = run_bounds(codes)
    for i in range(1, len(starts) - 1):
        if vals[i] == wake_code and vals[i - 1] == nrem_code and vals[i + 1] == nrem_code \
                and (stops[i] - starts[i]) * dt < max_s:
            ma[starts[i]:stops[i]] = True
    return ma
