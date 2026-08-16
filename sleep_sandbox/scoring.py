"""Per-recording sleep-scoring pipeline: runs the notebook's buzcode-concordant metric chain
(slow-wave PC1, peakTH theta ratio, intracranial EMG proxy, movement-conditioned theta
threshold) over a full preprocessed recording pair, driven entirely by scoring_config
(config/sleep_scoring.yml schema). Then summarize/plot/save the result."""

import json
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt

from sleep_sandbox.analysis import (
    broadband_pc1, select_channel_by_dip, select_theta_channel, select_theta_channel_peak,
    log_spectrogram, smooth_norm, find_thresh, theta_ratio, conditioned_theta_thresh,
    bin_max, bin_var, zscore, make_emg_pairs, emg_from_lfp,
    state_codes, merge_short_states, drop_short_packets, label_microarousals,
)

CORE_METRICS = ("sw_metric", "theta_metric", "motion_metric", "imu_speed")


STATES = ("nrem", "rem", "wake")


def classify(sw_metric, theta_metric, motion_metric, sw_thresh, startbins=12, maxbins=25,
             method="histogram", grid_n=512, dt=1.0, merge_shorter_than_s=None,
             min_state_s=None, microarousal_max_s=None, theta_conditioned=True,
             motion_thresh=None, th_thresh=None, min_prominence_frac=0.03):
    """buzcode ClusterStates_DetermineStates decision tree on already-smoothed/[0,1] metrics:
    motion threshold -> movement-conditioned theta threshold -> NREM/REM/WAKE masks, then the
    Watson duration criteria. Split out of score_recording so alternative motion signals (IMU
    variants) run through identical logic. Returns a dict of masks (nrem/rem/wake/qwake/ma/
    rem_cand/mov) and the thresholds used.

    wake = everything not NREM or REM; qwake (quiet wake) is buzcode's optional low-theta subset of
    wake, and ma (microarousal) is Watson's short-wake-between-NREM subset. States nrem/rem/wake
    partition every epoch exactly once; qwake and ma are subsets of wake, not separate states.

    method: 'histogram' (buzcode bz_BimodalThresh) or 'kde' -- see docs/threshold_comparison.md.
    Either can return a NaN threshold on a unimodal metric, which yields an empty mask downstream.
    merge_shorter_than_s: runs no longer than this take the preceding state (None = off).
    min_state_s: NREM/REM runs shorter than this become wake (None = off).
    microarousal_max_s: MA upper bound (None = off). rem_cand/mov are pre-merge intermediates.
    motion_thresh/th_thresh override the thresholds this would derive from the passed metrics, so
    thresholds estimated on one window can be applied to another (check_threshold_stability.py)."""
    if motion_thresh is None:
        motion_thresh = find_thresh(motion_metric, method, startbins, maxbins, grid_n,
                                    min_prominence_frac, label="motion")
    derived_th, mov = conditioned_theta_thresh(
        theta_metric, sw_metric, motion_metric, sw_thresh, motion_thresh, startbins, maxbins,
        method, grid_n, theta_conditioned, min_prominence_frac)
    if th_thresh is None:
        th_thresh = derived_th

    nrem = sw_metric > sw_thresh
    low_motion = motion_metric < motion_thresh
    rem_cand = ~nrem & low_motion
    rem = rem_cand & (theta_metric > th_thresh)
    wake = ~nrem & ~rem

    codes = state_codes({"nrem": nrem, "rem": rem, "wake": wake}, STATES)
    if merge_shorter_than_s:
        codes = merge_short_states(codes, dt, merge_shorter_than_s)
    if min_state_s:                       # after the merge, before MA: MA is defined against packets
        codes = drop_short_packets(codes, dt, min_state_s, STATES.index("wake"))
    if merge_shorter_than_s or min_state_s:
        nrem, rem, wake = (codes == i for i in range(len(STATES)))

    ma =(label_microarousals(codes, dt, STATES.index("wake"), STATES.index("nrem"), microarousal_max_s)
          if microarousal_max_s else np.zeros(len(codes), dtype=bool))

    return {"nrem": nrem, "rem": rem, "wake": wake, "qwake": wake & (theta_metric <= th_thresh),
            "ma": ma, "rem_cand": rem_cand, "mov": mov,
            "motion_thresh": motion_thresh, "th_thresh": th_thresh}


def score_recording(recording_lfp, recording_emg, scoring_config,
                     imu_t=None, imu_valid=None, imu_speed=None, theta_conventions=None):
    """Run the pipeline; recording_lfp/recording_emg are the preprocessed lfp_cmr/emg derivatives.
    imu_t/imu_valid/imu_speed: optional aligned IMU translational speed (align_bno055_to_lfp +
    imu_kinematics, already run by the caller); if given, IMU speed is binned onto the same grid.
    theta_conventions: extra THETA_BANDS convention names (besides the concordant one used for
    'theta_metric') to also report, each as 'theta_metric_<name>'. None (default) falls back to
    scoring_config['theta']['report_conventions']; pass [] explicitly to force none regardless of
    config. All conventions -- concordant and extra -- share the single peakTH-selected channel;
    only the band definition varies per convention, not the channel (see NB below; may change later).
    Returns a dict of per-timepoint arrays (on 'times'), boolean state masks, and scalars."""
    fs = recording_lfp.get_sampling_frequency()

    channel_stride = scoring_config["channel_selection"]["stride"]
    manual_channel = scoring_config["channel_selection"]["manual_slow_wave_channel"]
    pc_index = scoring_config["slow_wave"]["pc_index"]
    orientation_freq_hz = scoring_config["slow_wave"]["orientation_freq_hz"]
    spec_cfg = scoring_config["spectrogram"]
    spectrogram_kwargs = dict(window_s=spec_cfg["window_s"], step_s=spec_cfg["step_s"],
                              freq_min=spec_cfg["freq_min"], freq_max=spec_cfg["freq_max"],
                              n_freq_bins=spec_cfg["n_freq_bins"])
    step_s = spec_cfg["step_s"]
    smooth_win_s = scoring_config["smoothing"]["window_s"]
    bt_startbins = scoring_config["bimodal_threshold"]["startbins"]
    bt_maxbins = scoring_config["bimodal_threshold"]["maxbins"]
    thresh_cfg = scoring_config["threshold"]
    dur_cfg = scoring_config["duration_criteria"]

    # Slow-wave: dip-test channel selection -> PC1 -> smoothed/normed metric -> threshold.
    print(f"  [score_recording] slow-wave channel selection (dip test, every {channel_stride}th channel)...",
          flush=True)
    sw_channel, _, _ = select_channel_by_dip(
        recording_lfp, fs, channel_stride, manual_channel, pc=pc_index,
        orientation_freq_hz=orientation_freq_hz, **spectrogram_kwargs)
    print(f"  [score_recording] slow-wave channel = {sw_channel}; computing PC1 + threshold...", flush=True)
    sw_trace = recording_lfp.get_traces(channel_ids=[recording_lfp.channel_ids[sw_channel]]).squeeze()
    sw_pc1, _, _, _, times = broadband_pc1(
        sw_trace, fs, pc=pc_index, orientation_freq_hz=orientation_freq_hz, **spectrogram_kwargs)
    sw_metric = smooth_norm(sw_pc1, step_s=step_s, win_s=smooth_win_s)
    sw_thresh = find_thresh(sw_metric, thresh_cfg["method"], bt_startbins, bt_maxbins,
                            thresh_cfg["kde_grid_n"], thresh_cfg["min_prominence_frac"],
                            label="slow_wave")

    # Theta: peakTH channel selection, then the concordant convention's ratio on that channel.
    peak_band = scoring_config["theta"]["channel_peak_band"]
    concordant_conv = scoring_config["theta"]["concordant_convention"]
    if theta_conventions is None:
        theta_conventions = scoring_config["theta"].get("report_conventions", [])
    print(f"  [score_recording] theta channel selection (peakTH, every {channel_stride}th channel)...", flush=True)
    th_channel, _, _ = select_theta_channel_peak(
        recording_lfp, fs, theta=tuple(peak_band["theta"]), denom=tuple(peak_band["denom"]),
        stride=channel_stride, **spectrogram_kwargs)
    print(f"  [score_recording] theta channel = {th_channel}; computing ratio(s) "
          f"({concordant_conv} + {theta_conventions})...", flush=True)
    th_trace = recording_lfp.get_traces(channel_ids=[recording_lfp.channel_ids[th_channel]]).squeeze()
    th_spec, th_freqs, _ = log_spectrogram(th_trace, fs, **spectrogram_kwargs)
    theta_metric = smooth_norm(theta_ratio(th_spec, th_freqs, concordant_conv), step_s=step_s, win_s=smooth_win_s)
    # NB: all extra conventions reuse this same peakTH channel/spectrogram -- only the band
    # definition varies, not the channel. Each convention picking its own best channel (as the
    # notebook's diagnostic comparison does, via select_theta_channel/dip-test) is a possible
    # future change, not done here.
    extra_theta = {
        f"theta_metric_{conv}": smooth_norm(theta_ratio(th_spec, th_freqs, conv), step_s=step_s, win_s=smooth_win_s)
        for conv in (theta_conventions or [])
    }

    # EMG: cross-shank high-frequency correlation proxy, binned onto the same grid.
    emg_cfg = scoring_config["emg"]
    print(f"  [score_recording] EMG proxy ({emg_cfg['n_pairs']} channel pairs)...", flush=True)
    emg_shanks = recording_emg.get_probes()[0].shank_ids.astype(int)
    emg_pairs, _ = make_emg_pairs(
        emg_shanks, n_pairs=emg_cfg["n_pairs"], min_shank_dist=emg_cfg["min_shank_dist"], seed=emg_cfg["seed"])
    emg_score, emg_times = emg_from_lfp(recording_emg, emg_pairs, win_s=emg_cfg["window_s"])
    emg_b = np.interp(times, emg_times, emg_score)
    motion_metric = smooth_norm(emg_b, step_s=step_s, win_s=smooth_win_s)

    print("  [score_recording] movement-conditioned theta threshold + states...", flush=True)
    states = classify(sw_metric, theta_metric, motion_metric, sw_thresh, bt_startbins, bt_maxbins,
                      thresh_cfg["method"], thresh_cfg["kde_grid_n"], dt=step_s,
                      merge_shorter_than_s=dur_cfg["merge_shorter_than_s"],
                      min_state_s=dur_cfg["min_state_s"],
                      microarousal_max_s=dur_cfg["microarousal_max_s"],
                      theta_conditioned=scoring_config["theta"]["movement_conditioned"],
                      min_prominence_frac=thresh_cfg["min_prominence_frac"])

    result = {
        "times": times, "sw_pc1": sw_pc1,
        "sw_metric": sw_metric, "theta_metric": theta_metric, "motion_metric": motion_metric,
        "sw_channel": sw_channel, "theta_channel": th_channel, "sw_thresh": sw_thresh,
        **states,
        **extra_theta,
    }

    if imu_speed is not None:
        result["imu_speed"] = bin_max(imu_t[imu_valid], imu_speed[imu_valid], times)

    print("  [score_recording] done.", flush=True)
    return result


def _resolve_metrics(result, metrics):
    if metrics is not None:
        return [m for m in metrics if m in result]
    extra_theta = sorted(k for k in result if k.startswith("theta_metric_"))
    return [m for m in (*CORE_METRICS, *extra_theta) if m in result]


def summarize_run(result, metrics=None):
    """Distribution stats + pairwise Pearson correlations for the core per-timepoint metrics
    present in result (imu_speed and any extra theta_metric_<convention> series included only
    if score_recording was given imu_speed / theta_conventions)."""
    present = _resolve_metrics(result, metrics)
    stats = {}
    for m in present:
        x = np.asarray(result[m], dtype=float)
        ok = ~np.isnan(x)
        stats[m] = {
            "mean": float(np.mean(x[ok])), "std": float(np.std(x[ok])),
            "median": float(np.median(x[ok])),
            "q25": float(np.quantile(x[ok], 0.25)), "q75": float(np.quantile(x[ok], 0.75)),
        }

    correlations = {}
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            xa, xb = np.asarray(result[a], dtype=float), np.asarray(result[b], dtype=float)
            ok = ~np.isnan(xa) & ~np.isnan(xb)
            correlations[f"{a}__{b}"] = float(np.corrcoef(xa[ok], xb[ok])[0, 1])

    return {"stats": stats, "correlations": correlations}


def imu_wake_xcorr(result, max_lag_s=120.0):
    """Lagged cross-correlation between IMU translational speed and the neural wake proxy
    (-sw_pc1, z-scored), over the full session (unlike the notebook's illustrative-window
    version). Positive lag = speed leads the wake proxy. Requires 'imu_speed' and 'sw_pc1' in
    result. Returns (lags_s, xc); lags_s spacing follows result['times']'s actual bin width."""
    if "imu_speed" not in result:
        raise ValueError("imu_wake_xcorr requires score_recording to have been given IMU data")
    dt = np.median(np.diff(result["times"]))
    max_lag = int(round(max_lag_s / dt))

    speed = zscore(np.nan_to_num(result["imu_speed"], nan=0.0))
    wake = zscore(-result["sw_pc1"])
    n = len(speed)
    lags = np.arange(-max_lag, max_lag + 1)
    xc = np.array([
        np.dot(speed[max(0, l):n + min(0, l)], wake[max(0, -l):n + min(0, -l)]) / (n - abs(l))
        for l in lags
    ])
    return lags * dt, xc


def plot_summary(result, out_dir=None, metrics=None, xcorr_max_lag_s=120.0):
    """Histogram per metric + pairwise scatter/correlation grid for the core per-timepoint
    metrics present in result, plus an IMU-vs-wake-proxy cross-correlogram (imu_wake_xcorr) if
    both 'imu_speed' and 'sw_pc1' are present. Saves PNGs into out_dir if given, else returns
    the figures."""
    present = _resolve_metrics(result, metrics)

    fig_dist, axes = plt.subplots(1, len(present), figsize=(4 * len(present), 3.5))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, present):
        x = np.asarray(result[m], dtype=float)
        ax.hist(x[~np.isnan(x)], bins=50, density=True, alpha=0.6)
        ax.set_xlabel(m)
    axes[0].set_ylabel("density")
    fig_dist.tight_layout()

    n = len(present)
    fig_corr, cax = plt.subplots(n, n, figsize=(2.5 * n, 2.5 * n), squeeze=False)
    for i, a in enumerate(present):
        for j, b in enumerate(present):
            ax = cax[i][j]
            xa, xb = np.asarray(result[a], dtype=float), np.asarray(result[b], dtype=float)
            ok = ~np.isnan(xa) & ~np.isnan(xb)
            if i == j:
                ax.hist(xa[ok], bins=30, density=True, alpha=0.6)
            else:
                ax.scatter(xb[ok], xa[ok], s=2, alpha=0.3)
            if i == n - 1:
                ax.set_xlabel(b)
            if j == 0:
                ax.set_ylabel(a)
    fig_corr.tight_layout()

    figs = {"distributions": fig_dist, "correlations": fig_corr}

    if "imu_speed" in result and "sw_pc1" in result:
        lags_s, xc = imu_wake_xcorr(result, max_lag_s=xcorr_max_lag_s)
        peak = lags_s[np.argmax(xc)]
        fig_xcorr, ax = plt.subplots(figsize=(8, 3))
        ax.plot(lags_s, xc, lw=0.8)
        ax.axvline(peak, color="r", ls="--", label=f"peak lag = {peak:.0f} s")
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("lag (s): IMU speed relative to wake proxy (-sw_pc1)")
        ax.set_ylabel("cross-correlation")
        ax.legend()
        fig_xcorr.tight_layout()
        figs["imu_wake_xcorr"] = fig_xcorr

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(out_dir / f"{name}.png", dpi=150)
        return None
    return figs


def save_run(result, scoring_config, out_path, source_dirs):
    """Save the per-timepoint arrays (np.savez) plus a sidecar YAML recording scoring_config
    and, for each {stream_name: folder} in source_dirs, that derivative's canonical
    preprocessing params (its params.json) and seg_source (the folder's parent dir name,
    e.g. 'ProbeA_seg5-52')."""
    out_path = Path(out_path)
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    np.savez(out_path, **arrays)

    def to_native(v):
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.integer):
            return int(v)
        return v

    scalars = {k: to_native(v) for k, v in result.items() if not isinstance(v, np.ndarray)}

    sources = {}
    for stream, folder in source_dirs.items():
        folder = Path(folder)
        params = json.loads((folder / "params.json").read_text())
        sources[stream] = {"folder": folder.name, "seg_source": folder.parent.name, "params": params}

    sidecar = {"scoring_config": scoring_config, "result_scalars": scalars, "sources": sources}
    with open(out_path.with_suffix(".yml"), "w") as f:
        yaml.safe_dump(sidecar, f, sort_keys=False)


def notebook_extras(recording_lfp, recording_emg, scoring_config, times,
                     imu_t, imu_valid, imu_ang, imu_accel, result, cache_path=None, conventions=None):
    """Recompute the raw/per-convention signals behind the notebook-style diagnostic plots that
    score_recording's persisted result doesn't keep: the spectrogram on result['sw_channel'], each
    theta convention's own dip-selected channel + raw ratio (buzcode-style per-convention channel
    scan -- distinct from score_recording's shared peakTH channel), raw EMG score before smoothing,
    and IMU angular speed / |accel| / accel-variance binned onto `times`. If cache_path is given
    and already exists, loads and returns it instead of recomputing; otherwise computes and (if
    cache_path is given) saves it there for next time. conventions: THETA_BANDS names to scan
    (default: scoring_config's theta.conventions keys)."""
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path) as d:
            return {k: d[k] for k in d.files}

    fs = recording_lfp.get_sampling_frequency()
    spec_cfg = scoring_config["spectrogram"]
    spectrogram_kwargs = dict(window_s=spec_cfg["window_s"], step_s=spec_cfg["step_s"],
                              freq_min=spec_cfg["freq_min"], freq_max=spec_cfg["freq_max"],
                              n_freq_bins=spec_cfg["n_freq_bins"])
    channel_stride = scoring_config["channel_selection"]["stride"]
    conventions = conventions or list(scoring_config["theta"]["conventions"])

    print("  [notebook_extras] spectrogram on sw_channel...", flush=True)
    sw_trace = recording_lfp.get_traces(
        channel_ids=[recording_lfp.channel_ids[int(result["sw_channel"])]]).squeeze()
    sw_spec, sw_freqs, _ = log_spectrogram(sw_trace, fs, **spectrogram_kwargs)

    theta_own_channel, theta_own_ratio = {}, {}
    for conv in conventions:
        print(f"  [notebook_extras] theta channel scan ({conv})...", flush=True)
        ch, _, _ = select_theta_channel(recording_lfp, fs, conv, channel_stride, **spectrogram_kwargs)
        trace = recording_lfp.get_traces(channel_ids=[recording_lfp.channel_ids[ch]]).squeeze()
        spec, freqs, _ = log_spectrogram(trace, fs, **spectrogram_kwargs)
        theta_own_channel[conv] = ch
        theta_own_ratio[conv] = theta_ratio(spec, freqs, conv)

    print("  [notebook_extras] raw EMG score...", flush=True)
    emg_cfg = scoring_config["emg"]
    emg_shanks = recording_emg.get_probes()[0].shank_ids.astype(int)
    emg_pairs, _ = make_emg_pairs(
        emg_shanks, n_pairs=emg_cfg["n_pairs"], min_shank_dist=emg_cfg["min_shank_dist"], seed=emg_cfg["seed"])
    emg_score, emg_times = emg_from_lfp(recording_emg, emg_pairs, win_s=emg_cfg["window_s"])
    emg_b = np.interp(times, emg_times, emg_score)

    print("  [notebook_extras] IMU angular speed / |accel| / accel-variance...", flush=True)
    tv = imu_t[imu_valid]
    imu_ang_b = bin_max(tv, imu_ang[imu_valid], times)
    imu_accel_b = bin_max(tv, imu_accel[imu_valid], times)
    imu_accel_var_b = bin_var(tv, imu_accel[imu_valid], times)

    extras = {
        "sw_spec": sw_spec, "sw_freqs": sw_freqs,
        "emg_b": emg_b, "imu_ang_b": imu_ang_b, "imu_accel_b": imu_accel_b,
        "imu_accel_var_b": imu_accel_var_b,
        **{f"theta_own_channel_{c}": theta_own_channel[c] for c in conventions},
        **{f"theta_own_ratio_{c}": theta_own_ratio[c] for c in conventions},
    }
    if cache_path is not None:
        np.savez(cache_path, **extras)
    print("  [notebook_extras] done.", flush=True)
    return extras


def _pick_window(sw_metric, sw_thresh, dt, target_s=480.0, max_s=600.0):
    """Shortest centred window (target_s, growing up to max_s) around a NREM<->non-NREM
    transition that contains both above- and below-threshold epochs, for the Figure-1 overview
    plot. Falls back to the first target_s samples if sw_metric never crosses sw_thresh."""
    state = sw_metric > sw_thresh
    transitions = np.flatnonzero(np.diff(state.astype(int)) != 0)
    if len(transitions) == 0:
        return 0, min(len(sw_metric), int(round(target_s / dt)))
    mid = transitions[len(transitions) // 2]
    half = int(round(target_s / dt / 2))
    lo, hi = max(0, mid - half), min(len(sw_metric), mid + half)
    while state[lo:hi].mean() in (0.0, 1.0) and (hi - lo) * dt < max_s:
        half += int(round(30.0 / dt))
        lo, hi = max(0, mid - half), min(len(sw_metric), mid + half)
    return lo, hi


def plot_notebook_figures(result, extras, scoring_config, out_dir=None):
    """The notebook-style diagnostic figures (buzsaki_sleep_scoring.ipynb cells 18/19/29/30/34/42/
    53-54, plus a movement-conditioning comparison across all theta conventions extending cell 51):
    spectrogram, slow-wave metric + threshold, theta-convention comparison, Shin theta vs PC1,
    a Figure-1-style session overview, EMG vs slow-wave metric vs IMU speed, motion-candidate
    distributions, and per-convention movement-conditioned theta thresholds. result: score_recording
    output (or the loaded result.npz + result.yml scalars, merged). extras: notebook_extras output.
    Saves PNGs into out_dir if given (else returns the figures)."""
    times = result["times"]
    step_s = scoring_config["spectrogram"]["step_s"]
    smooth_win_s = scoring_config["smoothing"]["window_s"]
    bt_startbins = scoring_config["bimodal_threshold"]["startbins"]
    bt_maxbins = scoring_config["bimodal_threshold"]["maxbins"]
    thresh_cfg = scoring_config["threshold"]
    theta_cond = scoring_config["theta"]["movement_conditioned"]
    conventions = list(scoring_config["theta"]["conventions"])
    sw_channel = int(result["sw_channel"])
    sw_thresh = float(result["sw_thresh"])
    motion_thresh = float(result["motion_thresh"])
    figs = {}

    # Plot spectrogram of the selected channel.
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    pcm = ax.pcolormesh(times, extras["sw_freqs"], np.log10(extras["sw_spec"]), shading="auto", cmap="viridis")
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Spectrogram (sw_channel={sw_channel})")
    fig.colorbar(pcm, ax=ax, label="log10 power")
    figs["spectrogram"] = fig

    # Plot the buzcode-concordant slow-wave metric (15 s-smoothed, [0,1]) over time with its threshold.
    fig, ax = plt.subplots(figsize=(10, 3), dpi=150)
    ax.plot(times, result["sw_metric"], lw=0.5)
    ax.axhline(sw_thresh, color="r", ls="--", label=f"threshold = {sw_thresh:.3f}")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("slow-wave metric (smoothed, [0,1])")
    ax.set_title(f"Slow-wave metric (ch{sw_channel})")
    ax.legend()
    figs["sw_metric"] = fig

    # Compare the two conventions: z-scored log-ratio over time, and against each other.
    z_watson = zscore(np.log10(extras["theta_own_ratio_watson"]))
    z_shin = zscore(np.log10(extras["theta_own_ratio_shin"]))
    fig, (ax_ts, ax_sc) = plt.subplots(1, 2, figsize=(13, 3), dpi=150, gridspec_kw={"width_ratios": [3, 1]})
    ax_ts.plot(times, z_watson, lw=0.5, label=f"watson (ch {int(extras['theta_own_channel_watson'])})")
    ax_ts.plot(times, z_shin, lw=0.5, label=f"shin (ch {int(extras['theta_own_channel_shin'])})")
    ax_ts.set_xlabel("Time (s)"); ax_ts.set_ylabel("z-scored log theta ratio"); ax_ts.legend()
    ax_sc.scatter(z_watson, z_shin, s=2, alpha=0.3)
    r = np.corrcoef(z_watson, z_shin)[0, 1]
    ax_sc.set_xlabel("watson"); ax_sc.set_ylabel("shin"); ax_sc.set_title(f"r = {r:.3f}")
    fig.tight_layout()
    figs["theta_conventions"] = fig

    # Shin theta measure vs slow-wave PC1 (both z-scored for the overlay; same 10 s / 1 s time base).
    z_shin = zscore(np.log10(extras["theta_own_ratio_shin"]))
    z_sw = zscore(result["sw_pc1"])
    fig, (ax_sc, ax_ts) = plt.subplots(1, 2, figsize=(13, 3.5), dpi=150, gridspec_kw={"width_ratios": [1, 3]})
    ax_sc.scatter(z_sw, z_shin, s=2, alpha=0.3)
    r = np.corrcoef(z_sw, z_shin)[0, 1]
    ax_sc.set_xlabel("slow-wave PC1 (z)"); ax_sc.set_ylabel("shin theta (z)"); ax_sc.set_title(f"r = {r:.3f}")
    ax_ts.plot(times, z_sw, lw=0.5, label="slow-wave PC1")
    ax_ts.plot(times, z_shin, lw=0.5, label=f"Shin theta (ch {int(extras['theta_own_channel_shin'])})")
    ax_ts.set_xlabel("Time (s)"); ax_ts.set_ylabel("z-score"); ax_ts.legend()
    fig.tight_layout()
    figs["shin_vs_pc1"] = fig

    # Figure 1: PC1, Shin theta, and IMU movement over the session (shared 1 s time base).
    # NB: uses sw_metric (not raw sw_pc1) in panel 0 so sw_thresh -- a smoothed/[0,1] threshold --
    # is in the same units as what's plotted (the notebook's cell 34 overlays it on raw PC1, a
    # unit mismatch flagged in analysis.py's pc1_threshold docstring).
    dt = float(np.median(np.diff(times)))
    lo, hi = _pick_window(result["sw_metric"], sw_thresh, dt)
    sl = slice(lo, hi)
    t = times[sl]
    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(12, 9), dpi=150)
    ax[0].plot(t, result["sw_metric"][sl], lw=0.5)
    ax[0].axhline(sw_thresh, color="r", ls="--")
    ax[0].set_ylabel("slow-wave metric [0,1]")
    ax[1].plot(t, z_shin[sl], lw=0.5, color="C1")
    ax[1].set_ylabel("shin theta (z)")
    ax[2].plot(t, result["imu_speed"][sl], lw=0.5, color="C2")
    ax[2].set_ylabel("speed (m/s)")
    ax[3].plot(t, extras["imu_ang_b"][sl], lw=0.5, color="C3")
    ax[3].set_ylabel("ang. speed (deg/s)")
    ax[4].plot(t, extras["imu_accel_b"][sl], lw=0.5, color="C4")
    ax[4].set_ylabel("|accel| (m/s²)")
    ax[4].set_xlabel("Time (s)")
    fig.tight_layout()
    figs["figure1_overview"] = fig

    # EMG score vs slow-wave metric and IMU speed over the session (shared 1 s grid).
    z_emg = zscore(extras["emg_b"])
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 6), dpi=150)
    ax[0].plot(times, z_emg, lw=0.4); ax[0].set_ylabel("EMG (z)")
    ax[1].plot(times, result["sw_metric"], lw=0.4, color="C0")
    ax[1].axhline(sw_thresh, color="r", ls="--")
    ax[1].set_ylabel("slow-wave metric [0,1]")
    ax[2].plot(times, result["imu_speed"], lw=0.4, color="C2")
    ax[2].set_ylabel("speed (m/s)"); ax[2].set_xlabel("Time (s)")
    fig.suptitle(f"EMG vs slow-wave metric and IMU speed (sw_ch={sw_channel})")
    fig.tight_layout()
    figs["emg_vs_metrics"] = fig

    # Normalised motion distributions with their trough thresholds (MOV frac + conditioned THthresh per title).
    motion_candidates = {
        "EMG (tone)": extras["emg_b"],
        "IMU speed": np.nan_to_num(result["imu_speed"], nan=0.0),
        "IMU |accel|": np.nan_to_num(extras["imu_accel_b"], nan=0.0),
        "IMU angular": np.nan_to_num(extras["imu_ang_b"], nan=0.0),
        "IMU accel-var": np.nan_to_num(extras["imu_accel_var_b"], nan=0.0),
    }
    motion_metrics = {}
    for name, sig in motion_candidates.items():
        m = smooth_norm(sig, step_s=step_s, win_s=smooth_win_s)
        mt = find_thresh(m, thresh_cfg["method"], bt_startbins, bt_maxbins, thresh_cfg["kde_grid_n"],
                         thresh_cfg["min_prominence_frac"], label=f"motion:{name}")
        tht, mov = conditioned_theta_thresh(
            result["theta_metric"], result["sw_metric"], m, sw_thresh, mt, bt_startbins, bt_maxbins,
            thresh_cfg["method"], thresh_cfg["kde_grid_n"], theta_cond,
            thresh_cfg["min_prominence_frac"])
        motion_metrics[name] = (m, mt, mov, tht)
    fig, ax = plt.subplots(1, len(motion_metrics), figsize=(16, 3.2), dpi=150, sharey=True)
    for a, (name, (m, mt, mov, tht)) in zip(ax, motion_metrics.items()):
        a.hist(m, bins=50, density=True, alpha=0.5)
        a.axvline(mt, color="r", ls="--")
        a.set_title(f"{name}\nMOV={mov.mean():.2f}, THcond={tht:.3f}")
        a.set_xlabel("motion metric [0,1]")
    ax[0].set_ylabel("density")
    fig.tight_layout()
    figs["motion_distributions"] = fig

    # Effect of movement-conditioning on the theta threshold, extended across all theta conventions
    # (each on its own dip-selected channel), plus the motion (EMG) split.
    mov = result["mov"].astype(bool)
    fig, ax = plt.subplots(1, len(conventions) + 1, figsize=(4 * (len(conventions) + 1), 3.6), dpi=150)
    for i, conv in enumerate(conventions):
        tm = smooth_norm(extras[f"theta_own_ratio_{conv}"], step_s=step_s, win_s=smooth_win_s)
        tht, _ = conditioned_theta_thresh(
            tm, result["sw_metric"], result["motion_metric"], sw_thresh, motion_thresh, bt_startbins,
            bt_maxbins, thresh_cfg["method"], thresh_cfg["kde_grid_n"], theta_cond,
            thresh_cfg["min_prominence_frac"])
        ax[i].hist(tm, bins=50, density=True, alpha=0.4, label="all epochs")
        ax[i].hist(tm[~mov], bins=50, density=True, alpha=0.4, label="non-moving")
        ax[i].axvline(tht, color="r", ls="--", label=f"THthresh={tht:.3f}")
        ax[i].set_xlabel(f"{conv} theta metric [0,1]"); ax[i].legend()
    ax[0].set_ylabel("density")
    ax[-1].hist(result["motion_metric"], bins=50, density=True, alpha=0.5)
    ax[-1].axvline(motion_thresh, color="r", ls="--", label=f"MotionThresh={motion_thresh:.3f}")
    ax[-1].set_xlabel("motion metric (EMG) [0,1]"); ax[-1].legend()
    fig.tight_layout()
    figs["theta_movement_conditioning"] = fig

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(out_dir / f"{name}.png", dpi=150)
        return None
    return figs
