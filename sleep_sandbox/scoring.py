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
    broadband_pc1, select_channel_by_dip, select_theta_channel_peak, log_spectrogram,
    smooth_norm, bimodal_thresh, theta_ratio, conditioned_theta_thresh, bin_max, zscore,
    make_emg_pairs, emg_from_lfp,
)

CORE_METRICS = ("sw_metric", "theta_metric", "motion_metric", "imu_speed")


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
    sw_thresh = bimodal_thresh(sw_metric, bt_startbins, bt_maxbins)

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
    motion_thresh = bimodal_thresh(motion_metric, bt_startbins, bt_maxbins)

    print("  [score_recording] movement-conditioned theta threshold...", flush=True)
    th_thresh, movtimes = conditioned_theta_thresh(
        theta_metric, sw_metric, motion_metric, sw_thresh, motion_thresh, bt_startbins, bt_maxbins)

    not_nrem = sw_metric < sw_thresh
    low_tone = motion_metric < motion_thresh
    rem_cand = not_nrem & low_tone
    rem = rem_cand & (theta_metric > th_thresh)

    result = {
        "times": times, "sw_pc1": sw_pc1,
        "sw_metric": sw_metric, "theta_metric": theta_metric, "motion_metric": motion_metric,
        "nrem": ~not_nrem, "mov": movtimes, "rem_cand": rem_cand, "rem": rem,
        "sw_channel": sw_channel, "theta_channel": th_channel,
        "sw_thresh": sw_thresh, "motion_thresh": motion_thresh, "th_thresh": th_thresh,
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
