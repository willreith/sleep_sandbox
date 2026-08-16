"""Compare three ways of splitting a scoring metric into two modes, on every metric that appears
in distributions.png / motion_distributions.png:

  current   bimodal_thresh (buzcode bz_BimodalThresh histogram trough) + Hartigan dip test
  kde       gaussian_kde, deepest trough between the two most prominent peaks
  gmm       2-component Gaussian mixture, threshold at the equal-density crossing; BIC over
            k=1,2,3 as the bimodality signal (k=3 is diagnostic only -- see below)

Each metric is evaluated twice, raw and smooth_norm'd. NB the [0,1] step of smooth_norm cannot
change any result here: min-max is monotone-affine, and all three methods are affine-equivariant
(dip is invariant, bimodal_thresh's histogram spans the data range, KDE bandwidth scales with the
data, and GMM's per-k BIC shifts by a common log-Jacobian so dBIC is unchanged). The raw-vs-
normalised contrast is therefore purely a test of the 15 s moving average -- the only step that
can create or destroy modes.

k=3 deliberately yields no threshold: three components have two crossing points, so "the"
threshold is undefined. It is fit as a control for the failure mode that motivates this whole
comparison -- a heavy-tailed unimodal distribution getting fit by extra Gaussians. A low-weight,
narrow component parked in the tail (at k=2 or k=3) means the mixture is modelling skew, not
states, so per-component weight/mean/sd are reported alongside BIC.

Reads saved derivatives only (result.npz + result_extras.npz); no raw LFP or IMU. Writes
threshold_comparison_{diagnostics.json,raw.png,normalised.png} into each
{out_base}/{probe}/{variant}/threshold_methods/.
See docs/threshold_comparison.md.

Usage: python compare_threshold_methods.py [--out-base data/derivatives]
"""

import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from dotenv import load_dotenv
from diptest import diptest

from sleep_sandbox.analysis import smooth_norm, bimodal_thresh, kde_thresh

repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")

parser = argparse.ArgumentParser()
parser.add_argument("--seg", required=True,
                     help="segment range, e.g. 'seg5-148'; reads/writes under {out_base}/{seg}/")
parser.add_argument("--out-base", type=Path, default=repo_root / "data" / "derivatives")
args = parser.parse_args()

with open(repo_root / "config/sleep_scoring.yml") as f:
    scoring_config = yaml.safe_load(f)

step_s = scoring_config["spectrogram"]["step_s"]
smooth_win_s = scoring_config["smoothing"]["window_s"]
bt_startbins = scoring_config["bimodal_threshold"]["startbins"]
bt_maxbins = scoring_config["bimodal_threshold"]["maxbins"]
CONVENTIONS = list(scoring_config["theta"]["conventions"])

GRID_N = 512          # evaluation grid for the KDE / plotted mixture densities
KDE_MIN_PROMINENCE_FRAC = 0.01   # a peak must clear this fraction of peak density to count as a mode
GMM_KS = (1, 2, 3)
GMM_N_INIT = 10       # EM is a local optimiser; pc1_threshold_gmm's single init can land on a
                      # degenerate fit (one component collapsed onto a few tail points)


def kde_trough(x, grid):
    """Threshold via analysis.kde_thresh (the same code the pipeline uses), plus the density curve
    for plotting. Returns (threshold, density, info)."""
    dens = gaussian_kde(x)(grid)
    peaks, _ = signal.find_peaks(dens, prominence=KDE_MIN_PROMINENCE_FRAC * dens.max())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # NaN is an expected outcome here
        thresh = kde_thresh(x, grid_n=len(grid), min_prominence_frac=KDE_MIN_PROMINENCE_FRAC)
    return float(thresh), dens, {"n_peaks": int(len(peaks)),
                                 "peak_positions": [float(grid[p]) for p in peaks]}


def gmm_crossing(gmm):
    """Threshold where the two weighted component densities are equal, i.e. posterior = 0.5.
    Searched between the component means, which is the root that separates them (unequal variances
    give a second root outside that interval)."""
    order = np.argsort(gmm.means_.ravel())
    lo_mean, hi_mean = gmm.means_.ravel()[order]
    grid = np.linspace(lo_mean, hi_mean, 1000)
    post_hi = gmm.predict_proba(grid.reshape(-1, 1))[:, order[1]]
    return float(grid[np.argmin(np.abs(post_hi - 0.5))])


def gmm_components(gmm):
    """Per-component (weight, mean, sd), sorted by mean. A low weight with a small sd is the
    tail-catcher signature: the component is absorbing skew rather than marking a state."""
    order = np.argsort(gmm.means_.ravel())
    sds = np.sqrt(gmm.covariances_.ravel())
    return [{"weight": float(gmm.weights_[i]), "mean": float(gmm.means_.ravel()[i]),
             "sd": float(sds[i])} for i in order]


def analyse(x, grid):
    """All three methods on one series."""
    dip, dip_p = diptest(x)
    kde_t, dens, kde_info = kde_trough(x, grid)

    fits = {k: GaussianMixture(n_components=k, n_init=GMM_N_INIT, random_state=0).fit(x.reshape(-1, 1))
            for k in GMM_KS}
    bic = {k: float(f.bic(x.reshape(-1, 1))) for k, f in fits.items()}

    return {
        "n": int(x.size),
        "quantiles": {q: float(np.quantile(x, q)) for q in (0.5, 0.9, 0.99, 0.999)},
        "current": {"thresh": float(bimodal_thresh(x, bt_startbins, bt_maxbins)),
                    "dip": float(dip), "dip_p": float(dip_p)},
        "kde": {"thresh": kde_t, "bandwidth_factor": float(gaussian_kde(x).factor), **kde_info},
        "gmm": {"thresh": gmm_crossing(fits[2]), "bic": bic,
                "bic_1_minus_2": bic[1] - bic[2], "bic_2_minus_3": bic[2] - bic[3],
                "components": {k: gmm_components(fits[k]) for k in (2, 3)}},
    }, dens, fits


def frac_above(x, t):
    return float(np.mean(x > t)) if np.isfinite(t) else None


for probe in ["ProbeA", "ProbeB"]:
    for variant in ["lfp_cmr", "lfp_nocmr"]:
        out_dir = args.out_base / args.seg / probe / variant
        fig_dir = out_dir / "threshold_methods"
        fig_dir.mkdir(parents=True, exist_ok=True)
        print(f"--- {probe} {variant} ---", flush=True)

        result = dict(np.load(out_dir / "result.npz"))
        extras = dict(np.load(out_dir / "result_extras.npz"))

        # Every series in distributions.png + motion_distributions.png, deduped: result's
        # theta_metric is bit-identical to theta_metric_watson, and motion_metric is
        # smooth_norm(emg_b). Theta uses each convention's own dip-selected channel (the only raw
        # ratio saved; result's theta_metric_* are on the shared peakTH channel, so they are not
        # the same signal normalised), carried both linear and log10.
        series = {"slow_wave_pc1": result["sw_pc1"]}
        for conv in CONVENTIONS:
            # Linear is the pipeline-faithful one: score_recording thresholds
            # smooth_norm(theta_ratio(...)) with no log (scoring.py), as does buzcode
            # ClusterStates_GetMetrics. log10 is carried alongside only because the repo uses it
            # elsewhere (select_theta_channel's dip test, theta_conventions.png) -- it is not what
            # the classifier sees.
            series[f"theta_ratio_{conv}"] = extras[f"theta_own_ratio_{conv}"]
            series[f"theta_log10_{conv}"] = np.log10(extras[f"theta_own_ratio_{conv}"])
        series.update({
            "emg": extras["emg_b"],
            "imu_speed": result["imu_speed"],
            "imu_accel": extras["imu_accel_b"],
            "imu_angular": extras["imu_ang_b"],
            "imu_accel_var": extras["imu_accel_var_b"],
        })

        diagnostics = {"n_epochs": int(result["times"].size), "series": {}}
        plot_data = {"raw": {}, "normalised": {}}

        for name, raw in series.items():
            raw = np.asarray(raw, dtype=float)
            n_nan = int(np.isnan(raw).sum())
            raw = raw[np.isfinite(raw)]
            versions = {"raw": raw, "normalised": smooth_norm(raw, step_s=step_s, win_s=smooth_win_s)}

            diagnostics["series"][name] = {"n_dropped_nonfinite": n_nan}
            for scaling, x in versions.items():
                grid = np.linspace(x.min(), x.max(), GRID_N)
                stats, dens, fits = analyse(x, grid)
                stats["frac_above"] = {m: frac_above(x, stats[m]["thresh"])
                                       for m in ("current", "kde", "gmm")}
                diagnostics["series"][name][scaling] = stats
                plot_data[scaling][name] = (x, grid, dens, fits, stats)

        with open(fig_dir / "threshold_comparison_diagnostics.json", "w") as f:
            json.dump(diagnostics, f, indent=2)

        for scaling, panels in plot_data.items():
            fig, axes = plt.subplots(len(panels), 1, figsize=(9, 2.6 * len(panels)))
            for ax, (name, (x, grid, dens, fits, stats)) in zip(np.atleast_1d(axes), panels.items()):
                ax.hist(x, bins=60, density=True, alpha=0.35, color="0.5", label="histogram")
                ax.plot(grid, dens, color="k", lw=1.2, label="KDE")

                # Weighted component densities: k=2 solid, k=3 dotted. k=3 draws no threshold --
                # two crossings leave it undefined; it is here to expose tail-catcher components.
                for k, ls in ((2, "-"), (3, ":")):
                    f_k = fits[k]
                    sds = np.sqrt(f_k.covariances_.ravel())
                    for w, mu, sd in zip(f_k.weights_, f_k.means_.ravel(), sds):
                        comp = w * np.exp(-0.5 * ((grid - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
                        ax.plot(grid, comp, ls=ls, lw=0.9, color="tab:purple", alpha=0.8)

                for m, color in (("current", "tab:red"), ("kde", "tab:blue"), ("gmm", "tab:purple")):
                    t = stats[m]["thresh"]
                    if np.isfinite(t):
                        ax.axvline(t, color=color, ls="--", lw=1.3,
                                   label=f"{m} = {t:.3g} (frac above {stats['frac_above'][m]:.3f})")

                g = stats["gmm"]
                ax.set_title(f"{name}   dip={stats['current']['dip']:.4f} p={stats['current']['dip_p']:.3f}"
                             f"   BIC 1-2={g['bic_1_minus_2']:.0f}, 2-3={g['bic_2_minus_3']:.0f}",
                             fontsize=9)
                ax.legend(fontsize=7)
                ax.set_ylabel("density", fontsize=8)
            fig.suptitle(f"{probe} {variant}: threshold methods, {scaling} "
                         f"(purple solid = 2-component GMM, dotted = 3-component)")
            fig.tight_layout()
            fig.savefig(fig_dir / f"threshold_comparison_{scaling}.png", dpi=150)
            plt.close(fig)

        for name, d in diagnostics["series"].items():
            print(f"  {name:22s} " + "  ".join(
                f"{sc[:4]}: cur={d[sc]['current']['thresh']:.3g} kde={d[sc]['kde']['thresh']:.3g} "
                f"gmm={d[sc]['gmm']['thresh']:.3g}" for sc in ("raw", "normalised")), flush=True)
        print(f"--- {probe} {variant} done -> {fig_dir} ---", flush=True)

print("ALL THRESHOLD COMPARISONS COMPLETE")
