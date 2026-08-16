# Threshold methods: histogram trough vs KDE trough vs Gaussian mixture

Every threshold in the pipeline (`sw_thresh`, `motion_thresh`, `th_thresh`) is placed by the KDE
trough (`analysis.py:kde_thresh`, `config threshold.method: kde`). This documents the comparison
that led to that choice, against buzcode's histogram trough (`bz_BimodalThresh`) and a Gaussian
mixture.

Conclusions:

- KDE replaces the histogram trough. It agrees with the histogram wherever the histogram works,
  repairs several of its failures, and returns NaN with a warning rather than a degenerate threshold
  where no usable split exists.
- The GMM is not used. It cannot decline, and BIC prefers three components over two in 40 of 48
  cases, which removes the model-selection step that was its main attraction.
- The KDE prominence floor is 3% of peak density (`config threshold.min_prominence_frac`).
- ProbeA's theta ratio is unimodal under every method, scaling and floor tested. Its zero-REM result
  is a property of the signal, not of the thresholding.

Driver: `scripts/compare_threshold_methods.py`, writing to `{probe}/{variant}/threshold_methods/`.
Results below are from the 7.47 h `seg5-52` segment (~27-29k 1 s epochs per combo) unless noted.

## 1. The three methods

| | how the threshold is placed | bimodality signal |
|---|---|---|
| histogram | trough of the first 12→25-bin histogram resolving two peaks (`bimodal_thresh`) | Hartigan dip test |
| kde | `gaussian_kde` (Scott's rule) on a 512-point grid, deepest trough between the two most prominent peaks | returns no threshold if fewer than two modes clear the prominence floor |
| gmm | 2-component Gaussian mixture, threshold where the weighted component densities are equal | BIC over k=1,2,3 |

The histogram method was designed for far smaller recordings. Capping at 25 bins over ~28k epochs
leaves ~1100 samples per bin, so the extra data buys nothing and the coarse histogram is all that
stands between the pipeline and a usable density estimate.

Two notes on the GMM. Its equal-density crossing and the posterior = 0.5 crossover are the same
point, so this adds only BIC selection and `n_init=10` over the existing `pc1_threshold_gmm`. The
k=3 fit deliberately produces no threshold — three components have two crossings — and is fit purely
as a control for a non-Gaussian unimodal distribution being absorbed by extra components.

## 2. Results

48 series × combo cells per scaling, classifying each threshold as *no-threshold* (NaN),
*degenerate* (<1% or >99% of epochs above it, i.e. a cut through empty space), or *usable*:

| scaling | method | no-threshold | degenerate | usable |
|---|---|---|---|---|
| raw | histogram | 8 | 24 | 16 |
| raw | kde | 24 | 0 | 24 |
| raw | gmm | 0 | 0 | 48 |
| normalised | histogram | 3 | 15 | 30 |
| normalised | kde | 19 | 0 | 29 |
| normalised | gmm | 0 | 0 | 48 |

The column that matters is *degenerate*, not *usable*. KDE and the histogram find almost the same
number of usable thresholds (29 vs 30 normalised), but the histogram silently returns 15 that cut
through empty space and KDE returns none. KDE's larger no-threshold count is the same information
reported honestly. The GMM's 48/48 is not a virtue: it has no mechanism for declining.

A separate finding falls out of the raw-vs-normalised contrast. `smooth_norm` is a 15 s moving
average followed by min-max, and the min-max step cannot change any result here — it is
monotone-affine and all three methods are affine-equivariant. So that contrast isolates the
smoothing, which converts 8 no-threshold and 24 degenerate cases into 3 and 15. Much of the
pipeline's apparent bimodality is manufactured by the smoothing step rather than present in the
instantaneous metrics.

### 2a. How the histogram fails

`bimodal_thresh` produces an unusable threshold in 18 of 48 normalised cases. This was already known
for the IMU motion signals but is not confined to them — it fails on theta too, placing thresholds
above the 99.9th percentile so that REM is empty by construction:

| series | combo | threshold | frac above |
|---|---|---|---|
| theta_ratio_watson | ProbeA/cmr | 0.900 | 0.000 |
| theta_ratio_shin | ProbeA/cmr | 0.875 | 0.000 |
| theta_ratio_shin | ProbeB/nocmr | 0.875 | 0.000 |
| theta_ratio_shin_mod | ProbeB/cmr | 0.792 | 0.001 |

The recurring 0.875 and 0.792 values are histogram bin centres near the top of a 12-bin grid — the
signature of the zero-padded edge bin being taken as the second mode.

### 2b. Where KDE agrees, and where it declines

KDE tracks the histogram closely wherever the histogram is working and diverges only where it
breaks. On `slow_wave_pc1`, the one metric healthy under the histogram in all four combos, the two
agree to within 0.02 of threshold and 0.005 of movement fraction. On `imu_speed` the histogram's
above-the-99.9th-percentile threshold becomes a movement fraction of 0.64-0.66, comparable to EMG's
~0.70, in all four combos.

KDE's refusals concentrate exactly where independent evidence says the distribution has one mode:
all four `imu_accel_var` combos, the ProbeA theta series, and every `theta_log10_watson` case, all
with dip p > 0.99.

### 2c. Why the GMM is the wrong tool here

1. BIC prefers k=3 over k=2 in 40 of 48 cases, often by large margins (`imu_accel_var` raw:
   ΔBIC = 68,019). BIC-as-bimodality-gate would never stop at two components, which removes the main
   thing the GMM route was meant to add over the dip test.
2. The k=2 fits chase skew on the IMU series, splitting into a ~30%-weight narrow low component and
   a ~70% broad one. No series has exact zeros, so this is right skew compressing the bulk under
   min-max, not zero-inflation.
3. The resulting thresholds are displaced on those metrics: on `emg` (ProbeB/cmr) the GMM puts the
   movement fraction at 0.378 where both other methods agree on ~0.70.

It remains defensible on `slow_wave_pc1`, where k=2 finds two well-separated components and lands
within 0.06 of the other methods — the only place `pc1_threshold_gmm` was ever used.

### 2d. Bimodality testing is necessary but not sufficient

The dip test rejects unimodality in 32 of 48 normalised cases, including cases where the threshold is
degenerate. Non-unimodal is not the same as usefully bimodal. The diagnostic that actually catches
these failures is the threshold-versus-quantile comparison (`frac_above`), reported for every method
here.

## 3. The prominence floor

A KDE peak must rise a given fraction of peak density above its surroundings to count as a mode.
Some floor is required: without one, floating-point ripple in the near-zero tails registers as a
mode and a unimodal Gaussian yields a threshold out in its own tail — the KDE analogue of the
zero-padded edge bin in §2a.

The floor was swept at 1%, 2% and 3% across all three theta conventions and all three candidate
pools on the 22 h recording. It separates the pools, not the conventions, and the separation is wide
— second-mode prominence on ProbeB:

| pool | watson | shin | shin_mod |
|---|---|---|---|
| `~nrem & ~mov` (cmr / nocmr) | 0.223 / 0.127 | 0.114 / 0.284 | 0.283 / 0.288 |
| `~mov` (cmr / nocmr) | 0.024 / 0.018 | 0.007 / 0.002 | 0.002 / 0.001 |

Nothing lands between 0.024 and 0.114, so the floor's exact position inside that gap is not critical.
3% is chosen because it rejects the `~mov` pool, which should never have been thresholded: at 1% and
2%, `watson` on `~mov` returns 0.5675, the threshold that placed REM on the NREM/non-NREM split under
buzcode's fallback ordering. At 3% it returns NaN.

Raising the floor changes no production threshold — all three conventions give identical results at
1%, 2% and 3% on the `~nrem & ~mov` pool, the only pool production uses. The floor is insurance
against thresholding a pool with no real split.

Two consequences elsewhere. The floor rejects `imu_accel_var`, whose second mode sits at 0.0070;
that metric is therefore not flatly unimodal but has a mode too shallow to accept, and its retirement
rests on the motion-source evidence rather than on this alone. And on the unconditioned full
distribution `shin` goes from 0.1194 to NaN between 1% and 2% — not a live path, since conditioning
is unconditional, but it explains why unconditioned thresholds look unstable.

Genuinely unimodal cases are unambiguous at any floor: `theta_ratio_watson` on ProbeA/cmr has a
second-mode prominence of 0.0002 and returns NaN at every floor tested.

## 4. Series compared, and two caveats on theta

Twelve series: `slow_wave_pc1`, `theta_ratio_{watson,shin,shin_mod}`,
`theta_log10_{watson,shin,shin_mod}`, `emg`, `imu_speed`, `imu_accel`, `imu_angular`,
`imu_accel_var`.

- Linear theta is the pipeline-faithful one. `scoring.py` and buzcode both threshold the raw ratio
  with no log. The `theta_log10_*` panels are carried only because the repo uses log10 in channel
  selection; that is not what the classifier sees.
- These are own-channel ratios (`theta_own_ratio_*`), each convention on its own dip-selected
  channel — not `result.npz`'s shared-peakTH `theta_metric_*`. The two are different signals, not
  one signal rescaled, which is why numbers here do not always match the production thresholds in
  `sleep_classification_algorithm.md`.

`emg` and all IMU series are identical between `lfp_cmr` and `lfp_nocmr`, as expected: both LFP
variants are paired with the same `emg` derivative and the IMU does not depend on LFP referencing.

## 5. Caveats

- No ground truth. Every judgement here is whether a threshold splits the distribution somewhere
  defensible, not whether it produces correct sleep states.
- The KDE bandwidth is Scott's rule and remains unvalidated. Rule-of-thumb bandwidths assume a
  roughly Gaussian density and are biased toward over-smoothing the bimodal structure being detected.
  Silverman's critical-bandwidth test would unify the bimodality test and threshold placement into
  one computation; it is not being pursued, as it is not expected to change these conclusions and
  has no maintained Python implementation.
- The 48-case sweep was run on `seg5-52`. The floor sweep in §3 is from `seg5-148`.
- All ProbeB results rest on data now known to be truncated; see `docs/missing_data_note.md`.
