# Sleep classification approach and exploratory analysis

## Data sources

**LFP**
- Slow wave power (0.5-1Hz) —— high = NREM
- Delta/theta power ratio (1-4Hz vs 4-8Hz) —— high = NREM, low with no motion = REM
- Spindle power (12-16Hz) —— high = NREM
- Gamma —— high = wake/REM

**MUA**
— Sharp ON/OFF states = NREM (corresponding to SWS)

**IMU**
- High motion energy = awake
- Low motion energy = quiet wakefulness?
- Can you correlate sudden shifts in IMU signal to onset(/offset) of REM sleep?


| State              | Slow wave power | Delta power | Theta power | Delta/theta | Gamma power | Spindle power | Movement | Ripples  |
|--------------------|-----------------|-------------|-------------|-------------|-------------|---------------|----------|----------|
| **Awake (active)** | Low             | Low         | Medium/high | Low         | High        | Low           | High     | Few/none |
| **Awake (quiet)**  |                 |             |             |             |             |               | Low      | Many     |
| **NREM**           | High            | High        | Low         | High        | Low         | High          | Very low | Many     |
| **REM**            | Low             | Low         | High        | Low         | High        | Low           | None     | Few/none |


**Buzsaki strategy**

For all analyses in this section, recordings are downsampled to 1250 Hz

Intracranial EMG
- Bandpass (Butterworth, 4th order) filter in 300-600Hz range
- Subselect pairs of random, good channels ≥ 2 shanks apart
- Take mean of Pearson's R in 500ms windows --> **this is the EMG score**
- To implement later: 25Hz shoulder during bandpass (what does this mean?)

Low-frequency spectral features
1. Short-time FFT: 10s window, 1s step, evaluated at log-spaced frequencies between 1–100 Hz
2. z-score
3. PCA
4. Pick PC1. If negative loading on <20Hz frequencies, flip sign.
5. Pick trough in bimodal distribution as threshold between NREM and Wake/REM.

Narrowband theta (theta:denominator power ratio)
- Shin et al. (2026): theta (6-12Hz) / delta (1-4Hz).
- Watson et al. (2016): theta (5-10Hz) / broadband (2-16Hz).
- Both conventions are implemented side by side (`THETA_BANDS`) for comparison.

**Implementation** (reuses the STFT instead of Butterworth for narrowband theta
to stay consistent with buzcode and the slow-wave metric):
1. Band power comes from the same shared log-spectrogram as the slow-wave metric
   (`log_spectrogram`: 10 s Hann window, 1 s step, power interpolated onto a log-spaced 1-100 Hz
   grid), integrated over each band (trapezoid on the freq grid).
2. Channel selection (buzcode-style). buzcode picks the theta channel whose theta-ratio
   distribution separates best into two modes. We quantify that separation exactly
   as for the slow-wave channel — Hartigan's dip test on the `log10` theta ratio (ratios are
   right-skewed, so log first) — and take the max-dip channel. This replaces "highest theta power",
   which can just pick the noisiest/highest-amplitude channel rather than the most theta-modulated.
3. Ratio computed per convention in 10 s / 1 s windows on the chosen channel, aligned to `sw_times`.

**Classification algorithm**

## Implementation notes & deviations from Watson et al.

### Channel selection
- Channel is chosen automatically as the one with the strongest bimodal split in its
  broadband slow-wave (PC1) score, scored with Hartigan's dip test,
  scanning every Nth channel. `manual_channel` overrides the automatic pick.
- This follows buzcode `SleepScoreMaster`, which selects the most-bimodal channel.

### Common median reference (CMR)
- CMR is currently used for the slow-wave and theta metrics.
  This appears to reduce spurious NREM bouts in this data.
- May change this in future as slow waves are spatially broad and CMR may reduce this.
  Could use something like Watson et al.'s movement criterion (>60s) to filter spurious NREM bouts.

### IMU movement (BNO055) cross-check
- All BNO055 blocks are concatenated and reduced to translational speed, angular speed, and 
  `|linear accel|`, then binned to the 1 s slow-wave grid (max per bin) and compared to PC1 + Shin theta.
- Clock alignment: `Bno055_Clock_N` and `ProbeB_Clock_N` are the same free-running 250 MHz ONIX counter. 
  Each block's IMU clock span is mapped linearly onto its ephys segment's nominal sample-time span.
- Verified: binned IMU speed vs a wake proxy cross-correlates with peak lag ≈ 0 both, early and in a 
  tail window, meaning no cumulative drift.
- The aligned IMU timeline ends ~53 s past the LFP because the final segment's amplifier data stops
  before its clock, so that block's tail maps to no neural counterpart; those samples are dropped when
  binning. This is purely cosmetic; see `io_and_processing.md` §2b.
- Binning maps each IMU sample to the nearest slow wave PC bin (10 s STFT windows → 10 s smoothing).

**Current status: diagnostic-only.** IMU feeds `imu_wake_xcorr` (a QC cross-correlation against
the wake proxy) and is reported alongside the core metrics, but is not an input to any threshold
in `score_recording` -- `sw_thresh`/`motion_thresh`/`th_thresh` are all computed from LFP/EMG only.
Whether IMU (or some derivative of it) should ever gate a classification threshold, vs. remain
permanently QC-only, is an open question, not yet decided.

Substituting IMU for EMG as the motion input is under active consideration but has **no buzcode
precedent** to lean on: checked across all buzcode branches, `SleepScoreMaster`'s
`MotionSource='Accelerometer'` path (`bz_GetAccelerometerMotion.m`) is unimplemented everywhere,
and the one real accelerometer-extraction function that does exist (`bz_getIntanAccel.m`, `RHdev`
branch only) was never wired into a thresholding pipeline. See `ANALYSIS_FRAMEWORK.md` for the
full research writeup. The mechanistic concern: EMG reflects muscle tone, and REM is defined by
atonia, which IMU-derived movement/acceleration signals cannot straightforwardly detect (a still
animal in quiet wake looks the same to an IMU as one in REM atonia). Any IMU-based substitute
needs direct empirical validation against the EMG-based `rem` mask, not adoption by analogy.

## Broadband LFP slow-wave metric — implementation approach

Reference: Watson et al. 2016 / buzcode `SleepScoreLFP`. 
Implemented in `notebooks/buzsaki_sleep_scoring.ipynb`.

Input: One LFP channel at 1250 Hz, on the CMR variant.

0. Channel selection
- Scan every Nth channel; for each, run steps 1–4 and score the PC1 distribution's bimodality with
  Hartigan's dip test. Pick the max-dip channel.

1. Short-time FFT
- Get power spectrum for 10s Hann windows, 1s step.
- Interpolate each time bin's power onto a log-spaced frequency grid, 1–100 Hz, 100 bins.

2. log10 + z-score
- `log10` the power (compress 1/f dynamic range so low frequencies don't dominate),
- z-score each frequency across time.

3. PCA
- PCA over z-scored spectrogram with time bins as samples, the 100 frequencies as features.
- Take PC1. This is the dominant covarying spectral pattern (low-freq-up / high-freq-down mode
  that tracks NREM).
- Fit the full PCA and extract PC1 scores over time, PC1 loading per frequency.

4. Orient PC1
- Check mean PC1 loading on <20 Hz bins; if negative, flip sign of the component and its scores.

5. Threshold between the two modes.
- The distribution of PC1-over-time is bimodal (NREM vs Wake/REM). Split at the boundary
  between the two modes; above → NREM candidate, below → Wake/REM candidate.
- Method: fit a 2-component Gaussian mixture model to the PC1 scores and set the
  threshold at the posterior crossover (decision boundary) between the two component means.
