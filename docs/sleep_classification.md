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

Intracranial EMG
- Bandpass (Butterworth, 4th order) filter in 300-600Hz range
- Subselect pairs of random, good channels ≥ 2 shanks apart
- Take mean of Pearson's R in 500ms windows --> **this is the EMG score**

Low-frequency spectral features
1. Downsample to 1250 Hz
2. Short-time FFT: 10s window, 1s step, evaluated at log-spaced frequencies between 1–100 Hz
3. z-score
4. PCA
5. Pick PC1. If negative loading on <20Hz frequencies, flip sign.
6. Pick trough in bimodal distribution as threshold between NREM and Wake/REM.

Narrowband theta (theta:delta ratio)
- Shin et al. (2026) uses delta (1-4Hz) and theta (6-12Hz — quite high, should be closer to 4-8Hz to avoid contamination by spindle band).
– Watson et al. (2016) uses 5-10Hz:2-16Hz.
1. Bandpass filter for relevant frequency bands (4th order Butterworth filter).
2. Pick channel with highest theta power.
3. 

## Implementation notes & deviations from Watson et al.

### Channel selection
- Channel is chosen automatically as the one with the **strongest bimodal split** in its
  broadband slow-wave (PC1) score, scored with **Hartigan's dip test** (`diptest` package),
  scanning every Nth channel (`channel_stride`). `manual_channel` overrides the automatic pick.
- This follows buzcode `SleepScoreMaster`, which selects the most-bimodal channel. Scored on
  the **no-CMR** LFP variant (see CMR note below).
- The earlier default was a fixed channel 73 (theta-picked); it now survives only as a manual
  override, not the default.

### Common median reference (CMR)
- **CMR removed** for the LFP slow-wave metric.
- Cortical slow waves are spatially broad and largely *shared* across channels, so a
  global-median reference can subtract the very signal the slow-wave metric relies on.
  Watson's slow-wave LFP is not aggressively re-referenced. This deviates from the saved
  `lfp_preprocessed` recording (which was global-median CMR'd). Revisit if the bimodal
  split is weak.

## Broadband LFP slow-wave metric — implementation approach

Reference: Watson et al. 2016 / buzcode `SleepScoreLFP`. Status: channel selection + steps
1–5 implemented in `notebooks/buzsaki_sleep_scoring.ipynb`.

**Input / prep.** One LFP channel at 1250 Hz, on the **no-CMR** variant. The channel is chosen
by Step 0 below (or `manual_channel`). See channel-selection and CMR caveats above.

**Step 0 — Channel selection.** *(done)*
- Scan every Nth channel (`channel_stride`); for each, run steps 1–4 and score the PC1
  distribution's bimodality with **Hartigan's dip test** (`diptest`). Pick the max-dip channel.
- Steps 1–4 are collapsed into a `broadband_pc1(trace, fs)` helper reused by the scan and by
  the final run on the winning channel. `manual_channel` overrides the automatic pick.

**Step 1 — Short-time FFT.** *(done)*
- 10 s Hann window, 1 s step (~90% overlap) → power spectrogram.
- Interpolate each time bin's power onto a **log-spaced** frequency grid, 1–100 Hz, 100 bins
  (`np.logspace(0, 2, 100)`). Output: raw power, `[freq=100, time]`.
- Why log-spaced: roughly equal weight per octave, matching the ~1/f structure of LFP.

**Step 2 — log10 + z-score.** *(done)*
- `log10` the power (compresses the 1/f dynamic range so low frequencies don't dominate),
  then **z-score each frequency across time** (zero-mean/unit-variance per frequency row).
  This is buzcode's `zscore(log10(spec))`.

**Step 3 — PCA.** *(done)*
- PCA over the z-scored spectrogram with **time bins as samples, the 100 frequencies as
  features** (transpose to `[time, freq]`).
- Take **PC1** — the dominant covarying spectral pattern (low-freq-up / high-freq-down mode
  that tracks NREM). This is the `broadbandSlowWave` score, one value per second.
- Uses `sklearn.decomposition.PCA` (scikit-learn added to the env). Fits the full PCA;
  `sw_pc1` = PC1 scores over time, `sw_pc1_loading` = PC1 loading per frequency.

**Step 4 — Orient PC1.** *(done)*
- PCA sign is arbitrary; fix a convention so **high PC1 = more slow wave**. Check the mean
  PC1 loading on **<20 Hz** bins; if negative, flip the sign of the component and its scores.

**Step 5 — Threshold between the two modes.** *(done)*
- The distribution of PC1-over-time is bimodal (NREM vs Wake/REM). Split at the boundary
  between the two modes; above → NREM candidate, below → Wake/REM (later split by theta + EMG).
- Method: fit a **2-component `sklearn.mixture.GaussianMixture`** to the PC1 scores and set the
  threshold at the **posterior crossover** (decision boundary) between the two component means.
  `sw_nrem = sw_pc1 > sw_threshold`. (buzcode's analogue is `bz_BimodalThresh`.)

