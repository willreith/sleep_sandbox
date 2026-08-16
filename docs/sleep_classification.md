# Sleep classification: background and approach

Physiological background and the rationale for the method. The pipeline as implemented, and results,
are in `docs/sleep_classification_algorithm.md`.

## State signatures

| State | Slow wave | Delta | Theta | Delta/theta | Gamma | Spindle | Movement | Ripples |
|---|---|---|---|---|---|---|---|---|
| Awake (active) | Low | Low | Medium/high | Low | High | Low | High | Few/none |
| Awake (quiet) | | | | | | | Low | Many |
| NREM | High | High | Low | High | Low | High | Very low | Many |
| REM | Low | Low | High | Low | High | Low | None | Few/none |

The pipeline uses three of these columns: a broadband slow-wave score, a narrowband theta ratio, and
a movement/tone signal. Spindles, gamma, ripples and MUA ON/OFF structure are all diagnostic of
state in principle but are not inputs to the classifier.

Two of these signatures do most of the work, and one is the hard case. NREM separates cleanly on
slow-wave power. REM does not separate on any single band: it is defined by high theta *together
with* atonia, so a movement signal alone cannot distinguish REM from quiet wake, and theta alone
cannot distinguish REM from active exploration. This is why the theta threshold is conditioned on
the non-moving, non-NREM population rather than taken globally.

## The Watson / buzcode approach

Reference: Watson et al. 2016, buzcode `SleepScoreMaster` / `SleepScoreLFP`. Recordings are
downsampled to 1250 Hz. Three metrics are derived and each is split at the trough between the two
modes of its own distribution — Watson et al.'s supplementary states the principle as *"all
thresholds placed at dip between modes of bimodal data"*.

**Broadband slow-wave score.** Short-time FFT (10 s window, 1 s step, log-spaced 1-100 Hz), `log10`
to compress the 1/f dynamic range, z-score per frequency, then PCA over time bins. PC1 is the
dominant low-frequency-up / high-frequency-down mode that tracks NREM; its sign is fixed by checking
the mean loading below 20 Hz. High values are NREM.

**Narrowband theta ratio.** Theta band power over a denominator band, computed on the same
spectrogram rather than by separate Butterworth filtering, so it stays consistent with the slow-wave
metric. Two published conventions are carried side by side — Watson et al. (5-10 Hz / 2-16 Hz) and
Shin et al. (6-12 Hz / 1-4 Hz) — plus a local variant combining the two (5-10 Hz / 1-4 Hz). The
ratio is thresholded linearly, not log-scaled, as in buzcode.

**Intracranial EMG proxy.** Muscle activity appears as correlated high-frequency signal across
distant recording sites, so the mean Pearson correlation between well-separated channel pairs in the
300-600 Hz band indexes muscle tone without a dedicated electrode (Schomburg et al. 2014). Computed
on short windows and interpolated onto the 1 s grid.

**Channel selection.** Rather than picking the highest-power channel, which can select the noisiest
site, each candidate is scored by how well its metric separates into two modes, using Hartigan's dip
statistic. Applied independently to the slow-wave channel (on PC1) and the theta channel (on the
`log10` theta ratio, since ratios are right-skewed).

## Deviations from Watson et al. and buzcode

Each of these is documented with its supporting evidence in `sleep_classification_algorithm.md`.

- **KDE trough rather than a histogram trough** for every threshold. The histogram method was
  designed for far smaller recordings and returns a threshold cutting through empty space in a
  substantial fraction of cases here. See `docs/threshold_comparison.md`.
- **The theta threshold is taken on the `~nrem & ~mov` pool** — quiet wake plus REM — always, where
  buzcode takes it on `~mov` and excludes NREM only as a fallback. The motion threshold here is very
  nearly a sleep/wake detector, so `~mov` is mostly NREM and its trough marks the wrong boundary.
- **The non-CMR variant is used for the slow-wave metric.** Slow waves are spatially synchronous, so
  subtracting the across-channel median removes much of what PC1 is built to detect. CMR is retained
  as a control. Note that `run_scoring.py` pairs one LFP variant with both the slow-wave and theta
  metrics, so this choice currently carries theta with it; whether theta wants the same referencing
  has not been tested separately.
- **EMG derivation differs in band and pairing** from `bz_EMGFromLFP`: a plain 300-600 Hz bandpass
  rather than buzcode's 275-625 Hz filter design, and seeded random cross-shank pairs rather than
  each shank's top and bottom channel. Neither difference has been tested.
- **Watson's duration criteria are fully implemented**: runs ≤2 s take the preceding state, NREM/REM
  runs shorter than 20 s fall to wake, and microarousals (<40 s wake flanked by NREM) are reported
  as a subset mask of wake.
- **Explicit wake and quiet-wake states** are emitted, which this repo previously lacked.

## IMU

The BNO055 blocks are reduced to translational speed, angular speed and `|linear acceleration|`, and
binned onto the 1 s slow-wave grid. Clock alignment uses the shared free-running ONIX counter, with
each block's IMU clock span mapped linearly onto its ephys segment.

**The IMU is QC-only and does not gate any classification threshold.** `sw_thresh`, `motion_thresh`
and `th_thresh` are computed from LFP and EMG alone.

Substituting IMU for EMG as the motion input has no buzcode precedent to lean on: across all
branches, `SleepScoreMaster`'s `MotionSource='Accelerometer'` path is unimplemented, and the one real
accelerometer-extraction function that exists (`bz_getIntanAccel.m`, `RHdev` only) was never wired
into a thresholding pipeline. The mechanistic objection is stronger than the missing precedent: EMG
reflects muscle tone and REM is defined by atonia, which a movement signal cannot see — a still
animal in quiet wake looks the same to an accelerometer as one in REM atonia. Empirically, only the
EMG proxy produces a pool in which REM is separable.

Two alignment caveats. The aligned IMU timeline runs slightly past the LFP because the final
segment's amplifier data stops before its clock; those samples are dropped when binning, which is
cosmetic. More seriously, on the truncated ProbeB data the IMU alignment drifts against the LFP by
up to 6030 s, because block durations are taken from the complete `Clock` stream while the LFP
concatenation received only the truncated amplifier data. An earlier cross-correlation check found
no cumulative drift; that check predates the truncation finding and should not be relied on. See
`docs/missing_data_note.md`.
