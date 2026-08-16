# Sleep classification algorithm

Reference implementation: Watson et al. 2016 / buzcode `SleepScoreMaster`. This repo's
implementation is in `sleep_sandbox/analysis.py` (metric primitives and thresholding) and
`sleep_sandbox/scoring.py` (`score_recording`, `classify`). Parameters are in
`config/sleep_scoring.yml`; nothing below is hard-coded.

Results quoted here are from the 22.33 h `seg5-148` recording (80,361 epochs on ProbeB, 86,391 on
ProbeA) unless stated otherwise.

Every ProbeB result below was computed on data now known to be incomplete. 68 of the 144 blocks in
the analysis range are missing their last 60-98 s, the concatenation step abuts them end to end, and
the resulting timeline both contains 68 splice discontinuities and accumulates 6030 s of absolute
time error. ProbeA is unaffected. See `docs/missing_data_note.md` for the full analysis, including
which specific findings acquire a competing explanation; the most consequential are the slow-wave
mode drift, the ProbeB/`lfp_cmr` valley geometry, and every IMU-based motion source, since the IMU
alignment drifts against the LFP by up to 6030 s. Numbers below are reported as measured, without
attempting to correct for this.

## Data sources

Two Neuropixels probes recorded simultaneously in one session, each processed under two referencing
variants (`lfp_cmr`, `lfp_nocmr`), giving four combinations. ProbeA targets PFC; ProbeB targets
hippocampus.

The slow-wave metric is taken from the non-CMR variant. Slow waves are spatially synchronous across
the probe, so subtracting the common median removes much of the signal PC1 is built to detect; CMR
is designed to suppress exactly the kind of correlated activity that constitutes the metric. The
CMR variant is retained as a control rather than as a primary input — see the note on referencing.

| Input | Derivation | Role |
|---|---|---|
| LFP | 1250 Hz, per referencing variant | slow-wave and theta metrics |
| EMG proxy | cross-shank correlation of the 300-600 Hz LFP band | motion / muscle tone |
| IMU (BNO055) | translational speed, angular speed, \|linear acceleration\| | motion candidates and QC |

The EMG proxy is intracranial, not a separate electrode: it exploits the fact that muscle activity
appears as a common high-frequency signal across distant recording sites, so correlation between
well-separated channels indexes tone (Schomburg et al. 2014). Only the EMG proxy drives
classification; the IMU is retained for QC and for the motion-source comparison below.

Clock alignment between the IMU and ephys streams uses the shared free-running 250 MHz ONIX counter,
mapping each block's IMU clock span linearly onto its ephys segment. Cross-correlation against a
wake proxy peaks at lag 0 both early and late in the recording, so there is no cumulative drift.

## Processing

All metrics land on a shared 1 s grid derived from one spectrogram, so they are directly comparable
epoch by epoch.

1. Shared STFT front-end (`log_spectrogram`): 10 s Hann window, 1 s step, power interpolated onto a
   log-spaced 1-100 Hz grid of 100 bins.
2. Slow-wave metric: `log10` the power, z-score each frequency across time, run PCA over time bins,
   take PC1. If the mean PC1 loading below 20 Hz is negative, flip the component's sign so that high
   values correspond to NREM.
3. Theta metric: band power integrated over numerator and denominator bands on the shared
   spectrogram, taken as a linear ratio (not log-scaled — buzcode thresholds the raw ratio).
   Three conventions are computed and reported: `watson` (5-10 / 2-16 Hz), `shin` (6-12 / 1-4 Hz),
   and `shin_mod` (5-10 / 1-4 Hz, a local variant).
4. EMG proxy: mean Pearson *r* across 50 random channel pairs at least 2 shanks apart, computed on
   non-overlapping 1 s windows of the raw 300-600 Hz trace, then interpolated onto the 1 s grid.
5. Smoothing and normalisation (`smooth_norm`): 15 s centred moving average, then global min-max to
   [0,1], per buzcode `ClusterStates_GetMetrics`.

Channel selection follows buzcode in spirit: rather than picking the highest-power channel, which
can simply select the noisiest site, each candidate channel is scored by how well its metric
separates into two modes, using Hartigan's dip statistic, and the maximum is taken. This is done
independently for the slow-wave channel (on PC1) and the theta channel (on the `log10` theta ratio,
since ratios are right-skewed). Every Nth channel is scanned, with `stride` set in config.

Theta channel selection scores bimodality on the full theta distribution, not on the
`~nrem & ~mov` pool that the threshold is later taken on: the aim is to pick a channel with
prominent theta.

Two consequences of step 5 are worth keeping in view. The [0,1] scale is anchored to each
recording's own extremes, so threshold values are not comparable across separately-scored
recordings; and because the anchors are rare extremes, a shorter prefix of a recording can span a
noticeably narrower range than the whole (for both noCMR combinations the first 7.5 h reaches only
~0.60, while under 0.5% of epochs exceed that).

## Thresholding

Three thresholds are placed, all by the same routine (`find_thresh`, configured to `kde`): a
slow-wave threshold separating NREM from not-NREM, a motion threshold separating moving from still,
and a theta threshold separating REM from quiet wake. Each is the trough between the two modes of
the relevant distribution, located on a Gaussian KDE with Scott's-rule bandwidth. A candidate mode
must clear a prominence floor of 3% of peak density; if fewer than two modes qualify, the routine
returns NaN with a `RuntimeWarning` rather than inventing a threshold, and the dependent state mask
is empty. See the note on thresholding algorithms for why this method and this floor were chosen.

The theta threshold is not taken on the full theta distribution. It is taken on the pool of epochs
that are neither NREM nor moving, i.e. the quiet-wake plus REM population, which is 2-5% of the
recording:

```python
keep = ~(sw_metric > sw_thresh) & ~mov
```

This deviates from buzcode, which takes the trough on `~mov` and only excludes NREM as a fallback.
The reason is that the motion threshold in this dataset is very nearly a sleep/wake detector —
P(above motion threshold | NREM) = 0.03-0.07, P(above | not NREM) = 0.94-0.96 — so `~mov` is 85-92%
NREM and its theta distribution is essentially the NREM one. The trough it yields marks the
NREM/non-NREM boundary, which the `~nrem` term in the decision tree already handles, rather than the
REM/quiet-wake boundary that is actually wanted.

The fallback ordering is not merely redundant here but actively harmful, because on ProbeB the
`~mov` pool clears the prominence floor just well enough that the fallback never fires:

| ProbeB/lfp_cmr, watson | threshold | mode prominence |
|---|---|---|
| `~mov` (buzcode order) | 0.568 | 0.024 |
| `~nrem & ~mov` (used here) | 0.423 | 0.223 |

Excluding NREM leaves a split carrying roughly ten times the prominence. The pool is therefore used
unconditionally. The cost is that these thresholds rest on thin data and inherit any error in
`sw_thresh` or `motion_thresh` through pool membership. However, this should not be a problem when recordings are 24 hrs in duration, as the large amount of data will give us high confidence in the thresholds.

### Threshold stability

`scripts/check_threshold_stability.py` holds the metrics fixed and varies only the estimation window
(the full 22 h against its first 7.5 h), so the comparison is not confounded by renormalisation.
Motion thresholds move ~5% and theta ~3%; slow-wave thresholds move 3-24%.

| combo | kappa | sw_thresh full → prefix | wake mode shift | NREM mode shift | valley floor ÷ wake peak |
|---|---|---|---|---|---|
| A/cmr | 0.972 | 0.417 → 0.483 | −0.016 | −0.072 | 0.034 |
| A/nocmr | 0.980 | 0.297 → 0.317 | −0.010 | −0.030 | 0.079 |
| B/cmr | 0.854 | 0.487 → 0.604 | −0.001 | −0.090 | 0.153 |
| B/nocmr | 0.986 | 0.325 → 0.336 | −0.005 | −0.030 | 0.099 |

Two findings. First, the wake mode is essentially fixed while the NREM mode drifts downward across
the recording, three times more under CMR: slow-wave amplitude during NREM declines over 22 h. A
single global threshold does not avoid this, it averages over it, biasing early and late epochs in
opposite directions. Second, valley depth rather than threshold movement explains the spread in
agreement. A/cmr's trough moved 0.066 yet reached kappa 0.972, because its valley floor is 3% of the
wake peak and few epochs are ambiguous; B/cmr's moved 0.117 and reached only 0.854, because its
floor is 15% of the peak and flat over 0.10 of the range. ProbeB/lfp_cmr's slow-wave metric is
poorly conditioned for thresholding independent of recording length.

Both findings are confounded on ProbeB by the missing-data problem. Splice transients are smeared
across ±12 s each by the filtering margin, they fall predominantly in the later half of the
recording, and that is where the drift was measured; the comparison also assumes both windows are
continuous recordings, which holds for ProbeA but not ProbeB. The ProbeA rows are unaffected and
still show the same qualitative pattern, more weakly. Neither finding should be treated as settled
until the analysis is repeated on gap-aware data.

## Decision tree

All states derive from three binary comparisons. `nrem`, `rem` and `wake` partition every epoch
exactly once (asserted in the comparison driver); `qwake` and `microarousal` are reported as subset
masks of `wake`.

```python
nrem     = sw_metric > sw_thresh
rem_cand = ~nrem & (motion_metric < motion_thresh)     # not NREM, and low tone
rem      = rem_cand & (theta_metric > th_thresh)       # and high theta
wake     = ~nrem & ~rem
qwake    = wake & (theta_metric <= th_thresh)          # buzcode's optional quiet-wake state
```

This matches buzcode `ClusterStates_DetermineStates.m`
(`REMtimes = ~NREMtimes & ~highMotion & hightheta`), and adds the explicit `wake`/`qwake` states.

Watson's duration criteria are then applied in order (`config duration_criteria`):

1. Runs of 2 s or shorter take the preceding state, removing single-epoch flicker.
2. NREM and REM runs shorter than 20 s do not qualify as packets and fall to wake.
3. Wake runs shorter than 40 s that are flanked by NREM on both sides are labelled microarousals.

The 20 s minimum is applied after the merge and before microarousal labelling, so microarousals are
scored against qualifying packets rather than fragments. Its effect is to remove fragments rather
than sleep — bout counts roughly halve while total time in state barely moves:

| config | REM n | REM % | REM median | NREM n | NREM % | NREM median | MA % |
|---|---|---|---|---|---|---|---|
| cmr/watson, no minimum | 70 | 3.68% | 18 s | 261 | 31.72% | 42 s | 2.21% |
| cmr/watson, 20 s | 34 | 3.37% | 77 s | 164 | 30.53% | 100 s | 1.76% |
| cmr/shin, 20 s | 38 | 2.99% | 60 s | 164 | 30.52% | 99 s | 1.77% |
| nocmr/shin, 20 s | 38 | 2.65% | 55 s | 136 | 32.01% | 116 s | 1.34% |

Microarousal fraction falls rather than rising, because disqualifying a short flanking NREM packet
also disqualifies the microarousal it would have bracketed.

## Diagnostics

There is no ground truth for this recording, so every validation is agreement between variants that
differ in exactly one respect. The comparison scripts write to
`data/derivatives/{seg}/{probe}/{variant}/`, each taking a required `--seg` that selects both inputs
and outputs so the two cannot be pointed at different segment ranges.

| Diagnostic | What it distinguishes |
|---|---|
| Cohen's kappa, reported alongside raw agreement | raw agreement is inflated by the dominant states (wake and NREM are ~97% of epochs); kappa corrects for chance under that imbalance |
| Per-state Dice overlap | symmetric, so no variant is privileged as reference; REM Dice is the number that matters, since whole-sequence kappa hides REM almost entirely |
| Pearson *r* between continuous metrics | separates "genuinely different signal" from "same signal, different threshold" — high *r* with low kappa localises the problem to thresholding |
| Hartigan dip test per metric | the trough search assumes two modes; on a unimodal distribution any threshold it returns is arbitrary |
| Bout-duration distributions | a movement-only signal asked to detect a tone-defined state should fragment REM into many short bouts rather than produce continuous episodes — a falsifiable prediction |
| Threshold against distribution quantiles | catches a threshold placed in the empty tail, which the dip test alone does not |

The last two rows are there because bimodality testing proved necessary but not sufficient: the dip
test flagged every motion metric as significantly non-unimodal, including ones whose thresholds were
plainly unusable. Non-unimodal is not the same as usefully bimodal.

REM agreement across variants, which is the main confidence signal available:

| pair | Dice |
|---|---|
| B/cmr watson ↔ B/cmr shin | 0.917 |
| B/nocmr shin ↔ B/nocmr shin_mod | 0.968 |
| B/cmr watson ↔ B/nocmr watson | 0.794 |

The weakest pairing is CMR against noCMR within a fixed theta convention. The referencing choice
perturbs REM more than the convention choice does, which inverts what the two-convention agreement
proxy was intended to test — see the open questions.

Supporting scripts:

| script | output |
|---|---|
| `scripts/run_scoring.py` | metrics, channel selections, states → `result.npz` + `result.yml` |
| `scripts/reclassify.py` | recomputes thresholds and states in place from saved metrics after a config change; backs up to `.bak` |
| `scripts/plot_scoring_figures.py` | per-combo diagnostic figures into `metrics/` |
| `scripts/check_threshold_stability.py` | full-window vs prefix threshold comparison and mode/valley geometry |
| `scripts/plot_theta_conditioning.py`, `plot_theta_direct.py` | theta pool distributions and their KDEs, showing why each threshold succeeds or refuses |
| `scripts/plot_hypnograms_direct.py`, `plot_hypnograms_probeb.py` | hypnograms per combination and convention with bout statistics |
| `scripts/check_rem_agreement.py` | NREM spans, pairwise REM Dice, ProbeA sleep-block containment check |
| `scripts/compare_threshold_methods.py`, `compare_theta_conventions.py`, `compare_motion_sources.py` | the three head-to-head comparisons |

---

## Note on thresholding algorithms

Three methods were compared head to head across 48 series × combination cases
(`docs/threshold_comparison.md`): buzcode's histogram trough (`bz_BimodalThresh`), a KDE trough, and
a two-component Gaussian mixture. KDE is used.

The histogram method returns a degenerate threshold in 15 of the 48 cases. It zero-pads the
histogram so that edge bins can register as peaks, which on a heavy-tailed right-skewed distribution
means the apparent second mode is a handful of extreme epochs in the far tail and the trough between
them falls in empty space. The failure is not subtle: on the IMU speed and accel-variance metrics
the threshold landed above the 99.9th percentile of the metric's own distribution, so under 0.2% of
epochs counted as moving and nearly every non-NREM epoch became a REM candidate, assigning 62-66% of
the session to REM.

KDE agrees with the histogram wherever the histogram works, repairs several of its failures, and
where no usable split exists returns NaN with a warning rather than a number. That refusal is the
main reason for preferring it: a downstream state mask that is empty is visibly wrong, whereas an
invented threshold produces a plausible-looking hypnogram that is entirely artefact.

The GMM is defensible for the slow-wave metric but not generally. BIC prefers three components over
two in 40 of the 48 cases, so a fixed k=2 fit chases skew rather than finding modes, and the mixture
cannot act as a bimodality gate — it always returns a boundary.

A candidate mode must reach 3% of peak density to count, which stops tail ripple registering as a
mode. The floor was swept at 1%, 2% and 3% over all three theta conventions and all three candidate
pools on the 22 h recording. It separates the pools rather than the conventions, and the separation
is wide — second-mode prominence on ProbeB:

| pool | watson | shin | shin_mod |
|---|---|---|---|
| `~nrem & ~mov` (cmr / nocmr) | 0.223 / 0.127 | 0.114 / 0.284 | 0.283 / 0.288 |
| `~mov` (cmr / nocmr) | 0.024 / 0.018 | 0.007 / 0.002 | 0.002 / 0.001 |

Every `~mov` pool falls below 3%; every `~nrem & ~mov` pool clears it by at least a factor of
three. No case lands between 0.024 and 0.114, so the floor's exact position within that gap is not
critical, and 3% sits comfortably inside it.

The reason to prefer 3% over 1% is that it rejects the pool that should never have been thresholded.
At 1% and 2%, `watson` on the `~mov` pool returns 0.5675 — the threshold that placed REM on the
NREM/non-NREM split under buzcode's fallback ordering. At 3% it returns NaN. A 3% floor would have
surfaced that as a refusal rather than as a plausible number.

Raising the floor changes no current result: all three conventions give identical thresholds at 1%,
2% and 3% on the `~nrem & ~mov` pool, which is the only pool production uses. It is insurance
against thresholding a pool that carries no real split, not a change to any live threshold. The one
place it does bite is the unconditioned full distribution, where `shin` goes from 0.1194 to NaN
between 1% and 2%; that path is not in use, but it is why the unconditioned column is unstable.

The floor also rejects the accel-variance motion metric, whose second mode sits at 0.0070.

The Scott's-rule bandwidth remains unvalidated and is not currently being changed.

## Note on referencing

The slow-wave metric is taken from the non-CMR variant. PC1 is built to detect spatially synchronous
low-frequency activity, and CMR subtracts the across-channel median, which is the component carrying
that signal.

Three measurements agree, all favouring noCMR on ProbeB: a deeper threshold valley (floor at 10% of
the wake peak against 15%), a third of the NREM-mode drift (−0.030 against −0.090), and better
window-to-window threshold agreement (3% against 24%). ProbeB/`lfp_cmr` also places its first NREM
2.9 h earlier than the other three combinations, in the middle of a clear wake period. All were
measured on the truncated ProbeB data, so they support the mechanistic argument rather than
independently establishing it. Note though that a splice discontinuity is largely common-mode, so
CMR should suppress it — which cuts against splices explaining a CMR-specific artefact.

CMR is retained as a control. Referencing perturbs REM more than the theta convention does
(Dice 0.794 across referencing within a convention, against 0.917 across conventions within CMR).

## Note on choice of motion source

Four motion signals were compared under otherwise identical logic
(`scripts/compare_motion_sources.py`), so the motion input is the only thing varying:

| | Signal | Derivation | Provenance |
|---|---|---|---|
| A | EMG proxy | mean cross-shank Pearson *r* of the 300-600 Hz LFP band | buzcode `bz_EMGFromLFP`, the default and only wired-up motion source |
| B1 | bandpassed \|accel\| | \|LinearAcceleration\| → Butterworth bandpass 0.1-1 Hz → `abs` → per-bin mean | buzcode `bz_getIntanAccel` recipe (`RHdev` branch only; never wired into `SleepScoreMaster`) |
| B2 | translational speed | `imu_kinematics` speed → per-bin max | no buzcode precedent |
| B3 | \|accel\| variance | per-bin variance of \|LinearAcceleration\| | no buzcode precedent on any branch |

B1 is the only variant that must be filtered at the native IMU rate before binning, since a 1 Hz
corner is not representable on the 1 s grid.

The EMG proxy is used. At 22 h it is the only source that yields any REM at all:

| combo | version | motion_thresh | MOV frac | th_thresh | REM % | REM n | REM median |
|---|---|---|---|---|---|---|---|
| B/cmr | A_emg | 0.1624 | 0.632 | 0.4232 | 3.48% | 32 | 82 s |
| B/cmr | B1_accel_bp | 0.0685 | 0.507 | NaN | 0% | 0 | — |
| B/cmr | B2_speed | 0.0411 | 0.505 | NaN | 0% | 0 | — |
| B/cmr | B3_accel_var | NaN | 0.000 | NaN | 0% | 0 | — |
| B/nocmr | A_emg | 0.1624 | 0.632 | 0.4938 | 2.87% | 32 | 69.5 s |
| B/nocmr | B1/B2/B3 | — | — | NaN | 0% | 0 | — |

The failure mode is informative. B1 and B2 produce entirely reasonable-looking motion thresholds,
with roughly half of epochs classified as moving; what fails is the next step, because the
`~nrem & ~mov` pool those thresholds carve out is not bimodal in theta, so no theta threshold exists
and REM is empty. B3's motion metric is itself unimodal (dip statistic 0.0073, the lowest of the
four) and never yields a motion threshold at all.

This is the tone-versus-movement argument confirmed directly rather than assumed. An IMU measures
movement, not muscle tone; REM is defined by atonia, and an animal in REM is indistinguishable from
one in still quiet wake to an accelerometer. The two states therefore land in the same pool with no
second mode to separate them. The metric correlations support the same reading: on ProbeB/cmr,
EMG against B1/B2/B3 gives *r* = 0.332, 0.315, 0.293, while B1↔B3 gives 0.838 and B2↔B3 gives 0.805.
The IMU-derived signals agree closely with each other and not with EMG, consistent with their
measuring a different physical quantity rather than a noisier version of the same one.

Under the earlier histogram thresholding, B2 and B3 failed differently — an invented threshold in
the far tail rather than an honest refusal — which is worth recording because the visible symptom
(65% of the session labelled REM) was the opposite of the current one (no REM at all) while the
underlying cause was the same. This is also the failure the Buzsaki lab anticipated when declining
to wire the accelerometer into `SleepScoreMaster` (GitHub issue #299): an alternative motion signal
"will have different statistical properties (for example, may not be so cleanly bimodal but may have
a heavy tail)".

The IMU results specifically were computed against an alignment that drifts by up to 6030 s over the
recording (`docs/missing_data_note.md`), so B1-B3 were evaluated on a progressively mismatched motion
signal. This weakens the empirical case against them considerably; the mechanistic argument above
does not depend on it.

One caveat on the table: this comparison thresholds the shared-peakTH `theta_metric` from
`result.npz`, whereas the hypnogram scripts use the own-channel ratio from `result_extras.npz`, so
A_emg's REM count here (32 bouts) differs slightly from the 34 reported for the same configuration
elsewhere. Same pipeline, different theta series.

## Note on differences in EMG derivation from buzcode

`bz_EMGFromLFP.m` computes the same cross-shank high-frequency-correlation proxy, but the specifics
differ:

| | buzcode | this repo (`analysis.py:emg_from_lfp`) |
|---|---|---|
| Band | 275-625 Hz (300-600 Hz passband with 25 Hz transition shoulders, `fdesign.bandpass`) | 300-600 Hz plain bandpass |
| Channel pairing | top and bottom channel of each shank, exhaustive pairwise | 50 random pairs, shanks ≥2 apart, seeded, spread round-robin across qualifying shank combinations |
| Window | 0.5 s | 0.5 s function default, 1.0 s in config |
| Output rate | resampled to 2 Hz | interpolated onto the 1 s STFT grid |

The random-pair scheme covers more of the probe than buzcode's fixed top/bottom selection, which on
a Neuropixels shank samples only the two extreme sites; pairs are drawn round-robin across shank
combinations with each shank's pool refilled only once exhausted, so electrode coverage is as broad
as the pair budget allows. The seed is fixed so the derivation is reproducible.

The band difference is a filter-design detail rather than a deliberate choice: buzcode's 275/625 Hz
corners are the transition-band edges around the same 300-600 Hz passband. Neither difference has
been shown to matter for the resulting metric, and neither has been tested.

## Note on ProbeA REM

ProbeA returns a NaN theta threshold and zero REM under all three theta conventions and both
referencing variants. This is a real negative rather than a thresholding artefact:

- Neither log10-scaling nor the KDE method recovers a usable threshold, and the dip test gives
  p = 0.991-1.000 for ProbeA theta across all six convention × variant combinations. The KDE finds
  exactly one mode in all 18 combinations of convention × pool × prominence floor tested, so this is
  not a floor-placement artefact: no candidate second mode exists to be rejected.
- The two probes recorded simultaneously. Their sleep blocks coincide, and 100% of ProbeB's REM
  epochs, in every case that finds REM, fall inside ProbeA's own sleep block
  (`scripts/check_rem_agreement.py`). This holds at both recording lengths tested.

So the animal was in REM, ProbeA was recording, and ProbeA reports none of it. The PFC theta ratio
carries no REM signature here. This is the expected result given that theta is a hippocampal rhythm,
but it is now demonstrated rather than assumed. ProbeA is excluded from REM analysis and retained as
a control for the slow-wave and motion components, which do not depend on theta.

One consequence worth flagging: ProbeA's near-perfect motion-source kappas (0.997-1.000) are an
artefact rather than agreement. NREM is identical across motion variants by construction, since it
depends only on the slow-wave metric, so with REM empty the state sequences are literally identical
and kappa measures nothing. ProbeA results should not be read as validating any motion source.

## Open questions and decisions

Decided:

- The slow-wave metric uses the non-CMR variant; CMR is retained as a control.
- Theta channel selection scores bimodality on the full distribution, to select for prominent theta.
- The KDE prominence floor is 3% of peak density.
- The IMU is QC-only and does not gate any classification threshold.

Open:

1. Everything measured on ProbeB rests on truncated data. The results here should be reproduced once
   the concatenation is gap-aware, and the affected conclusions are enumerated in
   `docs/missing_data_note.md`. This is the largest outstanding item and it subsumes several below.

2. The theta conventions are not as independent as the two-convention agreement check assumes. All
   three survive the 3% floor comfortably at 22 h, so the check is available, but on the 7.5 h
   window `shin` was only marginally bimodal and `shin_mod` unimodal — so convention agreement is a
   weaker confidence signal on shorter recordings than on longer ones, in a way that is not visible
   from the agreement number itself. Referencing (CMR against noCMR) currently perturbs REM more
   than convention does, and may be the more informative comparison.

3. The Scott's-rule KDE bandwidth is unvalidated.

4. Cross-segment comparison currently compares two different theta rules, since the shorter
   `seg5-52` results predate the unconditional `~nrem & ~mov` pool and have not been reclassified.
   Note that `reclassify.py` overwrites existing `.bak` files.

5. The 250 MHz ONIX clock constant in `io.py` is wrong by 0.23%; the measured rate is
   249,423,512 Hz (`docs/missing_data_note.md`).
