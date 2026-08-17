# Ripple detection

Human-readable doc for the ripple-detection track. Separate from the sleep-classification track
(`sleep_classification.md`, `sleep_classification_algorithm.md`, `threshold_comparison.md`), which
it depends on only for NREM epoch masks.

Status: exploratory code exists; the refined pipeline described below is **planned, not built**.

## Scope

Ripple-band events only. No sharp-wave (deep-layer LFP deflection) component, no layer
identification by CSD — "ripple" throughout this doc means a ripple-band envelope event, not a
verified SWR complex. Revisit if the events don't look right.

## Method

Zhang et al., PNAS, 2021:

> The LFP from a selected channel (largest ripple power) was 140 to 250 Hz bandpass filtered by a
> fourth order Butterworth filter, and then the Hilbert transform were applied to filtered LFP to
> get ripple band amplitude. Candidate events was detected by choosing the periods that the ripple
> band amplitude is 2 SD above the mean, peak amplitudes >5 SD, and duration between 30 and 200 ms.
> After that, SPW-Rs were manually selected from candidate events by looking at the raw LFPs from
> neighboring channels.

This is the target configuration. The final manual-curation step is what we are automating.

## Existing code

| File | Role |
|---|---|
| `notebooks/ripple_detection.ipynb` | Original exploration: raw → LFP chain → Welch PSD → band power by depth → peak channel → Hilbert envelope → threshold → event list → plot one event |
| `sleep_sandbox/ripple.py` | The detection primitives (below) |
| `scripts/ripple_explorer_app.py` | Streamlit GUI: presets, sidebar params, depth/PSD/envelope plots, prev/next event browser |
| `docs/ripple_explorer.md` | The explorer's parameter/plot spec and the Zhang quote |
| `scripts/lfp_vis/band_viewer.py` | Adds a 100–180 Hz ephyviewer pane. Visualisation only, not detection |

The explorer is deliberately a **short-window interactive tool** — it is not the path for long
recordings. Long recordings go through the batch driver (below) and produce an event table.

`ripple.py`'s detection primitives, in pipeline order:

| Function | Role |
|---|---|
| `load_recording` | SI folder or raw `.bin`, probe attached, LFP chain applied lazily for raw |
| `compute_psd` / `band_power` | Welch PSD per channel → integrated ripple-band power |
| `shank_index` / `pick_channels_per_shank` | Geometry → highest-power channel per shank |
| `neighbour_channels` | Above/below/lateral neighbours on the same shank |
| `bandpass_envelope` | Zero-phase `sosfiltfilt` bandpass + Hilbert envelope, optional smoothing |
| `epoch_mask_to_samples` | Scoring's 1 s epoch mask → LFP sample grid |
| `zscore_envelope` | Z-score by mean/SD over a chosen baseline pool |
| `detect_events` | Threshold → merge → duration → peak → state restriction. Returns an event dict |
| `count_corroborating` | Per event, how many neighbours have an overlapping event |
| `peri_event_trace` | Trace ± window around an event, for plotting |

`detect_events` returns a dict of arrays (`start`, `end`, `peak`, `peak_z`, `duration_s`), matching
the dict-of-arrays convention `score_recording` already uses. Run-finding is vectorised; the
per-sample Python loop in the original would not survive ~100 M samples.

## Planned pipeline

Parameters live in `config/ripple.yml` (not hardcoded in the app or the library).

1. **Channel selection** — Welch PSD → integrate 140–250 Hz → highest-power channel per shank.
   At 22 h this cannot run on the full recording; PSD is estimated on a subsample of NREM windows.
2. **Filter** — 4th-order Butterworth bandpass, `sosfiltfilt` (zero-phase).
3. **Envelope** — `|hilbert|`.
4. **Baseline** — mean/SD of the envelope over **NREM samples only**, from the scoring track's
   `nrem` mask.
5. **Threshold** — extent at 2 SD, peak ≥5 SD.
6. **Merge** (optional, default off) — events separated by < `min_inter_event_ms`.
7. **Duration filter** — 30–200 ms.
8. **Neighbour corroboration** — require *k* adjacent channels to show a co-occurring event.
9. **Output** — event table + summary statistics.

Order of operations for 5–7 is load-bearing: merging after the duration filter is useless, because
the fragments have already been deleted.

## Integration with sleep classification

Ripple detection runs on NREM epochs only. The mask comes from the scoring track's saved
`result.npz` (`nrem`, `times`), expanded from the 1 s epoch grid onto the LFP sample grid.

Two consequences to keep in mind:

- **Epoch boundaries are soft.** `times` are STFT frame centres from a 10 s window, and the metrics
  are additionally smoothed with a 15 s moving average before thresholding. So an NREM boundary is
  accurate to several seconds, not to a sample. Fine for estimating a baseline; a caveat for any
  claim about events near a state transition.
- **The scoring run and the ripple run must be the same segment and probe**, or the masks are
  meaningless. The `--seg`/`--probe`/`--variant` triple selects both.

## Decisions taken (2026-08-17)

- **Referencing: `lfp_cmr`.** Not inherited from the scoring track's decision — slow-wave went
  noCMR because the global median removes the synchronous signal PC1 is built to detect
  (`sleep_classification_algorithm.md`, "Note on referencing"). That argument does not transfer to
  ripples, which are local. noCMR is retained as the control.
- **Strict neighbour criteria, separately parameterised.** A neighbour corroborates only if it has
  its own fully-qualified event (peak ≥ its own `peak_sd`, duration in range) overlapping the
  candidate in time. But the `neighbours:` block in `config/ripple.yml` carries its own
  `boundary_sd`/`peak_sd`/durations, so the corroboration bar moves independently of the
  candidate-detection bar. This is the escape hatch for the brittleness of strict matching — a
  neighbour event of 210 ms would otherwise fail max-duration and veto an otherwise good ripple.
- **The three neighbours are above, below and lateral on the same shank** (`neighbour_channels`):
  the next contact up and down the candidate's own column, plus the nearest contact in the other
  column. `neighbours.n_required` sets how many of the three must corroborate.

## Open decisions

- **Filter order under zero-phase filtering.** "Fourth order Butterworth" is ambiguous twice over:
  `scipy.signal.butter(4, [lo, hi], btype='bandpass')` already yields an 8-pole filter, and
  `sosfiltfilt` applies it twice again. Taking `order=4` + `sosfiltfilt` as the reading of Zhang;
  worth noting when comparing event counts against published rates.
- **Baseline drift over 22 h.** A single global mean/SD assumes a stationary envelope. The scoring
  track already documents NREM-mode drift over 22 h (`ANALYSIS_FRAMEWORK.md`, threshold-stability
  section). A sliding or per-bout baseline may be needed; not built.
- **Baseline is contaminated by the events it is meant to calibrate.** Envelope distributions are
  heavily right-skewed and the ripples themselves inflate the SD. FMAToolbox normalises the
  *squared, smoothed* signal; median/MAD or iterative exclusion of supra-threshold samples are the
  usual robust alternatives. Not addressed on the first pass.

## Event merging

Not in Zhang, and not an oversight on their part — see the reasoning below. Implemented as
`min_inter_event_ms`, **default 0 (off)** so the Zhang configuration is reproduced exactly, and
swept as a sensitivity parameter.

**The dual threshold already handles most fragmentation.** Detect-at-high / extend-to-low
(Csicsvari et al. 1999) is exactly Zhang's 2 SD extent / 5 SD peak. A ripple whose envelope dips
between cycles still reads as one event provided it stays above 2 SD.

**The reference implementation in the Buzsáki lineage nonetheless merges explicitly.** FMAToolbox
`FindRipples.m` (wrapped as buzcode's `bz_FindRipples.m` — already this repo's comparison baseline
for scoring) takes `'durations' = [minInterRippleInterval maxRippleDuration minRippleDuration]`,
default `[30 100 20]` ms, and merges events closer than 30 ms. It also squares the filtered signal
and moving-average smooths it (~11 samples ≈ 9 ms at 1250 Hz) before normalising — a second,
upstream anti-fragmentation device. The Frank-lab lineage (Kay et al. 2016; the `ripple_detection`
Python package) does the same with a Gaussian envelope smooth (~4 ms SD) plus a minimum
inter-event interval. Human iEEG work uses merge windows of ~10–25 ms.

**Zhang can omit it because the manual curation step *is* the merge rule.** A human reading raw LFP
across neighbouring channels resolves a fragmented pair as one ripple without deliberation. Once
that step is automated — the entire point of this track — the merge rule stops being optional.

**Why it matters for our outputs.** The headline results are event rate and duration distribution.
Fragmentation inflates the rate and biases durations downward, and because the min-duration
criterion runs after thresholding, both halves of a fragmented genuine ripple fall under 30 ms and
*both are deleted*. Fragmentation therefore produces false positives and false negatives at the
same time.

**The cost is real.** Ripple doublets and triplets are genuine, and long-duration/multi-ripple
events are associated with longer replay sequences (Fernández-Ruiz et al. 2019). A 30 ms merge
fuses a doublet into one event that may then exceed the 200 ms maximum and be discarded entirely.

Envelope smoothing (`envelope_smooth_ms`, also default off) is the alternative: it removes
fragmentation at source rather than stitching afterwards, at the cost of blurring true onsets.

## Neighbour corroboration

Automating Zhang's manual curation: an event on the detection channel is kept only if *k* adjacent
channels show a co-occurring event. Swept over k ∈ {1, 2, 3} to measure sensitivity.

**Adjacency is defined by depth on the same shank, not by channel index.** NeuropixelsV2 has two
columns 32 µm apart with 15 µm row pitch, so index ±1 is frequently a lateral neighbour at the same
depth rather than the next contact down.

**Expectation, recorded in advance.** Adjacent contacts are 15–32 µm apart, far inside a ripple's
spatial extent, so their LFP is near-identical by volume conduction. The k = 1→3 sweep is therefore
predicted to change event counts very little. The criterion rejects single-contact noise, which CMR
has largely handled already. The false-positive classes that Zhang's curation actually removes —
bandpass ringing on sharp transients, EMG/movement artefact, spike contamination from a nearby unit
— all appear on neighbouring channels too, so this criterion does not target them.

A flat sweep is a useful negative result and the test is cheap. But if it comes out flat, the
criteria that would target the real false positives are:

- a **distant out-of-layer noise-reference channel** — reject events coincident on it. This is what
  `bz_FindRipples`' `'noise'` parameter does, and it directly targets global artefacts.
- a **spectral-peak-in-band test** — a step/transient artefact rings broadband, so requiring
  ripple-band power to dominate a higher control band (e.g. 250–500 Hz) separates it from a real
  ripple.
- a **cycle-count / instantaneous-frequency test** — a real ripple has ≥3–4 in-band cycles.
  `ripple_detection.ipynb` already computes `inst_freq` and never uses it.

Neither is planned yet.

## Planned sensitivity analyses

| Parameter | Values | Question |
|---|---|---|
| `neighbours.k` | 0, 1, 2, 3 | How much does adjacent-channel corroboration actually reject? |
| `baseline.pool` | nrem, nrem+wake, all, wake | How much does the z-score baseline choice move detection? |
| `min_inter_event_ms` | 0, 30 | Fragmentation: how much does merging change rate and duration? |

Baseline-pool variation is deferred; NREM-only first.

## Outputs

`scripts/run_ripples.py --seg seg5-148 [--probe ProbeB] [--variant lfp_cmr] [--channel N]` writes
into `{out_base}/{seg}/{probe}/{variant}/`, alongside the scoring track's `result.npz`:

- **`ripples.npz`** — the event table: `start`/`end`/`peak` (samples), `start_s`/`end_s`/`peak_s`,
  `peak_z`, `duration_s`, `n_corroborating`.
- **`ripples.yml`** — the config used, source paths, detection channel + depth, the three neighbour
  channel indices, and the summary statistics.

**Every candidate is saved, filtered by nothing but the NREM restriction.** `n_required` is applied
only to compute the summary, and the run prints the full k = 0..3 sweep. So the headline sensitivity
test is a filter on the saved table rather than four separate runs.

Summary statistics: candidate and kept counts, NREM hours, rate per minute of NREM, duration
median/IQR, peak-amplitude median, median inter-event interval, and the corroboration sweep.

The driver requires `run_scoring.py` to have run for the same seg/probe/variant — that is where the
NREM mask comes from. It re-attaches the probe via `find_amplifier_files` (the same discovery
`run_scoring.py` uses), so the geometry matches what the derivatives were built with.

### Two implementation notes

- **Detection channel** defaults to the globally highest ripple-band-power channel, with band power
  averaged over a random sample of NREM windows (`channel_selection` in the config) because a PSD
  over the full recording is infeasible. The per-shank ranking is printed so the choice can be
  sanity-checked, and `--channel` overrides it.
- **Neighbours are detected without the NREM restriction.** The NREM test belongs to the candidate;
  a neighbour's own peak can legitimately fall the other side of a state boundary that is only
  accurate to a few seconds.

## References

- Zhang et al., PNAS, 2021 — the detection recipe above.
- Csicsvari et al., 1999 — dual-threshold detection.
- FMAToolbox `FindRipples.m` / buzcode `bz_FindRipples.m` — reference implementation; merging,
  squared-and-smoothed normalisation, noise-channel rejection.
- Kay et al., 2016 — Gaussian envelope smoothing + minimum inter-event interval.
- Fernández-Ruiz et al., 2019 — long-duration ripples and multi-ripple events.
