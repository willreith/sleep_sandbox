# Missing data in abcEphysPilot04 (2026-05-16) — ProbeB truncation

Investigated 2026-08-14. Read-only analysis of the raw session; no data or derivatives modified.

## Summary

133 of 281 ProbeB acquisition blocks are incomplete. Each affected block is missing the last
60–100 s of its 10-minute recording. The cause is the automated upload tool, which copied each
ProbeB amplifier file while acquisition was still writing to it. ProbeA is unaffected. The
follow-up session (`abcEphys01`, 2026-07-06), transferred manually, is completely clean.

## Block structure (reference)

| quantity | value |
|---|---|
| block length | 18,000,000 samples = 600.000 s @ 30 kHz |
| writer frame | 300 samples (10 ms, 230,400 bytes) |
| full `AmplifierData` block | 13,824,000,000 bytes (384 ch × int16) |
| blocks in session | 0–280 (281) |
| blocks in the analysis range | 5–148 (144, nominally 24 h) |

## Extent of the loss

| | ProbeA | ProbeB |
|---|---|---|
| complete blocks, range 5–148 | **144/144** | **76/144** |
| complete blocks, whole session | 279/281 (only 0 and 280 short — genuine start/stop partials) | 148/281 |
| `Clock` complete, range 5–148 | 144/144 | 144/144 |
| `HubSyncCounter` complete, range 5–148 | 144/144 | 144/144 |
| total duration, blocks 5–148 | **24.0000 h** | **22.3250 h** |

- Deficit over blocks 5–148: **6030.0 s = 1.675 h**. Over blocks 5–52: 7.470 h vs 8 h nominal.
- Per-block deficit on affected blocks: 59.8–98.3 s, mean 88.7 s.
- Truncation point within the 600 s block: 501.7 s min, **508.8 s median**, 540.2 s max.
- Affected blocks alternate: 6, 8, then odd 11–59, 65–71, 75–147 (and continuing to 277 outside
  the analysis range).

## Evidence for the cause

1. **Each truncated file is a clean contiguous prefix, not internally corrupted.** Cross-probe
   common-mode alignment (channel-mean of ProbeA vs ProbeB, ProbeA being complete) correlates at
   lag 0 with r = +0.16 to +0.57 from 2% to 99.5% through each short file, on blocks 6, 8, 75 and
   147. Testing lag = deficit gives r ≈ −0.05. A frame-boundary discontinuity test found no excess
   at 300-sample boundaries in the short blocks (ratio 0.95–1.02 vs a ProbeA control). There is no
   splice anywhere inside them — the data present is intact and correctly ordered.
2. **Truncation lands on exact 300-sample frame boundaries**, never on 4 KB / 64 KB / 1 MB
   filesystem boundaries — consistent with a reader seeing a frame-aligned EOF on a file the
   writer is still flushing, and inconsistent with a truncated byte-stream copy.
3. **File timestamps identify the affected blocks perfectly.** All 68 truncated blocks in range
   5–148 carry a `ProbeB_AmplifierData` mtime exactly one block period (10 min) earlier than the
   matching ProbeA file; **0 of 76** complete blocks show any lag. Zero false positives or
   negatives.
4. **The small companion streams are complete.** `Clock` and `HubSyncCounter` (144 MB/block) are
   full 18,000,000 entries for every truncated block, while `AmplifierData` (13.8 GB/block) is not
   — they finish writing long before the copier reaches them.

## Acquisition and clocks are not implicated

The ONIX clock is continuous within and across all blocks (uniform per-sample increments;
inter-block step exactly one sample, 0.0333 ms). Fitting all `HarpSync` records, the ONIX counter
runs at **249,423,512 Hz** — 0.2306% below its nominal 250 MHz — and is affine-locked to the HARP
clock with a **maximum residual of 1.4 ms over 46.7 h**, i.e. no drift. Applying the measured rate,
a full block is 600.0002 HARP s and the true ephys rate is 29,999.99 Hz.

Two consequences: nothing was dropped during acquisition (the loss happened at transfer), and the hard-coded 250 MHz clock
constant in `io.py` is wrong by 0.23%.

Note: one row in the final `HarpSync` CSV has truncated digits (a partial write at session end)
and must be excluded before fitting.

## Manual-transfer control session

`abcEphys01`, 2026-07-06, 299 blocks × 2 probes: `AmplifierData` == `Clock` == `HubSyncCounter` on
every block, both probes, with zero mismatches. The only short file is block 298 on both probes
(10.7 s), which is where recording stopped. Total 49.6696 h = 298 × 600 s + 10.7 s exactly.

## Impact on the current sleep classification results

All ProbeB results in `sleep_classification_algorithm.md` and `ANALYSIS_FRAMEWORK.md` were computed
on this data and are affected.

1. **The timeline is wrong, not merely short.** `run_preprocess.py` concatenates blocks with
   `si.concatenate_recordings`, which abuts them end-to-end. For seg5-148 this silently closes 68
   real gaps of 60–98 s each, so 68 sample-adjacent pairs splice data ~90 s apart, and absolute
   time error accumulates to 6030 s by the end of the recording. The contiguity assertion in
   `run_preprocess.py` checks block-index continuity only, which does not catch this.
2. **68 low-frequency transients land directly in the slow-wave metric.** `build_band` filters
   after concatenation with `margin_ms = 12000`, so each splice discontinuity is smeared across
   ±12 s in the 0.25–300 Hz band.
3. **The IMU is misaligned against the LFP by up to 6030 s.** `align_bno055_to_lfp` derives each
   block's duration from the `Clock` file (complete, 600 s) while the LFP concat received only
   ~509 s for that block, so the offset grows monotonically. Every IMU-derived motion source
   (B1 bandpassed |accel|, B2 speed, B3 accel-variance) was evaluated against a drifting IMU,
   which undermines the retirement of B1 and the B2/B3 conclusions.
4. **Specific prior findings now have a competing explanation** and should be treated as
   unresolved: the NREM slow-wave mode drift over 22 h read as physiological; ProbeB/`lfp_cmr`'s
   flat threshold valley floor and kappa 0.854 in the stability check; the spurious early NREM
   under CMR; and the CMR-vs-noCMR fragmentation gap. Seam artefacts are denser in the later half
   of the recording, which is where the drift was measured.
5. **The threshold-stability comparison assumed both segments are continuous recordings.** That
   holds for ProbeA but not for ProbeB, so the seg5-52 vs seg5-148 comparison is confounded.
6. **ProbeA is intact but is not a fallback**, having been excluded for carrying no theta REM
   signature. No current ProbeB result rests on complete data.

## Verification status

The contiguous-prefix finding was verified directly on 4 of the 133 truncated blocks. All four
agreed and the timestamp signature is uniform across all 133, but it is stated generally here on
that basis rather than from an exhaustive sweep.
