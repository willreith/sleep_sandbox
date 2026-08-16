# IO and processing of amplifier files

How raw ephys segments are discovered, filtered, downsampled, and concatenated into a single
memmapped LFP recording for sleep-scoring analysis.

## 1. On-disk data layout

A session lives under a datetime directory. abcGolden01 has a single probe (ProbeB) and one
continuous run of 12 segments (`_0`–`_11`):

```
.../abcGolden01/2026-05-11T07-50-11/
├── M81_ProbeB_4Shanks_1000_to_1700_um.json     # probe config
└── NeuropixelsV2/
    ├── NeuropixelsV2_ProbeB_AmplifierData_0.bin … _11.bin   # ~14 GB each: int16, 384 ch, 30 kHz
    ├── NeuropixelsV2_ProbeB_Clock_0.bin … _11.bin           # one uint64 per sample, 250 MHz counter
    ├── NeuropixelsV2_ProbeB_HubSyncCounter_0.bin … _11.bin
    ├── NeuropixelsV2_Bno055_{Clock,LinearAcceleration,Quaternion,Euler,GravityVector,HubSyncCounter}_0.bin … _11.bin
    ├── NeuropixelsV2_Commutator_{Clock,Turns}_0.bin … _11.bin
    └── NeuropixelsV2_HarpSync_*.csv                         # hourly
```

- The session dir directly contains the probe-config JSON (some sessions have two, e.g. ProbeA + ProbeB).
- The `NeuropixelsV2` subdir holds the raw per-block `.bin`s; all streams share the `_N` segment index.
- The trailing `_N` is the acquisition-segment index. Consecutive N = consecutive chunks of one
  continuous recording; non-consecutive N are separate epochs — do not concatenate across a gap (see §5).

## 2. IO module — `sleep_sandbox/io.py`

Covers raw-segment discovery, remote-mount detection, param-keyed save/load of preprocessed
derivatives, and the BNO055 IMU loaders (§2b). It names and hashes the preprocessing steps but does
not define them (that lives in `preprocessing`).

`find_amplifier_files(session_dir, probe, suffixes=None) -> (amplifier_paths, probe_config)`
- `probe` (`'ProbeA'`/`'ProbeB'`) filters both the `.bin`s and the config.
- `suffixes`: `None` = all, or an int / range / list of ints selecting by `_N`.
- `amplifier_paths` is sorted numerically by N; `probe_config` is the single matching config.
- Raises `FileNotFoundError` if nothing matches. Discovery only — no loading.

`remote_available(root) -> bool`
- `True` iff `root` is set and exists. Picks remote (build from raw) vs local (load derivatives) mode.
  Simplest possible check — a stale mount whose mountpoint still exists reads as available.

`get_or_build(recording, source, steps, base_dir, stream, verify=True, **save_kwargs)`
- Param-keyed load-or-build. The folder name encodes the chain (`{stream}__{readable_tag}__{sha1[:8]}`);
  loads it if present (verifying `params.json`), else saves it. `source` is the `seg_source`
  (e.g. `ProbeB_seg0-11`); `steps` must mirror the actual chain — `build_band` (§3) returns both together.

`load_preprocessed(parent, stream) -> recording`
- Local-mode loader. `parent` is a `seg_source` folder with one subdir per stream; matches on the
  `{stream}__` prefix and `si.load`s it. Raises on zero or multiple matches. Streams are named to be
  unique: `lfp_cmr`, `lfp_nocmr`, `emg`.

## 2a. Local-first vs remote mode

Local-first: if `LOCAL_DERIV_PARENT` is set and holds all needed streams, load them and skip raw
discovery / concat / build — even when the remote is mounted. Otherwise fall back to the remote build
path (guarded by `remote_available`). With neither local derivatives nor a reachable remote, the mode
cell raises.

Downstream analysis uses the CMR LFP (`rec = recording_lfp`); `lfp_nocmr` is also built but off the
default path (see the CMR note in `sleep_classification.md`).

## 2b. BNO055 IMU IO and clock alignment

- `load_bno055(data_dir, block, streams=None)` — read the per-stream `.bin`s for one block.
- `align_bno055_to_lfp(data_dir, ...)` — the aligner the notebook uses. `Bno055_Clock_N` and
  `ProbeB_Clock_N` are the same 250 MHz ONIX counter, so each block's IMU clock span is mapped
  linearly onto its ephys segment's nominal sample-time span, segments abutted gap-free. Returns
  `(data, t, valid, block_edges)`; `valid` masks samples outside their segment's clock span.
- `concat_bno055(...)` also exists (naive Clock-duration abutting) but is superseded by the aligner.

Final-segment overrun: the aligned IMU timeline ends ~53 s past the LFP (IMU 7030.0 s vs LFP 6976.9 s)
because the final segment's amplifier data stops before its clock, so that block's tail maps beyond
`sw_times[-1]`. Verified benign — binned IMU speed vs `−sw_pc1` cross-correlates at peak lag ≈ 0 both
early and at the tail, so there is no drift. `bin_max` drops samples outside `[sw_times[0], sw_times[-1]]`
so the overrun does not pollute the edge bins. For exact end-to-end coverage, anchor offsets to
per-segment `AmplifierData` sample counts instead of `ProbeB_Clock` lengths.

## 3. Processing pipeline

Produces one memmapped LFP recording for the whole ~2 h session, downsampled 30 kHz → 1250 Hz, so
downstream analysis runs over the full session at once. Concatenate raw, then filter — do not filter
per segment and stitch.

```python
amplifier_paths, probe_config_path = find_amplifier_files(session_dir, probe_name, suffixes)
segments  = [se.read_binary(p, sample_rate, dtype, n_channels) for p in amplifier_paths]  # lazy
recording = si.concatenate_recordings(segments)   # one continuous segment
recording_with_probe = recording.set_probe(active_probe)
rec, steps = build_band(recording_with_probe, freq_min, freq_max, resample_rate, margin_ms)
```

`build_band(...)` is the single builder for all three variants — `lfp_cmr` (`cmr=True`), `lfp_nocmr`,
and `emg` (300–600 Hz) — returning the lazy recording and its canonical `steps` list together so the
two can't drift. Each is saved via `get_or_build` as an SI binary folder (reload via `si.load`).

- Concatenate then filter reads real samples across every internal boundary, so internal boundaries
  carry no edge artifact — only the true first/last sample do.
- The chain is lazy; only `.save()` executes, streaming small chunks, so a full 14 GB segment is never
  held in RAM — only the ~1/24-size downsampled output is materialized.

## 4. Filter margin

Each chunk is filtered with a margin of neighbouring samples that is trimmed afterwards; it must be
long enough that the filter has forgotten the artificial chunk edge. The 0.25 Hz high-pass settles
over seconds, so SI's default `margin_ms=5.0` is far too short — low-frequency content near every
chunk edge would be slightly wrong (a global effect, not a boundary one).

`margin_ms = 12000` is set and encoded in the derivative tag (`mg12000`) + hash. Still to validate
empirically that 12000 ms is sufficient, not merely large. The margin is uniform, so once adequate,
boundaries need no special treatment.

## 5. Caveats / open questions

- Consecutive-only (guarded): the notebook asserts the selected N form a contiguous run before
  concatenating. This catches a missing index but not a same-index buffering gap — confirm segments
  abut sample-exactly (via clock/sync files) before trusting continuity.
- CMR vs no-CMR: both are built; the slow-wave metric uses CMR (see `sleep_classification.md`).
- Derivative naming: folders are keyed on `seg_source` (e.g.
  `ProbeB_seg0-11/lfp_nocmr__bp0.25-300_mg12000_rs1250__<hash>`); the exact ordered input set is pinned
  via a `concatenate` step in the hash + `params.json`, so the seg range alone never identifies inputs.
