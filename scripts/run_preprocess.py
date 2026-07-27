"""Stage A preprocessing: concatenate a segment range for one probe and build the
lfp_nocmr / lfp_cmr / emg derivatives. Mirrors the remote-build path of
buzsaki_sleep_scoring.ipynb. One probe per invocation (SLURM array maps task -> probe).

Usage: python run_preprocess.py {ProbeA|ProbeB}
"""

import os
import sys
import time
import yaml
import spikeinterface.extractors as se
import spikeinterface.full as si

from pathlib import Path

from dotenv import load_dotenv

from sleep_sandbox.io import find_amplifier_files, get_or_build
from sleep_sandbox.preprocessing import build_band

# ---------------------------------------------------------------------------
# Run config: paths from the gitignored .env, params inline (one-off; DataJoint later)
# ---------------------------------------------------------------------------
repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")               # already-exported env vars (e.g. from the sbatch) win

probe = sys.argv[1]                            # "ProbeA" or "ProbeB"

data_dir    = Path(os.environ["PREPRO_RAW_DIR"])
session_dir = data_dir.parent                 # datetime dir holding the probe config + NeuropixelsV2/
base_dir    = Path(os.environ["PREPRO_OUTPUT_DIR"])
suffixes    = range(5, 8)                  # 48 segments (~8.1 h)

margin_ms     = 12000                          # 0.25 Hz high-pass settles over seconds
emg_margin_ms = 100                            # 300-600 Hz band settles in ms
n_jobs        = 4        # per-worker peak is large (30s chunk + 24s margin, float32, + resample buffers)
chunk         = "30s"

with open(repo_root / "config/preprocessing.yml") as f:
    config = yaml.safe_load(f)

sample_rate   = config['recording']['sample_rate']
n_channels    = config['recording']['n_channels']
dtype         = config['recording']['dtype']
freq_min      = config['lfp']['bandpass_filter']['freq_min']
freq_max      = config['lfp']['bandpass_filter']['freq_max']
reference     = config['lfp']['common_reference']['reference']
operator      = config['lfp']['common_reference']['operator']
resample_rate = config['lfp']['resample_rate']

# ---------------------------------------------------------------------------
# Discover + concatenate the selected segments
# ---------------------------------------------------------------------------
amplifier_paths, _ = find_amplifier_files(session_dir, probe, suffixes)

seg_ns = sorted(int(p.stem.split("_")[-1]) for p in amplifier_paths)
assert seg_ns == list(range(seg_ns[0], seg_ns[-1] + 1)), f"non-contiguous segments: {seg_ns}"
seg_source = f"{probe}_seg{seg_ns[0]}-{seg_ns[-1]}"

print(f"Probe:      {probe}")
print(f"Segments:   {seg_ns[0]}-{seg_ns[-1]} ({len(seg_ns)} files)")
print(f"seg_source: {seg_source}")
print(f"Output:     {base_dir / seg_source}")
print(f"n_jobs={n_jobs}  chunk={chunk}", flush=True)

segments  = [se.read_binary(p, sample_rate, dtype, n_channels) for p in amplifier_paths]
recording = si.concatenate_recordings(segments)   # one continuous segment; bridges internal boundaries

# No set_probe: global-median CMR uses no geometry, and SI drops the probe on save anyway.

# Shared input provenance: the exact ordered concatenated segments, pinned into every hash.
concat_step = ("concatenate", {"inputs": [p.stem for p in amplifier_paths]})

# ---------------------------------------------------------------------------
# Build the three variants (lazy) then save each (skips any already on disk)
# ---------------------------------------------------------------------------
rec_nocmr, nocmr_steps = build_band(
    recording, freq_min, freq_max, resample_rate, margin_ms, source_steps=[concat_step])
rec_cmr, cmr_steps = build_band(
    recording, freq_min, freq_max, resample_rate, margin_ms,
    cmr=True, reference=reference, operator=operator, source_steps=[concat_step])
rec_emg, emg_steps = build_band(
    recording, 300, 600, resample_rate, emg_margin_ms, source_steps=[concat_step])

builds = [
    ("lfp_nocmr", rec_nocmr, nocmr_steps),
    ("lfp_cmr",   rec_cmr,   cmr_steps),
    ("emg",       rec_emg,   emg_steps),
]

for i, (stream, rec, steps) in enumerate(builds, 1):
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] building {probe} {stream} ({i}/{len(builds)})...", flush=True)
    get_or_build(rec, seg_source, steps, base_dir, stream,
                 n_jobs=n_jobs, chunk_duration=chunk, progress_bar=True)
    print(f"[{time.strftime('%H:%M:%S')}] done {probe} {stream} in {(time.time() - t0) / 60:.1f} min", flush=True)

print(f"[{time.strftime('%H:%M:%S')}] all builds complete for {probe}", flush=True)
