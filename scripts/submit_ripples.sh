#!/bin/bash
#SBATCH --job-name=ripples
#SBATCH --partition=cpu
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3:00:00

# ---------------------------------------------------------------------------
# Ripple detection on the per-shank channels chosen by select_ripple_channel.py
# (scripts/run_ripples.py). Requires that job AND the scoring job to have run for this seg/probe.
# Config: all paths come from the gitignored .env at the repo root.
# Submit from the repo root (cd there first) so $SLURM_SUBMIT_DIR points to it.
#
# Memory: all 16 channels (4 shanks x 1 candidate + 3 neighbours) are read in one pass and held,
# because a single-channel read of a sample-interleaved binary touches every page of the recording
# anyway -- 16 separate reads would be 16 full passes over ~123G. seg5-148 is 22.33 h at 1250 Hz =
# 8.0e7 samples, so the resident batch is 16 x 8.0e7 float32 = 5.1G. On top of that one envelope is
# computed at a time, peaking inside hilbert at ~3.2G (float64 trace + filtered + complex128 FFT
# workspace + envelope, 0.64G each). ~8.3G total, so 24G is ~3x headroom.
# Time: one pass over the derivative, then 16 x (sosfiltfilt + hilbert) on 8.0e7 samples, so now
# CPU-bound rather than I/O-bound. 3 h is headroom, not an estimate; /usr/bin/time -v measures the
# first run.
#
# Overrides (env, via --export):
#   SEG    preprocessed segment range   (default seg5-148)
# ---------------------------------------------------------------------------
set -a
source "$SLURM_SUBMIT_DIR/.env"    # PREPRO_RAW_DIR PREPRO_OUTPUT_DIR PREPRO_LOG_DIR PYTHON PYTHONPATH
set +a

SEG=${SEG:-seg5-148}

mkdir -p "$PREPRO_LOG_DIR"
LOG="$PREPRO_LOG_DIR/ripples_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
exec >>"$LOG.out" 2>>"$LOG.err"

export PYTHONUNBUFFERED=1

echo "Job ID:      $SLURM_JOB_ID"
echo "Array task:  $SLURM_ARRAY_TASK_ID"
echo "Node:        $SLURMD_NODENAME"
echo "Python:      $PYTHON"
echo "Segments:    $SEG"
echo "Start time:  $(date)"

# One array task per probe. ProbeA is PFC and is run as the negative control -- its channel
# selection is an arbitrary argmax on a flat profile, so whatever it detects is the null. Outputs
# land in data/derivatives/$SEG/{probe}/ripples/events/{run_id}/.
PROBES=(ProbeA ProbeB)
PROBE=${PROBES[$SLURM_ARRAY_TASK_ID]}
echo "Probe:       $PROBE"

/usr/bin/time -v "$PYTHON" "$SLURM_SUBMIT_DIR/scripts/run_ripples.py" \
    --seg "$SEG" --probe "$PROBE"

echo "Finished: $(date)"
