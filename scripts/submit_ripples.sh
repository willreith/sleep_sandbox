#!/bin/bash
#SBATCH --job-name=ripples
#SBATCH --partition=cpu
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00

# ---------------------------------------------------------------------------
# Ripple detection on the per-shank channels chosen by select_ripple_channel.py
# (scripts/run_ripples.py). Requires that job AND the scoring job to have run for this seg/probe.
# Config: all paths come from the gitignored .env at the repo root.
# Submit from the repo root (cd there first) so $SLURM_SUBMIT_DIR points to it.
#
# Sizing below is measured, not projected -- the first attempt (job 3391559, 24G/3h) TIMED OUT with
# MaxRSS pinned at exactly the 24G cap, having never finished reading. Cause was an unchunked
# get_traces over the whole memmap; see the comment in run_ripples.py. Do not "simplify" that read.
#
# Memory: seg5-148 is 24.00 h at 1250 Hz = 1.08e8 samples of int16, an 83G file. The resident batch
# is (1 candidate + up to 3 neighbours) x n_shanks channels kept as int16: 3.2G for ProbeA (15
# channels -- 16 minus the shank-1 pick, which sits at a column end and has no 'above' neighbour)
# and 2.6G for ProbeB (12; its channel map leaves the x=219 shank empty). One envelope is computed
# at a time on top of that, measured at 1.21G per 2e7 samples and linear, so ~12G at 1.08e8 (float64
# trace + filtered + two complex128 FFT workspaces + envelope). ~15.5G peak, so 32G is ~2x. 24G
# would likely fit, but the failure above is expensive enough to not retry at 1.5x.
# Time: measured 2.3 min for a full chunked pass, and 20.5 s per channel for sosfiltfilt + hilbert
# (2.0 s at 1e7, 3.8 s at 2e7, linear), so ~2.3 + 15 x 0.34 = ~8 min of real work. 2 h is headroom.
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
