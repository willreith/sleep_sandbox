#!/bin/bash
#SBATCH --job-name=ripple_channel
#SBATCH --partition=cpu
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00

# ---------------------------------------------------------------------------
# Ripple detection-channel selection + stability sweep (scripts/select_ripple_channel.py).
# Config: all paths come from the gitignored .env at the repo root.
# Submit from the repo root (cd there first) so $SLURM_SUBMIT_DIR points to it.
#
# Memory: unlike the scoring jobs this never holds the recording. It reads one 10 s x 384 ch window
# at a time (~19 MB as float32) and accumulates a (n_freqs x 384) PSD (~8 MB). 16G is almost all
# headroom for spikeinterface + the recording handle.
# Time: 5 seeds x 200 windows = 1000 reads of ~19 MB (~19 GB total off ceph) plus one Welch per
# window. 2 h is headroom, not an estimate -- the first run measures it via /usr/bin/time -v.
#
# Overrides (env, via --export):
#   SEG    preprocessed segment range   (default seg5-148)
# ---------------------------------------------------------------------------
set -a
source "$SLURM_SUBMIT_DIR/.env"    # PREPRO_RAW_DIR PREPRO_OUTPUT_DIR PREPRO_LOG_DIR PYTHON PYTHONPATH
set +a

SEG=${SEG:-seg5-148}

mkdir -p "$PREPRO_LOG_DIR"
LOG="$PREPRO_LOG_DIR/ripple_channel_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
exec >>"$LOG.out" 2>>"$LOG.err"

export PYTHONUNBUFFERED=1

echo "Job ID:      $SLURM_JOB_ID"
echo "Array task:  $SLURM_ARRAY_TASK_ID"
echo "Node:        $SLURMD_NODENAME"
echo "Python:      $PYTHON"
echo "Segments:    $SEG"
echo "Start time:  $(date)"

# One array task per probe. ProbeA is PFC and is expected to have no ripple channel -- it is run as
# the negative control, to show what an unconverged selection looks like. Outputs land in
# data/derivatives/$SEG/{probe}/ripples/channel_selection/.
PROBES=(ProbeA ProbeB)
PROBE=${PROBES[$SLURM_ARRAY_TASK_ID]}
echo "Probe:       $PROBE"

/usr/bin/time -v "$PYTHON" "$SLURM_SUBMIT_DIR/scripts/select_ripple_channel.py" \
    --seg "$SEG" --probe "$PROBE"

echo "Finished: $(date)"
