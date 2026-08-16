#!/bin/bash
#SBATCH --job-name=sleep_plots
#SBATCH --partition=cpu
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=4:00:00

# ---------------------------------------------------------------------------
# Config: all paths come from the gitignored .env at the repo root.
# Submit from the repo root (cd there first) so $SLURM_SUBMIT_DIR points to it.
#
# Sized from measurement, not projection: run_scoring.py on seg5-148 peaked at 167 GB
# (/usr/bin/time -v) in 39 min for one probe/variant. plot_scoring_figures.py repeats that
# spectrogram/channel-scan profile per theta convention, so 200 GB with both variants in one task.
# The shorter --time matters on this cluster: it is sched/backfill with PriorityWeightJobSize=0,
# so a small time request buys backfill windows, not priority. Overrunning it kills the job.
#
# Overrides (env, via --export):
#   SEG      preprocessed segment range           (default seg5-148)
# ---------------------------------------------------------------------------
set -a
source "$SLURM_SUBMIT_DIR/.env"    # PREPRO_RAW_DIR PREPRO_OUTPUT_DIR PREPRO_LOG_DIR PYTHON PYTHONPATH
set +a

SEG=${SEG:-seg5-148}

mkdir -p "$PREPRO_LOG_DIR"
LOG="$PREPRO_LOG_DIR/plot_scoring_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
exec >>"$LOG.out" 2>>"$LOG.err"

export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=120

# ---------------------------------------------------------------------------
# Job info
# ---------------------------------------------------------------------------
echo "Job ID:      $SLURM_JOB_ID"
echo "Array task:  $SLURM_ARRAY_TASK_ID"
echo "Node:        $SLURMD_NODENAME"
echo "Python:      $PYTHON"
echo "Segments:    $SEG"
echo "Start time:  $(date)"

# ---------------------------------------------------------------------------
# Map array task -> probe (both LFP variants run inside one task, sharing the
# probe's IMU/EMG load -- see plot_scoring_figures.py's variant loop).
# ---------------------------------------------------------------------------
PROBES=(ProbeA ProbeB)
PROBE=${PROBES[$SLURM_ARRAY_TASK_ID]}
echo "Probe:       $PROBE"

/usr/bin/time -v "$PYTHON" "$SLURM_SUBMIT_DIR/scripts/plot_scoring_figures.py" \
    --seg "$SEG" --probe "$PROBE"

echo "Finished: $(date)"
