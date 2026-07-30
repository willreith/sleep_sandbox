#!/bin/bash
#SBATCH --job-name=sleep_plots
#SBATCH --partition=cpu
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00

# ---------------------------------------------------------------------------
# Config: all paths come from the gitignored .env at the repo root.
# Submit from the repo root (cd there first) so $SLURM_SUBMIT_DIR points to it.
# Resource request mirrors submit_preprocess.sh (same whole-recording, multi-channel-scan cost
# profile as run_scoring.py, which plot_scoring_figures.py's notebook_extras partly repeats for
# the theta-convention channel scans); adjust mem/time down if that turns out to be too much.
# ---------------------------------------------------------------------------
set -a
source "$SLURM_SUBMIT_DIR/.env"    # PREPRO_RAW_DIR PROBEA_DERIV_DIR PROBEB_DERIV_DIR PREPRO_LOG_DIR PYTHON PYTHONPATH
set +a

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
echo "Start time:  $(date)"

# ---------------------------------------------------------------------------
# Map array task -> probe (both LFP variants run inside one task, sharing the
# probe's IMU/EMG load -- see plot_scoring_figures.py's variant loop).
# ---------------------------------------------------------------------------
PROBES=(ProbeA ProbeB)
PROBE=${PROBES[$SLURM_ARRAY_TASK_ID]}
echo "Probe:       $PROBE"

"$PYTHON" "$SLURM_SUBMIT_DIR/scripts/plot_scoring_figures.py" --probe "$PROBE"

echo "Finished: $(date)"
