#!/bin/bash
#SBATCH --job-name=sleep_prepro
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
# ---------------------------------------------------------------------------
set -a
source "$SLURM_SUBMIT_DIR/.env"    # PREPRO_RAW_DIR PREPRO_OUTPUT_DIR PREPRO_LOG_DIR PYTHON PYTHONPATH
set +a

# #SBATCH log directives can't read .env, so redirect output to the configured dir here.
mkdir -p "$PREPRO_LOG_DIR"
LOG="$PREPRO_LOG_DIR/preprocess_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
exec >>"$LOG.out" 2>>"$LOG.err"

export PYTHONUNBUFFERED=1      # stream stdout to the log during the run, not at exit
export TQDM_MININTERVAL=120    # throttle the SI progress bar to one line / 2 min

# ---------------------------------------------------------------------------
# Job info
# ---------------------------------------------------------------------------
echo "Job ID:      $SLURM_JOB_ID"
echo "Array task:  $SLURM_ARRAY_TASK_ID"
echo "Node:        $SLURMD_NODENAME"
echo "Python:      $PYTHON"
echo "Start time:  $(date)"

# ---------------------------------------------------------------------------
# Map array task -> probe, then run
# ---------------------------------------------------------------------------
PROBES=(ProbeA ProbeB)
PROBE=${PROBES[$SLURM_ARRAY_TASK_ID]}
echo "Probe:       $PROBE"

"$PYTHON" "$SLURM_SUBMIT_DIR/scripts/run_preprocess.py" "$PROBE"

echo "Finished: $(date)"
