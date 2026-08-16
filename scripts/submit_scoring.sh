#!/bin/bash
#SBATCH --job-name=sleep_scoring
#SBATCH --partition=cpu
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=12:00:00

# ---------------------------------------------------------------------------
# Config: all paths come from the gitignored .env at the repo root.
# Submit from the repo root (cd there first) so $SLURM_SUBMIT_DIR points to it.
#
# Memory: seg5-148 is 22.33 h vs seg5-52's 7.47 h (2.99x), and every unchunked path is linear in
# recording length -- the full candidate-channel load in select_channel_by_dip/select_theta_channel
# and, dominating it, the (n_segments x nperseg) array scipy materialises inside log_spectrogram.
# The 96G sleep_plots job on seg5-52 peaked at 84.9G, so 3x projects to ~255G. 256G also restricts
# the job to the 512G nodes (enc3-node{1,2,4,5}); the 240G nodes cannot satisfy it.
# Time: that job took 1:30 on seg5-52, so ~4.5 h projected; 12 h is headroom, not an estimate.
#
# Overrides (env, via --export):
#   SEG      preprocessed segment range           (default seg5-148)
#   VARIANT  lfp_cmr | lfp_nocmr                  (default: both, in one task)
# ---------------------------------------------------------------------------
set -a
source "$SLURM_SUBMIT_DIR/.env"    # PREPRO_RAW_DIR PREPRO_OUTPUT_DIR PREPRO_LOG_DIR PYTHON PYTHONPATH
set +a

SEG=${SEG:-seg5-148}

mkdir -p "$PREPRO_LOG_DIR"
LOG="$PREPRO_LOG_DIR/sleep_scoring_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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
# Map array task -> probe (both LFP variants run inside one task by default, sharing the probe's
# IMU/EMG load -- see run_scoring.py's variant loop). Outputs land in
# data/derivatives/$SEG/{probe}/{variant}/.
# ---------------------------------------------------------------------------
PROBES=(ProbeA ProbeB)
PROBE=${PROBES[$SLURM_ARRAY_TASK_ID]}
echo "Probe:       $PROBE"
echo "Variant:     ${VARIANT:-both}"

# /usr/bin/time -v reports peak RSS to the .err log, so the first run doubles as the memory
# measurement this request was sized on guesswork (sacct MaxRSS confirms it afterwards).
/usr/bin/time -v "$PYTHON" "$SLURM_SUBMIT_DIR/scripts/run_scoring.py" \
    --seg "$SEG" --probe "$PROBE" ${VARIANT:+--variant "$VARIANT"}

echo "Finished: $(date)"
