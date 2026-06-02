#!/bin/bash
#SBATCH --job-name=kilosort4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/ceph/behrens/wreith/sandbox/ephys/logs/kilosort4_%j.out
#SBATCH --error=/ceph/behrens/wreith/sandbox/ephys/logs/kilosort4_%j.err

# ---------------------------------------------------------------------------
# Print job info for debugging
# ---------------------------------------------------------------------------
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Start time:   $(date)"
echo "Working dir:  $(pwd)"

# ---------------------------------------------------------------------------
# Create log directory if it doesn't exist
# ---------------------------------------------------------------------------
mkdir -p /ceph/behrens/wreith/sandbox/ephys/logs

# ---------------------------------------------------------------------------
# Activate conda environment
# ---------------------------------------------------------------------------
module load miniconda
#source ~/.bashrc
conda activate sleep_sandbox

# ---------------------------------------------------------------------------
# Run the sorting pipeline
# ---------------------------------------------------------------------------
python ~/aeon/sleep_sandbox/scripts/run_sorting.py

echo "Finished: $(date)"
