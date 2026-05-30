#!/bin/bash
#SBATCH --job-name=tl_frozen
#SBATCH --output=tl_exp7_frozen_%j.out
#SBATCH --error=tl_exp7_frozen_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --clusters=wice
#SBATCH --account=lp_h_ecg_lbbb

echo "======================================"
echo "Starting TL Step 1: Frozen backbone"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "======================================"

START_TIME=$(date +%s)

source /data/leuven/376/vsc37666/miniconda3/etc/profile.d/conda.sh
conda activate lbbb_env

python train_tl_step1_frozen.py

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "======================================"
echo "Finished TL Step 1"
echo "Total runtime: ${ELAPSED} seconds"
echo "Total runtime: $(($ELAPSED / 60)) minutes"
echo "======================================"