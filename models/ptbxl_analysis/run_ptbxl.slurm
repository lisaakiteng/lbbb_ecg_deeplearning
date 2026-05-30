#!/bin/bash
#SBATCH --job-name=ptbxl_simple
#SBATCH --output=ptbxl_%j.out
#SBATCH --error=ptbxl_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --clusters=wice
#SBATCH --account=lp_h_ecg_lbbb

echo "======================================"
echo "Starting PTB-XL training"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "======================================"

START_TIME=$(date +%s)

# -------- Activate conda ----------
source /data/leuven/376/vsc37666/miniconda3/etc/profile.d/conda.sh
conda activate lbbb_env

# RUN MODEL
python train_ptbxl.py cnn_4layers_v1

# -------- Runtime --------
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "======================================"
echo "Finished"
echo "Total runtime: ${ELAPSED} seconds"
echo "Total runtime: $(($ELAPSED / 60)) minutes"
echo "======================================"