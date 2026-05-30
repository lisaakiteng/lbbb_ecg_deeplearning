#!/bin/bash
#SBATCH --job-name=cnn_lstm_fusion
#SBATCH --output=cnn_lstm_fusion_%j.out
#SBATCH --error=cnn_lstm_fusion_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --partition=batch
#SBATCH --clusters=wice
#SBATCH --account=lp_h_ecg_lbbb

echo "======================================"
echo "Starting CNN-LSTM Fusion job"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "======================================"

START_TIME=$(date +%s)

source /data/leuven/376/vsc37666/miniconda3/etc/profile.d/conda.sh
conda activate lbbb_env

cd ~/lbbb_analysis/cnn_lstm_analysis

python train_cnn_lstm_fusion.py

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "======================================"
echo "Finished"
echo "Runtime: $(($ELAPSED / 60)) minutes"
echo "======================================"