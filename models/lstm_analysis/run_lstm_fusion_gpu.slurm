#!/bin/bash
#SBATCH --job-name=lstm_fusion_gpu
#SBATCH --output=lstm_fusion_gpu_%j.out
#SBATCH --error=lstm_fusion_gpu_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --clusters=wice
#SBATCH --account=lp_h_ecg_lbbb

echo "======================================"
echo "Starting LSTM fusion GPU job"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "======================================"

START_TIME=$(date +%s)

source /data/leuven/376/vsc37666/miniconda3/etc/profile.d/conda.sh
conda activate lbbb_env

cd ~/lbbb_analysis/lstm_analysis

echo "Current directory:"
pwd

echo "Files:"
ls -lh

echo "Checking GPU..."
nvidia-smi

python - <<'PY'
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
PY

python train_lstm_fusion.py

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "======================================"
echo "Finished"
echo "Total runtime: ${ELAPSED} seconds"
echo "Total runtime: $(($ELAPSED / 60)) minutes"
echo "======================================"