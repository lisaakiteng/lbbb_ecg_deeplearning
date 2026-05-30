#!/bin/bash
#SBATCH --job-name=lstm_bidir_fusion_cpu
#SBATCH --output=lstm_bidir_fusion_cpu_%j.out
#SBATCH --error=lstm_bidir_fusion_cpu_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --partition=batch
#SBATCH --clusters=wice
#SBATCH --account=lp_h_ecg_lbbb

echo "======================================"
echo "Starting LSTM_Bidir fusion CPU job"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "======================================"

START_TIME=$(date +%s)

source /data/leuven/376/vsc37666/miniconda3/etc/profile.d/conda.sh
conda activate lbbb_env

echo "Checking CPU resources..."
nproc

echo "Python:"
which python

echo "TensorFlow check:"
python - <<'PY'
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
PY

cd ~/lbbb_analysis/lstm_bidir_analysis

echo "Current directory:"
pwd

echo "Files:"
ls -lh

python train_lstm_bidir_fusion.py

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "======================================"
echo "Finished"
echo "Total runtime: ${ELAPSED} seconds"
echo "Total runtime: $(($ELAPSED / 60)) minutes"
echo "======================================"