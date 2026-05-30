#!/bin/bash
#SBATCH --job-name=resnet_lbbb
#SBATCH --output=resnet_%j.out
#SBATCH --error=resnet_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=6
#SBATCH --partition=batch
#SBATCH --clusters=wice
#SBATCH --account=lp_h_ecg_lbbb

echo "======================================"
echo "Starting ResNet ECG job"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "======================================"

START_TIME=$(date +%s)

# -------- Activate conda --------
source /data/leuven/376/vsc37666/miniconda3/etc/profile.d/conda.sh
conda activate lbbb_env

# -------- Check resources --------
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

# -------- Go to experiment directory --------
cd ~/lbbb_analysis/resnet_analysis

echo "Current directory:"
pwd

echo "Files:"
ls -lh

# -------- Run training --------
python train_resnet.py

# -------- Runtime --------
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "======================================"
echo "Finished"
echo "Total runtime: ${ELAPSED} seconds"
echo "Total runtime: $(($ELAPSED / 60)) minutes"
echo "======================================"