#!/bin/bash
#SBATCH --job-name=cardioform
#SBATCH --output=cardioformer_like_%j.out
#SBATCH --error=cardioformer_like_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=10
#SBATCH --partition=batch
#SBATCH --clusters=wice
#SBATCH --account=lp_h_ecg_lbbb

echo "========================================================================"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_USER: $SLURM_JOB_USER"
echo "SLURM_JOB_ACCOUNT: $SLURM_JOB_ACCOUNT"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_CLUSTER_NAME: $SLURM_CLUSTER_NAME"
echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_JOB_CPUS_PER_NODE: $SLURM_JOB_CPUS_PER_NODE"
echo "Date: $(date)"
echo "Walltime: 00-08:00:00"
echo "========================================================================"

echo "======================================"
echo "Starting Cardioformer-like ECG job"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "======================================"

START_TIME=$(date +%s)

source /data/leuven/376/vsc37666/miniconda3/etc/profile.d/conda.sh
conda activate lbbb_env

cd ~/lbbb_analysis/cardioformer_like_analysis

echo "Current directory:"
pwd

echo "Python:"
which python

echo "TensorFlow check:"
python - <<'PY'
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
PY

python train_cardioformer_like.py

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "======================================"
echo "Finished"
echo "Runtime: $(($ELAPSED / 60)) minutes"
echo "======================================"