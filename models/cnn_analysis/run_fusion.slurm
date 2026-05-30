#!/bin/bash
#SBATCH --job-name=fusion_6mo
#SBATCH --output=fusion_6mo_%j.out
#SBATCH --error=fusion_6mo_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch
#SBATCH --clusters=wice
#SBATCH --account=lp_h_ecg_lbbb

echo "Starting fusion 6-month rerun"
echo "Node: $(hostname)"
echo "Time: $(date)"

source /data/leuven/376/vsc37666/miniconda3/etc/profile.d/conda.sh
conda activate lbbb_env

cd ~/lbbb_analysis/cnn_analysis/fusion_v2

which python
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

python train_fusion.py

echo "Finished"
echo "Time: $(date)"