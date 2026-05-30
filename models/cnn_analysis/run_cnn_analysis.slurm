#!/bin/bash
#SBATCH --job-name=cnn_analysis_6mo
#SBATCH --output=cnn_analysis_6mo_%j.out
#SBATCH --error=cnn_analysis_6mo_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch
#SBATCH --clusters=wice
#SBATCH --account=lp_h_ecg_lbbb

echo "Starting CNN analysis 6-month rerun"
echo "Node: $(hostname)"
echo "Time: $(date)"

source /data/leuven/376/vsc37666/miniconda3/etc/profile.d/conda.sh
conda activate lbbb_env

cd ~/lbbb_analysis/cnn_analysis_v2

which python
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

python 05_train_cnn_analysis.py

echo "Finished"
echo "Time: $(date)"