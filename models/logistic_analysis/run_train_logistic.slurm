#!/bin/bash
#SBATCH --job-name=logreg_6mo
#SBATCH --output=logreg_6mo_%j.out
#SBATCH --error=logreg_6mo_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --clusters=wice
#SBATCH --account=lp_h_ecg_lbbb

echo "Starting logistic regression 6-month baseline"
echo "Node: $(hostname)"
echo "Time: $(date)"

source /data/leuven/376/vsc37666/miniconda3/etc/profile.d/conda.sh
conda activate lbbb_env

cd ~/lbbb_analysis

which python

python 01_tabular_preprocess.py
python 05_train_logistic.py

echo "Finished"
echo "Time: $(date)"