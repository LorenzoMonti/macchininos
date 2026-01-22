#!/bin/bash
#SBATCH -A IscrC_TranRR
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00          # tempo massimo
#SBATCH -N 1                     # 1 nodo
#SBATCH --ntasks-per-node=1      # 1 task
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --mem=123000             # memoria in MB
#SBATCH --job-name=evaluation_rrls
#SBATCH --output=job_eval.out
#SBATCH --error=job_eval.err

module purge

# ---- carica Python 3.11 come modulo ----
module load python/3.11
python3 --version  # debug: deve stampare 3.11.x

# ---- Carica i moduli necessari ----
module load profile/deeplrn
module load cineca-ai/4.3.0

# ---- Attiva il virtualenv ----
source macch/bin/activate

export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

# ---- Esegui lo script di valutazione ----
# srun erediterà le variabili exportate sopra
srun python3 evaluation_cluster.py

# ---- Disattiva il virtualenv ----
deactivate

