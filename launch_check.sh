#!/bin/bash
#SBATCH -A IscrC_TranRR
#SBATCH -p boost_usr_prod
#SBATCH --time=00:10:00          # Bastano 10 minuti
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1             # Richiediamo esplicitamente 1 GPU
#SBATCH --mem=16000
#SBATCH --job-name=check_gpu
#SBATCH --output=check_gpu.out
#SBATCH --error=check_gpu.err

# Carica i moduli
module load profile/deeplrn
module load cineca-ai/4.1.1

# Attiva ambiente
source met/bin/activate

# Opzionale: Configurazione TF per evitare errori noti
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Esegui il check
srun python3 check_gpu.py