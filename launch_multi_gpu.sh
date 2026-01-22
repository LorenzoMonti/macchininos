#!/bin/bash
#SBATCH -A IscrC_TranRR
#SBATCH -p boost_usr_prod
#SBATCH --time=06:00:00          # 24h sono troppe con 4 GPU, 6h bastano e avanzano
#SBATCH -N 1                     # 1 nodo
#SBATCH --ntasks-per-node=4      # <--- MODIFICA: 4 processi (uno per GPU)
#SBATCH --gres=gpu:4             # <--- MODIFICA: Richiediamo tutte le 4 GPU
#SBATCH --cpus-per-task=8        # <--- MODIFICA: 32 core totali / 4 task = 8 core a testa
#SBATCH --mem=123000             # memoria in MB
#SBATCH --job-name=eval_multi
#SBATCH --output=logs/job_eval_%j.out   # Consiglio: salva i log in una cartella
#SBATCH --error=logs/job_eval_%j.err

# Crea cartella logs se non esiste
mkdir -p logs

module purge

# ---- Carica Python e Moduli ----
module load python/3.11
python3 --version

module load profile/deeplrn
module load cineca-ai/4.3.0

# ---- Attiva il virtualenv ----
source macch/bin/activate

# Variabili ENV per TensorFlow/Keras
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

# ---- ESECUZIONE MULTI-GPU ----
echo "Avvio valutazione su 4 GPU..."

srun bash -c 'python3 evaluation_cluster.py --shard_id=$SLURM_PROCID --num_shards=4'

echo "Job completato."

# ---- Disattiva il virtualenv ----
deactivate