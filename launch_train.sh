#!/bin/bash
#SBATCH -A IscrC_TranRR
#SBATCH -p boost_usr_prod
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=123000
#SBATCH --job-name=train_funnel
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

mkdir -p logs
mkdir -p saved_models

module purge
#module load profile/deeplrn
#module load cineca-ai/4.3.0

# --- FIX CRUCIALE PER LE LIBRERIE ---
# 1. Pulisce il PYTHONPATH ereditato dai moduli (che contiene il vecchio Pydantic)
unset PYTHONPATH

# 2. Attiva il tuo ambiente (che ora riempirà il PATH pulito)
source macch/bin/activate

# 3. Aggiunge la cartella corrente (per trovare utils)
export PYTHONPATH=$PYTHONPATH:.

# ---- SETUP TENSORFLOW ----
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"
export TF_CPP_MIN_LOG_LEVEL=2

export HF_HUB_OFFLINE=1
echo "=========================================="
echo "🚀 AVVIO TRAINING PIPELINE SU LEONARDO"
echo "=========================================="

# ---- DEBUG LIBRERIE ----
# Questo ci dirà nel file .out quale Pydantic sta caricando veramente
echo "🔍 VERIFICA VERSIONE PYDANTIC:"
python3 -c "import pydantic; print(f'Versione: {pydantic.VERSION}'); print(f'Percorso: {pydantic.__file__}')"
echo "------------------------------------------"

# ---- ESECUZIONE ----
srun python3 -u main.py

echo "=========================================="
echo "✅ TRAINING COMPLETATO"
echo "=========================================="

deactivate
