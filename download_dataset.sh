#!/bin/bash
#SBATCH -A IscrC_TranRR
#SBATCH -p boost_usr_prod
#SBATCH --time=06:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=download_data
#SBATCH --output=download.out
#SBATCH --error=download.err

# Go in project directory
cd $SLURM_SUBMIT_DIR

# directory destinazione
cd data

# venv temporaneo (non serve GPU)
python3 -m venv dlenv
source dlenv/bin/activate

# install gdown localmente dentro al venv
python3 -m pip install --user gdown

# download files (metti qui i tuoi file)
gdown --fuzzy "https://drive.google.com/file/d/1Gzzg0uVDlf4S09rLHFGKSHUVoMn93c-e/view?usp=drive_link" 
gdown --fuzzy "https://drive.google.com/file/d/1M0pAXNgG5kXBJuXDbBai1vnNMZLA2866/view?usp=drive_link"

deactivate

