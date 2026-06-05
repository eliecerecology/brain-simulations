#!/bin/bash
#SBATCH --job-name=kuramoto_sim_v2
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=60:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eliecer.diazdiaz@aalto.fi

# Load modules in correct order
module load triton/2024.1-gcc
module load cuda/12.2.1

# Activate venv
source $HOME/thesis/.venv/bin/activate

# Print environment info
echo "========================================"
echo "Job started at: $(date)"
echo "Running on node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"
nvidia-smi
echo "========================================"
python --version
echo "========================================"

# Move to thesis directory where scripts and data live
cd $HOME/thesis

# Run simulation
python kuramoto_sim_v2.py

echo "========================================"
echo "Job finished at: $(date)"
echo "========================================"
