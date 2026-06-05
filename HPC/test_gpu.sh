#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:25:00
#SBATCH --output=test_gpu_%j.out
#SBATCH --error=test_gpu_%j.err

# Load modules
module load triton/2024.1-gcc
module load cuda/12.2.1

# Activate venv
source $HOME/thesis/.venv/bin/activate

python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU name:', torch.cuda.get_device_name(0))
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = x @ y
print('GPU matrix multiplication OK')
print('Result shape:', z.shape)
"
