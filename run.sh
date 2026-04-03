#!/bin/bash
#SBATCH --job-name=ddpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/network/jl7339/rl/ddpo-pytorch/output/1.out
#SBATCH --error=/scratch/network/jl7339/rl/ddpo-pytorch/error/1.err

module load anaconda3/2025.12
source ~/.bashrc
conda activate rl

export HF_HOME=/scratch/network/jl7339/hf
export TRANSFORMERS_CACHE=/scratch/network/jl7339/hf
export HF_DATASETS_CACHE=/scratch/network/jl7339/hf
export HUGGINGFACE_HUB_CACHE=/scratch/network/jl7339/hf/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /scratch/network/jl7339/rl/ddpo-pytorch

accelerate launch scripts/train.py
