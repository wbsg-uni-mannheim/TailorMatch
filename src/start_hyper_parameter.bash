#!/bin/bash

#SBATCH --job-name=llama-testing
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=30G
#SBATCH --mail-user=aaron.steiner@uni-mannheim.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:2
#SBATCH --time=10:00:00
#SBATCH --chdir=/ceph/aasteine/fine-tuning-paper
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=logs/%x_%j.log      
#SBATCH --error=logs/%x_%j.err      


source /home/aasteine/miniconda3/etc/profile.d/conda.sh


cd /ceph/aasteine/fine-tuning-paper

# Activate your environment using the full path
conda activate /home/aasteine/miniconda3/envs/llama_matching

# Print activated environment information
echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python being used: $(which python)"
echo "Python version: $(python --version)"


cd src

torchrun --nproc_per_node=2 fine-tuning.py  # for 2 GPUs