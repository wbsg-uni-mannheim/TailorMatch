#!/bin/bash

#SBATCH --job-name=llama-validation
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


cd /ceph/aasteine/fine-tuning-paper

source /home/aasteine/miniconda3/etc/profile.d/conda.sh

conda activate /home/aasteine/miniconda3/envs/llama_matching

python src/validation.py 