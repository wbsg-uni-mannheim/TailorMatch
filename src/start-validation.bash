#!/bin/bash

#SBATCH --job-name=llama-validation
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=30G
#SBATCH --mail-user=aaron.steiner@uni-mannheim.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --chdir=/ceph/aasteine/fine-tuning-paper
#SBATCH --partition=gpu-vram-94gb
#SBATCH --output=logs/%x_%j.log      
#SBATCH --error=logs/%x_%j.err      


cd /ceph/aasteine/fine-tuning-paper

source /home/aasteine/miniconda3/etc/profile.d/conda.sh

conda activate /home/aasteine/miniconda3/envs/llama_matching

python src/validation.py --checkpoint-folder results/meta-llama/Meta-Llama-3.1-8B-Instruct/regular_train_small_all_fields/lr_0.0002/2025-05-12-12-20-23