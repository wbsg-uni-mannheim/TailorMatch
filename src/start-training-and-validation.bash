#!/bin/bash

#SBATCH --job-name=llama-training-validation
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --mail-user=aaron.steiner@uni-mannheim.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:2
#SBATCH --time=15:00:00
#SBATCH --chdir=/ceph/aasteine/fine-tuning-paper
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=logs/%j_%x.log      
#SBATCH --error=logs/%j_%x.err      

cd /ceph/aasteine/fine-tuning-paper

source /home/aasteine/miniconda3/etc/profile.d/conda.sh

conda activate /home/aasteine/miniconda3/envs/llama_matching

python src/fine-tuning.py

# Run training and capture the output directory
echo "Starting training..."
TRAINING_OUTPUT=$(python src/fine-tuning.py | grep "Output Directory:" | awk '{print $3}')

if [ -z "$TRAINING_OUTPUT" ]; then
    echo "Error: Could not determine training output directory"
    exit 1
fi

echo "Training completed. Output directory: $TRAINING_OUTPUT"

# Run validation with the output directory
echo "Starting validation..."
python src/validation.py --checkpoint-folder "$TRAINING_OUTPUT"

echo "Training and validation completed successfully." 