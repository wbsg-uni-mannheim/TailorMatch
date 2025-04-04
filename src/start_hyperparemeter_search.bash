#!/bin/bash

#SBATCH --job-name=llama-testing
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=30G
#SBATCH --mail-user=aaron.steiner@uni-mannheim.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --chdir=/ceph/aasteine/fine-tuning-paper/src
#SBATCH --partition=gpu-vram-32gb
#SBATCH --output=logs/%x_%j.log      
#SBATCH --error=logs/%x_%j.err      


cd /ceph/aasteine/fine-tuning-paper


# Activate environment
source venv/bin/activate
pip install -r requirements.txt
pip list


# Print activated environment information
echo "Python being used: $(which python)"
echo "Python version: $(python --version)"

cd src

# Define hyperparameter ranges
learning_rates=(0.0002 0.000001 0.00001 0.001)
batch_sizes=(8 16 32)
grad_accumulation_steps=(16 20 24)

# Fixed parameters
MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
TRAINING_FILE="../data/wdc/train_small/train_small_simple.csv"
VALIDATION_FILE="../data/wdc/preprocessed_wdcproducts80cc20rnd000un_valid_small.pkl"
VALIDATION_PROMPT="../prompts/test_prompt.json"
TEST_PROMPTS="../prompts/domain_promts.json"
DATASET_NAME="wdc_small"
BASE_PATH="../results/hyperparameter_search"
SEED=42
CUDA_DEVICES="4,5,6,7"
EPOCHS=10

# Create logs directory if it doesn't exist
mkdir -p hyperparameter_search_logs

# Log start time
echo "Starting hyperparameter search at $(date)"
echo "----------------------------------------"

# Nested loops for hyperparameter search
for lr in "${learning_rates[@]}"; do
    for bs in "${batch_sizes[@]}"; do
        for grad_steps in "${grad_accumulation_steps[@]}"; do
            # Create a unique identifier for this run
            RUN_ID="lr${lr}_bs${bs}_gs${grad_steps}"
            
            echo "Starting run: $RUN_ID at $(date)"
            echo "Parameters:"
            echo "- Learning Rate: $lr"
            echo "- Batch Size: $bs"
            echo "- Epochs: $EPOCHS (fixed)"
            echo "- Gradient Accumulation Steps: $grad_steps"
            
            # Run the training script with current hyperparameters
            python fine_tuning_comand_line.py \
                --base_path "$BASE_PATH" \
                --model_id "$MODEL_ID" \
                --training_file "$TRAINING_FILE" \
                --validation_file "$VALIDATION_FILE" \
                --validation_prompt "$VALIDATION_PROMPT" \
                --test_prompts "$TEST_PROMPTS" \
                --dataset_name "$DATASET_NAME" \
                --learning_rate "$lr" \
                --seed "$SEED" \
                --max_epochs "$EPOCHS" \
                --batch_size "$bs" \
                --grad_accumulation_steps "$grad_steps" \
                --cuda_devices "$CUDA_DEVICES" \
                2>&1 | tee "hyperparameter_search_logs/${RUN_ID}.log"
            
            # Check if the run was successful
            if [ $? -eq 0 ]; then
                echo "Run $RUN_ID completed successfully"
                
                # Archive validation results
                RESULTS_DIR="${BASE_PATH}/${MODEL_ID/meta-llama\//}/${DATASET_NAME}/lr_${lr}/$(date +%Y-%m-%d-%H-%M-%S)/${RUN_ID}"
                if [ -d "$RESULTS_DIR" ]; then
                    echo "Archiving validation results from $RESULTS_DIR"
                    cp -r "$RESULTS_DIR/validation_results.json" "hyperparameter_search_logs/${RUN_ID}_validation.json" 2>/dev/null || true
                    cp -r "$RESULTS_DIR/final_validation_results.json" "hyperparameter_search_logs/${RUN_ID}_final_validation.json" 2>/dev/null || true
                fi
            else
                echo "Run $RUN_ID failed with exit code $?"
            fi
            
            echo "----------------------------------------"
            
            # Clean up CUDA cache
            nvidia-smi --gpu-reset 2>/dev/null || true
            
            # Optional: Add a small delay between runs to ensure system stability
            sleep 30
        done
    done
done

echo "Hyperparameter search completed at $(date)"

# Calculate total number of experiments
total_experiments=$((${#learning_rates[@]} * ${#batch_sizes[@]} * ${#grad_accumulation_steps[@]}))
echo "Total experiments run: $total_experiments"

# Summarize results
echo "Creating summary of all runs..."
{
    echo "Run ID,Learning Rate,Batch Size,Epochs,Grad Accumulation Steps,Status,Best F1 Score"
    for lr in "${learning_rates[@]}"; do
        for bs in "${batch_sizes[@]}"; do
            for grad_steps in "${grad_accumulation_steps[@]}"; do
                RUN_ID="lr${lr}_bs${bs}_gs${grad_steps}"
                if [ -f "hyperparameter_search_logs/${RUN_ID}.log" ]; then
                    status="Completed"
                    # Try to extract best F1 score from validation results if available
                    final_f1=$(jq -r '.validation/f1' "hyperparameter_search_logs/${RUN_ID}_final_validation.json" 2>/dev/null || echo "N/A")
                else
                    status="Failed/Not Run"
                    final_f1="N/A"
                fi
                echo "$RUN_ID,$lr,$bs,$EPOCHS,$grad_steps,$status,$final_f1"
            done
        done
    done
} > hyperparameter_search_summary.csv

echo "Summary saved to hyperparameter_search_summary.csv"

# Create a final report with best performing model
echo "Creating final report..."
{
    echo "Hyperparameter Search Results"
    echo "============================"
    echo "Total experiments: $total_experiments"
    echo "Completed at: $(date)"
    echo ""
    echo "Best performing models (by F1 score):"
    # Sort results by F1 score (excluding N/A) and show top 3
    tail -n +2 hyperparameter_search_summary.csv | \
        grep -v "N/A" | \
        sort -t',' -k7 -nr | \
        head -n 3 | \
        awk -F',' '{printf "F1: %s - LR: %s, BS: %s, GS: %s\n", $7, $2, $3, $5}'
} > hyperparameter_search_report.txt

echo "Final report saved to hyperparameter_search_report.txt"