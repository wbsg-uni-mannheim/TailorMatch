from dotenv import load_dotenv
import helper
import time
import helper as analytics
import json
import pandas as pd
import torch
import os
from test_model import process_datasets
from utils import insert_product_descriptions_array, clean_response
from tqdm import tqdm
import wandb
from model_helpers import generate_answers, load_pipeline

# Load OPENAI_API_KEY from .env file
load_dotenv()

# Enable better CUDA error reporting
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

CHECKPOINT_FOLDER = "results/meta-llama/Meta-Llama-3.1-8B-Instruct/upsampled/lr_0.0002/2025-05-02-16-25-14"
VALIDATION_PROMPT_PATH = "./prompts/test_prompt.json"
VALIDATION_FILE_PATH = "./data/wdc/preprocessed_wdcproducts80cc20rnd000un_valid_small.pkl"

TEST_PROMPTS = "./prompts/domain_promts.json"

# Load run configuration
run_config_path = os.path.join(CHECKPOINT_FOLDER, "runconfig.json")

if not os.path.exists(run_config_path):
    print(f"Error: {run_config_path} not found")
    exit(1)

try:
    with open(run_config_path, 'r') as f:
        run_config = json.load(f)
    WANDDB_ID = run_config["wandb_run_id"]  # Get wandb run ID from config
except json.JSONDecodeError as e:
    print(f"Error parsing runconfig.json: {e}")
    exit(1)
except KeyError:
    print("Error: wandb_run_id not found in runconfig.json")
    exit(1)

batch_size = 32
device_map = "auto"

def load_run_config(config_path):
    """Load run configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def list_checkpoint_folders(directory):
    """List all checkpoint folders in the given directory"""
    checkpoint_folders = []
    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            if 'checkpoint' in folder:
                checkpoint_folders.append(os.path.join(root, folder))
    return checkpoint_folders

def get_checkpoint_number(path):
    """Extract checkpoint number from path"""
    return int(path.split('-')[-1])

def run_validation(checkpoint_path, config, wandb_run_id=None):
    """Run validation on a single checkpoint"""
    print(f"Processing checkpoint {checkpoint_path}")
    try:
        # Clear CUDA cache before loading new model
        torch.cuda.empty_cache()
        
        # Load the pipeline
        hf_pipeline = load_pipeline(checkpoint_path, config['batch_size'])
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Load validation dataset
        df = pd.read_pickle(VALIDATION_FILE_PATH)
        
        # Load validation prompts
        with open(VALIDATION_PROMPT_PATH, 'r') as file:
            prompts = json.load(file)
        
        result_rows = []
        
        for task in prompts:
            title = task['title']
            prompt_template = task['prompt']
            
            messages = [
                insert_product_descriptions_array(
                    prompt_template, row['title_left'], row['title_right']
                )
                for _, row in df.iterrows()
            ]
            
            try:
                responses = generate_answers(messages, hf_pipeline)
            except Exception as e:
                print(f"Error during generation: {e}")
                responses = [""] * len(df)
                torch.cuda.empty_cache()
            
            for idx, (index, row) in enumerate(df.iterrows()):
                response = responses[idx] if idx < len(responses) else ""
                try:
                    response = response[1].get("content")
                except Exception as e:
                    print(f"Error: {e}")
                    response = ""
                
                result_row = {
                    'task': title,
                    'chatbot_question': messages[idx],
                    'chatbot_response_raw': response,
                    'chatbot_response_clean': clean_response(response)
                }
                
                for col in df.columns:
                    result_row[col] = row[col]
                
                result_rows.append(result_row)
        
        # Convert results to DataFrame
        all_columns = ['task', 'chatbot_question', 'chatbot_response_raw',
                      'chatbot_response_clean'] + list(df.columns)
        results_df = pd.DataFrame(result_rows, columns=all_columns)
        results_df.loc[results_df['chatbot_response_clean'] == -1, 'chatbot_response_clean'] = 0
        
        # Calculate metrics
        f1, precision, recall = analytics.calculate_scores(results_df)
        print(f"F1: {f1}, Precision: {precision}, Recall: {recall}")
        
        # Log to wandb if run_id is provided
        if wandb_run_id:
            step = int(checkpoint_path.split("/")[-1].replace("checkpoint-", ""))
            epoch = helper.get_epoch_from_checkpoint(list_checkpoint_folders(os.path.dirname(checkpoint_path)), step)
            helper.log_metrics_to_existing_wandb_run(
                "First Paper", wandb_run_id, step, epoch, f1, precision, recall)
        
        # Save results
        results_df.to_json(f"{checkpoint_path}/validation_results.json")
        print("File Saved")
        
        return f1, precision, recall
        
    finally:
        # Clean up resources
        if 'hf_pipeline' in locals():
            del hf_pipeline
        torch.cuda.empty_cache()
        gc.collect()

def main():
    # Load run configuration
    config_path = os.path.join(CHECKPOINT_FOLDER, "runconfig.json")
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found")
        return
    
    config = load_run_config(config_path)
    
    # Get checkpoint paths
    checkpoint_paths = list_checkpoint_folders(CHECKPOINT_FOLDER)
    checkpoint_paths = sorted(checkpoint_paths, key=get_checkpoint_number)
    
    # Process each checkpoint
    results = []
    for checkpoint_path in tqdm(checkpoint_paths, desc="Processing checkpoints"):
        # Skip if validation results already exist
        if os.path.exists(f"{checkpoint_path}/validation_results.json"):
            print(f"Validation results already exist for {checkpoint_path}")
            continue
        
        # Run validation
        f1, precision, recall = run_validation(checkpoint_path, config, config.get('wandb_run_id'))
        
        # Store results
        result = {
            "checkpoint_path": checkpoint_path,
            "checkpoint_number": checkpoint_path.split("/")[-1].replace("checkpoint-", ""),
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
        results.append(result)
    
    # Save overall results
    if results:
        df = pd.DataFrame(results)
        df_sorted = df.sort_values(by='f1', ascending=False)
        df_sorted.to_csv(f"{config['output_dir']}/validation_results.csv", index=False)
        
        # Get best checkpoint
        best_checkpoint_path = df_sorted.iloc[0]['checkpoint_path']
        print(f"Best Checkpoint Path: {best_checkpoint_path}")
        print(f"Best F1: {df_sorted.iloc[0]['f1']}")
        
        # Run final validation on test datasets
        hf_pipeline = load_pipeline(best_checkpoint_path, config['batch_size'])
        test_datasets = [
            {
                "dataset_name": "wdc-fullsize",
                "dataset_path": "/ceph/aasteine/fine-tuning-paper/data/wdc/wdcproducts80cc20rnd050un_test_gs.pkl"
            }
        ]
        process_datasets(test_datasets, hf_pipeline, TEST_PROMPTS, CHECKPOINT_FOLDER)

if __name__ == "__main__":
    main()
