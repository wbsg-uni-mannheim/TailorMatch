from transformers import TrainerCallback, TrainerState, TrainerControl
import wandb
from dotenv import load_dotenv
import pandas as pd
import os
import json
import time
from pathlib import Path
from trl import SFTTrainer
from datasets import load_dataset
import torch
from datetime import datetime
import gc
from model_helpers import generate_answers, load_pipeline
from utils import (
    insert_product_descriptions_array, 
    clean_response, 
    calculate_results
)
import helper as analytics

from training_helpers import (
    clear_gpu_memory,
    set_seeds,
    setup_model_and_tokenizer,
    get_lora_config,
    get_sft_config
)

class TrainingConfig:
    def __init__(self):
        self.BASE_PATH = "../results"
        self.MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.TRAINING_FILE_PATH = "../data/wdc/train_small/train_small_simple.csv"
        self.VALIDATION_FILE_PATH = "../data/wdc/preprocessed_wdcproducts80cc20rnd000un_valid_small.pkl"
        self.VALIDATION_PROMPT_PATH = "../prompts/test_prompt.json"
        self.TEST_PROMPTS_PATH = "../prompts/domain_promts.json"
        self.DATASET_NAME = "wdc_no_quantization"
        self.LEARNING_RATES = [2.00E-04, 1.00E-06, 1.00E-05, 1.00E-03]
        self.SEED = 42
        self.MAX_EPOCHS = 10
        self.BATCH_SIZE = 4
        self.GRAD_ACCUMULATION_STEPS = 32
        
        # Set CUDA devices
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
        
        # Load environment variables
        load_dotenv()

def run_validation(checkpoint_path, config):
    """Run validation on a checkpoint"""
    print(f"Running validation for {checkpoint_path}")
    
    hf_pipeline = load_pipeline(checkpoint_path, config.BATCH_SIZE)
    df = pd.read_pickle(config.VALIDATION_FILE_PATH)
    
    with open(config.VALIDATION_PROMPT_PATH, 'r') as file:
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
            print(f"Error in validation: {e}")
            responses = [""] * len(df)
        
        for idx, (index, row) in enumerate(df.iterrows()):
            response = responses[idx] if idx < len(responses) else ""
            response = response[1].get("content") if isinstance(response, tuple) else ""
            
            result_row = {
                'task': title,
                'chatbot_question': messages[idx],
                'chatbot_response_raw': response,
                'chatbot_response_clean': clean_response(response)
            }
            
            for col in df.columns:
                result_row[col] = row[col]
            
            result_rows.append(result_row)
    
    results_df = pd.DataFrame(result_rows)
    results_df.loc[results_df['chatbot_response_clean'] == -1, 'chatbot_response_clean'] = 0
    
    metrics = analytics.calculate_scores(results_df)
    
    del hf_pipeline
    clear_gpu_memory()
    
    return metrics, results_df

class ValidationCallback(TrainerCallback):
    """Callback for validation during training"""
    def __init__(self, output_dir, config):
        self.output_dir = output_dir
        self.config = config
        self.best_f1 = 0
        self.best_checkpoint = None
    
    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called after a checkpoint save"""
        checkpoint_path = f"{self.output_dir}/checkpoint-{state.global_step}"
        if os.path.exists(checkpoint_path):
            metrics, results_df = run_validation(checkpoint_path, self.config)
            
            if not results_df.empty:
                # Save validation results
                results_df.to_json(f"{checkpoint_path}/validation_results.json")
                
                # Track best checkpoint
                if metrics['f1'] > self.best_f1:
                    self.best_f1 = metrics['f1']
                    self.best_checkpoint = checkpoint_path
                
                # Log to wandb
                wandb.log({
                    'validation/f1': metrics['f1'],
                    'validation/precision': metrics['precision'],
                    'validation/recall': metrics['recall'],
                    'epoch': state.epoch,
                    'global_step': state.global_step
                })
        return control

def train_and_evaluate(config, model, tokenizer, dataset, lr, output_dir, run_name):
    """Train model and run evaluation"""
    
    callback = ValidationCallback(output_dir, config)
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=get_sft_config(str(output_dir), lr, run_name, config),
        peft_config=get_lora_config(),
        tokenizer=tokenizer,
        callbacks=[callback]
    )
    
    # Train model
    trainer.train()
    
    # Save final model
    final_model_path = f"{output_dir}/{run_name}"
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(f"{final_model_path}_tokenizer")
    
    # Save to wandb
    model_artifact = wandb.Artifact(run_name, type='model')
    model_artifact.add_dir(run_name)
    wandb.log_artifact(model_artifact)
    
    # Clear memory
    del trainer
    clear_gpu_memory()

def main():
    """Main training loop"""
    # Initialize configuration
    config = TrainingConfig()
    set_seeds(config.SEED)
    
    # Load dataset
    dataset = load_dataset('csv', data_files=config.TRAINING_FILE_PATH, split="train")
    
    # Train for each learning rate
    for lr in config.LEARNING_RATES:
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        run_name = f"{timestamp}_{config.MODEL_ID.replace('meta-llama/', '')}_{config.DATASET_NAME}_lr_{lr}"
        output_dir = Path(config.BASE_PATH) / config.MODEL_ID / config.DATASET_NAME / f"lr_{lr}" / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Initialize wandb
            wandb.init(
                project="First Paper",
                name=run_name,
                tags=[config.DATASET_NAME, "simple", f"lr_{lr}"],
                config={"learning_rate": lr},
                reinit=True
            )
            
            # Setup model and tokenizer
            model, tokenizer = setup_model_and_tokenizer(config)
            
            # Train and evaluate
            train_and_evaluate(config, model, tokenizer, dataset, lr, output_dir, run_name)
            
        except Exception as e:
            print(f"Error during training with learning rate {lr}: {e}")
            
        finally:
            # Clean up
            try:
                del model
            except:
                pass
            
            clear_gpu_memory()
            
            # Close wandb run
            try:
                wandb.finish()
            except:
                pass
            
            # Additional cleanup
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    main()