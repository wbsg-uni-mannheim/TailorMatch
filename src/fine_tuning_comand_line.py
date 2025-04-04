import wandb
from dotenv import load_dotenv
import pandas as pd
import random
import numpy as np
from datetime import datetime
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from datasets import load_dataset
import torch
import os
import json
import gc
from pathlib import Path
import argparse
from model_helpers import generate_answers, load_pipeline
from utils import (
    insert_product_descriptions_array, 
    clean_response, 
    calculate_results
)
import helper as analytics

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training and evaluation script with command line arguments')
    parser.add_argument('--base_path', default="../results", help='Base path for output')
    parser.add_argument('--model_id', default="meta-llama/Meta-Llama-3.1-8B-Instruct", help='Model ID')
    parser.add_argument('--training_file', default="../data/wdc/train_small/train_small_simple.csv", help='Training file path')
    parser.add_argument('--validation_file', default="../data/wdc/preprocessed_wdcproducts80cc20rnd000un_valid_small.pkl", help='Validation file path')
    parser.add_argument('--validation_prompt', default="../prompts/test_prompt.json", help='Validation prompt path')
    parser.add_argument('--test_prompts', default="../prompts/domain_promts.json", help='Test prompts path')
    parser.add_argument('--dataset_name', default="wdc_no_quantization", help='Dataset name')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--grad_accumulation_steps', type=int, default=32, help='Gradient accumulation steps')
    parser.add_argument('--cuda_devices', default="0,1", help='CUDA devices to use')
    return parser.parse_args()

def clear_gpu_memory():
    """Clear GPU memory"""
    torch.cuda.empty_cache()
    gc.collect()

def set_seeds(seed_value):
    """Set all seeds for reproducibility"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_model_and_tokenizer(model_id):
    """Initialize model and tokenizer"""
    compute_dtype = getattr(torch, "float16")
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        token=os.getenv("HUGGINGFACE_TOKEN"),
        cache_dir=os.getenv("CACHE_DIR"),
        quantization_config=quant_config,
    )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=os.getenv("HUGGINGFACE_TOKEN")
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer

def get_lora_config():
    """Get LoRA configuration"""
    return LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

def get_sft_config(output_dir, args, run_name):
    """Get SFT configuration"""
    return SFTConfig(
        max_seq_length=240,
        packing=True,
        output_dir=output_dir,
        num_train_epochs=args.max_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation_steps,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=args.learning_rate,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=1,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="polynomial",
        report_to=["wandb"],
        run_name=run_name
    )

def run_validation(checkpoint_path, args):
    """Run validation on a checkpoint"""
    print(f"Running validation for {checkpoint_path}")
    
    hf_pipeline = load_pipeline(checkpoint_path, args.batch_size)
    df = pd.read_pickle(args.validation_file)
    
    with open(args.validation_prompt, 'r') as file:
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
    def __init__(self, output_dir, args):
        self.output_dir = output_dir
        self.args = args
        self.best_f1 = 0
        self.best_checkpoint = None
    
    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called after a checkpoint save"""
        checkpoint_path = f"{self.output_dir}/checkpoint-{state.global_step}"
        if os.path.exists(checkpoint_path):
            metrics, results_df = run_validation(checkpoint_path, self.args)
            
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

def main():
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    
    # Set seeds
    set_seeds(args.seed)
    
    # Load dataset
    dataset = load_dataset('csv', data_files=args.training_file, split="train")
    
    # Create unique run name and output directory
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    run_name = f"{timestamp}_{args.model_id.replace('meta-llama/', '')}_{args.dataset_name}_lr_{args.learning_rate}"
    output_dir = Path(args.base_path) / args.model_id / args.dataset_name / f"lr_{args.learning_rate}" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize wandb
        wandb.init(
            project="First Paper",
            name=run_name,
            tags=[args.dataset_name, "simple", f"lr_{args.learning_rate}"],
            config={"learning_rate": args.learning_rate},
        )
        
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(args.model_id)
        
        # Initialize validation callback
        callback = ValidationCallback(str(output_dir), args)
        
        # Initialize trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=get_sft_config(str(output_dir), args, run_name),
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
        
        # Run final validation
        final_metrics, final_results = run_validation(final_model_path, args)
        final_results.to_json(f"{final_model_path}/final_validation_results.json")
        
        # Log final metrics
        wandb.log({
            'final_validation/f1': final_metrics['f1'],
            'final_validation/precision': final_metrics['precision'],
            'final_validation/recall': final_metrics['recall']
        })
        
    except Exception as e:
        print(f"Error during training: {e}")
        
    finally:
        # Clean up
        try:
            del model
            del trainer
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