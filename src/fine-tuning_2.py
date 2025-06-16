import wandb
from dotenv import load_dotenv
import json

import random
import numpy as np
from datetime import datetime
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from datasets import load_dataset
import torch
import os
from pathlib import Path
import huggingface_hub

# Enable better CUDA error reporting
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Load environment variables
load_dotenv()

huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Configuration
class TrainingConfig:
    def __init__(self):
        self.BASE_PATH = "results"
        self.MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.TRAINING_FILE_PATH = "data/wdc/train_small/augmentation/preprocessed_wdcproducts80cc20rnd000un_train_small_all_augmentation_nplaug_matching_examples.csv"
        self.DATASET_NAME = "all_augmentation_nplaug_matching_examples"
        self.LEARNING_RATES = 2.00E-04
        self.SEED = 42
        self.MAX_EPOCHS = 30
        self.BATCH_SIZE = 4
        self.GRAD_ACCUMULATION_STEPS = 20
        

def set_seeds(seed_value):
    """Set all seeds for reproducibility"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_model_and_tokenizer(config):
    """Initialize model and tokenizer"""
    compute_dtype = getattr(torch, "float16")
    
    # Remove quantization config
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        device_map="auto",
        token=os.getenv("HUGGINGFACE_TOKEN"),
        cache_dir=os.getenv("CACHE_DIR"),
        torch_dtype=compute_dtype,  # Use float16 instead of quantization
    )
    
    model.config.use_cache = True
    model.config.pretraining_tp = 1
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID,
        trust_remote_code=True,
        token=os.getenv("HUGGINGFACE_TOKEN"),
        cache_dir=os.getenv("CACHE_DIR")
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print("--- Verification ---")
    print(f"Model Vocab Size (config): {model.config.vocab_size}")
    print(f"Model Embedding Size: {model.get_input_embeddings().weight.shape[0]}")
    print(f"Tokenizer Vocab Size (len): {len(tokenizer)}")
    print(f"Tokenizer Vocab Size (attr): {tokenizer.vocab_size}") # Base vocab without added tokens
    # Check if special tokens were added automatically
    print(f"Tokenizer Added Tokens: {len(tokenizer.added_tokens_decoder)}")
    # Ensure pad token ID is valid if used
    if tokenizer.pad_token_id is not None:
         print(f"Pad token ID: {tokenizer.pad_token_id} (Valid range: 0 to {len(tokenizer)-1})")
    print("--------------------")
    
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

def get_sft_config(output_dir, learning_rate, run_name, config, tokenizer):
    """Get SFT configuration"""
    return SFTConfig(
        max_seq_length=512,
        packing=True,
        output_dir=output_dir,
        num_train_epochs=config.MAX_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUMULATION_STEPS,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,
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

def save_artifacts(trainer, output_dir, run_name):
    """Save model artifacts and log to wandb"""
    # Save model and tokenizer locally
    trainer.model.save_pretrained(f"{output_dir}/{run_name}")
    trainer.tokenizer.save_pretrained(f"{output_dir}/{run_name}_tokenizer")
    
    # Create and log wandb artifacts
    model_artifact = wandb.Artifact(run_name, type='model')
    model_artifact.add_dir(run_name)
    wandb.log_artifact(model_artifact)
    
    tokenizer_artifact = wandb.Artifact(f"{run_name}_tokenizer", type='tokenizer')
    tokenizer_artifact.add_dir(f"{run_name}_tokenizer")
    wandb.log_artifact(tokenizer_artifact)

def main():
    # Initialize configuration
    config = TrainingConfig()
    set_seeds(config.SEED)
    
    # Load dataset
    dataset = load_dataset('csv', data_files=config.TRAINING_FILE_PATH, split="train")
    
    lr = config.LEARNING_RATES
    # Create unique run name and output directory
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    run_name = f"{timestamp}_{config.MODEL_ID.replace('meta-llama/', '')}_{config.DATASET_NAME}_lr_{lr}"
    output_dir = Path(config.BASE_PATH) / config.MODEL_ID / config.DATASET_NAME / f"lr_{lr}" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project="First Paper",
        name=run_name,
        tags=[config.DATASET_NAME, "simple", f"lr_{lr}"],
        config={"learning_rate": lr},
        reinit=True
    )
    
    # Save run configuration to JSON
    run_config = {
        "run_name": run_name,
        "wandb_run_id": wandb.run.id,
        "timestamp": timestamp,
        "model": config.MODEL_ID,
        "training_file": config.TRAINING_FILE_PATH,
        "dataset_name": config.DATASET_NAME,
        "learning_rate": lr,
        "max_epochs": config.MAX_EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "gradient_accumulation_steps": config.GRAD_ACCUMULATION_STEPS,
        "output_dir": str(output_dir),
        "training_params": {
            "max_seq_length": 240,
            "packing": True,
            "optim": "paged_adamw_32bit",
            "fp16": True,
            "bf16": False,
            "max_grad_norm": 1,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "polynomial"
        },
        "lora_config": {
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "r": 64,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
    }
    
    # Save the configuration to a JSON file
    config_file = output_dir / "runconfig.json"
    with open(config_file, 'w') as f:
        json.dump(run_config, f, indent=4)
    
    # Log training parameters
    print(f"Starting training with learning rate: {lr}")
    print(f"Training Data Path: {config.TRAINING_FILE_PATH}")
    print(f"Output Directory: {output_dir}")
    print(f"Run configuration saved to: {config_file}")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=get_sft_config(str(output_dir), lr, run_name, config, tokenizer),
        peft_config=get_lora_config(),
        processing_class=tokenizer,
    )
    
    # Train model
    trainer.train()
    
    # Save artifacts
    #save_artifacts(trainer, str(output_dir), run_name)
    
    # Close wandb run
    wandb.finish()
    
    # Return the output directory
    return str(output_dir)

if __name__ == "__main__":
    output_dir = main()
    print(f"Output Directory: {output_dir}")