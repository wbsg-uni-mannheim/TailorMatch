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
import pandas as pd
from datasets import load_dataset, Dataset
import torch
import os
from pathlib import Path
import huggingface_hub
from utils import serialize_product

# Enable better CUDA error reporting
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
USE_ALL_FIELDS = True
MAX_SEQ_LENGTH = 1024

# Load environment variables
load_dotenv()

huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))

PROMPT_TEMPLATE = "Do the two product descriptions refer to the same real-world product? Entity 1: 'Entity 1'. Entity 2: 'Entity 2'."

def insert_product_descriptions(prompt_template: str, product1: str, product2: str):
    # Replace placeholder texts with actual product descriptions
    prompt = prompt_template.replace("'Entity 1'", product1).replace("'Entity 2'", product2)
    return prompt

def transform_label(label: int):
    if label == 1 or label == "1":
        return "Yes"
    elif label == 0 or label == "0":
        return "No"
    else:
        raise ValueError("Label must be 0 or 1")

def create_training_example(prompt_template: str, product1: str, product2: str, label: int):
    # Create the prompt with product descriptions
    prompt = insert_product_descriptions(prompt_template, product1, product2)
    if label == 1 or label == "1":
        response = "Yes"
    elif label == 0 or label == "0":
        response = "No"
    else:
        raise ValueError("Label must be 0 or 1")
    
    # Create the training example in the format required for fine-tuning
    return prompt, response

# Configuration
class TrainingConfig:
    def __init__(self):
        self.BASE_PATH = "results"
        self.MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.TRAINING_FILE_PATH = "temp.csv"
        self.DATASET_NAME = "Walmart-Amazon with explanations"
        self.LEARNING_RATES = 2.00E-04
        self.SEED = 42
        self.MAX_EPOCHS = 15
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
        token=os.getenv("HUGGINGFACE_TOKEN"),
        cache_dir=os.getenv("CACHE_DIR"),
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))  # grow embedding matrix

    # The loss kernel uses model.config.vocab_size – keep it in sync!
    model.config.vocab_size = len(tokenizer)
    
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
    
    # Ensure model and tokenizer vocab sizes match – otherwise CrossEntropy will crash with CUDA device-side assert
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        print(f"[INFO] Resizing token embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)} to accommodate added special tokens")
        model.resize_token_embeddings(len(tokenizer))
    
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
        max_seq_length=MAX_SEQ_LENGTH,
        output_dir=output_dir,
        packing=True,
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
    # Convert output_dir to Path if it's not already
    output_path = Path(output_dir)
    
    # Save model and tokenizer locally
    trainer.model.save_pretrained(output_path / run_name)
    trainer.tokenizer.save_pretrained(output_path / f"{run_name}_tokenizer")
    
    # Create and log wandb artifacts
    model_artifact = wandb.Artifact(run_name, type='model')
    model_artifact.add_dir(str(output_path / run_name))
    wandb.log_artifact(model_artifact)
    
    tokenizer_artifact = wandb.Artifact(f"{run_name}_tokenizer", type='tokenizer')
    tokenizer_artifact.add_dir(str(output_path / f"{run_name}_tokenizer"))
    wandb.log_artifact(tokenizer_artifact)

def assert_ids_ok(data, tok, max_id):
    for ex in data.select(range(50)):          # first 50 samples are enough
        if max(tok(ex["prompt"]+ex["completion"])["input_ids"]) >= max_id:
            raise ValueError("Found token id ≥ vocab_size")

def main():
    # Initialize configuration
    config = TrainingConfig()
    set_seeds(config.SEED)
    
    if ".csv" in config.TRAINING_FILE_PATH:
        # Load dataset
        dataset = load_dataset('csv', data_files=config.TRAINING_FILE_PATH, split="train")
    elif USE_ALL_FIELDS:
        if ".pkl" in config.TRAINING_FILE_PATH:
            train_set = pd.read_pickle(config.TRAINING_FILE_PATH, compression="gzip")
        elif ".json" in config.TRAINING_FILE_PATH:
            train_set = pd.read_json(config.TRAINING_FILE_PATH, compression="gzip")
        else:
            raise ValueError(f"Unsupported file type: {config.TRAINING_FILE_PATH}")
        
        # Create training examples
        training_examples = []

        for index, row in train_set.iterrows():
            product_1 = serialize_product(row, "left")
            product_2 = serialize_product(row, "right")
            response = transform_label(row.get("label"))  # Convert label to string
            
            # check if the row has an explanation
            if "explanation" in row.index:
                response = row.get("explanation")
            
            prompt = insert_product_descriptions(PROMPT_TEMPLATE, product_1, product_2)
            training_examples.append({"prompt": prompt, "completion": response})

        dataset = Dataset.from_list(training_examples)
    else:
        train_set = pd.read_pickle(config.TRAINING_FILE_PATH, compression="gzip")
        # Create training examples
        training_examples = []

        for index, row in train_set.iterrows():
            product_1 = serialize_product(row, "left")
            product_2 = serialize_product(row, "right")
            label = str(row.get("label"))  # Convert label to string
            
            prompt, response = create_training_example(PROMPT_TEMPLATE, product_1, product_2, label)
            training_examples.append({"prompt": prompt, "completion": response})

        # Convert to Dataset format
        dataset = Dataset.from_list(training_examples)
    
    lr = config.LEARNING_RATES
    # Create unique run name and output directory
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    run_name = f"{timestamp}_{config.MODEL_ID.replace('meta-llama/', '')}_{config.DATASET_NAME}_lr_{lr}"
    output_dir = Path(config.BASE_PATH) / config.MODEL_ID / config.DATASET_NAME / f"lr_{lr}" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project="Example Selection",
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
            "max_seq_length": MAX_SEQ_LENGTH,
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
    
    # Save the configuration to a JSON file0
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
    return run_config.get("output_dir")

if __name__ == "__main__":
    output_dir = main()
    print(f"Output Directory: {output_dir}")