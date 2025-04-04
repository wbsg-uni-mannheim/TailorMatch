import torch
import random
import numpy as np
import gc
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTConfig

def clear_gpu_memory():
    """Clear GPU memory and cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        # Additional cleanup for all GPUs
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(f'cuda:{i}'):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

def set_seeds(seed_value):
    """Set all seeds for reproducibility"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_model_and_tokenizer(config):
    """Initialize model and tokenizer with proper configuration"""
    # Setup compute dtype
    compute_dtype = getattr(torch, "float16")
    
    # Configure quantization
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        device_map="auto",
        token=os.getenv("HUGGINGFACE_TOKEN"),
        cache_dir=os.getenv("CACHE_DIR"),
        quantization_config=quant_config,
    )
    
    # Configure model settings
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID,
        trust_remote_code=True,
        token=os.getenv("HUGGINGFACE_TOKEN")
    )
    
    # Configure tokenizer settings
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer

def get_lora_config():
    """Get LoRA configuration for fine-tuning"""
    return LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

def get_sft_config(output_dir, learning_rate, run_name, config):
    """Get Supervised Fine-tuning configuration"""
    return SFTConfig(
        max_seq_length=240,
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