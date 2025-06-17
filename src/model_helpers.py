from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import os
from dotenv import load_dotenv
from datetime import datetime
from tqdm import tqdm

# Load environment variables from a .env file
load_dotenv()

# Function to load a Hugging Face text generation pipeline with specific settings
def load_pipeline(model_path, batch_size, enable_compile=True):
    """
    Loads model with torch.compile optimization
    """
    # CRITICAL: Load tokenizer from the SAME checkpoint as the model
    # This ensures vocabulary sizes match
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,  # Use checkpoint path, not base model
        token=os.getenv("HUGGINGFACE_TOKEN"),
        trust_remote_code=True,  # Match training settings
    )

    # Match training tokenizer settings EXACTLY
    tokenizer.padding_side = "right"  # ← Same as training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Match training dtype and NO quantization
    compute_dtype = torch.float16  # Same as training
    
    # Load model WITHOUT quantization to match training
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        token=os.getenv("HUGGINGFACE_TOKEN"),
        cache_dir=os.getenv("CACHE_DIR"),
        torch_dtype=compute_dtype,  # FP16, no quantization
        trust_remote_code=True,
    )
    
    # SAFETY CHECK: Ensure vocabularies match
    model_vocab_size = model.get_input_embeddings().num_embeddings
    tokenizer_vocab_size = len(tokenizer)
    
    if tokenizer_vocab_size != model_vocab_size:
        print(f"WARNING: Vocab mismatch detected!")
        print(f"  Model vocab size: {model_vocab_size}")
        print(f"  Tokenizer vocab size: {tokenizer_vocab_size}")
        print(f"  Attempting to resize model embeddings...")
        
        # Try to fix the mismatch
        model.resize_token_embeddings(tokenizer_vocab_size)
        model.config.vocab_size = tokenizer_vocab_size
        
        print(f"  Fixed: Model now has {model.get_input_embeddings().num_embeddings} embeddings")
    
    print(f"✓ Model and tokenizer loaded successfully")
    print(f"  Vocabulary size: {len(tokenizer)}")
    print(f"  Model embeddings: {model.get_input_embeddings().num_embeddings}")
    print(f"  Padding side: {tokenizer.padding_side}")
    
    # 🚀 TORCH COMPILE OPTIMIZATION
    if enable_compile:
        print("🔥 Compiling model for faster inference...")
        model = torch.compile(
            model, 
            mode="reduce-overhead",  # Best for inference
            fullgraph=False,        # More robust compilation
            dynamic=True            # Handle variable sequence lengths
        )
        print("✅ Model compilation complete!")
    
    return model, tokenizer

# Function to generate text completions using a Hugging Face pipeline
def generate_answers(messages, model, tokenizer, n=5):
    """
    Generates answers using the provided model and tokenizer, and returns the top 10 token probabilities for each generated token.

    Args:
        messages (list): A list of strings containing input text messages.
        model: The Hugging Face model.
        tokenizer: The Hugging Face tokenizer.
        n (int): The number of tokens to generate.

    Returns:
        list: A list of dictionaries, each containing the generated text and the top 10 token probabilities for each generated token.
    """
    results = []
    for i, message in enumerate(tqdm(messages, desc="Generating responses")):
        # Debug print for first few messages
        if i < 3:
            print(f"Input message {i}: {message[:100]}...")
            
        # Handle different input formats
        if isinstance(message, list) and len(message) > 0 and isinstance(message[0], dict):
            # Chat format - extract content or use chat template
            # For probability analysis, use just the content without chat template
            # This will give us more natural probability distributions
            text_input = message[0].get('content', '') + " Answer:"
        else:
            text_input = message
            
        inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        if i < 3:
            print(f"Input shape: {inputs['input_ids'].shape}")
            print(f"Input tokens: {inputs['input_ids'][0][:10]}...")  # First 10 tokens

        # Generate with output_scores=True and return_dict_in_generate=True
        outputs = model.generate(
            **inputs,
            max_new_tokens=n,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,  # Use greedy decoding but still get probabilities
            # temperature=1.0,  # Use default temperature
            # top_p=1.0  # No nucleus sampling
        )

        if i < 3:
            print(f"Generated sequence shape: {outputs.sequences.shape}")
            print(f"Number of scores: {len(outputs.scores) if hasattr(outputs, 'scores') and outputs.scores else 0}")
            print(f"Generated sequence: {outputs.sequences[0]}")
        
        generated_ids = outputs.sequences
        # Only decode the newly generated tokens (excluding the input)
        input_length = inputs['input_ids'].shape[1]
        new_tokens = generated_ids[0][input_length:]
        
        if i < 3:
            print(f"Input length: {input_length}")
            print(f"New tokens: {new_tokens}")
            
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Extract top 10 token probabilities for each generated token
        top_probs = []
        if hasattr(outputs, 'scores') and outputs.scores:
            for step_idx, step_scores in enumerate(outputs.scores):
                if i < 3 and step_idx < 2:  # Debug first few steps
                    print(f"Raw logits shape: {step_scores[0].shape}")
                    print(f"Raw logits sample: {step_scores[0][:5]}")  # First 5 logits
                    print(f"Logits min/max: {step_scores[0].min():.4f}/{step_scores[0].max():.4f}")
                
                # Filter out -inf values to get meaningful probabilities
                logits = step_scores[0].clone()
                
                # Replace -inf with a very negative number to avoid numerical issues
                finite_mask = torch.isfinite(logits)
                if finite_mask.sum() > 1:  # Only if we have multiple finite values
                    # Set -inf values to minimum finite value minus a large number
                    min_finite = logits[finite_mask].min()
                    logits[~finite_mask] = min_finite - 100.0
                
                probs = torch.softmax(logits, dim=-1)
                
                if i < 3 and step_idx < 2:  # Debug first few steps
                    print(f"Probs shape: {probs.shape}")
                    print(f"Probs sum: {probs.sum():.6f}")
                    print(f"Probs min/max: {probs.min():.6f}/{probs.max():.6f}")
                    print(f"Number of finite logits: {finite_mask.sum()}")
                
                top_probs_step = torch.topk(probs, 10)
                
                if i < 3 and step_idx < 2:  # Debug first few steps
                    print(f"Top probs values: {top_probs_step.values[:3]}")  # First 3 values
                    print(f"Top probs indices: {top_probs_step.indices[:3]}")  # First 3 indices
                
                top_probs.append({
                    "tokens": [tokenizer.decode([idx]) for idx in top_probs_step.indices.tolist()],
                    "probabilities": [float(p) for p in top_probs_step.values.tolist()]  # Explicit float conversion
                })

        results.append({
            "generated_text": generated_text,
            "top_probs": top_probs
        })

    return results

def save_results(results_df, dataset_name, checkpoint_folder):
    """
    Saves the results DataFrame to a JSON file with a timestamp.

    Args:
        results_df (pd.DataFrame): The DataFrame containing the results.
        dataset_name (str): The name of the dataset.
        checkpoint_folder (str): Directory to save the results.

    Returns:
        None
    """
    now = datetime.now()
    directory = os.path.join(checkpoint_folder, "results", dataset_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, f"{now.strftime('%Y-%m-%d-%H-%M-%S')}_lama3.json")
    results_df.to_json(file_path)
    print(f"Results saved to {file_path}")

class DynamicBatcher:
    def __init__(self, max_batch_size=8, max_sequence_length=1024):
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
    
    def create_batches(self, messages, tokenizer):
        """
        Creates optimally sized batches based on sequence lengths
        """
        # Tokenize all messages first to get lengths
        tokenized_data = []
        for i, message in enumerate(messages):
            if isinstance(message, list) and len(message) > 0 and isinstance(message[0], dict):
                text_input = message[0].get('content', '') + " Answer:"
            else:
                text_input = message
            
            tokens = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=self.max_sequence_length)
            seq_length = tokens['input_ids'].shape[1]
            
            tokenized_data.append({
                'index': i,
                'text': text_input,
                'tokens': tokens,
                'length': seq_length,
                'message': message
            })
        
        # Sort by length for better batching efficiency
        tokenized_data.sort(key=lambda x: x['length'])
        
        # Create batches
        batches = []
        current_batch = []
        
        for item in tokenized_data:
            # Check if adding this item would exceed limits
            if len(current_batch) >= self.max_batch_size:
                # Batch is full, start new one
                batches.append(current_batch)
                current_batch = [item]
            elif current_batch and self._would_exceed_memory(current_batch + [item]):
                # Would use too much memory, start new batch
                batches.append(current_batch)
                current_batch = [item]
            else:
                # Add to current batch
                current_batch.append(item)
        
        # Add final batch
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _would_exceed_memory(self, batch_items):
        """
        Estimate if batch would use too much memory
        """
        max_length = max(item['length'] for item in batch_items)
        total_tokens = len(batch_items) * max_length
        
        # Rough heuristic: avoid batches with >8K total tokens
        return total_tokens > 8192

def prepare_batch_inputs(batch_items, tokenizer, device):
    """
    Prepare a batch for model input with optimal padding
    """
    texts = [item['text'] for item in batch_items]
    
    # Tokenize with padding to the longest sequence in THIS batch
    # (not global max - this is the key optimization!)
    batch_encoding = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,  # Pad to longest in batch
        truncation=True,
        max_length=1024  # Still respect global limits
    )
    
    # Move to device
    batch_inputs = {k: v.to(device) for k, v in batch_encoding.items()}
    
    return batch_inputs, texts