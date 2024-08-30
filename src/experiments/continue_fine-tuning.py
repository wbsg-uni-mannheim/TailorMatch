
import json
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
from dotenv import load_dotenv
from helper import calculate_scores, get_epoch_from_checkpoint
from utils import clean_response
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import random
import numpy as np
from datetime import datetime
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from datasets import load_dataset
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


# Load OPENAI_API_KEY from .env file
load_dotenv()

BASE_PATH = "../../results/meta-llama/Meta-Llama-3.1-8B-Instruct/error/embeddings/2024-08-12-14-59-11_explanation"
BATCH_SIZE = 32


# set seeds
seed_value = 42

random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # For multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def list_checkpoint_folders(directory):
    checkpoint_folders = []

    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            if 'checkpoint' in folder:
                checkpoint_folders.append(os.path.join(root, folder))

    return checkpoint_folders


def clean_response(response):
    if "yes" in response.lower():
        return 1
    elif "no" in response.lower():
        return 0
    else:
        return -1

# Function to insert product descriptions into the prompt


def insert_product_descriptions(prompt_template: str, product1: str, product2: str):
    # Replace placeholder texts with actual product descriptions
    prompt = prompt_template.replace(
        "'Entity 1'", product1).replace("'Entity 2'", product2)
    messages = [
        {"role": "user", "content": prompt},
    ]
    return messages

# Function to extract the checkpoint number


def get_checkpoint_number(path):
    return int(path.split('-')[-1])


def training(checkpoint_path, training_file_path, new_model_name):
    # get the current date and time
    now = datetime.now()

    # create new folder to store the model data
    new_folder = f"{BASE_PATH}/{new_model_name}"
    os.makedirs(new_folder, exist_ok=True)

    # Fine-tuned model
    new_model = f"{now.strftime('%Y-%m-%d-%H-%M-%S')}_{new_model_name}"

    # New dataset
    dataset = load_dataset('csv', data_files=training_file_path, split="train")

    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        quantization_config=quant_config,
        device_map="auto",
        token=os.getenv("HUGGINGFACE_TOKEN"),
        cache_dir=os.getenv("CHACHE_DIR"),
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path, trust_remote_code=True, token=os.getenv("HUGGINGFACE_TOKEN"))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load LoRA configuration
    peft_args = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # log the training data path to the logs section only
    print(f"Training Data Path: {training_file_path}")
    # Set supervised fine-tuning parameters
    sft_config = SFTConfig(
        max_seq_length=240,
        packing=True,
        output_dir=f"./{new_folder}",
        num_train_epochs=5,
        per_device_train_batch_size=20,  # used 20 for 7b and 13b, 4 for 70b
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=1e-4,  # Start with a lower learning rate for stability
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=1,
        max_steps=-1,
        warmup_ratio=0.03,  # Keep a warmup phase for better stability
        group_by_length=True,
        lr_scheduler_type="cosine",  # Use cosine decay
        run_name=new_model
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        peft_config=peft_args,
        tokenizer=tokenizer,
    )

    # Train model
    trainer.train()

    # After training and saving the model locally
    trainer.model.save_pretrained(f"{new_folder}/{new_model}")
    tokenizer.save_pretrained(f"{new_folder}/{new_model}_tokenizer")

# Function to generate answers using the pipeline


def generate_answers(messages, hf_pipeline):
    # Using KeyDataset for efficient batch processing
    dataset = Dataset.from_dict({"text": messages})
    results = []
    for out in hf_pipeline(KeyDataset(dataset, "text"), max_new_tokens=5):
        # Adjust based on your model output
        results.append(out[0]['generated_text'])
    return results


def validation(checkpoint_path, validation_file_path, validation_prompts_path):
    checkpoint_paths = list_checkpoint_folders(checkpoint_path)
    # Sorting the list by the checkpoint number
    checkpoint_paths = sorted(checkpoint_paths, key=get_checkpoint_number)

    # Loop through each checkpoint path
    for checkpoint_path in checkpoint_paths:
        # check if validation_results.json already exists
        if os.path.exists(f"{checkpoint_path}/validation_results.json"):
            print(f"Validation results already exist for {checkpoint_path}")
            continue

        print(f"Processing checkpoint {checkpoint_path}")
        torch.cuda.empty_cache()
        # Initialize the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            token=os.getenv("HUGGINGFACE_TOKEN")
        )

        tokenizer.padding_side = "left"

        compute_dtype = getattr(torch, "float16")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map="auto",
            quantization_config=quant_config,
            token=os.getenv("HUGGINGFACE_TOKEN"),
            cache_dir=os.getenv("CHACHE_DIR"),
        )

        # Set up the text generation pipeline without specifying device_map
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            batch_size=BATCH_SIZE  # Adjust batch size as needed
        )

        datasets = [{"dataset_name": "wdc",
                     "dataset_path": validation_file_path}]

        for dataset in datasets:
            # Load the dataset
            df = pd.read_pickle(dataset["dataset_path"])

            # Load all prompts we want to test
            with open(validation_prompts_path, 'r') as file:
                prompts = json.load(file)

            result_rows = []

            for task in prompts:
                title = task['title']
                prompt_template = task['prompt']

                messages = [insert_product_descriptions(
                    prompt_template, row['title_left'], row['title_right']) for _, row in df.iterrows()]

                try:
                    responses = generate_answers(messages, hf_pipeline)
                except Exception as e:
                    print(f"Error: {e}")
                    # Fill with empty responses in case of error
                    responses = [""] * len(df)

                for idx, (index, row) in enumerate(df.iterrows()):
                    response = responses[idx] if idx < len(responses) else ""
                    response = response[1].get("content")
                    result_row = {
                        'task': title,
                        'chatbot_question': messages[idx],
                        'chatbot_response_raw': response,
                        'chatbot_response_clean': clean_response(response)
                    }

                    for col in df.columns:
                        result_row[col] = row[col]

                    result_rows.append(result_row)

                print(f"Processed {len(df)} queries")

            all_columns = ['task', 'chatbot_question', 'chatbot_response_raw',
                           'chatbot_response_clean'] + list(df.columns)

            # Convert the list of dictionaries to a DataFrame
            results_df = pd.DataFrame(result_rows, columns=all_columns)
            results_df.loc[results_df['chatbot_response_clean']
                           == -1, 'chatbot_response_clean'] = 0
            f1, precision, recall = calculate_scores(results_df)
            print(f"F1: {f1}, Precision: {precision}, Recall: {recall}")

            # Save the dataframe as a json file
            results_df.to_json(f"{checkpoint_path}/validation_results.json")
            print("File Saved")
        del model
        del tokenizer
        del hf_pipeline


def analyse_validation(checkpoint_path, new_model_name):
    result_file = f"{checkpoint_path}/validation_results.csv"
    checkpoint_paths = list_checkpoint_folders(checkpoint_path)
    # Sorting the list by the checkpoint number
    checkpoint_paths = sorted(checkpoint_paths, key=get_checkpoint_number)

    results = []
    # get all validation files
    for checkpoint_path in checkpoint_paths:
        if not os.path.exists(f"{checkpoint_path}/validation_results.json"):
            print(f"Validation results already exist for {checkpoint_path}")
            continue
        df = pd.read_json(f"{checkpoint_path}/validation_results.json")
        df.loc[df['chatbot_response_clean']
               == -1, 'chatbot_response_clean'] = 0
        f1, precision, recall = calculate_scores(df)
        epoch = get_epoch_from_checkpoint(checkpoint_paths, int(
            checkpoint_path.split("/")[-1].replace("checkpoint-", "")))
        result = {
            "checkpoint_path": checkpoint_path,
            "checkpoint_number": checkpoint_path.split("/")[-1].replace("checkpoint-", ""),
            "epoch": epoch,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
        results.append(result)
    df = pd.DataFrame(results)

    # sort by highest f1
    df_sorted = df.sort_values(by='f1', ascending=False)
    # df_sorted.to_csv("{checkpoint_path}/validation_results.csv", index=False)

    # get the checkpoint path for the best f1
    best_checkpoint_path = df_sorted.iloc[0]['checkpoint_path']
    print(f"Best Checkpoint Path: {best_checkpoint_path}")
    print(f"Best F1: {df_sorted.iloc[0]['f1']}")
    df_sorted.to_csv(result_file, index=False)
    return best_checkpoint_path


# Optimized Cosine Similarity with Matrix Operations
def find_most_similar_examples(test_embedding, train_df, top_n=6):
    # Convert lists of embeddings to a numpy array if not already
    train_embeddings = np.array(list(train_df['embedding'].values))
    test_embedding = np.array(test_embedding).reshape(1, -1)

    # Calculate cosine similarities for all train embeddings at once
    similarities = cosine_similarity(test_embedding, train_embeddings)

    # Get indices of top_n highest similarities
    most_similar_indices = np.argsort(similarities[0])[::-1][:top_n]
    most_similar_examples = train_df.iloc[most_similar_indices].to_dict(
        orient='records')

    return most_similar_examples


def add_training_data(training_file_path, best_checkpoint_path, additional_training_file_path, new_model_name):
    training_data = pd.read_csv(training_file_path)
    additional_training_data = pd.read_pickle(
        additional_training_file_path, compression='gzip')
    validation_df = pd.read_pickle(VALIDATION_FILE_PATH)

    # Load the errors from the best checkpoint
    errors = pd.read_json(f"{best_checkpoint_path}/validation_results.json")
    print(f"{best_checkpoint_path}/validation_results.json")
    # Filter out the errors
    errors = errors[errors['chatbot_response_clean'] != errors['label']]
    errors.to_csv("errors.csv", index=False)

    # join the validationdf with the errors on pair_id
    # errors = errors.merge(validation_df, on='pair_id')

    # Calculate the number of additional training examples needed
    additional_training_examples = len(training_data) - len(errors)
    examples_per_error = int(additional_training_examples / len(errors))

    # Get the embeddings for the additional training data
    new_examples = []
    for _, row in errors.iterrows():
        new_example = find_most_similar_examples(
            row['embedding'],
            additional_training_data,
            examples_per_error
        )
        for example in new_example:
            label = example["label"]
            new_examples.append([insert_product_descriptions(
                "Do the two product descriptions refer to the same real-world product? Entity 1: 'Entity 1'. Entity 2: 'Entity 2'.", example['title_left'], example['title_right']), label])
    new_training_examples = pd.DataFrame(new_examples)
    new_training_examples.columns = ['prompt', 'completion']
    if len(new_training_examples) < len(training_data):
        difference = len(training_data) - len(new_training_examples)
        add_training_data = additional_training_data.sample(difference)
        more_examples = []
        for _, row in add_training_data.iterrows():
            more_examples.append([insert_product_descriptions(
                "Do the two product descriptions refer to the same real-world product? Entity 1: 'Entity 1'. Entity 2: 'Entity 2'.", row['title_left'], row['title_right']), row['label']])
        add_training_data = pd.DataFrame(more_examples)
        add_training_data.columns = ['prompt', 'completion']
        new_training_examples = pd.concat(
            [new_training_examples, add_training_data])
    elif len(new_training_examples) > len(training_data):
        new_training_examples = new_training_examples.sample(
            len(training_data))

    print(f"Number of new training examples: {len(new_training_examples)}")

    # drop the pair_id column
    # new_training_examples = new_training_examples.drop(columns=['pair_id'])

    # Combine the new training examples with the existing training data
    training_data = pd.concat([training_data, new_training_examples])

    # Save the new training data
    training_data.to_csv(
        f"{BASE_PATH}/{new_model_name}_enhanced_training.csv", index=False)
    print(
        f"Saved new training data to {BASE_PATH}/{new_model_name}_enhanced_training.csv")


"""
def add_training_data(training_file_path, best_checkpoint_path, additional_training_file_path, new_model_name):
    training_data = pd.read_csv(training_file_path)
    additional_training_data = pd.read_pickle(additional_training_file_path)
    validation_df = pd.read_pickle(VALIDATION_FILE_PATH)
    
    # Load the errors from the best checkpoint
    errors = pd.read_json(f"{best_checkpoint_path}/validation_results.json")
    print(f"{best_checkpoint_path}/validation_results.json")
    # Filter out the errors
    errors = errors[errors['chatbot_response_clean'] != errors['label']]
    
    # join the validationdf with the errors on pair_id
    errors = errors.merge(validation_df, on='pair_id')
    
    # filter additional training data to only include the pair_ids that are in the errors
    new_training_examples = additional_training_data[additional_training_data['pair_id'].isin(errors['pair_id'])]
    
    if len(new_training_examples) < len(training_data):
        difference = len(training_data) - len(new_training_examples)
        add_training_data = additional_training_data.sample(difference)
        new_training_examples = pd.concat([new_training_examples, add_training_data])
    elif len(new_training_examples) > len(training_data):
        new_training_examples = new_training_examples.sample(len(training_data))
        
    print(f"Number of new training examples: {len(new_training_examples)}")
    
        
    # drop the pair_id column
    new_training_examples = new_training_examples.drop(columns=['pair_id'])
        
    # Combine the new training examples with the existing training data
    training_data = pd.concat([training_data, new_training_examples])
    
    # Save the new training data
    training_data.to_csv(f"{BASE_PATH}/{new_model_name}_enhanced_training.csv", index=False)
    print(f"Saved new training data to {BASE_PATH}/{new_model_name}_enhanced_training.csv")
"""

print("Starting the training and validation process")
ORIGINAL_TRAINING_PATH = "../../data/wdc/filtered/small/wdc_train_small_filtered.csv"

first_checkpoint_path = "../../results/meta-llama/Meta-Llama-3.1-8B-Instruct/error/embeddings/2024-08-12-14-59-11_explanation/Meta-Llama-3.1-8B-Instruct-error-small_enhanced-1/checkpoint-44"
VALIDATION_PROMPT_PATH = "../../prompts/test_prompt.json"
VALIDATION_FILE_PATH = "../../data/wdc/preprocessed_wdcproducts80cc20rnd000un_valid_small_embeddings.pkl"
validation_df = pd.read_pickle(VALIDATION_FILE_PATH)
ADDITIONAL_TRAINING_FILE_PATH = "../../data/wdc/filtered/large/filtered_large_embeddings.pkl.gz"

add_training_data(
    training_file_path=ORIGINAL_TRAINING_PATH,
    best_checkpoint_path=first_checkpoint_path,
    additional_training_file_path=ADDITIONAL_TRAINING_FILE_PATH,
    new_model_name="Meta-Llama-3.1-8B-Instruct-error-small_enhanced-1"
)

for i in range(1, 6):
    new_model_name = f"Meta-Llama-3.1-8B-Instruct-error-small_enhanced-{i}"

    best_checkpoint_path = first_checkpoint_path
    if i != 1:
        checkpoint_paths = f"{BASE_PATH}/Meta-Llama-3.1-8B-Instruct-error-small_enhanced-{i-1}"
        best_checkpoint_path = analyse_validation(
            checkpoint_paths, new_model_name)

    training_file_path = f"{BASE_PATH}/{new_model_name}_enhanced_training.csv"

    training(best_checkpoint_path, training_file_path, new_model_name)
    validation(checkpoint_path=f"{BASE_PATH}/{new_model_name}",
               validation_file_path=VALIDATION_FILE_PATH,
               validation_prompts_path=VALIDATION_PROMPT_PATH
               )
    best_checkpoint_path = analyse_validation(checkpoint_path=f"{BASE_PATH}/{new_model_name}",
                                              new_model_name=new_model_name)
    add_training_data(ORIGINAL_TRAINING_PATH, best_checkpoint_path,
                      ADDITIONAL_TRAINING_FILE_PATH, f"Meta-Llama-3.1-8B-Instruct-error-small_enhanced-{i+1}")

    print(f"Finished training and validation for {new_model_name}")
    print(f"Added new training data for {new_model_name}")
    print(f"Continuing to the next iteration")
