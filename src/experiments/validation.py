from dotenv import load_dotenv
import helper
import time
from datasets import Dataset
import helper as analytics
from datetime import datetime
import json
import pandas as pd
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import transformers
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,4,5"

# Load OPENAI_API_KEY from .env file
load_dotenv()


CHECKPOINT_FOLDER = "../../results/meta-llama/Meta-Llama-3.1-8B-Instruct/wdc-small-synthetic-filtered-interesting-with-explanations/2024-08-28-08-11-22"
VALIDATION_PROMPT_PATH = "../../prompts/test_prompt.json"
VALIDATION_FILE_PATH = "../../data/wdc/preprocessed_wdcproducts80cc20rnd000un_valid_small.pkl"

TEST_PROMPTS = "../../prompts/domain_promts.json"
WANDDB_ID = "e4gkovo4"
batch_size = 32
device_map = "auto"


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


checkpoint_paths = list_checkpoint_folders(CHECKPOINT_FOLDER)

# Sorting the list by the checkpoint number
checkpoint_paths = sorted(checkpoint_paths, key=get_checkpoint_number)


# Function to generate answers using the pipeline
def generate_answers(messages, hf_pipeline):
    # Using KeyDataset for efficient batch processing
    dataset = Dataset.from_dict({"text": messages})
    results = []
    for out in hf_pipeline(KeyDataset(dataset, "text"), max_new_tokens=5):
        # Adjust based on your model output
        results.append(out[0]['generated_text'])
    return results


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
        device_map=device_map,
        quantization_config=quant_config,
        token=os.getenv("HUGGINGFACE_TOKEN"),
        cache_dir=os.getenv("CHACHE_DIR"),
    )

    # Set up the text generation pipeline without specifying device_map
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size  # Adjust batch size as needed
    )

    datasets = [{"dataset_name": "wdc", "dataset_path": VALIDATION_FILE_PATH}]

    for dataset in datasets:
        # Load the dataset
        df = pd.read_pickle(dataset["dataset_path"])

        # Load all prompts we want to test
        with open(VALIDATION_PROMPT_PATH, 'r') as file:
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
        f1, precision, recall = analytics.calculate_scores(results_df)
        print(f"F1: {f1}, Precision: {precision}, Recall: {recall}")
        step = int(checkpoint_path.split("/")[-1].replace("checkpoint-", ""))
        epoch = helper.get_epoch_from_checkpoint(checkpoint_paths, step)
        helper.log_metrics_to_existing_wandb_run(
            "First Paper", WANDDB_ID, step, epoch, f1, precision, recall)
        # Save the dataframe as a json file
        results_df.to_json(f"{checkpoint_path}/validation_results.json")
        print("File Saved")
    del model
    del tokenizer
    del hf_pipeline


results = []
# get all validation files
for checkpoint_path in checkpoint_paths:
    if not os.path.exists(f"{checkpoint_path}/validation_results.json"):
        print(f"Validation results already exist for {checkpoint_path}")
        continue
    df = pd.read_json(f"{checkpoint_path}/validation_results.json")
    df.loc[df['chatbot_response_clean'] == -1, 'chatbot_response_clean'] = 0
    f1, precision, recall = analytics.calculate_scores(df)
    epoch = helper.get_epoch_from_checkpoint(checkpoint_paths, int(
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
df_sorted
df.to_csv(f"{CHECKPOINT_FOLDER}/validation_results.csv")


# get the checkpoint path for the best f1
best_checkpoint_path = df_sorted.iloc[0]['checkpoint_path']

# Initialize the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    best_checkpoint_path,
    token=os.getenv("HUGGINGFACE_TOKEN")
)

tokenizer.padding_side = "left"
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    best_checkpoint_path,
    device_map=device_map,
    offload_folder="offload",  # Ensure offloading happens to a specific folder if needed
    token=os.getenv("HUGGINGFACE_TOKEN"),
    cache_dir=os.getenv("CHACHE_DIR"),
)

# Set up the text generation pipeline without specifying device_map
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    batch_size=batch_size  # Adjust batch size as needed
)


datasets = [
    {"dataset_name": "wdc-fullsize",
        "dataset_path": "../../data/wdc/wdcproducts80cc20rnd050un_test_gs.pkl"},
    {"dataset_name": "abt-buy-full", "dataset_path": "../../data/abt-buy/abt-buy-gs.pkl"}, {
        "dataset_name": "amazon-google-full", "dataset_path": "../../data/amazon-google/amazon-google-gs.pkl"},
    {"dataset_name": "dblp-acm", "dataset_path": "../../data/dblp-acm/dblp-acm-gs.pkl"},
    {"dataset_name": "dblp-scholar",
        "dataset_path": "../../data/dblp-scholar/dblp-scholar-gs.pkl"},
    {"dataset_name": "walmart-amazon",
        "dataset_path": "../../data/walmart-amazon/walmart-amazon-gs.pkl"}
]

for dataset in datasets:
    print(f"Processing dataset: {dataset['dataset_name']}")
    # Load the dataset
    df = pd.read_pickle(dataset["dataset_path"])

    # Load all prompts we want to test
    with open(TEST_PROMPTS, 'r') as file:
        prompts = json.load(file)

    result_rows = []

    for task in prompts:
        title = task['title']
        prompt_template = task['prompt']
        print(
            f"Processing dataset: {dataset['dataset_name']} \n Processing task:  {title}")

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
        print(analytics.calculate_stats(results_df))

    all_columns = ['task', 'chatbot_question', 'chatbot_response_raw',
                   'chatbot_response_clean'] + list(df.columns)

    # Convert the list of dictionaries to a DataFrame
    results_df = pd.DataFrame(result_rows, columns=all_columns)

    # get the current date and time
    now = datetime.now()

    directory = f"{CHECKPOINT_FOLDER}/results/{dataset['dataset_name']}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    print(analytics.calculate_stats(results_df))
    # save the dataframe as a json file
    results_df.to_json(
        f"{CHECKPOINT_FOLDER}/results/{dataset['dataset_name']}/{now.strftime('%Y-%m-%d-%H-%M-%S')}_lama3.json")
    print(
        f"{CHECKPOINT_FOLDER}/{dataset['dataset_name']}/{now.strftime('%Y-%m-%d-%H-%M-%S')}_lama3.json")
