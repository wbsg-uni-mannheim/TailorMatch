import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from transformers.pipelines.pt_utils import KeyDataset
import pandas as pd
import json
import os
from datetime import datetime
import pandas as pd
import helper as analytics
from datasets import Dataset
import time
import helper
from dotenv import load_dotenv

# Set CUDA_VISIBLE_DEVICES to limit GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,5,6,7"
# Load OPENAI_API_KEY from .env file
load_dotenv()


CHECKPOINT_FOLDER = "../../results/meta-llama/Meta-Llama-3.1-70B-Instruct/wdc-small/2024-08-26-12-36-12"
best_checkpoint_path = CHECKPOINT_FOLDER

TEST_PROMPTS = "../../prompts/domain_promts.json"

batch_size = 64


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


# Function to generate answers using the pipeline
def generate_answers(messages, hf_pipeline):
    # Using KeyDataset for efficient batch processing
    dataset = Dataset.from_dict({"text": messages})
    results = []
    for out in hf_pipeline(KeyDataset(dataset, "text"), max_new_tokens=5):
        # Adjust based on your model output
        results.append(out[0]['generated_text'])
    return results


# Load the validation results
df = pd.read_csv(f"{CHECKPOINT_FOLDER}/validation_results.csv")

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

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    best_checkpoint_path,
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
