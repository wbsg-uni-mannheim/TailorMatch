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

# Load OPENAI_API_KEY from .env file
load_dotenv()


CHECKPOINT_FOLDER = "../../results/meta-llama/Meta-Llama-3.1-8B-Instruct/small/explanations/2024-08-01-19-25-40"

TEST_PROMPTS = "../../prompts/explanation_structured.json"


# Function to insert product descriptions into the prompt
def insert_product_descriptions(prompt_template: str, product1: str, product2: str, label: str):
    # Replace placeholder texts with actual product descriptions
    prompt = prompt_template.replace(
        "{{product_1}}", product1).replace("{{product_2}}", product2)
    label = "Yes" if label == 1 else "No"
    prompt = prompt.replace("{{label}}", label)
    messages = [
        {"role": "user", "content": prompt},
    ]
    return messages


# Function to generate answers using the pipeline
def generate_answers(messages, hf_pipeline):
    # Using KeyDataset for efficient batch processing
    dataset = Dataset.from_dict({"text": messages})
    results = []
    for out in hf_pipeline(KeyDataset(dataset, "text"), max_new_tokens=300):
        # Adjust based on your model output
        results.append(out[0]['generated_text'])
    return results


# Load the validation dataset
df = pd.read_csv(f"{CHECKPOINT_FOLDER}/validation_results.csv")

# sort by highest f1
df_sorted = df.sort_values(by='f1', ascending=False)


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
    device_map="auto",
    offload_folder="offload",  # Ensure offloading happens to a specific folder if needed
    token=os.getenv("HUGGINGFACE_TOKEN"),
    cache_dir=os.getenv("CHACHE_DIR"),
)

# Set up the text generation pipeline without specifying device_map
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    batch_size=16,  # Adjust batch size as needed
)

datasets = [{"dataset_name": "wdc",
             "dataset_path": "../../data/wdc/preprocessed_wdcproducts80cc20rnd000un_train_small.pkl.gz"}]

for dataset in datasets:
    # Load the dataset
    df = pd.read_pickle(dataset["dataset_path"], compression='gzip')

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
            prompt_template, row['title_left'], row['title_right'], row["label"]) for _, row in df.iterrows()]

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
                'chatbot_response_raw': response
            }

            for col in df.columns:
                result_row[col] = row[col]

            result_rows.append(result_row)

        print(f"Processed {len(df)} queries")

        all_columns = ['chatbot_response_raw'] + list(df.columns)

        # Convert the list of dictionaries to a DataFrame
        results_df = pd.DataFrame(result_rows, columns=all_columns)

    all_columns = ['chatbot_response_raw'] + list(df.columns)

    # Convert the list of dictionaries to a DataFrame
    results_df = pd.DataFrame(result_rows, columns=all_columns)

    # get the current date and time
    now = datetime.now()

    path = f"{CHECKPOINT_FOLDER}/explanations/{dataset['dataset_name']}"
    directory = path
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save the dataframe as a json file
    results_df.to_csv(
        f"{path}/{now.strftime('%Y-%m-%d-%H-%M-%S')}_{dataset['dataset_name']}.csv")
    print(f"{path}/{now.strftime('%Y-%m-%d-%H-%M-%S')}_lama3.json")
