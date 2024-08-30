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
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from utils import clean_response
from model_helpers import generate_answers, load_pipeline
from dotenv import load_dotenv

# Set CUDA_VISIBLE_DEVICES to limit GPUs
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,5,6,7"
# Load OPENAI_API_KEY from .env file
load_dotenv()


CHECKPOINT_PATH = "../../results/meta-llama/Meta-Llama-3.1-8B-Instruct/small/explanations/2024-08-01-19-25-40/checkpoint-208"
BASELINE_FOLDER = "../../results/meta-llama/Meta-Llama-3.1-8B-Instruct/small/explanations/2024-08-01-19-25-40/few_shot_in_context"
TEST_PROMPTS = "../../prompts/domain_promts_few_shot.json"

batch_size = 8

def insert_product_descriptions(prompt_template: str, product1: str, product2: str, examples: list):
    # Replace placeholder texts with actual product descriptions and examples
    example_str = "\n".join(
        [f"Example {i+1}:\nEntity 1: '{example['title_left']}'\nEntity 2: '{example['title_right']}'\nAnswer: {'Yes' if example['label'] == 1 else 'No'}" for i, example in enumerate(examples)])
    prompt_with_examples = prompt_template.replace("[EXAMPLES]", example_str)
    prompt = prompt_with_examples.replace(
        "'Entity 1'", product1).replace("'Entity 2'", product2)
    messages = [{"role": "user", "content": prompt}]
    return messages


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


hf_pipeline = load_pipeline(CHECKPOINT_PATH, batch_size)


datasets = [
    {"dataset_name": "wdc-fullsize",
        "dataset_path": "../../data/wdc/wdcproducts80cc20rnd050un_test_gs_embeddings.pkl"},
    {"dataset_name": "abt-buy-full",
        "dataset_path": "../../data/abt-buy/abt-buy-gs_embeddings.pkl"},
    {"dataset_name": "amazon-google-full",
        "dataset_path": "../../data/amazon-google/amazon-google-gs_embeddings.pkl"},
    {"dataset_name": "dblp-acm",
        "dataset_path": "../../data/dblp-acm/dblp-acm-gs_embeddings.pkl"},
    {"dataset_name": "dblp-scholar",
        "dataset_path": "../../data/dblp-scholar/dblp-scholar-gs_embeddings.pkl"},
    {"dataset_name": "walmart-amazon",
        "dataset_path": "../../data/walmart-amazon/walmart-amazon-gs_embeddings.pkl"}
]

train_datasets = [
    {"dataset_name": "wdc-fullsize",
        "dataset_path": "../../data/wdc/preprocessed_wdcproducts80cc20rnd000un_train_small_with_embeddings.pkl.gz"},
    {"dataset_name": "abt-buy-full",
        "dataset_path": "../../data/abt-buy/abt-buy-train_embeddings.pkl"},
    {"dataset_name": "amazon-google-full",
        "dataset_path": "../../data/amazon-google/amazon-google-train_embeddings.pkl"},
    {"dataset_name": "dblp-acm",
        "dataset_path": "../../data/dblp-acm/dblp-acm-train_embeddings.pkl"},
    {"dataset_name": "dblp-scholar",
        "dataset_path": "../../data/dblp-scholar/dblp-scholar-train_embeddings.pkl"},
    {"dataset_name": "walmart-amazon",
        "dataset_path": "../../data/walmart-amazon/walmart-amazon-train.gz_embeddings.pkl"}
]


for index, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset['dataset_name']}")
    # Load the dataset
    df = pd.read_pickle(dataset["dataset_path"])
    if ".gz" in dataset["dataset_path"]:
        df = pd.read_pickle(dataset["dataset_path"], compression='gzip')
    else:
        train_df = pd.read_pickle(train_datasets[index]["dataset_path"])

    # Load all prompts we want to test
    with open(TEST_PROMPTS, 'r') as file:
        prompts = json.load(file)

    result_rows = []

    for task in prompts:
        title = task['title']
        prompt_template = task['prompt']
        print(
            f"Processing dataset: {dataset['dataset_name']} \n Processing task:  {title}")

        messages = []
        for _, row in df.iterrows():
            # Find the 6 most similar examples based on embeddings
            most_similar_examples = find_most_similar_examples(
                row['embedding'], train_df, top_n=6)
            message = insert_product_descriptions(
                prompt_template, row['title_left'], row['title_right'], most_similar_examples)
            messages.append(message)

        print(f"Generated {len(messages)} prompts")
        print(messages[0])

        try:
            responses = generate_answers(messages, hf_pipeline)
        except Exception as e:
            print(f"Error: {e}")
            # Fill with empty responses in case of error
            responses = [""] * len(df)

        for idx, (index, row) in enumerate(df.iterrows()):
            response = responses[idx] if idx < len(responses) else ""

            try:
                response = response[1].get("content")
            except:
                print(f"Response: {response}")
                pass

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

    directory = f"{BASELINE_FOLDER}/{dataset['dataset_name']}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    print(analytics.calculate_stats(results_df))
    # save the dataframe as a json file
    results_df.to_json(
        f"{BASELINE_FOLDER}/{dataset['dataset_name']}/{now.strftime('%Y-%m-%d-%H-%M-%S')}_lama3.json")
