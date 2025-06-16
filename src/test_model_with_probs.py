import pandas as pd
import json
import os
import time
import argparse
from utils import clean_response, insert_product_descriptions_array, calculate_results
from model_helpers import generate_answers, load_pipeline, save_results
from dotenv import load_dotenv

# Load OPENAI_API_KEY from .env file
load_dotenv()

def process_datasets(datasets, model, tokenizer, test_prompts_path, checkpoint_folder, n_tokens):
    """
    Processes a list of datasets with specified prompts and saves the results, including top 10 token probabilities.

    Args:
        datasets (list): A list of dictionaries, each containing 'dataset_name' and 'dataset_path'.
        model: The Hugging Face model.
        tokenizer: The Hugging Face tokenizer.
        test_prompts_path (str): Path to the JSON file containing the test prompts.
        checkpoint_folder (str): Directory to save the results.
        n_tokens (int): The number of tokens to generate.

    Returns:
        None
    """
    with open(test_prompts_path, 'r') as file:
        prompts = json.load(file)

    for dataset in datasets:
        print(f"Processing dataset: {dataset['dataset_name']}")
        df = pd.read_pickle(dataset["dataset_path"])
        
        result_rows = []

        for task in prompts:
            start_time = time.time()
            title = task['title']
            prompt_template = task['prompt']
            print(f"Processing task: {title}")

            if "dblp" in dataset['dataset_name']:
                messages = [
                    insert_product_descriptions_array(
                        prompt_template=prompt_template,
                        product1=f"{row['title_left']}; {row['authors_left']}; {row['venue_left']}; {row['year_left']}",
                        product2=f"{row['title_right']}; {row['authors_right']}; {row['venue_right']}; {row['year_right']}"
                    )
                    for _, row in df.iterrows()
                ]
            else:
                messages = [
                    insert_product_descriptions_array(
                        prompt_template, row['title_left'], row['title_right']
                    )
                    for _, row in df.iterrows()
                ]

            try:
                responses = generate_answers(messages, model, tokenizer, n=n_tokens)
            except Exception as e:
                print(f"Error: {e}")
                responses = [{"generated_text": "", "top_probs": []}] * len(df)

            for idx, (index, row) in enumerate(df.iterrows()):
                response = responses[idx] if idx < len(responses) else {"generated_text": "", "top_probs": []}
                generated_text = response["generated_text"]
                top_probs = response["top_probs"]
                
                # Debug print
                if idx < 3:  # Print first 3 for debugging
                    print(f"Generated text: {generated_text}")
                    print(f"Top probs length: {len(top_probs)}")
                    for token_idx, token_probs in enumerate(top_probs):
                        print(f"Token {token_idx} probs: {token_probs}")
                        if token_idx >= 2:  # Only show first 3 tokens to avoid too much output
                            print(f"... and {len(top_probs) - 3} more tokens")
                            break
                
                result_row = {
                    'task': title,
                    'chatbot_question': messages[idx],
                    'chatbot_response_raw': generated_text,
                    'chatbot_response_clean': clean_response(generated_text),
                    'top_probs': top_probs
                }

                for col in df.columns:
                    result_row[col] = row[col]

                # Final debug check for first few rows
                if idx < 3:
                    print(f"Saving result with {len(result_row['top_probs'])} token probabilities")
                
                result_rows.append(result_row)

            print(f"Processed {len(df)} queries in {time.time() - start_time:.2f} seconds")
            results_df = pd.DataFrame(result_rows, columns=[
                'task', 'chatbot_question', 'chatbot_response_raw',
                'chatbot_response_clean', 'top_probs'] + list(df.columns))

        save_results(results_df, dataset['dataset_name'], checkpoint_folder)
    calculate_results(f"{checkpoint_folder}/results")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process datasets with token probabilities.')
    parser.add_argument('--n_tokens', type=int, default=5, help='Number of tokens to generate')
    args = parser.parse_args()

    CHECKPOINT_FOLDERS = [
        "/ceph/aasteine/fine-tuning-paper/results/meta-llama/Meta-Llama-3.1-8B-Instruct/regular_train_small_all_fields/lr_0.0002/2025-05-12-12-20-23"
    ]

    TEST_PROMPTS = "/ceph/aasteine/fine-tuning-paper/prompts/test_prompt.json"

    BATCH_SIZE = 32

    datasets = [
        {"dataset_name": "wdc-fullsize",
           "dataset_path": "./../data/wdc/wdcproducts80cc20rnd050un_test_gs.pkl"},
        # {"dataset_name": "abt-buy-full", "dataset_path": "../data/abt-buy/abt-buy-gs.pkl"}, {
        #    "dataset_name": "amazon-google-full", "dataset_path": "../data/amazon-google/amazon-google-gs.pkl"},
        #{"dataset_name": "dblp-acm", "dataset_path": "../data/dblp-acm/dblp-acm-gs.pkl"},
        #{"dataset_name": "dblp-scholar",
        #    "dataset_path": "../data/dblp-scholar/dblp-scholar-gs.pkl"},
        # {"dataset_name": "walmart-amazon",
        #    "dataset_path": "../data/walmart-amazon/walmart-amazon-gs.pkl"}
    ]
    for CHECKPOINT_FOLDER in CHECKPOINT_FOLDERS:
        print(f"Processing checkpoint folder: {CHECKPOINT_FOLDER}")
        # Determine the best checkpoint path
        # Load the validation results
        df = pd.read_csv(f"{CHECKPOINT_FOLDER}/validation_results.csv")
        # sort by highest f1
        df_sorted = df.sort_values(by='f1', ascending=False)
        # get the checkpoint path for the best f1
        best_checkpoint_path = df_sorted.iloc[0]['checkpoint_path']
        model, tokenizer = load_pipeline(best_checkpoint_path, BATCH_SIZE)
        process_datasets(datasets, model, tokenizer, TEST_PROMPTS, CHECKPOINT_FOLDER, args.n_tokens) 