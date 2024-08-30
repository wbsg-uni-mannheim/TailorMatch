import os
from openai import OpenAI

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA

from sklearn.metrics.pairwise import cosine_similarity

# Load OPENAI_API_KEY from .env file
load_dotenv()

client = OpenAI()


def get_embedding(row, model="text-embedding-3-small"):
    product_title_left = row['title_left']
    product_title_right = row['title_right']
    
    text = f"{product_title_left} - {product_title_right}"
    return client.embeddings.create(input = [text], model=model).data[0].embedding


datasets = [
    {"dataset_name": "amazon-google-train", "dataset_path": "../../data/amazon-google/amazon-google-train.json"},
    {"dataset_name": "dblp-acm", "dataset_path": "../../data/dblp-acm/dblp-acm-train.json.gz"},
    {"dataset_name": "dblp-scholar", "dataset_path": "../../data/dblp-scholar/dblp-scholar-train.json.gz"},
    {"dataset_name": "walmart-amazon", "dataset_path": "../../data/walmart-amazon/walmart-amazon-train.json.gz"}
]

for dataset in datasets:
    dataset_name = dataset["dataset_name"]
    dataset_path = dataset["dataset_path"]
    
    print(f"Processing {dataset_name}")
    if ".json.gz" in dataset_path:
        train_df = pd.read_json(dataset_path, lines=True, compression='gzip')
    else:
        train_df = pd.read_json(dataset_path)
    
    # Use tqdm with apply to show a progress bar
    tqdm.pandas(desc="Processing Embeddings")
    train_df['embedding'] = train_df.progress_apply(get_embedding, axis=1)
    
    # Save the dataframe with embeddings
    train_df.to_pickle(f"{dataset_path.replace('.json', '')}_embeddings.pkl")