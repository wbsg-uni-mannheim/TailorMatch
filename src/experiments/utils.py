import os
import pandas as pd
from helper import get_all_files_in_directory, calculate_stats

def clean_response(response):
    """
    Cleans the chatbot response by converting it to a binary value.

    Args:
        response (str): The raw response from the chatbot.

    Returns:
        int: 1 if the response contains "yes", 0 if it contains "no", and -1 otherwise.
    """
    if "yes" in response.lower():
        return 1
    elif "no" in response.lower():
        return 0
    else:
        return -1

# Function to insert product descriptions into the prompt
def insert_product_descriptions(prompt_template: str, product1: str, product2: str):
    """
    Inserts product descriptions into the provided prompt template.

    Args:
        prompt_template (str): The template string containing placeholders.
        product1 (str): The description of the first product.
        product2 (str): The description of the second product.

    Returns:
        str: The prompt with product descriptions inserted.
    """
    # Replace placeholder texts with actual product descriptions
    prompt = prompt_template.replace("'Entity 1'", product1).replace("'Entity 2'", product2)
    return prompt

def generate_question(prompt, entity1, entity2):
    """
    Generates a question by inserting product descriptions into the prompt template.

    Args:
        prompt (str): The prompt template.
        entity1 (str): The description of the first entity.
        entity2 (str): The description of the second entity.

    Returns:
        list: A list containing a dictionary with the role and content of the generated question.
    """
    prompt = insert_product_descriptions(prompt, entity1, entity2)
    return [
        {"role": "user", "content": prompt},
    ]

def insert_product_descriptions_array(prompt_template: str, product1: str, product2: str):
    """
    Inserts product descriptions into the prompt template and returns a formatted message.

    Args:
        prompt_template (str): The template string containing placeholders.
        product1 (str): The description of the first product.
        product2 (str): The description of the second product.

    Returns:
        list: A list containing a dictionary with the role and content of the generated message.
    """
    # Replace placeholder texts with actual product descriptions
    prompt = prompt_template.replace("'Entity 1'", product1).replace("'Entity 2'", product2)
    messages = [
        {"role": "user", "content": prompt},
    ]
    return messages

def get_checkpoint_number(path):
    """
    Extracts the checkpoint number from a given file path.

    Args:
        path (str): The file path containing the checkpoint number.

    Returns:
        int: The checkpoint number extracted from the file path.
    """
    return int(path.split('-')[-1])

def calculate_results(result_dir):
    """
    Processes all experiment files in the given directory, calculates statistics for each dataset,
    and saves the combined results as a CSV file.

    Args:
        result_dir (str): The directory containing the experiment result files.

    Returns:
        pd.DataFrame: The DataFrame containing the combined statistics for all datasets.
    """
    # Get all file paths in the result directory
    experiment_paths = get_all_files_in_directory(result_dir)

    # Initialize a list to store stats DataFrames
    stats_dataframes = []

    # Process each experiment file
    for experiment_path in experiment_paths:
        # Extract the dataset name from the file path
        dataset_name = os.path.basename(os.path.dirname(experiment_path))
        print(f"Processing {dataset_name}")

        # Load the dataset from the JSON file
        df = pd.read_json(experiment_path)

        # Calculate statistics for the dataset
        stats_df = calculate_stats(df)

        # Add a column with the dataset name for reference
        stats_df['Dataset'] = dataset_name

        # Append the stats DataFrame to the list
        stats_dataframes.append(stats_df)

    # Concatenate all stats DataFrames into a single DataFrame
    result_df = pd.concat(stats_dataframes)

    # Save the combined results to a CSV file
    result_df.to_csv(os.path.join(result_dir, "results.csv"), index=False)

    return result_df