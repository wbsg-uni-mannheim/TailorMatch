import os
from dotenv import load_dotenv
import json
import pandas as pd
from datetime import datetime
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load OPENAI_API_KEY from .env file
load_dotenv()

# Initialize the OpenAI client
client = OpenAI()

def insert_product_descriptions(prompt_template: str, product1: str, product2: str):
    # Replace placeholder texts with actual product descriptions
    prompt = prompt_template.replace("'Entity 1'", product1).replace("'Entity 2'", product2)
    return prompt

def create_training_example(prompt_template: str, product1: str, product2: str, label: str):
    # Create the prompt with product descriptions
    prompt = insert_product_descriptions(prompt_template, product1, product2)
    if label == 1:
        label = "Yes"
    else:
        label = "No"
    
    # Create the training example in the format required for fine-tuning
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": label}
        ]
    }

def process_results_and_calculate_metrics(output_dir: str) -> None:
    """
    Process the results from a fine-tuning run and calculate metrics.
    
    Args:
        output_dir (str): Path to the directory containing the run configuration and results
    """
    # Load the run configuration
    with open(os.path.join(output_dir, "run_config.json"), "r") as f:
        run_config = json.load(f)
    
    job_id = run_config['job_id']
    
    # Retrieve the fine-tuning job
    job = client.fine_tuning.jobs.retrieve(job_id)
    print(f"Job status: {job.status}")
    print(f"Created at: {datetime.utcfromtimestamp(job.created_at).strftime('%Y-%m-%d %H:%M:%S')}")
    
    if job.status != "succeeded":
        print("Fine-tuning job has not completed successfully.")
        return
    
    # Get the fine-tuned model name
    model_name = job.fine_tuned_model
    print(f"Fine-tuned model: {model_name}")
    
    # Load the test set
    test_set = pd.read_pickle("../../data/wdc/wdcproducts80cc20rnd050un_test_gs.pkl")
    
    # Create test examples
    test_examples = []
    PROMPT_TEMPLATE = "Do the two product descriptions refer to the same real-world product? Entity 1: 'Entity 1'. Entity 2: 'Entity 2'."
    
    for index, row in test_set.iterrows():
        product1, product2 = row['title_left'], row['title_right']
        label = row.get('label')
        pair_id = row['pair_id']
        
        example = create_training_example(PROMPT_TEMPLATE, product1, product2, label)
        test_examples.append({
            "pair_id": pair_id,
            "label": label,
            "messages": example["messages"]
        })
    
    # Get predictions from the fine-tuned model
    predictions = []
    for example in test_examples:
        response = client.chat.completions.create(
            model=model_name,
            messages=example["messages"],
            temperature=0
        )
        predictions.append({
            "pair_id": example["pair_id"],
            "label": example["label"],
            "prediction": 1 if "Yes" in response.choices[0].message.content else 0
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(predictions)
    
    # Calculate metrics
    y_true = results_df['label']
    y_pred = results_df['prediction']
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred)
    }
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    output_path = os.path.join(output_dir, 'stats.csv')
    metrics_df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")
    
    return metrics

# Example usage:
# process_results_and_calculate_metrics("../../results/gpt-4.1-mini-2025-04-14/fine-tune-wdc-small-regular/20240321_123456")

# Load the test set
train_set = pd.read_pickle("../../data/wdc/train_small/preprocessed_wdcproducts80cc20rnd000un_train_small.pkl.gz")

# Create output directory structure
run_name = "fine-tune-wdc-small-regular"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"../../results/{run_name}/{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Create training examples
training_examples = []
print(f"Creating training examples for {run_name}. The dataset has {len(train_set)} pairs.")

PROMPT_TEMPLATE = "Do the two product descriptions refer to the same real-world product? Entity 1: 'Entity 1'. Entity 2: 'Entity 2'."

for index, row in train_set.iterrows():
    product1, product2 = row['title_left'], row['title_right']
    label = row.get('label')
    pair_id = row['pair_id']
    
    example = create_training_example(PROMPT_TEMPLATE, product1, product2, label)
    training_examples.append(example)

# Save the training file
training_file_path = os.path.join(output_dir, "training.jsonl")
with open(training_file_path, "w") as f:
    for example in training_examples:
        f.write(json.dumps(example) + "\n")

# Upload the training file using the SDK
training_file = client.files.create(
    file=open(training_file_path, "rb"),
    purpose="fine-tune"
)

print(f"Training file uploaded successfully. File ID: {training_file.id}")

# Start the fine-tuning job using the SDK
fine_tune_job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    model="gpt-3.5-turbo",
    hyperparameters={
        "n_epochs": 3
    }
)

print(f"Fine-tuning job started successfully. Job ID: {fine_tune_job.id}")

# Save the run configuration
run_config = {
    "run_name": run_name,
    "timestamp": timestamp,
    "model": "gpt-3.5-turbo",
    "file_id": training_file.id,
    "job_id": fine_tune_job.id,
    "status": fine_tune_job.status,
    "created_at": fine_tune_job.created_at
}

with open(os.path.join(output_dir, "run_config.json"), "w") as f:
    json.dump(run_config, f, indent=2)

print(f"Run configuration and training files saved to: {output_dir}")

# Optional: Monitor the fine-tuning job
print("\nMonitoring fine-tuning job status...")
job = client.fine_tuning.jobs.retrieve(fine_tune_job.id)
print(f"Current status: {job.status}")
if job.status == "succeeded":
    print(f"Fine-tuned model: {job.fine_tuned_model}") 