# TailorMatch: Fine-Tuning Large Language Models for Enhanced Entity Matching

This repository contains the code and examples to reproduce and extend the experiments from our paper **"Fine-tuning Large Language Models for Entity Matching"**. The preprint is available on [arxiv](####).

## Requirements

- Python 3.11+
- [venv](https://docs.python.org/3/library/venv.html)

## Setup

1. **Create and activate virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Jupyter Notebooks

This section provides an overview of the key Jupyter notebooks used in the project for various tasks such as calculating metrics, testing models, and transforming datasets.

- **`src/calculate_results.ipynb`**: 
  This notebook is used to calculate the evaluation metrics reported in the paper based on the outputs of different models. It processes the results from both base and fine-tuned models and computes key metrics like precision, recall, and F1 scores. This is essential for comparing model performance.

- **`src/test_gpt_model.ipynb`**: 
  This notebook is used to test both base and fine-tuned GPT models. It allows you to input different datasets and evaluate how the models perform.

- **`src/create_data_set.ipynb`**: 
  This notebook handles various dataset transformations.

- **`src/error_analysis.ipynb`**: 
  This notebook performs an error analysis of model predictions using GPT.

- **`src/filter_data.ipynb`**: 
  This notebook is used to filter existing datasets, either to remove irrelevant or noisy data or to subset the data based on specific criteria. Filtering improves the quality of the dataset and ensures that only the most relevant data is used for training or evaluation.

- **`src/fine-tune-gpt.ipynb`**: 
  This notebook handles the fine-tuning of GPT models. 

- **`src/generate_embeddings.ipynb`**: 
  This notebook is used to generate embeddings for a given dataset using pre-trained language models. Embeddings are low-dimensional representations of the data that can be used for tasks such as similarity search.

- **`src/gpt_generate_explanations.ipynb`**: 
  This notebook generates explanations for the dataset, typically using GPT models. 

- **`src/gpt_generate_new_examples.ipynb`**: 
  This notebook is used to generate new examples or synthetic data using GPT models. It allows for the creation of additional training data, which can be useful for expanding small datasets, generating diverse examples, or improving model generalization.

## Python Files

- **`src/baseline_few_shot.py`**: Evaluates models on few-shot learning tasks, assessing how well the models perform with minimal training examples.
  
- **`src/continue_fine_tuning.py`**: Trains a model iteratively, with each iteration focusing on correcting errors from the previous iteration to improve performance.
  
- **`src/fine_tuning.py`**: Handles the fine-tuning of open-source models.

- **`src/test_model.py`**: Evaluates fine-tuned models to measure their performance after training, and validation.

- **`src/validation.py`**: Validates the performance of fine-tuned models.
```​⬤

## Datasets

We use the [WDC Products](https://webdatacommons.org/largescaleproductcorpus/wdc-products/#toc5) dataset and other standard benchmarks like Abt-Buy, Amazon-Google, and DBLP-Scholar.


## Results

You can refer to `Fine-tuning-results.xlsx` for a comprehensive overview of the model's performance.