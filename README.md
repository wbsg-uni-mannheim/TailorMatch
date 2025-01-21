# TailorMatch: Fine-Tuning Large Language Models for Enhanced Entity Matching

	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fine-tuning-large-language-models-for-entity/entity-resolution-on-wdc-products)](https://paperswithcode.com/sota/entity-resolution-on-wdc-products?p=fine-tuning-large-language-models-for-entity)

This repository contains the code and examples to reproduce and extend the experiments from our paper **"Fine-tuning Large Language Models for Entity Matching"**. The preprint is available on [arxiv](https://arxiv.org/abs/2409.08185).

## Requirements

- Python 3.9
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


## Datasets

The datasets used is the [WDC Products](https://webdatacommons.org/largescaleproductcorpus/wdc-products/#toc5) dataset containing 80% corner-cases and the Abt-Buy, Amazon-Google, Walmart-Amazon, DBLP-Scholar and DBLP-ACM datasets using the [downloads and splits from the Deepmatcher paper](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md).

When working with open-source models, the datasets are provided in CSV format. For models using OpenAI APIs, the datasets are formatted as JSONL files to align with the preferred data input format for OpenAI models.

## Results

You can refer to `Fine-tuning-results.xlsx` for a comprehensive overview. The file consolidates and aggregates the results from various scenarios, providing a complete analysis.

Each run produces a results.csv or stats.csv file containing the metrics specific to that particular scenario. Additionally, the results folder for each scenario includes the model’s raw outputs.


## Prompts

Below, we provide examples of the different prompt designs. 

### Evaluation prompts

```json
[
    {
        "title": "domain-complex-free (Product)",
        "prompt": "Do the two product descriptions refer to the same real-world product? Entity 1: 'Entity 1'. Entity 2: 'Entity 2'."
    },
    {
        "title": "domain-simple-free (Product)",
        "prompt": "Do the two product descriptions match? Entity 1: 'Entity 1'. Entity 2: 'Entity 2'."
    },
    {
        "title": "domain-complex-force (Product)",
        "prompt": "Do the two product descriptions refer to the same real-world product? Answer with 'Yes' if they do and 'No' if they do not. Entity 1: 'Entity 1'. Entity 2: 'Entity 2'."
    },
    {
        "title": "domain-simple-force (Product)",
        "prompt": "Do the two product descriptions match? Answer with 'Yes' if they do and 'No' if they do not. Entity 1: 'Entity 1'. Entity 2: 'Entity 2'."
    }
]
````


### Explanation generation

```json
[
    {
        "title": "explanation-long-textual",
        "prompt": "Do the two entity descriptions refer to the same real-world entity? Entity 1: 'Entity 1'. Entity 2: 'Entity 2'. Please provide an explanation. The correct answer is 'label'"
    },
    {
        "title": "explanation-structured",
        "prompt": "Do the two entity descriptions refer to the same real-world entity?\\nEntity 1: {{product_1}}\\nEntity 2: {{product_2}}\\n\\nThe correct answer is {{label}}.\\n\\nPlease provide an explanation for this answer in a structured format, listing the attributes that you compared for reaching this answer. Each attribute should be accompanied by the attribute values and a score between -1 and 1 that shows the importance of the attribute for the decision. If the attribute influenced the decision towards non-match the importance score should be negative. If the attribute pointed towards a match, the importance score should be positive. Also provide a similarity score for the attribute values. If an attribute only occurs in one item, specify the value of that attribute for the other item as 'missing'. An example output is the following:\\n\\nattribute=brand|||importance=0.05|||values=Logitech###Logitech|||similarity=1.00\\nattribute=model|||importance=-0.95|||values=MX G500###MX Master 3S|||similarity=0.20\\nattribute=color|||importance=0.00|||values=missing###Graphite|||similarity=0.00\\n\\nHere is a complete example:\\nDo the two product descriptions refer to the same real-world product? Entity 1: 'WD 4TB Black My Passport Portable External Hard Drive - USB 3.0 - WDBYFT0040BBK-WESN'. Entity 2: 'Dysk WD My Passport 1TB USB 3.0 black'.\\nNo. \\nattribute=brand|||importance=0.05|||values=Western Digital###Western Digital|||similarity=1.00\\nattribute=model|||importance=0.95|||values=My Passport###My Passport|||similarity=1.00\\nattribute=storage capacity|||importance=0.9|||values=4TB###1TB|||similarity=0.25\\nattribute=color|||importance=0.1|||values=Black###Black|||similarity=1.00\\nattribute=USB version|||importance=0.05|||values=USB 3.0###USB 3.0|||similarity=1.00\\n\\nDo not provide a explanation in a different format. The explanation should be in the format described above. Only provide the answer and explanation dont repeat the question."
    },
    {
        "title": "explanation-wadhwa",
        "prompt": "<s>[INST] Given the following two examples, provide an explanation for the third example for why the two entities do or do not match. [/INST]\n\nEntity A: [NAME] samsung dlp tv stand in black tr72bx [DESCRIPTION] samsung dlp tv stand in black tr72bx designed to fit samsung hlt7288, hlt7288, hl72a650, and hl67a650 television sets tempered 6mm tinted glass shelves wide audio storage shelves to accommodate 4 or more components wire management system easy to assemble high gloss black finish [PRICE] 369.0\nEntity B: [NAME] samsung tr72b tv stand [DESCRIPTION] glass black [PRICE] 232.14\nLabel: MATCH\nExplanation: Both entities refer to samsung TV stand in black and therefore have substantially similar specifications, therefore they’re a match. </s>\n\nEntity A: [NAME] canon high capacity color ink cartridge color ink cl51 [DESCRIPTION] canon high capacity color ink cartridge cl51 compatible with pixma ip6210d, ip6220d, mp150, mp170 and mp450 printers [PRICE] 35.0\nEntity B: [NAME] canon pg-40 twin pack black ink cartridge 0615b013 [DESCRIPTION] black [PRICE] Label: NOT A MATCH\nExplanation: Entity A refers to a color ink cartridge while Entity B is a black ink cartridge, therefore they are not a match. </s>\n\nEntity A: [NAME] {product_1_name} [DESCRIPTION] {product_1_description} [PRICE] {product_1_price}\nEntity B: [NAME] {product_2_name} [DESCRIPTION] {product_2_description} [PRICE] {product_2_price}\nLabel: {label}\nExplanation:"
    }
]
```

### Generate new examples

```json

[
    {
        "name": "simple generation",
        "message": "Please generate similar examples for enitity matching. The results should only be presented as JSON containing the generated entity, one and entity two as well as information if they are a match or not represented by boolean and value. Only return JSON.Generate one match and three non matches. The example that was misclassified is: Entity 1: {product_1} Entity 2: {product_2} Label: {label}}"
    },
    {
        "name": "textual generation",
        "message": "I'm currently testing large language, models on the task of entity matching. In this context, I am first fine-tuning them, and then testing their weaknesses and strengths. The example I will show you is wrongly classified by the model and that idea is to generate four new examples three of which should be negative, i.e. non-matches, and one of them match. For context, two products are considered to be a match if the two entity descriptions refer to the same real world entity. This does not mean that the descriptions need to be the same but that the entity the decription refers to needs to match. Secondly products are not a match if the two descriptions refer to different products.  As a model has previously made an error on these two entity descriptions it is important to create examples that present a similar challenge. Please focus on corner cases meaning examples that are quite difficult to get correct. The generated examples should belong to the same category as the presented product and should be very similar to it. However even if they are a match the strings should never match exactly. The results should only be presented as JSON containing the generated entity, one and entity two as well as information if they are a match or not represented by boolean and value. Only return JSON. The example that was misclassified is: Entity 1: {product_1} Entity 2: {product_2} Label: {label}}"
    },
    {
        "name": "textual generation examples",
        "message": "I'm currently testing large language, models on the task of entity matching. In this context, I am first fine-tuning them, and then testing their weaknesses and strengths. The example I will show you is wrongly classified by the model and that idea is to generate four new examples three of which should be negative, i.e. non-matches, and one of them match. For context, two products are considered to be a match if the two entity descriptions refer to the same real world entity. This does not mean that the descriptions need to be the same but that the entity the decription refers to needs to match. Secondly products are not a match if the two descriptions refer to different products.  As a model has previously made an error on these two entity descriptions it is important to create examples that present a similar challenge. Please focus on corner cases meaning examples that are quite difficult to get correct. The generated examples should belong to the same category as the presented product and should be very similar to it. However even if they are a match the strings should never match exactly. The results should only be presented as JSON containing the generated entity, one and entity two as well as information if they are a match or not represented by boolean and value. Only return JSON. Here are some relevant examples: {examples} The example that was misclassified is: Entity 1: {product_1} Entity 2: {product_2} Label: {label}}"
    }
]
```
