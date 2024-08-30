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

The code is organized into several Jupyter notebooks for different tasks, including zero-shot, in-context learning and fine-tuning.

## Datasets

We use the [WDC Products](https://webdatacommons.org/largescaleproductcorpus/wdc-products/#toc5) dataset and other standard benchmarks like Abt-Buy, Amazon-Google, and DBLP-Scholar.

