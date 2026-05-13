# SRL-ICL: Enhancing Semantic Role Labeling through Optimized Example Selection and Reordering in In-Context Learning

This repository provides the source code for reproducing the experiments described in the paper. The pipeline applies In-Context Learning (ICL) with Large Language Models (LLMs) for Semantic Role Labeling (SRL), with optimized example selection and ordering strategies.

## Pipeline Overview

```
[Training Data] ──► [BERT-CRF Training] ──► [Encoder Checkpoint]
                                                     │
[Training Data] ──► [Encode & Build DB] ◄────────────┘
                           │
[Test Data] ──► [Example Selection (Top-K / MMR)] ◄──┘
                           │
                    [ConE Ordering Optimization]  (optional)
                           │
                    [LLM Inference & Evaluation]
```

**Step 1.** Train a BERT-CRF model to serve as the sentence encoder for retrieval.
**Step 2.** Encode all training examples and build a vector database.
**Step 3.** For each test instance, retrieve similar examples using Top-K or MMR.
**Step 4.** (Optional) Find the optimal example ordering via ConE (Conditional Entropy).
**Step 5.** Run LLM inference with the selected examples and evaluate F1.


## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- ~48GB VRAM for 27B model (or ~16GB for 9B with 4-bit quantization)

```bash
pip install -r requirements.txt
```

## Data Preparation

Due to licensing restrictions, only sample data (10 sentences per split) is included. Please obtain the full datasets separately:

- **English**: CoNLL 2009 Shared Task data from [LDC](https://www.ldc.upenn.edu/)
- **Korean**: Korean PropBank from the dataset authors

Place the data files according to the paths in `configs/en_config.yaml` or `configs/ko_config.yaml`.

See `data/README.md` for format details.

## Quick Start

```bash
# Run the full pipeline with ConE ordering (English)
bash scripts/run_all.sh --config configs/en_config.yaml

# Run WITHOUT ConE ordering (fixed order 0,1,2,3,4)
bash scripts/run_without_cone.sh --config configs/en_config.yaml

# Run with a custom fixed order
bash scripts/run_without_cone.sh --config configs/en_config.yaml --order 2,0,4,1,3

# Specify GPU device
bash scripts/run_all.sh --config configs/en_config.yaml --gpu 1

# Use MMR selection strategy
bash scripts/run_all.sh --config configs/en_config.yaml --strategy mmr --lambda 0.9
```

## Step-by-Step Reproduction

```bash
# Step 1: Train BERT-CRF encoder (uses dev set for early stopping)
python scripts/01_train_crf.py --config configs/en_config.yaml

# Step 2: Build retrieval database
python scripts/02_build_retrieval_db.py --config configs/en_config.yaml

# Step 3: Select examples (Top-K or MMR)
python scripts/03_select_examples.py --config configs/en_config.yaml --strategy topk
python scripts/03_select_examples.py --config configs/en_config.yaml --strategy mmr --lambda_param 0.9

# Step 4: Optimize example ordering with ConE (uses cone_llm)
python scripts/04_optimize_order.py --config configs/en_config.yaml

# Step 5: Evaluate (uses eval_llm; loads order from Step 4 automatically)
python scripts/05_evaluate.py --config configs/en_config.yaml
# Or specify a custom ordering:
python scripts/05_evaluate.py --config configs/en_config.yaml --order 2,1,3,4,0

```

## Inference (Single Sentence)

```python
from inference.pipeline import SRLPipeline

pipe = SRLPipeline.from_config("configs/en_config.yaml")

result = pipe.predict(
    sentence="The role of Celimene was mistakenly attributed to Christina Haag.",
    predicate="attribute.01",     # Verb sense ID (for frame lookup)
    predicate_index=6,            # 0-based word index of the predicate
    output_format="dict",         # "conll" or "dict"
    verbose=True,                 # Print per-step timings
)

print(result["prediction"])
```

**Note**: Predicate identification is assumed to be already completed. The input requires the verb sense number (e.g., `attribute.01`) and its position in the sentence.

## Project Structure

```
SRL-ICL/
├── models/          # BERT-CRF architecture (encoder, CRF layer, BiLSTM)
├── retrieval/       # Example retrieval (Euclidean/Mahalanobis, CRF/pretrained encoder)
├── ordering/        # ConE ordering optimization
├── prompts/         # Prompt templates (EN/KO × CoNLL/Dict) and builder
├── evaluation/      # Metrics (micro-F1) and format conversion
├── inference/       # End-to-end inference pipeline
├── utils/           # GPU setup and shared utilities
├── scripts/         # Step-by-step reproduction scripts + shell runners
├── data/            # Sample data and format documentation
├── configs/         # YAML configuration files
├── docs/            # Pseudocode and pipeline diagrams
├── requirements.txt
└── LICENSE
```

## Configuration

All hyperparameters and file paths are controlled via YAML config files in `configs/`. Key settings include:

| Parameter | Description | Default |
|---|---|---|
| `language` | `"en"` or `"ko"` | — |
| `output_format` | `"conll"` or `"dict"` | — |
| `gpu_id` | CUDA device index | `0` |
| `cone_llm.model_id` | LLM for ConE ordering (smaller) | `"google/gemma-2-9b-it"` |
| `eval_llm.model_id` | LLM for final evaluation (larger) | `"google/gemma-2-27b-it"` |
| `retrieval.encoder_type` | `"crf"` or `"pretrained"` | `"crf"` |
| `retrieval.metric` | `"euclidean"` or `"mahalanobis"` | `"euclidean"` |
| `retrieval.strategy` | `"topk"` or `"mmr"` | `"topk"` |
| `cone.num_examples` | Number of examples to permute (k) | `5` |
| `crf_training.dev_file` | Dev set for early stopping | — |

## Environment

Experiments were conducted on:
- **OS**: Ubuntu 22.04
- **GPU**: NVIDIA RTX A6000 (48GB)
- **Framework**: PyTorch 2.7.1, Transformers 4.40+

