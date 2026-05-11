#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 3: Select in-context examples for each test instance.

Retrieves the top-k or MMR-selected examples from the training database
for each test instance and saves the prompt data as JSON.

Supports encoder_type selection via config:
  - "crf": Use the fine-tuned BERT-CRF checkpoint (default).
  - "pretrained": Use a vanilla pre-trained BERT model.

Usage:
    python scripts/03_select_examples.py --config configs/en_config.yaml
    python scripts/03_select_examples.py --config configs/en_config.yaml --strategy mmr --lambda_param 0.9
    python scripts/03_select_examples.py --config configs/en_config.yaml --gpu 1
"""

import os
import sys
import json
import argparse

# --- Set GPU BEFORE importing any CUDA libraries ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.gpu import set_gpu_before_import
set_gpu_before_import()

import yaml
from tqdm import tqdm

from utils.io import load_json
from retrieval.encoder import BertEncoder
from retrieval.database import VectorDatabase
from retrieval.selector import select_topk, select_mmr


def load_srl_data(data_path, framefiles):
    """Load SRL data in the ICL format."""
    temp = []
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if len(line) < 1:
                if not temp:
                    continue
                sentence_org = temp[0].strip()
                text = []
                v_position = []
                verbs = []
                for part in temp[1].split("\t"):
                    if part != "_":
                        verbs.append(part)
                for i, row in enumerate(temp[2:]):
                    cols = row.split("\t")
                    text.append(cols[0])
                    if cols[1] != "_":
                        v_position.append(i)
                for ii, (v, idx) in enumerate(zip(verbs, v_position)):
                    sentence = []
                    v_org = ""
                    for i, word in enumerate(text):
                        if i == idx:
                            sentence.append("<predicate>")
                            sentence.append(word)
                            sentence.append("</predicate>")
                            v_org = word
                        else:
                            sentence.append(word)
                    gold_conll = ""
                    for row in temp[2:]:
                        cols = row.split("\t")
                        gold_conll += cols[0] + "\t" + cols[ii + 2] + "\n"
                    gold_conll = gold_conll.strip()
                    sentence_str = " ".join(sentence)
                    roles = []
                    if v in framefiles:
                        for rr in framefiles[v]:
                            roles.append(rr)
                    roles_str = "\n".join(roles)
                    data.append({
                        "sentence_org": sentence_org,
                        "sentence": sentence_str,
                        "v_org": v_org,
                        "v": v,
                        "roles": roles_str,
                        "gold": gold_conll,
                    })
                temp = []
            else:
                temp.append(line)
    return data


def main():
    parser = argparse.ArgumentParser(description="Select in-context examples")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--strategy", default=None, choices=["topk", "mmr"],
                        help="Selection strategy (overrides config)")
    parser.add_argument("--lambda_param", type=float, default=None,
                        help="MMR lambda parameter (overrides config)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="CUDA device index (overrides config gpu_id)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # GPU was already configured by set_gpu_before_import() above.
    ret_cfg = cfg["retrieval"]
    data_cfg = cfg["data"]

    # Resolve strategy and lambda from CLI or config
    strategy = args.strategy or ret_cfg.get("strategy", "topk")
    lambda_param = args.lambda_param if args.lambda_param is not None else ret_cfg.get("mmr_lambda", 0.7)

    # Load framefiles (auto-detect encoding for Korean files)
    framefiles = load_json(data_cfg["framefiles"])

    # Load test data
    test_data = load_srl_data(data_cfg["test_file"], framefiles)
    print(f"Loaded {len(test_data)} test instances")

    # Load train data
    train_json_path = ret_cfg["db_path"] + "_train_data.json"
    with open(train_json_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    print(f"Loaded {len(train_data)} training examples")

    # Load database
    db = VectorDatabase.load(ret_cfg["db_path"], use_gpu=True)
    print(f"Loaded database with {len(db.data)} vectors (metric={db.metric})")

    # Determine encoder type and model path
    encoder_type = ret_cfg.get("encoder_type", "crf")
    if encoder_type == "pretrained":
        encoder_model = ret_cfg.get("pretrained_model", "bert-base-uncased")
        print(f"Using pre-trained BERT encoder: {encoder_model}")
    else:
        encoder_model = ret_cfg["encoder_model"]
        print(f"Using fine-tuned CRF encoder: {encoder_model}")

    # Initialize encoder
    encoder = BertEncoder(
        bert_model_name=encoder_model, use_gpu=True, encoder_type=encoder_type
    )

    # Select examples for each test instance
    num_examples = ret_cfg.get("num_examples", 50)
    prompt_data = []
    print(f"Strategy: {strategy}, num_examples: {num_examples}")

    for ii, data in tqdm(enumerate(test_data), desc="Selecting", total=len(test_data)):
        query = data["sentence_org"] + " [SEP] " + data["v_org"] + " [SEP]"
        query_vector = encoder.encode(query)

        if strategy == "topk":
            results = select_topk(db, query_vector, k=num_examples)
        else:
            results = select_mmr(
                db, query_vector, k=num_examples,
                lambda_param=lambda_param,
                candidate_pool_size=min(num_examples * 5, len(db.data)),
            )

        examples = [train_data[idx] for _, idx in results]
        prompt_data.append({"test": data, "examples": examples})

    # Save
    output_path = ret_cfg["prompt_output"]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prompt_data, f, ensure_ascii=False)
    print(f"Saved {len(prompt_data)} prompt instances to: {output_path}")


if __name__ == "__main__":
    main()
