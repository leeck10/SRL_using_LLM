#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 2: Build a retrieval database from training data.

Encodes all training examples using the BERT encoder (fine-tuned CRF or
pre-trained) and stores them in a VectorDatabase.

Supports encoder_type selection via config:
  - "crf": Use the fine-tuned BERT-CRF checkpoint (default).
  - "pretrained": Use a vanilla pre-trained BERT model.

Usage:
    python scripts/02_build_retrieval_db.py --config configs/en_config.yaml
    python scripts/02_build_retrieval_db.py --config configs/en_config.yaml --gpu 1
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


def load_srl_data(data_path, framefiles):
    """Load SRL data in the ICL format (sentence, verb, roles, gold)."""
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
    parser = argparse.ArgumentParser(description="Build retrieval database")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--gpu", type=int, default=None,
                        help="CUDA device index (overrides config gpu_id)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # GPU was already configured by set_gpu_before_import() above.
    ret_cfg = cfg["retrieval"]
    data_cfg = cfg["data"]

    # Load framefiles (auto-detect encoding for Korean files)
    framefiles = load_json(data_cfg["framefiles"])

    # Load training data
    train_data = load_srl_data(data_cfg["train_file"], framefiles)
    print(f"Loaded {len(train_data)} training examples")

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
        bert_model_name=encoder_model,
        use_gpu=True,
        encoder_type=encoder_type,
    )

    # Initialize database
    db = VectorDatabase(metric=ret_cfg["metric"], use_gpu=True)

    # Encode all training examples
    for i, dd in tqdm(enumerate(train_data), desc="Encoding", total=len(train_data)):
        query_str = dd["sentence_org"] + " [SEP] " + dd["v_org"] + " [SEP]"
        vector = encoder.encode(query_str)
        db.add_item(vector, i)

    # Update covariance matrix (for Mahalanobis distance)
    db.update_covariance_matrix()

    # Save database
    os.makedirs(os.path.dirname(ret_cfg["db_path"]) or ".", exist_ok=True)
    db.save(ret_cfg["db_path"])
    print(f"Database saved to: {ret_cfg['db_path']}.pkl")

    # Also save train_data as JSON for later use
    train_json_path = ret_cfg["db_path"] + "_train_data.json"
    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False)
    print(f"Training data saved to: {train_json_path}")


if __name__ == "__main__":
    main()
