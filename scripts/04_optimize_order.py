#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 4: Optimize example ordering using ConE (Conditional Entropy).

Evaluates all k! permutations of example orderings using the LLM's
cross-entropy loss and selects the ordering with the lowest ConE score.

Uses ``cone_llm`` from config (separate from eval_llm) to reduce VRAM usage,
since ConE requires holding the entire model in memory while computing
cross-entropy for k! * 3 forward passes per instance.

Falls back to ``llm`` config key if ``cone_llm`` is not present.

Usage:
    python scripts/04_optimize_order.py --config configs/en_config.yaml
    python scripts/04_optimize_order.py --config configs/en_config.yaml --gpu 1
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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from ordering.cone import find_optimal_order, save_order_file


def main():
    parser = argparse.ArgumentParser(description="Optimize example ordering with ConE")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--gpu", type=int, default=None,
                        help="CUDA device index (overrides config gpu_id)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # GPU was already configured by set_gpu_before_import() above.
    ret_cfg = cfg["retrieval"]
    cone_cfg = cfg.get("cone", {})

    # Use cone_llm for ordering (falls back to llm if cone_llm not defined)
    llm_cfg = cfg.get("cone_llm", cfg.get("llm", {}))

    # Load prompt data
    with open(ret_cfg["prompt_output"], "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test instances")

    # Load LLM (ConE-specific, typically smaller to save VRAM)
    print(f"Loading ConE model: {llm_cfg['model_id']}")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(llm_cfg["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        llm_cfg["model_id"],
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )

    # Find optimal ordering
    results = find_optimal_order(
        model=model,
        tokenizer=tokenizer,
        test_data=test_data,
        language=cfg["language"],
        output_format=cfg["output_format"],
        num_examples=cone_cfg.get("num_examples", 5),
        sample_size=cone_cfg.get("sample_size", 300),
        max_input_length=cone_cfg.get("max_input_length", 3000),
        seed=cone_cfg.get("seed", 42),
    )

    # Save results
    output_path = cone_cfg.get("order_output", "order.txt")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_order_file(results, output_path)
    print(f"\nOrder file saved to: {output_path}")
    print(f"Best ordering: {results[0][1]} (score: {results[0][0]:.8f})")


if __name__ == "__main__":
    main()
