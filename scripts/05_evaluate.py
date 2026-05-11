#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 5: Evaluate SRL performance with the optimized example ordering.

Runs the LLM with the selected examples in the optimized order
and computes micro-F1 on the test set.

Uses ``eval_llm`` from config (separate from cone_llm).
Falls back to ``llm`` config key if ``eval_llm`` is not present.

The example order can be specified in three ways (in priority order):
  1. CLI ``--order`` flag (e.g., ``--order 2,1,3,4,0``)
  2. Order file from ConE optimization (config ``cone.order_output``)
  3. Default sequential order ``[0, 1, 2, ..., k-1]``

Usage:
    python scripts/05_evaluate.py --config configs/en_config.yaml
    python scripts/05_evaluate.py --config configs/en_config.yaml --order 2,1,3,4,0
    python scripts/05_evaluate.py --config configs/en_config.yaml --gpu 1
"""

import os
import sys
import json
import argparse
import time

# --- Set GPU BEFORE importing any CUDA libraries ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.gpu import set_gpu_before_import
set_gpu_before_import()

import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

from prompts.builder import build_prompt
from evaluation.metrics import eval_conll, eval_dict, compute_f1
from evaluation.format_converter import conll_to_dict
from ordering.cone import load_order_file


def main():
    parser = argparse.ArgumentParser(description="Evaluate SRL with LLM")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--order", default=None,
                        help="Comma-separated example order (e.g., '2,1,3,4,0'). "
                             "If not given, loads from the order file in config.")
    parser.add_argument("--gpu", type=int, default=None,
                        help="CUDA device index (overrides config gpu_id)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # GPU was already configured by set_gpu_before_import() above.
    ret_cfg = cfg["retrieval"]
    cone_cfg = cfg.get("cone", {})
    language = cfg["language"]
    output_format = cfg["output_format"]

    # Use eval_llm for evaluation (falls back to llm if eval_llm not defined)
    llm_cfg = cfg.get("eval_llm", cfg.get("llm", {}))

    # Determine example order
    if args.order:
        example_order = [int(x.strip()) for x in args.order.split(",")]
        print(f"Example order (from CLI): {example_order}")
    else:
        order_file = cone_cfg.get("order_output", "order.txt")
        if os.path.exists(order_file):
            example_order = load_order_file(order_file)
            print(f"Example order (from {order_file}): {example_order}")
        else:
            num_ex = cone_cfg.get("num_examples", 5)
            example_order = list(range(num_ex))
            print(f"Example order (default sequential): {example_order}")

    # Load prompt data
    with open(ret_cfg["prompt_output"], "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test instances")

    # Load LLM (evaluation-specific, typically larger than ConE model)
    print(f"Loading eval model: {llm_cfg['model_id']}")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(llm_cfg["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        llm_cfg["model_id"],
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )

    eos_token = tokenizer.eos_token or ""

    # Evaluate
    total_gold, total_pred, total_correct = 0, 0, 0
    results = []
    eval_fn = eval_conll if output_format == "conll" else eval_dict

    data_iter = tqdm(enumerate(test_data), desc="Evaluating", total=len(test_data))

    for iii, tt in data_iter:
        # Build prompt
        prompt = build_prompt(
            tt["test"], tt["examples"], example_order,
            language, output_format, eos_token,
        )

        # Tokenize and generate
        input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = model.generate(
            **input_ids,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
        )

        pred_result = tokenizer.decode(
            outputs[0][input_ids["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

        # Prepare gold for evaluation
        if output_format == "dict":
            gold_str = conll_to_dict(tt["test"]["gold"])
        else:
            gold_str = tt["test"]["gold"]

        a, b, c = eval_fn(gold_str, pred_result)
        total_gold += a
        total_pred += b
        total_correct += c

        results.append(pred_result)

        # Running F1
        r, p, f1 = compute_f1(total_gold, total_pred, total_correct)
        data_iter.set_postfix(F1=f"{f1:.4f}")

    # Final scores
    recall, precision, f1 = compute_f1(total_gold, total_pred, total_correct)
    print(f"\n{'='*50}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"{'='*50}")

    # Save predictions
    output_path = ret_cfg["prompt_output"] + ".predictions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
