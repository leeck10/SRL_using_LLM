# -*- coding: utf-8 -*-
"""Conditional Entropy (ConE) based example ordering optimization.

Evaluates all permutations of k examples by computing a normalized
cross-entropy score using the LLM, then selects the ordering that
minimizes the score.

Score = (CE_full - CE_examples_only) / CE_query_only

Lower scores indicate better orderings.
"""

from typing import List, Tuple, Dict
from itertools import permutations

import torch
import numpy as np
from tqdm import tqdm

from prompts.builder import build_prompt


def compute_ce_loss(
    model, tokenizer, text: str, device: str = "cuda"
) -> float:
    """Compute the normalized cross-entropy loss of a text sequence.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        text: Input text.
        device: Device string.

    Returns:
        Normalized cross-entropy loss (total loss / sequence length).
    """
    input_ids = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**input_ids)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = input_ids["input_ids"][..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(
        reduction="none", ignore_index=tokenizer.pad_token_id
    )
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    ).view(shift_labels.size())

    seq_len = (input_ids["input_ids"] != tokenizer.pad_token_id).sum(-1).cpu().float().numpy()
    ce_loss = loss.sum(-1).cpu().detach().float().numpy() / seq_len

    return float(ce_loss)


def find_optimal_order(
    model,
    tokenizer,
    test_data: List[Dict],
    language: str,
    output_format: str,
    num_examples: int = 5,
    sample_size: int = 300,
    max_input_length: int = 3000,
    seed: int = 42,
    device: str = "cuda",
) -> List[Tuple[float, Tuple[int, ...]]]:
    """Find the optimal example ordering by evaluating all permutations.

    For each sampled test instance, computes the ConE score for all k!
    permutations, then aggregates across instances.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        test_data: List of dicts, each with ``test`` and ``examples`` keys.
        language: ``'en'`` or ``'ko'``.
        output_format: ``'conll'`` or ``'dict'``.
        num_examples: Number of examples (k) to permute.
        sample_size: Number of test instances to sample for evaluation.
        max_input_length: Skip instances exceeding this token length.
        seed: Random seed for sampling.
        device: Device string.

    Returns:
        List of ``(avg_score, order_tuple)`` sorted ascending (best first).
    """
    import random
    random.seed(seed)

    if len(test_data) > sample_size:
        test_data = random.sample(test_data, sample_size)

    eos_token = tokenizer.eos_token or ""

    # Initialize scores for all permutations
    candidate_orders = list(permutations(range(num_examples), num_examples))
    order_scores = {order: 0.0 for order in candidate_orders}

    valid_count = 0
    data_iter = tqdm(enumerate(test_data), desc="ConE Ordering", total=len(test_data))

    for iii, tt in data_iter:
        for jjj, current_order in enumerate(candidate_orders):
            # Build full prompt (examples + query)
            prompt_full = build_prompt(
                tt["test"], tt["examples"], list(current_order),
                language, output_format, eos_token
            )

            # Check length only for the first permutation
            if jjj == 0:
                input_ids = tokenizer(prompt_full, return_tensors="pt")
                in_len = input_ids["input_ids"].shape[-1]
                if in_len > max_input_length:
                    break  # skip this instance entirely

            # Build examples-only prompt (no query)
            from prompts.templates import get_templates
            sys_prompt, ex_tmpl, _ = get_templates(language, output_format)
            from evaluation.format_converter import conll_to_dict

            prompt_ex_only = sys_prompt + "\n\n"
            for idx in current_order:
                ex = tt["examples"][idx]
                gold = ex["gold"]
                if output_format == "dict":
                    gold = conll_to_dict(gold)
                prompt_ex_only += ex_tmpl.format(
                    sentence=ex["sentence"], v_org=ex["v_org"],
                    roles=ex["roles"], gold=gold, eos=eos_token,
                )

            # Build query-only prompt (system + query, no examples)
            _, _, q_tmpl = get_templates(language, output_format)
            prompt_q_only = sys_prompt + "\n\n" + q_tmpl.format(
                sentence=tt["test"]["sentence"],
                v_org=tt["test"]["v_org"],
                roles=tt["test"]["roles"],
            )

            ce_full = compute_ce_loss(model, tokenizer, prompt_full, device)
            ce_ex = compute_ce_loss(model, tokenizer, prompt_ex_only, device)
            ce_q = compute_ce_loss(model, tokenizer, prompt_q_only, device)

            if ce_q != 0:
                score = (ce_full - ce_ex) / ce_q
            else:
                score = 0.0

            order_scores[current_order] += score
        else:
            valid_count += 1
            continue
        # break from inner loop means we skipped this instance
        continue

    # Normalize and sort
    if valid_count > 0:
        order_scores = {k: v / valid_count for k, v in order_scores.items()}

    sorted_orders = sorted(order_scores.items(), key=lambda x: x[1])
    return [(score, order) for order, score in sorted_orders]


def save_order_file(results: List[Tuple[float, Tuple[int, ...]]], filepath: str):
    """Save ordering results to a text file.

    Format: ``[score](idx0, idx1, idx2, ...)``

    Args:
        results: List of ``(score, order_tuple)`` from :func:`find_optimal_order`.
        filepath: Output file path.
    """
    with open(filepath, "w") as f:
        for score, order in results:
            f.write(f"[{score:.8f}]{order}\n")


def load_order_file(filepath: str) -> List[int]:
    """Load the best ordering from an order file.

    Args:
        filepath: Path to the order file.

    Returns:
        List of example indices representing the best ordering.
    """
    with open(filepath, "r") as f:
        first_line = f.readline().strip()
    # Parse: [score](idx0, idx1, ...)
    order_str = first_line.split("]")[1]
    order_str = order_str.strip("()")
    return [int(x.strip()) for x in order_str.split(",")]
