# -*- coding: utf-8 -*-
"""Evaluation metrics for SRL output.

Supports two evaluation modes matching the output format:
  - CoNLL format: Position-based token-level matching.
  - Dict format: Role-word pair exact matching.

No special handling is applied for malformed LLM outputs.
"""

from typing import Tuple

from evaluation.format_converter import conll_to_dict, parse_dict_output


def eval_conll(gold: str, pred: str) -> Tuple[int, int, int]:
    """Evaluate CoNLL-format output by position-based matching.

    Compares gold and predicted labels at each token position.
    If the prediction has fewer lines, missing lines count as ``_``.
    If the prediction has more lines, extra lines are ignored.

    Args:
        gold: Gold CoNLL string (``word\\tlabel`` per line).
        pred: Predicted CoNLL string.

    Returns:
        Tuple of (gold_count, pred_count, correct_count).
    """
    gold = gold.strip()
    pred = pred.strip()

    g_labels = [line.split("\t")[-1] for line in gold.split("\n")]
    p_lines = pred.split("\n")

    # Align prediction length to gold length
    p_labels = []
    for i in range(len(g_labels)):
        if i < len(p_lines):
            p_labels.append(p_lines[i].split("\t")[-1])
        else:
            p_labels.append("_")

    gold_count = 0
    pred_count = 0
    correct_count = 0

    for g, p in zip(g_labels, p_labels):
        is_g_arg = "ARG" in g or "AUX" in g
        is_p_arg = "ARG" in p or "AUX" in p

        if is_g_arg:
            gold_count += 1
        if is_p_arg:
            pred_count += 1
        if is_g_arg and is_p_arg and g == p:
            correct_count += 1

    return gold_count, pred_count, correct_count


def eval_dict(gold: str, pred: str) -> Tuple[int, int, int]:
    """Evaluate Dict-format output by role-word pair exact matching.

    Args:
        gold: Gold in Dict format (``label\\tword`` per line).
        pred: Predicted in Dict format.

    Returns:
        Tuple of (gold_count, pred_count, correct_count).
    """
    gold_pairs = parse_dict_output(gold)
    pred_pairs = parse_dict_output(pred)

    gold_count = len(gold_pairs)
    pred_count = len(pred_pairs)
    correct_count = sum(1 for p in pred_pairs if p in gold_pairs)

    return gold_count, pred_count, correct_count


def compute_f1(
    gold_count: int, pred_count: int, correct_count: int
) -> Tuple[float, float, float]:
    """Compute Recall, Precision, and F1 from counts.

    Args:
        gold_count: Number of gold arguments.
        pred_count: Number of predicted arguments.
        correct_count: Number of correctly predicted arguments.

    Returns:
        Tuple of (recall, precision, f1).
    """
    recall = correct_count / gold_count if gold_count > 0 else 0.0
    precision = correct_count / pred_count if pred_count > 0 else 0.0
    f1 = (
        2 * recall * precision / (recall + precision)
        if (recall + precision) > 0
        else 0.0
    )
    return recall, precision, f1
