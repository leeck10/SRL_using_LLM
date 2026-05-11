# -*- coding: utf-8 -*-
"""Prompt builder — assembles system prompt + examples + query into a single string."""

from typing import List, Dict

from prompts.templates import get_templates
from evaluation.format_converter import conll_to_dict


def build_prompt(
    test_instance: Dict,
    examples: List[Dict],
    example_order: List[int],
    language: str,
    output_format: str,
    eos_token: str = "",
) -> str:
    """Build a complete prompt for LLM inference.

    Args:
        test_instance: Dict with keys ``sentence``, ``v_org``, ``roles``, ``gold``.
        examples: List of example dicts, each with the same keys as test_instance.
        example_order: List of indices into ``examples`` specifying the ordering.
        language: ``'en'`` or ``'ko'``.
        output_format: ``'conll'`` or ``'dict'``.
        eos_token: End-of-sequence token from the tokenizer.

    Returns:
        Complete prompt string ready for tokenization.
    """
    system_prompt, example_template, query_template = get_templates(language, output_format)

    prompt = system_prompt + "\n\n"

    for idx in example_order:
        ex = examples[idx]
        gold = ex["gold"]
        # Convert gold to dict format if needed
        if output_format == "dict":
            gold = conll_to_dict(gold)

        prompt += example_template.format(
            sentence=ex["sentence"],
            v_org=ex["v_org"],
            roles=ex["roles"],
            gold=gold,
            eos=eos_token,
        )

    prompt += query_template.format(
        sentence=test_instance["sentence"],
        v_org=test_instance["v_org"],
        roles=test_instance["roles"],
    )

    return prompt
