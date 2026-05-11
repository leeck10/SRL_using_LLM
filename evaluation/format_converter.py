# -*- coding: utf-8 -*-
"""Format conversion utilities between CoNLL and Dict SRL output formats.

CoNLL format (one token per line):
    word1\\tlabel1
    word2\\tlabel2
    ...

Dict format (only labeled tokens):
    LABEL1\\tword1
    LABEL2\\tword2
    ...
"""


def conll_to_dict(conll_str: str) -> str:
    """Convert CoNLL format to Dict format.

    Filters out tokens with ``_`` labels, swaps columns to ``label\\tword``.

    Args:
        conll_str: CoNLL-formatted string.

    Returns:
        Dict-formatted string.
    """
    lines = []
    for line in conll_str.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2 and parts[-1] != "_":
            lines.append(f"{parts[-1]}\t{parts[0]}")
    return "\n".join(lines)


def dict_to_conll(dict_str: str, words: list) -> str:
    """Convert Dict format back to CoNLL format.

    Args:
        dict_str: Dict-formatted string (``label\\tword`` per line).
        words: Complete list of words in the sentence.

    Returns:
        CoNLL-formatted string with ``_`` for unlabeled tokens.
    """
    label_map = {}
    for line in dict_str.strip().split("\n"):
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            label, word = parts[0], parts[-1]
            label_map[word] = label

    lines = []
    for word in words:
        label = label_map.get(word, "_")
        lines.append(f"{word}\t{label}")
    return "\n".join(lines)


def parse_dict_output(text: str) -> list:
    """Parse Dict-format output into a list of (label, word) tuples.

    Handles noisy LLM output by filtering malformed lines.

    Args:
        text: Raw Dict-format output string.

    Returns:
        List of unique ``(label, word)`` tuples.
    """
    seen = set()
    result = []
    for line in text.strip().split("\n"):
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            key = f"{parts[0]}\t{parts[-1]}"
            if key not in seen:
                seen.add(key)
                result.append(key)
    return result
