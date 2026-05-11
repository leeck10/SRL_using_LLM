# -*- coding: utf-8 -*-
"""Prompt templates for SRL In-Context Learning.

Four combinations: {English, Korean} x {CoNLL format, Dict format}

Each template defines:
  - SYSTEM_PROMPT: Adjunct argument definitions shown at the top of the prompt.
  - EXAMPLE_TEMPLATE: Format for each in-context example (Q/A pair).
  - QUERY_TEMPLATE: Format for the test query (Q only, no answer).
"""

# =============================================================================
# English Templates
# =============================================================================

EN_SYSTEM_PROMPT = (
    "<Adjunct Arguments>\n"
    "AM-ADV: Adverbial modification\n"
    "AM-CAU: Cause\n"
    "AM-DIR: Directional\n"
    "AM-DIS: Discourse marker\n"
    "AM-EXT: Extent\n"
    "AM-LOC: Location\n"
    "AM-MNR: Manner\n"
    "AM-MOD: Modal verb\n"
    "AM-NEG: Negation marker\n"
    "AM-PNC: Purpose\n"
    "AM-PRD: Secondary predication\n"
    "AM-PRT: Verb particle\n"
    "AM-REC: Reciprocal\n"
    "AM-TMP: Temporal\n"
    "\n"
    "<Special Tags>\n"
    "C-[ARG]: Continuation - used when an argument is split into discontinuous parts\n"
    "R-[ARG]: Reference - used when an argument references another argument, "
    "often in relative clauses\n"
    "\n"
    "SU: Support - used to mark supporting elements that help establish "
    "the predicate-argument structure\n"
)

# --- English CoNLL format ---

EN_CONLL_EXAMPLE_TEMPLATE = (
    "Q: For the sentence\n"
    "[{sentence}]\n"
    "Please provide the semantic role labels for the predicate "
    "[<predicate>{v_org}</predicate>] in CoNLL format. "
    "Use the original word tokens without sentence splitting. "
    "The predicate has the following core roles\n"
    "{roles}.\n"
    "A: {gold}{eos}\n\n"
)

EN_CONLL_QUERY_TEMPLATE = (
    "Q: For the sentence\n"
    "[{sentence}]\n"
    "Please provide the semantic role labels for the predicate "
    "[<predicate>{v_org}</predicate>] in CoNLL format. "
    "Use the original word tokens without sentence splitting. "
    "The predicate has the following core roles\n"
    "{roles}.\n"
    "A: "
)

# --- English Dict format ---

EN_DICT_EXAMPLE_TEMPLATE = (
    "Q: For the sentence\n"
    "[{sentence}]\n"
    "Please provide the semantic role labels for the predicate "
    "[<predicate>{v_org}</predicate>]. "
    "Use the original word tokens without sentence splitting. "
    "The predicate has the following core roles\n"
    "{roles}.\n"
    "A: {gold}{eos}\n\n"
)

EN_DICT_QUERY_TEMPLATE = (
    "Q: For the sentence\n"
    "[{sentence}]\n"
    "Please provide the semantic role labels for the predicate "
    "[<predicate>{v_org}</predicate>]. "
    "Use the original word tokens without sentence splitting. "
    "The predicate has the following core roles\n"
    "{roles}.\n"
    "A: "
)

# =============================================================================
# Korean Templates
# =============================================================================

KO_SYSTEM_PROMPT = (
    "다음은 한국어 의미역 결정의 부가격 의미역 정의이다.\n"
    "<ARGM-LOC 장소 (locatives)>\n"
    "<ARGM-DIR 방향 (directional)>\n"
    "<ARGM-CND 조건 (condition)>\n"
    "<ARGM-MNR 방법 (manner)>\n"
    "<ARGM-TMP 시간 (temporal)>\n"
    "<ARGM-EXT 범위 (extent)>\n"
    "<ARGM-PRD 보조 서술(secondary predication)>\n"
    "<ARGM-PRP 목적 (purpose clauses)>\n"
    "<ARGM-CAU 발생 이유 (cause clauses)>\n"
    "<ARGM-DIS 담화 연결 (discourse)>\n"
    "<ARGM-NEG 부정 (negation)>\n"
    "<ARGM-INS 도구 (instrument)>\n"
    "<ARGM-ADV 부사어 (adverbial)>\n"
    "<AUX 보조용언 (Auxiliary Verb)>\n"
)

# --- Korean CoNLL format ---

KO_CONLL_EXAMPLE_TEMPLATE = (
    "Q: 다음 문장\n"
    "[{sentence}]\n"
    "에서 동사 [<predicate>{v_org}</predicate>]에 대한 의미역 결정 분석을 해서 "
    "conll 포맷으로 출력해줘. 이때 문장 분리를 하지말고 어절 그대로 사용하며, "
    "가질 수 있는 필수격은\n{roles}\n이다. "
    "이전의 지시사항이나 예시는 반복하지 말고 반드시 의미역 분석 결과만 답변 할 것. "
    "추가설명이나 반복은 하지 말 것.\n"
    "A: {gold}{eos}\n\n"
)

KO_CONLL_QUERY_TEMPLATE = (
    "Q: 다음 문장\n"
    "[{sentence}]\n"
    "에서 동사 [<predicate>{v_org}</predicate>]에 대한 의미역 결정 분석을 해서 "
    "conll 포맷으로 출력해줘. 이때 문장 분리를 하지말고 어절 그대로 사용하며, "
    "가질 수 있는 필수격은\n{roles}\n이다. "
    "이전의 지시사항이나 예시는 반복하지 말고 반드시 의미역 분석 결과만 답변 할 것. "
    "추가설명이나 반복은 하지 말 것.\n"
    "A: "
)

# --- Korean Dict format ---

KO_DICT_EXAMPLE_TEMPLATE = (
    "Q: 다음 문장\n"
    "[{sentence}]\n"
    "에서 동사 [<predicate>{v_org}</predicate>]에 대한 의미역 결정 분석을 해줘. "
    "이때 문장 분리를 하지말고 어절 그대로 사용하며, "
    "가질 수 있는 필수격은\n{roles}\n이다. "
    "이전의 지시사항이나 예시는 반복하지 말고 반드시 의미역 분석 결과만 답변 할 것. "
    "추가설명이나 반복은 하지 말 것.\n"
    "A: {gold}{eos}\n\n"
)

KO_DICT_QUERY_TEMPLATE = (
    "Q: 다음 문장\n"
    "[{sentence}]\n"
    "에서 동사 [<predicate>{v_org}</predicate>]에 대한 의미역 결정 분석을 해줘. "
    "이때 문장 분리를 하지말고 어절 그대로 사용하며, "
    "가질 수 있는 필수격은\n{roles}\n이다. "
    "이전의 지시사항이나 예시는 반복하지 말고 반드시 의미역 분석 결과만 답변 할 것. "
    "추가설명이나 반복은 하지 말 것.\n"
    "A: "
)


def get_templates(language: str, output_format: str):
    """Return (system_prompt, example_template, query_template) for a given config.

    Args:
        language: ``'en'`` or ``'ko'``.
        output_format: ``'conll'`` or ``'dict'``.

    Returns:
        Tuple of (system_prompt, example_template, query_template).
    """
    templates = {
        ("en", "conll"): (EN_SYSTEM_PROMPT, EN_CONLL_EXAMPLE_TEMPLATE, EN_CONLL_QUERY_TEMPLATE),
        ("en", "dict"): (EN_SYSTEM_PROMPT, EN_DICT_EXAMPLE_TEMPLATE, EN_DICT_QUERY_TEMPLATE),
        ("ko", "conll"): (KO_SYSTEM_PROMPT, KO_CONLL_EXAMPLE_TEMPLATE, KO_CONLL_QUERY_TEMPLATE),
        ("ko", "dict"): (KO_SYSTEM_PROMPT, KO_DICT_EXAMPLE_TEMPLATE, KO_DICT_QUERY_TEMPLATE),
    }
    key = (language.lower(), output_format.lower())
    if key not in templates:
        raise ValueError(f"Unknown language/format combination: {key}")
    return templates[key]
