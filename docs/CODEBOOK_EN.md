# English Codebook for Korean-Language Content in the SRL-ICL Repository

This codebook provides English explanations for every piece of Korean text that appears in the supplemental source code and data files (`final/` folder) of the manuscript "Enhancing Semantic Role Labeling through Optimized Example Selection and Reordering in In-Context Learning."

The Korean content cannot be removed because it is the linguistic input/output of the **Korean SRL task itself** (Korean PropBank corpus and Korean-language LLM prompts). This codebook is provided so that editors, reviewers, and readers who do not read Korean can fully understand:

1. The Korean dataset samples (`final/data/ko/`).
2. The Korean prompt templates used to query the LLM (`final/prompts/templates.py`).
3. Korean PropBank predicate / frame conventions.
4. The Korean-language README (`final/README_ko.md`) — an English-equivalent README is already provided as `final/README.md`.

All file paths below are relative to the `final/` directory (the supplemental code archive).

---

## 1. Korean Dataset Samples

### 1.1 `data/ko/sample.conll`

This file contains a single Korean sentence in the same CoNLL-style block format used by the full Korean PropBank dataset. The format is documented in `data/README.md` and is identical to the English file `data/en/sample.conll`. Only the surface tokens are Korean.

**Raw file content (Korean):**

```
한국탁구가 2000년 시드니올림픽 본선에 남녀복식 2개조씩을 파견할 수 있게 됐다.
_	_	파견.01
한국탁구가	_	ARG0
2000년	_	_
시드니올림픽	_	_
본선에	_	ARG2
남녀복식	_	_
2개조씩을	_	ARG1
파견할	파견.01	_
수	_	_
있게	_	AUX
됐다.	_	AUX
```

**English-glossed equivalent (for reader understanding only — the file itself is Korean):**

| Korean line          | Word-for-word English gloss                                  | Role assigned |
|----------------------|--------------------------------------------------------------|---------------|
| Sentence (line 1)    | "Korean table-tennis (will) be able to dispatch 2 pairs each in men's-and-women's doubles to the main draw of the 2000 Sydney Olympics." | (raw sentence) |
| `_ _ 파견.01`         | Predicate-sense header line: predicate sense ID is `파견.01` ("dispatch.01") | predicate sense |
| `한국탁구가`           | "Korean table-tennis-NOM"                                    | ARG0 (agent)  |
| `2000년`              | "year 2000"                                                  | (no role)     |
| `시드니올림픽`         | "Sydney Olympics"                                            | (no role)     |
| `본선에`               | "to-the-main-draw"                                           | ARG2 (destination) |
| `남녀복식`             | "men's-and-women's doubles"                                  | (no role)     |
| `2개조씩을`            | "two pairs each-ACC"                                         | ARG1 (entity dispatched) |
| `파견할`               | "dispatch (will)"                                            | (predicate token) |
| `수`                  | dependent noun forming "be-able-to"                          | (no role)     |
| `있게`                | "be-able-to" (auxiliary)                                     | AUX           |
| `됐다.`                | "became" (past tense auxiliary)                              | AUX           |

**Column convention (identical to English file):**

```
<column 1: surface token>
<column 2: predicate sense ID, or "_" if the token is not a predicate>
<column 3: SRL label for verb #1, or "_" if no role assigned>
(additional columns appear only when a sentence contains multiple predicates)
```

### 1.2 `data/ko/sample_frames.json`

Defines core semantic roles for Korean predicate senses. The JSON keys are Korean predicate-sense identifiers in Korean PropBank format `<verb-stem>.<sense-number>`.

**Korean predicate-sense IDs and their English meaning:**

| Frame ID (Korean) | Romanization | English meaning of the verb sense |
|-------------------|--------------|-----------------------------------|
| `파견.01`          | *pagyeon.01* | "to dispatch" (sense 01)          |
| `이기.01`          | *igi.01*     | "to win / defeat" (sense 01)      |
| `확보.01`          | *hwakbo.01*  | "to secure / acquire" (sense 01)  |

The role inventories inside each frame (`agent`, `winner`, `thing secured`, etc.) are already in English and need no translation.

---

## 2. Korean Prompt Templates (`prompts/templates.py`)

The file `prompts/templates.py` defines four template combinations: `{English, Korean} × {CoNLL format, Dict format}`. The English templates (`EN_*`) are self-explanatory. The Korean templates (`KO_*`) contain Korean instructions because they are the actual prompts sent to the LLM when performing **Korean** SRL — they cannot be translated to English without changing the experimental conditions.

Below is a faithful English translation of every Korean variable in `templates.py`. These translations are provided **only for reader comprehension**; the source file itself must remain in Korean to reproduce the experiment.

### 2.1 `KO_SYSTEM_PROMPT`

**Korean (as in source code):**

```
다음은 한국어 의미역 결정의 부가격 의미역 정의이다.
<ARGM-LOC 장소 (locatives)>
<ARGM-DIR 방향 (directional)>
<ARGM-CND 조건 (condition)>
<ARGM-MNR 방법 (manner)>
<ARGM-TMP 시간 (temporal)>
<ARGM-EXT 범위 (extent)>
<ARGM-PRD 보조 서술(secondary predication)>
<ARGM-PRP 목적 (purpose clauses)>
<ARGM-CAU 발생 이유 (cause clauses)>
<ARGM-DIS 담화 연결 (discourse)>
<ARGM-NEG 부정 (negation)>
<ARGM-INS 도구 (instrument)>
<ARGM-ADV 부사어 (adverbial)>
<AUX 보조용언 (Auxiliary Verb)>
```

**Faithful English translation:**

```
The following are the definitions of adjunct semantic roles in Korean SRL.
<ARGM-LOC: locatives>
<ARGM-DIR: directional>
<ARGM-CND: condition>
<ARGM-MNR: manner>
<ARGM-TMP: temporal>
<ARGM-EXT: extent>
<ARGM-PRD: secondary predication>
<ARGM-PRP: purpose clauses>
<ARGM-CAU: cause clauses>
<ARGM-DIS: discourse>
<ARGM-NEG: negation>
<ARGM-INS: instrument>
<ARGM-ADV: adverbial>
<AUX: Auxiliary Verb>
```

Note that the English glosses of every tag are already present in the Korean source between parentheses — only the framing sentence and a few Korean glosses (`장소`, `방향`, `조건`, `방법`, `시간`, `범위`, `목적`, `발생 이유`, `담화 연결`, `부정`, `도구`, `부사어`, `보조용언`) need translation.

### 2.2 `KO_CONLL_EXAMPLE_TEMPLATE` and `KO_CONLL_QUERY_TEMPLATE`

**Korean (as in source code):**

```
Q: 다음 문장
[{sentence}]
에서 동사 [<predicate>{v_org}</predicate>]에 대한 의미역 결정 분석을 해서
conll 포맷으로 출력해줘. 이때 문장 분리를 하지말고 어절 그대로 사용하며,
가질 수 있는 필수격은
{roles}
이다. 이전의 지시사항이나 예시는 반복하지 말고 반드시 의미역 분석 결과만 답변 할 것.
추가설명이나 반복은 하지 말 것.
A: {gold}{eos}
```

**Faithful English translation:**

```
Q: For the following sentence
[{sentence}]
perform semantic role labeling for the verb [<predicate>{v_org}</predicate>] and
output the result in CoNLL format. Do not perform sentence splitting; use the
original word tokens (eojeol) as-is. The available core roles are
{roles}.
Do not repeat the previous instructions or examples; output only the SRL result.
Do not provide additional explanation or repetition.
A: {gold}{eos}
```

The query template (`KO_CONLL_QUERY_TEMPLATE`) is identical except that it ends with `A:` (no gold answer), because it is the test query the LLM must complete.

### 2.3 `KO_DICT_EXAMPLE_TEMPLATE` and `KO_DICT_QUERY_TEMPLATE`

**Korean (as in source code):**

```
Q: 다음 문장
[{sentence}]
에서 동사 [<predicate>{v_org}</predicate>]에 대한 의미역 결정 분석을 해줘.
이때 문장 분리를 하지말고 어절 그대로 사용하며,
가질 수 있는 필수격은
{roles}
이다. 이전의 지시사항이나 예시는 반복하지 말고 반드시 의미역 분석 결과만 답변 할 것.
추가설명이나 반복은 하지 말 것.
A: {gold}{eos}
```

**Faithful English translation:**

```
Q: For the following sentence
[{sentence}]
perform semantic role labeling for the verb [<predicate>{v_org}</predicate>].
Do not perform sentence splitting; use the original word tokens (eojeol) as-is.
The available core roles are
{roles}.
Do not repeat the previous instructions or examples; output only the SRL result.
Do not provide additional explanation or repetition.
A: {gold}{eos}
```

The only structural difference from the CoNLL version is that this template asks for a **dictionary-style** output (no fixed CoNLL grid), as described in the manuscript Materials and Methods section.

### 2.4 Format-string placeholders

The placeholders in both Korean and English templates have identical meaning:

| Placeholder    | Meaning                                                                      |
|----------------|------------------------------------------------------------------------------|
| `{sentence}`   | Raw input sentence (Korean for KO templates, English for EN templates).      |
| `{v_org}`      | Surface form of the predicate token in the sentence.                         |
| `{roles}`      | Core argument roles available for the predicate sense (from frame file).     |
| `{gold}`       | Gold-standard SRL answer (only present in `EXAMPLE_TEMPLATE`, not in queries). |
| `{eos}`        | End-of-sequence token specific to the LLM's tokenizer.                       |

---

## 3. Korean PropBank Conventions

The Korean SRL data used in this study follows **Korean PropBank** (Palmer et al., 2006; LDC2006T03). Key conventions a non-Korean reader needs to know:

- **Predicate-sense IDs** combine a Korean verb stem with a two-digit sense number, e.g. `파견.01` ("dispatch", sense 01). These IDs are atomic strings — no further decomposition is performed by the code.
- **Eojeol (어절)**: the "word" unit in Korean is a whitespace-delimited eojeol that often bundles a content morpheme with one or more functional morphemes (postpositions, endings). Throughout the prompt templates the instruction "어절 그대로 사용하며" tells the LLM not to perform sub-eojeol tokenization. The English translation in §2 above renders this as "use the original word tokens (eojeol) as-is."
- **Argument labels** (`ARG0`, `ARG1`, `ARG2`, `ARGM-*`, `AUX`) are written in ASCII and identical in form to English PropBank. Only the verb-sense IDs and the surface tokens are Korean.

---

## 4. Korean README (`README_ko.md`)

`final/README_ko.md` is a Korean-language version of `final/README.md`. The two files are content-equivalent; the English `README.md` is the canonical reference for editors and reviewers. `README_ko.md` is provided only as a convenience for Korean-speaking readers and contains no information that is absent from the English README.

---

## 5. Summary Table — All Korean Strings in the Repository

| File path                       | Korean content present                          | Why it must remain Korean                                  | Translation provided in this codebook |
|---------------------------------|-------------------------------------------------|------------------------------------------------------------|---------------------------------------|
| `data/ko/sample.conll`          | One Korean sentence + Korean tokens + `파견.01` | Sample of the Korean PropBank corpus (the task input).     | §1.1                                  |
| `data/ko/sample_frames.json`    | Korean predicate-sense keys (`파견.01`, `이기.01`, `확보.01`) | Korean PropBank frame identifiers (atomic IDs).            | §1.2                                  |
| `prompts/templates.py`          | `KO_SYSTEM_PROMPT`, `KO_CONLL_EXAMPLE_TEMPLATE`, `KO_CONLL_QUERY_TEMPLATE`, `KO_DICT_EXAMPLE_TEMPLATE`, `KO_DICT_QUERY_TEMPLATE` | These are the literal LLM prompts for the Korean SRL experiment; translating them would change the experimental condition. | §2.1 – §2.3                           |
| `README_ko.md`                  | Full Korean README                               | Convenience file for Korean readers; equivalent English `README.md` already exists. | §4                                    |

No other file in the `final/` repository contains Korean text. All code (`*.py`), configuration files (`*.yaml`), shell scripts (`*.sh`), and the English README/data are written in English.
