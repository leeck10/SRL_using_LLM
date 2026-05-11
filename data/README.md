# Data

Due to licensing restrictions, only 10 randomly sampled sentences per split are included.
These samples illustrate the data format; **full datasets must be obtained separately**.

## English: CoNLL 2009 Shared Task

- **Source**: [CoNLL-2009 Shared Task](https://ufal.mff.cuni.cz/conll2009-st/)
- **License**: LDC (Linguistic Data Consortium)
- **How to obtain**: Contact LDC or the CoNLL-2009 organizers for the English SRL data (PropBank/NomBank annotations on WSJ).

## Korean: Korean PropBank

- **Source**: Korean PropBank
- **License**: Contact the dataset authors for licensing information.
- **How to obtain**: Request from the Korean PropBank project maintainers.

## Data Format

Each sentence block consists of:

```
<raw sentence text>
<tab-separated verb sense IDs (_ for non-predicate positions)>
<word>\t<predicate marker or _>\t<SRL label for verb1>\t<SRL label for verb2>\t...
<word>\t<predicate marker or _>\t<SRL label for verb1>\t<SRL label for verb2>\t...
...
<blank line>
```

### Labels

- `ARG0`-`ARG5`: Core argument roles (defined per verb frame)
- `ARGM-*`: Adjunct argument roles (e.g., `ARGM-TMP`, `ARGM-LOC`)
- `AUX`: Auxiliary verb (Korean only)
- `_`: No role assigned

## Frame Files

Frame files (`sample_frames.json`) define the core semantic roles for each verb sense.
Full frame files are also subject to licensing restrictions.

## Sample Data Sizes

Each language includes 10 sentence blocks per split (randomly sampled):

| Language | Train | Dev | Test |
|----------|-------|-----|------|
| English (ICL) | 10 | 10 | 10 |
| English (CRF) | 10 | 10 | 10 |
| Korean (ICL) | 10 | 10 | 10 |
| Korean (CRF) | 10 | 10 | 10 |
