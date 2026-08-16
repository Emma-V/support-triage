# Data

## Source

**Bitext - Customer Service Tagged Training Dataset for LLM-based Virtual Assistants**

- HuggingFace: [`bitext/Bitext-customer-support-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- File used: `Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`
- sha256: `6f81102b0100b97b8468eb04368033a23206bf1fde9d53500d5806ec1001a434`
- (c) Bitext Innovations, 2024

## License

**CDLA-Sharing-1.0** (Community Data License Agreement - Sharing, version 1.0).

The license permits use and redistribution and **requires attribution**, which is what
this file provides. Any redistribution of the data, or of data derived from it, must
carry the same license and the same attribution.

## Contents

| | |
|---|---|
| Rows | 26,872 |
| Intents (the label) | 27 |
| Categories (derived from intent) | 11 |
| Rows per intent | 950-1,000, close to balanced |
| Longest `instruction` | 92 characters (~24 tokens) |

Columns: `flags`, `instruction`, `category`, `intent`, `response`.
Only `instruction` is a model input; only `intent` is a label. See
`notebooks/01_data.ipynb` for what each column is for and why `flags` and
`response` must never reach the feature side.

## This data is hybrid-synthetic, and that matters

The dataset was not collected from real customers. Linguists extracted a few dozen
base templates from natural text, and an NLG engine expanded each template into
dozens of variants: inflections, synonyms, question form instead of request,
politeness, colloquial register, keyword-only phrasing, deliberate typos.

The consequence is that many rows are **siblings of one template rather than
independent examples**. In a plain random split those siblings scatter across train
and test, and the model is scored on sentences it has effectively already seen.
Measured on this data: in a naive split, 10.5% of test rows appear verbatim in the
training set and 52.0% sit at cosine similarity >= 0.90 to a training row.

`data/processed/clean/` is the split that groups siblings into families and keeps
each family on one side of the split. `data/processed/naive/` is the ordinary random
split, kept as the control group.

Two smaller points, both verified against the file rather than the documentation:

- The dataset card documents 12 generation tags. The file contains 14 - `S` (80 rows
  after de-duplication, only in `delivery_options`) and `V` (77 rows) are documented
  nowhere.
- The dataset card states "27 intents assigned to 10 categories" and lists 21 intents.
  The file has 27 intents under 11 categories. This is why `artifacts/intent2cat.json`
  is built from the DataFrame and validated, never typed in from the documentation.

## What is in git and what is not

Nothing under `data/` is committed except this file, the `.gitkeep` markers and
`data/processed/split_manifest.json`.

- `data/raw/` - downloaded once from HuggingFace (19 MB), gitignored.
- `data/processed/{clean,naive}/*.csv` - regenerated deterministically from the
  frozen seed, gitignored.
- `data/processed/split_manifest.json` - **committed**. Seed, threshold, row counts
  and a sha256 per split. It holds no rows, and it is what proves a rebuild produced
  the same split. If the file upstream is ever updated, the rebuilt split stops
  matching these hashes and the assert at the top of every notebook fires.

## How to rebuild

Run `notebooks/01_data.ipynb` top to bottom. It downloads the raw file if it is
missing, rebuilds all six CSVs from seed 42, and finishes by re-reading them off disk
and checking their hashes against `split_manifest.json`.
