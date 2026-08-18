# support-triage

Customer Support Ticket Triage + Auto-Draft Agent. Solo student project, Applied
Language Models course.

<!-- TODO: project overview, pipeline diagram, results summary — filled in at the end. -->

## Data

### Source

**Bitext - Customer Service Tagged Training Dataset for LLM-based Virtual Assistants**

- HuggingFace: [`bitext/Bitext-customer-support-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- File used: `Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`
- sha256: `6f81102b0100b97b8468eb04368033a23206bf1fde9d53500d5806ec1001a434`
- (c) Bitext Innovations, 2024

### License

**CDLA-Sharing-1.0** (Community Data License Agreement - Sharing, version 1.0).

The license permits use and redistribution and **requires attribution**, which is what
this section provides. Any redistribution of the data, or of data derived from it, must
carry the same license and the same attribution.

### Contents

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

### This data is hybrid-synthetic, and that matters

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

### What is in git and what is not

Nothing under `data/` is committed except the `.gitkeep` markers and
`data/processed/split_manifest.json`.

- `data/raw/` - downloaded once from HuggingFace (19 MB), gitignored.
- `data/processed/{clean,naive}/*.csv` - regenerated deterministically from the
  frozen seed, gitignored.
- `data/processed/full_corpus.csv` - regenerated with the splits, gitignored. The
  registry of **every** raw row: its template family, the split side that family
  landed on, and role flags (`is_representative`, `is_exact_duplicate`). Stage 1
  never reads it; it exists so stage 2 can build its retrieval corpus from
  train-side families only, and so nothing is silently discarded by the
  one-representative-per-family collapse.
- `data/processed/split_manifest.json` - **committed**. Seed, threshold, row counts
  and a sha256 per split (the full-corpus registry included). It holds no rows, and
  it is what proves a rebuild produced the same split. If the file upstream is ever
  updated, the rebuilt split stops matching these hashes and the assert at the top
  of every notebook fires.

### Rules fixed now for the later stages (stage 2+)

Three rules recorded while the split was being designed, because they are cheap to
state now and expensive to rediscover later:

1. The stage-2 retrieval corpus is built from **train-side families only**
   (`full_corpus.csv`, rows with `split == "train"`). A test-family row in the
   corpus would let the end-to-end evaluation retrieve the answer key.
2. **Filter by intent, rank by text.** Several intents measurably bundle more than
   one user goal under one label - `newsletter_subscription` covers both subscribe
   and unsubscribe, `switch_account` covers both upgrade-tier and switch-user (the
   measurement is in `notebooks/01_data.ipynb`, "subgoal separability"). A reply
   chosen from the intent alone answers the wrong goal for a large minority of
   tickets; retrieval ranked by the ticket's own text recovers the distinction.
3. Responses are joined back from `data/raw/` via `row_id`. No file under
   `data/processed/` contains the `response` column.

### How to rebuild

Run `notebooks/01_data.ipynb` top to bottom. It downloads the raw file if it is
missing, rebuilds all six CSVs and `full_corpus.csv` from seed 42, and finishes by
re-reading them off disk and checking their hashes against `split_manifest.json`.
