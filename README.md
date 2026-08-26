# support-triage

Customer support ticket triage: a 27-way intent classifier over customer support
messages, fine-tuned with LoRA and measured against a leakage-controlled split.
Team project, Applied Language Models course. This part of the repository —
data pipeline and baselines — is Emma Vainshtein's contribution.

<!-- TODO: project overview, team members, pipeline diagram, results summary — filled in at the end. -->

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

### Every cleaning step, and what it buys

The project is deliberately short of cleaning steps, and each one is measured rather
than assumed. `results/metrics/corpus_cleaning_funnel.csv` holds these numbers: the
rows surviving each stage, with the whitespace counters on the stage they explain.

| step | rows in | rows out | what it buys |
|---|---|---|---|
| whitespace normalisation | 26,872 | 26,872 | Removes no rows at all. It rewrites 551 texts, and that lets **81 additional** `(instruction, intent)` pairs be recognised as exact duplicates (2,318 against 2,237 without it). Those pairs differ only by a double space; without this step they are two rows that can land on opposite sides of the split. |
| exact de-duplication | 26,872 | 24,554 | Removes 2,318 rows. The subset matters: `drop_duplicates()` with its default arguments compares `flags` and `response` too and removes **zero** rows here. |
| family clustering @ 0.90 | 24,554 | 14,133 | Collapses near-duplicate template families to one representative each. This is what makes the split clean, and its cost is measured — see the ablation below. |

Nothing else is done. No lowercasing, no punctuation stripping, no stopword removal,
no stemming, and no typo correction — the deliberate typos are 19.7% of the corpus and
they are the single largest error source for every baseline, which is exactly why they
are kept.

Two smaller points, both verified against the file rather than the documentation:

- The dataset card documents 12 generation tags. The file contains 14 - `S` (80 rows
  after de-duplication, only in `delivery_options`) and `V` (77 rows) are documented
  nowhere.
- The dataset card states "27 intents assigned to 10 categories" and lists 21 intents.
  The file has 27 intents under 11 categories. This is why `artifacts/intent2cat.json`
  is built from the DataFrame and validated, never typed in from the documentation.

### What is in git and what is not

Everything under `data/processed/` is committed - the six split CSVs, the corpus
registry and the manifest, 6 MB in total. The raw file is not.

- `data/raw/` - fetched from HuggingFace on demand and not kept on disk (19 MB,
  gitignored). `load_raw()` re-downloads it when it is missing, `verify_raw()`
  raises if the shape moved, and the manifest stores its sha256 - so the integrity
  guarantee lives in the manifest rather than in a local copy of the file.
- `data/processed/{clean,naive}/*.csv` - **committed** (3.4 MB). They are
  regenerated deterministically from the frozen seed, so in principle the repo
  could hold the seed alone - but only on a machine with the pinned versions
  installed, and a dataset a reader cannot open is not a reproducible one. 3.4 MB
  buys a clone that runs.
- `data/processed/full_corpus.csv` - **committed** (2.8 MB), regenerated with the
  splits. The registry of **every** raw row: its template family, the split side
  that family landed on, and role flags (`is_representative`,
  `is_exact_duplicate`). Nothing trains on it. It exists so that the
  one-representative-per-family collapse discards nothing silently — every one
  of the 26,872 raw rows keeps a family and a split side — and it is what the
  uncollapsed alternative in "What the family collapse costs" below is built
  from, without needing a second split.
- `data/processed/split_manifest.json` - **committed**. Seed, threshold, row counts
  and a sha256 per split (the full-corpus registry included). It holds no rows, and
  it is what proves a rebuild produced the same split. If the file upstream is ever
  updated, the rebuilt split stops matching these hashes and the assert at the top
  of every notebook fires.

`results/` **is** committed, and is kept to one file per finding. Each table is
keyed by one thing — a threshold, an intent, a run — and nothing is written twice:
the per-run detail of all 18 baseline runs lives in `baselines_summary.csv` rather
than in a JSON file per run beside it. Frames of individual rows are not written at
all. Where reading the rows is the point, the notebook prints worked examples
inline and saves a small curated set - `corpus_cross_intent_examples.csv` holds
seven rows, one per conflicting intent pair, rather than the 203 rows behind them.
File names say what a table contains, not which day it was produced.

### Two properties of the committed split

Both are guarantees rather than conventions — they hold by construction, so no
later step has to remember them:

1. **A family never straddles the split.** Every member of a template family
   inherits the side its representative landed on, so any view of
   `full_corpus.csv` filtered by `split` is leak-free: it cannot contain a
   sibling of a sentence held out on the other side. That is what lets
   `train_side_rows()` build the uncollapsed training set out of the frozen
   split instead of a new one.
2. **No file under `data/processed/` contains the `response` column.** The
   response is the templated answer, and a model that can see it reads the
   label off its own input and scores ~0.999. Leaving it out of the files makes
   that mistake impossible rather than merely discouraged; the same argument
   applies to `flags`, which stays only because error slicing needs it and
   never reaches the feature side.

### How to rebuild

```
python -m pip install -r requirements.txt
```

The versions in `requirements.txt` are pinned rather than ranged, and that is
deliberate. The split is rebuilt from a seed and then checked against the sha256
fingerprints in `split_manifest.json`; a different scikit-learn can allocate a
stratified split differently, which moves every hash and makes
`verify_against_manifest()` raise. That assert is working as designed, but it is
easier to install the versions that produced the committed hashes than to debug it.

The split is already in the repo, so `02_baselines.ipynb` and `03_train.ipynb` run
on a fresh clone without rebuilding anything. Rebuilding is a check, not a
prerequisite: run `notebooks/01_data.ipynb` top to bottom. It downloads the raw file if it is
missing, rebuilds all six CSVs and `full_corpus.csv` from seed 42, and finishes by
re-reading them off disk and checking their hashes against `split_manifest.json`.

Then `notebooks/02_baselines.ipynb`, which verifies the same manifest before producing
any number and writes every run to `results/metrics/`. Both notebooks are CPU-only and
take roughly five to eight minutes together, most of it in the two exhaustive
all-pairs similarity scans.

Re-running `01_data.ipynb` end to end reproduces `split_manifest.json` byte for byte,
which is the check worth doing if you want to confirm the pipeline rather than trust
it.

## Baselines

Reference points measured before any fine-tuning, so that later scores have a scale.
All numbers are on `val`; the test set is opened once, at the end. Full table in
`results/metrics/baselines_summary.csv`.

| model | trained on | rows | scored on | accuracy | macro-F1 |
|---|---|---|---|---|---|
| majority class | `clean/train` | 9,893 | `clean/val` | 0.0580 | 0.0041 |
| majority class (11 categories) | `clean/train` | 9,893 | `clean/val` | 0.2528 | 0.0367 |
| TF-IDF word(1,2) + LogReg | `clean/train` | 9,893 | `clean/val` | 0.9835 | **0.9787** |
| TF-IDF word(1,2) + LogReg | `naive/train` | 17,187 | `naive/val` | 0.9886 | 0.9881 |
| TF-IDF `char_wb`(3,5) + LogReg | `clean/train` | 9,893 | `clean/val` | 0.9948 | 0.9928 |
| TF-IDF word(1,2), `class_weight="balanced"` | `clean/train` | 9,893 | `clean/val` | 0.9873 | 0.9842 |
| TF-IDF word(1,2) *(collapse ablation)* | train-side families, all members | 17,031 | `clean/val` | 0.9858 | 0.9818 |

Six things these numbers establish:

1. **Why macro-F1.** A model that reads nothing scores accuracy 0.253 on the 11
   categories but macro-F1 0.037 - a 0.216 gap, because `ACCOUNT` alone bundles six
   intents. On the 27 intents the floor is 0.058 accuracy, not the 1/27 = 0.037 that
   balanced classes would give, because the family collapse left them unbalanced.

2. **What the leakage is worth.** Day 1 measured that ~47% of committed naive test
   rows sit within 0.10 cosine of a training row. Converting that to a score requires
   controlling for two confounds at once - training-set size and test-set identity -
   so the decisive comparison trains two models on 9,893 rows each and scores both on
   the *same* 2,120 `clean/val` rows, where one training set contains ~40% of them
   verbatim and the other 0%. The gap is **+0.0040 macro-F1** against a subsample-draw
   noise floor of 0.0007: real, statistically separable, and much smaller than the
   52%-of-rows headline suggests. Both halves of that sentence matter.

3. **How much headroom is left, stated against the strongest cheap model rather than
   the most flattering one.** The headline baseline is `word(1,2)` at 0.9787, chosen
   because it is a feature space independent of the `char_wb` space the families were
   built in. But the strongest model here that needs no pretraining at all is
   `char_wb`(3,5) at **0.9928**, so the honest headroom for a fine-tuned Qwen3 is
   **0.0072**, not the 0.021 that quoting the weaker baseline would give. Character
   n-grams absorb the deliberate typos that a word-level model cannot see; that is a
   real advantage of the feature space, not an artefact.

   A tempting further claim is **not** made here, and the reason is worth recording.
   Section 1.4 of `notebooks/01_data.ipynb` ("The ambiguity ceiling, measured
   exhaustively") measures that **0.83%** of
   corpus rows have a near-identical twin carrying a *different* intent (73% of them
   `check_invoice` against `get_invoice`), which looks like an accuracy ceiling of
   about 0.9917. It is not one. Only 6 of the 2,120 `clean/val` rows have such a twin
   at all — the family collapse removes most of them, since they cluster inside the big
   invoice families — and **both baselines classify all 6 correctly**. A row with a
   near-identical neighbour under another label is not unclassifiable: the single token
   that differs, *see* against *get*, is exactly what a bag-of-words model keys on.

   What the 0.83% does measure is how finely the **taxonomy** divides intents, and it
   has a mirror image that belongs beside it. The subgoal-separability measurement in
   `notebooks/01_data.ipynb` shows that several intents bundle more than one user goal
   under a single label — `newsletter_subscription` covers both subscribe and
   unsubscribe, `switch_account` covers both upgrade-tier and switch-user — and that
   the customer's own wording reliably predicts which one is meant, while the label
   throws that distinction away.

   So the label scheme is fine where the *wording* is and coarse where the *goal* is.
   A correct intent does not by itself say what the customer wanted, which is a real
   limitation of intent classification on this taxonomy and a caution about paraphrase
   robustness. Neither is a bound on the score.

4. **Where the remaining errors are.** Typo-flagged rows (`Z`) score 0.955 against
   0.995 for the rest - a 0.041 gap, the largest of any slice, and the one place a
   pretrained model has obvious room to help.

5. **What the family collapse costs.** Keeping one representative per family discards
   42% of the corpus, so the obvious objection is that the score is paying for it. It
   is, and the amount is measured. Keeping *every* member of every train-side family
   instead - which has the identical leakage guarantee, because a family never
   straddles the split - gives 17,031 training rows and **0.9818** macro-F1 against the
   committed design's 0.9787, so the collapse costs **+0.0031**. It also improves class
   balance from 4.06:1 to 2.62:1.

   This is reported as an ablation, not adopted: the committed split stays as it is.
   Two caveats belong with the number. There is no error bar on it and there cannot be
   a meaningful one - both models are deterministic, so the only variance that would
   matter is the split seed, and the split seed is frozen. And the uncollapsed set is
   not strictly better: it reverts the training set to the corpus flag mix (19.3% typo
   rows against 27.8%), because a mangled sentence survives collapse as its own
   singleton family while thirty polite paraphrases become one row. The collapse buys a
   training set enriched for hard rows, and this is what that enrichment costs.

6. **The leakage numbers are all in one vector space, and it is named.** Every
   similarity figure in this repository is computed in the `char_wb`(3,5) space fitted
   on the 24,554 deduplicated rows (`src.data.build_canonical_corpus`). This matters
   because an earlier version of the similarity profile re-fitted the vectoriser on
   `train + val` alone: `min_df=2` over 12,013 rows keeps a smaller vocabulary, every
   idf weight shifts, and the >= 0.90 tail inflated from 0.09% to 1.70% for a split
   that had not changed at all. Two numbers for one quantity, in units nothing
   declared. Quoting a similarity figure now requires saying which corpus it was
   fitted on.

### Is the clean split clean, or only clean above 0.90?

**The honest answer is that in the space the families were built in, the question
cannot be answered.** Families are the connected components of the >= 0.90 similarity
graph *within* one intent, and a whole family always lands on one side of the split.
So a held-out row and a training row that share an intent and sit at >= 0.90 would have
to be one family on two sides, which cannot happen. The measured maximum is
**0.899984** against a threshold of 0.90, with nothing above it: that is a constraint,
not a result.

So the same measurement is repeated in `word(1,2)`, a feature space that had no part in
building the split and where the answer was free to be non-zero
(`results/metrics/corpus_residual_leakage.csv`):

| space | relation | >= 0.95 | >= 0.90 | median | max |
|---|---|---|---|---|---|
| `char_wb`(3,5) — the clustering space | same intent | 0.00% | **0.00%** (by construction) | 0.807 | 0.899984 |
| `char_wb`(3,5) | any intent | 0.05% | 0.24% | 0.809 | 0.961 |
| `word(1,2)` — independent | same intent | 0.38% | **1.79%** | 0.692 | 1.000 |
| `word(1,2)` | any intent | 0.38% | 1.79% | 0.695 | 1.000 |

The 1.79% is the number worth quoting, and reading the pairs behind it — printed in
section 1.4 of `notebooks/01_data.ipynb`, under "What are those word-space pairs,
actually?" — shows it is not sibling leakage. Seven
`clean/test` rows are *word-identical* to a training row; three differ only in
punctuation or casing, and four differ only in a rare typo token that `min_df=2` prunes
out of the vocabulary. In character n-grams those same pairs score 0.275 to 0.898 —
every one below the clustering threshold, which is why the families were right to keep
them apart. The residual is a property of the measurement space, not of the split.

A second-order consequence belongs in the limitations: on those rows the word-level
baseline is not merely uncertain, it is **blind** — two differently-worded sentences
share one feature vector. That is part of why `char_wb` scores higher here.

### The caveat this project states rather than hides

The split is clean of sibling leakage. It is still a template-generated corpus, and the
*median* `clean/val` row sits at 0.808 cosine from its nearest training row. Accuracy
broken down by that similarity (`results/metrics/baseline_clean_similarity_profile.csv`):

| nearest-train similarity | rows | accuracy |
|---|---|---|
| < 0.6 | 51 | **0.882** |
| 0.6 - 0.7 | 257 | 0.949 |
| 0.7 - 0.8 | 667 | 0.984 |
| 0.8 - 0.9 | 1,143 | 0.996 |
| >= 0.9 | 2 | 1.000 |

Accuracy rises monotonically with similarity to the training set. The headline 0.9835
is an average over that gradient: on the 51 rows furthest from anything seen in
training the same model scores **0.882**, about 0.10 below its headline. De-duplication
removes sibling leakage; it does not turn a template-generated corpus into a sample of
real customer language, and no threshold could.

## Fine-tuning

`notebooks/03_train.ipynb` is the only notebook that needs a GPU, and `src/train.py` is
the only module that imports torch. Everything below is method; the numbers land in
`results/metrics/gpu_runs_summary.csv` and are quoted here once they exist.

### The model

`Qwen3-1.7B` with a 27-way classification head (`AutoModelForSequenceClassification`,
`task_type=SEQ_CLS`), adapted with LoRA. The **instruct** variant, not `-Base`: the
untrained baseline below has to be measured on the same weights the adapter sits on, or
the before/after difference mixes a model swap into a fine-tune. `Qwen3-0.6B` is used for
the debug run only and none of its numbers are reported.

LoRA on `q_proj` and `v_proj`, `dropout=0.05`, and the head in `modules_to_save`. The
sweep is over **`r` alone**, with `alpha = 2r` tied to it — with alpha held fixed instead,
changing `r` would change both the adapter's capacity and how strongly its output is
scaled, and a difference between two rows could not be attributed to either. Every other
hyper-parameter is a module-level constant in `src/train.py` rather than a notebook cell,
so that a mid-day nudge appears as a diff in git instead of as an invisible difference
between two runs.

`MAX_LENGTH = 32`, measured on day 1 rather than chosen: p99 is 19 tokens and the longest
row is 24. The naive default of 512 would cost roughly 250x the attention for identical
predictions, and a slip to 64 is 4x slower with no error and no warning.

### The three baselines are measured before any of it

Majority class and TF-IDF were measured on CPU (above). The third needs a GPU: **Qwen3-1.7B
with no adapter and no training at all**, zero-shot and with one demonstration per intent.

It is not measured through the classification head. That head is initialised at random, so a
score taken through it would be a score of the random numbers in it — roughly guessing, and
it would inflate the apparent improvement for free. Free generation is the other option, and
it needs mapping rules for outputs matching none of the 27 labels; those rules are themselves
an arbitrary choice that moves the number. So the measurement is **label scoring**: for each
of the 27 intents, how likely the model finds that label as the continuation of the prompt,
highest wins. Always a legal label, no mapping rules, and the untrained head is never touched.

Two things are fixed before the run rather than after:

- **The demonstrations come from `clean/train` only**, one per intent, drawn at random with a
  recorded seed. An example taken from `clean/val` puts a validation sentence inside the
  prompt used to score `clean/val`, and nothing external would show it.
- **Length-normalised scoring is the headline.** Labels are 1 to about 7 tokens long
  (`review` against `set_up_shipping_address`) and a plain sum of log-probabilities is
  systematically biased towards short ones — a property of the scoring rule, not of the model.
  The unnormalised sum comes free from the same forward pass and is recorded next to it.

The prompt is a hyper-parameter even though it does not look like one: reword it and the
number moves. The exact text is written to `results/prompts/` and its sha256 goes into every
run record, so a change is visible in the table rather than being an unexplained shift.

### Two silent failures, and the one check that catches both

A wrong `task_type`, and a classification head left out of `modules_to_save`. Both train
successfully, both produce a falling loss, and both are only visible when what was written to
disk is read back in a fresh process — the head comes back randomly initialised and the score
collapses with no error message anywhere. "No exception was raised" does not test this.

The check that does is a **save, load from scratch, predict round trip** with an exact
comparison of the predicted labels, and it is why the debug run exists at all: it costs ten
minutes on a 0.6B model and an hour of wasted training on a 1.7B one. Predicted labels are
compared rather than logits, because half-precision arithmetic is not bit-reproducible across
two loads and demanding identical floats would fail for a reason unrelated to what is being
tested.

### What is in git and what is not

Adapters go to Drive, one directory per run; git gets the run's JSON record. An adapter is a
training output, not code.

The day-2 rule — no per-run JSON beside the summary table, because every field in it was
already a column — is kept, and its *reason* is what makes the GPU runs an exception. A
training run carries two things no summary table can hold, the per-step loss curve and the
per-epoch validation score, and re-creating them costs compute units rather than a second. So
`results/metrics/run_XX.json` is the archive and `gpu_runs_summary.csv` is a view generated
from it. Run numbers are never reused, including for runs that failed: a run that disappeared
is a run that will be run again.
