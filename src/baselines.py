"""
Reference points for evaluating the task without a language model.

A baseline defines the scale against which later results are judged: a
macro-F1 score is uninformative on its own, but becomes a meaningful claim
when stated against a floor and a non-neural reference model.

Two baselines are implemented here. The third baseline - zero/few-shot
scoring with an untrained Qwen3 - requires a GPU and lives outside this
module.

  1. Majority class. Predicts the most frequent training label regardless
     of input text. Establishes the floor: any model that does not beat it
     performs worse than one that ignores the input entirely.

  2. TF-IDF + logistic regression. A linear classifier over weighted word
     counts, with no pretraining and negligible CPU cost. On a 27-way task
     built from a small number of templates, this is a strong reference
     point, and a fine-tuned model that does not clearly beat it is a
     result worth reporting rather than a failure to hide.

Both baselines also convert the corpus-level leakage measured in
notebooks/01_data.ipynb from a property of the data into a measured
difference in score.

Functions here follow the same conventions as src/data.py: pure functions
that take a table and return a table or an object, with no printing,
plotting, disk I/O, or torch dependency - notebooks/02_baselines.ipynb runs
these functions and reports the results.

Scoring itself lives in src/evaluate.py rather than here, since the
fine-tuned model must later be scored by identical code; a module named
`baselines` is the wrong place to import a metric from.
"""

from __future__ import annotations

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data import SPLIT_SEED

# Token that replaces every {{Placeholder}} in the ablation. Angle brackets
# so it cannot collide with a real word in the corpus.
ENTITY_TOKEN = "<ENT>"

# Matches {{Order Number}}, {{Person Name}}, {{Currency Symbol}} and ~30 more.
# Non-greedy and newline-free so that one runaway `{{` swallows one
# placeholder rather than the rest of the sentence.
_PLACEHOLDER = re.compile(r"\{\{[^{}]*\}\}")


# =========================================================================
# 1. MAJORITY CLASS
# =========================================================================

def majority_label(train: pd.DataFrame, label_column: str = "intent") -> str:
    """Returns the most frequent label in `train`.

    Computed from the training split and applied to validation/test, as any
    other model would be - this is why the function takes a frame rather
    than a hardcoded string. Reading the majority label off the validation
    set would fit the baseline on the same data it is later compared
    against.

    The result differs between split families: after the family collapse,
    the clean and naive sets no longer share a class distribution, so their
    floors must be measured separately rather than assumed to be 1/27.
    """
    counts = train[label_column].value_counts()
    # value_counts sorts by count descending; ties break on first-seen
    # order, which is deterministic per file and is recorded here to avoid
    # an invisible source of run-to-run variation.
    return str(counts.index[0])


# =========================================================================
# 2. TF-IDF + LOGISTIC REGRESSION
# =========================================================================

def build_tfidf_pipeline(analyzer: str = "word",
                         ngram_range: tuple[int, int] = (1, 2),
                         min_df: int = 2,
                         sublinear_tf: bool = True,
                         lowercase: bool = True,
                         class_weight: str | None = None,
                         max_iter: int = 2000,
                         seed: int = SPLIT_SEED) -> Pipeline:
    """Builds a vectoriser + classifier pipeline as a single unfitted object.

    Combining the vectoriser and classifier into one Pipeline is a
    structural safeguard: there is exactly one `fit` call, it receives
    exactly one frame, and it is not possible to accidentally fit the
    vocabulary on validation text. The alternative -
    `vectorizer.fit_transform(all_text)` followed by `clf.fit(train_slice)`
    - is a two-line mistake that raises no error and inflates the score.

    Word-level (1,2) is the default here, while char_wb (3,5) is the space
    in which the near-duplicate families were defined and collapsed (see
    notebooks/01_data.ipynb). Evaluating the clean split in that same space
    would be circular: rows judged similar by char-level TF-IDF were
    removed, then scored with char-level TF-IDF. Word (1,2) is an
    independent feature space and the standard baseline for text
    classification; the char_wb configuration is still run as a separate
    row, since quantifying that bias is more informative than avoiding it.

    `lowercase=True` does not conflict with the minimal-cleaning policy
    documented in the README ("Every cleaning step, and what it buys"),
    which governs what is written to disk. Lowercasing here is applied
    identically to training text and live tickets by the same fitted
    object, so there is no train/serving skew.

    Known limitation, not addressed here: scikit-learn's default word
    token pattern drops punctuation and single characters, so the question
    mark that distinguishes a query from a command is invisible to this
    baseline - one of the properties the char_wb row is measured against.

    `seed` reaches `LogisticRegression.random_state`, which the lbfgs
    solver does not use (lbfgs is deterministic given the data). It is
    passed regardless so that switching to a stochastic solver (e.g. saga)
    cannot silently introduce unrecorded run-to-run variance.
    """
    vectoriser = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        lowercase=lowercase,
    )
    classifier = LogisticRegression(
        solver="lbfgs",
        max_iter=max_iter,   # generous, to avoid a swallowed ConvergenceWarning
                             # understating this baseline's true strength
        class_weight=class_weight,
        random_state=seed,
    )
    return Pipeline([("tfidf", vectoriser), ("clf", classifier)])


def pipeline_converged(pipeline: Pipeline) -> bool:
    """Reports whether lbfgs converged rather than hitting the iteration cap.

    scikit-learn emits a ConvergenceWarning in this case, which is easy to
    miss among other notebook output. A non-converged classifier does not
    raise an error - it silently returns an under-trained model, weakening
    the baseline and flattering any comparison against it. Calling this
    after every `fit` turns the warning into an assertable check.
    """
    return bool((pipeline.named_steps["clf"].n_iter_ < pipeline.named_steps["clf"].max_iter).all())


# =========================================================================
# 3. CONTROLS AND ABLATIONS
# =========================================================================

def subsample_stratified(df: pd.DataFrame, n_rows: int,
                         seed: int = SPLIT_SEED,
                         stratify_column: str = "intent") -> pd.DataFrame:
    """Draws exactly `n_rows`, preserving the class proportions of `df`.

    This is the size control. naive/train has 17,187 rows and clean/train
    has 9,893, so any score difference between them could otherwise be
    attributed to training-set size alone. Drawing 9,893 rows from
    naive/train removes that explanation, leaving one remaining variable:
    whether the training set contains siblings of the test sentences.

    Stratified for the same reason the split itself is stratified: a flat
    random draw can leave a small intent with almost no rows, and macro-F1
    weights that intent as heavily as a large one, which would turn the
    comparison into a measurement of draw luck.

    `train_test_split` is reused rather than reimplemented, since it
    already performs proportional allocation with a fixed seed, and reusing
    it keeps one splitting behaviour in the project instead of two.

    `seed` here is the subsample seed, distinct from the split seed even
    though it holds the same value - which rows get drawn and how the data
    was partitioned are different decisions, and a record naming the wrong
    one would not look wrong. Which rows get drawn is the only stochastic
    element in this baseline, and it is fixed: the draw is made once, at
    seed 42, and written to data/processed/naive_sub/ so that every later
    comparison reads the same rows instead of redrawing them.
    """
    if n_rows >= len(df):
        raise ValueError(
            f"n_rows={n_rows} is not smaller than the frame ({len(df)} rows) - "
            "there is nothing to subsample and the control would be a no-op."
        )
    drawn, _ = train_test_split(
        df,
        train_size=n_rows,
        stratify=df[stratify_column],
        random_state=seed,
    )
    return drawn.copy()


def normalise_placeholders(texts, token: str = ENTITY_TOKEN) -> list[str]:
    """Replaces every {{Placeholder}} with one shared token (the <ENT> ablation).

    The dataset is synthetic and writes entities as {{Order Number}},
    {{Invoice Number}}, {{Person Name}} and similar. A real customer would
    write "#84213". If a placeholder is concentrated in a small number of
    intents - {{Invoice Number}} is close to this - the model can identify
    those intents without reading any other content, i.e. shortcut
    learning that works here and collapses on live traffic.

    Collapsing every placeholder to one token and re-running the identical
    baseline puts a number on this effect. A large drop is evidence of the
    shortcut; no drop weighs against it.

    This removes the placeholder's lexical content only, and words such as
    "invoice" inside {{Invoice Number}} are partly legitimate signal too.
    The measured drop is therefore an upper bound on the shortcut, not a
    clean estimate of it.

    Operates on texts rather than a DataFrame, so it cannot touch a label
    column and cannot be mistaken for a new split. This keeps the ablation
    running on the frozen split with only the vectorised text changed,
    isolating the placeholder effect from any change in row membership.
    """
    return [_PLACEHOLDER.sub(token, str(t)) for t in texts]


def placeholder_stats(texts, intents) -> pd.DataFrame:
    """Reports how concentrated each placeholder is within a small number of intents.

    Supporting evidence for the ablation above: a placeholder appearing in
    one or two intents is a give-away token, while one spread evenly across
    twenty is ordinary vocabulary.

    Returns one row per distinct placeholder: how many rows contain it, how
    many intents it appears in, and the share of its occurrences falling in
    its single most common intent.
    """
    rows = []
    frame = pd.DataFrame({"text": [str(t) for t in texts], "intent": list(intents)})
    frame["found"] = frame["text"].map(lambda t: set(_PLACEHOLDER.findall(t)))

    all_placeholders = sorted({p for s in frame["found"] for p in s})
    for placeholder in all_placeholders:
        hit = frame[frame["found"].map(lambda s: placeholder in s)]
        by_intent = hit["intent"].value_counts()
        rows.append({
            "placeholder": placeholder,
            "n_rows": len(hit),
            "n_intents": int(by_intent.size),
            "top_intent": str(by_intent.index[0]),
            "top_intent_share": round(float(by_intent.iloc[0] / len(hit)), 3),
        })
    return (pd.DataFrame(rows)
            .sort_values(["top_intent_share", "n_rows"], ascending=[False, False])
            .reset_index(drop=True))
