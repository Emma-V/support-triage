"""
The reference points. How good is this task without any language model at all?

A baseline is not a failed attempt - it is the unit of measurement for
everything that comes after it. "macro-F1 0.91" is a number with no scale;
"0.91 against a floor of 0.06 and a TF-IDF baseline of 0.83" is a claim.

Two baselines live here. The third one named in CLAUDE.md section 4 - the
zero/few-shot untrained Qwen3 - needs a GPU and is therefore not in this file.

  1. Majority class. Always answer the most common training label, without
     reading the text. The absolute floor: any model that does not beat it is
     worse than a model that cannot read.

  2. TF-IDF + logistic regression. Weighted word counting plus a linear
     classifier. No pretraining, no language understanding, seconds on CPU.
     On a 27-way task built from a few dozen templates this is a serious
     opponent, and if the fine-tuned model does not clearly beat it, that is
     the finding - not a failure to hide.

Both of them are also how the leakage measured on day 1 stops being a property
of the DATA and becomes a difference in SCORE.

Same house rules as src/data.py: pure functions, a table goes in and a table or
an object comes out. No print, no plots, no writing to disk, and no torch -
notebooks/02_baselines.ipynb is what runs these and shows the results.

Scoring lives in src/evaluate.py, not here, because tomorrow's fine-tuned model
must be scored by exactly the same code. A module named `baselines` is the
wrong place to import a metric from.
"""

from __future__ import annotations

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data import SPLIT_SEED

# The token that replaces every {{Placeholder}} in the ablation. Angle brackets
# so it cannot collide with a real word in the corpus.
ENTITY_TOKEN = "<ENT>"

# Matches {{Order Number}}, {{Person Name}}, {{Currency Symbol}} and ~30 more.
# Non-greedy and newline-free on purpose: one runaway `{{` should swallow one
# placeholder, not the rest of the sentence.
_PLACEHOLDER = re.compile(r"\{\{[^{}]*\}\}")


# =========================================================================
# 1. MAJORITY CLASS
# =========================================================================

def majority_label(train: pd.DataFrame, label_column: str = "intent") -> str:
    """The most frequent label in TRAIN. The model that does not read.

    Learned from train and then applied to val, exactly like any other model -
    which is the whole reason this is a function taking a frame rather than a
    hardcoded string. Reading the majority label off the validation set would
    be fitting on the test data to build the baseline that the test data is
    then compared against.

    Note that the answer differs per split family here: after the family
    collapse the clean and naive sets no longer have the same class
    distribution, so their floors are two different numbers and must be
    measured separately rather than assumed to be 1/27.
    """
    counts = train[label_column].value_counts()
    # value_counts sorts by count descending; ties break on first-seen order,
    # which is deterministic for a given file. Recorded here because a tie
    # would otherwise be an invisible source of run-to-run difference.
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
    """Vectoriser + classifier as ONE object. Returns an unfitted Pipeline.

    Why a Pipeline and not two separate objects: this is structural protection,
    not tidiness. When the vectoriser and the classifier are a single estimator
    there is exactly one `fit`, it receives exactly one frame, and it is
    physically impossible to accidentally fit the vocabulary on validation
    text. The alternative - vectorizer.fit_transform(all_text) followed by
    clf.fit(train_slice) - is a two-line mistake that raises nothing and lifts
    the score.

    Why word-level (1,2) by default, when day 1 used char_wb (3,5):
    char_wb (3,5) is the space in which the near-duplicate FAMILIES were
    defined and collapsed. Measuring the clean split with that same space is
    circular - "I removed everything char-TF-IDF calls similar, then measured
    how well char-TF-IDF does on the remainder". Word (1,2) is an independent
    feature space, and it is also the standard text-classification baseline a
    reader expects. The char_wb configuration is still run as a separate row,
    because quantifying that bias is more convincing than avoiding it.

    On `lowercase=True`: this does not contradict the "minimal cleaning" rule
    in CLAUDE.md section 4. That rule is about what is written to disk, and its
    real content is that train and inference must be treated identically.
    Lowercasing inside the pipeline is applied to training text and to live
    tickets by the same fitted object, so there is no train/serving skew.

    Known limitation, deliberately not fixed: scikit-learn's default word
    token pattern drops punctuation and single characters, so the question mark
    that distinguishes a query from a command is invisible to this baseline.
    That is one of the things the char_wb row measures.

    `seed` reaches LogisticRegression's random_state, which the lbfgs solver
    does not use - lbfgs is deterministic given the data. It is passed anyway
    so that swapping in a stochastic solver (saga) cannot silently introduce
    unrecorded run-to-run variance.
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
        max_iter=max_iter,   # generous: a swallowed ConvergenceWarning reads as
                             # a genuinely weak baseline, which flatters the
                             # model it is compared against
        class_weight=class_weight,
        random_state=seed,
    )
    return Pipeline([("tfidf", vectoriser), ("clf", classifier)])


def pipeline_converged(pipeline: Pipeline) -> bool:
    """Did lbfgs actually finish, or did it hit the iteration cap?

    scikit-learn emits ConvergenceWarning for this, and a warning in a notebook
    scrolls past behind a wall of output. A non-converged classifier does not
    error - it returns an under-trained model, so the baseline looks weaker
    than it is and every later comparison is flattered by the difference.
    Calling this after every fit turns that into an assert.
    """
    return bool((pipeline.named_steps["clf"].n_iter_ < pipeline.named_steps["clf"].max_iter).all())


# =========================================================================
# 3. CONTROLS AND ABLATIONS
# =========================================================================

def subsample_stratified(df: pd.DataFrame, n_rows: int,
                         seed: int = SPLIT_SEED,
                         stratify_column: str = "intent") -> pd.DataFrame:
    """Draw exactly n_rows, keeping the class proportions of the input.

    This is the size control. naive/train has 17,187 rows and clean/train has
    9,893, so any score difference between them can be dismissed with "you
    simply had less data". Drawing 9,893 rows from naive/train removes that
    explanation and leaves one variable: whether the training set contains
    siblings of the test sentences.

    Stratified for the same reason the split itself is stratified - a flat
    random draw can leave a small intent with almost no rows, and macro-F1
    weights that intent exactly as heavily as a large one, so the comparison
    would then be measuring the luck of the draw.

    train_test_split is reused rather than reimplemented: it already does
    proportional allocation with a fixed seed, and using the same mechanism as
    split_stratified() keeps one behaviour in the project instead of two.

    `seed` here is the SUBSAMPLE seed, and it is not the split seed even though
    it defaults to the same value. Which rows get drawn is the only stochastic
    element in this entire baseline, so it is the only thing there is a noise
    floor to measure - hence three draws (42/43/44) in the notebook.
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
    """Replace every {{Placeholder}} with one shared token. The <ENT> ablation.

    The dataset is synthetic and writes entities as {{Order Number}},
    {{Invoice Number}}, {{Person Name}} and so on. A real customer writes
    "#84213". If a placeholder is concentrated in a couple of intents - and
    {{Invoice Number}} very nearly is - the model can identify those intents
    without reading a single real word. That is shortcut learning: it works
    here and collapses on live traffic.

    Collapsing every placeholder to one token and re-running the identical
    baseline puts a number on it. A large drop is measured evidence of the
    shortcut; no drop refutes the concern, and both are worth a paragraph.

    Read the result carefully, though: this removes the placeholder's LEXICAL
    content, and words like "invoice" inside {{Invoice Number}} are partly
    legitimate signal too. The measured drop is therefore an upper bound on the
    shortcut, not a clean estimate of it.

    Takes and returns TEXTS, never a DataFrame. That is deliberate: it cannot
    touch a label column, and it cannot return something that looks like a new
    split. Decision E6 requires this ablation to run on the frozen split with
    only the vectorised text changed.
    """
    return [_PLACEHOLDER.sub(token, str(t)) for t in texts]


def placeholder_stats(texts, intents) -> pd.DataFrame:
    """How concentrated is each placeholder in a small number of intents?

    Supporting evidence for the ablation, and it answers the "why did you
    expect a shortcut at all" question with a table instead of an intuition.
    A placeholder appearing in one or two intents is a give-away token; one
    spread evenly over twenty is just vocabulary.

    Returns one row per distinct placeholder: how many rows contain it, how
    many intents it touches, and what share of its occurrences fall in its
    single most common intent.
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
