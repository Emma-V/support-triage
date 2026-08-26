"""
Evaluation: how well a trained model performs, measured consistently.

Scope of this module:
- Scores. The main metric is macro-F1: computed separately for each of the
  27 intents and then averaged, so a small intent the model fails on
  completely is not hidden by the large ones.
- A classification report, a table with a score per intent.
- A confusion matrix: which intents the model mixes up with each other,
  for example get_invoice and check_invoice, where the difference is one
  verb.
- Error slices: whether the model fails more on rows with typos, or on
  rows that are keyword-only rather than full sentences, using the
  flags column the dataset provides.

The plots and tables in the report are produced from this file.

--------------------------------------------------------------------------
Every score in this project is computed by this module - the majority-
class baseline, TF-IDF, and the fine-tuned Qwen3 alike. That is the reason
it is a separate file: two runs scored by two different implementations
are not comparable, however similar the implementations look, and the
difference usually turns out to be something like an implicit label list.

Nothing here knows what produced the predictions. Everything takes plain
arrays of labels, so it works identically for a model that reads the text
and one that does not.

No torch, no plots and no disk writes in this file either - it returns
dictionaries and tables, and the notebook decides what to draw and save.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import datetime as _dt
import platform

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# Bumped only when the shape of a run record changes, so old JSON records
# can be checked for compatibility with the current summary code.
RUN_RECORD_SCHEMA = 1


# =========================================================================
# 1. SCORES
# =========================================================================

def evaluate_predictions(y_true, y_pred, labels: list[str]) -> dict:
    """Computes the scores for one run. `labels` is required, and that is deliberate.

    Without an explicit label list, scikit-learn averages macro-F1 over
    whichever classes happen to appear in y_true or y_pred. If the model
    never predicts a rare intent and that intent is thin in the slice being
    measured, the average silently runs over 26 classes instead of 27. The
    score increases with no warning printed, and two runs averaged over
    different denominators are then compared as if they were the same
    measurement.

    Passing all 27 labels from artifacts/labels.json is what keeps the
    number comparable across runs, splits, and models. An intent the model
    never gets right then contributes a genuine 0.0 instead of dropping
    out of the average.

    Returns accuracy, macro-F1, weighted-F1, and a per-class breakdown.

    Both macro and weighted are reported because macro treats every intent
    equally and is the decisive metric (a small intent the model always
    misses drags it down), while weighted scales each intent by its row
    count. The gap between them measures the size of the imbalance
    problem, so reporting one without the other discards information. The
    raw dataset is near-balanced, but the clean training set is not - the
    family collapse leaves a 4.06:1 ratio - so the two numbers diverge at
    the intent level here, not only at the category level.
    """
    labels = list(labels)
    if len(labels) != len(set(labels)):
        raise ValueError("`labels` contains duplicates - the per-class table would be wrong")

    unseen = sorted(set(map(str, y_true)) - set(labels))
    if unseen:
        raise ValueError(
            f"These true labels are missing from `labels`: {unseen}. They would be "
            "dropped from every average without any warning."
        )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )

    return {
        "n_rows": int(len(y_true)),
        "n_labels": len(labels),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, labels=labels,
                                         average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, labels=labels,
                                            average="weighted", zero_division=0)), 4),
        "per_class": {
            label: {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(f), 4),
                "support": int(s),
            }
            for label, p, r, f, s in zip(labels, precision, recall, f1, support)
        },
    }


def per_class_frame(metrics: dict) -> pd.DataFrame:
    """Reformats the per-class block of evaluate_predictions() as a table, worst first.

    This table answers "which intents are actually hard", a more useful
    question than "what is the average". Sorted ascending by F1 so the
    weakest intents appear first, and written to CSV so later models can be
    compared to it intent by intent rather than on one aggregate number.
    """
    return (pd.DataFrame(metrics["per_class"]).T
            .reset_index()
            .rename(columns={"index": "intent"})
            .astype({"support": int})
            .sort_values("f1")
            .reset_index(drop=True))


# =========================================================================
# 2. CONFUSION
# =========================================================================

def top_confusions(y_true, y_pred, k: int = 10, labels: list[str] | None = None) -> pd.DataFrame:
    """Returns the k largest off-diagonal cells: which pairs the model actually mixes up.

    The diagonal holds the correct answers, so everything interesting is
    off it. This table is what confirms or refutes the confusions
    predicted in notebooks/01_data.ipynb before any training - get_invoice
    vs check_invoice, the three refund intents, and similar.

    This comparison matters because if a model with no language
    understanding at all confuses exactly the pairs predicted from reading
    the label scheme, the confusion is a property of the taxonomy rather
    than of the model - a stronger claim than "the model got confused",
    and one that survives the fine-tuned model achieving a better score.

    `share_of_true` is included because a raw count favours large classes:
    40 errors out of 500 rows is a different phenomenon from 40 out of 60.
    """
    if labels is None:
        labels = sorted(set(map(str, y_true)) | set(map(str, y_pred)))

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    support = matrix.sum(axis=1)

    rows = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i == j or matrix[i, j] == 0:
                continue
            rows.append({
                "true": true_label,
                "predicted": pred_label,
                "count": int(matrix[i, j]),
                "share_of_true": round(float(matrix[i, j] / support[i]), 3) if support[i] else 0.0,
            })

    return (pd.DataFrame(rows)
            .sort_values("count", ascending=False)
            .head(k)
            .reset_index(drop=True))


def check_predicted_pairs(confusions: pd.DataFrame,
                          predicted_pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Scores a pre-registered confusion prediction against what actually happened.

    A prediction written down before seeing any result carries more weight
    than one asserted after the fact, and a refuted one is informative too
    - but only if it is checked mechanically rather than by eye afterwards,
    which is how a prediction quietly becomes a description.

    Direction-insensitive: predicting that A and B get confused is a claim
    about the pair, not about which way round the error falls.
    """
    seen = {frozenset((r.true, r.predicted)): r.count
            for r in confusions.itertuples()}
    rows = []
    for a, b in predicted_pairs:
        key = frozenset((a, b))
        rows.append({
            "pair": f"{a} <-> {b}",
            "confirmed": key in seen,
            "count": int(seen.get(key, 0)),
        })
    return pd.DataFrame(rows)


# =========================================================================
# 3. ERROR SLICES
# =========================================================================

def error_frame(eval_df: pd.DataFrame, y_pred, confidence=None,
                label_column: str = "intent") -> pd.DataFrame:
    """Returns every misclassified row, with enough context to inspect by hand.

    Individual misclassified sentences reveal more about what to fix than
    any aggregate score, and cost nothing to capture at prediction time -
    but only if captured during the run. Reconstructing them later would
    mean re-running the model, which for a GPU run means paying for the
    same compute twice.

    `confidence` is optional, since not every baseline produces one
    (majority class has no meaningful notion of it). When present, sorting
    by it separates two distinct failure modes: confident-and-wrong points
    to a systematic problem with the label scheme or features, while
    unconfident-and-wrong reflects the model correctly signalling that the
    sentence is ambiguous.
    """
    out = eval_df.copy().reset_index(drop=True)
    out["predicted"] = list(y_pred)
    if confidence is not None:
        out["confidence"] = np.asarray(confidence, dtype=float)

    errors = out[out[label_column] != out["predicted"]].copy()

    columns = [c for c in ["row_id", "instruction", label_column, "predicted",
                           "confidence", "category", "flags"] if c in errors.columns]
    errors = errors[columns]

    if "confidence" in errors.columns:
        errors = errors.sort_values("confidence", ascending=False)
    return errors.reset_index(drop=True)


def flag_slices(eval_df: pd.DataFrame, y_pred, label_column: str = "intent",
                min_rows: int = 30) -> pd.DataFrame:
    """Reports accuracy per linguistic-phenomenon flag.

    The dataset tags every row with letters describing how the sentence
    was generated: Z for typos, K for keyword-only phrasing, P for
    politeness, and similar. Slicing accuracy by letter turns "93%
    overall" into a robustness claim, or reveals that the average is
    carried by clean, well-formed rows while noisy ones fail.

    The letters are read from the data rather than hardcoded from the
    dataset card, since the published documentation lists twelve while
    this file contains fourteen; a hardcoded list would silently skip the
    two undocumented ones.

    `min_rows` suppresses slices too small to interpret - accuracy over
    eight rows moves by 12.5 points per row and would otherwise read as a
    finding when it is noise.
    """
    out = eval_df.copy().reset_index(drop=True)
    out["predicted"] = list(y_pred)
    out["correct"] = out[label_column] == out["predicted"]

    flags = out["flags"].fillna("").astype(str)
    letters = sorted({ch for s in flags for ch in s})

    rows = []
    for letter in letters:
        mask = flags.str.contains(letter, regex=False)
        if int(mask.sum()) < min_rows:
            continue
        rows.append({
            "flag": letter,
            "n_rows": int(mask.sum()),
            "accuracy": round(float(out.loc[mask, "correct"].mean()), 4),
            "accuracy_without": round(float(out.loc[~mask, "correct"].mean()), 4),
        })

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["delta"] = (frame["accuracy"] - frame["accuracy_without"]).round(4)
        frame = frame.sort_values("delta").reset_index(drop=True)
    return frame


# =========================================================================
# 4. RUN RECORDS
# =========================================================================

def library_versions() -> dict:
    """Records the library versions that produced a run, for reproducibility."""
    import sklearn
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "platform": platform.system(),
    }


def run_record(name: str, config: dict, metrics: dict,
               runtime_seconds: float, notes: str = "") -> dict:
    """Assembles one run into a single dictionary of fixed shape.

    Across many runs it becomes difficult to track which number came from
    which configuration, and a dictionary assembled by hand in a notebook
    cell tends to omit a different field each time, letting the summary
    table silently compare runs that were not configured the same way.

    On seeds: `config` is expected to name them separately - split_seed,
    subsample_seed, train_seed - and never to contain a single key called
    "seed". Conflating three distinct seeds under one name is how variance
    that is really different data gets reported as training noise. For
    these baselines train_seed is None, since lbfgs is deterministic; the
    only stochastic element is which rows the subsample drew.
    """
    for forbidden in ("seed",):
        if forbidden in config:
            raise ValueError(
                "config must not contain a bare 'seed' key - name it split_seed, "
                "subsample_seed or train_seed so runs stay comparable."
            )
    return {
        "schema": RUN_RECORD_SCHEMA,
        "name": name,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "metrics": metrics,
        "runtime_seconds": round(float(runtime_seconds), 2),
        "library_versions": library_versions(),
        "notes": notes,
    }


def summarise_runs(records: list[dict]) -> pd.DataFrame:
    """Combines all run records into one comparison table.

    Deliberately drops the per-class block: this is the comparison view,
    and 27 columns per run would make it unreadable. The per-class detail
    remains available in the individual JSON files and via
    per_class_frame().
    """
    rows = []
    for record in records:
        row = {"run": record["name"]}
        row.update({k: v for k, v in record["config"].items()})
        row.update({k: v for k, v in record["metrics"].items() if k != "per_class"})
        row["runtime_seconds"] = record["runtime_seconds"]
        rows.append(row)
    return pd.DataFrame(rows)
