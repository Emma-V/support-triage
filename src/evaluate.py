"""
Measurement. Once the model is trained, how good is it actually?

What goes in here:
- The scores. The main metric is macro-F1: the score is computed separately
  for each of the 27 intents and then averaged, so a small intent that the
  model fails on completely is not hidden by the big ones.
- A classification report, a table with a score per intent.
- A confusion matrix: which intents the model mixes up with each other. For
  example get_invoice and check_invoice, where the difference is one verb.
- Error slices: check whether the model fails more on rows with typos, or on
  rows that are keywords rather than a full sentence. The dataset marks this
  in the flags column.

The plots and tables that go into the report come from this file.

--------------------------------------------------------------------------
Every score in this project comes from THIS module - the majority-class
baseline, TF-IDF, and tomorrow's fine-tuned Qwen3 alike. That is the point of
it being its own file. Two runs scored by two implementations are not
comparable, however similar the two implementations look, and the difference
usually turns out to be something like an implicit label list.

Nothing here knows what produced the predictions. Everything takes plain
arrays of labels, so it works identically for a model that reads and a model
that does not.

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

# Bumped only if the shape of a run record changes. Twelve runs from now, this
# is what tells you whether an old JSON can be read by the current summary code.
RUN_RECORD_SCHEMA = 1


# =========================================================================
# 1. SCORES
# =========================================================================

def evaluate_predictions(y_true, y_pred, labels: list[str]) -> dict:
    """The scores for one run. `labels` is REQUIRED, and that is the whole point.

    Without an explicit label list, scikit-learn averages macro-F1 over the
    classes it happens to see in y_true or y_pred. If the model never predicts
    a rare intent AND that intent is thin in the slice being measured, the
    average silently runs over 26 classes instead of 27. The score goes UP, no
    warning is printed, and two runs that averaged over different denominators
    get compared as if they were the same measurement.

    Passing all 27 from artifacts/labels.json is what makes the number
    comparable across runs, across splits, and across models. An intent the
    model never gets right contributes a genuine 0.0 instead of vanishing.

    Returns accuracy, macro-F1, weighted-F1, and a per-class breakdown.

    Why both macro and weighted: macro treats every intent equally and is the
    decisive metric (a small intent the model always misses drags it down);
    weighted scales each intent by how many rows it has. The GAP between them
    is the size of the imbalance problem, so reporting one without the other
    throws away information. The raw dataset is near-balanced, but the clean
    training set is not - the family collapse left a 4.06:1 ratio - so the two
    numbers diverge at the intent level here, not only at the category level.
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
    """The per-class block of evaluate_predictions() as a table, worst first.

    This is the table that answers "which intents are actually hard", which is
    a different and more useful question than "what is the average". Sorted
    ascending by F1 so the problems are at the top, and written to CSV so
    tomorrow's model can be compared to it intent by intent rather than on one
    aggregate number.
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
    """The k largest off-diagonal cells: which pairs the model actually mixes up.

    The diagonal is the correct answers, so everything interesting is off it.
    This is the table that confronts the prediction pre-registered in
    01_data.ipynb before anything was trained - get_invoice vs check_invoice,
    the three refund intents, and so on.

    Why this matters more than it looks: if a model with no language
    understanding at all confuses exactly the pairs that were predicted from
    reading the label scheme, the confusion is a property of the TAXONOMY, not
    of the model. That is a much stronger claim than "the model got confused",
    and it survives the fine-tuned model getting a better score.

    `share_of_true` is included because a raw count favours large classes: 40
    errors out of 500 rows is a different phenomenon from 40 out of 60.
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
    """Score the day-1 pre-registration against what actually happened.

    A prediction written down before seeing any result is the most convincing
    paragraph available in the analysis chapter, and one that is refuted is
    more interesting still - but only if it is checked mechanically rather than
    by eye afterwards, which is how a prediction quietly becomes a description.

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
    """Every misclassified row, with enough context to read it by hand.

    Twenty actual sentences the model got wrong tell you more about what to fix
    than any aggregate score does, and they cost nothing at prediction time -
    but only if they are captured DURING the run. Reconstructing them later
    means re-running the model, which for tomorrow's GPU runs means paying for
    the same compute twice.

    `confidence` is optional because not every baseline has one (majority class
    has no meaningful notion of it). When present, sorting by it separates two
    very different failures: confident-and-wrong is a systematic problem with
    the label scheme or the features, while unconfident-and-wrong is the model
    correctly reporting that the sentence is ambiguous.
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
    """Accuracy per linguistic-phenomenon flag. Where does the model actually break?

    The dataset tags every row with letters describing how the sentence was
    generated: Z for typos, K for keyword-only phrasing, P for politeness, and
    so on. Slicing accuracy by letter turns "93% overall" into a robustness
    claim - or reveals that the average is carried by the clean, well-formed
    rows while the noisy ones fail.

    The letters are read from the data rather than hardcoded from the dataset
    card. The published documentation lists twelve; this file contains
    fourteen. A hardcoded list would silently skip the two undocumented ones.

    `min_rows` suppresses slices too small to interpret - accuracy over eight
    rows moves by 12.5 points per row and reads as a finding when it is noise.
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
    """Which versions produced this number. Cheap now, unreconstructable later."""
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
    """One run, one dictionary, one JSON file. Always the same shape.

    After a dozen runs it is impossible to remember which number came from
    which configuration, and a dictionary assembled by hand in a notebook cell
    omits a different field every time - so the summary table quietly compares
    runs that were not configured the same way.

    On seeds: `config` is expected to name them separately - split_seed,
    subsample_seed, train_seed - and never to contain one key called "seed".
    Three different things called "seed" is how variance that is really
    different DATA gets reported as training noise. For these baselines
    train_seed is None, because lbfgs is deterministic; the only stochastic
    element is which rows the subsample drew.
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
    """All run records as one table - the thing that gets pasted into the report.

    Deliberately drops the per-class block: this is the comparison view, and
    27 columns per run would make it unreadable. The per-class detail stays in
    the individual JSON files and in per_class_frame().
    """
    rows = []
    for record in records:
        row = {"run": record["name"]}
        row.update({k: v for k, v in record["config"].items()})
        row.update({k: v for k, v in record["metrics"].items() if k != "per_class"})
        row["runtime_seconds"] = record["runtime_seconds"]
        rows.append(row)
    return pd.DataFrame(rows)
