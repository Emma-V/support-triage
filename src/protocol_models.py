"""
The two models the leakage experiment requires but does not yet have.

An earlier stage trained three adapters, all on `clean/train` - they
differ only in LoRA rank. Nothing in this project has trained on
`naive/train` or `naive_sub/train`, and those two models are the
experiment: without them there is nothing for the test set to compare
against, and the leakage finding remains a percentage of similar rows
rather than a difference in score.

--------------------------------------------------------------------------
THE ONE PROPERTY THIS MODULE HAS

It cannot open the test set. There is no path to `clean/test` or
`naive/test` anywhere in this file, and `notebooks/03d_protocol_models.ipynb`
has none either - a property checked by grep, not merely claimed, and
`tools/smoke_protocol_models.py` runs that check.

That is what makes training here safe to crash, resume and re-run at
will. The rule that the test set opens exactly once is not a matter of
care; it is a rule about a file that must be read exactly once, and a
notebook with no way to read it cannot break that rule. The ceremony
belongs to the notebook that scores, and this is not that notebook.

--------------------------------------------------------------------------
WHY THESE PARTS ARE HERE AND NOT IN A NOTEBOOK CELL

`TRAINING_PLAN` lives in the module so that changing what gets trained is
a line in a diff with a date on it. `matches_freeze` lives in the module
so the comparison between a run and the frozen configuration is the same
comparison every time it is made. Both are the kind of logic that is easy
to leave in a notebook cell right up until the moment someone needs to
know whether it matched an earlier run.

Module conventions, as elsewhere in src/: no prints, no plots, no writing
to results/ - except save_val_outputs(), which exists to write files and
says so in its name. Scoring goes through src/evaluate.py rather than a
re-implementation, so these models are measured by the identical function
that measured the TF-IDF baselines.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as E


# =========================================================================
# 1. WHAT GETS TRAINED
# =========================================================================
# The most recent prior run was run_08. Run numbers are never reused -
# including for runs that were lost before completion, since a run that
# disappeared will be run again and the two must not share a name.
RUN_NUMBER_BASE = 9

# The validation frame each model is selected on is its own protocol's,
# and that is a deliberate decision rather than a convenience. Selecting
# the naive model's best epoch on `clean/val` would hand it a selection
# signal the clean model never received, and the comparison table would
# then silently include that difference alongside the one it claims to
# measure.
#
# `naive_sub` is selected on `naive/val` for the same reason: it is the
# naive protocol run at a different size, so it lives in the naive
# protocol's world end to end. What makes it a size control is its row
# count, not its validation set.
TRAINING_PLAN = (
    {
        "key": "naive",
        "label": "fine-tuned - naive",
        "train": "naive/train",
        "val": "naive/val",
        "why": "a plain random split, start to finish - what this project would "
               "report under an ordinary protocol, without the near-duplicate "
               "family control. The headline the ordinary protocol would produce.",
    },
    {
        "key": "naive_sub",
        "label": "fine-tuned - naive, size-matched",
        "train": "naive_sub/train",
        "val": "naive/val",
        "why": "the same naive protocol on clean/train's row count, so that "
               "'you simply trained on less data' cannot explain the gap. This "
               "is the control that makes the leakage claim survive scrutiny.",
    },
)

PLAN_BY_KEY = {plan["key"]: plan for plan in TRAINING_PLAN}


def run_name(key: str, r: int) -> str:
    """Builds the run's name identically wherever it is needed.

    Built rather than written out by hand, since it is needed in three
    places - the adapter directory, the record filename and the resume
    check - and three hand-written strings are three chances for one to
    disagree with the others.
    """
    if key not in PLAN_BY_KEY:
        raise KeyError(f"{key!r} is not in TRAINING_PLAN - keys are {sorted(PLAN_BY_KEY)}")
    offset = [plan["key"] for plan in TRAINING_PLAN].index(key)
    return f"run_{RUN_NUMBER_BASE + offset:02d}_{key}_r{r}"


# =========================================================================
# 2. THE FREEZE COMPARISON
# =========================================================================
# The check that a comparison between protocols is not also a comparison
# between configurations. It is inexpensive, mechanical, and the only
# thing distinguishing "the naive split scores higher" from "the naive run
# happened to use a different learning rate".

FROZEN_FIELDS = (
    "base_model", "task_type", "r", "lora_alpha", "lora_dropout",
    "target_modules", "modules_to_save",
    "learning_rate", "epochs", "batch_size", "grad_accum", "warmup_ratio",
    "weight_decay", "precision", "max_length", "selection_metric",
)

# Expected to differ, and listed rather than omitted so a reader can see
# they were considered rather than overlooked. The entire point of this
# stage is that the data moves while the configuration does not, so a run
# whose train_sha256 matched the freeze would mean the naive model had
# trained on clean/train.
EXPECTED_TO_DIFFER = (
    "trained_on", "scored_on", "train_rows", "eval_rows", "train_sha256",
    "best_epoch", "subsample_seed",
)


def _freeze_value(freeze: dict, field: str):
    """Looks one frozen field up wherever it lives in the freeze record."""
    for section in ("model", "training", "data", "hardware"):
        if field in freeze.get(section, {}):
            return freeze[section][field]
    raise KeyError(
        f"{field} is not anywhere in the freeze record - its sections are "
        f"{sorted(freeze)}. Either FROZEN_FIELDS names a field the freeze does "
        "not carry, or this is not a freeze record.")


def matches_freeze(record: dict, freeze: dict) -> pd.DataFrame:
    """Compares one run against the frozen configuration, field by field.

    Returns one row per field so the notebook can display the whole
    comparison rather than only its verdict - a table of eighteen `True`s
    is evidence, while a bare "passed" is only a claim.

    Values are compared through repr() so that 32 and "32" do not compare
    equal. A string where an int belongs is exactly the kind of drift a
    JSON round trip can introduce, and one that == would hide.
    """
    config = record["config"]
    rows = []
    for field in FROZEN_FIELDS:
        want, got = _freeze_value(freeze, field), config.get(field)
        rows.append({
            "field": field,
            "frozen": repr(want),
            "this run": repr(got),
            "identical": repr(want) == repr(got),
        })
    return pd.DataFrame(rows)


def differences_from_freeze(record: dict, freeze: dict) -> pd.DataFrame:
    """The fields that are supposed to differ, shown side by side.

    Not a check but a display: it lets a reader see directly that "the
    data moved and the configuration did not", rather than inferring it
    from the absence of a complaint.
    """
    config = record["config"]
    rows = []
    for field in EXPECTED_TO_DIFFER:
        # `best_epoch` and `subsample_seed` are properties of a run rather
        # than of a configuration, so the freeze carries no value for them
        # and the column reads "-" rather than "None", which would look
        # like an actual value.
        try:
            frozen = repr(_freeze_value(freeze, field))
        except KeyError:
            frozen = "-"
        rows.append({
            "field": field,
            "the frozen run": frozen,
            "this run": repr(config.get(field)),
            # "-" rather than True where there is no frozen value to differ
            # from. Reporting a difference against a field the freeze never
            # carried would be inventing one.
            "differs": "-" if frozen == "-" else frozen != repr(config.get(field)),
        })
    return pd.DataFrame(rows)


def freeze_violations(record: dict, freeze: dict) -> list[str]:
    """The differing frozen fields, as strings, for an assert message."""
    frame = matches_freeze(record, freeze)
    return [f"{row['field']}: frozen {row['frozen']} but this run {row['this run']}"
            for _, row in frame.loc[~frame["identical"]].iterrows()]


def assert_matches_freeze(record: dict, freeze: dict) -> None:
    """Raises before anything is kept, naming every field that drifted."""
    violations = freeze_violations(record, freeze)
    if violations:
        raise AssertionError(
            f"{record['name']} does not match artifacts/config_freeze.json:\n  "
            + "\n  ".join(violations) +
            "\n\nComparing this run against the others would measure the "
            "configuration as well as the protocol, and the table would not say "
            "which. Fix the configuration and retrain, or break the freeze in "
            "writing - see its `breaking_this_freeze` field.")


# =========================================================================
# 3. SCORING AND ROW-LEVEL OUTPUTS
# =========================================================================

def score_from_logits(logits, frame: pd.DataFrame, labels: list[str],
                      label_column: str = "intent") -> tuple[dict, pd.DataFrame]:
    """Computes metrics and the per-row frame from one forward pass.

    Both come from the same argmax, so the table in the report and the
    rows behind it cannot disagree. Returning the rows is not a
    convenience: downstream error analysis runs on them, on CPU, and a
    row-level frame that was not captured during the run would cost a GPU
    hour to rebuild.
    """
    logits = np.asarray(logits, dtype=float)
    if logits.shape != (len(frame), len(labels)):
        raise ValueError(
            f"logits are {logits.shape}, expected {(len(frame), len(labels))}. "
            "A mismatch here silently pairs each row with another row's scores.")

    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    predicted_ids = probabilities.argmax(axis=1)
    predicted = [labels[i] for i in predicted_ids]
    confidence = probabilities.max(axis=1)

    metrics = E.evaluate_predictions(frame[label_column], predicted, labels)

    rows = frame.copy().reset_index(drop=True)
    rows["predicted"] = predicted
    rows["confidence"] = confidence
    rows["correct"] = rows[label_column].to_numpy() == np.asarray(predicted)
    return metrics, rows


def save_row_outputs(name: str, logits, rows: pd.DataFrame,
                     metrics_dir: Path | str, part: str = "val") -> list[Path]:
    """Saves the logit matrix and the per-row predictions for one run on one split.

    Logits are kept alongside the predictions because they are what later
    confidence and calibration analysis reads, and an argmax cannot be
    un-taken once discarded - an earlier run kept only the argmax, which is
    why its untrained-model results could not enter the confidence
    analysis afterwards.

    `part` is included in the filename rather than left implicit because
    the same adapter can be scored on more than one split, and two files
    that differ only in which rows produced them are exactly the pair most
    worth being unable to confuse. It is also what lets later analysis run
    with no GPU at all: with `test_predictions_*.csv` on disk the analysis
    is a CPU job over CSVs, and without them it would require re-opening
    the test set.
    """
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logits_path = metrics_dir / f"{part}_logits_{name}.npy"
    rows_path = metrics_dir / f"{part}_predictions_{name}.csv"
    np.save(logits_path, np.asarray(logits, dtype=np.float32))
    rows.to_csv(rows_path, index=False, encoding="utf-8")
    return [logits_path, rows_path]


def save_val_outputs(name: str, logits, rows: pd.DataFrame,
                     metrics_dir: Path | str) -> list[Path]:
    """save_row_outputs() on the validation split.

    Kept as its own name because `03d_protocol_models.ipynb` was run
    against it and is committed with its outputs: the notebook that
    produced run_09 and run_10 should stay re-runnable as written, rather
    than referencing a name that no longer exists.
    """
    return save_row_outputs(name, logits, rows, metrics_dir, part="val")


# =========================================================================
# 4. THE JOURNAL
# =========================================================================
# One line per event, appended, never rewritten. The value of the file is
# its order: it is what lets a reader who was not present see that the
# freeze was recovered before the models were trained, and that both
# happened before the test set was opened. A journal that can be edited in
# the middle proves nothing, so entries are appended and timestamped at
# the moment they happen.

def write_journal_entry(path: Path | str, entry: dict) -> Path:
    """Append one timestamped event. Returns the path, for printing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamped = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(stamped, ensure_ascii=False) + "\n")
    return path


def read_journal(path: Path | str) -> pd.DataFrame:
    """The journal as a frame, in the order it was written."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["at", "event"])
    entries = [json.loads(line) for line in
               path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(entries)
