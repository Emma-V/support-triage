"""
Day 5, step 18a: the two models the leakage experiment is missing.

Day 3 trained three adapters and all three trained on `clean/train` - they
differ only in LoRA rank. Nothing in this project has ever seen `naive/train`
or `naive_sub/train`, and those two models ARE the experiment: without them
there is nothing for the test set to compare, and the leakage finding is a
percentage of similar rows rather than a difference in a score.

--------------------------------------------------------------------------
THE ONE PROPERTY THIS MODULE HAS

It cannot open the test set. There is no path to `clean/test` or `naive/test`
anywhere in this file, and `notebooks/03d_protocol_models.ipynb` has none
either - which is a grep, not a promise, and tools/smoke_protocol_models.py
runs that grep.

That is what makes today's training safe to crash, resume and re-run at will.
Day 5's "the test set opens once" rule is not a rule about care; it is a rule
about a file that must be read exactly once, and a notebook that cannot read
it cannot break the rule. The ceremony belongs to the notebook that scores,
and this is not that notebook.

--------------------------------------------------------------------------
WHY THESE PARTS ARE HERE AND NOT IN A CELL

`TRAINING_PLAN` in the module means changing what gets trained is a line in a
diff with a date on it. `matches_freeze` in the module means the comparison
between a run and the frozen configuration is the same comparison every time
it is made. Both were the kind of thing that lives in a cell right up until
the moment somebody needs to know whether it was the same yesterday.

HOUSE RULES, as everywhere else in src/: no prints, no plots, no writing to
results/ - except save_val_outputs(), which exists to write files and says so
in its name. Scoring goes through src/evaluate.py, never a re-implementation,
so that today's models are measured by the identical function that measured
the TF-IDF baselines on day 2.
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
# Day 4 ended at run_08. Run numbers are never reused - including for the three
# runs day 4 lost, because a run that disappeared is a run that will be run
# again and the two must not share a name.
RUN_NUMBER_BASE = 9

# The validation frame each model is selected on is its OWN protocol's, and
# that is a decision rather than a convenience. Selecting the naive model's
# best epoch on `clean/val` would hand it a selection signal the clean model
# never got, and the comparison in tomorrow's table would silently include
# that difference alongside the one it claims to measure.
#
# `naive_sub` is selected on `naive/val` for the same reason: it is the naive
# protocol run at a different size, so it lives in the naive protocol's world
# end to end. What makes it a size control is its row count, not its
# validation set.
TRAINING_PLAN = (
    {
        "key": "naive",
        "label": "fine-tuned - naive",
        "train": "naive/train",
        "val": "naive/val",
        "why": "a plain random split, start to finish - what this project would "
               "have reported without day 1. The headline the ordinary protocol "
               "would have produced.",
    },
    {
        "key": "naive_sub",
        "label": "fine-tuned - naive, size-matched",
        "train": "naive_sub/train",
        "val": "naive/val",
        "why": "the same naive protocol on clean/train's row count, so that "
               "'you simply trained on less data' cannot explain the gap. This "
               "is the control that makes the leakage claim survive a reviewer.",
    },
)

PLAN_BY_KEY = {plan["key"]: plan for plan in TRAINING_PLAN}


def run_name(key: str, r: int) -> str:
    """The run's name, built the same way every time it is needed.

    Built rather than written down because it is needed in three places - the
    adapter directory, the record filename and the resume check - and three
    hand-written strings are three chances for one of them to disagree.
    """
    if key not in PLAN_BY_KEY:
        raise KeyError(f"{key!r} is not in TRAINING_PLAN - keys are {sorted(PLAN_BY_KEY)}")
    offset = [plan["key"] for plan in TRAINING_PLAN].index(key)
    return f"run_{RUN_NUMBER_BASE + offset:02d}_{key}_r{r}"


# =========================================================================
# 2. THE FREEZE COMPARISON
# =========================================================================
# The check that a comparison between PROTOCOLS is not also a comparison
# between CONFIGURATIONS. It is cheap, it is mechanical, and it is the only
# thing standing between "the naive split scores higher" and "the naive run
# happened to use a different learning rate".

FROZEN_FIELDS = (
    "base_model", "task_type", "r", "lora_alpha", "lora_dropout",
    "target_modules", "modules_to_save",
    "learning_rate", "epochs", "batch_size", "grad_accum", "warmup_ratio",
    "weight_decay", "precision", "max_length", "selection_metric",
)

# Expected to differ, and listed rather than omitted so that a reader can see
# they were considered rather than overlooked. Today's entire point is that the
# DATA moves while the configuration does not, so a run whose train_sha256
# matched the freeze would mean the naive model had trained on clean/train.
EXPECTED_TO_DIFFER = (
    "trained_on", "scored_on", "train_rows", "eval_rows", "train_sha256",
    "best_epoch", "subsample_seed",
)


def _freeze_value(freeze: dict, field: str):
    """Look one frozen field up wherever it lives in the freeze record."""
    for section in ("model", "training", "data", "hardware"):
        if field in freeze.get(section, {}):
            return freeze[section][field]
    raise KeyError(
        f"{field} is not anywhere in the freeze record - its sections are "
        f"{sorted(freeze)}. Either FROZEN_FIELDS names a field the freeze does "
        "not carry, or this is not a freeze record.")


def matches_freeze(record: dict, freeze: dict) -> pd.DataFrame:
    """Compare one run against the frozen configuration, field by field.

    Returns one row per field so the notebook can display the whole comparison
    rather than only its verdict - a table of eighteen `True`s is evidence, and
    a bare "passed" is a claim.

    Values are compared through repr() so that 32 and "32" do not compare
    equal. A string where an int belongs is exactly the kind of drift that a
    JSON round trip introduces and that == would hide.
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
    """The fields that are SUPPOSED to differ, shown side by side.

    Not a check - a display. It exists so that "the data moved and the
    configuration did not" is something a reader can see in one place instead
    of inferring from the absence of a complaint.
    """
    config = record["config"]
    rows = []
    for field in EXPECTED_TO_DIFFER:
        # `best_epoch` and `subsample_seed` are properties of a run rather than
        # of a configuration, so the freeze carries no value for them and the
        # column reads "-" rather than "None" - which would look like a value.
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
    """Raise before anything is kept, naming every field that drifted."""
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
    """Metrics AND the per-row frame, from one forward pass.

    Both come out of the same argmax, so the table in the report and the rows
    behind it cannot disagree. Returning the rows is not a convenience: day 6's
    analysis runs on them, on CPU, and a row-level frame that was not captured
    during the run costs a GPU hour to rebuild.
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


def save_val_outputs(name: str, logits, rows: pd.DataFrame,
                     metrics_dir: Path | str) -> list[Path]:
    """The logit matrix and the per-row predictions, for one run's validation set.

    The logits are kept as well as the predictions because they are what the
    confidence and calibration work reads, and an argmax cannot be un-taken.
    Day 3 kept only the argmax, which is why the untrained model could not
    enter the confidence analysis afterwards.
    """
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logits_path = metrics_dir / f"val_logits_{name}.npy"
    rows_path = metrics_dir / f"val_predictions_{name}.csv"
    np.save(logits_path, np.asarray(logits, dtype=np.float32))
    rows.to_csv(rows_path, index=False, encoding="utf-8")
    return [logits_path, rows_path]


# =========================================================================
# 4. THE JOURNAL
# =========================================================================
# One line per event, appended, never rewritten. The value of the file is its
# ORDER: it is what lets somebody who was not here see that the freeze was
# recovered before the models were trained, and that both happened before the
# test set was opened. A journal that can be edited in the middle proves
# nothing, so entries are appended and timestamped at the moment they happen.

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
