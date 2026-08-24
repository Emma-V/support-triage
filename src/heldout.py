"""
Day 5, step 18b: the test set opens once, and this module is what "once" means.

Every other module in src/ describes a computation. This one describes a
PROCEDURE, and the difference is the point. The scores it produces are not more
accurate than the validation scores next to them - they are 2,120 rows measured
by the same code that measured 2,120 other rows yesterday. What makes them worth
more is a claim about history: that nothing in this project was chosen after
seeing them. That claim is not verifiable from the numbers, only from the order
in which things happened, so the order has to be written down somewhere that a
reader can check.

--------------------------------------------------------------------------
THE THREE THINGS THAT LIVE HERE, AND WHY EACH IS IN A MODULE

1. `EVALUATION_PLAN` - the exact seven rows that will be measured. In a module
   rather than a cell so that the run list is a commit with a date on it, made
   before the session that spends it. "I decided in advance" and "git says I
   decided in advance" are different statements, and only one of them survives
   a sceptical reader.

2. The seal. `assert_seal_absent()` refuses to open a test set that has already
   been opened, unless a written reason is supplied - and then records that
   reason in the artifact. This is decision ז4 as code: a technical failure
   justifies a re-run and a disappointing number does not, and the difference
   between them is a sentence somebody has to type and commit. A second opening
   is not prevented, which would be dishonest engineering - it is made
   impossible to perform silently.

3. `readings()` - the three subtractions the table exists to support, each
   carrying whether the noise floor applies to it. Written here because the
   dangerous one is reading 1: the headline comparison spans two DIFFERENT test
   sets, so the between-seed floor measured on one of them does not govern it,
   and the natural thing to do with a floor in a table is apply it to every row.

--------------------------------------------------------------------------
WHAT THIS MODULE DELIBERATELY DOES NOT DO

It does not read a test file. Every function here takes frames it is handed.
The reading happens in `notebooks/04_test.ipynb`, in one place, after the
journal entry that says it is about to happen - and `tools/smoke_heldout.py`
checks that this file contains no path to one.

HOUSE RULES, as everywhere else in src/: no prints, no plots, no writing to
results/ - except save_test_outputs() and seal(), which exist to write files and
say so in their names. Scoring goes through src/evaluate.py, never a
re-implementation, so today's numbers are produced by the identical function
that produced day 2's TF-IDF baselines.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as E
from . import protocol_models as P


# =========================================================================
# 1. THE RUN LIST
# =========================================================================
# Decisions ז1 and ז2, as data. The order is the order of the table in the
# report: the cheap models first and the headline in the middle, so that a
# reader meets the number the fine-tuned model has to beat before meeting the
# number it got.
#
# `same_test_rows_as_headline` is the field that does the real work. Four of
# these seven rows are measured on `clean/test` and three are not, and every
# difference taken between rows on different evaluation sets is a difference
# that includes "these are not the same sentences". Carrying that as a field
# means the readings below cannot forget it.

EVALUATION_PLAN = (
    {
        "key": "majority",
        "kind": "baseline",
        "family": "majority_class",
        "label": "majority class",
        "trained_on": "clean/train",
        "test": "clean/test",
        "needs_gpu": False,
        "role": "floor",
        "why": "the score of a model that does not read. Everything else in the "
               "table is only meaningful as a distance from this.",
    },
    {
        "key": "tfidf",
        "kind": "baseline",
        "family": "tfidf+logreg",
        "label": "TF-IDF + LogReg",
        "trained_on": "clean/train",
        "test": "clean/test",
        "needs_gpu": False,
        "role": "baseline",
        "why": "the linear baseline in the independent word (1,2) feature space, "
               "the same configuration day 2 reported as tfidf_clean. This is what "
               "the fine-tuned model has to be worth more than.",
    },
    {
        "key": "zero_shot",
        "kind": "baseline",
        "family": "qwen3+label_scoring",
        "label": "zero-shot (no training)",
        "trained_on": "-",
        "test": "clean/test",
        "needs_gpu": True,
        "role": "before",
        "why": "the same base weights the adapter sits on, asked the same question "
               "with no training at all - the honest 'before' for a before-and-after "
               "claim. Measured on clean/test so the pair is on the same rows.",
    },
    {
        "key": "clean",
        "kind": "model",
        "model": "clean",
        "label": "fine-tuned - clean",
        "trained_on": "clean/train",
        "test": "clean/test",
        "needs_gpu": True,
        "role": "headline",
        "why": "the system's actual performance, on rows whose near-duplicate "
               "families were held out of training. This is the number the report "
               "leads with.",
    },
    {
        "key": "naive",
        "kind": "model",
        "model": "naive",
        "label": "fine-tuned - naive",
        "trained_on": "naive/train",
        "test": "naive/test",
        "needs_gpu": True,
        "role": "protocol",
        "why": "what this project would have reported without day 1 - an ordinary "
               "random split, end to end. Not a better or worse model: a different "
               "protocol's headline.",
    },
    {
        "key": "naive_sub",
        "kind": "model",
        "model": "naive_sub",
        "label": "fine-tuned - naive, size-matched",
        "trained_on": "naive_sub/train",
        "test": "naive/test",
        "needs_gpu": True,
        "role": "protocol",
        "why": "the same naive protocol at clean/train's row count, so that 'the "
               "clean number is lower because you trained on less data' is answered "
               "before it is asked.",
    },
    {
        "key": "naive_sub_on_clean",
        "kind": "model",
        "model": "naive_sub",
        "label": "fine-tuned - naive, size-matched",
        "trained_on": "naive_sub/train",
        "test": "clean/test",
        "needs_gpu": True,
        "role": "control",
        "why": "THE CONTROL, and not a performance number. Same test rows as the "
               "headline, same training-set size as the headline, and the only "
               "thing that moves is whether the training set was allowed to "
               "contain siblings of those test rows. Reported as a control, with "
               "the sentence that says so, or it reads as a mistake.",
    },
)

PLAN_BY_KEY = {row["key"]: row for row in EVALUATION_PLAN}

# Which test file each row reads. Two files, seven rows - stated as a set so the
# notebook can open exactly these and no others.
TEST_SPLITS = tuple(dict.fromkeys(row["test"] for row in EVALUATION_PLAN))

# The row every "same rows?" question is asked against.
HEADLINE_KEY = "clean"
HEADLINE_TEST = PLAN_BY_KEY[HEADLINE_KEY]["test"]


def same_test_rows_as_headline(key: str) -> bool:
    """Is this row measured on the same sentences as the headline row?

    The one question that decides whether a difference between two rows can be
    judged against the noise floor at all. Kept as a function of the plan rather
    than a hand-maintained field so that it cannot drift away from `test`.
    """
    return PLAN_BY_KEY[key]["test"] == HEADLINE_TEST


def model_keys() -> list[str]:
    """The distinct adapters this plan needs, in the order they are first used.

    Distinct, because `naive_sub` appears in two rows and loading a 1.7B base
    model twice to score two frames is four minutes of nothing. The notebook
    loads each key once and scores every row that names it.
    """
    return list(dict.fromkeys(row["model"] for row in EVALUATION_PLAN
                              if row["kind"] == "model"))


def rows_for_model(model_key: str) -> list[dict]:
    """Every plan row that this adapter is responsible for."""
    return [row for row in EVALUATION_PLAN
            if row["kind"] == "model" and row["model"] == model_key]


def adapter_run_name(model_key: str, freeze: dict, r: int) -> str:
    """Which run directory holds this model's adapter.

    The clean model's name is read from the freeze record rather than written
    down, because the freeze is the thing that decides which run is the frozen
    one; a literal here could disagree with it, and the disagreement would be
    invisible - a plausible adapter, a plausible score, and the wrong model.
    The other two are built by the same function that named them on day 5.
    """
    if model_key == "clean":
        return freeze["frozen_from_run"]
    return P.run_name(model_key, r)


def plan_fingerprint() -> str:
    """A hash of the run list as it stands in this file.

    Recorded in the seal so that the plan which ran and the plan which was
    committed can be compared later by a reader who was not here. If the two
    differ, the plan was edited between the commit and the run - which is
    allowed, and is exactly the thing that has to be visible.
    """
    payload = json.dumps(EVALUATION_PLAN, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def plan_frame() -> pd.DataFrame:
    """The run list as a table, for printing before anything is opened."""
    return pd.DataFrame([{
        "#": i + 1,
        "row": row["key"],
        "what": row["label"],
        "trained on": row["trained_on"],
        "measured on": row["test"],
        "role": row["role"],
        "GPU": "yes" if row["needs_gpu"] else "no",
    } for i, row in enumerate(EVALUATION_PLAN)])


# =========================================================================
# 2. THE SEAL
# =========================================================================
# Decision ז4, made mechanical. The rule written before any number existed is
# that a technical failure justifies re-opening the test set and a disappointing
# result does not. A rule like that is worth exactly as much as the friction
# behind it, and prose in a work log has none: re-running a notebook is one
# keystroke and leaves no trace that it was the second time.
#
# So the first successful run writes an artifact, and the notebook refuses to
# start when it finds one. Refusing is not the interesting part - `REOPEN_REASON`
# defeats it in one line, and it is meant to. The interesting part is that
# defeating it APPENDS to the artifact: the reason, the timestamp, and the
# scores that were already on the table when the decision to re-run was taken.
# A second opening stays possible and stops being deniable.

SEAL_FILENAME = "test_seal.json"


def read_seal(path: Path | str) -> dict | None:
    """The seal, or None if the test set has never been opened."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def assert_seal_absent(path: Path | str, reason: str | None = None) -> dict | None:
    """Refuse to re-open a sealed test set without a written reason. Returns the seal.

    Returns None on a first opening and the existing seal on a justified
    re-opening, so the caller can carry the previous scores into the new seal.

    The error message quotes the scores the previous run produced on purpose.
    The failure this guards against is not somebody deciding to cheat; it is
    somebody re-running a notebook out of habit at the end of a long session and
    quietly overwriting a number. Being shown the number that already exists is
    what turns that from a reflex into a decision.
    """
    seal = read_seal(path)
    if seal is None:
        return None

    if not reason or not str(reason).strip():
        already = "\n  ".join(
            f"{key:22s} accuracy {value['accuracy']:.4f}  macro-F1 {value['f1_macro']:.4f}"
            for key, value in seal.get("results", {}).items())
        raise AssertionError(
            f"The test set was already opened at {seal.get('opened_at')} and these "
            f"numbers exist:\n  {already}\n\n"
            "Re-running would overwrite them, and decision ז4 was written before any "
            "of them existed:\n"
            "  a re-run is justified by a TECHNICAL failure - a crash, a file that was "
            "not written, an adapter that did not load, a metric computed on the wrong "
            "column;\n"
            "  a re-run is NOT justified by the size of the number, by trying another "
            "epoch, or by another threshold.\n\n"
            "If the reason is technical, set REOPEN_REASON to it in the notebook. It is "
            "appended to the seal with a timestamp, so the second opening is part of the "
            "record rather than a thing that quietly happened.")
    return seal


def seal(path: Path | str, *, freeze: dict, manifest: dict, records: dict,
         test_frames: dict, reopened_from: dict | None = None,
         reason: str | None = None) -> dict:
    """Write the receipt: what was opened, when, with which plan, and what came out.

    Deliberately a small, dull, machine-readable file rather than a paragraph.
    The paragraph goes in the report and can be written to say anything; this
    holds the hashes of the files that were read, the fingerprint of the run
    list, and the scores, and it is committed in the same commit as the results
    it describes.
    """
    path = Path(path)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    document = {
        "what": "the record that the held-out test set was opened, and of what was "
                "run in that one session. Day 5, step 18b.",
        "opened_at": now,
        "opened_by": "notebooks/04_test.ipynb",
        "plan_sha256": plan_fingerprint(),
        "plan": [row["key"] for row in EVALUATION_PLAN],
        "freeze": {
            "frozen_at": freeze.get("frozen_at"),
            "frozen_from_run": freeze.get("frozen_from_run"),
            "chosen_r": freeze.get("selection", {}).get("chosen_r"),
        },
        "test_files": {
            split: {
                "n_rows": int(len(frame)),
                "sha256": manifest["splits"][split.split("/")[0]][split.split("/")[1]]["sha256"],
            }
            for split, frame in test_frames.items()
        },
        "results": {
            key: {
                "accuracy": record["metrics"]["accuracy"],
                "f1_macro": record["metrics"]["f1_macro"],
                "measured_on": record["config"]["scored_on"],
            }
            for key, record in records.items()
        },
        "closed": "from here the test set is closed. Day 6 reads the row-level "
                  "prediction files in results/metrics/ and changes nothing.",
        "reopenings": list(reopened_from.get("reopenings", [])) if reopened_from else [],
    }

    if reopened_from is not None:
        document["reopenings"].append({
            "at": now,
            "reason": reason,
            "superseded": reopened_from.get("results", {}),
            "previously_opened_at": reopened_from.get("opened_at"),
        })
        document["first_opened_at"] = reopened_from.get(
            "first_opened_at", reopened_from.get("opened_at"))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return document


# =========================================================================
# 3. SCORING ONE ROW OF THE TABLE
# =========================================================================

def test_record(key: str, metrics: dict, config: dict, runtime_seconds: float,
                notes: str = "") -> dict:
    """One plan row's result, in the shape every other run record in this project has.

    Goes through E.run_record so that the seven rows measured today land in the
    same summary table as the eleven runs measured before them, with the same
    column names. The plan row's `role` and `why` are copied in, because six
    weeks from now "why is the naive model being scored on the clean test set"
    is a question the file should answer by itself.
    """
    row = PLAN_BY_KEY[key]
    full_config = {
        "model": row["family"] if row["kind"] == "baseline" else "qwen3+lora",
        "role": row["role"],
        "trained_on": row["trained_on"],
        "scored_on": row["test"],
        "same_test_rows_as_headline": same_test_rows_as_headline(key),
        **config,
    }
    record = E.run_record(f"test_{key}", full_config, metrics, runtime_seconds,
                          notes=notes or row["why"])
    record["plan_sha256"] = plan_fingerprint()
    return record


def save_test_outputs(key: str, logits, rows: pd.DataFrame,
                      metrics_dir: Path | str) -> list[Path]:
    """The row-level predictions for one plan row, under a name that says test.

    This is the single most important write of day 5, and it is worth being
    blunt about why: day 6 is a whole day of analysis - confusion pairs, slices,
    reading twenty errors by hand - and it runs on CPU, from these files. If
    they are not written, day 6 cannot be done without opening the test set a
    second time, which would break the rule this entire day is built around. The
    scores are recoverable from the seal. These are not recoverable from
    anything.

    Named by the PLAN ROW rather than by the run that produced it, which is not
    a cosmetic choice: `naive_sub` is one adapter and two rows of the table, so
    a file named after the adapter would be written twice and the second write
    would silently destroy the control. The plan key is the only name that
    distinguishes them.
    """
    return P.save_row_outputs(key, logits, rows, metrics_dir, part="test")


def label_score_rows(key: str, scores: np.ndarray, frame: pd.DataFrame,
                     labels: list[str], label_column: str = "intent",
                     ) -> tuple[dict, pd.DataFrame]:
    """The zero-shot equivalent of score_from_logits(), for label scoring.

    Kept separate from P.score_from_logits() rather than folded into it, because
    the two take different quantities and a function that silently accepts both
    is a function that will one day softmax a log-prob. These are mean log-probs
    per token over 27 candidate strings; they are not a classifier's logits, the
    softmax of them is not a calibrated probability, and `confidence` here is
    exp(best log-prob) - a per-token likelihood, on its own scale.

    The metrics come from E.evaluate_predictions all the same, so the score in
    the table is produced by the identical function as every other row.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(frame), len(labels)):
        raise ValueError(
            f"label scores are {scores.shape}, expected {(len(frame), len(labels))}. "
            "A mismatch here silently pairs each row with another row's scores.")

    best = scores.argmax(axis=1)
    predicted = [labels[i] for i in best]
    metrics = E.evaluate_predictions(frame[label_column], predicted, labels)

    rows = frame.copy().reset_index(drop=True)
    rows["predicted"] = predicted
    rows["confidence"] = np.exp(scores.max(axis=1))
    rows["correct"] = rows[label_column].to_numpy() == np.asarray(predicted)
    return metrics, rows


def frame_from_predictions(frame: pd.DataFrame, predicted, confidence=None,
                           label_column: str = "intent") -> pd.DataFrame:
    """The row-level frame for a model with no score matrix - the majority class.

    Same columns as score_from_logits() returns, so day 6 reads all seven rows
    of the table with one function instead of a special case per baseline.
    `confidence` is genuinely absent for the majority class rather than filled
    with 1.0: a constant model has no notion of it, and inventing one would put
    a number in a column that a later confidence analysis would happily average.
    """
    rows = frame.copy().reset_index(drop=True)
    rows["predicted"] = list(predicted)
    if confidence is not None:
        rows["confidence"] = np.asarray(confidence, dtype=float)
    rows["correct"] = rows[label_column].to_numpy() == np.asarray(list(predicted))
    return rows


# =========================================================================
# 4. THE TABLE, AND THE THREE READINGS
# =========================================================================

def results_table(records: dict) -> pd.DataFrame:
    """The seven rows, in plan order, as the table that goes in the report.

    Rows that have not been run are dropped rather than shown as NaN, so a table
    printed after a partial session says what it measured instead of implying a
    number that does not exist.
    """
    rows = []
    for row in EVALUATION_PLAN:
        record = records.get(row["key"])
        if record is None:
            continue
        rows.append({
            "row": row["label"],
            "trained on": row["trained_on"],
            "train rows": record["config"].get("train_rows"),
            "measured on": row["test"],
            "accuracy": record["metrics"]["accuracy"],
            "macro-F1": record["metrics"]["f1_macro"],
            "role": row["role"],
        })
    return pd.DataFrame(rows)


# The three subtractions the table exists to support. `same_rows` is not a note
# on them - it is what decides whether the noise floor may be applied at all.
READINGS = (
    {
        "key": "headline_inflation",
        "question": "how much the ordinary protocol would have inflated the headline",
        "minuend": "naive",
        "subtrahend": "clean",
        "for": "the abstract. It is the number that makes the project worth doing.",
        "caveat": "the two rows are measured on DIFFERENT test sets, so this "
                  "difference contains 'these are not the same sentences' as well as "
                  "the protocol. It is the honest headline and it is not the "
                  "controlled result - reading 2 is.",
    },
    {
        "key": "controlled_leakage",
        "question": "how much of that survives when training size and test rows are held fixed",
        "minuend": "naive_sub_on_clean",
        "subtrahend": "clean",
        "for": "the claim that can be defended. Same test rows, same training-set "
               "size, and only the leakage moves.",
        "caveat": "expected to be much smaller than reading 1. Reporting the two "
                  "together is what makes the claim credible; reporting only the "
                  "first would be the thing this project exists to criticise.",
    },
    {
        "key": "what_tuning_bought",
        "question": "what LoRA fine-tuning bought over a linear model of negligible cost",
        "minuend": "clean",
        "subtrahend": "tfidf",
        "for": "the question of when a large model is worth bringing at all.",
        "caveat": "read against the noise floor, never alone. The linear baseline "
                  "reached 0.9787 macro-F1 on clean/val, so there was about 0.021 of "
                  "headroom in existence before anything was trained.",
    },
)


def judge(delta: float, floor: float, same_rows: bool) -> dict:
    """Is a difference big enough to report? And is the floor even applicable?

    Two different answers, and conflating them is the mistake this exists to
    prevent. The noise floor measured on day 4 is the spread between three
    training runs of one configuration scored on ONE evaluation set. It says
    what a difference has to clear to be a real difference between models
    measured on those rows. It says nothing at all about a difference between
    two models measured on two different sets of sentences, because that
    difference has a second source the floor never sampled.

    So `applicable` is returned separately from `reportable`, and `reportable`
    is None rather than False when the floor does not apply - "cannot be judged
    by this instrument" and "judged and found too small" must not print alike.
    """
    result = {
        "delta": round(float(delta), 4),
        "floor": round(float(floor), 6),
        "in_floor_units": round(float(abs(delta) / floor), 1) if floor else None,
        "floor_applies": bool(same_rows),
    }
    if not same_rows:
        result["reportable"] = None
        result["verdict"] = ("the two rows are different sentences - day 4's floor "
                             "does not govern this difference")
        return result

    result["reportable"] = bool(abs(delta) > floor)
    result["verdict"] = (
        f"{abs(delta) / floor:.1f}x the noise floor - reportable" if abs(delta) > floor
        else f"{abs(delta) / floor:.1f}x the noise floor - within it, report as no "
             "measurable difference")
    return result


def readings(records: dict, floor: float, metric: str = "f1_macro") -> pd.DataFrame:
    """The three readings, computed, judged, and labelled with their caveats.

    A difference is only computed when both of its rows were actually measured,
    so a partial session produces a shorter table rather than a wrong one.
    """
    rows = []
    for reading in READINGS:
        high, low = records.get(reading["minuend"]), records.get(reading["subtrahend"])
        if high is None or low is None:
            continue
        delta = high["metrics"][metric] - low["metrics"][metric]
        same = (PLAN_BY_KEY[reading["minuend"]]["test"]
                == PLAN_BY_KEY[reading["subtrahend"]]["test"])
        verdict = judge(delta, floor, same)
        rows.append({
            "reading": reading["question"],
            "minus": f"{reading['minuend']} - {reading['subtrahend']}",
            metric: round(delta, 4),
            "same test rows": same,
            "vs noise floor": verdict["verdict"],
            "caveat": reading["caveat"],
        })
    return pd.DataFrame(rows)


def val_test_gap(val_metrics: dict, test_metrics: dict, floor: float,
                 metric: str = "f1_macro") -> dict:
    """val minus test for the headline model. A measurement of the METHOD, not the model.

    This is the number the guide calls free and says most projects do not bother
    to report. It costs one subtraction, both halves already exist, and it is
    the only quantitative evidence available for how hard the hyper-parameter
    search was pushed against the validation set. A large gap means the choices
    were fitted to val; a small one means the search was narrow - which it was,
    on purpose: one axis, three values, and a decision rule registered before
    the sweep ran.

    Reported whichever way it comes out. A gap that embarrasses the method is
    the one worth having written down in advance.
    """
    gap = val_metrics[metric] - test_metrics[metric]
    return {
        "metric": metric,
        "val": val_metrics[metric],
        "test": test_metrics[metric],
        "gap": round(float(gap), 4),
        "floor": round(float(floor), 6),
        "in_floor_units": round(float(abs(gap) / floor), 1) if floor else None,
        "reading": (
            "the test score is BELOW the validation score by more than the noise "
            "floor - some of the validation number was selection, and the size of "
            "the drop is the size of it"
            if gap > floor else
            "the test score is ABOVE the validation score by more than the noise "
            "floor - the validation set was, if anything, the harder sample"
            if gap < -floor else
            "val and test agree to within the noise floor - the choices made "
            "against the validation set did not overfit it measurably"),
    }


# =========================================================================
# 5. WHY THE CONTROL ROW READS THE WAY IT DOES
# =========================================================================

def overlap_report(pairs: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
                   column: str = "instruction") -> pd.DataFrame:
    """Exact text overlap between each training set and each test set it faces.

    One cheap number that makes the control row legible. `naive_sub/train` was
    drawn from a random split of the deduplicated corpus, and `clean/test` holds
    family representatives from the same corpus - so a clean/test sentence can
    sit verbatim in naive_sub/train, and the whole point of the control is what
    that does to the score. Without this column a reader has to take on trust
    that the mechanism named in the caption is the mechanism at work.

    Measured on test rows and computed once, in the session that is allowed to
    read them. It involves no model and decides nothing, which is what keeps it
    on the right side of the rule.
    """
    from . import data as D

    rows = []
    for name, (train, evaluation) in pairs.items():
        rows.append({
            "pair": name,
            "train rows": len(train),
            "test rows": len(evaluation),
            "exact overlap %": round(100 * D.exact_overlap(train, evaluation, column), 2),
        })
    return pd.DataFrame(rows)
