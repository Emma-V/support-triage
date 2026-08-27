"""How small a difference this metric can express at all.

An r sweep produced three macro-F1 values for r in {4, 8, 16} that differ
by 0.0007, too small to interpret without a reference scale. This module
supplies that scale.

WHAT THIS MEASURES, AND WHAT IT DOES NOT. The floor here is the
*resolution* of the metric: the amount macro-F1 moves when exactly one
validation row changes its answer. A difference finer than that cannot be
expressed by the metric, let alone believed. It is a property of the
evaluation set - 2,120 rows over 27 classes - and not of training.

It is deliberately NOT a measurement of training variance. Estimating
that would mean training the same configuration several times under
different seeds, and this project trains under exactly one seed
(TRAIN_SEED = 42, the same value as SPLIT_SEED and every other seed
that remains in the repository). So no claim of the form "this difference survives
retraining" is available here, and none is made. The weaker claim that
remains is still the one that matters for the sweep: a gap narrower than
one validation row is not a gap.

Reporting it this way is the honest reading. A resolution floor is a
lower bound on the true floor - real run-to-run variance can only be
larger, never smaller - so every verdict this module produces errs
toward calling differences *unreportable*. That is the safe direction,
and it is why the rule below can be fixed in advance rather than after
seeing the numbers.

NO RANDOMNESS. Nothing in this module draws a random number. The
granularity probe walks a deterministic stride over the correctly
classified rows, so re-running it returns the same floor without
depending on any seed at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# How many rows the granularity probe flips. The probe is O(n_probe) calls
# to f1_score over the full validation set, so this trades runtime against
# the stability of the mean. 200 is enough for the mean drop to settle to
# the precision that is reported.
N_PROBE = 200


def metric_granularity(y_true, y_pred, labels: list[str],
                       n_probe: int = N_PROBE) -> dict:
    """How much macro-F1 moves when exactly one validation row changes answer.

    With 2,120 rows spread over 27 classes, macro-F1 is not a continuous
    quantity - it moves in visible steps, because flipping one row changes
    one class's recall by 1/support and that class contributes 1/27 of the
    average. Below that step size there is nothing to measure.

    Measured rather than derived: correctly classified rows are walked in a
    deterministic stride, each is flipped to a wrong label in turn, and the
    resulting drop in macro-F1 is recorded. The mean of those drops is what
    one validation row is worth.

    The stride, rather than a random sample, is what keeps this seedless.
    Every row is equally eligible and the selection is reproducible from the
    inputs alone, so the floor does not depend on a seed that would then have
    to be recorded and defended.

    This lets the r sweep be read in units a reader can picture: "the spread
    across r is worth about one and a half validation rows" needs no
    statistics to be understood.
    """
    from sklearn.metrics import f1_score

    y_true = list(y_true)
    y_pred = list(y_pred)
    labels = list(labels)
    base = f1_score(y_true, y_pred, labels=labels, average="macro",
                    zero_division=0)

    correct = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t == p]
    if not correct:
        return {"base_f1_macro": float(base), "n_probed": 0,
                "mean_drop_per_row": float("nan"),
                "min_drop": float("nan"), "max_drop": float("nan")}

    # Deterministic stride across the correct rows: take every k-th one so
    # the probe spans the whole set rather than clustering at the front,
    # and does so without drawing a random number.
    step = max(1, len(correct) // n_probe)
    probe = correct[::step][:n_probe]

    drops = []
    for i in probe:
        # Flip to some other label - the next one round the list, so the
        # choice is deterministic and does not itself need a seed.
        predicted = y_pred[i]
        wrong = labels[(labels.index(predicted) + 1) % len(labels)]
        flipped = y_pred.copy()
        flipped[i] = wrong
        drops.append(base - f1_score(y_true, flipped, labels=labels,
                                     average="macro", zero_division=0))

    drops = np.asarray(drops, dtype=float)
    return {
        "base_f1_macro": float(base),
        "n_probed": int(drops.size),
        "mean_drop_per_row": float(drops.mean()),
        "min_drop": float(drops.min()),
        "max_drop": float(drops.max()),
    }


def resolution_floor(granularity: dict) -> dict:
    """The floor itself, with the reasoning that produced it attached.

    A thin wrapper over one number on purpose. The floor is a claim about
    what may be believed, and a claim travels better with its own
    justification and its own stated limit than as a bare float.
    """
    per_row = granularity["mean_drop_per_row"]
    if per_row is None or not np.isfinite(per_row) or per_row <= 0:
        raise ValueError(
            f"the granularity probe returned {per_row!r}, which cannot be a "
            "floor. With no metric resolution there is nothing to judge a gap "
            "against.")
    return {
        "floor": float(per_row),
        "basis": "one_validation_row",
        "measured_on": f"{granularity['n_probed']} single-row flips",
        "means": ("the smallest difference macro-F1 can express on this "
                  "validation set; differences below it are not differences"),
        "limit": ("this is metric resolution, not training variance - the "
                  "project trains under one seed, so run-to-run spread was "
                  "never measured and the true floor can only be larger"),
    }


def decision_rule(sweep: pd.DataFrame, floor: float,
                  metric_column: str = "macro-F1 (val)",
                  r_column: str = "r") -> dict:
    """Applies the pre-registered rule for selecting r from the sweep.

    The rule, in full, fixed before the floor existed:

      1. If the gap between the best and the worst r is smaller than the
         floor, there is no winner. Choose the smallest r - fewest
         parameters, cheapest, easiest to defend.
      2. If the gap is larger, choose the r with the highest macro-F1 and
         record how far past the floor it sits.

    Returning the rule text alongside the outcome is the point. A rule
    quoted next to the result it produced cannot have been written after
    the fact, and that distinction is what this function exists to
    preserve.
    """
    if floor <= 0 or not np.isfinite(floor):
        raise ValueError(
            f"floor must be a positive finite number, got {floor!r}. Without a "
            "floor the rule has nothing to compare the gap against.")

    best_row = sweep.loc[sweep[metric_column].idxmax()]
    worst_row = sweep.loc[sweep[metric_column].idxmin()]
    gap = float(best_row[metric_column] - worst_row[metric_column])
    smallest_r = int(sweep[r_column].min())
    distinguishable = gap > floor

    if distinguishable:
        chosen = int(best_row[r_column])
        branch = 2
        reason = (f"the gap across r ({gap:.4f}) is larger than the "
                  f"resolution floor ({floor:.4f}), so the difference is at "
                  f"least expressible by the metric: r={chosen} leads by "
                  f"{gap / floor:.1f} validation rows' worth.")
    else:
        chosen = smallest_r
        branch = 1
        reason = (f"the gap across r ({gap:.4f}) is smaller than the "
                  f"resolution floor ({floor:.4f}), so no r is "
                  f"distinguishable from another: the cheapest is chosen, "
                  f"r={chosen}.")

    # How far past the floor the gap actually sits. This does NOT change
    # which branch is taken - the rule was registered before any run
    # existed, and rewriting its threshold now would defeat the purpose of
    # registering it. It annotates the answer instead, since "larger than
    # the floor" and "comfortably larger than the floor" are different
    # claims and only the second should be reported as an established
    # difference.
    ratio = gap / floor
    if not distinguishable:
        margin_note = ("the gap is inside the floor; there is nothing to "
                       "establish and the cheapest configuration wins")
    elif ratio < 2.0:
        margin_note = (
            f"MARGINAL: the gap clears the floor by only {ratio:.1f}x, and the "
            "floor is metric resolution rather than measured training variance, "
            "which can only be larger. The rule's second branch is triggered, "
            "but a difference this close to the smallest expressible one should "
            "be reported as suggestive and not as an established improvement.")
    else:
        margin_note = (f"the gap clears the floor by {ratio:.1f}x, which is "
                       "comfortable enough to report as a real difference")

    return {
        "rule": ("gap across r vs the resolution floor: gap < floor -> take "
                 "the smallest r; gap > floor -> take the highest-scoring r"),
        "registered": "written before any floor existed",
        "gap": gap,
        "floor": floor,
        "gap_in_row_units": ratio,
        "distinguishable": bool(distinguishable),
        "marginal": bool(distinguishable and ratio < 2.0),
        "margin_note": margin_note,
        "branch_taken": branch,
        "chosen_r": chosen,
        "reason": reason,
    }


def freeze_record(chosen: dict, record: dict, manifest: dict,
                  floor: dict) -> dict:
    """Builds the configuration freeze, as a file rather than an intention.

    From the moment this is written the hyper-parameters do not move.
    Changing one afterwards breaks the freeze, and breaking it has to be
    written down - runs after that point belong to a second configuration
    and are not comparable to the ones before it.

    Everything here is copied out of a run record rather than retyped, so
    the freeze cannot disagree with the run it claims to describe.
    """
    from datetime import datetime, timezone

    config = record["config"]
    return {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen_from_run": record["name"],
        "selection": {
            "rule": chosen["rule"],
            "registered": chosen["registered"],
            "gap_across_r": chosen["gap"],
            "resolution_floor": chosen["floor"],
            "distinguishable": chosen["distinguishable"],
            "chosen_r": chosen["chosen_r"],
            "reason": chosen["reason"],
        },
        "resolution_floor": {
            "metric": "f1_macro on clean/val",
            "seed": config["train_seed"],
            **floor,
        },
        "model": {
            "base_model": config["base_model"],
            "task_type": config["task_type"],
            "r": config["r"],
            "lora_alpha": config["lora_alpha"],
            "lora_dropout": config["lora_dropout"],
            "target_modules": config["target_modules"],
            "modules_to_save": config["modules_to_save"],
        },
        "training": {
            "learning_rate": config["learning_rate"],
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "grad_accum": config["grad_accum"],
            "warmup_ratio": config["warmup_ratio"],
            "weight_decay": config["weight_decay"],
            "precision": config["precision"],
            "max_length": config["max_length"],
            "selection_metric": config["selection_metric"],
            "train_seed_of_frozen_run": config["train_seed"],
        },
        "data": {
            "trained_on": config["trained_on"],
            "scored_on": config["scored_on"],
            "train_rows": config["train_rows"],
            "eval_rows": config["eval_rows"],
            "train_sha256": config["train_sha256"],
            "split_seed": config["split_seed"],
            "manifest_clean_train_sha256":
                manifest["splits"]["clean"]["train"]["sha256"],
            "labels": "artifacts/labels.json, order frozen",
        },
        "hardware": {
            "gpu_name": config["gpu_name"],
            "precision": config["precision"],
        },
        "still_open": [
            "the confidence threshold for routing - an operational decision, "
            "not a training hyper-parameter, and calibratable later",
            "which slices and figures the analysis cuts",
        ],
        "breaking_this_freeze": (
            "requires a written note naming what changed and why; runs after "
            "such a change belong to configuration 2 and are not comparable "
            "to the ones before it"
        ),
    }
