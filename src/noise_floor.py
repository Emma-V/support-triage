"""
How much the score moves when nothing about the configuration moves.

An earlier r sweep produced three macro-F1 values for r in {4, 8, 16} that
differ by 0.0007, too small to interpret without a reference scale. This
module supplies that scale: run the same configuration several times,
varying only the training seed, and measure how far apart the results land
anyway. Any difference smaller than that spread is not a difference.

TWO SEEDS THAT ARE NOT THE SAME SEED. `SPLIT_SEED` (data.py) decides which
rows are in which split. `TRAIN_SEED` (train.py) decides head
initialisation, LoRA initialisation, shuffling order and dropout masks.
Both happen to be 42, and they are otherwise unrelated; confusing them
would make this measurement describe something else entirely - the
spread would be far larger and would look like training variance. In this
project it is structurally impossible: the split is not rebuilt at run
time, it is read from committed CSVs and hashed against
split_manifest.json before anything trains. `same_configuration` below
checks the fingerprint anyway, since "impossible" and "checked" are
different guarantees.

WHY THREE SEEDS. Three is a small sample, and the standard deviation it
produces is itself a noisy estimate. It does not support a significance
test, and none is claimed. It supports exactly one statement - "a gap
this small is not distinguishable from training variance" - which is a
lower bound on what may be claimed, and is the intended use of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The seeds, fixed here rather than passed in, so that "three runs" is a
# property of the code rather than of an ad hoc choice made at run time.
# Adding a fourth seed informally is the exact drift this constant exists
# to make visible in a diff.
SEEDS = (42, 43, 44)

# The fields that must be identical across the seed runs for their spread to
# mean anything. If any of these differs, the runs vary in more than one thing
# and their standard deviation is measuring the wrong quantity.
MUST_MATCH = (
    "base_model", "r", "lora_alpha", "lora_dropout", "target_modules",
    "modules_to_save", "epochs", "learning_rate", "batch_size", "grad_accum",
    "warmup_ratio", "weight_decay", "max_length", "precision",
    "train_rows", "eval_rows", "train_sha256", "split_seed", "selection_metric",
)


def same_configuration(records: list[dict]) -> pd.DataFrame:
    """Compares the seed runs field by field and reports what differs.

    Checked against the record rather than assumed, since the failure
    being guarded against is precisely the belief that the runs were
    identical. `train_sha256` is the load-bearing row: it is the
    fingerprint of the exact training rows the run saw, so a match proves
    the split did not move underneath the experiment.

    `train_seed` is expected to differ - that is the experiment - so it is
    reported separately rather than as a fault.
    """
    rows = []
    for field in MUST_MATCH:
        values = [record["config"].get(field) for record in records]
        # Lists are unhashable, so compare their repr rather than set() them.
        shown = [repr(v) for v in values]
        rows.append({
            "field": field,
            "identical": len(set(shown)) == 1,
            "value": shown[0] if len(set(shown)) == 1 else " | ".join(shown),
        })
    frame = pd.DataFrame(rows)
    seeds = [record["config"].get("train_seed") for record in records]
    if len(set(seeds)) != len(seeds):
        raise AssertionError(
            f"the seed runs do not have distinct train_seeds: {seeds}. "
            "Two runs with the same seed are the same run and contribute no "
            "information about variance.")
    return frame


def seed_table(records: list[dict], metric: str = "f1_macro") -> pd.DataFrame:
    """One row per seed: the score, the epoch chosen, and what it cost."""
    rows = []
    for record in records:
        rows.append({
            "seed": record["config"]["train_seed"],
            "macro-F1 (val)": record["metrics"]["f1_macro"],
            "accuracy (val)": record["metrics"]["accuracy"],
            "epoch chosen": record["config"]["best_epoch"],
            "runtime (s)": record["runtime_seconds"],
            "run": record["name"],
        })
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def noise_floor(values, ddof: int = 1) -> dict:
    """Mean, standard deviation and range of the seed scores.

    ddof=1 - the sample standard deviation. These three runs are a sample
    from the distribution of runs this recipe could produce, not the
    entire population of them, and at n=3 the difference between dividing
    by 2 and by 3 is not cosmetic.
    """
    values = np.asarray(list(values), dtype=float)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=ddof)),
        "min": float(values.min()),
        "max": float(values.max()),
        "range": float(values.max() - values.min()),
    }


def effective_noise_floor(std: float, per_row: float) -> dict:
    """The larger of the measured seed spread and what one validation row is worth.

    Two distinct factors limit how small a difference may be believed to
    be, and the binding constraint is whichever is larger.

    The first is training variance, measured across seeds. The second is
    the resolution of the metric itself: macro-F1 over a finite validation
    set does not move continuously, and a difference finer than one row
    changing its answer cannot be expressed at all, let alone trusted.

    This is not a hypothetical case at 0.999 accuracy: three runs can land
    on exactly the same score, giving a measured standard deviation of
    0.0000. That does not mean there is no noise. It means the noise is
    below what three runs can resolve, and the metric's own step size
    becomes the honest lower bound instead.

    Taking the maximum makes the decision rule strictly more conservative
    - it can only make a difference harder to call real, never easier.
    That is the safe direction, and it is what allows this rule to be
    settled in advance rather than after seeing which of the two turns out
    to be larger.
    """
    candidates = {"between_seed_std": std, "one_validation_row": per_row}
    usable = {k: v for k, v in candidates.items()
              if v is not None and np.isfinite(v) and v > 0}
    if not usable:
        raise ValueError(
            f"neither ruler is usable: {candidates}. With no measured noise and "
            "no metric granularity there is nothing to judge a gap against.")
    binding = max(usable, key=usable.get)
    return {
        "floor": float(usable[binding]),
        "binding": binding,
        "between_seed_std": std,
        "one_validation_row": per_row,
    }


def metric_granularity(y_true, y_pred, labels: list[str], n_probe: int = 200,
                       seed: int = 0) -> dict:
    """How much macro-F1 moves when exactly one validation row changes answer.

    This is the argument the standard deviation cannot make on its own,
    and at this accuracy level it is the stronger of the two. With 2,120
    rows spread over 27 classes, macro-F1 is not a continuous quantity - it
    moves in visible steps, because flipping one row changes one class's
    recall by 1/support and that class contributes 1/27 of the average.
    Below that step size there is nothing to measure, regardless of what
    the seeds happen to show.

    Measured rather than derived: correctly classified rows are sampled,
    each is flipped to a wrong label in turn, and the resulting drop in
    macro-F1 is recorded. The mean of those drops is what one validation
    row is worth.

    This lets the r sweep be read in units a reader can picture: "the
    spread across r is worth about two validation rows" needs no
    statistics to be understood, and holds regardless of what the seed
    variance turns out to be.
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

    rng = np.random.default_rng(seed)
    probe = rng.choice(correct, size=min(n_probe, len(correct)), replace=False)

    drops = []
    for i in probe:
        # Flip to some other label - the next one round the list, so the choice
        # is deterministic and does not itself need a seed.
        true_label = y_pred[i]
        wrong = labels[(labels.index(true_label) + 1) % len(labels)]
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


def decision_rule(sweep: pd.DataFrame, std: float,
                  metric_column: str = "macro-F1 (val)",
                  r_column: str = "r") -> dict:
    """Applies the pre-registered rule for selecting r from the sweep.

    The rule, in full, fixed before the standard deviation existed:

      1. If the gap between the best and the worst r is smaller than the
         between-seed standard deviation, there is no winner. Choose the
         smallest r - fewest parameters, cheapest, easiest to defend.
      2. If the gap is larger, choose the r with the highest macro-F1 and
         record how many standard deviations ahead it is.

    Returning the rule text alongside the outcome is the point. A rule
    quoted next to the result it produced cannot have been written after
    the fact, and that distinction is what this function exists to
    preserve.
    """
    if std <= 0 or not np.isfinite(std):
        raise ValueError(
            f"std must be a positive finite number, got {std!r}. Without a "
            "noise floor the rule has nothing to compare the gap against.")

    best_row = sweep.loc[sweep[metric_column].idxmax()]
    worst_row = sweep.loc[sweep[metric_column].idxmin()]
    gap = float(best_row[metric_column] - worst_row[metric_column])
    smallest_r = int(sweep[r_column].min())
    distinguishable = gap > std

    if distinguishable:
        chosen = int(best_row[r_column])
        branch = 2
        reason = (f"the gap across r ({gap:.4f}) is larger than the "
                  f"between-seed standard deviation ({std:.4f}), so the "
                  f"difference survives training variance: r={chosen} leads by "
                  f"{gap / std:.1f} standard deviations.")
    else:
        chosen = smallest_r
        branch = 1
        reason = (f"the gap across r ({gap:.4f}) is smaller than the "
                  f"between-seed standard deviation ({std:.4f}), so no r is "
                  f"distinguishable from another: the cheapest is chosen, "
                  f"r={chosen}.")

    # How far past the floor the gap actually sits. This does NOT change
    # which branch is taken - the rule was registered before any run
    # existed, and rewriting its threshold now would defeat the purpose of
    # registering it. It annotates the answer instead, since "larger than
    # the floor" and "comfortably larger than the floor" are different
    # claims and only the second should be reported as an established
    # difference.
    ratio = gap / std
    if not distinguishable:
        margin_note = ("the gap is inside the floor; there is nothing to "
                       "establish and the cheapest configuration wins")
    elif ratio < 2.0:
        margin_note = (
            f"MARGINAL: the gap clears the floor by only {ratio:.1f}x, and that "
            "floor is itself estimated from three runs. The rule's second branch "
            "is triggered, but a difference this close to its own noise should be "
            "reported as suggestive and not as an established improvement.")
    else:
        margin_note = (f"the gap clears the floor by {ratio:.1f}x, which is "
                       "comfortable enough to report as a real difference")

    return {
        "rule": ("gap across r vs the noise floor: gap < floor -> take the "
                 "smallest r; gap > floor -> take the highest-scoring r"),
        "registered": "written before any noise floor existed",
        "gap": gap,
        "std": std,
        "gap_in_std_units": ratio,
        "distinguishable": bool(distinguishable),
        "marginal": bool(distinguishable and ratio < 2.0),
        "margin_note": margin_note,
        "branch_taken": branch,
        "chosen_r": chosen,
        "reason": reason,
    }


def freeze_record(chosen: dict, record: dict, manifest: dict,
                  noise: dict, seeds=SEEDS) -> dict:
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
            "between_seed_std": chosen["std"],
            "distinguishable": chosen["distinguishable"],
            "chosen_r": chosen["chosen_r"],
            "reason": chosen["reason"],
        },
        "noise_floor": {
            "seeds": list(seeds),
            "metric": "f1_macro on clean/val",
            **noise,
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
