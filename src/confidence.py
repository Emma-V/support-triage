"""
Confidence scores, and what they are worth as evidence.

A macro-F1 of 0.9994 identifies which predictions were right. It says
nothing about the number printed beside each one, and three questions hang
on that number: whether the confidence is calibrated, whether a `top3` list
is worth returning when the model hesitates, and at what score a ticket
should stop being answered automatically and be routed to a human instead.
This module measures all three.

Everything here runs on a saved logit matrix and never touches a GPU. That
is a deliberate consequence of losing an earlier run's predictions when a
disposable Colab clone died with the runtime: re-answering a new question
about those predictions then required paying for the model again. A
committed (n_rows, n_labels) array of logits is 230 KB and answers every
question below indefinitely, on a laptop.

Logits are saved rather than probabilities for one specific reason.
Temperature scaling divides the logits before the softmax, so it cannot be
done from probabilities without taking their log - and at this model's
confidence level a float32 probability underflows to exactly 0.0, whose log
is -inf. Saving the pre-softmax layer keeps every later question answerable.

WHAT CONFIDENCE IS NOT. A high value here means "of the 27 intents, this
one is most likely", not "this ticket is one of the 27". A message about
something the taxonomy has no label for still receives a label, possibly a
confident one - a classification head has no way to express "none of
these". This is a stated limit of the system as built. Where to place the
routing cutoff is an operational decision left open here; this module
measures what the confidence number is worth, not what to do with it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Bins the calibration table uses unless told otherwise. Ten equal-width
# bins on [0, 1] is the convention used in the ECE literature, adopted here
# so the number is comparable to values reported elsewhere.
N_CALIBRATION_BINS = 10

# Thresholds the coverage-accuracy curve is swept over. Dense near 1.0
# because a fine-tuned model puts most of its mass there; a curve sampled
# at 0.1 intervals would be flat with all the interesting behaviour hidden
# inside the last point.
COVERAGE_THRESHOLDS = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9999)


def probabilities(logits: np.ndarray) -> np.ndarray:
    """Softmax over the last axis, in float64, shift-stabilised.

    float64 is required rather than merely cautious: the gap between the
    top logit and the rest is large after fine-tuning, and in float32 the
    tail of the distribution rounds to zero, which silently sets the
    entropy to exactly 0 and makes every ticket look equally unambiguous.
    """
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def row_frame(logits: np.ndarray, y_true, labels: list[str],
              texts=None, row_ids=None) -> pd.DataFrame:
    """Builds one row per ticket, with three measures of prediction certainty.

    All three are kept because they disagree in exactly the cases worth
    finding:

    - `confidence` is the top probability, the value the predict() contract
      exposes.
    - `margin` is first minus second. A ticket at 0.55 with a runner-up at
      0.44 is a genuine two-way tie; one at 0.55 with a runner-up at 0.03
      is not, a distinction `confidence` alone cannot make.
    - `entropy` is the spread over all 27 labels, and is the only measure
      that detects the case where no candidate stands out at all, rather
      than two standing out equally.

    `top3` is included because the deployment contract promises it, and the
    gap between top-1 and top-3 accuracy determines whether that field
    earns its place in the output.
    """
    probs = probabilities(logits)
    labels = list(labels)
    order = np.argsort(-probs, axis=1)

    top1, top2, top3 = order[:, 0], order[:, 1], order[:, 2]
    p_sorted = np.take_along_axis(probs, order[:, :3], axis=1)

    y_true = np.asarray(y_true, dtype=object)
    predicted = np.array([labels[i] for i in top1], dtype=object)

    # log(0) is -inf and 0 * -inf is nan, so the zero terms are masked
    # rather than nudged with an epsilon, which would quietly change the
    # number.
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probs > 0, probs * np.log(probs), 0.0)
    entropy = -terms.sum(axis=1)

    frame = pd.DataFrame({
        "true": y_true,
        "predicted": predicted,
        "correct": predicted == y_true,
        "confidence": p_sorted[:, 0],
        "margin": p_sorted[:, 0] - p_sorted[:, 1],
        "entropy": entropy,
        "top1": predicted,
        "top2": [labels[i] for i in top2],
        "top3": [labels[i] for i in top3],
        "p2": p_sorted[:, 1],
        "p3": p_sorted[:, 2],
    })
    frame["top3_hit"] = [
        t in (a, b, c) for t, a, b, c in
        zip(frame["true"], frame["top1"], frame["top2"], frame["top3"])
    ]
    if texts is not None:
        frame.insert(0, "instruction", list(texts))
    if row_ids is not None:
        frame.insert(0, "row_id", list(row_ids))
    return frame


def topk_accuracy(logits: np.ndarray, y_true, labels: list[str],
                  ks=(1, 2, 3, 5)) -> pd.DataFrame:
    """Reports accuracy when the true label is allowed anywhere in the top k.

    This determines whether the `top3` field is worth returning. If top-3
    accuracy is far above top-1, the runner-up carries information and a
    human handed three candidates is better served than one handed a
    single answer. If the two are equal to three decimal places, the field
    is honest but uninformative, which is itself a finding worth reporting.
    """
    probs = probabilities(logits)
    labels = list(labels)
    index = {label: i for i, label in enumerate(labels)}
    truth = np.array([index[t] for t in y_true])
    order = np.argsort(-probs, axis=1)

    rows = []
    for k in ks:
        hit = (order[:, :k] == truth[:, None]).any(axis=1)
        rows.append({"k": k, "accuracy": float(hit.mean()),
                     "errors": int((~hit).sum())})
    return pd.DataFrame(rows)


def coverage_accuracy_curve(logits: np.ndarray, y_true, labels: list[str],
                            thresholds=COVERAGE_THRESHOLDS,
                            score: str = "confidence") -> pd.DataFrame:
    """Reports how many tickets clear a confidence threshold, and how well.

    This turns a human-in-the-loop routing policy into two numbers: at
    threshold t the system answers `coverage` of the tickets at
    `accuracy_covered`, and the rest are routed to a person.

    `accuracy_abstained` is reported alongside it. If the tickets held back
    are answered correctly just as often as the ones let through, the
    threshold is not selecting for doubt - it is discarding work for
    nothing, a failure mode this curve would otherwise obscure.

    Thresholds are applied with `>=`, so 0.0 selects the whole set and is
    included deliberately: it is the row reporting unfiltered accuracy.
    """
    frame = row_frame(logits, y_true, labels)
    if score not in {"confidence", "margin"}:
        raise ValueError(f"score must be 'confidence' or 'margin', got {score!r}")
    values = frame[score].to_numpy()
    correct = frame["correct"].to_numpy()
    n = len(frame)

    rows = []
    for t in thresholds:
        keep = values >= t
        n_kept = int(keep.sum())
        rows.append({
            "threshold": t,
            "coverage": n_kept / n,
            "n_answered": n_kept,
            "n_deferred": n - n_kept,
            "accuracy_covered": float(correct[keep].mean()) if n_kept else float("nan"),
            "accuracy_abstained": (float(correct[~keep].mean())
                                   if n_kept < n else float("nan")),
            "errors_covered": int((~correct[keep]).sum()),
        })
    return pd.DataFrame(rows)


def calibration_table(logits: np.ndarray, y_true, labels: list[str],
                      n_bins: int = N_CALIBRATION_BINS) -> pd.DataFrame:
    """Bins tickets by confidence and compares each bin's claim to its outcome.

    A calibrated model that reports 0.9 is correct about 90% of the time it
    says so. Accuracy and calibration are distinct properties, and a model
    can have one without the other - relevant here because every routing
    threshold is read off the confidence number, so a number that overstates
    itself sends tickets to customers that should have gone to a person.

    Empty bins are dropped rather than reported as zero: a bin nothing
    landed in has no defined accuracy, and reporting 0.0 there would read as
    a failure that did not occur.
    """
    frame = row_frame(logits, y_true, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right=True so the top bin is closed and a confidence of exactly 1.0
    # lands inside it rather than falling outside every bin.
    which = np.clip(np.digitize(frame["confidence"], edges[1:-1], right=True),
                    0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = which == b
        if not mask.any():
            continue
        confidence = float(frame.loc[mask, "confidence"].mean())
        accuracy = float(frame.loc[mask, "correct"].mean())
        rows.append({
            "bin": f"({edges[b]:.2f}, {edges[b + 1]:.2f}]",
            "n": int(mask.sum()),
            "mean_confidence": confidence,
            "accuracy": accuracy,
            "gap": confidence - accuracy,   # positive = overconfident
        })
    return pd.DataFrame(rows)


def expected_calibration_error(logits: np.ndarray, y_true, labels: list[str],
                               n_bins: int = N_CALIBRATION_BINS) -> dict:
    """Computes ECE, MCE, and the signed average gap.

    ECE averages |confidence - accuracy| over the bins, weighted by bin
    size. It is the number most commonly quoted, but on its own does not
    say which direction the model is wrong in. `signed_gap` does: positive
    means the model claims more than it delivers, negative means it is
    needlessly modest, and only the former is a risk for routing decisions.
    """
    table = calibration_table(logits, y_true, labels, n_bins)
    if table.empty:
        return {"ece": float("nan"), "mce": float("nan"),
                "signed_gap": float("nan"), "n_bins_used": 0}
    weight = table["n"] / table["n"].sum()
    return {
        "ece": float((weight * table["gap"].abs()).sum()),
        "mce": float(table["gap"].abs().max()),
        "signed_gap": float((weight * table["gap"]).sum()),
        "n_bins_used": int(len(table)),
    }


def fit_temperature(logits: np.ndarray, y_true, labels: list[str],
                    bounds: tuple[float, float] = (0.05, 20.0)) -> dict:
    """Fits a single temperature parameter by minimising negative log-likelihood.

    Temperature scaling divides every logit by a single positive number
    before the softmax. T > 1 flattens the distribution and reduces
    confidence; T < 1 sharpens it. Because it is a monotone transform
    applied identically to all 27 logits, it cannot change which label is
    the argmax - accuracy, macro-F1, and every confusion are bit-identical
    afterwards. Only the confidence values move.

    That property is what makes it safe to fit here: it is fitted on
    clean/val while the test set remains sealed, making it part of the
    model rather than part of the evaluation. Fitting it after seeing the
    test set would be a leak with no way to undo afterwards.

    Fitted on NLL rather than directly on ECE because NLL is smooth and
    convex in 1/T, so the optimiser converges to the same answer every
    time; ECE is a step function of the bin edges, which an optimiser can
    chase as noise.
    """
    from scipy.optimize import minimize_scalar

    logits = np.asarray(logits, dtype=np.float64)
    index = {label: i for i, label in enumerate(labels)}
    truth = np.array([index[t] for t in y_true])
    rows = np.arange(len(truth))

    def nll(temperature: float) -> float:
        scaled = logits / temperature
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        log_norm = np.log(np.exp(scaled).sum(axis=1))
        return float(-(scaled[rows, truth] - log_norm).mean())

    result = minimize_scalar(nll, bounds=bounds, method="bounded")
    temperature = float(result.x)

    before = expected_calibration_error(logits, y_true, labels)
    after = expected_calibration_error(logits / temperature, y_true, labels)
    return {
        "temperature": temperature,
        "nll_before": nll(1.0),
        "nll_after": float(result.fun),
        "ece_before": before["ece"],
        "ece_after": after["ece"],
        "signed_gap_before": before["signed_gap"],
        "signed_gap_after": after["signed_gap"],
        "direction": ("overconfident, softened" if temperature > 1
                      else "underconfident, sharpened"),
    }
