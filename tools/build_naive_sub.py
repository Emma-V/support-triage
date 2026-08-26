#!/usr/bin/env python
"""Writes the size control as a committed, hashed file, verified against the baseline scores.

--------------------------------------------------------------------------
WHY THIS FILE EXISTS

`naive_sub` is `naive/train` cut down to `clean/train`'s 9,893 rows. It is
the control that answers the obvious objection to every naive-versus-clean
number in this project: "the clean score is lower because you trained on
less data". With the row counts equal, that explanation is removed and one
variable remains.

This subsample was previously drawn inline inside `02_baselines.ipynb` and
never written to disk. That was survivable while it fed only a two-second
TF-IDF fit that could be re-run at will. It is not survivable once a model
trains on it and other numbers are compared against that model: a frame
redrawn independently has the right row count and class balance but not
necessarily the same rows, and produces a believable score that cannot be
compared to the original one - with nothing raising anywhere.

--------------------------------------------------------------------------
THE PROOF IS THE SCORE, NOT THE SEED

"seed=42 was passed, so these are the same rows" is an argument about
intent, and it fails silently the moment scikit-learn changes how
`train_test_split` allocates a stratified draw. So this script does not
assert the seed: it refits the original TF-IDF pipeline on the rows it
just wrote and requires the previously committed numbers back, on both
evaluation sets, from results/metrics/baselines_summary.csv.

That is a real test because the baseline notebook also swept the
subsample seed over 42/43/44 and committed all three results. Seeds 43
and 44 land at 0.9847 and 0.9859 macro-F1 on naive/val against seed 42's
0.9874 - far outside any tolerance a rounding difference could hide in. A
wrong draw cannot pass this.

`n_features` is checked alongside the scores and is the sharpest of the
three: the vocabulary surviving `min_df=2` is a direct function of which
rows are in the frame, and it is an integer.

--------------------------------------------------------------------------
    python tools/build_naive_sub.py          # verify, then write
    python tools/build_naive_sub.py --check  # verify only
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import baselines as B  # noqa: E402
from src import data as D       # noqa: E402
from src import evaluate as E   # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
MANIFEST = PROCESSED / "split_manifest.json"
BASELINES = REPO_ROOT / "results" / "metrics" / "baselines_summary.csv"

# The two baseline runs whose rows this frame has to reproduce, and where
# each was scored. Named by their row in baselines_summary.csv rather than
# by their numbers, so the targets are read from the committed table
# instead of typed here.
DAY2_RUNS = (
    ("tfidf_naive_sub_seed42", "naive", "val"),
    ("tfidf_naive_sub_on_cleanval_seed42", "clean", "val"),
)

# The baseline scores were rounded to four decimals before being
# committed, so the comparison is performed at that precision. It is not a
# loose tolerance: the nearest wrong draw (seed 43) differs in the third
# decimal.
PLACES = 4


def day2_targets() -> pd.DataFrame:
    """The committed rows this rebuild has to reproduce."""
    table = pd.read_csv(BASELINES).set_index("run")
    missing = [name for name, _, _ in DAY2_RUNS if name not in table.index]
    if missing:
        raise LookupError(
            f"{missing} are not in {BASELINES.name}. Without the previously "
            "committed baseline numbers there is nothing to verify the draw "
            "against, and an unverified draw is the thing this script exists "
            "to prevent.")
    return table


def score_like_day2(train_frame: pd.DataFrame, eval_frame: pd.DataFrame,
                    labels: list[str]) -> dict:
    """Refits the baseline headline pipeline; its defaults are the original configuration.

    `build_tfidf_pipeline()`'s defaults are the word(1,2) / min_df=2 /
    sublinear_tf configuration used for these two runs, so they are not
    restated here - restating them is how the two implementations would
    drift apart.
    """
    pipe = B.build_tfidf_pipeline()
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UserWarning)
        pipe.fit(train_frame["instruction"], train_frame["intent"])
    assert B.pipeline_converged(pipe), (
        "lbfgs hit max_iter - the score would be an under-estimate and the "
        "comparison against the baseline would fail for a reason unrelated to the rows")

    predicted = pipe.predict(eval_frame["instruction"])
    metrics = E.evaluate_predictions(eval_frame["intent"], predicted, labels)
    metrics["n_features"] = len(pipe.named_steps["tfidf"].vocabulary_)
    return metrics


def verify(frame: pd.DataFrame, splits: dict, labels: list[str]) -> pd.DataFrame:
    """Refits on the drawn rows and requires the previously committed numbers back. Raises if not."""
    targets = day2_targets()
    rows, failures = [], []

    for run_name, split_name, part in DAY2_RUNS:
        want = targets.loc[run_name]
        got = score_like_day2(frame, splits[split_name][part], labels)
        checks = {
            "accuracy": (round(got["accuracy"], PLACES), round(float(want["accuracy"]), PLACES)),
            "macro-F1": (round(got["f1_macro"], PLACES), round(float(want["f1_macro"]), PLACES)),
            "n_features": (got["n_features"], int(want["n_features"])),
        }
        for field, (mine, theirs) in checks.items():
            if mine != theirs:
                failures.append(f"{run_name} {field}: baseline {theirs}, this draw {mine}")
        rows.append({
            "baseline run": run_name,
            "scored on": f"{split_name}/{part}",
            "accuracy": checks["accuracy"][0],
            "baseline accuracy": checks["accuracy"][1],
            "macro-F1": checks["macro-F1"][0],
            "baseline macro-F1": checks["macro-F1"][1],
            "n_features": checks["n_features"][0],
            "baseline n_features": checks["n_features"][1],
        })

    if failures:
        raise AssertionError(
            "these are not the rows the baseline run drew:\n  " + "\n  ".join(failures) +
            "\n\nThe baseline's own seed sweep puts 43 and 44 several thousandths away "
            "from 42, so this is not a rounding difference - it is a different "
            "subsample. Nothing scored against these rows would be comparable "
            "to any number already in the repository. Not written."
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="verify the draw without writing anything")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    splits = D.load_all_splits(PROCESSED)
    D.verify_against_manifest(manifest, splits)
    print("manifest verified - the six split files match their frozen fingerprints")

    labels = json.loads((REPO_ROOT / "artifacts" / "labels.json").read_text(encoding="utf-8"))

    naive_train, clean_train = splits["naive"]["train"], splits["clean"]["train"]
    frame = D.build_naive_sub(naive_train, clean_train, seed=D.SUBSAMPLE_SEED)
    print(f"drew {len(frame):,} rows from naive/train ({len(naive_train):,}) "
          f"at clean/train's size ({len(clean_train):,}), seed {D.SUBSAMPLE_SEED}")

    # The class proportions are what "stratified" is supposed to buy, so they
    # are shown rather than trusted. The ratio is the readable form: macro-F1
    # weights the smallest intent as heavily as the largest.
    counts = frame["intent"].value_counts()
    print(f"  27 intents present: {frame['intent'].nunique() == len(labels)}   "
          f"largest/smallest = {counts.max() / counts.min():.2f}:1")
    print()

    comparison = verify(frame, splits, labels)
    print(comparison.to_string(index=False))
    print("\n[PASS] refitting the baseline pipeline on these rows reproduces the "
          "previously committed scores on both evaluation sets.")

    if args.check:
        print("\n--check: nothing written.")
        return 0

    sub_manifest = D.write_naive_sub(frame, manifest, PROCESSED, seed=D.SUBSAMPLE_SEED)
    print(f"\n  written  data/processed/naive_sub/train.csv   "
          f"{sub_manifest['n_rows']:,} rows")
    print(f"  written  data/processed/naive_sub/subsample_manifest.json")
    print(f"           sha256 {sub_manifest['sha256']}")

    # Read it straight back through the verifying loader. Writing a file
    # and trusting it without reading it back is how the original draw
    # was lost in the first place.
    reloaded = D.load_naive_sub(PROCESSED)
    assert len(reloaded) == len(frame)
    print(f"\n  [PASS] load_naive_sub() reads it back and its sha256 verifies")
    print("\nNow check `git status` actually lists data/processed/naive_sub/.")
    print("Everything under data/processed/ is ignored by default.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
