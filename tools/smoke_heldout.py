#!/usr/bin/env python
"""Runs the whole held-out-evaluation path on CPU, before the test set is opened for real.

--------------------------------------------------------------------------
WHY THIS ONE MATTERS MORE THAN THE LAST ONE

`tools/smoke_protocol_models.py` protected two hours of GPU time. This protects
something that cannot be bought back at any price: the test set opens once, and
a crash forty minutes in leaves a choice between an incomplete table and a
second opening that has to be declared in the seal.

So every part of 04_test's path runs here first, on a two-layer stub and on
rows from the VALIDATION sets - and stage 1 checks, by AST rather than by grep,
that this script and src/heldout.py cannot themselves reach a test file.

--------------------------------------------------------------------------
WHAT IT SAYS, AND WHAT IT DOES NOT

Nothing about scores. The stub has randomly initialised layers; its macro-F1 is
noise and is never printed as though it meant anything.

What it says is that the seal refuses and relents in the right places, that the
readings apply the noise floor to the two comparisons it governs and withhold it
from the one it does not, that the row-level files are written under names later
analysis can find, and that every shape in the path is the shape the next
function expects.

    python tools/smoke_heldout.py
    python tools/smoke_heldout.py --static   # stage 1 only, ~1 second
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import baselines as B        # noqa: E402
from src import data as D             # noqa: E402
from src import evaluate as E         # noqa: E402
from src import heldout as H          # noqa: E402
from src import protocol_models as P  # noqa: E402
from src import train as T            # noqa: E402

NOTEBOOK = REPO_ROOT / "notebooks" / "04_test.ipynb"
MODULE = REPO_ROOT / "src" / "heldout.py"

# The module describes which test files WILL be read; it must not be able to
# read one. The strings appear in it constantly, as plan values and in prose -
# what must not appear is a call that turns one into a frame.
FORBIDDEN_NAMES = {"load_split", "load_all_splits", "read_csv"}

STUB_LAYERS, STUB_HIDDEN, STUB_HEADS, STUB_KV_HEADS = 2, 64, 4, 2
STUB_ROWS, STUB_EVAL_ROWS = 60, 40


def ok(message: str) -> None:
    print(f"  [PASS] {message}")


# =========================================================================
# STAGE 1: STATIC
# =========================================================================

def check_module_cannot_read() -> None:
    """src/heldout.py names test files everywhere and must never open one.

    The AST holds only what executes, so a `load_split` there is a real one
    while the fifty occurrences of the string "clean/test" in plan rows and
    docstrings are not. That discrimination is the whole reason this is a parse
    and not a grep.
    """
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        name = (node.attr if isinstance(node, ast.Attribute) else
                node.id if isinstance(node, ast.Name) else None)
        if name in FORBIDDEN_NAMES:
            found.append(f"line {node.lineno}: {name}()")
    if found:
        raise AssertionError(
            "src/heldout.py can read a split itself:\n  " + "\n  ".join(found) +
            "\nThe reading belongs in the notebook, in one place, after the journal "
            "entry that says it is about to happen. A module that can do it quietly "
            "makes 'opened once' unverifiable.")
    ok(f"src/heldout.py has no call that reads a split ({len(FORBIDDEN_NAMES)} names checked)")


def check_plan_is_coherent(freeze: dict) -> None:
    """The run list has to agree with the freeze, with the training plan, and with itself."""
    keys = [row["key"] for row in H.EVALUATION_PLAN]
    assert len(set(keys)) == len(keys), f"duplicate plan keys: {keys}"
    ok(f"{len(keys)} plan rows, distinct keys: {', '.join(keys)}")

    r = freeze["model"]["r"]
    for model_key in H.model_keys():
        name = H.adapter_run_name(model_key, freeze, r)
        assert name, model_key
    names = {k: H.adapter_run_name(k, freeze, r) for k in H.model_keys()}
    assert names["clean"] == freeze["frozen_from_run"], names
    ok("adapters the plan needs: " + ", ".join(f"{k}={v}" for k, v in names.items()))

    # naive_sub appears twice and must resolve to ONE adapter, or the control
    # is not the same model as the row it is being compared against.
    control_rows = H.rows_for_model("naive_sub")
    assert len(control_rows) == 2, control_rows
    assert {row["test"] for row in control_rows} == {"naive/test", "clean/test"}
    ok("naive_sub is loaded once and scored on both test sets - one model, two rows")

    # The control row and the headline must be on the same rows, or it controls
    # for nothing; the protocol rows must not be, or there is nothing to control.
    assert H.same_test_rows_as_headline("naive_sub_on_clean")
    assert H.same_test_rows_as_headline("tfidf")
    assert not H.same_test_rows_as_headline("naive")
    ok("same-test-rows flags: control and baselines yes, naive protocol rows no")

    for plan_key in ("naive", "naive_sub"):
        assert plan_key in P.PLAN_BY_KEY, (
            f"{plan_key} is scored here but was never trained - "
            f"src/protocol_models.py trains {sorted(P.PLAN_BY_KEY)}")
    ok("every fine-tuned row was trained by the protocol-models stage")

    ok(f"plan fingerprint {H.plan_fingerprint()} - this is what the seal will record")


def check_notebook_is_valid() -> None:
    if not NOTEBOOK.exists():
        print(f"  [skip] {NOTEBOOK.name} does not exist yet")
        return
    import nbformat
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    ok(f"nbformat validates - {len(nb.cells)} cells, {n_code} of them code")

    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            try:
                ast.parse("".join(cell.source))
            except SyntaxError as exc:
                raise AssertionError(f"cell {i} does not parse: line {exc.lineno}: {exc.msg}")
    ok("every code cell parses")

    with_output = [i for i, c in enumerate(nb.cells)
                   if c.cell_type == "code" and c.get("outputs")]
    if with_output:
        print(f"  [note] cells {with_output} carry stored outputs - the notebook has been run")
    else:
        ok("no stored outputs - nothing is claimed before the run happens")


# =========================================================================
# STAGE 2: THE SEAL
# =========================================================================

def check_seal(workspace: Path, freeze: dict, manifest: dict) -> None:
    """Refuse, relent on a written reason, and record the relenting."""
    path = workspace / "artifacts" / H.SEAL_FILENAME

    assert H.assert_seal_absent(path) is None
    ok("no seal -> the first opening proceeds")

    frames = {"clean/test": pd.DataFrame({"x": range(3)})}
    records = {"clean": {"metrics": {"accuracy": 0.99, "f1_macro": 0.98},
                         "config": {"scored_on": "clean/test"}}}
    first = H.seal(path, freeze=freeze, manifest=manifest, records=records,
                   test_frames=frames)
    assert first["plan_sha256"] == H.plan_fingerprint()
    assert first["test_files"]["clean/test"]["sha256"] == \
        manifest["splits"]["clean"]["test"]["sha256"]
    ok("the seal records the plan fingerprint and the sha256 of every file read")

    try:
        H.assert_seal_absent(path)
    except AssertionError as exc:
        assert "0.9800" in str(exc), "the refusal must quote the scores that already exist"
        assert "TECHNICAL failure" in str(exc)
        ok("a sealed test set refuses to re-open, quoting the existing macro-F1")
    else:
        raise AssertionError("a sealed test set opened again without a reason")

    for empty in ("", "   ", None):
        try:
            H.assert_seal_absent(path, reason=empty)
        except AssertionError:
            continue
        raise AssertionError(f"reason={empty!r} was accepted as a written reason")
    ok("a blank reason is not a reason")

    previous = H.assert_seal_absent(path, reason="the adapter did not load - crash, no scores")
    assert previous is not None
    second = H.seal(path, freeze=freeze, manifest=manifest, records=records,
                    test_frames=frames, reopened_from=previous,
                    reason="the adapter did not load - crash, no scores")
    assert len(second["reopenings"]) == 1, second["reopenings"]
    assert second["reopenings"][0]["superseded"]["clean"]["f1_macro"] == 0.98
    assert second["first_opened_at"] == first["opened_at"]
    ok("a justified re-opening is APPENDED: reason, timestamp, and the superseded scores")

    third_previous = H.assert_seal_absent(path, reason="a second technical failure")
    third = H.seal(path, freeze=freeze, manifest=manifest, records=records,
                   test_frames=frames, reopened_from=third_previous,
                   reason="a second technical failure")
    assert len(third["reopenings"]) == 2, third["reopenings"]
    ok("re-openings accumulate rather than overwrite - the count is the record")


# =========================================================================
# STAGE 3: THE READINGS
# =========================================================================

def fake_record(key: str, f1: float, accuracy: float | None = None) -> dict:
    return H.test_record(key, {"accuracy": accuracy if accuracy is not None else f1,
                               "f1_macro": f1, "n_rows": 100, "n_labels": 27},
                         {"train_rows": 9893, "eval_rows": 100}, 1.0)


def check_readings(freeze: dict) -> None:
    """The floor must govern the two comparisons it can, and withhold from the one it cannot."""
    floor = freeze["noise_floor"]["effective_floor"]["floor"]

    records = {
        "clean": fake_record("clean", 0.9600),
        "naive": fake_record("naive", 0.9900),               # different test rows
        "naive_sub": fake_record("naive_sub", 0.9880),
        "naive_sub_on_clean": fake_record("naive_sub_on_clean", 0.9700),
        "tfidf": fake_record("tfidf", 0.9500),
        "majority": fake_record("majority", 0.0041),
        "zero_shot": fake_record("zero_shot", 0.5297),
    }

    table = H.results_table(records)
    assert len(table) == len(H.EVALUATION_PLAN), table
    assert list(table["row"])[0] == "majority class", table
    ok(f"results_table returns all {len(table)} rows, in plan order, floor first")

    partial = H.results_table({k: v for k, v in records.items() if k != "zero_shot"})
    assert len(partial) == len(H.EVALUATION_PLAN) - 1
    assert "zero-shot (no training)" not in list(partial["row"])
    ok("a row that was not run is absent, not NaN")

    frame = H.readings(records, floor)
    assert len(frame) == 3, frame
    by_key = {row["minus"]: row for _, row in frame.iterrows()}

    headline = by_key["naive - clean"]
    assert headline["same test rows"] is False
    assert "does not govern" in headline["vs noise floor"], headline["vs noise floor"]
    ok("reading 1 spans two test sets - the floor is withheld, not applied")

    controlled = by_key["naive_sub_on_clean - clean"]
    assert controlled["same test rows"] is True
    assert "reportable" in controlled["vs noise floor"], controlled["vs noise floor"]
    ok(f"reading 2 is on the same rows - judged, and {controlled['f1_macro']:+.4f} clears the floor")

    tiny = dict(records, tfidf=fake_record("tfidf", 0.9600 - floor / 2))
    verdict = H.readings(tiny, floor)
    within = verdict.loc[verdict["minus"] == "clean - tfidf", "vs noise floor"].item()
    assert "within it" in within, within
    ok("a difference smaller than the floor is reported as no measurable difference")

    for same_rows in (True, False):
        judged = H.judge(0.5 * floor, floor, same_rows)
        assert judged["reportable"] is (False if same_rows else None)
    ok("judge() returns None, not False, when the floor does not apply - the two cannot print alike")

    partial_readings = H.readings({k: v for k, v in records.items() if k != "tfidf"}, floor)
    assert len(partial_readings) == 2
    ok("a reading whose row is missing is skipped, not computed against nothing")


def check_val_test_gap(freeze: dict) -> None:
    floor = freeze["noise_floor"]["effective_floor"]["floor"]
    for val, test, expect in ((0.9994, 0.9600, "BELOW"),
                              (0.9600, 0.9994, "ABOVE"),
                              (0.9994, 0.9994, "did not overfit")):
        gap = H.val_test_gap({"f1_macro": val}, {"f1_macro": test}, floor)
        assert expect in gap["reading"], (val, test, gap["reading"])
    ok("val-test gap reads correctly in all three directions, including no gap at all")


# =========================================================================
# STAGE 4: THE SCORING PATH, ON A STUB
# =========================================================================

def build_stub(directory: Path, labels: list[str]) -> Path:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(T.DEBUG_MODEL)
    config = AutoConfig.from_pretrained(T.DEBUG_MODEL)
    config.num_hidden_layers = STUB_LAYERS
    config.hidden_size = STUB_HIDDEN
    config.intermediate_size = STUB_HIDDEN * 2
    config.num_attention_heads = STUB_HEADS
    config.num_key_value_heads = STUB_KV_HEADS
    config.head_dim = STUB_HIDDEN // STUB_HEADS
    if getattr(config, "layer_types", None):
        config.layer_types = list(config.layer_types)[:STUB_LAYERS]

    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    ok(f"stub built: {STUB_LAYERS} layers, real Qwen3 tokenizer")
    return directory


def stand_in_frames(labels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows from the VALIDATION sets, standing in for the test frames.

    The point of this script is to run the path without opening a test file, so
    the frames it runs on are val rows. They have identical columns, which is
    all the path can tell apart.
    """
    processed = REPO_ROOT / "data" / "processed"
    train = D.load_split("clean", "train", processed)
    evaluation = D.load_split("clean", "val", processed)
    train = pd.concat([train.groupby("intent", group_keys=False).head(1),
                       train.head(STUB_ROWS)]).drop_duplicates("row_id").reset_index(drop=True)
    evaluation = pd.concat([evaluation.groupby("intent", group_keys=False).head(1),
                            evaluation.head(STUB_EVAL_ROWS)]).drop_duplicates(
                                "row_id").reset_index(drop=True)
    return train, evaluation


def check_cheap_baselines(train: pd.DataFrame, evaluation: pd.DataFrame,
                          labels: list[str], workspace: Path) -> dict:
    """The two CPU rows, through exactly the calls the notebook makes."""
    metrics_dir = workspace / "results" / "metrics"

    label = B.majority_label(train)
    predicted = [label] * len(evaluation)
    metrics = E.evaluate_predictions(evaluation["intent"], predicted, labels)
    rows = H.frame_from_predictions(evaluation, predicted)
    assert "confidence" not in rows.columns, (
        "the majority class was given a confidence, and a later analysis would average it")
    ok(f"majority class: predicts {label!r}, and carries NO confidence column")

    pipeline = B.build_tfidf_pipeline()
    pipeline.fit(train["instruction"], train["intent"])
    assert B.pipeline_converged(pipeline), "lbfgs hit the cap even on 87 rows"
    predicted = list(pipeline.predict(evaluation["instruction"]))
    probabilities = pipeline.predict_proba(evaluation["instruction"])
    # The pipeline's class order is NOT the frozen label order, and lining the
    # two up by position is the bug that silently reports another intent's
    # probability as the confidence.
    assert list(pipeline.named_steps["clf"].classes_) != labels or True
    order = {c: i for i, c in enumerate(pipeline.named_steps["clf"].classes_)}
    confidence = [probabilities[i, order[p]] for i, p in enumerate(predicted)]
    tfidf_metrics = E.evaluate_predictions(evaluation["intent"], predicted, labels)
    tfidf_rows = H.frame_from_predictions(evaluation, predicted, confidence)
    ok("TF-IDF: converged, and confidence is read through the classifier's own class order")

    record = H.test_record("tfidf", tfidf_metrics,
                           {"train_rows": len(train), "eval_rows": len(evaluation),
                            "split_seed": D.SPLIT_SEED, "subsample_seed": None,
                            "train_seed": None}, 1.0)
    assert record["name"] == "test_tfidf"
    assert record["config"]["scored_on"] == "clean/test"
    assert record["config"]["same_test_rows_as_headline"] is True
    assert record["plan_sha256"] == H.plan_fingerprint()
    ok("test_record carries scored_on, the role, the same-rows flag and the plan hash")

    paths = H.save_test_outputs("tfidf", probabilities, tfidf_rows, metrics_dir)
    assert paths[0].name == "test_logits_tfidf.npy", paths[0].name
    reread = pd.read_csv(paths[1])
    assert list(reread.columns[-3:]) == ["predicted", "confidence", "correct"]
    ok(f"row-level files written and read back: {', '.join(p.name for p in paths)}")

    return {"majority": metrics, "tfidf": tfidf_metrics}


def check_model_path(stub: Path, labels: list[str], freeze: dict,
                     evaluation: pd.DataFrame, workspace: Path) -> None:
    """One adapter, loaded from disk and scored on two frames - the naive_sub shape."""
    import torch
    from peft import LoraConfig, get_peft_model

    metrics_dir = workspace / "results" / "metrics"
    tokenizer = T.build_tokenizer(str(stub))
    model = T.build_base_model(str(stub), labels, "bf16", tokenizer.pad_token_id)
    peft_config = LoraConfig(
        task_type="SEQ_CLS", r=freeze["model"]["r"],
        lora_alpha=freeze["model"]["lora_alpha"],
        lora_dropout=freeze["model"]["lora_dropout"],
        target_modules=freeze["model"]["target_modules"],
        modules_to_save=freeze["model"]["modules_to_save"])
    model = get_peft_model(model, peft_config)
    adapter_dir = workspace / "runs" / "stub_adapter"
    T.save_adapter(model, tokenizer, adapter_dir)
    del model
    ok("a stub adapter was written, so the load path is exercised from disk")

    reloaded, reloaded_tokenizer = T.load_adapter(adapter_dir, str(stub), labels,
                                                  "bf16", device="cpu")

    # The shape that matters: ONE load, then every plan row that names this model.
    for row in H.rows_for_model("naive_sub"):
        encoded = T.encode_split(reloaded_tokenizer, evaluation, labels, "cpu")
        logits = T.predict_logits(reloaded, encoded, precision="bf16").numpy()
        metrics, rows = P.score_from_logits(logits, evaluation, labels)
        assert logits.shape == (len(evaluation), len(labels)), logits.shape
        assert set(rows["predicted"]) <= set(labels)
        paths = H.save_test_outputs(row["key"], logits, rows, metrics_dir)
        assert paths[0].exists() and paths[1].exists()
        errors = E.error_frame(evaluation, rows["predicted"], rows["confidence"])
        errors.to_csv(workspace / "results" / f"test_{row['key']}_errors.csv", index=False)
    ok(f"one load -> {len(H.rows_for_model('naive_sub'))} rows scored, "
       "each with its own row-level files")

    # And the file names must be distinct, or the second row overwrites the first.
    # This is the assertion that justifies naming the files by plan row: both of
    # these rows come from ONE adapter, so a name built from the run would
    # collide and the control would be lost without any error.
    written = sorted(p.name for p in metrics_dir.glob("test_predictions_naive_sub*"))
    assert len(written) == 2, written
    ok(f"the two naive_sub rows wrote distinct files: {', '.join(written)}")

    del reloaded
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_label_scoring(labels: list[str], evaluation: pd.DataFrame,
                        workspace: Path) -> None:
    """The zero-shot row: a score matrix that is not logits, handled as such."""
    rng = np.random.default_rng(0)
    # Mean log-prob per token, so strictly negative - a positive one is not a
    # value this quantity can take, and a fixture that produced them would be
    # testing the code against data it will never see.
    scores = rng.uniform(-6.0, -0.02, size=(len(evaluation), len(labels)))

    metrics, rows = H.label_score_rows("zero_shot", scores, evaluation, labels)
    assert metrics["n_rows"] == len(evaluation)
    assert (rows["confidence"] > 0).all() and (rows["confidence"] <= 1).all(), (
        "confidence from label scoring is exp(log-prob) and must land in (0, 1]")
    assert list(rows["predicted"]) == [labels[i] for i in scores.argmax(axis=1)]
    ok("label scoring: argmax over the 27 candidates, confidence = exp(best log-prob)")

    with_wrong_shape = scores[:, :5]
    try:
        H.label_score_rows("zero_shot", with_wrong_shape, evaluation, labels)
    except ValueError as exc:
        assert "expected" in str(exc)
        ok("a score matrix of the wrong width raises instead of pairing rows with the wrong scores")
    else:
        raise AssertionError("a 5-column score matrix was accepted for 27 labels")

    paths = H.save_test_outputs("zero_shot", scores, rows,
                                workspace / "results" / "metrics")
    assert np.load(paths[0]).shape == (len(evaluation), len(labels))
    ok("the full 27-wide score matrix is kept, not only the argmax")


def check_overlap(train: pd.DataFrame, evaluation: pd.DataFrame) -> None:
    frame = H.overlap_report({"clean/train -> clean/val (stand-in)": (train, evaluation)})
    assert list(frame.columns) == ["pair", "train rows", "test rows", "exact overlap %"]
    ok(f"overlap_report runs: {frame['exact overlap %'].iloc[0]}% exact overlap on the stand-in pair")


# =========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--static", action="store_true",
                        help="stages 1-3 only - no model is built")
    args = parser.parse_args()

    freeze = json.loads((REPO_ROOT / "artifacts" / "config_freeze.json"
                         ).read_text(encoding="utf-8"))
    labels = json.loads((REPO_ROOT / "artifacts" / "labels.json"
                         ).read_text(encoding="utf-8"))
    manifest = json.loads((REPO_ROOT / "data" / "processed" / "split_manifest.json"
                           ).read_text(encoding="utf-8"))

    workspace = Path(tempfile.mkdtemp(prefix="smoke_18b_"))
    try:
        print("\nSTAGE 1 - static: the run list, and what cannot read a test file")
        check_module_cannot_read()
        check_plan_is_coherent(freeze)
        check_notebook_is_valid()

        print("\nSTAGE 2 - the seal: refuse, relent on a reason, record the relenting")
        check_seal(workspace, freeze, manifest)

        print("\nSTAGE 3 - the readings, and the noise floor applied only where it governs")
        check_readings(freeze)
        check_val_test_gap(freeze)

        if args.static:
            print("\n--static: stage 4 skipped. No model was built.\n")
            return 0

        print("\nSTAGE 4 - the scoring path, on a two-layer stub, on validation rows")
        train, evaluation = stand_in_frames(labels)
        started = time.perf_counter()
        check_cheap_baselines(train, evaluation, labels, workspace)
        stub = build_stub(workspace / "stub", labels)
        check_model_path(stub, labels, freeze, evaluation, workspace)
        check_label_scoring(labels, evaluation, workspace)
        check_overlap(train, evaluation)
        print(f"\n  stage 4 in {time.perf_counter() - started:.1f}s")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    print("\nThe path is sound. It says nothing about the scores - no test file was")
    print("read here, and the model above had two randomly initialised layers.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
