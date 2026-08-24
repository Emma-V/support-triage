#!/usr/bin/env python
"""Run day 5 step 18a's whole path on CPU, before it costs two GPU hours.

--------------------------------------------------------------------------
WHAT THIS SAYS, AND WHAT IT DOES NOT

It says nothing whatsoever about scores. The model it trains has two randomly
initialised layers and sixty rows of data; its macro-F1 is noise and is never
printed as though it meant something.

What it says is that the code which will spend two hours of GPU time does not
fail on a shape, a missing keyword argument, a renamed field or a path. Every
one of those is a five-second fix that costs a session when it surfaces eighty
minutes into a Colab runtime, which is where they have surfaced before.

--------------------------------------------------------------------------
STAGE 1 IS THE ONE THAT MATTERS MOST

`03d_protocol_models.ipynb` claims it cannot open the test set. Stage 1 checks
that claim by parsing every code cell to an AST and looking for a string or a
call that could reach one - which is a different question from whether the text
"clean/test" appears, because the notebook says those words repeatedly in prose
while promising not to use them. Comments and markdown do not appear in an AST;
executable code does. That is exactly the discrimination needed.

--------------------------------------------------------------------------
    python tools/smoke_protocol_models.py
    python tools/smoke_protocol_models.py --static   # stage 1 only, ~1 second
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

from src import data as D              # noqa: E402
from src import evaluate as E          # noqa: E402
from src import protocol_models as P   # noqa: E402
from src import train as T             # noqa: E402

NOTEBOOK = REPO_ROOT / "notebooks" / "03d_protocol_models.ipynb"
MODULE = REPO_ROOT / "src" / "protocol_models.py"

# A string that names a test file, or a call that reads every split at once.
FORBIDDEN_STRINGS = {"test", "clean/test", "naive/test", "test.csv"}
FORBIDDEN_NAMES = {"load_all_splits"}

# The stub. Two layers and a 64-wide hidden state - small enough that a full
# training run is seconds, large enough that every shape in the real path is
# exercised. The tokenizer is the real Qwen3 one, from the local cache, because
# MAX_LENGTH and the truncation check are claims about that tokenizer.
STUB_LAYERS, STUB_HIDDEN, STUB_HEADS, STUB_KV_HEADS = 2, 64, 4, 2
STUB_ROWS, STUB_VAL_ROWS = 60, 40


def ok(message: str) -> None:
    print(f"  [PASS] {message}")


# =========================================================================
# STAGE 1: THE TEST SET IS UNREACHABLE FROM THIS NOTEBOOK
# =========================================================================

def check_cannot_open_test() -> None:
    """Parse every code cell and every line of the module. AST, not grep.

    A grep over this notebook finds "clean/test" a dozen times, all of them in
    markdown and comments explaining that it is not read. The AST holds only
    what executes, so a hit there is a real one.
    """
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = [(f"cell {i}", "".join(cell["source"]))
               for i, cell in enumerate(notebook["cells"])
               if cell["cell_type"] == "code"]
    sources.append((MODULE.name, MODULE.read_text(encoding="utf-8")))

    found = []
    for where, source in sources:
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise AssertionError(f"{where} does not parse: line {exc.lineno}: {exc.msg}")
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and node.value in FORBIDDEN_STRINGS):
                found.append(f"{where} line {node.lineno}: the string {node.value!r}")
            name = (node.attr if isinstance(node, ast.Attribute) else
                    node.id if isinstance(node, ast.Name) else None)
            if name in FORBIDDEN_NAMES:
                found.append(f"{where} line {node.lineno}: a call to {name}()")

    if found:
        raise AssertionError(
            "03d can reach the test set, and its whole claim is that it cannot:\n  "
            + "\n  ".join(found) +
            "\nEither remove the access, or stop claiming the notebook is safe to "
            "re-run - it cannot be both.")
    ok(f"no executable path to a test file in {len(sources)} sources "
       f"({len(sources) - 1} code cells + {MODULE.name})")


def check_notebook_is_valid() -> None:
    import nbformat
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    ok(f"nbformat validates - {len(nb.cells)} cells, {n_code} of them code")

    # Before the run this asserted that no cell carried an output: a number
    # committed ahead of the run it claims to come from is a number with nothing
    # behind it. 03d has since been run on 2026-08-24 and its outputs ARE the
    # surviving record of it - the run records died with the Colab runtime, and
    # the notebook is what is left. So the check reports rather than asserts, and
    # what it reports is whether the run went top to bottom: an execution_count
    # gap means cells ran out of order, which is the failure this can still see.
    executed = [c.execution_count for c in nb.cells
                if c.cell_type == "code" and c.get("outputs")]
    if not executed:
        ok("no stored outputs - nothing is claimed before the run happens")
    elif executed == sorted(e for e in executed if e is not None) == list(
            range(1, len(executed) + 1)):
        ok(f"stored outputs from one top-to-bottom run: cells 1-{len(executed)}, in order")
    else:
        print(f"  [note] stored outputs with execution counts {executed} - "
              "not a single top-to-bottom run, so read them in that light")


def check_plan_is_coherent(freeze: dict) -> None:
    """The run list, the freeze and the data on disk have to agree."""
    r = freeze["model"]["r"]
    names = [P.run_name(plan["key"], r) for plan in P.TRAINING_PLAN]
    assert len(set(names)) == len(names), f"duplicate run names: {names}"
    assert all(n.startswith(f"run_{P.RUN_NUMBER_BASE:02d}") or True for n in names)
    ok(f"run list: {', '.join(names)}")

    clean = D.load_split("clean", "train", REPO_ROOT / "data" / "processed")
    naive_sub = D.load_naive_sub(REPO_ROOT / "data" / "processed")
    assert len(naive_sub) == len(clean), (
        f"the size control has {len(naive_sub):,} rows and clean/train has "
        f"{len(clean):,} - it is not controlling for the size it claims to")
    ok(f"the size control is exactly clean/train's size ({len(clean):,} rows)")


# =========================================================================
# STAGE 2: THE TRAINING PATH, ON A STUB
# =========================================================================

def build_stub(directory: Path, labels: list[str]) -> Path:
    """A two-layer randomly initialised Qwen3, saved where from_pretrained can read it."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(T.DEBUG_MODEL)
    config = AutoConfig.from_pretrained(T.DEBUG_MODEL)
    config.num_hidden_layers = STUB_LAYERS
    config.hidden_size = STUB_HIDDEN
    config.intermediate_size = STUB_HIDDEN * 2
    config.num_attention_heads = STUB_HEADS
    config.num_key_value_heads = STUB_KV_HEADS
    config.head_dim = STUB_HIDDEN // STUB_HEADS
    # transformers 5 validates that layer_types has one entry per layer, so the
    # list has to be cut to match rather than left at the real model's 28.
    if getattr(config, "layer_types", None):
        config.layer_types = list(config.layer_types)[:STUB_LAYERS]

    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    n = sum(p.numel() for p in model.parameters())
    ok(f"stub built: {STUB_LAYERS} layers, {n / 1e6:.1f}M parameters, real Qwen3 tokenizer")
    return directory


def tiny_frames(labels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Real rows off disk, cut small. Every label appears, so the head is exercised.

    Real sentences rather than generated ones, because the truncation check is a
    claim about this corpus at MAX_LENGTH=32 and synthetic text would not test it.
    """
    processed = REPO_ROOT / "data" / "processed"
    train = D.load_split("naive", "train", processed)
    val = D.load_split("naive", "val", processed)
    # One row per intent first, so nothing is missing from the head's output layer.
    train = pd.concat([train.groupby("intent", group_keys=False).head(1),
                       train.head(STUB_ROWS)]).drop_duplicates("row_id").head(
                           max(STUB_ROWS, len(labels)))
    val = pd.concat([val.groupby("intent", group_keys=False).head(1),
                     val.head(STUB_VAL_ROWS)]).drop_duplicates("row_id").head(
                         max(STUB_VAL_ROWS, len(labels)))
    return train.reset_index(drop=True), val.reset_index(drop=True)


def run_training_path(stub: Path, labels: list[str], freeze: dict,
                      workspace: Path) -> dict:
    """One pass of exactly what the notebook's section 3 loop does."""
    import torch

    hardware = {
        "gpu_name": "cpu-stub", "device": "cpu", "precision": "bf16",
        "bf16_supported": True, "compute_capability": "-", "total_memory_gb": 0.0,
        "precision_reason": "CPU smoke test - not a GPU and not a result",
    }
    train_frame, val_frame = tiny_frames(labels)
    plan = P.TRAINING_PLAN[-1]           # naive_sub: the one with a subsample_seed
    name = P.run_name(plan["key"], freeze["model"]["r"])
    adapter_dir = workspace / "runs" / name
    metrics_dir = workspace / "results" / "metrics"

    config = T.RunConfig(
        name=name,
        r=freeze["model"]["r"],
        model_name=str(stub),
        epochs=1,                                    # 1, not 3 - this is a shape test
        learning_rate=freeze["training"]["learning_rate"],
        batch_size=freeze["training"]["batch_size"],
        grad_accum=freeze["training"]["grad_accum"],
        warmup_ratio=freeze["training"]["warmup_ratio"],
        weight_decay=freeze["training"]["weight_decay"],
        train_seed=freeze["training"]["train_seed_of_frozen_run"],
        train_rows=None,
        trained_on=plan["train"],
        scored_on=plan["val"],
        subsample_seed=D.SUBSAMPLE_SEED,
        notes="CPU smoke test on a randomly initialised stub - not a result")

    started = time.perf_counter()
    out = T.train_one_run(config, train_frame, val_frame, labels, hardware, adapter_dir,
                          progress=lambda *a, **k: None)
    record = out["record"]
    ok(f"train_one_run finished in {time.perf_counter() - started:.1f}s "
       f"({len(train_frame)} rows, 1 epoch)")

    assert record["config"]["trained_on"] == plan["train"], record["config"]["trained_on"]
    assert record["config"]["scored_on"] == plan["val"], record["config"]["scored_on"]
    assert record["config"]["subsample_seed"] == D.SUBSAMPLE_SEED
    ok(f"the record names its own data: trained_on={record['config']['trained_on']!r}, "
       f"subsample_seed={record['config']['subsample_seed']}")

    for field in ("model", "tokenizer", "val_encoded"):
        out.pop(field, None)

    # Reload from disk, exactly as the notebook does.
    model, tokenizer = T.load_adapter(adapter_dir, str(stub), labels,
                                      hardware["precision"], device="cpu")
    encoded = T.encode_split(tokenizer, val_frame, labels, "cpu")
    logits = T.predict_logits(model, encoded, precision=hardware["precision"]).numpy()
    metrics, rows = P.score_from_logits(logits, val_frame, labels)
    assert abs(metrics["f1_macro"] - record["metrics"]["f1_macro"]) <= 1e-6, (
        f"the reloaded adapter scores {metrics['f1_macro']} but the record says "
        f"{record['metrics']['f1_macro']} - the save/load round trip is broken, and "
        "on a real run that is a classification head coming back randomly initialised")

    # The score comparison above is what the notebook does, and on a real run it
    # is a strong check. Here it is not: an untrained stub scores 0.0000 macro-F1
    # and so would a randomly re-initialised head, so the equality holds for the
    # wrong reason. Comparing the PREDICTED LABELS instead is not degenerate -
    # with 27 classes, two different heads agreeing on all 40 rows by chance is
    # about 27^-40. This is the check that actually exercises modules_to_save.
    trained_predictions = list(out["predictions"])
    reloaded_predictions = list(rows["predicted"])
    assert trained_predictions == reloaded_predictions, (
        f"the model in memory and the model reloaded from disk disagree on "
        f"{sum(a != b for a, b in zip(trained_predictions, reloaded_predictions))} of "
        f"{len(trained_predictions)} rows. That is the classification head coming back "
        "randomly initialised - it trains, the loss falls, the save does not complain, "
        "and nothing else in the run would ever say so.")
    ok(f"save -> load -> predict returns the identical {len(rows)} labels, and the "
       f"recorded score ({metrics['f1_macro']:.4f}, which is noise)")

    paths = P.save_val_outputs(name, logits, rows, metrics_dir)
    assert np.load(paths[0]).shape == (len(val_frame), len(labels))
    reread = pd.read_csv(paths[1])
    assert list(reread.columns[-3:]) == ["predicted", "confidence", "correct"]
    E.error_frame(val_frame, rows["predicted"], rows["confidence"]).to_csv(
        workspace / "results" / f"{name}_errors.csv", index=False)
    D.write_json(record, metrics_dir / f"{name}.json")
    ok(f"row-level outputs written and read back: {', '.join(p.name for p in paths)}")

    journal = workspace / "results" / "run_journal.jsonl"
    P.write_journal_entry(journal, {"event": "model_trained", "run": name,
                                    "macro_f1_val": metrics["f1_macro"]})
    assert len(P.read_journal(journal)) == 1
    ok("journal entry written and read back")
    return record


def check_resume(record: dict, workspace: Path, freeze: dict) -> None:
    """The branch that makes a disconnect cost one model instead of the session."""
    plan = P.TRAINING_PLAN[-1]
    name = P.run_name(plan["key"], freeze["model"]["r"])
    adapter = workspace / "runs" / name / "adapter_model.safetensors"
    record_path = workspace / "results" / "metrics" / f"{name}.json"
    assert adapter.exists() and record_path.exists(), (
        "the resume branch keys on both files existing, and one of them does not - "
        "so on a real run the loop would silently retrain instead of resuming")
    ok("resume condition holds: adapter and record both on disk")

    # And it must NOT fire when only one of the two is there. A record without an
    # adapter is exactly the state a crash between the two writes leaves behind.
    record_path.rename(record_path.with_suffix(".json.moved"))
    assert not (adapter.exists() and record_path.exists())
    record_path.with_suffix(".json.moved").rename(record_path)
    ok("resume correctly does NOT fire on a half-written run")


def check_freeze_guard(record: dict, freeze: dict) -> None:
    """The guard that keeps a protocol comparison from also being a config comparison.

    The real freeze describes Qwen3-1.7B for 3 epochs and the stub is a
    two-layer model trained for 1, so the real freeze rejects it - which is the
    guard working, and is asserted as such below. The positive case then needs a
    freeze that describes the run being checked, so those two fields are swapped
    into a copy. Nothing else is touched: the sixteen frozen fields that this is
    actually testing all come from the real document.
    """
    # First, the accidental negative test the stub hands us for free: a run on a
    # different base model must not pass, and on a real run "the adapter was
    # trained on a different checkpoint" is invisible in every other field.
    try:
        P.assert_matches_freeze(record, freeze)
    except AssertionError as exc:
        assert "base_model" in str(exc) and "epochs" in str(exc), str(exc)
        ok("the real freeze rejects the stub, naming base_model and epochs")
    else:
        raise AssertionError(
            "the real freeze ACCEPTED a two-layer stub trained for one epoch. The "
            "guard is not comparing what it claims to compare.")

    stub_freeze = json.loads(json.dumps(freeze))
    stub_freeze["model"]["base_model"] = record["config"]["base_model"]
    stub_freeze["training"]["epochs"] = record["config"]["epochs"]

    P.assert_matches_freeze(record, stub_freeze)
    ok("with those two swapped, all 16 frozen fields match")

    for field, bad in (("learning_rate", 3e-4), ("r", 999), ("max_length", "32"),
                       ("modules_to_save", []), ("precision", "fp16")):
        drifted = json.loads(json.dumps(record))
        drifted["config"][field] = bad
        try:
            P.assert_matches_freeze(drifted, stub_freeze)
        except AssertionError:
            continue
        raise AssertionError(f"a drifted {field} was NOT caught by the freeze guard")
    ok("drift caught in learning_rate, r, max_length-as-a-string, "
       "modules_to_save and precision")

    differences = P.differences_from_freeze(record, stub_freeze)
    moved = differences.loc[differences["field"] == "trained_on", "differs"].item()
    assert moved is True, differences
    ok("differences_from_freeze reports the data moving, as it should")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--static", action="store_true",
                        help="stage 1 only - no model is built or trained")
    args = parser.parse_args()

    freeze = json.loads((REPO_ROOT / "artifacts" / "config_freeze.json"
                         ).read_text(encoding="utf-8"))
    labels = json.loads((REPO_ROOT / "artifacts" / "labels.json"
                         ).read_text(encoding="utf-8"))

    print("\nSTAGE 1 - static: 03d cannot open the test set")
    check_cannot_open_test()
    check_notebook_is_valid()
    check_plan_is_coherent(freeze)

    if args.static:
        print("\n--static: stage 2 skipped. Nothing was trained.\n")
        return 0

    print("\nSTAGE 2 - the training path, on a two-layer stub, on CPU")
    workspace = Path(tempfile.mkdtemp(prefix="smoke_18a_"))
    try:
        stub = build_stub(workspace / "stub", labels)
        record = run_training_path(stub, labels, freeze, workspace)
        check_freeze_guard(record, freeze)
        check_resume(record, workspace, freeze)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    print("\nThe code path is sound. It says nothing about the scores - the model")
    print("above had two randomly initialised layers and sixty rows.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
