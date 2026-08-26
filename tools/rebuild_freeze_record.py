#!/usr/bin/env python
"""Rebuilds the configuration freeze on disk, from evidence already committed to git.

--------------------------------------------------------------------------
WHY THIS FILE EXISTS

An earlier session measured the between-seed noise floor, applied the
pre-registered decision rule, and wrote `artifacts/config_freeze.json`.
None of it reached the repository: Drive did not mount, so every output of
that session went to /content/_local_runs and was lost with the runtime,
and `.gitignore` matched `artifacts/*` with only three files on the
allowlist, so the freeze record would have been skipped even if the final
commit cell had run. No error, no output, no file.

The project rule for opening the test set is blunt: without a freeze
record the test set does not open. The freeze record is also the
authority the later training runs are checked against - a run that
quietly used a different learning rate would turn a comparison between
protocols into a comparison between configurations, and nothing in the
resulting table would look wrong.

--------------------------------------------------------------------------
WHAT IS AND IS NOT RECOVERED

The evidence is `notebooks/03b_noise_floor.ipynb` with its stored outputs,
committed at adaebe9 on 2026-08-22 - before any test number existed
anywhere in this project. That timestamp is the whole reason this is a
recovery rather than a re-decision.

Nothing here is retyped. The numbers are parsed out of the saved outputs,
and the record is rebuilt by calling the original session's own
`src.noise_floor.freeze_record()` on them - the same function, so the same
document. The rebuild is then required to match, character for character,
the 2,000-character prefix that session printed. If a constant in
src/train.py has drifted since, that comparison fails and this script
refuses to write anything.

Deliberately not invented: the three seed adapters,
`val_logits_r8_seed*.npy`, and the full run_06/07/08 records. Only their
summary rows were ever printed, so only their summary rows come back.

--------------------------------------------------------------------------
    python tools/rebuild_freeze_record.py          # verify, then write
    python tools/rebuild_freeze_record.py --check  # verify only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import data as D          # noqa: E402
from src import noise_floor as NF  # noqa: E402
from src import train as T         # noqa: E402

NOTEBOOK = REPO_ROOT / "notebooks" / "03b_noise_floor.ipynb"
SWEEP_CSV = REPO_ROOT / "results" / "metrics" / "lora_r_sweep.csv"
MANIFEST = REPO_ROOT / "data" / "processed" / "split_manifest.json"

FREEZE_OUT = REPO_ROOT / "artifacts" / "config_freeze.json"
NOISE_OUT = REPO_ROOT / "results" / "metrics" / "noise_floor.json"
SEEDS_OUT = REPO_ROOT / "results" / "metrics" / "seed_noise_floor.csv"

# The commit that carries the outputs being read. Named so that "which run is
# this describing" has an answer that does not depend on this file.
EVIDENCE_COMMIT = "adaebe9"


# =========================================================================
# 1. READING THE NOTEBOOK
# =========================================================================
# Cells are found by what their output contains, never by index. A cell
# inserted into 03b later must not silently shift which output is parsed
# as the seed table.

def cell_outputs(notebook: dict) -> list[str]:
    """Every output of every cell, flattened to text, in notebook order."""
    blocks = []
    for cell in notebook["cells"]:
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                blocks.append("".join(out.get("text", [])))
            elif out.get("output_type") in ("display_data", "execute_result"):
                plain = out.get("data", {}).get("text/plain")
                if plain:
                    blocks.append("".join(plain))
    return blocks


def find_block(blocks: list[str], *must_contain: str) -> str:
    """The one output block containing all of these strings."""
    hits = [b for b in blocks if all(needle in b for needle in must_contain)]
    if len(hits) != 1:
        raise LookupError(
            f"expected exactly one output block containing {must_contain!r}, "
            f"found {len(hits)}. 03b's outputs are not the ones this script was "
            "written against - do not guess, read the notebook."
        )
    return hits[0]


def parse_seed_table(blocks: list[str]) -> pd.DataFrame:
    """The three noise-floor runs, from the frame the original session displayed.

    pandas wraps a six-column frame into two printed blocks, so the numbers and
    the run names are parsed separately and then zipped back together. The row
    count is asserted rather than assumed: a wrapped frame that lost a line
    would otherwise produce a two-seed 'noise floor'.
    """
    numbers = find_block(blocks, "seed", "macro-F1 (val)", "epoch chosen")
    rows = re.findall(
        r"^\s*\d+\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s*$",
        numbers, flags=re.MULTILINE)
    names = re.findall(r"^\s*\d+\s+(run_\d+_\S+)\s*$", numbers, flags=re.MULTILINE)

    if len(rows) != len(NF.SEEDS) or len(names) != len(NF.SEEDS):
        raise ValueError(
            f"parsed {len(rows)} score rows and {len(names)} run names from the "
            f"seed table, expected {len(NF.SEEDS)} of each. A standard deviation "
            "over the wrong number of runs is not the noise floor that was "
            "originally measured."
        )

    table = pd.DataFrame({
        "seed": [int(r[0]) for r in rows],
        "macro-F1 (val)": [float(r[1]) for r in rows],
        "accuracy (val)": [float(r[2]) for r in rows],
        "epoch chosen": [int(r[3]) for r in rows],
        "runtime (s)": [float(r[4]) for r in rows],
        "run": names,
    })
    if tuple(table["seed"]) != tuple(NF.SEEDS):
        raise ValueError(
            f"the notebook's seeds are {tuple(table['seed'])} but "
            f"src.noise_floor.SEEDS is {NF.SEEDS}. One of the two moved after "
            "the run, and the freeze must describe the run.")
    return table


def parse_granularity(blocks: list[str]) -> dict:
    """The metric-granularity probe: what one validation row is worth.

    Printed to six decimals, which is not enough precision to reproduce the
    freeze byte for byte - the full value is recovered from the freeze's own
    printed prefix below. This block is kept as the independent cross-check on
    that value, at the precision it was printed.
    """
    block = find_block(blocks, "base_f1_macro", "mean_drop_per_row", "max_drop")
    found = {}
    for key in ("base_f1_macro", "n_probed", "mean_drop_per_row",
                "min_drop", "max_drop"):
        match = re.search(rf"^\s*{key}\s+([\d.]+)\s*$", block, flags=re.MULTILINE)
        if match is None:
            raise LookupError(f"{key} is not in the granularity output")
        found[key] = float(match.group(1))
    found["n_probed"] = int(found["n_probed"])
    return found


def parse_hardware(blocks: list[str]) -> dict:
    """GPU name and precision, from the hardware report the original session printed."""
    block = find_block(blocks, "gpu_name", "compute_capability", "bf16_supported")
    gpu = re.search(r"^\s*gpu_name\s+(.+?)\s*$", block, flags=re.MULTILINE)
    precision = re.search(r"^\s*precision\s+(\S+)\s*$", block, flags=re.MULTILINE)
    if gpu is None or precision is None:
        raise LookupError("gpu_name / precision are not in the hardware report")
    return {"gpu_name": gpu.group(1), "precision": precision.group(1)}


def parse_measured_at_r(blocks: list[str]) -> int:
    """The r the noise floor was measured on - 8, the middle of the sweep.

    It differs from the chosen r on purpose (the estimate is deliberately not
    taken around the highest scorer), and that difference is the reason the
    freeze had to be built from constants instead of from a run record.
    """
    block = find_block(blocks, "Building the freeze from the constants")
    match = re.search(r"was measured on \(r=(\d+)\)", block)
    if match is None:
        raise LookupError("the 'measured on (r=N)' line is not in that output")
    return int(match.group(1))


def _printed_value(printed: str, key: str) -> str:
    """One value out of the truncated freeze prefix, by key name.

    The prefix stops mid-document, so it is not loadable JSON. Both keys read
    this way appear well inside the first 2,000 characters and both are
    cross-checked against an independently parsed figure before use.
    """
    match = re.search(rf'"{key}":\s*([^,\n]+)', printed)
    if match is None:
        raise LookupError(
            f'"{key}" is not in the 2,000 characters 03b printed. It is the only '
            "place that value survives at full precision.")
    return match.group(1).strip()


def parse_printed_freeze(blocks: list[str]) -> str:
    """The 2,000-character prefix of the freeze record, as originally printed.

    This is the target. It is not a source for anything except the two values
    that were printed at full precision only here - `frozen_at` and
    `one_val_row_is_worth` - both of which are cross-checked against
    independently parsed numbers before being used.
    """
    block = find_block(blocks, '"frozen_at"', '"frozen_from_run"', '"noise_floor"')
    prefix = block[block.index("{"):][:2000]
    if len(prefix) != 2000:
        raise ValueError(
            f"the printed freeze is {len(prefix)} characters, expected 2000. "
            "03b printed json.dumps(FREEZE, indent=2)[:2000]; a different length "
            "means a different notebook.")
    return prefix


# =========================================================================
# 2. REBUILDING
# =========================================================================

def synthesise_frozen_run(chosen_r: int, best_epoch: int, hardware: dict,
                          manifest: dict) -> dict:
    """Rebuilds the run record originally handed to freeze_record(), from constants.

    This step was performed originally as well, and its output says so:
    "the rule chose r=16, which is not the r the noise floor was measured
    on (r=8). Building the freeze from the constants." It deep-copied the
    seed-42 record, swapped in the chosen r, its alpha and its best epoch,
    and renamed it. There is no surviving record to copy here, so the same
    config is assembled directly from the module-level constants those
    runs were themselves built from.

    Every field is a constant, a manifest entry, or a parsed measurement.
    None is a literal typed into this file.
    """
    clean = manifest["splits"]["clean"]
    return {
        "name": f"run_0{3 + list(T.R_VALUES).index(chosen_r)}_lora_r{chosen_r}",
        "config": {
            "base_model": T.MAIN_MODEL,
            "task_type": "SEQ_CLS",
            "r": chosen_r,
            "lora_alpha": T.LORA_ALPHA_MULTIPLIER * chosen_r,
            "lora_dropout": T.LORA_DROPOUT,
            "target_modules": list(T.TARGET_MODULES),
            "modules_to_save": list(T.MODULES_TO_SAVE),
            "learning_rate": T.LEARNING_RATE,
            "epochs": T.EPOCHS,
            "best_epoch": best_epoch,
            "batch_size": T.BATCH_SIZE,
            "grad_accum": T.GRAD_ACCUM,
            "warmup_ratio": T.WARMUP_RATIO,
            "weight_decay": T.WEIGHT_DECAY,
            "precision": hardware["precision"],
            "max_length": D.MAX_LENGTH,
            "selection_metric": T.SELECTION_METRIC,
            "train_seed": T.TRAIN_SEED,
            "trained_on": "clean/train",
            "scored_on": "clean/val",
            "train_rows": clean["train"]["n_rows"],
            "eval_rows": clean["val"]["n_rows"],
            "train_sha256": clean["train"]["sha256"],
            "split_seed": D.SPLIT_SEED,
            "gpu_name": hardware["gpu_name"],
        },
    }


def cross_check_constants(blocks: list[str], record: dict, measured_at_r: int) -> None:
    """The field-by-field table originally printed, checked against the current constants.

    03b printed `same_configuration`, which lists the value of every frozen
    field as the three seed runs actually had it. If a constant in src/train.py
    has been edited since - and nothing would otherwise announce that - the
    rebuilt record disagrees with that table and the recovery is not a
    recovery.

    Two fields are expected to differ and are skipped: `r` and `lora_alpha`
    were measured at r=8 and the freeze describes the chosen r=16.
    """
    block = find_block(blocks, "field", "identical", "selection_metric")

    # pandas wraps the three-column frame into two printed blocks - `field` and
    # `identical` in the first, `value` in the second - each keyed by the row
    # index. They are parsed separately and rejoined on that index. Read as one
    # block, the regex pairs a field with the NEXT row's value and every
    # comparison below becomes nonsense that still looks like a real failure.
    fields = dict(re.findall(r"^\s*(\d+)\s+(\w+)\s+(?:True|False)\s*$",
                             block, flags=re.MULTILINE))
    values = dict(re.findall(r"^\s*(\d+)\s+(\S.*?)\s*$", block, flags=re.MULTILINE))
    printed = {name: values[i] for i, name in fields.items()
               if i in values and values[i] not in ("True", "False")}
    if len(printed) < 15:
        raise LookupError(
            f"only {len(printed)} of the {len(fields)} fields in the "
            "originally printed same_configuration table were paired with a "
            "value. A cross-check that passes by being empty is worse than "
            "one that fails.")

    config = record["config"]
    mismatches = []
    for field, shown in printed.items():
        if field in ("r", "lora_alpha"):
            continue                       # measured at r=8, frozen at r=16
        if field not in config:
            continue
        if shown.endswith("..."):
            # pandas truncated a long value (train_sha256) - compare the prefix
            if not repr(config[field]).startswith(shown[:-3]):
                mismatches.append(f"{field}: originally {shown}, now {config[field]!r}")
        elif repr(config[field]) != shown:
            mismatches.append(f"{field}: originally {shown}, now {config[field]!r}")

    # The three runs in that table ARE the noise-floor runs, so the r they show
    # must be the r the floor was measured at. If those two disagree, one of
    # the two parsers has locked onto the wrong output block.
    if printed.get("r") != repr(measured_at_r):
        mismatches.append(
            f"the same_configuration table shows r={printed.get('r')} but the "
            f"freeze cell said the floor was measured at r={measured_at_r}")

    if mismatches:
        raise AssertionError(
            "the constants in src/train.py no longer agree with the run "
            "originally described:\n  " + "\n  ".join(mismatches) +
            "\nA freeze rebuilt from drifted constants describes a configuration "
            "that never ran. Fix the constant, or re-run 03b for real.")


def rebuild(notebook: dict) -> tuple[dict, dict, pd.DataFrame, str]:
    """Everything, in the order 03b did it. Returns (freeze, noise, seeds, target)."""
    blocks = cell_outputs(notebook)

    seeds_table = parse_seed_table(blocks)
    granularity = parse_granularity(blocks)
    hardware = parse_hardware(blocks)
    measured_at_r = parse_measured_at_r(blocks)
    printed = parse_printed_freeze(blocks)

    # The two values that were only ever printed at full precision inside the
    # freeze itself. The prefix is truncated mid-document and cannot be parsed
    # as JSON, so both are pulled out by name rather than by loading it.
    per_row = float(_printed_value(printed, "one_val_row_is_worth"))
    frozen_at = _printed_value(printed, "frozen_at").strip('"')
    if round(per_row, 6) != granularity["mean_drop_per_row"]:
        raise AssertionError(
            f"the freeze says one validation row is worth {per_row}, but the "
            f"granularity probe printed {granularity['mean_drop_per_row']}. Those "
            "are two different measurements of one quantity.")

    noise = NF.noise_floor(seeds_table["macro-F1 (val)"])
    noise_acc = NF.noise_floor(seeds_table["accuracy (val)"])
    floor = NF.effective_noise_floor(noise["std"], per_row)

    sweep = pd.read_csv(SWEEP_CSV).rename(columns={"f1_macro_val": "macro-F1 (val)"})
    chosen = NF.decision_rule(sweep, floor["floor"])

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    best_epoch = int(sweep.loc[sweep["r"] == chosen["chosen_r"], "best_epoch"].iloc[0])
    record = synthesise_frozen_run(chosen["chosen_r"], best_epoch, hardware, manifest)
    cross_check_constants(blocks, record, measured_at_r)

    freeze = NF.freeze_record(chosen, record, manifest, noise, seeds=NF.SEEDS)
    freeze["noise_floor"]["measured_at_r"] = measured_at_r
    freeze["noise_floor"]["one_val_row_is_worth"] = per_row
    freeze["noise_floor"]["effective_floor"] = floor
    freeze["smoke"] = False
    # freeze_record() stamps datetime.now(). A freeze re-stamped at rebuild
    # time would be a freeze written AFTER the models it governs, which is
    # the one property this document exists to preserve.
    freeze["frozen_at"] = frozen_at

    noise_json = {
        "macro_f1": noise,
        "accuracy": noise_acc,
        "granularity": granularity,
        "decision": chosen,
        "effective_floor": floor,
        "measured_at_r": measured_at_r,
        "seeds": list(NF.SEEDS),
        "provenance": {
            "recovered_by": "tools/rebuild_freeze_record.py",
            "from": "notebooks/03b_noise_floor.ipynb stored outputs",
            "committed_at": EVIDENCE_COMMIT,
            "verified": "the rebuilt freeze matches the 2,000-character prefix "
                        "03b printed, character for character",
            "precision_note": (
                "granularity.mean_drop_per_row is the six decimals 03b printed; "
                "the full-precision value it was computed with survives as "
                "config_freeze.json's one_val_row_is_worth"),
            "not_recovered": ["val_logits_r8_seed*.npy",
                              "val_predictions_r8_seed*.csv",
                              "the full run_06/07/08 records"],
        },
    }
    return freeze, noise_json, seeds_table, printed


# =========================================================================
# 3. THE COMPARISON THAT MAKES THIS A RECOVERY
# =========================================================================

def verify(freeze: dict, printed: str) -> None:
    rebuilt = json.dumps(freeze, indent=2)[:2000]
    if rebuilt == printed:
        return
    for i, (a, b) in enumerate(zip(rebuilt, printed)):
        if a != b:
            break
    else:
        i = min(len(rebuilt), len(printed))
    raise AssertionError(
        "the rebuild does not match what was originally printed. First "
        f"difference at character {i}:\n"
        f"  rebuilt : ...{rebuilt[max(0, i - 60):i + 60]!r}\n"
        f"  original: ...{printed[max(0, i - 60):i + 60]!r}\n"
        "Do not write this file. Something that fed the freeze has moved.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="verify the rebuild without writing anything")
    args = parser.parse_args()

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    freeze, noise_json, seeds_table, printed = rebuild(notebook)
    verify(freeze, printed)

    print(f"[PASS] the rebuild matches the 2,000 characters 03b printed, "
          f"character for character")
    print(f"       evidence: notebooks/03b_noise_floor.ipynb @ {EVIDENCE_COMMIT}")
    print()
    print(f"  frozen at        {freeze['frozen_at']}")
    print(f"  frozen from      {freeze['frozen_from_run']}")
    print(f"  chosen r         {freeze['model']['r']}  "
          f"(alpha {freeze['model']['lora_alpha']})")
    print(f"  noise floor      {freeze['noise_floor']['effective_floor']['floor']:.6f}"
          f"  binding: {freeze['noise_floor']['effective_floor']['binding']}")
    print(f"  between-seed std {freeze['noise_floor']['std']:.6f}"
          f"  over seeds {freeze['noise_floor']['seeds']}, measured at "
          f"r={freeze['noise_floor']['measured_at_r']}")
    print(f"  {noise_json['decision']['margin_note']}")

    if args.check:
        print("\n--check: nothing written.")
        return 0

    D.write_json(freeze, FREEZE_OUT)
    D.write_json(noise_json, NOISE_OUT)
    SEEDS_OUT.parent.mkdir(parents=True, exist_ok=True)
    seeds_table.to_csv(SEEDS_OUT, index=False)

    print()
    for path in (FREEZE_OUT, NOISE_OUT, SEEDS_OUT):
        print(f"  written  {path.relative_to(REPO_ROOT).as_posix()}")
    print("\nNow check `git status` actually lists artifacts/config_freeze.json.")
    print("The .gitignore rule that matches artifacts/* previously caused this "
          "exact file to be skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
