"""
Training. This is the file that teaches the model to sort a customer message
into one of the 27 intents.

What goes in here:
- A function that builds the model: take Qwen3, freeze it, and add two things.
  LoRA, which is a small set of extra weights that do get trained instead of
  training the whole model, and a classification head, which is the final
  layer that returns a probability for each of the 27 options.
- One function that takes a run configuration (which LoRA r, how many epochs,
  which learning rate) and returns the results of that run.
- The three blocking checks that have to pass before a single training step
  runs, because each of them fails SILENTLY, or with an error that points at
  the wrong place.
- The "before" number: scoring the 27 labels with an untrained Qwen3, zero-shot
  and few-shot. That is a baseline rather than training, but it needs a GPU and
  src/baselines.py is deliberately torch-free, so it lives here.

Why a function that takes a configuration instead of code written top to
bottom: you are going to run this many times with different values in order to
compare them. If the settings are a parameter, every run is one row in a
table. If they are written inside the code, every run means editing the code,
and then the runs are not comparable.

This file runs on GPU, and it is the only one that does.

--------------------------------------------------------------------------
HOUSE RULES, the same ones as src/data.py and src/baselines.py

No prints except through an explicit progress callback, no plots, and no
writing to results/ - notebooks/03_train.ipynb runs these functions and decides
what to draw and save. The one exception is save_adapter(): a trained adapter
IS a file, so the function that produces it is the one that writes it.

Scoring goes through src/evaluate.py, not through a re-implementation that
looks the same. The fine-tuned model has to be measured by the identical
function that measured the TF-IDF baseline, otherwise the comparison in the
report is between two rulers rather than between two models.

READING ORDER (the file is written to be read top to bottom)
  1. Constants        - every hyper-parameter that must not drift, in one place.
  2. Blocking checks  - the four things that fail without an honest error.
  3. Building         - tokenizer, model, LoRA.
  4. Encoding         - text to tensors, once, on the GPU.
  5. Training         - the loop. One config in, one record out.
  6. Save and reload  - the round-trip that catches the two silent failures.
  7. Label scoring    - the "before" number, measured without a trained head.
  8. Environment      - what has to be recorded now or is lost forever.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as E
from .data import MAX_LENGTH, SPLIT_SEED

# torch, transformers and peft are imported inside the functions that need
# them. Importing src.train on a CPU machine - to read it, to lint it, or
# because something else imports the package - must not drag in a CUDA stack
# or fail where torch is not installed.


# =========================================================================
# 1. CONSTANTS
# =========================================================================
# Decision E4: today's sweep is over `r` ALONE. That is only true if every
# other number is fixed before the first run and untouched for the rest of the
# day. They are fixed here, in the module, rather than in a notebook cell, for
# the same reason NEAR_DUP_THRESHOLD lives in src/data.py: a value that lives
# in a cell gets nudged mid-day, and two runs stop being comparable with no
# error message anywhere.

# Decision B2: three sizes, one code path. 0.6B is the debug run and its
# numbers are never reported; 1.7B is the reported result; 4B in 4-bit is the
# scaling row, later. The instruct variant, never `-Base` - the "before"
# number has to be measured on the SAME weights the adapter will sit on, or
# the before/after difference mixes two changes into one.
DEBUG_MODEL = "Qwen/Qwen3-0.6B"
MAIN_MODEL = "Qwen/Qwen3-1.7B"

# Decision B0 fallback condition. transformers below this does not know the
# Qwen3 architecture, and the error it raises names the model - which reads
# exactly like a typo in the model id and costs half an hour spent in the
# wrong place.
MIN_TRANSFORMERS = (4, 51)

# Decision B4. `alpha = 2r` is the rule that keeps the sweep one-dimensional.
# With alpha fixed instead, changing r would change both how much capacity the
# adapter has AND how strongly its output is scaled, and a difference between
# two rows could not be attributed to either.
LORA_ALPHA_MULTIPLIER = 2
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]

# The classification head. If it is not in this list the adapter trains fine,
# scores fine and saves without complaint - and comes back RANDOMLY
# INITIALISED on the next load, because it is not part of the base model and
# nothing else saves it. That failure prints nothing. The round-trip check in
# section 6 is what catches it.
MODULES_TO_SAVE = ["score"]

R_VALUES = (4, 8, 16)

# The frozen group (decision E4). Chosen once, before the first run.
LEARNING_RATE = 2e-4      # LoRA convention: about 10x a full fine-tune, because
                          # only ~0.1% of the parameters move on each step
EPOCHS = 3
BATCH_SIZE = 32           # the sequences are 32 tokens long; the usual reason
                          # to keep batches small does not apply here
GRAD_ACCUM = 1            # the FIRST knob to turn if memory runs out - not
                          # max_length, which is measured and must not move
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# The checkpoint selection criterion, fixed in advance: macro-F1 on clean/val
# at the end of every epoch, keep the best. Named here so that "which epoch
# was reported" is a recorded property of the run rather than a decision made
# while looking at the numbers.
SELECTION_METRIC = "f1_macro"

# Seeds. All named separately, and all different from each other on purpose.
# SPLIT_SEED (42, in src/data.py) decides which rows are in which split and is
# frozen forever. TRAIN_SEED decides weight init and shuffling, and it is the
# one that VARIES across tomorrow's three noise-floor runs. FEWSHOT_SEED
# decides which 27 sentences are shown inside the few-shot prompt. Three
# different things called "seed" is the classic way to report variance that is
# really different DATA, so they never share a name - and here they do not
# even share a value, so a mix-up is visible in the record rather than silent.
TRAIN_SEED = 42
FEWSHOT_SEED = 13

# The debug run (step 13). Small model, small slice, one epoch. Nothing it
# produces is reported: it exists to prove the plumbing before an hour of GPU
# time is spent discovering that the plumbing leaks.
DEBUG_ROWS = 512          # 16 optimiser steps at BATCH_SIZE - few enough to be
DEBUG_EPOCHS = 1          # cheap, enough for "did the loss move" to be a real answer

# Where adapters go. Colab's local disk is wiped when the runtime dies, and a
# run whose output vanished is a run that has to be paid for twice.
DEFAULT_DRIVE_DIR = Path("/content/drive/MyDrive/support-triage/runs")

# Decision E3: one demonstration per intent, drawn at random from clean/train
# with a recorded seed. Hand-picking the "nice" examples is exactly where
# information that would not exist at prediction time gets in.
FEWSHOT_PER_INTENT = 1

# The instruction text. It is a hyper-parameter even though it does not look
# like one: reword it and the "before" number moves. It lives in the module so
# that it is under version control and cannot drift between two runs, and its
# sha256 goes into every run record so that a change is visible in the table
# instead of being an unexplained shift in a score.
PROMPT_SYSTEM = (
    "You are a customer-support ticket router. You read one customer message "
    "and answer with exactly one intent label from the list you are given. "
    "Answer with the label and nothing else."
)
PROMPT_USER = "Intent labels:\n{label_list}\n\nCustomer message: {text}\n\nIntent label:"
# The few-shot demonstrations already contain every one of the 27 labels, so
# the list is not repeated in every user turn - it is stated once, in the
# system turn, and the per-turn template gets shorter.
PROMPT_USER_FEWSHOT = "Customer message: {text}\n\nIntent label:"

# Scoring rule for the "before" number, DECLARED BEFORE THE RUN. The candidate
# labels are between 1 and about 7 tokens long ("review" against
# "set_up_shipping_address"), and a plain sum of token log-probabilities is
# systematically biased towards the short ones - that is a property of the
# scoring rule, not of the model. The headline number therefore uses the mean
# log-probability per token. The unnormalised sum comes out of the same forward
# pass, costs nothing, and is recorded alongside it - but which one is the
# headline was fixed here, before either number existed.
LABEL_SCORING = "mean_logprob_per_token"


# =========================================================================
# 2. BLOCKING CHECKS
# =========================================================================
# Four checks, all of them run before a single training step. What they have
# in common is the reason they are worth a section of their own: each one, when
# it fails, either prints nothing at all or prints something that points at the
# wrong file.

def check_transformers_version() -> dict:
    """Blocking check A: is transformers new enough to know what Qwen3 is?

    Below 4.51 the architecture is unknown and the loader raises a KeyError on
    the architecture name. That error reads like a wrong model id - the model
    id is in the message and the version is not - so the natural response is
    to go and check the spelling of "Qwen/Qwen3-1.7B", which is fine, for half
    an hour.

    Returns the version rather than only asserting it, because the notebook has
    to write it into the work log either way.
    """
    import transformers

    version = transformers.__version__
    parts = []
    for chunk in version.split(".")[:2]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    major_minor = tuple(parts)

    ok = major_minor >= MIN_TRANSFORMERS
    if not ok:
        raise RuntimeError(
            f"transformers {version} does not know the Qwen3 architecture "
            f"(need >= {MIN_TRANSFORMERS[0]}.{MIN_TRANSFORMERS[1]}). The error you would "
            "get instead names the MODEL, not the version, which is why this check "
            "exists. Either upgrade, or fall back to Qwen2.5 at the same sizes "
            "(decision B0) - and record in the work log which one happened."
        )
    return {"transformers": version, "min_required": f"{MIN_TRANSFORMERS[0]}.{MIN_TRANSFORMERS[1]}"}


def gpu_report() -> dict:
    """Blocking check B: which GPU did Colab hand out, and does it support bf16?

    T4 (Turing) has no bf16 support: asking for it either raises or falls back
    to something very slow. L4 and A100 do support it, and there bf16 is the
    better choice because it has the exponent range of fp32 and therefore does
    not need loss scaling.

    The point of returning this rather than just picking a dtype is the work
    log. Two runs on different hardware are not comparable, and the runtimes
    recorded today are meaningless without the name of the machine that
    produced them. `precision` here is a decision made from a measurement, not
    a default that was left alone.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU visible. In Colab: Runtime > Change runtime type > T4 GPU. "
            "Nothing in this notebook should be run on CPU - not because it would "
            "fail, but because it would take hours and look like it was working."
        )

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    try:
        bf16 = bool(torch.cuda.is_bf16_supported())
    except Exception:            # very old torch has no such function
        bf16 = capability[0] >= 8

    return {
        "gpu_name": name,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        "bf16_supported": bf16,
        "device": "cuda",
        "precision": "bf16" if bf16 else "fp16",
        "precision_reason": (
            "bf16 supported by this GPU - preferred, no loss scaling needed"
            if bf16 else
            f"{name} does not support bf16 (compute capability "
            f"{capability[0]}.{capability[1]} < 8.0), so fp16 with a gradient scaler"
        ),
    }


def assert_no_truncation(tokenizer, texts) -> int:
    """Blocking check C: does MAX_LENGTH = 32 actually fit every row?

    MAX_LENGTH was measured on day 1 with this same tokenizer (p99 = 19 tokens,
    max = 24) and 32 was chosen to truncate nothing. Truncation is the failure
    that produces no error and no warning: the row still classifies, it is just
    missing its ending, and the ending is where "cancel my order" and "change
    my order" differ.

    Returns the longest sequence seen, which is the number to put in the log.
    """
    lengths = [len(ids) for ids in
               tokenizer(list(texts), add_special_tokens=True)["input_ids"]]
    longest = max(lengths)
    if longest > MAX_LENGTH:
        raise AssertionError(
            f"{sum(l > MAX_LENGTH for l in lengths)} rows are longer than "
            f"MAX_LENGTH={MAX_LENGTH} (longest is {longest}). Day 1 measured that this "
            "cannot happen for this corpus with this tokenizer, so either the data or "
            "the tokenizer is not the one that was measured. Do not raise MAX_LENGTH to "
            "make this pass."
        )
    return longest


def assert_pad_token_agrees(tokenizer, model) -> int:
    """Blocking check D: one padding token, and both halves agree on its id.

    A decoder was never trained to pad, so it often arrives with no padding
    token defined at all, and the first attempt to build a batch fails with an
    error that reads like a tokenizer bug. Setting one is easy; the part that
    silently goes wrong is that the MODEL keeps a second copy of the id in its
    config, and a sequence-classification head reads the label off the last
    NON-PAD position. If the model's idea of "pad" differs from the
    tokenizer's, it reads the wrong position, learns something, and scores
    badly for no visible reason.
    """
    if tokenizer.pad_token_id is None:
        raise AssertionError("tokenizer has no pad token - build_tokenizer() sets one")
    if model.config.pad_token_id != tokenizer.pad_token_id:
        raise AssertionError(
            f"model.config.pad_token_id={model.config.pad_token_id} but "
            f"tokenizer.pad_token_id={tokenizer.pad_token_id}. A sequence-classification "
            "decoder pools the LAST NON-PAD token; disagreeing about which token that is "
            "makes it pool a padding position, and nothing raises."
        )
    if getattr(tokenizer, "padding_side", "right") != "right":
        raise AssertionError(
            f"padding_side is {tokenizer.padding_side!r}. Classification pools the last "
            "non-pad token, so padding goes on the RIGHT here. Left padding is for "
            "generation, which is a different section of this file."
        )
    return int(tokenizer.pad_token_id)


# =========================================================================
# 3. BUILDING THE MODEL
# =========================================================================

def build_tokenizer(model_name: str, padding_side: str = "right"):
    """The tokenizer, with a padding token guaranteed to exist.

    `padding_side` is an argument rather than a constant because the two jobs
    in this file want different answers. Classification pools the last non-pad
    token, so it pads on the RIGHT, and assert_pad_token_agrees() enforces it.
    Prompt scoring reads the model's prediction for the position after the
    prompt, so every prompt in a batch would have to END at the same index -
    left padding. Section 7 as written scores one prompt at a time and pads
    nothing, so the side is currently inert there; it is still set to "left" so
    that batching those prompts later is correct by default rather than
    silently off by however much padding each row happened to get.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        # eos is the conventional stand-in. It only has to be a token that
        # never appears inside a real instruction, so that "not pad" and
        # "part of the sentence" stay distinguishable.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def _dtype_kwarg(dtype):
    """`dtype=` on transformers >= 4.56, `torch_dtype=` before it.

    Not defensive programming for its own sake: Colab decides which
    transformers is preinstalled, the argument was renamed between the version
    this project pins and the versions Colab ships, and passing the old name to
    a new library is a TypeError in the middle of a GPU session.
    """
    import transformers

    parts = []
    for chunk in transformers.__version__.split(".")[:2]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return {"dtype": dtype} if tuple(parts) >= (4, 56) else {"torch_dtype": dtype}


def build_base_model(model_name: str, labels: list[str], precision: str,
                     pad_token_id: int):
    """Qwen3 with a 27-way classification head on top, and nothing else.

    Decision B1: `AutoModelForSequenceClassification`, not a generative model
    that writes the label as text. Four reasons hold on their own - it satisfies
    the LoRA requirement, the metrics come out of the box, it cannot invent a
    28th class, and softmax gives the confidence the stage-4 gate needs - and
    two more come from the stages after this one: stage 2 filters the vector
    store by intent, and a filter needs a value from a closed set; a stage-3
    agent needs a typed tool.

    `id2label` and `label2id` are built from the frozen artifacts/labels.json
    order and travel INSIDE the saved config. That is what makes the mapping a
    property of the checkpoint rather than of whichever notebook happens to
    load it. Rebuilding the list from `sorted(df["intent"].unique())` at
    prediction time works right up until an intent is missing from the frame
    being predicted, and then every label after it shifts by one.

    The head is created with random weights. That is expected, and it is also
    the reason section 7 exists: measuring an untrained model THROUGH this head
    would measure the random numbers in it.
    """
    import torch
    from transformers import AutoModelForSequenceClassification

    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label={i: label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
        **_dtype_kwarg(dtype),
    )
    model.config.pad_token_id = pad_token_id
    return model


def add_lora(model, r: int, precision: str):
    """Freeze the base, attach the adapter, and put the trainable parts in fp32.

    `task_type=SEQ_CLS` is not cosmetic. With the wrong task type the run
    trains, the loss goes down, and the model learns a different objective than
    the one being measured - the single most expensive silent failure available
    today, which is why it is also the first thing the round-trip check in
    section 6 verifies.

    The dtype line at the end is the standard LoRA recipe and it prevents a
    specific crash. Loading the base in fp16 saves half the memory, but fp16
    gradients cannot be unscaled by a gradient scaler and torch raises
    "Attempting to unscale FP16 gradients" on the first optimiser step. Casting
    only the ~0.1% of parameters that are trainable back to fp32 costs nothing
    and makes the update numerically sane; the forward pass stays in half
    precision under autocast.
    """
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=LORA_ALPHA_MULTIPLIER * r,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(TARGET_MODULES),
        modules_to_save=list(MODULES_TO_SAVE),
        bias="none",
    )
    model = get_peft_model(model, config)

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    return model


def trainable_parameter_counts(model) -> dict:
    """Trained parameters against total, and the ratio between them.

    This is the number that turns "we used LoRA" from a statement into an
    argument: how much capacity was added, at what cost. It goes in the report
    table next to the runtime, and together they are what makes r=16 beating
    r=4 by a hair an operational finding rather than a footnote.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": int(trainable),
        "total_params": int(total),
        "trainable_pct": round(100.0 * trainable / total, 4),
    }


def set_all_seeds(seed: int) -> None:
    """Every source of randomness this file touches, from one number."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================================
# 4. ENCODING
# =========================================================================

def encode_split(tokenizer, frame: pd.DataFrame, labels: list[str],
                 device, label_column: str = "intent") -> dict:
    """One split, tokenised once, as tensors already on the GPU.

    Padding to a fixed MAX_LENGTH rather than to the longest row in each batch.
    Dynamic padding is the usual advice and it is right when sequences are long
    and uneven; here every row is under 24 tokens, the saving is invisible, and
    fixed length removes a collator, a sampler and a whole class of shape bugs.

    The whole training set is 9,893 x 32 int64 = about 2.5 MB, so it lives on
    the GPU for the duration and there is no data loader at all. That is not a
    micro-optimisation - it removes worker processes, their seeds, and their
    non-determinism from the experiment.

    The label column is mapped through the frozen order, and an unknown label
    raises instead of becoming -1 and poisoning the loss silently.
    """
    import torch

    label2id = {label: i for i, label in enumerate(labels)}
    unknown = sorted(set(frame[label_column]) - set(label2id))
    if unknown:
        raise ValueError(
            f"labels not in artifacts/labels.json: {unknown}. The frozen list is the "
            "contract - the data changed, or the wrong split was loaded."
        )

    encoded = tokenizer(
        list(frame["instruction"]),
        truncation=True, max_length=MAX_LENGTH,
        padding="max_length", return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "labels": torch.tensor([label2id[x] for x in frame[label_column]],
                               dtype=torch.long, device=device),
        "n_rows": len(frame),
    }


# =========================================================================
# 5. TRAINING
# =========================================================================

@dataclass
class RunConfig:
    """One row of the experiment table, before it has been run.

    Everything that moves a score is a field here. Everything that is frozen
    for the day gets its module-level default and is recorded anyway, because
    "it was the default" is not something a reader of the table can verify a
    week later.
    """
    name: str
    r: int
    model_name: str = MAIN_MODEL
    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    batch_size: int = BATCH_SIZE
    grad_accum: int = GRAD_ACCUM
    warmup_ratio: float = WARMUP_RATIO
    weight_decay: float = WEIGHT_DECAY
    max_grad_norm: float = MAX_GRAD_NORM
    train_seed: int = TRAIN_SEED
    train_rows: int | None = None      # None means "all of them"; the debug run sets it
    notes: str = ""

    @property
    def lora_alpha(self) -> int:
        return LORA_ALPHA_MULTIPLIER * self.r


def _linear_schedule(optimizer, total_steps: int, warmup_steps: int):
    """Warm up linearly, then decay linearly to zero.

    Written out rather than imported so that the schedule is visible in the
    file that claims to fix it, and so that one more transformers API is not
    part of the day's version risk.
    """
    import torch

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        remaining = total_steps - warmup_steps
        return max(0.0, (total_steps - step) / max(1, remaining))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def predict_logits(model, encoded: dict, batch_size: int = 128, precision: str = "fp16"):
    """Forward pass over an encoded split. No gradients, model in eval mode.

    `model.eval()` matters more than it looks: LoRA dropout is active in
    training mode, so scoring without it makes the number depend on which
    dropout mask happened to be drawn.
    """
    import torch

    device = next(model.parameters()).device.type
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    was_training = model.training
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, encoded["n_rows"], batch_size):
            stop = start + batch_size
            with torch.autocast(device, dtype=dtype):
                logits = model(
                    input_ids=encoded["input_ids"][start:stop],
                    attention_mask=encoded["attention_mask"][start:stop],
                ).logits
            out.append(logits.float().cpu())
    if was_training:
        model.train()
    return torch.cat(out)


def score_encoded(model, encoded: dict, labels: list[str], precision: str,
                  batch_size: int = 128) -> tuple[dict, list[str], np.ndarray]:
    """Predictions, confidences, and the metrics - through src/evaluate.py.

    Returns the same dictionary shape the TF-IDF baselines returned yesterday,
    from the same function, so the two are on one ruler. That is the whole
    reason evaluate.py is a separate module.
    """
    import torch

    logits = predict_logits(model, encoded, batch_size, precision)
    probabilities = torch.softmax(logits, dim=-1)
    predicted_ids = probabilities.argmax(dim=-1).numpy()
    confidence = probabilities.max(dim=-1).values.numpy()

    y_pred = [labels[i] for i in predicted_ids]
    y_true = [labels[i] for i in encoded["labels"].cpu().numpy()]
    return E.evaluate_predictions(y_true, y_pred, labels), y_pred, confidence


def train_one_run(config: RunConfig, train_frame: pd.DataFrame, val_frame: pd.DataFrame,
                  labels: list[str], hardware: dict, output_dir: Path | str,
                  progress=print) -> dict:
    """The whole of one run: build, train, score every epoch, keep the best.

    One configuration in, one run record out (inside a dict that also carries
    the live model and its predictions, so the notebook can check and slice
    them without paying for a second forward pass). The record is the same
    shape src/evaluate.py produces for every other model in this project, so
    the fine-tuned rows and the baseline rows go into one table without any
    reshaping - and the per-epoch history, which no summary table can hold,
    rides along inside it.

    Checkpoint policy: evaluate on clean/val at the end of every epoch and
    write the adapter to disk only when macro-F1 improves. When the loop ends,
    what is on disk IS the best epoch, so there is no "load the best checkpoint"
    step that can be forgotten and no second copy in memory. The criterion was
    fixed in advance (SELECTION_METRIC), not chosen after seeing the curve.

    Nothing here touches clean/test. Today selects a configuration out of
    three, and a number used to select is no longer a neutral estimate of
    anything - that is what day 5 and a sealed test set are for.
    """
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    precision = hardware["precision"]
    # The device comes out of the hardware report rather than being hardcoded.
    # Not so this can be run on CPU - it cannot, usefully - but so that the
    # loop is exercisable by a test with a randomly initialised toy model, on
    # a laptop, before an hour of GPU time is spent finding a shape bug.
    device = hardware.get("device", "cuda")

    set_all_seeds(config.train_seed)

    tokenizer = build_tokenizer(config.model_name, padding_side="right")
    if config.train_rows is not None and config.train_rows < len(train_frame):
        # Stratified so that a small debug slice still contains all 27 classes;
        # a slice missing a class would make the head's output layer partly
        # untrained and the debug metrics meaningless in a confusing way.
        from .baselines import subsample_stratified
        train_frame = subsample_stratified(train_frame, config.train_rows,
                                           seed=config.train_seed)

    longest = assert_no_truncation(tokenizer, train_frame["instruction"])
    # The fingerprint of the rows this run actually saw, computed the same way
    # split_manifest.json computes it. The manifest assert at the top of the
    # notebook proves the FILES are right; this proves that the frame handed to
    # this function is the one that came out of them, subsample included.
    from .data import sha256_of_split
    train_sha = sha256_of_split(train_frame)

    model = build_base_model(config.model_name, labels, precision,
                             tokenizer.pad_token_id)
    assert_pad_token_agrees(tokenizer, model)
    model = add_lora(model, config.r, precision)
    model.to(device)
    counts = trainable_parameter_counts(model)

    train = encode_split(tokenizer, train_frame, labels, device)
    val = encode_split(tokenizer, val_frame, labels, device)

    steps_per_epoch = max(1, train["n_rows"] // config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    scheduler = _linear_schedule(optimizer, total_steps,
                                 int(round(WARMUP_RATIO * total_steps)))
    scaler = torch.amp.GradScaler(device, enabled=(precision == "fp16"))
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    generator = torch.Generator(device="cpu").manual_seed(config.train_seed)
    history: list[dict] = []
    # Every step's loss, not only the per-epoch mean. Three points cannot show
    # whether a run diverged and recovered, whether the warmup was too short,
    # or - the question the debug run exists to answer - whether the loss moved
    # at all inside a single epoch. It is a few thousand floats, and
    # regenerating it costs GPU time rather than a second.
    step_losses: list[float] = []
    best_score = -1.0
    best_epoch = -1
    best_metrics: dict | None = None
    best_predictions: list[str] | None = None
    best_confidence: np.ndarray | None = None

    started = time.perf_counter()
    model.train()
    for epoch in range(1, config.epochs + 1):
        order = torch.randperm(train["n_rows"], generator=generator).to(device)
        running_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for step in range(steps_per_epoch):
            index = order[step * config.batch_size:(step + 1) * config.batch_size]
            with torch.autocast(device, dtype=autocast_dtype):
                loss = model(
                    input_ids=train["input_ids"][index],
                    attention_mask=train["attention_mask"][index],
                    labels=train["labels"][index],
                ).loss
            scaler.scale(loss / config.grad_accum).backward()
            step_loss = float(loss.detach())
            step_losses.append(round(step_loss, 5))
            running_loss += step_loss
            n_batches += 1

            if (step + 1) % config.grad_accum == 0 or step + 1 == steps_per_epoch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        metrics, predictions, confidence = score_encoded(model, val, labels, precision)
        history.append({
            "epoch": epoch,
            "train_loss": round(running_loss / max(1, n_batches), 4),
            "val_accuracy": metrics["accuracy"],
            "val_f1_macro": metrics["f1_macro"],
            "val_f1_weighted": metrics["f1_weighted"],
            "learning_rate_end": scheduler.get_last_lr()[0],
            "elapsed_seconds": round(time.perf_counter() - started, 1),
        })
        progress(f"  epoch {epoch}/{config.epochs}  loss {history[-1]['train_loss']:.4f}"
                 f"  val macro-F1 {metrics['f1_macro']:.4f}")

        if metrics[SELECTION_METRIC] > best_score:
            best_score = metrics[SELECTION_METRIC]
            best_epoch = epoch
            best_metrics = metrics
            best_predictions = predictions
            best_confidence = confidence
            save_adapter(model, tokenizer, output_dir)

    runtime = time.perf_counter() - started

    record = E.run_record(
        name=config.name,
        config={
            "model": "qwen3+lora+seq_cls",
            "base_model": config.model_name,
            "task_type": "SEQ_CLS",
            "label_level": "intent",
            "train_rows": train["n_rows"],
            "eval_rows": val["n_rows"],
            "trained_on": "clean/train",
            "scored_on": "clean/val",
            "train_sha256": train_sha,
            "r": config.r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": list(TARGET_MODULES),
            "modules_to_save": list(MODULES_TO_SAVE),
            "epochs": config.epochs,
            "best_epoch": best_epoch,
            "selection_metric": SELECTION_METRIC,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "grad_accum": config.grad_accum,
            "warmup_ratio": config.warmup_ratio,
            "weight_decay": config.weight_decay,
            "max_length": MAX_LENGTH,
            "longest_sequence": longest,
            "precision": precision,
            "gpu_name": hardware["gpu_name"],
            "split_seed": SPLIT_SEED,
            "subsample_seed": config.train_seed if config.train_rows else None,
            "train_seed": config.train_seed,
            **counts,
        },
        metrics=best_metrics,
        runtime_seconds=runtime,
        notes=config.notes,
    )
    # run_record() fills library_versions() with the CPU stack, which is the
    # right answer for a scikit-learn baseline and the wrong one here: torch,
    # transformers, peft and the CUDA build are what produced this number, and
    # none of them can be reconstructed from the file next week.
    record["library_versions"] = gpu_environment(hardware)
    record["history"] = history
    record["step_losses"] = step_losses
    record["adapter_dir"] = str(output_dir)

    # A dict rather than a six-tuple. The notebook needs the live model for the
    # round-trip check, the encoded validation set for the error frame, and the
    # predictions for the confusion matrix - and a tuple of six things is read
    # by counting commas.
    return {
        "record": record,
        "model": model,
        "tokenizer": tokenizer,
        "val_encoded": val,
        "predictions": best_predictions,
        "confidence": best_confidence,
        "history": history,
        "step_losses": step_losses,
    }


# =========================================================================
# 6. SAVE, RELOAD, PREDICT
# =========================================================================
# The justification for the entire debug run. Two failures that the day-2 notes
# marked as critical - the wrong task type, and a classification head that is
# not saved with the adapter - both train successfully, both produce a falling
# loss, and both are only visible when the thing that was written to disk is
# read back in a fresh process. "No exception was raised" does not test this.

def save_adapter(model, tokenizer, output_dir: Path | str) -> Path:
    """Write the adapter and the tokenizer, and nothing else.

    A few megabytes rather than a few gigabytes, because the base weights are
    unchanged and are re-downloaded at load time. That is what makes it
    affordable to keep every run of the sweep instead of overwriting.

    The tokenizer goes with it. It has the padding token that the model's
    config now depends on, and a checkpoint that needs a tokenizer stored
    somewhere else is a checkpoint that will one day be loaded with the wrong
    one.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def load_adapter(adapter_dir: Path | str, base_model_name: str, labels: list[str],
                 precision: str, device: str = "cuda"):
    """Rebuild the model from disk the way tomorrow's notebook will have to.

    Deliberately built from scratch - a fresh base model, then the adapter on
    top - rather than by touching the object still in memory. The point of the
    check this feeds is to test what was WRITTEN, and reusing the live object
    would test nothing at all.
    """
    from peft import PeftModel

    tokenizer = build_tokenizer(str(adapter_dir), padding_side="right")
    base = build_base_model(base_model_name, labels, precision, tokenizer.pad_token_id)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.to(device)
    model.eval()
    return model, tokenizer


def roundtrip_check(model, tokenizer, encoded: dict, labels: list[str],
                    adapter_dir: Path | str, base_model_name: str,
                    precision: str, device: str = "cuda") -> dict:
    """Save, load from scratch, predict, and require identical predictions.

    This is check 6 of the seven, and it is the one that justifies the debug
    run existing at all. It is the only check today that catches BOTH of the
    silent failures: a wrong task type, and a classification head left out of
    `modules_to_save` and therefore re-initialised at random on load.

    Predicted LABELS are compared, not logits. fp16 arithmetic is not bit
    reproducible across two loads of the same weights, so demanding identical
    floats would fail for a reason that has nothing to do with the thing being
    tested. A randomly initialised head does not produce a slightly different
    argmax - it produces a completely different one - so labels are the right
    granularity. The largest logit difference is reported anyway, because a
    correct round trip keeps it near zero and a suspicious one does not.
    """
    import torch

    save_adapter(model, tokenizer, adapter_dir)
    before = predict_logits(model, encoded, precision=precision)

    reloaded, _ = load_adapter(adapter_dir, base_model_name, labels, precision, device)
    after = predict_logits(reloaded, encoded, precision=precision)

    before_ids = before.argmax(dim=-1).numpy()
    after_ids = after.argmax(dim=-1).numpy()
    agreement = float((before_ids == after_ids).mean())

    result = {
        "n_rows": int(len(before_ids)),
        "prediction_agreement": round(agreement, 6),
        "max_logit_difference": round(float((before - after).abs().max()), 6),
        "passed": agreement == 1.0,
    }

    del reloaded
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not result["passed"]:
        raise AssertionError(
            f"save -> reload -> predict changed {(1 - agreement) * 100:.1f}% of the "
            "predictions. STOP HERE. The two known causes are a task_type that is not "
            "SEQ_CLS, and 'score' missing from modules_to_save - both of which train "
            "perfectly and only fail on reload. There is no point running a day of "
            "experiments whose outputs cannot be loaded back."
        )
    return result


# =========================================================================
# 7. LABEL SCORING - THE "BEFORE" NUMBER
# =========================================================================
# Decision E2. The classification head of an untrained model is random, so
# measuring through it would measure the random numbers rather than the model.
# Free generation is the other option and it needs mapping rules for outputs
# that match none of the 27 labels, and those rules are themselves an
# arbitrary decision that moves the score.
#
# So: for each of the 27 intents, compute how strongly the model believes that
# label follows the prompt, and take the highest. Always returns a legal label,
# needs no mapping rules, and never touches the untrained head.

def build_causal_model(model_name: str, precision: str, pad_token_id: int):
    """Qwen3 with its ORIGINAL language-model head. No classification head at all.

    This is the point of decision E2 made concrete. build_base_model() puts a
    fresh 27-way head on the model and initialises it at random; asking that
    object what it thinks measures the random numbers. Here the model is loaded
    the way it was trained - predicting the next token - and the question asked
    is "how likely is the string `track_order` to come next", which is a
    question about the pretrained weights and nothing else.

    The weights are the same ones the adapter will be trained on, so the
    before/after difference in the report is one change and not two.
    """
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, **_dtype_kwarg(dtype))
    model.config.pad_token_id = pad_token_id
    model.eval()
    return model


def build_fewshot_examples(train_frame: pd.DataFrame, labels: list[str],
                           seed: int = FEWSHOT_SEED,
                           per_intent: int = FEWSHOT_PER_INTENT) -> list[tuple[str, str]]:
    """One demonstration per intent, drawn from clean/train with a fixed seed.

    From TRAIN only. A demonstration taken from clean/val puts a validation
    sentence inside the prompt used to score clean/val, which inflates the
    "before" number with no external symptom whatsoever.

    Random with a recorded seed rather than hand-picked. Choosing the clearest
    example of each intent by eye is exactly the step that smuggles in
    knowledge that would not be available on a live ticket, and it cannot be
    reproduced by anyone else.
    """
    rows = []
    for label in labels:                    # frozen order, so the prompt is stable
        pool = train_frame[train_frame["intent"] == label]
        if pool.empty:
            raise ValueError(f"no training rows for intent {label!r}")
        picked = pool.sample(n=min(per_intent, len(pool)), random_state=seed)
        rows.extend((row.instruction, label) for row in picked.itertuples())
    return rows


def build_prompt(tokenizer, text: str, labels: list[str],
                 examples: list[tuple[str, str]] | None = None) -> str:
    """The exact string the model sees, as one chat-formatted prompt.

    `enable_thinking=False` is not optional for Qwen3. With thinking on, the
    generation prompt opens a reasoning block, and the tokens being scored stop
    being the label - they become whatever the model would think first. The
    same flag matters again in stage 3, where a `<think>` block would end up
    inside a draft reply to a customer.
    """
    label_list = "\n".join(labels)
    messages = [{"role": "system", "content": PROMPT_SYSTEM if examples is None
                 else PROMPT_SYSTEM + "\n\nIntent labels:\n" + label_list}]
    for example_text, example_label in (examples or []):
        messages.append({"role": "user",
                         "content": PROMPT_USER_FEWSHOT.format(text=example_text)})
        messages.append({"role": "assistant", "content": example_label})
    template = PROMPT_USER_FEWSHOT if examples else PROMPT_USER
    messages.append({"role": "user",
                     "content": template.format(label_list=label_list, text=text)})

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def prompt_fingerprint(prompt: str) -> str:
    """sha256 of the exact prompt text, for the run record.

    The prompt is a hyper-parameter. Recording a hash of it is what makes
    "yesterday's number was different" a question with an answer instead of a
    mystery, and it is cheaper than storing the whole string in every record -
    the string itself is written once, to results/prompts/.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _cache_key_value_pairs(past):
    """Read (keys, values) per layer out of a KV cache, across library versions.

    transformers moved this from `.key_cache` / `.value_cache` lists to
    `.layers[i].keys` / `.layers[i].values` between 4.x and 5.x. This is the
    only place in the project that reaches inside a library object, and it is
    guarded by the self-test in score_labels().
    """
    layers = getattr(past, "layers", None)
    if layers is not None:
        return [(layer.keys, layer.values) for layer in layers]
    if hasattr(past, "key_cache"):
        return list(zip(past.key_cache, past.value_cache))
    raise TypeError(f"unrecognised cache object: {type(past)!r}")


def _repeat_cache(past, n: int):
    """A copy of a batch-1 KV cache, repeated n times along the batch axis.

    Why this exists: the prompt is identical for all 27 candidate labels, and
    it is the expensive part - about 180 tokens zero-shot and about 900
    few-shot, against roughly 4 tokens for a label. Scoring the candidates as
    27 independent sequences re-computes that prompt 27 times, which turns a
    ten-minute measurement into a two-and-a-half-hour one.

    A fresh cache object is built rather than the original being expanded in
    place, because the forward pass that follows APPENDS the candidate tokens
    to whatever cache it is handed, and mutating the prompt's cache would
    corrupt the next candidate.
    """
    from transformers.cache_utils import DynamicCache

    new = DynamicCache()
    for layer_index, (keys, values) in enumerate(_cache_key_value_pairs(past)):
        new.update(keys.expand(n, -1, -1, -1).contiguous(),
                   values.expand(n, -1, -1, -1).contiguous(), layer_index)
    return new


def _label_token_ids(tokenizer, labels: list[str]) -> list[list[int]]:
    """Each label as the token sequence the model would have to produce."""
    return [tokenizer(label, add_special_tokens=False)["input_ids"] for label in labels]


def cache_memory_estimate(model, prompt_tokens: int, n_labels: int = 27) -> dict:
    """How much GPU memory the 27-way cache copy will want, before it asks for it.

    The saving in _repeat_cache() is paid for in memory: the prompt's key-value
    cache is duplicated once per candidate label. For a few-shot prompt of
    ~1,200 tokens on Qwen3-1.7B that is a few gigabytes, which fits on a T4
    next to the model and would not fit next to a larger one.

    Printed by the notebook before the run rather than discovered as a CUDA
    out-of-memory error forty minutes in. If it does not fit, the fix is
    fewer demonstrations - not fast=False, which fits easily and takes hours.

    That last clause was false for one afternoon and is worth keeping honest:
    fast=False fits easily *because* _score_one_prompt() bounds the logits it
    asks for. Without that bound the naive path wanted 7.5 GB of vocabulary
    logits at few-shot length and died inside the self-test - the number this
    function estimates was never the ceiling that was actually hit.
    """
    config = model.config
    n_layers = config.num_hidden_layers
    n_kv = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    bytes_per_element = 2                       # fp16 / bf16
    per_token = 2 * n_layers * n_kv * head_dim * bytes_per_element
    return {
        "prompt_tokens": prompt_tokens,
        "kv_bytes_per_token": per_token,
        "cache_gb_per_sequence": round(per_token * prompt_tokens / 1e9, 3),
        "cache_gb_for_all_labels": round(per_token * prompt_tokens * n_labels / 1e9, 2),
    }


def _score_one_prompt(model, tokenizer, prompt: str, label_ids: list[list[int]],
                      precision: str, fast: bool) -> np.ndarray:
    """Sum of token log-probabilities for each candidate label. Shape (27, 2).

    Column 0 is the sum, column 1 is the number of tokens - the caller turns
    those into whichever of the two declared scores it wants. Both come out of
    the same forward pass, so recording both is free.

    Two implementations of the same quantity: `fast` reuses the prompt's
    key-value cache across the 27 candidates, `naive` scores 27 complete
    sequences. The naive one is obviously correct and is the reference the
    self-test compares against.

    `model` must be the causal-LM form (build_causal_model). A
    sequence-classification model returns one vector per SEQUENCE rather than
    one per token, so it cannot answer this question at all - and the very
    reason for measuring this way is that its head is untrained.
    """
    import torch

    device = next(model.parameters()).device
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    prompt_ids = tokenizer(prompt, add_special_tokens=False,
                           return_tensors="pt")["input_ids"].to(device)
    n_labels = len(label_ids)
    lengths = [len(ids) for ids in label_ids]
    width = max(lengths)
    pad_id = tokenizer.pad_token_id

    candidates = torch.full((n_labels, width), pad_id, dtype=torch.long, device=device)
    for i, ids in enumerate(label_ids):
        candidates[i, :len(ids)] = torch.tensor(ids, device=device)

    with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
        if fast:
            # logits_to_keep=1: the only row of prompt logits this path reads is the
            # last one, and running the head over all ~900 prompt positions allocates
            # a few hundred MB to throw them away. The read below is [-1] either way,
            # so a version that ignores the argument still gets the right row.
            prompt_out = model(input_ids=prompt_ids, use_cache=True, logits_to_keep=1)
            prompt_length = prompt_ids.shape[1]
            first_logits = prompt_out.logits[:, -1, :].float().expand(n_labels, -1)

            mask = torch.ones((n_labels, prompt_length + width),
                              dtype=torch.long, device=device)
            for i, length in enumerate(lengths):
                mask[i, prompt_length + length:] = 0
            step_out = model(
                input_ids=candidates,
                attention_mask=mask,
                past_key_values=_repeat_cache(prompt_out.past_key_values, n_labels),
                use_cache=True,   # a passed cache is ignored by some versions when this
                                  # is False; the returned cache is thrown away anyway
                cache_position=torch.arange(prompt_length, prompt_length + width,
                                            device=device),
            )
            # position t of a candidate is predicted by the logits at t-1;
            # position 0 is predicted by the last position of the prompt.
            logits = torch.cat([first_logits.unsqueeze(1),
                                step_out.logits[:, :-1, :].float()], dim=1)
        else:
            prompt_length = prompt_ids.shape[1]
            full = torch.cat([prompt_ids.expand(n_labels, -1), candidates], dim=1)
            mask = torch.ones_like(full)
            for i, length in enumerate(lengths):
                mask[i, prompt_length + length:] = 0
            # logits_to_keep is not a speed tweak here, it is what makes this path
            # runnable at all. Without it the head is applied at every position of
            # every sequence: 27 candidates x ~900 few-shot tokens x a 151,936-token
            # vocabulary in half precision is 7.5 GB, requested as ONE contiguous
            # block, on top of the model and the 27-way cache the fast path just
            # built. A T4 refuses, and it refuses inside the self-test - the check
            # that exists to make this path trustworthy is the thing that dies.
            # Only the `width` rows that predict candidate tokens are ever read.
            keep_last = width + 1        # ... of which prompt_length - 1 is the first
            out = model(input_ids=full, attention_mask=mask, use_cache=False,
                        logits_to_keep=keep_last)
            # A transformers that does not know the argument absorbs it into **kwargs
            # and returns full-length logits - and then `[:, :width]` would silently
            # read the START of the prompt and score nonsense. Slice according to what
            # came back, not according to what was asked for.
            if out.logits.shape[1] == keep_last:
                logits = out.logits[:, :width, :].float()
            else:
                logits = out.logits[:, prompt_length - 1:prompt_length - 1 + width, :].float()

    logprobs = torch.log_softmax(logits, dim=-1)
    taken = logprobs.gather(2, candidates.unsqueeze(-1)).squeeze(-1)   # (27, width)
    keep = torch.zeros_like(taken)
    for i, length in enumerate(lengths):
        keep[i, :length] = 1.0
    totals = (taken * keep).sum(dim=1).cpu().numpy()
    return np.stack([totals, np.asarray(lengths, dtype=float)], axis=1)


def score_labels(model, tokenizer, texts, labels: list[str], precision: str,
                 examples: list[tuple[str, str]] | None = None,
                 fast: bool = True, self_test_rows: int = 3,
                 progress=print) -> dict:
    """The "before" number: classify by scoring all 27 labels, no head involved.

    Runs the fast and the naive implementation against each other on the first
    few rows before trusting either. The fast path reaches into a library
    object to reuse a key-value cache, that object changed shape between
    transformers versions, and a wrong answer there would look like a plausible
    score rather than a crash. Disagreement raises; there is no silent
    fallback, because a silent fallback would turn a two-and-a-half-hour run
    into a surprise.

    Returns predictions under both declared scoring rules. The headline was
    fixed in LABEL_SCORING before any of this ran.
    """
    import torch

    label_ids = _label_token_ids(tokenizer, labels)
    texts = list(texts)

    if fast and self_test_rows:
        for text in texts[:self_test_rows]:
            prompt = build_prompt(tokenizer, text, labels, examples)
            quick = _score_one_prompt(model, tokenizer, prompt, label_ids, precision, True)
            slow = _score_one_prompt(model, tokenizer, prompt, label_ids, precision, False)
            gap = float(np.abs(quick[:, 0] - slow[:, 0]).max())
            # Top-1 agreement is the decisive check - that is the quantity the
            # measurement actually uses. The numeric tolerance is loose on
            # purpose: half-precision arithmetic does not reassociate the same
            # way in the two paths, and failing the day over 0.01 of log-prob
            # would be a false alarm.
            if gap > 0.5 or quick[:, 0].argmax() != slow[:, 0].argmax():
                raise RuntimeError(
                    f"cached and uncached label scoring disagree by {gap:.4f}. The "
                    "key-value cache layout of this transformers version is not the one "
                    "_repeat_cache() handles. Re-run with fast=False - it is correct and "
                    "about 20x slower - and record the fallback in the work log."
                )
        progress(f"  self-test passed on {self_test_rows} rows (cached == uncached)")

    sums = np.zeros((len(texts), len(labels)))
    counts = np.zeros((len(texts), len(labels)))
    started = time.perf_counter()
    for i, text in enumerate(texts):
        prompt = build_prompt(tokenizer, text, labels, examples)
        scored = _score_one_prompt(model, tokenizer, prompt, label_ids, precision, fast)
        sums[i], counts[i] = scored[:, 0], scored[:, 1]
        if (i + 1) % 200 == 0 or i + 1 == len(texts):
            elapsed = time.perf_counter() - started
            progress(f"  {i + 1}/{len(texts)} rows  {elapsed:.0f}s elapsed  "
                     f"~{elapsed / (i + 1) * (len(texts) - i - 1):.0f}s left")

    normalised = sums / counts
    example_prompt = build_prompt(tokenizer, texts[0], labels, examples)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "prompt_example": example_prompt,
        "prompt_sha256": prompt_fingerprint(example_prompt),
        "prompt_tokens": len(tokenizer(example_prompt, add_special_tokens=False)["input_ids"]),
        "scoring_rule": LABEL_SCORING,
        "runtime_seconds": time.perf_counter() - started,
        "pred_mean_logprob_per_token": [labels[i] for i in normalised.argmax(axis=1)],
        "pred_sum_logprob": [labels[i] for i in sums.argmax(axis=1)],
        "confidence": np.exp(normalised.max(axis=1)),
        "n_distinct_predictions": int(len(set(normalised.argmax(axis=1)))),
    }


def label_scoring_record(name: str, scored: dict, y_true, labels: list[str],
                         model_name: str, hardware: dict, shots: int,
                         fewshot_seed: int | None, notes: str = "") -> dict:
    """One run record for a zero-shot or few-shot measurement.

    Built here rather than in the notebook so that the two "before" runs and
    the three sweep runs land in the summary table with the same column names.
    A record assembled by hand in a cell omits a different field every time,
    and then the table quietly compares runs that were not configured alike.

    Both declared scoring rules are scored and both go in. The headline is
    LABEL_SCORING, which was fixed before any of these numbers existed; the
    other is recorded because it came free from the same forward pass and
    because a reader is entitled to see that the choice was not the flattering
    one.
    """
    headline = scored["pred_" + LABEL_SCORING]
    metrics = E.evaluate_predictions(y_true, headline, labels)
    alternative = E.evaluate_predictions(y_true, scored["pred_sum_logprob"], labels)

    record = E.run_record(
        name=name,
        config={
            "model": "qwen3+label_scoring",
            "base_model": model_name,
            "task_type": "label_scoring",
            "label_level": "intent",
            "train_rows": 0,               # nothing was trained
            "eval_rows": len(headline),
            "trained_on": "-",
            "scored_on": "clean/val",
            "shots": shots,
            "fewshot_seed": fewshot_seed,
            "fewshot_source": "clean/train" if shots else "-",
            "scoring_rule": scored["scoring_rule"],
            "prompt_sha256": scored["prompt_sha256"],
            "prompt_tokens": scored["prompt_tokens"],
            "decoding": "greedy (argmax over the 27 label scores)",
            "max_length": None,            # the prompt is not truncated
            "precision": hardware["precision"],
            "gpu_name": hardware["gpu_name"],
            "split_seed": SPLIT_SEED,
            "subsample_seed": None,
            "train_seed": None,
        },
        metrics=metrics,
        runtime_seconds=scored["runtime_seconds"],
        notes=notes,
    )
    record["library_versions"] = gpu_environment(hardware)
    record["alternative_scoring"] = {
        "rule": "sum_logprob",
        "accuracy": alternative["accuracy"],
        "f1_macro": alternative["f1_macro"],
    }
    record["n_distinct_predictions"] = scored["n_distinct_predictions"]
    return record


def history_frame(records: list[dict]) -> pd.DataFrame:
    """Every epoch of every run as one long table, keyed by (run, epoch).

    This is the one thing summarise_runs() cannot hold and the reason the GPU
    runs keep a per-run file at all: the loss curve and the epoch-by-epoch
    validation score. Re-creating it costs compute units rather than a second,
    which is exactly the distinction that made the day-2 rule - no per-run JSON
    beside the summary - the right rule there and the wrong one here.
    """
    rows = []
    for record in records:
        for epoch in record.get("history", []):
            rows.append({"run": record["name"], **epoch})
    return pd.DataFrame(rows)


# =========================================================================
# 8. ENVIRONMENT
# =========================================================================

def gpu_environment(hardware: dict) -> dict:
    """Everything about this machine that cannot be reconstructed next week.

    The GPU name and the precision are in every run's config already; this is
    the version half. A run recorded without them is a run that cannot be
    compared to anything, and a sweep of three numbers whose runtimes came from
    two different GPUs is not a sweep.
    """
    import torch
    import transformers

    try:
        import peft
        peft_version = peft.__version__
    except Exception:
        peft_version = None

    return {
        **E.library_versions(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft_version,
        "cuda": torch.version.cuda,
        **hardware,
    }
