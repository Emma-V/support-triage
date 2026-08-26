"""
Data preparation. Runs once, on CPU, ahead of any model work.

Scope of this module:
- Download the Bitext dataset, or read it from disk if already present.
- Light text cleaning: collapse repeated whitespace, drop empty rows.
  Nothing more.
- Identify near-duplicate sentences. The dataset was generated
  automatically, so many rows are variations of one template. Sibling rows
  must land on the same side of the split, otherwise the model is
  evaluated on sentences it has effectively already seen.
- Split the data into three parts: train, val (used while tuning), and
  test (opened only at the end).
- Save a fingerprint of the split - the seed used and the row count of
  each part - so the split can be reproduced exactly, and the same numbers
  reported.
- Maintain a registry of every raw row (full_corpus.csv): its template
  family, the split side that family was assigned to, and whether it is
  the row chosen to represent the family in the classification files. The
  training files keep one row per family, which drops 42% of the corpus;
  this registry is what turns that into a documented collapse rather than
  a silent deletion, and it is the source for the uncollapsed-alternative
  ablation.

This file contains no print statements and no plots. Only functions that
take a table and return a table. notebooks/01_data.ipynb runs them and
reports the results.

--------------------------------------------------------------------------
READING ORDER (the file is organised to be read top to bottom)
  1. Constants          - every number that must never drift lives here.
  2. Loading            - fetch the raw CSV, verify its shape, and read
                          back the frozen split files.
  3. Cleaning           - whitespace, empty rows, exact duplicates.
  4. Near-duplicates    - grouping sibling sentences into families.
  5. Splitting          - stratified 70/15/15, for both clean and naive.
  6. Leakage measurement- how close a test row sits to its nearest train row.
  7. Artifacts          - labels / intent->category, the output contract.
  8. Manifest           - the identity card of the split.
  9. Token lengths      - instruction lengths in real model tokens.
  10. Taxonomy diagnostic - how much each intent label bundles distinct goals.
  11. Canonical space   - the one corpus every similarity number is defined
                          against, so notebooks cannot drift apart.
  12. Residual leakage  - the same measurement repeated in an independent
                          feature space, since one of the two cannot fail
                          by construction.
  13. Ambiguity ceiling - near-identical sentences carrying different labels.
  14. Cleaning audit    - what each cleaning step measurably removes.
  15. Uncollapsed set   - the training set the family collapse gives up.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# =========================================================================
# 1. CONSTANTS
# =========================================================================
# Anything that could silently change a result is a named constant here,
# and nowhere else. This is not a matter of tidiness: if the near-duplicate
# threshold lived inside a notebook cell, an incidental edit from 0.90 to
# 0.88 would change the split, and two training runs would no longer be
# comparable, with no error raised anywhere.

HF_REPO = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
RAW_FILENAME = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")

# The two split families and the three parts each has. Named here so that
# load_split() can reject a typo instead of raising FileNotFoundError from
# somewhere inside pandas - "naive" vs "clean" is a two-character difference
# that produces a plausible but wrong number.
SPLIT_NAMES = ("clean", "naive")
SPLIT_PARTS = ("train", "val", "test")

# The three numbers to verify before anything else runs. If the upstream
# file changes, these stop matching, and every figure in the report would
# otherwise silently describe a different dataset.
EXPECTED_N_ROWS = 26_872
EXPECTED_N_INTENTS = 27
EXPECTED_N_CATEGORIES = 11

# The split seed, frozen for the life of the project.
# This is NOT the training seed - that is a separate constant, in a
# different file, and is the one that varies across the noise-floor runs.
# Two seeds sharing a name is a common way to mistake "different data" for
# "training variance".
SPLIT_SEED = 42

# The subsample seed - a third, distinct "seed" in this project, named
# apart from the other two for the same reason they are named apart from
# each other. It determines which 9,893 of naive/train's 17,187 rows form
# the size control. Changing it silently would produce a frame with the
# right row count and class balance but the wrong rows.
SUBSAMPLE_SEED = 42

# The size control's own directory. It is not a split but a subset of one,
# so it lives beside clean/ and naive/ with its own manifest rather than
# being folded into split_manifest.json, whose hashes several other checks
# depend on.
NAIVE_SUB_DIR = "naive_sub"

# 70 / 15 / 15. train_test_split does not support a three-way split
# directly, so the split is performed twice: first 70/30, then the
# remaining 30 is halved.
HOLDOUT_FRACTION = 0.30
VAL_SHARE_OF_HOLDOUT = 0.50

# Cosine-similarity threshold above which two sentences are treated as
# siblings of the same template. This value is derived rather than typed
# in by preference: it is the output of the selection rule in
# notebooks/01_data.ipynb (section 1.3). The rule is fixed before any
# candidate is measured - take the loosest threshold, i.e. the one that
# retains the most training data, at which the resulting split is still
# clean - applied to a scan over {0.80, 0.85, 0.90, 0.91, 0.92, 0.95}, and
# the notebook asserts that it lands on this constant. Changing the rule or
# the scan is what would need to update this value.
NEAR_DUP_THRESHOLD = 0.90

# One vector space, defined once, used for three purposes: clustering
# near-duplicates, measuring leakage in the naive split, and verifying the
# clean split. Using one vectorizer for clustering and a different one for
# measurement would leave "cleaned at 0.90" and "0.9 similar" as numbers in
# different units, and the claim would not be checkable. char_wb (character
# n-grams inside word boundaries) is used rather than word n-grams because
# the corpus deliberately contains typos: "delivery" and "deliverly" share
# almost every 3-gram but zero words.
TFIDF_PARAMS = dict(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,          # an n-gram seen once is noise, and it doubles the vocabulary
    sublinear_tf=True,  # 1+log(tf): a word repeated 5x is not 5x more important
)

# Number of tokens the model is given per ticket. Measured, not guessed:
# token_lengths() over the whole corpus with the real Qwen3 tokenizer gives
# p99 = 19 and max = 24, so 32 truncates zero rows with room to spare (see
# section 1.7 of notebooks/01_data.ipynb).
# Defined here rather than in the training module because it is a property
# of this dataset. Attention cost grows quadratically with sequence
# length: the naive default of 512 would cost roughly 250x for identical
# predictions, and even a slip to 64 is 4x slower with no error raised.
MAX_LENGTH = 32

# Columns written to the split CSVs.
# `response` is deliberately excluded. It is the templated answer, and if
# it reaches the feature side the model reads the answer off its own input
# and scores ~0.999. Omitting it from the file makes that mistake
# impossible rather than merely discouraged.
# `flags` and `category` remain: neither is a feature either, but both are
# needed for error slicing and category-level metrics.
SPLIT_COLUMNS = ["row_id", "instruction", "intent", "category", "flags", "dup_group"]

# Columns of data/processed/full_corpus.csv - the registry of every raw
# row. Not a training file: nothing trains on it. It exists so that no row
# is silently discarded by the one-representative-per-family collapse, and
# so the ~2.3k exact-duplicate rows retain a split assignment rather than
# falling outside the guarantee. train_side_rows() reads it to build the
# uncollapsed alternative that the family-collapse ablation is measured
# against. `response` is absent here too, for the same reason as above:
# nothing under data/processed/ ever contains the answer text.
FULL_CORPUS_COLUMNS = ["row_id", "instruction", "intent", "category", "flags",
                       "dup_group", "split", "is_representative", "is_exact_duplicate"]


# =========================================================================
# 2. LOADING
# =========================================================================

def load_raw(raw_dir: Path | str = DEFAULT_RAW_DIR, download_if_missing: bool = True) -> pd.DataFrame:
    """Returns the raw Bitext table, downloading it once if not already on disk.

    A local copy matters because Colab wipes the machine when the session
    ends. Re-downloading 19 MB every session is slow and, more importantly,
    would silently pick up any upstream change. A local file plus the hash
    recorded in the manifest turns "the dataset changed" from an invisible
    event into a failed assertion.
    """
    raw_dir = Path(raw_dir)
    local_path = raw_dir / RAW_FILENAME

    if not local_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"{local_path} not found and download_if_missing=False")
        # Imported here, not at module level, so that `import src.data` works
        # on a machine with no huggingface_hub installed as long as the CSV
        # is already on disk.
        from huggingface_hub import hf_hub_download

        raw_dir.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(HF_REPO, RAW_FILENAME, repo_type="dataset")
        # copy rather than symlink: Windows/OneDrive does not handle symlinks well
        local_path.write_bytes(Path(cached).read_bytes())

    return pd.read_csv(local_path, encoding="utf-8")


def verify_raw(df: pd.DataFrame) -> dict:
    """Checks the three headline shape numbers and raises if any moved.

    This is deliberately an exception rather than a warning: a warning
    scrolls past in a notebook, after which every figure in the report
    would describe a dataset nobody actually inspected.
    """
    found = {
        "n_rows": len(df),
        "n_intents": df["intent"].nunique(),
        "n_categories": df["category"].nunique(),
    }
    expected = {
        "n_rows": EXPECTED_N_ROWS,
        "n_intents": EXPECTED_N_INTENTS,
        "n_categories": EXPECTED_N_CATEGORIES,
    }
    mismatches = {k: (expected[k], found[k]) for k in expected if expected[k] != found[k]}
    if mismatches:
        raise ValueError(
            "The raw dataset does not match the expected shape (expected, found): "
            f"{mismatches}. The upstream file may have been updated - stop and "
            "re-check every number in the report before continuing."
        )
    return found


def sha256_of_file(path: Path | str) -> str:
    """Hash of the raw CSV as it sits on disk. Recorded in the manifest."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split(split_name: str, part: str,
               processed_dir: Path | str = DEFAULT_PROCESSED_DIR) -> pd.DataFrame:
    """Reads one frozen split file, e.g. load_split("clean", "train").

    Centralising this avoids notebook cells writing a path by hand. The two
    split families differ by one word in the middle of the path, and
    getting it wrong does not crash: training on `naive/train` while
    scoring `clean/val` runs without error and answers a question nobody
    asked. Validating both arguments against a fixed list turns that into a
    ValueError naming the typo.

    The `response` guard enforces the rule that no file under
    data/processed/ carries the `response` column. `response` is the
    templated answer; if it reaches the feature side the model reads the
    answer off its own input and scores ~0.999, which looks like success.
    SPLIT_COLUMNS already excludes it, so this can only fire if the CSVs
    are regenerated with a different column list - exactly the situation
    this check is meant to catch.
    """
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"split_name must be one of {SPLIT_NAMES}, got {split_name!r}")
    if part not in SPLIT_PARTS:
        raise ValueError(f"part must be one of {SPLIT_PARTS}, got {part!r}")

    path = Path(processed_dir) / split_name / f"{part}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The split CSVs are gitignored and rebuilt from the "
            "seed - run notebooks/01_data.ipynb first."
        )

    df = pd.read_csv(path, encoding="utf-8")
    if "response" in df.columns:
        raise AssertionError(
            f"{path} contains a `response` column. That is the answer text, and any "
            "model that sees it scores ~0.999 while learning nothing. Rebuild the "
            "split with SPLIT_COLUMNS before using this file."
        )
    return df


def load_all_splits(processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
                    ) -> dict[str, dict[str, pd.DataFrame]]:
    """Loads all six CSVs, in the nested shape verify_against_manifest() expects.

    {"clean": {"train": df, "val": df, "test": df}, "naive": {...}}

    Returning exactly this shape means the manifest check is two lines at
    the top of a notebook rather than a dict assembled by hand, where an
    omitted part would verify five files while reporting success.

    Note that `test` is loaded as well. Loading it is not the same as
    inspecting it - the manifest check must hash all six files to actually
    verify the split. Scoring on test is a separate, deliberate step that
    stays gated until the final run.
    """
    return {
        name: {part: load_split(name, part, processed_dir) for part in SPLIT_PARTS}
        for name in SPLIT_NAMES
    }


# -------------------------------------------------------------------------
# 2b. THE SIZE CONTROL
# -------------------------------------------------------------------------
# naive/train has 17,187 rows and clean/train has 9,893, so any score
# difference between the two protocols could otherwise be attributed to
# training-set size alone. naive_sub is naive/train cut down to
# clean/train's row count, which removes that explanation and leaves one
# variable: whether the training set contains siblings of the evaluation
# sentences.
#
# This subsample is written to disk with its own manifest rather than
# redrawn inline, because a frame redrawn independently would have the
# right row count and class balance but not necessarily the same rows,
# producing a plausible score that is not actually comparable to earlier
# runs against it.

def build_naive_sub(naive_train: pd.DataFrame, clean_train: pd.DataFrame,
                    seed: int = SUBSAMPLE_SEED) -> pd.DataFrame:
    """Draws the size control: naive/train, cut to clean/train's row count.

    The target size is read from clean/train rather than hardcoded as
    9893, so the two frames cannot drift apart without this raising. The
    draw itself calls baselines.subsample_stratified rather than
    reimplementing it, since a second implementation of "draw n rows
    keeping class proportions" would be a second source of truth.
    """
    from .baselines import subsample_stratified
    return subsample_stratified(naive_train, n_rows=len(clean_train), seed=seed)


def write_naive_sub(frame: pd.DataFrame, manifest: dict,
                    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
                    seed: int = SUBSAMPLE_SEED) -> dict:
    """Writes naive_sub/train.csv and its own small manifest. Returns the manifest.

    `drawn_from_sha256` fingerprints the frame it was drawn from, so the
    control cannot outlive the split it controls for: if naive/train is
    ever rebuilt differently, this file's provenance stops matching and
    load_naive_sub() reports it.
    """
    directory = Path(processed_dir) / NAIVE_SUB_DIR
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / "train.csv", index=False, encoding="utf-8")

    sub_manifest = {
        "what": "naive/train subsampled to clean/train's row count - the size "
                "control for every naive-versus-clean comparison",
        "drawn_from": "naive/train",
        "drawn_from_sha256": manifest["splits"]["naive"]["train"]["sha256"],
        "subsample_seed": seed,
        "stratified_on": "intent",
        "n_rows": len(frame),
        "sha256": sha256_of_split(frame),
        "drawn_by": "src.baselines.subsample_stratified",
        "not_in_split_manifest": (
            "this is a subset of a split rather than a split, and split_manifest.json "
            "is settled - rewriting it would move hashes that the assert at the top "
            "of every notebook depends on"),
    }
    write_json(sub_manifest, directory / "subsample_manifest.json")
    return sub_manifest


def load_naive_sub(processed_dir: Path | str = DEFAULT_PROCESSED_DIR) -> pd.DataFrame:
    """Reads the size control, verifying its sha256 on the way in.

    Verified here rather than left to a notebook cell because the failure
    being guarded against is silent. A naive_sub with the wrong rows trains
    fine, scores plausibly, and controls for nothing - there is no
    exception anywhere to surface that.
    """
    directory = Path(processed_dir) / NAIVE_SUB_DIR
    path, manifest_path = directory / "train.csv", directory / "subsample_manifest.json"
    if not path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"{path} or its manifest is missing. Build it with "
            "`python tools/build_naive_sub.py`, which verifies the draw against "
            "the committed baseline scores before writing.")

    frame = pd.read_csv(path, encoding="utf-8")
    sub_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = sha256_of_split(frame)
    if actual != sub_manifest["sha256"]:
        raise AssertionError(
            f"{path} does not match its manifest.\n"
            f"  on disk   {actual}\n"
            f"  manifest  {sub_manifest['sha256']}\n"
            "These are not the rows the committed baselines were measured on, so "
            "nothing scored against them is comparable. Rebuild with "
            "tools/build_naive_sub.py.")
    if len(frame) != sub_manifest["n_rows"]:
        raise AssertionError(
            f"{path} has {len(frame)} rows, manifest says {sub_manifest['n_rows']}")
    return frame


# =========================================================================
# 3. CLEANING
# =========================================================================
# The governing rule for this section: whatever is not done to a live
# incoming ticket must not be done to the training data either. Departing
# from that is train/serving skew, which is why there is no lowercasing, no
# punctuation stripping, no stopword removal and no stemming here. A
# question mark is a real signal for query intents, and a preposition is
# literally the only thing separating get_invoice from check_invoice.

_WHITESPACE = re.compile(r"\s+")


def normalise_whitespace(df: pd.DataFrame, column: str = "instruction") -> pd.DataFrame:
    """Collapses runs of whitespace to a single space and strips the ends.

    This is safe under the skew rule because it is also what would be done
    to a live ticket: a double space is a typing artefact, not a signal,
    and would otherwise make two identical sentences look different to the
    exact-duplicate check that runs next.
    """
    out = df.copy()
    out[column] = out[column].astype(str).str.replace(_WHITESPACE, " ", regex=True).str.strip()
    return out


def drop_empty_rows(df: pd.DataFrame, column: str = "instruction") -> pd.DataFrame:
    """Drop rows whose text is empty after normalisation."""
    return df[df[column].str.len() > 0].copy()


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keeps one row per (instruction, intent) pair.

    The `subset` argument is the key detail. pandas' default compares
    every column, including `flags` and `response`; two rows with
    identical text but a different flag string both survive, and the data
    appears cleaned when nothing was removed. On this dataset the default
    removes 0 rows, while the correct subset removes 2,318 - the notebook
    reports both numbers side by side as the cheapest demonstration of the
    trap.
    """
    return df.drop_duplicates(subset=["instruction", "intent"]).copy()


def find_label_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where the exact same sentence carries more than one intent.

    Not a defect to patch, but a ceiling to measure: if the same sentence
    appears with two labels, no model can be correct on both, so 100% is
    unreachable and the honest headline is "93.1 against a ceiling of
    98.5" rather than "93.1". Returns an empty frame when there are none,
    which is itself a result worth stating.
    """
    per_text = df.groupby("instruction")["intent"].nunique()
    conflicted = per_text[per_text > 1].index
    return df[df["instruction"].isin(conflicted)].sort_values("instruction")


# =========================================================================
# 4. NEAR-DUPLICATES
# =========================================================================

def build_vector_space(texts: pd.Series | list[str]) -> tuple[TfidfVectorizer, csr_matrix]:
    """Fits the single TF-IDF space used by every similarity computation here.

    Fitted on all rows, before splitting, deliberately. This can look like
    a violation of "fit on train only", so it is worth stating explicitly
    why it is not:
    - "fit on train only" protects a model from seeing test data. Nothing
      here is a model; no parameter of this vectorizer reaches the
      classifier.
    - Grouping siblings is data curation and must happen before splitting.
      Clustering each split separately would leave each one internally
      clean while still leaking across the boundary.
    - Using one space for both clustering and measurement keeps the two
      claims in the same units: "removed everything above 0.90" and
      "0.24% of test rows are above 0.90" are then comparable statements.

    That last point has a limit that is worth stating plainly. One space
    makes the two numbers comparable; it does not make the second one
    evidence for the first. Restricted to a single intent, "% of test rows
    within 0.90 of a train row" measured in this space is zero by
    construction, since that is exactly the relation the families were
    built from. Section 12 re-measures it in an independent space, where
    the checkable version of the claim lives.

    TF-IDF output is L2-normalised, so X @ X.T is cosine similarity
    directly - no separate normalisation step, and no 27k x 27k dense
    matrix.
    """
    vec = TfidfVectorizer(**TFIDF_PARAMS)
    X = vec.fit_transform(texts)
    return vec, X


def near_duplicate_groups(X: csr_matrix, intents: pd.Series | np.ndarray,
                          threshold: float = NEAR_DUP_THRESHOLD) -> np.ndarray:
    """Assigns every row a template-family id. Returns an int array, one per row.

    Three design points worth being able to justify:

    1. Blocked by intent. All-pairs over 26,872 rows is 722M cells (~2.9 GB
       in float32), which is prohibitive on a CPU-only runtime. Near-
       duplicates always fall inside one intent, since they were expanded
       from the same template, so the computation runs as 27 blocks of
       ~900 rows instead - seconds rather than minutes.
       (Cross-intent near-duplicates are a distinct phenomenon - semantic
       conflicts - and are deliberately kept; see the notebook.)

    2. Character n-grams, from the shared space above.

    3. `connected_components`, not pairwise deletion. If A~B and B~C but A
       and C are not similar to each other, all three still originate from
       one template. Deleting in pairs would leave A and C on opposite
       sides of the split: cleaned in appearance, still leaking in effect.
       `connected_components` computes the transitive closure of the
       "is similar to" relation, which is exactly the family.
    """
    intents = np.asarray(intents)
    groups = np.full(X.shape[0], -1, dtype=np.int64)
    next_id = 0

    # np.unique keeps the order deterministic across runs and machines,
    # which matters because these ids end up in a committed CSV.
    for intent in np.unique(intents):
        idx = np.flatnonzero(intents == intent)
        Xi = X[idx]

        # Sparse cosine similarity inside the block.
        S = (Xi @ Xi.T).tocoo()

        # Keep only the edges above the threshold. Building the adjacency
        # matrix from the surviving (row, col) pairs is cheaper and
        # clearer than thresholding the sparse matrix in place.
        keep = S.data >= threshold
        adjacency = coo_matrix(
            (np.ones(keep.sum()), (S.row[keep], S.col[keep])),
            shape=S.shape,
        )

        n_components, labels = connected_components(adjacency, directed=False)
        groups[idx] = labels + next_id
        next_id += n_components

    return groups


def threshold_scan(X: csr_matrix, intents: pd.Series | np.ndarray,
                   thresholds=(0.80, 0.85, 0.90, 0.91, 0.92, 0.95)) -> pd.DataFrame:
    """Reports how many families survive at each candidate threshold.

    One of two tables the selection rule in notebooks/01_data.ipynb is
    applied to: this one prices each candidate in data retained, while the
    leakage measurement beside it prices the same candidates in leakage
    left behind. NEAR_DUP_THRESHOLD is whatever the rule returns from the
    pair.

    The candidates bracket the chosen threshold on both sides, with 0.91
    and 0.92 included so that "why not just above it?" is answered by a
    measured row rather than an assertion. A flat curve indicates the
    choice barely matters; a steep one indicates a real sensitivity, which
    is a finding in its own right rather than a problem.
    """
    rows = []
    n = X.shape[0]
    for t in thresholds:
        g = near_duplicate_groups(X, intents, threshold=t)
        n_groups = len(np.unique(g))
        rows.append({
            "threshold": t,
            "n_groups": n_groups,
            "retained_pct": round(100 * n_groups / n, 1),
            "mean_family_size": round(n / n_groups, 2),
        })
    return pd.DataFrame(rows)


def pick_representatives(df: pd.DataFrame, group_column: str = "dup_group",
                         seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Keeps exactly one row per family. This is what makes the clean split clean.

    The representative is drawn at random, with a fixed seed.
    """
    shuffled = df.sample(frac=1.0, random_state=seed)
    representatives = shuffled.drop_duplicates(subset=group_column)
    # Restored to the original row order so the written CSV is
    # deterministic and readable alongside the raw file.
    return representatives.sort_index().copy()


# =========================================================================
# 5. SPLITTING
# =========================================================================

def split_stratified(df: pd.DataFrame, seed: int = SPLIT_SEED,
                     holdout_fraction: float = HOLDOUT_FRACTION,
                     val_share: float = VAL_SHARE_OF_HOLDOUT,
                     stratify_column: str = "intent",
                     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70 / 15 / 15, stratified on intent, via the two-step split sklearn requires.

    Stratified on `intent` (27 classes), never on `category` (11). Stratifying
    on the 11 categories would not guarantee that all 27 intents appear in
    val: a rare intent could end up absent, macro-F1 would then average 26
    classes instead of 27, and nothing would flag it.
    """
    train, holdout = train_test_split(
        df,
        test_size=holdout_fraction,
        stratify=df[stratify_column],
        random_state=seed,
    )
    val, test = train_test_split(
        holdout,
        test_size=1.0 - val_share,
        stratify=holdout[stratify_column],
        random_state=seed,
    )
    return train.copy(), val.copy(), test.copy()


def build_full_corpus(df_all: pd.DataFrame, df_deduped: pd.DataFrame,
                      clean_parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every raw row, stamped with its family, split side, and role.

    Rationale for this file's existence: the clean split keeps one row per
    family, which is correct for classification - train and test stay
    template-uniform, so the score measures generalisation across
    templates - but would otherwise act as a silent archive loss, discarding
    the other ~10k rows with no record of their existence or split
    assignment. This function discards nothing. Every row inherits the
    split side of its family's representative:

    - one family -> one representative -> exactly one side, so the
      inheritance is well-defined;
    - a whole family is always on one side, so any view of this file
      filtered by `split` is leak-free by construction: it cannot contain a
      sibling of a sentence held out on the other side. This is what makes
      train_side_rows() a different view of the frozen split rather than a
      second split requiring its own seed and hash.

    Exact duplicates (dropped before clustering) are re-attached through
    their (instruction, intent) key - unambiguous, since the dataset has
    zero exact label conflicts - and marked is_exact_duplicate.

    The `split` column refers to the clean split only; the naive control
    split keeps its own six CSVs and plays no role here.

    df_all      : whitespace-normalised rows incl. exact duplicates (has row_id)
    df_deduped  : after drop_exact_duplicates, with dup_group assigned
    clean_parts : {"train": ..., "val": ..., "test": ...} - the representative
                  frames exactly as split_stratified returned them
    """
    # 1. every raw row -> its family, through the exact-duplicate key.
    #    validate= raises if the key is not unique on the deduped side.
    full = df_all.merge(df_deduped[["instruction", "intent", "dup_group"]],
                        on=["instruction", "intent"], how="left",
                        validate="many_to_one")
    if full["dup_group"].isna().any():
        raise AssertionError(
            "some raw rows matched no family - the dedup key drifted between "
            "df_all and df_deduped")

    # 2. family -> side, read off where the family's representative landed
    split_of_family: dict[int, str] = {}
    for part_name, frame in clean_parts.items():
        for g in frame["dup_group"]:
            split_of_family[int(g)] = part_name
    full["split"] = full["dup_group"].map(split_of_family)
    if full["split"].isna().any():
        raise AssertionError(
            "some families have no split side - clean_parts does not cover "
            "every dup_group")

    # 3. roles
    rep_ids = set(pd.concat([f["row_id"] for f in clean_parts.values()]))
    full["is_representative"] = full["row_id"].isin(rep_ids)
    full["is_exact_duplicate"] = ~full["row_id"].isin(set(df_deduped["row_id"]))

    # deterministic file order: the raw file order, which row_id preserves
    return (full[FULL_CORPUS_COLUMNS]
            .sort_values("row_id")
            .reset_index(drop=True))


# =========================================================================
# 6. LEAKAGE MEASUREMENT
# =========================================================================

def max_similarity_to_train(X: csr_matrix, train_index: np.ndarray, test_index: np.ndarray,
                            chunk_size: int = 500) -> np.ndarray:
    """For every test row: the cosine similarity to its nearest train row.

    Chunked deliberately. The full product for the naive split is
    3,684 x 17,187, which is 63M float32 = 253 MB dense; in chunks of 500 it
    is 34 MB at a time, for the same result with no memory spike.

    `train_index` / `test_index` are positions in X, so the caller must
    pass row positions rather than pandas labels - this is what `row_id` is
    for.
    """
    A = X[train_index]
    B = X[test_index]
    out = np.zeros(B.shape[0], dtype=np.float32)
    for start in range(0, B.shape[0], chunk_size):
        block = (B[start:start + chunk_size] @ A.T).toarray()
        out[start:start + chunk_size] = block.max(axis=1)
    return out


def leakage_summary(max_sim: np.ndarray, exact_overlap_fraction: float) -> dict:
    """The three headline leakage figures, plus the median for context."""
    return {
        "exact_pct": round(100 * float(exact_overlap_fraction), 2),
        "ge_0.95_pct": round(100 * float((max_sim >= 0.95).mean()), 2),
        "ge_0.90_pct": round(100 * float((max_sim >= 0.90).mean()), 2),
        "ge_0.80_pct": round(100 * float((max_sim >= 0.80).mean()), 2),
        "median_max_similarity": round(float(np.median(max_sim)), 3),
    }


def exact_overlap(train: pd.DataFrame, test: pd.DataFrame, column: str = "instruction") -> float:
    """Fraction of test rows whose text appears verbatim in train."""
    return float(test[column].isin(set(train[column])).mean())


# =========================================================================
# 7. ARTIFACTS  (the output contract: two small JSON files, committed)
# =========================================================================

def build_labels(df: pd.DataFrame) -> list[str]:
    """The 27 intents, sorted - the one artifact deliberately built from all rows.

    The model outputs integers 0..26, not names; this list is the
    translation. It is a fixed contract rather than a learned parameter, so
    it is built once from every label in the dataset. If the order ever
    shifted between runs, a saved checkpoint would return confidently wrong
    class names with no error raised.
    """
    return sorted(df["intent"].unique().tolist())


def build_intent2cat(df: pd.DataFrame) -> dict[str, str]:
    """intent -> category, derived from the file itself rather than the docs.

    The published dataset card disagrees with the file in at least two
    ways (it lists 10 categories and omits several intents), so the
    mapping is derived and then verified. A failure of the assertion below
    would mean one intent maps to two categories - a finding to report,
    not something to patch quietly.
    """
    per_intent = df.groupby("intent")["category"].nunique()
    ambiguous = per_intent[per_intent > 1]
    if len(ambiguous) > 0:
        raise ValueError(
            f"These intents map to more than one category: {ambiguous.to_dict()}. "
            "This contradicts the assumption that intent -> category is a function "
            "and must be reported, not silently fixed."
        )
    return df.groupby("intent")["category"].first().to_dict()


# =========================================================================
# 8. MANIFEST
# =========================================================================

def sha256_of_split(df: pd.DataFrame) -> str:
    """A fingerprint of one split file: its texts and labels, in file order.

    Text and label are both included, since a rebuild that shuffled the
    labels but kept the texts would otherwise pass. Row order is included
    too, since order is part of the file's content and is deterministic
    given the seed.
    """
    payload = "\n".join(f"{t}\t{y}" for t, y in zip(df["instruction"], df["intent"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_of_full_corpus(df: pd.DataFrame) -> str:
    """Fingerprint of the full-corpus registry.

    Unlike sha256_of_split, this includes dup_group, split and
    is_representative, since those assignments are part of the file's
    content. A rebuild that kept every text but moved one family to the
    other side must fail the manifest check rather than pass it.
    """
    payload = "\n".join(
        f"{r}\t{t}\t{y}\t{g}\t{s}\t{int(rep)}"
        for r, t, y, g, s, rep in zip(df["row_id"], df["instruction"], df["intent"],
                                      df["dup_group"], df["split"],
                                      df["is_representative"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest(raw_path: Path | str, counts: dict, splits: dict[str, dict[str, pd.DataFrame]],
                   threshold: float = NEAR_DUP_THRESHOLD, seed: int = SPLIT_SEED,
                   library_versions: dict | None = None,
                   full_corpus: pd.DataFrame | None = None) -> dict:
    """The identity card of the split: small, committed to git, and load-bearing.

    The split CSVs themselves are gitignored and rebuilt from the seed.
    This file is the proof that a rebuild produces the same result.
    Without it, an upstream dataset update or a scikit-learn version bump
    could move the split unnoticed; with it, the bootstrap assertion fires
    immediately.

    When the full-corpus registry is passed, its fingerprint is included
    too: the family-to-side assignment is as much a part of the frozen
    split as the six CSVs are.
    """
    manifest = {
        "seed": seed,
        "near_dup_threshold": threshold,
        "scheme": "70/15/15 stratified on intent, two-step",
        "source": {
            "hf_repo": HF_REPO,
            "filename": RAW_FILENAME,
            "sha256": sha256_of_file(raw_path),
        },
        "tfidf_params": {**TFIDF_PARAMS, "ngram_range": list(TFIDF_PARAMS["ngram_range"])},
        "counts": counts,
        "splits": {
            name: {
                part: {"n_rows": len(frame), "sha256": sha256_of_split(frame)}
                for part, frame in parts.items()
            }
            for name, parts in splits.items()
        },
        "library_versions": library_versions or {},
    }
    if full_corpus is not None:
        manifest["full_corpus"] = {
            "n_rows": len(full_corpus),
            "n_families": int(full_corpus["dup_group"].nunique()),
            "n_representatives": int(full_corpus["is_representative"].sum()),
            "sha256": sha256_of_full_corpus(full_corpus),
        }
    return manifest


def verify_against_manifest(manifest: dict, splits: dict[str, dict[str, pd.DataFrame]],
                            full_corpus: pd.DataFrame | None = None) -> None:
    """Re-hashes the splits on disk and compares to the manifest, raising on mismatch.

    This assertion is what makes the reproducibility claim self-verifying.
    It runs at the top of every notebook, the only place it is useful: a
    check that runs after the numbers are produced is decoration.

    `full_corpus` is optional so that a notebook reading only the six split
    CSVs can still verify them without rebuilding the registry.
    """
    problems = []
    for name, parts in splits.items():
        for part, frame in parts.items():
            expected = manifest["splits"][name][part]
            actual = sha256_of_split(frame)
            if actual != expected["sha256"]:
                problems.append(f"{name}/{part}: sha256 {actual[:12]} != {expected['sha256'][:12]}")
            if len(frame) != expected["n_rows"]:
                problems.append(f"{name}/{part}: {len(frame)} rows != {expected['n_rows']}")
    if full_corpus is not None and "full_corpus" in manifest:
        expected = manifest["full_corpus"]
        actual = sha256_of_full_corpus(full_corpus)
        if actual != expected["sha256"]:
            problems.append(f"full_corpus: sha256 {actual[:12]} != {expected['sha256'][:12]}")
        if len(full_corpus) != expected["n_rows"]:
            problems.append(f"full_corpus: {len(full_corpus)} rows != {expected['n_rows']}")
    if problems:
        raise AssertionError(
            "The rebuilt split does not match split_manifest.json:\n  " + "\n  ".join(problems)
            + "\nDo not continue - every number produced from here would describe a "
              "different dataset than the one in the report."
        )


def write_json(obj, path: Path | str) -> None:
    """Writes JSON with consistent formatting, used for every JSON file in the project."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# =========================================================================
# 9. TOKEN LENGTHS
# =========================================================================

def token_lengths(texts: pd.Series | list[str], model_name: str = "Qwen/Qwen3-1.7B") -> np.ndarray:
    """Length of every text in tokens of the actual model tokenizer.

    Character counts cannot answer this question: max_length is measured
    in tokens, and training time grows roughly with the square of the
    sequence length. Measuring gives a real number (p99 is 19 here) rather
    than the naive default of 512, which would cost roughly 250x the
    attention for the same result.

    transformers is imported inside the function deliberately: this is the
    CPU data module, and `import src.data` should not pull in a heavy
    library on a machine that is only rebuilding the split.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    encoded = tok(list(texts), add_special_tokens=True)["input_ids"]
    return np.array([len(ids) for ids in encoded])


# =========================================================================
# 10. TAXONOMY DIAGNOSTIC
# =========================================================================

def subgoal_separability(df: pd.DataFrame, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Measures how much each intent label bundles more than one user goal.

    Method, per intent: cluster the responses into two groups (the NLG
    engine used different reply templates for different user goals, so
    response text is a cheap proxy for what the customer actually wanted),
    then train a small classifier to predict the response cluster from the
    instruction alone, 3-fold cross-validated.

    Reading the `lift` column (accuracy minus the majority baseline):
      ~0.0 - the two response clusters are template noise; the instruction
             does not encode them, and the label is as fine as the data
             supports.
      high - the customer's own words reliably signal a distinction that
             the intent label discards (subscribe vs unsubscribe, file-a-
             claim vs complain, upgrade-tier vs switch-user).

    This is a diagnostic rather than training code: nothing it fits is
    kept, and it does not touch the split. It measures a limit of the
    label scheme: where lift is high on most intents, the intent is
    coarser than the customer's actual goal, so a correct intent does not
    by itself identify what the customer wanted. This is a stated
    limitation of intent classification on this taxonomy, not a defect in
    the model.

    k=2 is a deliberately coarse instrument - the true number of goals per
    intent is not necessarily two - so `lift` is a lower bound on the
    bundling, not an exact measurement of it.
    """
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    rows = []
    for intent, g in df.groupby("intent"):
        R = TfidfVectorizer(max_features=3000, stop_words="english").fit_transform(g["response"])
        response_cluster = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(R).labels_
        I = TfidfVectorizer(**TFIDF_PARAMS).fit_transform(g["instruction"])
        accuracy = cross_val_score(LogisticRegression(max_iter=2000),
                                   I, response_cluster, cv=3).mean()
        majority = float(np.bincount(response_cluster).max()) / len(g)
        rows.append({
            "intent": intent,
            "n_rows": len(g),
            "smaller_cluster_pct": round(100 * (1 - majority), 1),
            "instruction_predicts_cluster": round(accuracy, 3),
            "majority_baseline": round(majority, 3),
            "lift": round(accuracy - majority, 3),
        })
    return (pd.DataFrame(rows)
            .sort_values("lift", ascending=False)
            .reset_index(drop=True))


# =========================================================================
# 11. CANONICAL SPACE
# =========================================================================
# Everything below this line is measurement. Nothing here changes what is
# written to disk: the six split CSVs, full_corpus.csv and the manifest
# are frozen, and a function in this section that altered them would
# invalidate every committed hash. These functions read the frozen split
# and describe it.
#
# This section exists because of a discrepancy found by measuring the same
# quantity twice. notebooks/01_data.ipynb measured "how similar is a
# held-out row to its nearest training row" in a TF-IDF space fitted on
# all 24,554 deduplicated rows, and reported 0.24%. notebooks/02_baselines.ipynb
# measured what reads like the same quantity, but re-fitted the vectoriser
# on train+val alone (12,013 rows), and reported 1.70%. Neither number is
# wrong; they are simply not in the same units, since min_df=2 over a
# smaller corpus keeps a smaller vocabulary and every cosine value shifts.
# Two figures describing "residual leakage" that differ by 7x, with
# nothing on disk marking them as incomparable, is exactly how an
# incorrect figure reaches a report.
#
# The fix is not a larger warning comment. It is to make the corpus that
# defines the space a named, reusable object, so that measuring in a
# different space is a deliberate act rather than an accident.


def build_canonical_corpus(raw_dir: Path | str = DEFAULT_RAW_DIR,
                           download_if_missing: bool = False) -> pd.DataFrame:
    """The deduplicated frame that every similarity number in this project uses.

    This is exactly the sequence notebooks/01_data.ipynb runs inline before
    clustering anything:

        load_raw -> row_id -> normalise_whitespace -> drop_empty_rows
                 -> drop_exact_duplicates

    and it returns the same 24,554 rows in the same order, so a vector
    space fitted on this frame is the vector space the split was built in.

    Notebook 01 keeps its inline version, since watching the row count fall
    step by step is most of what that notebook demonstrates. It asserts
    that this function reproduces it rather than calling it directly,
    giving one definition and a proof that the two agree.

    The returned frame carries `pos`, its row position in that frame, which
    is what a fitted matrix is indexed by. Rows should generally be looked
    up by `row_id` through the helpers below; `pos` exists so those helpers
    can do their job, not so callers need to reason in positions.
    """
    df = load_raw(raw_dir, download_if_missing=download_if_missing)
    df = df.reset_index(names="row_id")
    df = drop_empty_rows(normalise_whitespace(df))
    df = drop_exact_duplicates(df).reset_index(drop=True)
    df["pos"] = np.arange(len(df))
    return df


def position_index(corpus: pd.DataFrame) -> dict[int, int]:
    """Maps row_id -> row position in the matrix fitted on `corpus`.

    A short function with a real purpose. Similarity code indexes a matrix
    by position, but every frame in this project is identified by
    `row_id`, and positions are specific to one particular frame. Passing
    positions taken from `clean/train` into a matrix fitted on the full
    corpus would not raise an error - it would silently compare the wrong
    sentences and return a plausible number. Going through row_id every
    time removes that class of mistake.
    """
    return {int(r): i for i, r in enumerate(corpus["row_id"])}


def similarity_profile(train: pd.DataFrame, evaluation: pd.DataFrame,
                       X: csr_matrix, corpus: pd.DataFrame,
                       chunk_size: int = 500) -> pd.DataFrame:
    """Per evaluation row: how close it is to training, and to what.

    Returns one row per evaluation row:
      row_id           the evaluation row
      intent           its true label
      max_sim_same     cosine to the nearest training row of the same intent
      max_sim_any      cosine to the nearest training row, any intent
      nearest_intent   the label of that nearest training row

    Splitting the similarity into two columns is the central point, since
    the single number `max_similarity_to_train` returns bundles two
    opposite phenomena:

    - `max_sim_same` is leakage. A held-out sentence that paraphrases a
      training sentence carrying the same label is one the model has
      effectively already been trained on, and scoring on it flatters the
      model.
    - a high `max_sim_any` with a low `max_sim_same` is ambiguity, the
      opposite of leakage. The nearest neighbour is a near-identical
      sentence under a different label ("refund {{X}}" vs "I expect a
      refund of {{X}}"), so the row is not easier than average - it is one
      of the hardest rows in the set, and it caps what any model can
      score.

    Reporting the bundled number alone would let a reader mistake an
    ambiguity ceiling for leftover leakage, the more damaging and
    incorrect of the two readings.

    `X` must be a matrix fitted on `corpus`, in `corpus` row order. Rows
    are looked up by row_id, so `train` and `evaluation` may come from any
    frame as long as their rows exist in `corpus`.
    """
    where = position_index(corpus)
    missing = sorted({int(r) for r in evaluation["row_id"]} - set(where)) \
        + sorted({int(r) for r in train["row_id"]} - set(where))
    if missing:
        raise KeyError(
            f"{len(missing)} row_ids are not in `corpus` (first few: {missing[:5]}). "
            "The frame the matrix was fitted on and the frames being compared do "
            "not describe the same dataset."
        )

    train_pos = np.array([where[int(r)] for r in train["row_id"]])
    eval_pos = np.array([where[int(r)] for r in evaluation["row_id"]])
    train_intents = train["intent"].to_numpy()
    eval_intents = evaluation["intent"].to_numpy()

    A = X[train_pos]
    B = X[eval_pos]
    n = B.shape[0]
    max_same = np.zeros(n, dtype=np.float32)
    max_any = np.zeros(n, dtype=np.float32)
    nearest = np.empty(n, dtype=object)

    # Chunked for the same reason as max_similarity_to_train: the dense
    # product of every eval row against every train row is hundreds of MB,
    # and is never needed all at once.
    for start in range(0, n, chunk_size):
        block = (B[start:start + chunk_size] @ A.T).toarray()
        for j in range(block.shape[0]):
            i = start + j
            row = block[j]
            best = int(row.argmax())
            max_any[i] = row[best]
            nearest[i] = train_intents[best]
            same = train_intents == eval_intents[i]
            # An intent with no training rows at all cannot leak into this
            # row. This also cannot occur in a stratified split, so 0.0
            # here is a defined answer rather than a silent one.
            max_same[i] = row[same].max() if same.any() else 0.0

    return pd.DataFrame({
        "row_id": evaluation["row_id"].to_numpy(),
        "intent": eval_intents,
        "max_sim_same": max_same,
        "max_sim_any": max_any,
        "nearest_intent": nearest,
    })


# =========================================================================
# 12. RESIDUAL LEAKAGE, MEASURED IN TWO SPACES
# =========================================================================
# The sharpest question this project invites is: if everything the
# similarity metric calls similar has been removed, and that same metric
# is then used to measure what remains, what should be expected?
#
# The honest answer is that in the clustering space the residual is not
# merely low, it is zero by construction. Families are the connected
# components of the >=0.90 graph within an intent, and a whole family
# always lands on one side of the split. So a held-out row and a training
# row that are same-intent and >=0.90 similar would have to be one family
# on two sides, which cannot happen. Measured on the committed split, the
# largest same-intent similarity observed is 0.8999 - the distribution is
# truncated exactly at the threshold, which is the signature of a
# constraint rather than a measurement.
#
# That does not make the split dirty. It makes the evidence circular, and
# a circular claim carries less weight in a report than a smaller honest
# one. The remedy is to measure the same quantity in a feature space that
# played no part in building the split, where the answer is free to be
# non-zero.

# Word unigrams and bigrams: the standard text-classification feature
# space, and - the reason it is used here - one that shares no
# construction with char_wb(3,5). A residual measured here is not implied
# by how the families were built.
INDEPENDENT_TFIDF_PARAMS = dict(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)


def residual_leakage_two_spaces(train: pd.DataFrame, evaluation: pd.DataFrame,
                                corpus: pd.DataFrame) -> pd.DataFrame:
    """Produces the residual-leakage table that avoids the circularity above.

    Runs the identical measurement in both feature spaces and returns one
    row per (space, relation):

      space     `char_wb(3,5)` - the space the families were defined in
                `word(1,2)`    - an independent space, no part in the split
      relation  `same_intent`  - leakage
                `any_intent`   - leakage plus ambiguity

    Both spaces are fitted on the same `corpus`, so the only difference
    between the two rows is the analyzer. That matters: re-fitting on a
    smaller frame changes min_df and every idf weight, and it was exactly
    that difference - not a real change in leakage - that produced two
    incompatible numbers (0.24% and 1.70%) for the same split.

    Reading the table: in char_wb, `same_intent >= 0.90` is 0.00% and that
    is guaranteed rather than observed. The number worth quoting is the
    word-space one, which is small but was free to have been large.

    The percentage alone is not the full picture. `nearest_train_pairs`
    returns the sentences behind it, and on this split they turn out not
    to be sibling leakage at all, but pairs whose only distinguishing
    token was pruned by min_df=2. A residual with an explanation is
    evidence; a residual without one is only a number.
    """
    spaces = {
        "char_wb(3,5)": TFIDF_PARAMS,
        "word(1,2)": INDEPENDENT_TFIDF_PARAMS,
    }
    rows = []
    for space_name, params in spaces.items():
        X = TfidfVectorizer(**params).fit_transform(corpus["instruction"])
        profile = similarity_profile(train, evaluation, X, corpus)
        for relation, column in (("same_intent", "max_sim_same"),
                                 ("any_intent", "max_sim_any")):
            sim = profile[column].to_numpy()
            rows.append({
                "space": space_name,
                "relation": relation,
                "guaranteed_zero": space_name.startswith("char_wb") and relation == "same_intent",
                "ge_0.95_pct": round(100 * float((sim >= 0.95).mean()), 2),
                "ge_0.90_pct": round(100 * float((sim >= 0.90).mean()), 2),
                "ge_0.80_pct": round(100 * float((sim >= 0.80).mean()), 2),
                "median": round(float(np.median(sim)), 3),
                # six decimals, not four: the char_wb same-intent maximum is
                # 0.899984, and rounding that to 0.9000 would make a number
                # strictly below the threshold look like it sits on it.
                "max": round(float(sim.max()), 6),
            })
    return pd.DataFrame(rows)


def nearest_train_pairs(train: pd.DataFrame, evaluation: pd.DataFrame,
                        X: csr_matrix, corpus: pd.DataFrame,
                        threshold: float = NEAR_DUP_THRESHOLD,
                        same_intent_only: bool = True,
                        chunk_size: int = 500) -> pd.DataFrame:
    """Returns the actual sentence pairs behind a residual-leakage percentage.

    `residual_leakage_two_spaces` reports that 2.36% of clean/val rows sit
    within 0.10 cosine of a same-intent training row in the independent
    word space. That number alone does not explain why. This function
    returns the pairs, which is what turns the percentage into a claim.

    On the committed split the explanation is specific and worth stating
    directly. The word-space residual is not sibling leakage; it is
    `min_df=2` pruning the one token that distinguishes two sentences:

        val  : "do ya ship toFinland"      train: "do ya ship toUSA"
        val  : "makie complaint"           train: "complaint"
        val  : "seeingbill from {{X}}"     train: "seebills from {{X}}"

    "toFinland", "makie" and "seeingbill" each occur once in the corpus, so
    min_df drops them, and what remains is identical. In character n-grams
    the same pairs score 0.367 to 0.869 - all below the 0.90 clustering
    threshold, confirming the families were right to keep them apart.

    So the residual is a property of the measurement space, not of the
    split. That is a stronger answer to "how do you know your split is
    clean" than the zero the clustering space returns by construction,
    because it is an answer that could have come out the other way.

    A second-order finding belongs in the limitations discussion: for
    those rows the word-level TF-IDF baseline is not merely uncertain, it
    is blind - two sentences with different meanings share one feature
    vector. That is a property of the baseline rather than of the data,
    and it is part of why char_wb scores higher.
    """
    where = position_index(corpus)
    train_pos = np.array([where[int(r)] for r in train["row_id"]])
    eval_pos = np.array([where[int(r)] for r in evaluation["row_id"]])
    train_intents = train["intent"].to_numpy()
    train_texts = train["instruction"].to_numpy()
    train_ids = train["row_id"].to_numpy()
    eval_intents = evaluation["intent"].to_numpy()
    eval_texts = evaluation["instruction"].to_numpy()
    eval_ids = evaluation["row_id"].to_numpy()

    A = X[train_pos]
    B = X[eval_pos]
    found = []
    for start in range(0, B.shape[0], chunk_size):
        block = (B[start:start + chunk_size] @ A.T).toarray()
        for j in range(block.shape[0]):
            i = start + j
            row = block[j]
            if same_intent_only:
                # Copy first: `row` is a view into `block`, and zeroing it
                # in place would not corrupt anything in this loop but
                # would if it ever read the block twice. Cheap insurance
                # against a future edit.
                row = row.copy()
                row[train_intents != eval_intents[i]] = 0.0
            best = int(row.argmax())
            if row[best] >= threshold:
                found.append({
                    "similarity": round(float(row[best]), 4),
                    "eval_row_id": int(eval_ids[i]),
                    "intent": eval_intents[i],
                    "eval_instruction": eval_texts[i],
                    "train_row_id": int(train_ids[best]),
                    "train_intent": train_intents[best],
                    "train_instruction": train_texts[best],
                })

    return (pd.DataFrame(found, columns=["similarity", "eval_row_id", "intent",
                                         "eval_instruction", "train_row_id",
                                         "train_intent", "train_instruction"])
            .sort_values("similarity", ascending=False)
            .reset_index(drop=True))


# =========================================================================
# 13. AMBIGUITY CEILING
# =========================================================================

def cross_intent_neighbours(corpus: pd.DataFrame, X: csr_matrix,
                            threshold: float = NEAR_DUP_THRESHOLD,
                            chunk_size: int = 400) -> pd.DataFrame:
    """Rows whose nearest neighbour is near-identical but carries a different label.

    `find_label_conflicts` answers the exact-match version of this question
    and returns nothing: no sentence in this dataset appears verbatim under
    two intents. That is a real result, but an earlier reading of it drew
    too strong a conclusion ("no measurable labelling ceiling from
    ambiguity"). The soft version is not empty:
    "refund {{Currency Symbol}}{{Refund Amount}}" is labelled get_refund
    and "I expect a refund of {{Currency Symbol}}{{Refund Amount}}" is
    labelled track_refund, and no classifier can be correct about both.

    These are not defects to clean away. Removing them would mean deleting
    the hard cases to inflate the score, which this project is built to
    avoid.

    It is worth resisting the temptation to call this an accuracy ceiling.
    It is not one, and this was checked rather than assumed: only 6 of the
    2,120 clean/val rows have a cross-intent twin (the family collapse
    removes most of them, since they cluster in the large invoice
    families), and both TF-IDF baselines classify all 6 correctly. A row
    with a near-identical neighbour under another label is not
    unclassifiable - the single token that differs, "see" against "get",
    is exactly what a bag-of-words model keys on.

    What the measurement does support is a statement about the taxonomy:
    the label scheme separates intents on distinctions this fine, so 0.83%
    of the corpus sits one word away from a different label. That is a
    caution about paraphrase robustness, and the companion to the
    subgoal-separability finding - this taxonomy is fine where the wording
    is and coarse where the goal is - not a bound on the score.

    Exhaustive rather than sampled. An earlier 3,000-row sample was in fact
    accurate for the frame it ran on, estimating 1.23% against a true 1.21%
    on the 26,872 pre-deduplication rows, so sampling was not the source of
    error. The exhaustive computation is used anyway because it takes
    twenty seconds, and the corpus matters: the same measurement over the
    24,554 deduplicated rows the split is actually drawn from gives 0.83%.
    Any quoted figure should state which corpus it is measured on.

    Each row is reported once, against its single nearest cross-intent
    neighbour, so the count is "rows that have a cross-intent twin" - the
    correct unit for a ceiling. A mutual pair therefore appears twice, once
    from each side.
    """
    intents = corpus["intent"].to_numpy()
    texts = corpus["instruction"].to_numpy()
    row_ids = corpus["row_id"].to_numpy()
    found = []

    for start in range(0, X.shape[0], chunk_size):
        block = (X[start:start + chunk_size] @ X.T).toarray()
        for j in range(block.shape[0]):
            i = start + j
            row = block[j]
            row[i] = 0.0                      # a row is not its own neighbour
            row[intents == intents[i]] = 0.0  # same-intent neighbours are families
            best = int(row.argmax())
            if row[best] >= threshold:
                found.append({
                    "similarity": round(float(row[best]), 3),
                    "row_id": int(row_ids[i]),
                    "intent": intents[i],
                    "instruction": texts[i],
                    "twin_row_id": int(row_ids[best]),
                    "twin_intent": intents[best],
                    "twin_instruction": texts[best],
                })

    return (pd.DataFrame(found, columns=["similarity", "row_id", "intent", "instruction",
                                         "twin_row_id", "twin_intent", "twin_instruction"])
            .sort_values("similarity", ascending=False)
            .reset_index(drop=True))


# =========================================================================
# 14. CLEANING AUDIT
# =========================================================================

def whitespace_impact(df_raw: pd.DataFrame) -> dict:
    """Measures what whitespace normalisation actually removes, in rows.

    This step is applied unconditionally to every row, and on its own
    appears to do nothing: the row count before and after is identical,
    26,872 both times, as the manifest records. It is reasonable to ask
    what a cleaning step that removes no rows is for.

    It is for the step that runs immediately after it. Collapsing runs of
    whitespace changes 551 texts, and those changes cause 81 additional
    (instruction, intent) pairs to be recognised as exact duplicates -
    2,318 removed with normalisation against 2,237 without. Those 81 pairs
    are sentences that differ only by a double space; without this step
    they would remain two distinct rows that can land on opposite sides of
    the split and leak.

    So the justification is not general convention, but that the step
    removes 81 leaking pairs, a number this function exists to compute
    rather than assert in a comment.
    """
    normalised = drop_empty_rows(normalise_whitespace(df_raw))
    changed = int((df_raw["instruction"].astype(str)
                   != normalised["instruction"].reindex(df_raw.index)).sum())
    without = len(df_raw) - len(df_raw.drop_duplicates(subset=["instruction", "intent"]))
    with_normalisation = len(normalised) - len(drop_exact_duplicates(normalised))
    return {
        "rows_in": len(df_raw),
        "texts_changed": changed,
        "empty_rows_dropped": len(df_raw) - len(normalised),
        "exact_duplicates_found_without_normalisation": int(without),
        "exact_duplicates_found_with_normalisation": int(with_normalisation),
        "extra_pairs_merged": int(with_normalisation - without),
    }


# =========================================================================
# 15. THE UNCOLLAPSED ALTERNATIVE
# =========================================================================

def train_side_rows(full_corpus: pd.DataFrame, part: str = "train",
                    include_exact_duplicates: bool = False) -> pd.DataFrame:
    """Every member of every family on one side of the split, not just the representative.

    The committed design keeps one row per template family, dropping 42%
    of the corpus. A second design removes exactly as much leakage while
    discarding nothing: keep every row, and assign whole families to a
    side. full_corpus.csv already records this, so the alternative
    training set needs no new split and no new hash - it is a different
    view of the frozen one.

    The leakage guarantee is identical, and for the same reason: if a
    held-out representative were >=0.90 similar to any member of a
    train-side family under the same intent, the two would be connected
    and would therefore be one family, which cannot straddle the split.
    Verified on the committed files - same-intent nearest neighbour
    0.000%, largest observed 0.8999, verbatim overlap 0.

    So the collapse is not what makes the split clean; the family
    assignment is. The collapse is a separate decision about class
    weighting, and this function is what lets its cost be measured instead
    of assumed.

    `include_exact_duplicates=False` by default. Exact duplicates carry no
    text the model has not already seen, so including them would only
    reweight the training set; excluding them keeps the comparison about
    family membership rather than repeated rows.
    """
    if part not in SPLIT_PARTS:
        raise ValueError(f"part must be one of {SPLIT_PARTS}, got {part!r}")
    rows = full_corpus[full_corpus["split"] == part]
    if not include_exact_duplicates:
        rows = rows[~rows["is_exact_duplicate"]]
    return rows.reset_index(drop=True).copy()
