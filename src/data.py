"""
Data preparation. Runs once at the start, on CPU. After that we barely touch it.

What goes in here:
- Download the Bitext dataset (or read it from disk if it is already there).
- Light text cleaning: collapse double spaces, drop empty rows. Nothing more.
- Find rows that are almost the same sentence. The dataset was generated
  automatically, so many rows are variations of one template. Rows like that
  must end up in the same split, otherwise the model is tested on sentences
  it has already seen.
- Split the data into three parts: train (learn from it), val (check while
  tuning), test (check only at the very end).
- Save a fingerprint of the split: the seed used, how many rows in each part.
  This is what lets the lecturer run the code and get exactly the same split,
  and therefore exactly the same numbers as the report.
- Keep a registry of EVERY raw row (full_corpus.csv): its template family, the
  split side that family landed on, and whether it is the row that represents
  the family in the classification files. Stage 1 never reads it; it is what
  lets stage 2 (RAG) build a retrieval corpus from train-side families only,
  so a sibling of a test sentence can never end up in the corpus.

Note: no print statements and no plots in this file. Only functions that take
a table and return a table. notebooks/01_data.ipynb is what runs them and
shows the results.

--------------------------------------------------------------------------
READING ORDER (the file is written to be read top to bottom)
  1. Constants          - every number that must never drift lives here.
  2. Loading            - get the raw CSV, verify it is the file we expect.
  3. Cleaning           - whitespace, empty rows, exact duplicates.
  4. Near-duplicates    - the interesting part: group sibling sentences.
  5. Splitting          - stratified 70/15/15, twice (clean and naive).
  6. Leakage measurement- how close is a test row to its nearest train row.
  7. Artifacts          - labels / intent->category / urgency, the contract.
  8. Manifest           - the identity card of the split.
  9. Token lengths      - instruction lengths in real model tokens.
  10. Taxonomy diagnostic - how much each intent label bundles distinct goals.
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
# Everything that could silently change a result is a named constant here,
# and nowhere else. The reason is not tidiness: if the near-duplicate
# threshold lives inside a notebook cell, someone nudges it from 0.90 to
# 0.88 "just to look", the split changes, and two training runs are no
# longer comparable - with no error message anywhere.

HF_REPO = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
RAW_FILENAME = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
DEFAULT_RAW_DIR = Path("data/raw")

# The three numbers the guide says to verify by hand before doing anything
# else. If the file upstream is updated these stop matching, and every number
# in the report would silently be about a different dataset.
EXPECTED_N_ROWS = 26_872
EXPECTED_N_INTENTS = 27
EXPECTED_N_CATEGORIES = 11

# The split seed. This one is frozen forever (decision C2).
# It is NOT the training seed - that one is a different constant, in a
# different file, and it is the one that varies across the three noise-floor
# runs. Two seeds with the same name is a classic way to report "training
# variance" that is really "different data".
SPLIT_SEED = 42

# 70 / 15 / 15. train_test_split cannot do three-way, so we split twice:
# first 70/30, then cut the 30 in half.
HOLDOUT_FRACTION = 0.30
VAL_SHARE_OF_HOLDOUT = 0.50

# Cosine-similarity threshold above which two sentences are called siblings
# of the same template. 0.90 is a choice, not a fact - notebooks/01_data.ipynb
# justifies it with a scan over {0.80, 0.85, 0.90, 0.95}.
NEAR_DUP_THRESHOLD = 0.90

# One vector space, defined once, used for three different jobs:
# clustering near-duplicates, measuring leakage in the naive split, and
# verifying the clean split. If clustering used one vectorizer and the
# leakage measurement used another, "cleaned at 0.90" and "0.9 similar"
# would be numbers in two different units and the claim would not be
# checkable. char_wb (character n-grams inside word boundaries) rather than
# words, because the dataset deliberately contains typos: "delivery" vs
# "deliverly" share almost every 3-gram but zero words.
TFIDF_PARAMS = dict(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,          # an n-gram seen once is noise, and it doubles the vocabulary
    sublinear_tf=True,  # 1+log(tf): a word repeated 5x is not 5x more important
)

# The columns that get written to the split CSVs.
# `response` is deliberately NOT here. It is the templated answer, and if it
# ever reaches the feature side the model reads the answer off the input and
# scores ~0.999. Leaving it out of the file makes that mistake impossible
# rather than merely discouraged.
# `flags` and `category` DO stay: they are never features either, but they are
# needed for error slicing and for category-level metrics.
SPLIT_COLUMNS = ["row_id", "instruction", "intent", "category", "flags", "dup_group"]

# The columns of data/processed/full_corpus.csv - the registry of EVERY raw row
# (decision C6). Not a training file: stage 1 never reads it. It exists so that
# stage 2 can build its retrieval corpus from train-side families only, and so
# that the ~2.3k exact-duplicate rows (whose responses are unique) keep a split
# assignment instead of silently falling outside the guarantee.
# `response` is deliberately absent here too: stage 2 joins it back from
# data/raw/ via row_id, so nothing under data/processed/ ever contains the
# answer text.
FULL_CORPUS_COLUMNS = ["row_id", "instruction", "intent", "category", "flags",
                       "dup_group", "split", "is_representative", "is_exact_duplicate"]

# The urgency table (decision A2). A business rule, not a learned task.
# Written once, by hand, and validated against the real 27 labels below.
URGENCY_PRIOR: dict[str, str] = {
    # high - the customer is blocked from acting, or their money is stuck
    "payment_issue": "high",
    "complaint": "high",
    "get_refund": "high",
    "track_refund": "high",
    "cancel_order": "high",
    "delete_account": "high",
    "recover_password": "high",
    "registration_problems": "high",
    "contact_human_agent": "high",
    # medium - time-sensitive, but there is a window to act
    "change_order": "medium",
    "track_order": "medium",
    "change_shipping_address": "medium",
    "check_invoice": "medium",
    "get_invoice": "medium",
    "check_refund_policy": "medium",
    "check_cancellation_fee": "medium",
    "edit_account": "medium",
    "switch_account": "medium",
    "contact_customer_service": "medium",
    "delivery_period": "medium",
    # low - an information request, or a proactive action
    "newsletter_subscription": "low",
    "review": "low",
    "create_account": "low",
    "place_order": "low",
    "set_up_shipping_address": "low",
    "delivery_options": "low",
    "check_payment_methods": "low",
}

URGENCY_FALLBACK = "medium"


# =========================================================================
# 2. LOADING
# =========================================================================

def load_raw(raw_dir: Path | str = DEFAULT_RAW_DIR, download_if_missing: bool = True) -> pd.DataFrame:
    """Return the raw Bitext table, downloading it once if it is not on disk.

    Why a local copy at all: Colab wipes the machine when the session dies.
    Re-downloading 19 MB every session is slow, and - worse - it silently
    picks up any upstream change. A local file plus the hash in the manifest
    turns "the dataset changed" from an invisible event into a failed assert.
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
        # copy rather than symlink: Windows/OneDrive does not do symlinks well
        local_path.write_bytes(Path(cached).read_bytes())

    return pd.read_csv(local_path, encoding="utf-8")


def verify_raw(df: pd.DataFrame) -> dict:
    """Check the three headline numbers, and raise if any of them moved.

    This is the "stop and report" check from the guide. It is deliberately an
    exception and not a warning: a warning scrolls past in a notebook, and
    then every figure in the report describes a dataset nobody looked at.
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
    """Hash of the raw CSV as it sits on disk. Goes into the manifest."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =========================================================================
# 3. CLEANING
# =========================================================================
# The rule for this section: whatever we do NOT do to a live incoming ticket,
# we must not do to the training data either. That is train/serving skew, and
# it is why there is no lowercasing, no punctuation stripping, no stopword
# removal and no stemming here. A question mark is a real signal for query
# intents, and the preposition is literally the only thing separating
# get_invoice from check_invoice.

_WHITESPACE = re.compile(r"\s+")


def normalise_whitespace(df: pd.DataFrame, column: str = "instruction") -> pd.DataFrame:
    """Collapse runs of whitespace to a single space and strip the ends.

    This is safe under the skew rule because it is also what we would do to a
    live ticket: a double space is a typing artefact, not a signal, and it
    would otherwise make two identical sentences look different to the exact
    duplicate check that runs next.
    """
    out = df.copy()
    out[column] = out[column].astype(str).str.replace(_WHITESPACE, " ", regex=True).str.strip()
    return out


def drop_empty_rows(df: pd.DataFrame, column: str = "instruction") -> pd.DataFrame:
    """Drop rows whose text is empty after normalisation."""
    return df[df[column].str.len() > 0].copy()


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per (instruction, intent) pair.

    The subset argument is the whole point. The default compares EVERY column,
    including `flags` and `response`; two rows with identical text but a
    different flag string both survive, and it looks like the data was cleaned
    when nothing was removed at all. On this dataset the default removes 0 rows
    and the correct subset removes 2,318 - the notebook prints both numbers
    side by side, because it is the cheapest demonstration of the trap.
    """
    return df.drop_duplicates(subset=["instruction", "intent"]).copy()


def find_label_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where the exact same sentence carries more than one intent.

    Not a bug to patch - a ceiling to measure. If the same sentence appears
    with two labels, no model can be right on both, so 100% is unreachable and
    the honest headline is "93.1 against a ceiling of 98.5" rather than "93.1".
    Returns an empty frame when there are none, which is itself a result.
    """
    per_text = df.groupby("instruction")["intent"].nunique()
    conflicted = per_text[per_text > 1].index
    return df[df["instruction"].isin(conflicted)].sort_values("instruction")


# =========================================================================
# 4. NEAR-DUPLICATES
# =========================================================================

def build_vector_space(texts: pd.Series | list[str]) -> tuple[TfidfVectorizer, csr_matrix]:
    """Fit the single TF-IDF space used by every similarity computation here.

    Fitted on ALL rows, before splitting, on purpose. This looks like it
    violates "fit on train only", so it is worth being able to answer:
    - "fit on train only" protects a MODEL from seeing test data. Nothing here
      is a model; no parameter of this vectorizer ever reaches the classifier.
    - Grouping siblings is data curation and has to happen before splitting.
      Clustering each split separately would leave each one internally clean
      and still leaking across the boundary.
    - Using one space for both clustering and measuring is what makes the
      claim checkable: "removed everything above 0.90" and "0.3% of test rows
      are above 0.90" are then statements in the same units.

    TF-IDF output is L2-normalised, so X @ X.T IS cosine similarity - no
    separate normalisation step, and no 27k x 27k dense matrix.
    """
    vec = TfidfVectorizer(**TFIDF_PARAMS)
    X = vec.fit_transform(texts)
    return vec, X


def near_duplicate_groups(X: csr_matrix, intents: pd.Series | np.ndarray,
                          threshold: float = NEAR_DUP_THRESHOLD) -> np.ndarray:
    """Assign every row a 'template family' id. Returns an int array, one per row.

    Three design points, all of which the examiner can ask about:

    1. Blocked by intent. All-pairs over 26,872 rows is 722M cells (~2.9 GB in
       float32) and kills a CPU Colab. Near-duplicates are always inside one
       intent, because they were expanded from the same template, so we do 27
       blocks of ~900 rows instead. Seconds, not minutes.
       (Cross-intent near-duplicates are a different phenomenon - semantic
       conflicts - and we deliberately keep them; see the notebook.)

    2. Character n-grams, from the shared space above.

    3. connected_components, not pairwise deletion. If A~B and B~C but A and C
       are not similar to each other, all three still came from one template.
       Delete in pairs and A and C survive on opposite sides of the split:
       cleaned, and still leaking. connected_components is the transitive
       closure of the "is similar to" relation, which is exactly the family.
    """
    intents = np.asarray(intents)
    groups = np.full(X.shape[0], -1, dtype=np.int64)
    next_id = 0

    # np.unique keeps the order deterministic across runs and machines, which
    # matters because these ids end up in a committed CSV.
    for intent in np.unique(intents):
        idx = np.flatnonzero(intents == intent)
        Xi = X[idx]

        # Sparse cosine similarity inside the block.
        S = (Xi @ Xi.T).tocoo()

        # Keep only the edges above the threshold. Building the adjacency
        # matrix from the surviving (row, col) pairs is cheaper and clearer
        # than thresholding the sparse matrix in place.
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
                   thresholds=(0.80, 0.85, 0.90, 0.95)) -> pd.DataFrame:
    """How many families survive at each threshold. This is what justifies 0.90.

    A flat curve around the chosen value means the choice is stable and one
    sentence closes it in the report. A steep curve means the threshold is a
    real sensitivity - which is a finding in its own right, not a problem.
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
    """Keep exactly one row per family. This is what makes the clean split clean.

    The representative is drawn at RANDOM (with a fixed seed), not taken as the
    first row of the family. The dataset is ordered by construction: the first
    member of a family tends to be the plain base variant, and the typo /
    keyword-only / colloquial variants come after it. Always taking the first
    would quietly build a clean set made of tidy grammatical sentences and
    strip out exactly the noisy rows the robustness analysis needs.
    """
    shuffled = df.sample(frac=1.0, random_state=seed)
    representatives = shuffled.drop_duplicates(subset=group_column)
    # Back to the original row order so the written CSV is deterministic and
    # readable next to the raw file.
    return representatives.sort_index().copy()


# =========================================================================
# 5. SPLITTING
# =========================================================================

def split_stratified(df: pd.DataFrame, seed: int = SPLIT_SEED,
                     holdout_fraction: float = HOLDOUT_FRACTION,
                     val_share: float = VAL_SHARE_OF_HOLDOUT,
                     stratify_column: str = "intent",
                     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70 / 15 / 15, stratified on intent, in the two steps sklearn forces on us.

    Stratified on `intent` (27) and never on `category` (11). Stratifying on
    the 11 categories does not guarantee that all 27 intents appear in val: a
    rare intent can end up absent, macro-F1 then averages 26 classes instead
    of 27, and nothing complains.
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
    """Every raw row, stamped with its family, its split side, and its role.

    Why this file exists (decision C6). The clean split keeps one row per
    family, which is right for CLASSIFICATION - train and test stay template-
    uniform, so the score measures generalisation across templates - and wrong
    as an archive: the other ~10k rows would simply be discarded, and they are
    exactly the raw material stage 2 (RAG) needs, since every row's `response`
    is unique. This function throws nothing away. Every row inherits the split
    side of its family's representative:

    - one family -> one representative -> exactly one side, so the
      inheritance is well-defined;
    - a whole family is always on ONE side, so a stage-2 corpus built as
      `split == "train"` can never contain a sibling of a test sentence.
      Contamination becomes impossible by construction instead of a
      convention someone has to remember.

    Exact duplicates (dropped before clustering) are re-attached through their
    (instruction, intent) key - unambiguous because the dataset has zero exact
    label conflicts - and marked is_exact_duplicate.

    The `split` column refers to the CLEAN split only; the naive control split
    keeps its own six CSVs and plays no role here.

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

    Chunked on purpose. The full product for the naive split is 3,684 x 17,187
    which is 63M float32 = 253 MB dense; in chunks of 500 it is 34 MB at a
    time. Same answer, no memory spike.

    `train_index` / `test_index` are POSITIONS in X, so the caller must pass
    row positions and not pandas labels - that is what `row_id` is for.
    """
    A = X[train_index]
    B = X[test_index]
    out = np.zeros(B.shape[0], dtype=np.float32)
    for start in range(0, B.shape[0], chunk_size):
        block = (B[start:start + chunk_size] @ A.T).toarray()
        out[start:start + chunk_size] = block.max(axis=1)
    return out


def leakage_summary(max_sim: np.ndarray, exact_overlap_fraction: float) -> dict:
    """The three numbers the guide asks for, plus the median for context."""
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
# 7. ARTIFACTS  (the contract - see section 6 of CLAUDE.md)
# =========================================================================

def build_labels(df: pd.DataFrame) -> list[str]:
    """The 27 intents, sorted. The one thing deliberately built from ALL rows.

    The model knows integers 0..26, not names; this list is the translation.
    It is a contract, not a learned parameter, so it is built once from every
    label in the dataset. If the order ever shifts between runs, yesterday's
    checkpoint returns confidently wrong class names with no error at all.
    """
    return sorted(df["intent"].unique().tolist())


def build_intent2cat(df: pd.DataFrame) -> dict[str, str]:
    """intent -> category, built from the file itself and never from the docs.

    The published dataset card disagrees with the file in at least two ways
    (it lists 10 categories, and it omits several intents), so the mapping is
    derived and then verified. A failure of the assert below would mean one
    intent appears under two categories - a finding to report, not something
    to patch quietly.
    """
    per_intent = df.groupby("intent")["category"].nunique()
    ambiguous = per_intent[per_intent > 1]
    if len(ambiguous) > 0:
        raise ValueError(
            f"These intents map to more than one category: {ambiguous.to_dict()}. "
            "This contradicts decision A1 (intent -> category is a function) and "
            "must be reported, not silently fixed."
        )
    return df.groupby("intent")["category"].first().to_dict()


def build_urgency_prior(labels: list[str]) -> dict[str, str]:
    """The hand-written urgency rule, validated against the real label list.

    Two checks: every one of the 27 intents has an entry (a missing key would
    silently fall back to 'medium' at routing time), and no entry refers to an
    intent that does not exist (a typo in the table).
    """
    missing = sorted(set(labels) - set(URGENCY_PRIOR))
    unknown = sorted(set(URGENCY_PRIOR) - set(labels))
    if missing or unknown:
        raise ValueError(f"urgency table mismatch - missing: {missing}, unknown: {unknown}")
    return {intent: URGENCY_PRIOR[intent] for intent in labels}


# =========================================================================
# 8. MANIFEST
# =========================================================================

def sha256_of_split(df: pd.DataFrame) -> str:
    """A fingerprint of one split file: its texts and labels, in file order.

    Text AND label, because a rebuild that shuffled the labels but kept the
    texts would otherwise pass. Row order is included, because order is part
    of the file - and it is deterministic given the seed.
    """
    payload = "\n".join(f"{t}\t{y}" for t, y in zip(df["instruction"], df["intent"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_of_full_corpus(df: pd.DataFrame) -> str:
    """Fingerprint of the full-corpus registry.

    Unlike sha256_of_split, this includes dup_group, split and
    is_representative: those assignments ARE the file's content. A rebuild
    that kept every text but moved one family to the other side must fail
    the manifest check, not pass it.
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
    """The identity card of the split. Small, committed to git, and load-bearing.

    The split CSVs themselves are gitignored - they are rebuilt from the seed.
    This file is the proof that a rebuild produced the same thing. Without it,
    an upstream dataset update or a scikit-learn major version bump moves the
    split and nobody notices; with it, the bootstrap assert fires immediately.

    When the full-corpus registry is passed, its fingerprint is included too:
    the family-to-side assignment is as much a part of the frozen split as the
    six CSVs are.
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
    """Re-hash the splits on disk and compare to the manifest. Raise on mismatch.

    This is the assert that makes the reproducibility claim self-verifying.
    It runs at the top of every notebook, which is the only place it is useful:
    a check that runs after the numbers are produced is decoration.

    `full_corpus` is optional so that a notebook which only reads the six split
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
    """Small helper so every JSON in the project is written the same way."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# =========================================================================
# 9. TOKEN LENGTHS
# =========================================================================

def token_lengths(texts: pd.Series | list[str], model_name: str = "Qwen/Qwen3-1.7B") -> np.ndarray:
    """Length of every text in TOKENS of the actual model tokenizer.

    In characters this question is unanswerable: max_length is counted in
    tokens, and training time grows roughly with the square of the sequence
    length. Measuring gives a real number (p99 here is 19) instead of the
    naive default of 512, which would be ~250x the attention cost for the same
    result.

    transformers is imported inside the function on purpose: this module is
    the CPU data module, and `import src.data` should not pull in a heavy
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
    """How much does each intent label bundle more than one user goal?

    Method, per intent: cluster the RESPONSES into two groups (the NLG engine
    wrote different reply templates for different user goals, so the response
    text is a cheap proxy for what the customer actually wanted), then train a
    small classifier to predict the response cluster from the INSTRUCTION
    alone, 3-fold cross-validated.

    Reading the `lift` column (accuracy minus the majority baseline):
      ~0.0 - the two response clusters are template noise; the instruction does
             not encode them, and the label is as fine as the data supports.
      high - the customer's own words reliably signal a distinction that the
             intent label throws away (subscribe vs unsubscribe, file-a-claim
             vs complain, upgrade-tier vs switch-user).

    This is a diagnostic, not training code: nothing it fits is kept, and
    nothing here touches the split. It is the measured basis for the stage-2
    rule "filter by intent, RANK BY TEXT": with high lift on most intents,
    mapping intent -> one canned reply would answer the wrong goal for a large
    minority of tickets.

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
