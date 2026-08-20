from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _split_stratify_labels(df: pd.DataFrame) -> np.ndarray:
    return (df["domain"].astype(str) + "__label_" + df["label"].astype(str)).to_numpy()


def _safe_stratify_or_none(labels, test_size, split_name: str):
    labels = np.asarray(labels)
    counts = pd.Series(labels).value_counts()
    if len(counts) < 2:
        print(f"[{split_name}] Only one stratification class; using an unstratified split.")
        return None
    n_test = int(np.ceil(len(labels) * test_size)) if isinstance(test_size, float) else int(test_size)
    n_train = len(labels) - n_test
    if counts.min() < 2 or n_test < len(counts) or n_train < len(counts):
        print(f"[{split_name}] Not enough examples per domain/label stratum; using an unstratified split.")
        return None
    return labels


def build_shared_example_split(
    acts,
    test_fraction: float = 0.20,
    val_fraction_of_train: float = 0.20,
    random_seed: int = 0,
):
    """Create the single example-level split reused by INLP and both WTA classifiers."""
    ex_df = pd.DataFrame({
        "ex_idx": acts["example_index_mean"],
        "domain": acts["domain_mean"],
        "label": acts["y_mean"],
    }).drop_duplicates("ex_idx").reset_index(drop=True)

    all_positions = np.arange(len(ex_df))
    strat_labels = _split_stratify_labels(ex_df)
    trainval_pos, test_pos = train_test_split(
        all_positions,
        test_size=test_fraction,
        random_state=random_seed,
        stratify=_safe_stratify_or_none(strat_labels, test_fraction, "shared train/test"),
    )

    trainval_df = ex_df.iloc[trainval_pos].reset_index(drop=True)
    trainval_positions = np.arange(len(trainval_df))
    trainval_strat = _split_stratify_labels(trainval_df)
    train_pos_local, val_pos_local = train_test_split(
        trainval_positions,
        test_size=val_fraction_of_train,
        random_state=random_seed,
        stratify=_safe_stratify_or_none(trainval_strat, val_fraction_of_train, "shared train/val"),
    )

    split_ids = {
        "train": np.sort(trainval_df.iloc[train_pos_local]["ex_idx"].to_numpy()),
        "val": np.sort(trainval_df.iloc[val_pos_local]["ex_idx"].to_numpy()),
        "test": np.sort(ex_df.iloc[test_pos]["ex_idx"].to_numpy()),
    }
    split_sets = {k: set(map(int, v)) for k, v in split_ids.items()}
    assert split_sets["train"].isdisjoint(split_sets["val"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["val"].isdisjoint(split_sets["test"])
    return split_ids


def mask_for_split(example_indices, split_name: str, split_ids) -> np.ndarray:
    if split_name not in split_ids:
        raise ValueError(f"Unknown split: {split_name}")
    return np.isin(example_indices, split_ids[split_name])


def split_indices_from_example_ids(example_ids, split_ids):
    example_ids = np.asarray(example_ids, dtype=np.int64)
    return {
        split_name: np.flatnonzero(np.isin(example_ids, ids)).astype(np.int64)
        for split_name, ids in split_ids.items()
    }


def split_label_counts(y, split_indices):
    out = {}
    for split_name, idxs in split_indices.items():
        labels, counts = np.unique(y[idxs], return_counts=True) if len(idxs) else ([], [])
        out[split_name] = {int(k): int(v) for k, v in zip(labels, counts)}
    return out
