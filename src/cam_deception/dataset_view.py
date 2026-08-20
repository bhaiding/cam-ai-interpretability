from __future__ import annotations

import numpy as np

from .cam_rows import get_scope_mask


def make_train_data_for_scope(
    acts_obj,
    scope: str,
    split_ids,
    train_on: str = "means",
    split: str | None = None,
    return_example_ids: bool = False,
):
    suffix = "mean" if train_on == "means" else "token" if train_on == "tokens" else None
    if suffix is None:
        raise ValueError(train_on)
    X = acts_obj[f"X_{suffix}"]
    y = acts_obj[f"y_{suffix}"]
    domains = acts_obj[f"domain_{suffix}"]
    example_ids = acts_obj[f"example_index_{suffix}"]

    mask = get_scope_mask(domains, scope)
    if split is not None:
        if split not in split_ids:
            raise ValueError(f"Unknown split: {split}")
        mask = mask & np.isin(example_ids, split_ids[split])

    Xs = X[mask]
    ys = np.asarray(y[mask], dtype=np.int64)
    exs = np.asarray(example_ids[mask], dtype=np.int64)
    if return_example_ids:
        return Xs, ys, exs
    return Xs, ys
