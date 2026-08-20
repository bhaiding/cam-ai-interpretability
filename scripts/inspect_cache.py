#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from cam_deception.cache import load_saved_residual_streams


def main():
    p = argparse.ArgumentParser(description="Inspect a saved residual-stream cache without loading an LLM.")
    p.add_argument("--cache", type=Path, required=True)
    args = p.parse_args()
    acts = load_saved_residual_streams(args.cache)
    print("Arrays:")
    for key, value in acts.items():
        if hasattr(value, "shape"):
            print(f"  {key:24s} shape={tuple(value.shape)} dtype={value.dtype}")
    df = pd.DataFrame({"domain": np.asarray(acts["domain_mean"]).astype(str), "label": np.asarray(acts["y_mean"])})
    print("\nMean-activation label counts:")
    print(df.groupby(["domain", "label"]).size().to_string())


if __name__ == "__main__":
    main()
