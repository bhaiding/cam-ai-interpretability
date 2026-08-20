#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

from cam_deception.activations import extract_or_load_activations, load_model_and_tokenizer, refresh_activation_labels_from_data
from cam_deception.config import ExperimentConfig, PathsConfig, seed_everything
from cam_deception.data import load_all_datasets, load_external_benchmarks


def parse_args():
    p = argparse.ArgumentParser(description="Build the Llama residual-stream cache used by the CAM experiments.")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--truth-spec-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("results/llama70b_L33"))
    p.add_argument("--clone-truth-spec", action="store_true")
    p.add_argument("--include-external", action="store_true", help="Append TruthfulQA and FEVER like the main notebook")
    p.add_argument("--max-examples-per-domain", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--train-on", choices=["means", "tokens"], default="means")
    p.add_argument("--force", action="store_true", help="Discard an existing activation cache")
    return p.parse_args()


def main():
    args = parse_args()
    paths = PathsConfig(
        project_root=args.project_root,
        truth_spec_dir=args.truth_spec_dir,
        output_dir=args.output_dir,
    )
    cfg = ExperimentConfig(paths=paths)
    cfg.max_examples_per_domain = args.max_examples_per_domain
    cfg.activations.batch_size = args.batch_size
    cfg.activations.train_on = args.train_on
    cfg.activations.force_reextract = args.force
    seed_everything(cfg.split.random_seed)

    if not cfg.paths.truth_spec_dir.exists() and args.clone_truth_spec:
        subprocess.run(
            ["git", "clone", "https://github.com/zfying/truth_spec.git", str(cfg.paths.truth_spec_dir)],
            check=True,
        )

    data, missing = load_all_datasets(
        cfg.domains,
        cfg.paths.truth_spec_dir,
        max_examples_per_domain=cfg.max_examples_per_domain,
        random_seed=cfg.split.random_seed,
    )
    if args.include_external:
        external = load_external_benchmarks(random_seed=cfg.split.random_seed)
        if len(external):
            data = pd.concat([data, external], ignore_index=True)
    if len(data) == 0:
        raise RuntimeError("No dataset rows were loaded. Use --clone-truth-spec or point --truth-spec-dir at the paper repository.")

    print(data.groupby(["domain", "label"]).size())
    if missing:
        print("Missing repo domains:", missing)

    model, tokenizer = load_model_and_tokenizer(cfg.model)
    cache_path = cfg.paths.cache_dir / (
        f"{cfg.model.model_id.replace('/', '__')}_L{cfg.model.layer_index}_{cfg.model.hidden_state_mode}_activations.npz"
    )
    acts = extract_or_load_activations(data, cache_path, model, tokenizer, cfg.model, cfg.activations)
    acts = refresh_activation_labels_from_data(acts, data)
    print("Cache arrays:", {k: tuple(v.shape) for k, v in acts.items() if hasattr(v, "shape")})
    print("Hamming/projected runners should use:", cache_path.with_suffix("").with_name(cache_path.stem + "_mmap"))


if __name__ == "__main__":
    main()
