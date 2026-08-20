#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from cam_deception.cache import load_saved_residual_streams, optionally_flip_cached_labels
from cam_deception.config import ExperimentConfig, PathsConfig, seed_everything
from cam_deception.pipeline import prepare_cam_banks, run_hamming_experiment, save_hamming_results


def parse_args():
    p = argparse.ArgumentParser(description="Train weighted or unweighted binary Hamming CAM-WTA classifiers.")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/llama70b_L33"))
    p.add_argument("--method", choices=["sign", "learned_matrix", "both"], default="both")
    p.add_argument("--unweighted", action="store_true", help="Match the Hamming_Unweighted notebook")
    p.add_argument("--fixed-epochs", type=int, default=150)
    p.add_argument("--learned-epochs", type=int, default=30)
    p.add_argument("--train-on", choices=["means", "tokens"], default="means")
    p.add_argument("--flip-cached-domain", action="append", default=[])
    p.add_argument("--save-learned-hash-matrix", action="store_true", help="Save the full learned d×d matrix (very large for 8192-d residuals)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig(paths=PathsConfig(output_dir=args.output_dir))
    cfg.activations.train_on = args.train_on
    cfg.hamming.weighted_rows = not args.unweighted
    cfg.hamming.fixed_epochs = args.fixed_epochs
    cfg.hamming.learned_epochs = args.learned_epochs
    cfg.hamming.methods = ("sign", "learned_matrix") if args.method == "both" else (args.method,)
    cfg.hamming.save_learned_hash_matrices = args.save_learned_hash_matrix
    seed_everything(cfg.split.random_seed)

    acts = load_saved_residual_streams(args.cache)
    acts = optionally_flip_cached_labels(acts, set(args.flip_cached_domain))
    prepared = prepare_cam_banks(acts, cfg)
    artifacts, metrics = run_hamming_experiment(acts, prepared, cfg)
    saved = save_hamming_results(artifacts, metrics, prepared, cfg)
    print(metrics.sort_values(["method", "train_scope", "eval_scope", "split"]).to_string(index=False))
    print("Saved:")
    for path in saved:
        print(" -", path)


if __name__ == "__main__":
    main()
