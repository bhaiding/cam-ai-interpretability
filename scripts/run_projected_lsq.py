#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from cam_deception.cache import load_saved_residual_streams, optionally_flip_cached_labels
from cam_deception.config import ExperimentConfig, PathsConfig, seed_everything
from cam_deception.pipeline import prepare_cam_banks, run_projected_experiment, save_projected_results


def parse_args():
    p = argparse.ArgumentParser(description="Train projected LSQ/QAT CAM-WTA classifiers from cached residual streams.")
    p.add_argument("--cache", type=Path, required=True, help="mmap cache directory, .npz, .pt, or .pth")
    p.add_argument("--output-dir", type=Path, default=Path("results/llama70b_L33"))
    p.add_argument("--projection-dim", type=int, default=128)
    p.add_argument("--bits", type=int, default=3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--train-on", choices=["means", "tokens"], default="means")
    p.add_argument("--eval-mode", choices=["euclidean", "dot"], default="euclidean")
    p.add_argument("--no-projection", action="store_true")
    p.add_argument("--no-quantization", action="store_true")
    p.add_argument("--no-global-bank", action="store_true")
    p.add_argument("--flip-cached-domain", action="append", default=[], help="Explicitly flip labels for an old cache")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig(paths=PathsConfig(output_dir=args.output_dir))
    cfg.activations.train_on = args.train_on
    cfg.projected_wta.projection_dim = args.projection_dim
    cfg.projected_wta.lsq_bit_precision = args.bits
    cfg.projected_wta.epochs = args.epochs
    cfg.projected_wta.eval_distance_mode = args.eval_mode
    cfg.projected_wta.use_projection = not args.no_projection
    cfg.projected_wta.use_lsq_qat = not args.no_quantization
    seed_everything(cfg.split.random_seed)

    acts = load_saved_residual_streams(args.cache)
    acts = optionally_flip_cached_labels(acts, set(args.flip_cached_domain))
    prepared = prepare_cam_banks(acts, cfg)
    results, metrics = run_projected_experiment(acts, prepared, cfg, train_global_bank=not args.no_global_bank)
    saved = save_projected_results(results, metrics, prepared, cfg)
    print(metrics.sort_values(["train_scope", "eval_scope", "split"]).to_string(index=False))
    print("Saved:")
    for path in saved:
        print(" -", path)


if __name__ == "__main__":
    main()
