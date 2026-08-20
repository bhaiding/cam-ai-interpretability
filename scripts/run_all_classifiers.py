#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from cam_deception.cache import load_saved_residual_streams
from cam_deception.config import ExperimentConfig, PathsConfig, seed_everything
from cam_deception.pipeline import (
    prepare_cam_banks,
    run_hamming_experiment,
    run_projected_experiment,
    save_hamming_results,
    save_projected_results,
)


def main():
    parser = argparse.ArgumentParser(description="Run all three uploaded-notebook experiment families from one cache.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/llama70b_L33"))
    parser.add_argument("--skip-learned-hash", action="store_true", help="Avoid the very large d×d learned hash matrix")
    args = parser.parse_args()

    cfg = ExperimentConfig(paths=PathsConfig(output_dir=args.output_dir))
    seed_everything(cfg.split.random_seed)
    acts = load_saved_residual_streams(args.cache)
    prepared = prepare_cam_banks(acts, cfg)

    projected_results, projected_metrics = run_projected_experiment(acts, prepared, cfg)
    save_projected_results(projected_results, projected_metrics, prepared, cfg)

    methods = ("sign",) if args.skip_learned_hash else ("sign", "learned_matrix")
    for weighted in [True, False]:
        hcfg = deepcopy(cfg)
        hcfg.hamming.weighted_rows = weighted
        hcfg.hamming.methods = methods
        artifacts, metrics = run_hamming_experiment(acts, prepared, hcfg)
        save_hamming_results(artifacts, metrics, prepared, hcfg)


if __name__ == "__main__":
    main()
