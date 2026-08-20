from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from .cam_rows import extract_cam_banks
from .dataset_view import make_train_data_for_scope
from .hamming import score_dataset_batched as score_hamming_batched
from .hamming import train_hamming_wta
from .metrics import binary_metrics
from .plotting import save_auroc_heatmap
from .projected_lsq import score_dataset_batched as score_projected_batched
from .projected_lsq import train_projected_wta
from .splits import build_shared_example_split, mask_for_split, split_indices_from_example_ids, split_label_counts


def prepare_cam_banks(acts, config):
    """Build one shared split, then extract combined + domain-specific INLP CAM rows from train only."""
    split_ids = build_shared_example_split(
        acts,
        test_fraction=config.split.test_fraction,
        val_fraction_of_train=config.split.val_fraction_of_train,
        random_seed=config.split.random_seed,
    )
    train_mask = mask_for_split(acts["example_index_mean"], "train", split_ids)
    X_train = np.asarray(acts["X_mean"][train_mask], dtype=np.float32)
    y_train = np.asarray(acts["y_mean"][train_mask], dtype=np.int64)
    domains_train = np.asarray(acts["domain_mean"][train_mask]).astype(str)

    rows, meta, deployment_rows, deployment_meta = extract_cam_banks(
        X_train,
        y_train,
        domains_train,
        n_single=config.probe.n_directions_single_domain,
        n_combined=config.probe.n_directions_combined,
        alpha=config.probe.logreg_alpha,
        use_scaler=config.probe.use_standard_scaler,
        remove_general_before_single=config.probe.remove_general_before_single_domain,
        positive_label_name=config.probe.positive_label_name,
    )
    return {
        "split_ids": split_ids,
        "all_scope_rows": rows,
        "all_scope_meta": meta,
        "deployment_rows": deployment_rows,
        "deployment_meta": deployment_meta,
    }


def _scopes_from_banks(all_scope_rows):
    return ["combined"] + sorted(k for k, rows in all_scope_rows.items() if k != "combined" and len(rows))


def _evaluate_projected(clf, X, y, threshold, config):
    scores = score_projected_batched(
        clf,
        X,
        mode=config.projected_wta.eval_distance_mode,
        batch_size=config.projected_wta.eval_batch_size,
    )
    pred = (scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    acc = accuracy_score(y, pred)
    try:
        auc = roc_auc_score(y, scores)
    except Exception:
        auc = np.nan
    return {
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "accuracy": float(acc), "auroc": float(auc),
        "eval_distance_mode": config.projected_wta.eval_distance_mode,
        "lsq_bit_precision": int(config.projected_wta.lsq_bit_precision),
        "projection_dim": int(config.projected_wta.projection_dim),
        "projection_method": config.projected_wta.projection_method,
        "normalize_after_projection": bool(config.projected_wta.normalize_after_projection),
    }


def run_projected_experiment(acts, prepared, config, train_global_bank: bool = True):
    all_scope_rows = prepared["all_scope_rows"]
    all_scope_meta = prepared["all_scope_meta"]
    split_ids = prepared["split_ids"]
    scopes = _scopes_from_banks(all_scope_rows)
    results, metrics_rows = {}, []

    for train_scope in scopes:
        rows = all_scope_rows[train_scope]
        Xs, ys, exs = make_train_data_for_scope(
            acts, train_scope, split_ids, train_on=config.activations.train_on, return_example_ids=True
        )
        local_splits = split_indices_from_example_ids(exs, split_ids)
        if len(np.unique(ys)) < 2 or len(np.unique(ys[local_splits["train"]])) < 2:
            print(f"Skipping {train_scope}: insufficient class coverage")
            continue
        print(f"Training projected WTA: {train_scope}; rows={rows.shape}; X={Xs.shape}; counts={split_label_counts(ys, local_splits)}")
        clf, artifact = train_projected_wta(
            rows,
            Xs,
            ys,
            config=config.projected_wta,
            split_config=config.split,
            split_indices=local_splits,
            example_indices=exs,
        )
        results[train_scope] = artifact
        for metric in artifact["metrics"]:
            m = dict(metric)
            m.update({
                "train_scope": train_scope,
                "eval_scope": train_scope,
                "n_rows": rows.shape[0],
                "train_on": config.activations.train_on,
            })
            metrics_rows.append(m)

        threshold = artifact["threshold"]
        for eval_scope in scopes:
            if eval_scope == train_scope:
                continue
            X_eval, y_eval = make_train_data_for_scope(
                acts, eval_scope, split_ids, train_on=config.activations.train_on, split="test"
            )
            if len(y_eval) == 0 or len(np.unique(y_eval)) < 2:
                continue
            m = _evaluate_projected(clf, X_eval, y_eval, threshold, config)
            m.update({
                "split": "cross_eval", "source_split": "shared_test", "n": len(y_eval),
                "threshold": threshold, "train_scope": train_scope, "eval_scope": eval_scope,
                "n_rows": rows.shape[0], "train_on": config.activations.train_on,
            })
            metrics_rows.append(m)

    if train_global_bank and len(prepared["deployment_rows"]):
        Xg, yg, exg = make_train_data_for_scope(
            acts, "combined", split_ids, train_on=config.activations.train_on, return_example_ids=True
        )
        global_splits = split_indices_from_example_ids(exg, split_ids)
        _, artifact = train_projected_wta(
            prepared["deployment_rows"], Xg, yg,
            config=config.projected_wta, split_config=config.split,
            split_indices=global_splits, example_indices=exg,
        )
        results["global_all_rows"] = artifact

    return results, pd.DataFrame(metrics_rows)


def run_hamming_experiment(acts, prepared, config):
    all_scope_rows = prepared["all_scope_rows"]
    split_ids = prepared["split_ids"]
    scopes = _scopes_from_banks(all_scope_rows)
    artifacts = {method: {} for method in config.hamming.methods}
    metric_rows = []

    for method in config.hamming.methods:
        for train_scope in scopes:
            rows = all_scope_rows[train_scope]
            Xs, ys, exs = make_train_data_for_scope(
                acts, train_scope, split_ids, train_on=config.activations.train_on, return_example_ids=True
            )
            local_splits = split_indices_from_example_ids(exs, split_ids)
            if len(np.unique(ys[local_splits["train"]])) < 2:
                continue
            print(f"Training Hamming WTA: method={method}; weighted={config.hamming.weighted_rows}; scope={train_scope}")
            model, artifact = train_hamming_wta(rows, Xs, ys, local_splits, method, config.hamming)
            artifacts[method][train_scope] = artifact

            for metric in artifact["metrics"]:
                m = dict(metric)
                m.update({
                    "method": method,
                    "weighted_rows": config.hamming.weighted_rows,
                    "train_scope": train_scope,
                    "eval_scope": train_scope,
                    "n_rows": rows.shape[0],
                    "train_on": config.activations.train_on,
                })
                metric_rows.append(m)

            threshold = artifact["threshold"]
            for eval_scope in scopes:
                if eval_scope == train_scope:
                    continue
                X_eval, y_eval = make_train_data_for_scope(
                    acts, eval_scope, split_ids, train_on=config.activations.train_on, split="test"
                )
                if len(y_eval) == 0 or len(np.unique(y_eval)) < 2:
                    continue
                scores = score_hamming_batched(model, X_eval, method, config.hamming)
                m = binary_metrics(y_eval, scores, threshold)
                m.update({
                    "split": "cross_eval", "source_split": "shared_test",
                    "method": method, "weighted_rows": config.hamming.weighted_rows,
                    "train_scope": train_scope, "eval_scope": eval_scope,
                    "n_rows": rows.shape[0], "train_on": config.activations.train_on,
                })
                metric_rows.append(m)
    return artifacts, pd.DataFrame(metric_rows)


def save_projected_results(results, metrics_df, prepared, config):
    out_dir = config.paths.artifact_dir
    saved = []
    for scope, artifact in results.items():
        if scope == "global_all_rows":
            rows = prepared["deployment_rows"]
            meta = prepared["deployment_meta"]
            filename_scope = "GLOBAL_all_rows"
        else:
            rows = prepared["all_scope_rows"][scope]
            meta = prepared["all_scope_meta"][scope]
            filename_scope = scope
        projection_weight = artifact["projection_weight"]
        if projection_weight is None:
            projection_weight = np.empty((0, 0), dtype=np.float32)
        path = out_dir / (
            f"cam_wta_{filename_scope}_llama70b_L{config.model.layer_index}_{config.model.hidden_state_mode}"
            f"_proj{artifact['projection_dim']}_lsq{config.projected_wta.lsq_bit_precision}bit_{config.projected_wta.eval_distance_mode}.npz"
        )
        np.savez_compressed(
            path,
            cam_rows=np.asarray(rows, dtype=np.float32),
            projection_weight=np.asarray(projection_weight, dtype=np.float32),
            rows_projected_fp32=artifact["rows_projected_fp32"],
            rows_projected_quantized=artifact["rows_projected_quantized"],
            euclidean_baked_rows=artifact["euclidean_baked_rows"],
            euclidean_constant=np.array([artifact["euclidean_constant"]], dtype=np.float32),
            alphas=artifact["alphas"], bias=np.array([artifact["bias"]], dtype=np.float32),
            threshold=np.array([artifact["threshold"]], dtype=np.float32),
            row_metadata_json=np.array([json.dumps(meta)], dtype=object),
            metrics_json=np.array([json.dumps(artifact["metrics"])], dtype=object),
            history_json=np.array([json.dumps(artifact["history"])], dtype=object),
            model_id=np.array([config.model.model_id], dtype=object),
            layer_index=np.array([config.model.layer_index], dtype=np.int64),
            hidden_state_mode=np.array([config.model.hidden_state_mode], dtype=object),
        )
        saved.append(path)
    metrics_path = out_dir / "projected_wta_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    heldout = metrics_df[metrics_df["split"].isin(["test", "cross_eval"])].copy()
    if len(heldout):
        save_auroc_heatmap(heldout, "Cross-Domain Projected LSQ WTA AUROC", out_dir / "projected_wta_auroc_heatmap.png")
    return saved + [metrics_path]


def save_hamming_results(artifacts, metrics_df, prepared, config):
    mode = "weighted" if config.hamming.weighted_rows else "unweighted"
    out_dir = config.paths.hamming_artifact_dir / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for method, scope_artifacts in artifacts.items():
        for scope, artifact in scope_artifacts.items():
            path = out_dir / f"cam_hamming_{method}_{scope}_L{config.model.layer_index}_{config.model.hidden_state_mode}.npz"
            payload = {
                "cam_rows": np.asarray(prepared["all_scope_rows"][scope], dtype=np.float32),
                "cam_binary_codes": np.asarray(artifact["cam_binary_codes"], dtype=np.int8),
                "alphas": np.asarray(artifact["alphas"], dtype=np.float32),
                "bias": np.array([artifact["bias"]], dtype=np.float32),
                "threshold": np.array([artifact["threshold"]], dtype=np.float32),
                "method": np.array([method], dtype=object),
                "weighted_rows": np.array([config.hamming.weighted_rows], dtype=bool),
                "history_json": np.array([json.dumps(artifact["history"])], dtype=object),
                "metrics_json": np.array([json.dumps(artifact["metrics"])], dtype=object),
            }
            # Avoid multi-GB artifacts unless explicitly requested.
            if method == "learned_matrix" and config.hamming.save_learned_hash_matrices and "hash_matrix" in artifact:
                payload["hash_matrix"] = np.asarray(artifact["hash_matrix"], dtype=np.float32)
            np.savez_compressed(path, **payload)
            saved.append(path)

    metrics_path = out_dir / "hamming_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    heldout = metrics_df[metrics_df["split"].isin(["test", "cross_eval"])].copy()
    for method in config.hamming.methods:
        method_df = heldout[heldout["method"] == method]
        if len(method_df):
            save_auroc_heatmap(
                method_df,
                f"Cross-Domain Hamming WTA AUROC — {method} ({mode})",
                out_dir / f"{method}_auroc_heatmap.png",
            )
    return saved + [metrics_path]
