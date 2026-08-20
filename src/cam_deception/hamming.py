from __future__ import annotations

import math
from contextlib import nullcontext
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HammingConfig
from .metrics import binary_metrics, safe_auroc, select_validation_threshold


def binary_sign_np(x):
    return np.where(np.asarray(x) >= 0, 1, -1).astype(np.int8)


def hamming_similarity_np(query_codes, row_codes):
    q = np.asarray(query_codes, dtype=np.float32)
    r = np.asarray(row_codes, dtype=np.float32)
    return (q @ r.T) / q.shape[1]


def hard_binary_sign(z):
    return torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))


def straight_through_binary_sign(z, temperature: float = 1.0, eps: float = 1e-6):
    hard = hard_binary_sign(z)
    surrogate_scale = z.detach().abs().mean(dim=-1, keepdim=True).clamp_min(eps)
    soft = torch.tanh(z / (temperature * surrogate_scale))
    return soft + (hard - soft).detach()


class HammingWTAClassifier(nn.Module):
    """Full-dimensional binary CAM classifier with optional learned row weights."""

    def __init__(self, cam_rows, method: str = "sign", weighted_rows: bool = True, ste_temperature: float = 1.0):
        super().__init__()
        if method not in {"sign", "learned_matrix"}:
            raise ValueError(method)
        rows = torch.as_tensor(np.asarray(cam_rows, dtype=np.float32))
        self.register_buffer("cam_rows", rows)
        self.method = method
        self.weighted_rows = bool(weighted_rows)
        self.ste_temperature = float(ste_temperature)
        self.d_model = int(rows.shape[1])
        self.n_rows = int(rows.shape[0])

        if method == "learned_matrix":
            self.hash_matrix = nn.Linear(self.d_model, self.d_model, bias=False)
            with torch.no_grad():
                self.hash_matrix.weight.zero_()
                self.hash_matrix.weight.diagonal().fill_(1.0)
        else:
            self.hash_matrix = None

        if self.weighted_rows:
            initial_raw_alpha = math.log(math.expm1(1.0))
            self.raw_alphas = nn.Parameter(torch.full((self.n_rows,), initial_raw_alpha))
        else:
            self.register_parameter("raw_alphas", None)
        self.bias = nn.Parameter(torch.zeros(()))

    @property
    def alphas(self):
        if self.raw_alphas is None:
            return torch.ones(self.n_rows, dtype=self.bias.dtype, device=self.bias.device)
        return F.softplus(self.raw_alphas) + 1e-6

    def transform(self, z):
        return z if self.hash_matrix is None else self.hash_matrix(z)

    def encode(self, z):
        transformed = self.transform(z)
        if self.method == "learned_matrix" and self.training:
            return straight_through_binary_sign(transformed, temperature=self.ste_temperature)
        return hard_binary_sign(transformed)

    def forward(self, x):
        query_codes = self.encode(x)
        row_codes = self.encode(self.cam_rows)
        hamming_similarity = (query_codes.float() @ row_codes.float().T) / self.d_model
        normalized_hamming_distance = 0.5 * (1.0 - hamming_similarity)
        row_scores = hamming_similarity * self.alphas.unsqueeze(0)
        logits, winners = row_scores.max(dim=1)
        return logits + self.bias, winners, {
            "normalized_hamming": normalized_hamming_distance,
            "query_codes": query_codes,
            "row_codes": row_codes,
        }

    def identity_penalty(self):
        if self.hash_matrix is None:
            return self.bias.new_zeros(())
        W = self.hash_matrix.weight
        d = self.d_model
        return (W.square().sum() - 2.0 * W.diagonal().sum() + d) / (d * d)

    @torch.no_grad()
    def export_small(self):
        was_training = self.training
        self.eval()
        out = {
            "method": self.method,
            "weighted_rows": self.weighted_rows,
            "alphas": self.alphas.detach().cpu().numpy().astype(np.float32),
            "bias": float(self.bias.detach().cpu()),
            "cam_binary_codes": self.encode(self.cam_rows).cpu().numpy().astype(np.int8),
            "d_model": self.d_model,
        }
        self.train(was_training)
        return out


def _batch_size(config: HammingConfig, method: str, evaluation: bool = False):
    if method == "sign":
        return config.fixed_eval_batch_size if evaluation else config.fixed_batch_size
    return config.learned_eval_batch_size if evaluation else config.learned_batch_size


def _autocast_context(config: HammingConfig, method: str):
    if config.device == "cuda" and config.use_bf16 and method == "learned_matrix":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def score_dataset_batched(model, X, method: str, config: HammingConfig, return_winners=False):
    model.eval()
    scores, winners = [], []
    batch_size = _batch_size(config, method, evaluation=True)
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.as_tensor(
                np.asarray(X[start:start + batch_size], dtype=np.float32),
                dtype=torch.float32,
                device=config.device,
            )
            with _autocast_context(config, method):
                logits, batch_winners, _ = model(xb)
            scores.append(logits.float().cpu().numpy())
            if return_winners:
                winners.append(batch_winners.cpu().numpy())
    score_array = np.concatenate(scores) if scores else np.array([], dtype=np.float32)
    if return_winners:
        winner_array = np.concatenate(winners) if winners else np.array([], dtype=np.int64)
        return score_array, winner_array
    return score_array


def train_hamming_wta(cam_rows, X, y, split_indices, method: str, config: HammingConfig | None = None):
    config = deepcopy(config) if config is not None else HammingConfig()
    y = np.asarray(y, dtype=np.int64)
    train_idx = np.asarray(split_indices["train"], dtype=np.int64)
    val_idx = np.asarray(split_indices["val"], dtype=np.int64)
    test_idx = np.asarray(split_indices["test"], dtype=np.int64)
    if len(np.unique(y[train_idx])) < 2:
        raise ValueError("Training split must contain both labels.")

    model = HammingWTAClassifier(
        cam_rows,
        method=method,
        weighted_rows=config.weighted_rows,
        ste_temperature=config.ste_temperature,
    ).to(config.device)

    readout_params = [model.bias]
    if model.raw_alphas is not None:
        readout_params.insert(0, model.raw_alphas)

    if method == "learned_matrix":
        optimizer = torch.optim.AdamW([
            {"params": readout_params, "lr": config.readout_lr, "weight_decay": config.readout_weight_decay},
            {"params": model.hash_matrix.parameters(), "lr": config.matrix_lr, "weight_decay": config.matrix_weight_decay},
        ])
        epochs, eval_every = config.learned_epochs, config.learned_eval_every
    else:
        optimizer = torch.optim.AdamW(readout_params, lr=config.readout_lr, weight_decay=config.readout_weight_decay)
        epochs, eval_every = config.fixed_epochs, config.fixed_eval_every

    y_train = np.asarray(y[train_idx], dtype=np.float32)
    positive_count = max(float(y_train.sum()), 1.0)
    negative_count = max(float(len(y_train) - y_train.sum()), 1.0)
    pos_weight = torch.tensor(negative_count / positive_count, dtype=torch.float32, device=config.device)

    best_state, best_val_auc = None, -np.inf
    evaluations_without_improvement = 0
    history = []
    rng = np.random.default_rng(config.random_seed)

    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(len(train_idx))
        epoch_losses = []
        batch_size = _batch_size(config, method, evaluation=False)
        identity_penalty = torch.tensor(0.0, device=config.device)
        balance_penalty = torch.tensor(0.0, device=config.device)

        for start in range(0, len(permutation), batch_size):
            local = permutation[start:start + batch_size]
            batch_idx = train_idx[local]
            xb = torch.as_tensor(np.asarray(X[batch_idx], dtype=np.float32), device=config.device)
            yb = torch.as_tensor(y_train[local], dtype=torch.float32, device=config.device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(config, method):
                logits, _, extra = model(xb)
                bce = F.binary_cross_entropy_with_logits(logits.float(), yb, pos_weight=pos_weight)
                if method == "learned_matrix":
                    identity_penalty = model.identity_penalty().float()
                    balance_penalty = extra["query_codes"].float().mean(dim=0).square().mean()
                    loss = bce + config.identity_reg * identity_penalty + config.bit_balance_reg * balance_penalty
                else:
                    loss = bce
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        should_eval = ((epoch + 1) % eval_every == 0) or (epoch == epochs - 1)
        if not should_eval:
            continue
        val_scores = score_dataset_batched(model, X[val_idx], method, config)
        val_auc = safe_auroc(y[val_idx], val_scores)
        history.append({
            "epoch": int(epoch + 1),
            "loss": float(np.mean(epoch_losses)),
            "val_auroc": float(val_auc),
            "identity_penalty": float(identity_penalty.detach().cpu()),
            "bit_balance_penalty": float(balance_penalty.detach().cpu()),
        })
        if not np.isnan(val_auc) and val_auc > best_val_auc + 1e-5:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            evaluations_without_improvement = 0
        else:
            evaluations_without_improvement += 1
        if evaluations_without_improvement >= config.early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    val_scores = score_dataset_batched(model, X[val_idx], method, config)
    threshold = select_validation_threshold(y[val_idx], val_scores, config.threshold_selection_metric)
    test_scores = score_dataset_batched(model, X[test_idx], method, config)

    val_metrics = binary_metrics(y[val_idx], val_scores, threshold)
    val_metrics.update({"split": "val", "source_split": "shared_val"})
    test_metrics = binary_metrics(y[test_idx], test_scores, threshold)
    test_metrics.update({"split": "test", "source_split": "shared_test"})
    artifact = model.export_small()
    if method == "learned_matrix" and config.save_learned_hash_matrices:
        artifact["hash_matrix"] = model.hash_matrix.weight.detach().float().cpu().numpy().astype(np.float32)
    artifact.update({
        "threshold": float(threshold),
        "history": history,
        "metrics": [val_metrics, test_metrics],
        "best_val_auroc": float(best_val_auc),
    })
    return model, artifact
