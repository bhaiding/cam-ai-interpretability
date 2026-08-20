from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from .cam_rows import l2_normalize_rows
from .config import ProjectedWTAConfig, SplitConfig


def grad_scale(x, scale):
    return (x - x * scale).detach() + x * scale


def round_pass(x):
    return (x.round() - x).detach() + x


class LSQQuantizer(nn.Module):
    """Minimal LSQ-style fake quantizer used by the uploaded projected-WTA notebook."""

    def __init__(
        self,
        bit_precision: int = 3,
        signed: bool = True,
        init_scale: float | None = None,
        eps: float = 1e-8,
        reference_dim: int = 128,
    ):
        super().__init__()
        self.bit_precision = int(bit_precision)
        self.signed = bool(signed)
        self.eps = float(eps)
        if self.signed:
            self.qmin = -(2 ** (self.bit_precision - 1))
            self.qmax = (2 ** (self.bit_precision - 1)) - 1
        else:
            self.qmin = 0
            self.qmax = (2 ** self.bit_precision) - 1
        if init_scale is None:
            denom = max(abs(self.qmin), abs(self.qmax), 1)
            init_scale = 1.0 / (math.sqrt(max(reference_dim, 1)) * denom)
        init_scale = float(max(init_scale, eps))
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale), dtype=torch.float32))

    @property
    def scale(self):
        return self.log_scale.exp().clamp_min(self.eps)

    def forward(self, x):
        if self.bit_precision >= 32:
            return x
        scale = self.scale
        grad_factor = 1.0 / math.sqrt(max(x.numel() * max(abs(self.qmax), 1), 1))
        scale = grad_scale(scale, grad_factor)
        q = round_pass((x / scale).clamp(self.qmin, self.qmax))
        return q * scale

    def quantize_np(self, x_np: np.ndarray) -> np.ndarray:
        scale = float(self.scale.detach().cpu())
        return (np.clip(np.round(x_np / scale), self.qmin, self.qmax) * scale).astype(np.float32)

    def metadata(self):
        return {
            "bit_precision": self.bit_precision,
            "signed": self.signed,
            "qmin": int(self.qmin),
            "qmax": int(self.qmax),
            "scale": float(self.scale.detach().cpu()),
        }


def _fix_projection_signs(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=np.float32).copy()
    for i in range(W.shape[0]):
        j = int(np.argmax(np.abs(W[i])))
        if W[i, j] < 0:
            W[i] *= -1.0
    return W


def build_dot_product_preserving_projection(
    rows_np: np.ndarray,
    d_proj: int = 128,
    seed: int = 0,
    rcond: float = 1e-6,
    pad_with_random_orthogonal: bool = True,
):
    """Build an unsupervised basis for the CAM-row span.

    If d_proj covers the row-bank rank, dot products x·row are preserved exactly
    (up to floating-point error) after projecting both x and row with this basis.
    """
    rows = l2_normalize_rows(np.asarray(rows_np, dtype=np.float32))
    if rows.ndim != 2 or rows.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2D CAM rows, got shape={rows.shape}")
    n_rows, d_model = rows.shape
    d_proj = int(min(max(d_proj, 1), d_model))
    _, s, vt = np.linalg.svd(rows, full_matrices=False)
    if len(s) == 0:
        raise ValueError("Could not compute a projection from an empty row bank.")
    tol = float(max(s[0] * rcond, np.finfo(np.float32).eps))
    rank = int(np.sum(s > tol))
    keep = int(min(max(rank, 1), d_proj, vt.shape[0]))
    basis = _fix_projection_signs(vt[:keep])

    if basis.shape[0] < d_proj and pad_with_random_orthogonal:
        rng = np.random.default_rng(seed)
        pad = rng.normal(size=(d_proj - basis.shape[0], d_model)).astype(np.float32)
        q, _ = np.linalg.qr(np.vstack([basis, pad]).T, mode="reduced")
        W = _fix_projection_signs(q.T[:d_proj].astype(np.float32))
    else:
        W = basis.astype(np.float32)

    rows_proj = rows @ W.T
    max_row_gram_error = float(np.max(np.abs(rows @ rows.T - rows_proj @ rows_proj.T)))
    metadata = {
        "method": "cam_row_subspace",
        "d_model": int(d_model),
        "requested_projection_dim": int(d_proj),
        "actual_projection_dim": int(W.shape[0]),
        "n_rows": int(n_rows),
        "row_bank_rank": int(rank),
        "rank_tolerance": float(tol),
        "exact_when_projection_dim_covers_rank": bool(d_proj >= rank),
        "max_row_gram_error": max_row_gram_error,
        "uses_labels": False,
    }
    return W.astype(np.float32), metadata


class WTAProjectedLSQClassifier(nn.Module):
    """Fixed CAM rows + optional projection + LSQ fake quantization + weighted WTA."""

    def __init__(self, rows_np: np.ndarray, config: ProjectedWTAConfig | None = None):
        super().__init__()
        self.config = deepcopy(config) if config is not None else ProjectedWTAConfig()
        rows = torch.tensor(l2_normalize_rows(np.asarray(rows_np, dtype=np.float32)), dtype=torch.float32)
        self.register_buffer("rows", rows)
        self.d_model = int(rows.shape[1])
        self.use_projection = bool(self.config.use_projection)
        self.d_proj = int(self.config.projection_dim if self.use_projection else self.d_model)
        self.use_lsq_qat = bool(self.config.use_lsq_qat)
        self.projection_method = self.config.projection_method if self.use_projection else "none"
        self.normalize_after_projection = bool(self.config.normalize_after_projection)
        self.projection_metadata = {"method": self.projection_method, "uses_labels": False}
        self.proj = None
        self.register_buffer("fixed_projection_weight", None)

        if self.use_projection:
            if self.projection_method == "cam_row_subspace":
                W_np, meta = build_dot_product_preserving_projection(
                    rows_np,
                    d_proj=self.d_proj,
                    seed=self.config.random_seed,
                    rcond=self.config.projection_svd_rcond,
                    pad_with_random_orthogonal=self.config.projection_pad_random_orthogonal,
                )
                self.d_proj = int(W_np.shape[0])
                self.register_buffer("fixed_projection_weight", torch.tensor(W_np, dtype=torch.float32))
                self.projection_metadata = meta
            elif self.projection_method in {"learned_supervised", "learned"}:
                self.proj = nn.Linear(self.d_model, self.d_proj, bias=False)
                nn.init.orthogonal_(self.proj.weight)
                self.projection_metadata = {"method": self.projection_method, "uses_labels": True}
            else:
                raise ValueError(f"Unknown projection method: {self.projection_method}")

        self.act_quant = LSQQuantizer(
            bit_precision=self.config.lsq_bit_precision,
            signed=self.config.lsq_signed,
            init_scale=self.config.lsq_initial_scale,
            eps=self.config.lsq_eps,
            reference_dim=self.d_proj,
        )
        self.row_quant = LSQQuantizer(
            bit_precision=self.config.lsq_bit_precision,
            signed=self.config.lsq_signed,
            init_scale=self.config.lsq_initial_scale,
            eps=self.config.lsq_eps,
            reference_dim=self.d_proj,
        )
        self.raw_alpha = nn.Parameter(torch.zeros(rows.shape[0], dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def get_alphas(self):
        return F.softplus(self.raw_alpha)

    def project(self, x):
        x = x.float()
        if not self.use_projection:
            return x
        if self.fixed_projection_weight is not None:
            return F.linear(x, self.fixed_projection_weight)
        if self.proj is not None:
            return self.proj(x)
        raise RuntimeError("Projection is enabled but no projection operator is initialized.")

    def project_maybe_normalize(self, x):
        z = self.project(x)
        return F.normalize(z, dim=-1) if self.normalize_after_projection else z

    def encode_query(self, x, quantize: bool | None = None):
        quantize = self.use_lsq_qat if quantize is None else quantize
        z = self.project_maybe_normalize(x)
        return self.act_quant(z) if quantize else z

    def encode_rows(self, quantize: bool | None = None):
        quantize = self.use_lsq_qat if quantize is None else quantize
        r = self.project_maybe_normalize(self.rows)
        return self.row_quant(r) if quantize else r

    def forward_dot(self, x, quantize: bool | None = None):
        q = self.encode_query(x, quantize=quantize)
        r = self.encode_rows(quantize=quantize)
        weighted = (q @ r.T) * self.get_alphas()[None, :]
        max_score, winner = weighted.max(dim=1)
        return max_score + self.bias, winner, weighted

    def euclidean_baked_tensors(self, quantize: bool | None = None, eps: float = 1e-8):
        r = self.encode_rows(quantize=quantize)
        weighted_rows = r * self.get_alphas()[:, None]
        row_norm_sq = (weighted_rows ** 2).sum(dim=1)
        C = row_norm_sq.max().detach() + eps
        extra = torch.sqrt(torch.clamp(C - row_norm_sq, min=0.0))
        return torch.cat([weighted_rows, extra[:, None]], dim=1), C

    def forward_euclidean(self, x, quantize: bool | None = None):
        q = self.encode_query(x, quantize=quantize)
        baked_rows, C = self.euclidean_baked_tensors(quantize=quantize)
        weighted_rows = baked_rows[:, :-1]
        q_norm_sq = (q ** 2).sum(dim=1, keepdim=True)
        dist_sq = q_norm_sq + C - 2.0 * (q @ weighted_rows.T)
        min_dist_sq, winner = dist_sq.min(dim=1)
        score = -0.5 * (min_dist_sq - q_norm_sq.squeeze(1) - C) + self.bias
        return score, winner, dist_sq

    def forward(self, x):
        if self.config.eval_distance_mode == "euclidean":
            return self.forward_euclidean(x, quantize=self.use_lsq_qat)
        return self.forward_dot(x, quantize=self.use_lsq_qat)

    @torch.no_grad()
    def export_numpy(self):
        was_training = self.training
        self.eval()
        rows_fp32 = self.encode_rows(quantize=False).cpu().numpy().astype(np.float32)
        rows_q = self.encode_rows(quantize=True).cpu().numpy().astype(np.float32)
        baked_rows, C = self.euclidean_baked_tensors(quantize=True)
        projection_weight = None
        if self.fixed_projection_weight is not None:
            projection_weight = self.fixed_projection_weight.cpu().numpy().astype(np.float32)
        elif self.proj is not None:
            projection_weight = self.proj.weight.cpu().numpy().astype(np.float32)
        out = {
            "projection_weight": projection_weight,
            "rows_projected_fp32": rows_fp32,
            "rows_projected_quantized": rows_q,
            "euclidean_baked_rows": baked_rows.cpu().numpy().astype(np.float32),
            "euclidean_constant": float(C.cpu()),
            "alphas": self.get_alphas().cpu().numpy().astype(np.float32),
            "bias": float(self.bias.cpu()),
            "act_quant": self.act_quant.metadata(),
            "row_quant": self.row_quant.metadata(),
            "use_projection": self.use_projection,
            "projection_dim": self.d_proj,
            "projection_method": self.projection_method,
            "projection_metadata": dict(self.projection_metadata),
            "normalize_after_projection": self.normalize_after_projection,
            "use_lsq_qat": self.use_lsq_qat,
        }
        self.train(was_training)
        return out


def _get_scores_for_eval(clf, X_tensor, mode: str):
    if mode == "euclidean":
        return clf.forward_euclidean(X_tensor, quantize=clf.use_lsq_qat)
    if mode == "dot":
        return clf.forward_dot(X_tensor, quantize=clf.use_lsq_qat)
    raise ValueError(f"Unknown eval mode: {mode}")


def score_dataset_batched(clf, X, mode: str | None = None, batch_size: int | None = None, return_winners=False):
    mode = mode or clf.config.eval_distance_mode
    batch_size = batch_size or clf.config.eval_batch_size
    device = next(clf.parameters()).device
    clf.eval()
    scores, winners = [], []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.as_tensor(np.asarray(X[start:start + batch_size], dtype=np.float32), device=device)
            logits, batch_winners, _ = _get_scores_for_eval(clf, xb, mode)
            scores.append(logits.detach().cpu().numpy())
            if return_winners:
                winners.append(batch_winners.detach().cpu().numpy())
    score_arr = np.concatenate(scores) if scores else np.array([], dtype=np.float32)
    if return_winners:
        winner_arr = np.concatenate(winners) if winners else np.array([], dtype=np.int64)
        return score_arr, winner_arr
    return score_arr


def _evaluate_split(y, split_idx, scores, threshold: float, config: ProjectedWTAConfig, name: str, split_source: str):
    pred = (scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y[split_idx], pred, average="binary", zero_division=0)
    acc = accuracy_score(y[split_idx], pred)
    try:
        auc = roc_auc_score(y[split_idx], scores)
    except Exception:
        auc = np.nan
    return {
        "split": name,
        "n": int(len(split_idx)),
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "accuracy": float(acc), "auroc": float(auc), "threshold": float(threshold),
        "eval_distance_mode": config.eval_distance_mode,
        "lsq_bit_precision": int(config.lsq_bit_precision),
        "projection_dim": int(config.projection_dim),
        "projection_method": config.projection_method,
        "normalize_after_projection": bool(config.normalize_after_projection),
        "split_source": split_source,
    }


def train_projected_wta(
    rows,
    X,
    y,
    config: ProjectedWTAConfig | None = None,
    split_config: SplitConfig | None = None,
    split_indices=None,
    example_indices=None,
):
    """Train the projected LSQ/QAT WTA readout using the shared split when supplied."""
    config = deepcopy(config) if config is not None else ProjectedWTAConfig()
    split_config = deepcopy(split_config) if split_config is not None else SplitConfig(random_seed=config.random_seed)
    y = np.asarray(y, dtype=np.int64)
    if len(np.unique(y)) != 2:
        raise ValueError("Need both classes")

    if split_indices is None:
        idx = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            idx, test_size=split_config.test_fraction, random_state=split_config.random_seed, stratify=y
        )
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=split_config.val_fraction_of_train,
            random_state=split_config.random_seed,
            stratify=y[train_idx],
        )
        split_source = "internal_fallback_split"
    else:
        train_idx = np.asarray(split_indices["train"], dtype=np.int64)
        val_idx = np.asarray(split_indices["val"], dtype=np.int64)
        test_idx = np.asarray(split_indices["test"], dtype=np.int64)
        split_source = "shared_example_split"

    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError(f"Empty split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    if len(np.unique(y[train_idx])) < 2:
        raise ValueError("WTA train split has only one class for this scope.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = WTAProjectedLSQClassifier(rows, config=config).to(device)
    optimizer = torch.optim.AdamW(clf.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    y_train = np.asarray(y[train_idx], dtype=np.float32)
    X_val = X[val_idx]
    y_val = y[val_idx]
    best_state, best_val_auc = None, -np.inf
    history = []

    for epoch in range(config.epochs):
        perm = np.random.default_rng(config.random_seed + epoch).permutation(len(train_idx))
        losses = []
        clf.train()
        for start in range(0, len(perm), config.batch_size):
            local = perm[start:start + config.batch_size]
            batch_idx = train_idx[local]
            xb = torch.as_tensor(np.asarray(X[batch_idx], dtype=np.float32), device=device)
            yb = torch.as_tensor(y_train[local], dtype=torch.float32, device=device)
            logits, _, _ = clf.forward_dot(xb, quantize=config.use_lsq_qat)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_scores = score_dataset_batched(clf, X_val)
        try:
            val_auc = float(roc_auc_score(y_val, val_scores))
        except Exception:
            val_auc = float("nan")
        history.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_auc": val_auc,
            "act_scale": float(clf.act_quant.scale.detach().cpu()),
            "row_scale": float(clf.row_quant.scale.detach().cpu()),
        })
        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}

    if best_state is not None:
        clf.load_state_dict(best_state)
    clf.eval()
    val_scores, _ = score_dataset_batched(clf, X[val_idx], return_winners=True)
    test_scores, _ = score_dataset_batched(clf, X[test_idx], return_winners=True)

    candidates = np.quantile(val_scores, np.linspace(0.01, 0.99, 99))
    best_th, best_f1 = 0.0, -1.0
    for th in candidates:
        pred = (val_scores >= th).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y[val_idx], pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, float(th)

    metrics = [
        _evaluate_split(y, val_idx, val_scores, best_th, config, "val", split_source),
        _evaluate_split(y, test_idx, test_scores, best_th, config, "test", split_source),
    ]
    export = clf.export_numpy()
    artifact = {
        "rows": l2_normalize_rows(np.asarray(rows, dtype=np.float32)),
        "alphas": export["alphas"], "bias": export["bias"], "threshold": best_th,
        "history": history, "metrics": metrics,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "split_source": split_source,
        **{k: export[k] for k in [
            "projection_weight", "rows_projected_fp32", "rows_projected_quantized",
            "euclidean_baked_rows", "euclidean_constant", "act_quant", "row_quant",
            "use_projection", "projection_dim", "use_lsq_qat", "projection_method",
            "projection_metadata", "normalize_after_projection",
        ]},
        "lsq_bit_precision": int(config.lsq_bit_precision),
        "eval_distance_mode": config.eval_distance_mode,
    }
    if example_indices is not None:
        example_indices = np.asarray(example_indices, dtype=np.int64)
        artifact["train_example_ids"] = np.unique(example_indices[train_idx])
        artifact["val_example_ids"] = np.unique(example_indices[val_idx])
        artifact["test_example_ids"] = np.unique(example_indices[test_idx])
    return clf, artifact
