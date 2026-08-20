from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def l2_normalize_rows(X, eps: float = 1e-8):
    X = np.asarray(X, dtype=np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def l2_normalize_vec(w, eps: float = 1e-8):
    w = np.asarray(w, dtype=np.float32)
    return w / (np.linalg.norm(w) + eps)


def project_out_directions(X, directions):
    """Project activations onto the nullspace of the supplied directions."""
    if directions is None or len(directions) == 0:
        return X
    D = l2_normalize_rows(np.asarray(directions, dtype=np.float32))
    Q = []
    for d in D:
        v = d.copy()
        for q in Q:
            v = v - np.dot(v, q) * q
        n = np.linalg.norm(v)
        if n > 1e-6:
            Q.append(v / n)
    if not Q:
        return X
    Q = np.stack(Q)
    return X - (X @ Q.T) @ Q


def fit_logreg_direction(
    X,
    y,
    alpha: float = 1e-4,
    use_scaler: bool = False,
):
    """Fit a logistic probe and return a normalized direction in raw activation coordinates."""
    if len(np.unique(y)) < 2:
        raise ValueError("Need both classes to fit a direction.")

    if use_scaler:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_fit = scaler.fit_transform(X)
    else:
        scaler = None
        X_fit = X

    clf = LogisticRegression(
        C=1.0 / alpha,
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_fit, y)
    w_scaled = clf.coef_[0].astype(np.float32)
    b_scaled = float(clf.intercept_[0])

    if scaler is not None:
        scale = scaler.scale_.astype(np.float32)
        mean = scaler.mean_.astype(np.float32)
        safe_scale = np.maximum(scale, 1e-12)
        w = w_scaled / safe_scale
        b = b_scaled - float(np.dot(w_scaled / safe_scale, mean))
        scaler_info = {"mean": mean, "scale": scale}
    else:
        w, b = w_scaled, b_scaled
        scaler_info = None

    return l2_normalize_vec(w), b, clf, scaler_info


def extract_inlp_directions(
    X,
    y,
    n_directions: int,
    name_prefix: str,
    alpha: float = 1e-4,
    use_scaler: bool = False,
    positive_label_name: str = "deceptive_or_false",
):
    """Repeatedly fit a probe, save its direction, then remove that direction from X."""
    X_work = np.asarray(X, dtype=np.float32).copy()
    rows, metadata = [], []
    for k in range(n_directions):
        try:
            w, b, _, _ = fit_logreg_direction(X_work, y, alpha=alpha, use_scaler=use_scaler)
        except Exception as exc:
            print(f"[{name_prefix}] stopping at k={k}: {exc}")
            break
        rows.append(w)
        metadata.append({
            "row_name": f"{name_prefix}__inlp_{k}",
            "scope": name_prefix,
            "k": k,
            "bias_from_probe": b,
            "alpha": alpha,
            "positive_label": positive_label_name,
            "split_source": "shared_train_examples",
        })
        X_work = project_out_directions(X_work, [w])

    if not rows:
        return np.empty((0, np.asarray(X).shape[1]), dtype=np.float32), metadata
    return np.stack(rows).astype(np.float32), metadata


def get_scope_mask(domains_array, scope: str):
    if scope == "combined":
        return np.ones(len(domains_array), dtype=bool)
    return np.asarray(domains_array).astype(str) == scope


def extract_cam_banks(
    X_train,
    y_train,
    domains_train,
    n_single: int = 4,
    n_combined: int = 10,
    alpha: float = 1e-4,
    use_scaler: bool = False,
    remove_general_before_single: bool = False,
    positive_label_name: str = "deceptive_or_false",
):
    """Extract the combined bank and one INLP bank per domain from shared-train examples."""
    combined_rows, combined_meta = extract_inlp_directions(
        X_train,
        y_train,
        n_directions=n_combined,
        name_prefix="combined",
        alpha=alpha,
        use_scaler=use_scaler,
        positive_label_name=positive_label_name,
    )
    all_scope_rows = {"combined": combined_rows}
    all_scope_meta = {"combined": combined_meta}

    for domain in sorted(set(np.asarray(domains_train).astype(str).tolist())):
        mask = get_scope_mask(domains_train, domain)
        Xd = np.asarray(X_train[mask], dtype=np.float32)
        yd = np.asarray(y_train[mask], dtype=np.int64)
        if remove_general_before_single and len(combined_rows):
            Xd = project_out_directions(Xd, combined_rows)
        rows, meta = extract_inlp_directions(
            Xd,
            yd,
            n_directions=n_single,
            name_prefix=domain,
            alpha=alpha,
            use_scaler=use_scaler,
            positive_label_name=positive_label_name,
        )
        all_scope_rows[domain] = rows
        all_scope_meta[domain] = meta

    deployment_rows = [all_scope_rows["combined"]]
    deployment_meta = list(all_scope_meta["combined"])
    for domain in sorted(k for k in all_scope_rows if k != "combined"):
        deployment_rows.append(all_scope_rows[domain])
        deployment_meta.extend(all_scope_meta[domain])
    nonempty = [r for r in deployment_rows if len(r)]
    if nonempty:
        deployment = l2_normalize_rows(np.concatenate(nonempty, axis=0).astype(np.float32))
    else:
        deployment = np.empty((0, np.asarray(X_train).shape[1]), dtype=np.float32)
    return all_scope_rows, all_scope_meta, deployment, deployment_meta


def score_with_rows(X, rows):
    Xn = l2_normalize_rows(np.asarray(X, dtype=np.float32))
    Rn = l2_normalize_rows(np.asarray(rows, dtype=np.float32))
    sims = Xn @ Rn.T
    return sims.max(axis=1), sims.argmax(axis=1)
