from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm.auto import tqdm


def activation_cache_dir(cache_path: Path) -> Path:
    cache_path = Path(cache_path)
    return cache_path.with_suffix("").with_name(cache_path.stem + "_mmap")


def load_saved_residual_streams(path: Path):
    """Load cached residual streams without loading the LLM."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Saved residual cache not found at {path}")

    if path.is_dir():
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            names = list(manifest.get("arrays", {}).keys())
        else:
            names = [p.stem for p in sorted(path.glob("*.npy"))]
        if not names:
            raise ValueError(f"No .npy arrays found in residual cache directory: {path}")
        out = {}
        for name in names:
            arr_path = path / f"{name}.npy"
            if not arr_path.exists():
                raise FileNotFoundError(arr_path)
            mmap_mode = "r" if name.startswith("X_") else None
            out[name] = np.load(arr_path, allow_pickle=False, mmap_mode=mmap_mode)
        source_type = "mmap directory"
    elif path.suffix.lower() == ".npz":
        obj = np.load(path, allow_pickle=True)
        out = {k: obj[k] for k in obj.files}
        source_type = "npz"
    elif path.suffix.lower() in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict):
            raise TypeError("A .pt/.pth residual cache must contain a dictionary of arrays.")
        out = {
            k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v))
            for k, v in obj.items()
        }
        source_type = "torch dictionary"
    else:
        raise ValueError(f"Unsupported residual cache format: {path}")

    required = {"X_mean", "y_mean", "domain_mean", "example_index_mean"}
    missing = sorted(required.difference(out))
    if missing:
        raise KeyError(f"Saved residual cache is missing required arrays: {missing}")

    print(f"Loaded saved residual streams from {source_type}: {path}")
    print({k: tuple(v.shape) for k, v in out.items() if hasattr(v, "shape")})
    return out


def optionally_flip_cached_labels(acts_obj, domains_to_flip):
    """Flip cached labels only for caches known to predate a label-orientation fix."""
    if not domains_to_flip:
        return acts_obj
    out = dict(acts_obj)
    for suffix in ["mean", "token"]:
        y_key = f"y_{suffix}"
        d_key = f"domain_{suffix}"
        if y_key not in out or d_key not in out:
            continue
        labels = np.asarray(out[y_key], dtype=np.int64).copy()
        domains = np.asarray(out[d_key]).astype(str)
        mask = np.isin(domains, sorted(domains_to_flip))
        labels[mask] = 1 - labels[mask]
        out[y_key] = labels
        print(f"Flipped {int(mask.sum())} cached {suffix} labels for {sorted(domains_to_flip)}")
    return out


def load_mmap_activation_cache(cache_dir: Path):
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    out = {}
    for name in manifest["arrays"]:
        arr_path = cache_dir / f"{name}.npy"
        mmap = "r" if name.startswith("X_") else None
        out[name] = np.load(arr_path, allow_pickle=False, mmap_mode=mmap)
    return out


def write_mmap_activation_cache_from_shards(
    shard_paths: Iterable[Path],
    cache_dir: Path,
    cache_dtype=np.float16,
    extract_token_activations: bool = False,
):
    """Materialize mmap-backed .npy arrays from extraction shards without concatenating in RAM."""
    shard_paths = [Path(p) for p in shard_paths]
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not shard_paths:
        raise RuntimeError("No activation shards were written.")

    n_mean = n_tok = 0
    d_model = None
    max_domain_len = 1
    for sp in shard_paths:
        z = np.load(sp, allow_pickle=False)
        n_mean += z["X_mean"].shape[0]
        n_tok += z["X_token"].shape[0]
        if d_model is None and z["X_mean"].ndim == 2 and z["X_mean"].shape[1] > 0:
            d_model = z["X_mean"].shape[1]
        for key in ["domain_mean", "domain_token"]:
            if key in z and len(z[key]):
                max_domain_len = max(max_domain_len, max(len(str(x)) for x in z[key]))
        z.close()
    if d_model is None:
        raise RuntimeError("Could not infer d_model from activation shards.")

    cache_dtype = np.dtype(cache_dtype)
    domain_dtype = f"<U{max_domain_len}"
    arrays = {
        "X_mean": np.lib.format.open_memmap(cache_dir / "X_mean.npy", mode="w+", dtype=cache_dtype, shape=(n_mean, d_model)),
        "y_mean": np.lib.format.open_memmap(cache_dir / "y_mean.npy", mode="w+", dtype=np.int64, shape=(n_mean,)),
        "domain_mean": np.lib.format.open_memmap(cache_dir / "domain_mean.npy", mode="w+", dtype=domain_dtype, shape=(n_mean,)),
        "example_index_mean": np.lib.format.open_memmap(cache_dir / "example_index_mean.npy", mode="w+", dtype=np.int64, shape=(n_mean,)),
        "X_token": np.lib.format.open_memmap(cache_dir / "X_token.npy", mode="w+", dtype=cache_dtype, shape=(n_tok, d_model)),
        "y_token": np.lib.format.open_memmap(cache_dir / "y_token.npy", mode="w+", dtype=np.int64, shape=(n_tok,)),
        "domain_token": np.lib.format.open_memmap(cache_dir / "domain_token.npy", mode="w+", dtype=domain_dtype, shape=(n_tok,)),
        "example_index_token": np.lib.format.open_memmap(cache_dir / "example_index_token.npy", mode="w+", dtype=np.int64, shape=(n_tok,)),
        "token_position": np.lib.format.open_memmap(cache_dir / "token_position.npy", mode="w+", dtype=np.int64, shape=(n_tok,)),
    }

    mean_pos = tok_pos = 0
    for sp in tqdm(shard_paths, desc="Writing mmap activation cache"):
        z = np.load(sp, allow_pickle=False)
        m, t = z["X_mean"].shape[0], z["X_token"].shape[0]
        if m:
            arrays["X_mean"][mean_pos:mean_pos + m] = z["X_mean"].astype(cache_dtype, copy=False)
            arrays["y_mean"][mean_pos:mean_pos + m] = z["y_mean"]
            arrays["domain_mean"][mean_pos:mean_pos + m] = z["domain_mean"].astype(domain_dtype, copy=False)
            arrays["example_index_mean"][mean_pos:mean_pos + m] = z["example_index_mean"]
        if t:
            arrays["X_token"][tok_pos:tok_pos + t] = z["X_token"].astype(cache_dtype, copy=False)
            arrays["y_token"][tok_pos:tok_pos + t] = z["y_token"]
            arrays["domain_token"][tok_pos:tok_pos + t] = z["domain_token"].astype(domain_dtype, copy=False)
            arrays["example_index_token"][tok_pos:tok_pos + t] = z["example_index_token"]
            arrays["token_position"][tok_pos:tok_pos + t] = z["token_position"]
        mean_pos += m
        tok_pos += t
        z.close()

    for key in list(arrays):
        arrays[key].flush()
        del arrays[key]

    manifest = {
        "cache_dtype": str(cache_dtype),
        "d_model": int(d_model),
        "extract_token_activations": bool(extract_token_activations),
        "n_mean": int(n_mean),
        "n_token": int(n_tok),
        "arrays": {
            "X_mean": {"dtype": str(cache_dtype), "shape": [int(n_mean), int(d_model)]},
            "y_mean": {"dtype": "int64", "shape": [int(n_mean)]},
            "domain_mean": {"dtype": domain_dtype, "shape": [int(n_mean)]},
            "example_index_mean": {"dtype": "int64", "shape": [int(n_mean)]},
            "X_token": {"dtype": str(cache_dtype), "shape": [int(n_tok), int(d_model)]},
            "y_token": {"dtype": "int64", "shape": [int(n_tok)]},
            "domain_token": {"dtype": domain_dtype, "shape": [int(n_tok)]},
            "example_index_token": {"dtype": "int64", "shape": [int(n_tok)]},
            "token_position": {"dtype": "int64", "shape": [int(n_tok)]},
        },
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return cache_dir
