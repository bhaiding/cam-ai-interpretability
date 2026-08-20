from __future__ import annotations

import contextlib
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .cache import activation_cache_dir, load_mmap_activation_cache, write_mmap_activation_cache_from_shards


def load_model_and_tokenizer(model_config):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_id,
        torch_dtype=model_config.dtype,
        device_map=model_config.device_map,
        attn_implementation=model_config.attn_implementation,
        low_cpu_mem_usage=True,
    )
    model.config.output_hidden_states = False
    model.config.use_cache = False
    model.eval()
    return model, tokenizer


def build_chat_ids_and_assistant_span(tokenizer, user_prompt: str, assistant_text: str):
    """Return input IDs and the assistant-token span [start, end)."""
    user_msg = [{"role": "user", "content": user_prompt}]
    full_msg = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_text},
    ]
    prefix_text = tokenizer.apply_chat_template(user_msg, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(full_msg, tokenize=False, add_generation_prompt=False)
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    start, end = len(prefix_ids), len(full_ids)
    if end > start and full_ids[end - 1] == tokenizer.eos_token_id:
        end -= 1
    if start >= end:
        assistant_ids = tokenizer(assistant_text, add_special_tokens=False)["input_ids"]
        if not assistant_ids:
            raise ValueError("Empty assistant text after tokenization.")
        start, end = max(0, len(full_ids) - len(assistant_ids)), len(full_ids)
    return torch.tensor(full_ids, dtype=torch.long), start, end


def get_llama_layer_module(model, layer_index: int):
    base = getattr(model, "model", model)
    if hasattr(base, "layers"):
        return base.layers[layer_index]
    if hasattr(base, "decoder") and hasattr(base.decoder, "layers"):
        return base.decoder.layers[layer_index]
    raise AttributeError("Could not find transformer layers on this model.")


def get_model_input_device(model):
    if hasattr(model, "hf_device_map"):
        for key in ["model.embed_tokens", "model.tok_embeddings", "transformer.wte", "model"]:
            dev = model.hf_device_map.get(key)
            if dev is not None and dev not in {"cpu", "disk"}:
                return torch.device(dev)
    return next(model.parameters()).device


@contextlib.contextmanager
def capture_selected_layer_hidden_state(model, layer_index: int, hidden_state_mode: str):
    """Capture only one selected residual stream instead of all hidden states."""
    layer = get_llama_layer_module(model, layer_index)
    cache = {}

    def hook(module, inputs, output):
        if hidden_state_mode == "pre_layer":
            hidden = inputs[0]
        elif hidden_state_mode == "post_layer":
            hidden = output[0] if isinstance(output, (tuple, list)) else output
        else:
            raise ValueError(hidden_state_mode)
        cache["hidden"] = hidden.detach().to("cpu", dtype=torch.float32)

    handle = layer.register_forward_hook(hook)
    try:
        yield cache
    finally:
        handle.remove()


def refresh_activation_labels_from_data(acts_obj, data_df: pd.DataFrame):
    """Reuse cached activations while refreshing labels after normalization/label fixes."""
    if len(data_df) == 0:
        return acts_obj
    out = dict(acts_obj)
    for suffix in ["mean", "token"]:
        y_key, ex_key = f"y_{suffix}", f"example_index_{suffix}"
        if y_key not in out or ex_key not in out:
            continue
        ex = np.asarray(out[ex_key], dtype=np.int64)
        if len(ex) == 0:
            continue
        if ex.min() < 0 or ex.max() >= len(data_df):
            print(f"WARNING: cannot refresh {y_key}; cached example IDs do not match current data.")
            continue
        refreshed = data_df.iloc[ex]["label"].astype(np.int64).to_numpy()
        out[y_key] = refreshed
    return out


def extract_or_load_activations(
    data: pd.DataFrame,
    cache_path: Path,
    model,
    tokenizer,
    model_config,
    activation_config,
):
    """OOM-conscious layer activation extraction with shard + mmap caching."""
    cache_path = Path(cache_path)
    cache_dir = activation_cache_dir(cache_path)
    if activation_config.use_mmap and cache_dir.exists() and not activation_config.force_reextract:
        return load_mmap_activation_cache(cache_dir)
    if cache_path.exists() and not activation_config.force_reextract and not activation_config.use_mmap:
        obj = np.load(cache_path, allow_pickle=True)
        return {k: obj[k] for k in obj.files}

    if activation_config.force_reextract:
        if cache_path.exists():
            cache_path.unlink()
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    shard_dir = cache_path.with_suffix("").with_name(cache_path.stem + "_shards")
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = []

    X_mean, y_mean, domains, ex_indices = [], [], [], []
    X_tok, y_tok, tok_domains, tok_ex_indices, tok_positions = [], [], [], [], []
    n_extracted = 0
    shard_idx = 0
    cache_dtype = np.dtype(activation_config.cache_dtype)

    def flush_shard(force=False):
        nonlocal shard_idx, X_mean, y_mean, domains, ex_indices
        nonlocal X_tok, y_tok, tok_domains, tok_ex_indices, tok_positions
        if not X_mean or (not force and len(X_mean) < activation_config.shard_size):
            return
        X_mean_arr = np.stack(X_mean).astype(cache_dtype, copy=False)
        d_model = X_mean_arr.shape[1]
        X_tok_arr = np.stack(X_tok).astype(cache_dtype, copy=False) if X_tok else np.empty((0, d_model), dtype=cache_dtype)
        shard_path = shard_dir / f"activations_shard_{shard_idx:05d}.npz"
        np.savez(
            shard_path,
            X_mean=X_mean_arr,
            y_mean=np.array(y_mean, dtype=np.int64),
            domain_mean=np.array(domains, dtype=str),
            example_index_mean=np.array(ex_indices, dtype=np.int64),
            X_token=X_tok_arr,
            y_token=np.array(y_tok, dtype=np.int64),
            domain_token=np.array(tok_domains, dtype=str),
            example_index_token=np.array(tok_ex_indices, dtype=np.int64),
            token_position=np.array(tok_positions, dtype=np.int64),
        )
        shard_paths.append(shard_path)
        shard_idx += 1
        X_mean, y_mean, domains, ex_indices = [], [], [], []
        X_tok, y_tok, tok_domains, tok_ex_indices, tok_positions = [], [], [], [], []
        gc.collect()

    input_device = get_model_input_device(model)
    batch_size = activation_config.batch_size
    for start_idx in tqdm(range(0, len(data), batch_size), desc="Extracting activations"):
        batch_df = data.iloc[start_idx:start_idx + batch_size]
        batch_input_ids, batch_spans, valid_indices = [], [], []
        for i, row in batch_df.iterrows():
            try:
                ids, span_start, span_end = build_chat_ids_and_assistant_span(tokenizer, row.user_prompt, row.assistant_text)
                batch_input_ids.append(ids)
                batch_spans.append((span_start, span_end))
                valid_indices.append(i)
            except Exception as exc:
                print(f"Skipping row {i}: {exc}")
        if not batch_input_ids:
            continue

        max_len = max(len(ids) for ids in batch_input_ids)
        padded_ids = torch.full((len(batch_input_ids), max_len), tokenizer.pad_token_id, dtype=torch.long)
        attn_mask = torch.zeros((len(batch_input_ids), max_len), dtype=torch.long)
        for b_idx, ids in enumerate(batch_input_ids):
            padded_ids[b_idx, :len(ids)] = ids
            attn_mask[b_idx, :len(ids)] = 1

        try:
            with torch.inference_mode(), capture_selected_layer_hidden_state(
                model, model_config.layer_index, model_config.hidden_state_mode
            ) as layer_cache:
                _ = model(
                    input_ids=padded_ids.to(input_device),
                    attention_mask=attn_mask.to(input_device),
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            hidden_states = layer_cache["hidden"]
            for b_idx, (span_start, span_end) in enumerate(batch_spans):
                df_idx = valid_indices[b_idx]
                row = data.loc[df_idx]
                h = hidden_states[b_idx, span_start:span_end]
                if h.shape[0] == 0:
                    continue
                X_mean.append(h.mean(dim=0).numpy())
                y_mean.append(int(row.label))
                domains.append(str(row.domain))
                ex_indices.append(int(df_idx))
                if activation_config.extract_token_activations:
                    max_t = min(h.shape[0], activation_config.max_tokens_per_example)
                    for t in range(max_t):
                        X_tok.append(h[t].numpy())
                        y_tok.append(int(row.label))
                        tok_domains.append(str(row.domain))
                        tok_ex_indices.append(int(df_idx))
                        tok_positions.append(int(t))
                n_extracted += 1
            flush_shard(force=False)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise
            print(f"Skipping batch {start_idx}: {exc}")
        finally:
            gc.collect()
            batch_no = start_idx // max(1, batch_size)
            if torch.cuda.is_available() and batch_no % activation_config.clear_cuda_cache_every_n_batches == 0:
                torch.cuda.empty_cache()

    flush_shard(force=True)
    if n_extracted == 0:
        raise RuntimeError("All rows failed extraction.")

    if activation_config.use_mmap:
        write_mmap_activation_cache_from_shards(
            shard_paths,
            cache_dir,
            cache_dtype=cache_dtype,
            extract_token_activations=activation_config.extract_token_activations,
        )
        shutil.rmtree(shard_dir, ignore_errors=True)
        return load_mmap_activation_cache(cache_dir)

    parts = [np.load(sp, allow_pickle=False) for sp in shard_paths]
    arrays = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0].files}
    for p in parts:
        p.close()
    np.savez(cache_path, **arrays)
    shutil.rmtree(shard_dir, ignore_errors=True)
    return arrays
