from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASET_SPECS, FLIP_LABEL_DOMAINS


def apply_domain_label_fixes(df: pd.DataFrame, domains_to_flip=FLIP_LABEL_DOMAINS) -> pd.DataFrame:
    """Apply the empirical-label orientation fix used by the uploaded notebook."""
    if len(df) == 0 or "domain" not in df or "label" not in df:
        return df
    out = df.copy()
    mask = out["domain"].isin(domains_to_flip)
    if mask.any():
        before = out.loc[mask, "label"].value_counts().sort_index().to_dict()
        out.loc[mask, "label"] = 1 - out.loc[mask, "label"].astype(int)
        after = out.loc[mask, "label"].value_counts().sort_index().to_dict()
        print(f"Applied label flip for domains {sorted(out.loc[mask, 'domain'].unique())}: before={before}, after={after}")
    return out


def find_dataset_files(root: Path, domain: str, dataset_specs=DATASET_SPECS):
    aliases = dataset_specs[domain]
    exts = {".json", ".jsonl", ".csv", ".parquet", ".pkl", ".pickle"}
    candidates = []
    for p in Path(root).rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        name = p.name.lower()
        if any(a in name for a in aliases):
            if any(skip in name for skip in ["result", "metric", "plot", "cache", "activation", "feat"]):
                continue
            candidates.append(p)
    return sorted(set(candidates), key=lambda x: (len(str(x)), str(x)))


def read_any_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        obj = json.loads(path.read_text())
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            for key in ["data", "train", "examples", "records", "samples"]:
                if key in obj and isinstance(obj[key], list):
                    return pd.DataFrame(obj[key])
            try:
                return pd.DataFrame(obj)
            except Exception:
                rows = []
                for k, value in obj.items():
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                item = dict(item)
                                item["_split_or_key"] = k
                                rows.append(item)
                if rows:
                    return pd.DataFrame(rows)
        raise ValueError(f"Unsupported JSON layout: {path}")
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        obj = pd.read_pickle(path)
        return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
    raise ValueError(f"Unsupported file suffix: {path}")


def pick_first_existing(row, candidates):
    for c in candidates:
        if c in row:
            val = row[c]
            if isinstance(val, float) and pd.isna(val):
                continue
            if val is not None:
                if isinstance(val, list):
                    if val:
                        return val[0]
                    continue
                return val
    return None


def infer_binary_label(row) -> int:
    label_cols = [
        "label", "labels", "label_name", "class", "target", "truth", "truthful", "is_true", "is_truthful",
        "is_false", "is_deceptive", "is_lie", "lie", "deceptive", "honest", "correct", "answer_label",
        "expected_label", "ground_truth", "expect_deception", "graded_deception", "claim_label", "sycophancy_occurred",
    ]
    val = pick_first_existing(row, label_cols)
    if val is None:
        raise ValueError("Could not infer label from row.")
    if isinstance(val, str):
        lower = val.strip().lower()
        if lower == "true":
            val = True
        elif lower == "false":
            val = False
    if isinstance(val, (bool, np.bool_)):
        for c in ["truth", "truthful", "is_true", "is_truthful", "honest", "correct"]:
            if c in row and pd.notna(row[c]):
                return 0 if bool(row[c]) else 1
        return int(bool(val))
    if isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, bool):
        return int(val)

    s = str(val).strip().lower()
    truthful = {"true", "truth", "truthful", "honest", "supported", "correct", "yes", "nondeceptive", "not deceptive", "not_lie", "not lie", "normal", "0"}
    deceptive = {"false", "deceptive", "deception", "lie", "lying", "refuted", "incorrect", "wrong", "sycophantic", "dishonest", "sandbag", "sandbagging", "insider_trading", "1"}
    if s in truthful:
        return 0
    if s in deceptive:
        return 1
    raise ValueError(f"Unknown label value: {val!r}")


def default_user_prompt_for_domain(domain: str, row=None):
    if domain == "ethical":
        if row is not None:
            action = pick_first_existing(row, ["action", "scenario", "description", "input", "question", "prompt", "user_prompt"])
            if action is not None:
                return str(action)
        return "Is this wrong?"
    return "Please tell me a fact."


def flatten_dict(d, max_depth: int = 3):
    out = dict(d)
    for _ in range(max_depth):
        for k in list(out):
            if isinstance(out[k], str) and out[k].strip().startswith("{") and out[k].strip().endswith("}"):
                try:
                    out[k] = ast.literal_eval(out[k])
                except Exception:
                    pass
            if isinstance(out[k], dict):
                for sub_k, sub_v in out[k].items():
                    out.setdefault(sub_k, sub_v)
    return out


def normalize_raw_frame(df: pd.DataFrame, domain: str, source_path: Path) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        r = flatten_dict(row.to_dict())
        if "Claim" in r and "Negated Claim" in r:
            rows.extend([
                {"domain": domain, "user_prompt": "Is this statement true?", "assistant_text": str(r["Claim"]), "label": 0, "source_path": str(source_path)},
                {"domain": domain, "user_prompt": "Is this statement true?", "assistant_text": str(r["Negated Claim"]), "label": 1, "source_path": str(source_path)},
            ])
            continue

        assistant_text = pick_first_existing(r, [
            "assistant_text", "assistant", "response", "completion", "generation", "claim", "statement", "text", "answer",
            "model_response", "output_str", "output_strs", "biography_answer_text",
        ])
        user_prompt = pick_first_existing(r, ["user_prompt", "prompt", "question", "input", "user", "query", "input_str", "input_strs", "plain_messages"])

        if assistant_text is None and "messages" in r and isinstance(r["messages"], (list, tuple)):
            for m in r["messages"]:
                if isinstance(m, dict) and m.get("role") == "assistant":
                    assistant_text = m.get("content")
                    break
            for m in r["messages"]:
                if isinstance(m, dict) and m.get("role") == "user":
                    user_prompt = m.get("content")
                    break
        if assistant_text is None:
            for k, v in r.items():
                if isinstance(v, str) and k not in ["domain", "label", "expected_label", "dataset_class", "base_name", "variant", "model", "rollouts"]:
                    if len(v) > 20:
                        assistant_text = v
                        break
        if assistant_text is None:
            continue
        if user_prompt is None or str(user_prompt).strip() == str(assistant_text).strip():
            user_prompt = default_user_prompt_for_domain(domain, r)
        try:
            y = infer_binary_label(r)
        except Exception:
            if "is_true" in r:
                y = 0 if bool(r["is_true"]) else 1
            else:
                continue
        rows.append({
            "domain": domain,
            "user_prompt": str(user_prompt),
            "assistant_text": str(assistant_text),
            "label": int(y),
            "source_path": str(source_path),
        })
    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise ValueError(f"No usable rows parsed from {source_path}")
    return out[out["label"].isin([0, 1])].reset_index(drop=True)


def load_domain_dataset(
    domain: str,
    root: Path,
    max_examples_per_domain: int | None = None,
    random_seed: int = 0,
    dataset_specs=DATASET_SPECS,
) -> pd.DataFrame:
    files = find_dataset_files(root, domain, dataset_specs)
    if not files:
        raise FileNotFoundError(f"Could not find a data file for domain={domain!r} under {root}")
    last_error = None
    for p in files:
        try:
            norm = normalize_raw_frame(read_any_table(p), domain, p)
            if norm["label"].nunique() < 2 and domain in ["roleplaying", "insider_trading", "sandbagging"]:
                alpaca_files = find_dataset_files(root, "alpaca", dataset_specs)
                if alpaca_files:
                    alpaca_norm = normalize_raw_frame(read_any_table(alpaca_files[0]), domain, alpaca_files[0])
                    alpaca_norm["label"] = 0
                    norm = pd.concat([norm, alpaca_norm], ignore_index=True)
            if norm["label"].nunique() < 2:
                continue
            if max_examples_per_domain is not None and len(norm) > max_examples_per_domain:
                per_label = max(1, max_examples_per_domain // 2)
                norm = pd.concat([
                    g.sample(min(len(g), per_label), random_state=random_seed)
                    for _, g in norm.groupby("label")
                ], ignore_index=True)
            return norm
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Found candidate files for {domain}, but none parsed successfully. Last error: {last_error}")


def load_all_datasets(
    domains,
    root: Path,
    max_examples_per_domain: int | None = None,
    random_seed: int = 0,
):
    frames, missing = [], []
    for domain in domains:
        try:
            frames.append(load_domain_dataset(domain, root, max_examples_per_domain, random_seed))
        except Exception as exc:
            print(f"WARNING: failed to load {domain}: {exc}")
            missing.append(domain)
    if not frames:
        return pd.DataFrame(columns=["domain", "user_prompt", "assistant_text", "label", "source_path"]), missing
    data = apply_domain_label_fixes(pd.concat(frames, ignore_index=True))
    return data, missing


def load_external_benchmarks(random_seed: int = 0, fever_per_class: int = 1000) -> pd.DataFrame:
    """Load the optional TruthfulQA and FEVER rows added in the main notebook."""
    rows = []
    try:
        from datasets import load_dataset
        tqa = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        for item in tqa:
            rows.append({
                "domain": "truthfulqa", "user_prompt": item["question"],
                "assistant_text": item["best_answer"], "label": 0, "source_path": "hf://truthful_qa",
            })
            if item["incorrect_answers"]:
                rows.append({
                    "domain": "truthfulqa", "user_prompt": item["question"],
                    "assistant_text": item["incorrect_answers"][0], "label": 1, "source_path": "hf://truthful_qa",
                })
    except Exception as exc:
        print(f"Failed to load TruthfulQA: {exc}")

    try:
        url = "https://huggingface.co/datasets/fever/fever/resolve/refs%2Fconvert%2Fparquet/v1.0/train/0000.parquet"
        fever_df = pd.read_parquet(url)
        fever_df = fever_df[fever_df["label"] != "NOT ENOUGH INFO"].copy()
        truthful = fever_df[fever_df["label"] == "SUPPORTS"].sample(fever_per_class, random_state=random_seed)
        deceptive = fever_df[fever_df["label"] == "REFUTES"].sample(fever_per_class, random_state=random_seed)
        for label, frame in [(0, truthful), (1, deceptive)]:
            for _, row in frame.iterrows():
                rows.append({
                    "domain": "fever", "user_prompt": "Is this claim true?",
                    "assistant_text": row["claim"], "label": label, "source_path": "hf://fever_parquet",
                })
    except Exception as exc:
        print(f"Failed to load FEVER: {exc}")
    return pd.DataFrame(rows)
