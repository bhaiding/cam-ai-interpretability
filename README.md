# CAM-Based LLM Deception Detection

Research code for detecting truthfulness/deception from transformer residual-stream activations using **INLP-derived CAM rows** and **winner-take-all (WTA)** classifiers.

This repository is a cleaned, modular conversion of three Llama-3.3-70B research notebooks:

- projected CAM/WTA with an unsupervised CAM-row-subspace projection, LSQ quantization-aware training, and Euclidean-compatible scoring;
- full-dimensional Hamming WTA with learned positive per-row weights;
- the same Hamming classifier with all CAM rows unweighted.

The original notebooks are preserved in `notebooks/original/`, but the main implementation now lives in importable Python modules under `src/cam_deception/`.

## Core pipeline

```text
Dataset rows
   ↓
Llama-3.3-70B residual stream (layer 33, post-layer)
   ↓
Shared example-level train / validation / test split
   ↓
INLP logistic-probe directions
   ↓
CAM row banks
   ├── combined/general bank
   └── per-domain banks
   ↓
Classifier family
   ├── Projected LSQ/QAT WTA
   ├── Weighted Hamming WTA
   └── Unweighted Hamming WTA
   ↓
Cross-domain AUROC / accuracy / artifacts / heatmaps
```

The split is constructed once and reused for both CAM-row extraction and WTA training so held-out examples do not leak into the learned probe directions.

## Repository layout

```text
cam-deception-detection/
├── src/cam_deception/
│   ├── activations.py      # Llama loading + layer-33 residual extraction
│   ├── cache.py            # mmap/NPZ/PT residual cache I/O
│   ├── cam_rows.py         # logistic probes + INLP + CAM banks
│   ├── config.py           # experiment dataclasses and defaults
│   ├── data.py             # truth_spec normalization + optional TQA/FEVER
│   ├── dataset_view.py     # scope/split views over cached activations
│   ├── hamming.py          # weighted/unweighted Hamming WTA
│   ├── metrics.py          # AUROC, thresholding, binary metrics
│   ├── pipeline.py         # cross-domain experiment orchestration
│   ├── plotting.py         # AUROC heatmaps
│   ├── projected_lsq.py    # projection, LSQ, Euclidean/dot WTA
│   └── splits.py           # shared example-level split
├── scripts/
│   ├── extract_activations.py
│   ├── inspect_cache.py
│   ├── run_projected_lsq.py
│   ├── run_hamming.py
│   └── run_all_classifiers.py
├── tests/
│   └── test_core.py
├── notebooks/original/
│   ├── CAM_Deception_Detection_70B.ipynb
│   ├── CAM_Deception_Detection_70B_Hamming_Weighted.ipynb
│   └── CAM_Deception_Detection_70B_Hamming_Unweighted.ipynb
├── results/
├── data/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <your-repository-url>
cd cam-deception-detection

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development/testing:

```bash
pip install -e ".[dev]"
pytest
```

## 1. Build or locate the residual-stream cache

The Hamming notebooks do **not** need to load Llama-70B. They operate directly on the activation cache produced by the main extraction pipeline.

To reproduce extraction from the paper datasets:

```bash
python scripts/extract_activations.py \
  --clone-truth-spec \
  --include-external \
  --output-dir results/llama70b_L33
```

This uses the main notebook defaults:

- model: `meta-llama/Llama-3.3-70B-Instruct`
- layer: `33`
- hidden state: `post_layer`
- example representation: mean over assistant-output-token residual streams
- disk cache: float16 mmap arrays

> Llama-3.3-70B is a large model. Activation extraction requires an appropriately provisioned GPU/multi-GPU environment and Hugging Face access to the model checkpoint. If you already have the residual cache, skip extraction entirely.

Inspect an existing cache with:

```bash
python scripts/inspect_cache.py --cache /path/to/..._activations_mmap
```

## 2. Projected LSQ/QAT WTA experiment

Equivalent to the main projected/quantized notebook path:

```bash
python scripts/run_projected_lsq.py \
  --cache /path/to/..._activations_mmap \
  --projection-dim 128 \
  --bits 3 \
  --eval-mode euclidean
```

Useful ablations are now command-line switches instead of notebook edits:

```bash
# No projection, quantization still enabled
python scripts/run_projected_lsq.py --cache /path/to/cache --no-projection

# Projection only; no LSQ quantization
python scripts/run_projected_lsq.py --cache /path/to/cache --no-quantization

# Neither projection nor quantization
python scripts/run_projected_lsq.py --cache /path/to/cache --no-projection --no-quantization
```

### Dot-product-preserving projection

The default projection is **unsupervised**. It constructs an orthonormal basis from the CAM-row span and does not use class labels. When the projection dimension covers the CAM-row rank, projecting both a residual vector and a CAM row preserves their dot product up to numerical precision.

### Euclidean-compatible CAM scoring

For deployment, learned row weights are baked into augmented rows:

```text
query_aug = [q, 0]
row_aug_j = [alpha_j * r_j, sqrt(C - ||alpha_j * r_j||²)]
```

Every augmented row has equal norm, so nearest Euclidean row and maximum weighted dot-product row have the same winner.

## 3. Hamming WTA experiments

### Weighted CAM rows

```bash
python scripts/run_hamming.py \
  --cache /path/to/..._activations_mmap \
  --method both
```

The weighted classifier learns a positive scalar for each CAM row before the WTA max.

### Unweighted CAM rows

```bash
python scripts/run_hamming.py \
  --cache /path/to/..._activations_mmap \
  --method both \
  --unweighted
```

The unweighted version uses the maximum **raw normalized Hamming similarity** and one global bias. This replaces the separate unweighted notebook with one shared implementation.

### Hash methods

`run_hamming.py` supports:

- `sign`: direct sign hash over the original residual dimensions;
- `learned_matrix`: a full `d × d` shared matrix initialized to identity and trained through a straight-through sign estimator;
- `both`: run both experiments.

For an 8192-dimensional residual stream, the learned matrix is extremely large. Use `--method sign` when you only need the hardware-friendly direct-sign baseline.
If you explicitly need the trained full matrix in the saved artifact, add `--save-learned-hash-matrix`; it is omitted by default because an 8192×8192 fp32 matrix is hundreds of MiB before compression.

## 4. Run all notebook experiment families

Once a residual cache exists:

```bash
python scripts/run_all_classifiers.py --cache /path/to/..._activations_mmap
```

To avoid the expensive full learned hash matrix:

```bash
python scripts/run_all_classifiers.py \
  --cache /path/to/..._activations_mmap \
  --skip-learned-hash
```

This runs:

1. projected LSQ/QAT WTA;
2. weighted Hamming WTA;
3. unweighted Hamming WTA.

## CAM-row extraction

The repository preserves the notebook's INLP procedure. A logistic probe is fit to the shared training activations, its normalized weight vector is stored as a CAM row, that direction is projected out, and the process repeats.

Default bank sizes are:

- 4 directions per single domain;
- 10 directions for the combined/general bank.

The positive label convention is:

```text
1 = deceptive / false / target behavior
0 = truthful / honest
```

The heuristic paper-dataset loader retains the notebook's explicit empirical-domain label-orientation fix. Old caches that predate that fix can be corrected explicitly with `--flip-cached-domain empirical` rather than silently flipping every cache.

## Evaluation

The main reported metric is **AUROC**. Each trained source-domain classifier is evaluated on the shared held-out test examples from every other domain.

The runners save:

- CSV metric tables;
- cross-domain AUROC heatmaps;
- compact CAM/classifier `.npz` artifacts;
- projected/quantized rows for the LSQ path;
- binary CAM codes for the Hamming path.

Large activation caches, model checkpoints, and generated artifacts are excluded by `.gitignore`.

## Tests

The included tests verify several invariants that were implicit in the notebooks:

- no example ID leaks across the shared train/validation/test split;
- the CAM-row-subspace projection preserves CAM dot products when its dimension covers the row-bank rank;
- augmented Euclidean scoring agrees with weighted dot-product scoring;
- unweighted Hamming WTA is exactly the maximum raw Hamming similarity plus the global bias.

Run:

```bash
pytest
```

## Notes on the conversion

The repository intentionally keeps the experimental behavior close to the uploaded notebooks while removing notebook-specific duplication and hidden state. In particular:

- weighted and unweighted Hamming classifiers now share one implementation controlled by `weighted_rows` / `--unweighted`;
- shared INLP, split, cache, metrics, and cross-domain evaluation logic is defined once;
- experiment settings are collected into dataclasses instead of scattered notebook globals;
- the original notebooks remain available for provenance and result comparison;
- the global deployment artifact naming is consistently `llama70b` rather than the stray `llama8b` filename that appeared in the notebook save cell.

## Research motivation

The broader goal is to connect LLM interpretability with hardware-efficient model monitoring. Instead of keeping a single software-only linear probe, the project studies whether multiple learned residual-stream directions can be stored as a compact CAM feature bank and queried using simple similarity operations, including aggressively quantized vectors and one-bit Hamming codes.
