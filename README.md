# CAM-Based LLM Deception Detection

This repository contains research code for detecting **truthfulness and deception in large language models using residual-stream activations and content-addressable memory (CAM)**.

The project explores whether learned activation-space directions associated with truthful or deceptive behavior can be stored in a compact CAM-style feature bank and used for efficient inference. Alongside classification performance, the work studies the effects of **dimensionality reduction, low-bit quantization, cross-domain generalization, and feature-selection strategies** with the goal of making mechanistic interpretability methods more compatible with hardware-efficient deployment.

## Project Overview

Modern LLMs encode information about their internal behavior within high-dimensional activation spaces. This project investigates whether those internal representations can be used to distinguish truthful from deceptive responses.

The general pipeline is:

1. Extract residual-stream activations from an intermediate transformer layer.
2. Learn directions in activation space associated with truthfulness or deception.
3. Store selected feature vectors as rows in a CAM-style classifier.
4. Compare an incoming activation against the stored feature bank.
5. Use learned feature weights and a classification threshold to predict whether the example is truthful or deceptive.
6. Evaluate how well the system generalizes across datasets and how performance changes under hardware-oriented compression.

Experiments were conducted primarily with **Llama-3.1-8B-Instruct** and **Llama-3.3-70B-Instruct**.

---

## Key Research Questions

This project investigates several questions:

* Can residual-stream activations reliably separate truthful and deceptive LLM behavior?
* Can multiple learned activation directions be used as a CAM-based alternative to a conventional linear probe?
* How well do learned deception features generalize across domains?
* How many CAM rows are actually used during classification?
* How much performance is lost when feature vectors are compressed?
* Can an **unsupervised dot-product-preserving projection** reduce activation dimensionality while preserving classifier behavior?
* How does **low-bit quantization using LSQ/QAT** affect classification accuracy and AUROC?
* Do domain-general feature directions outperform domain-specific directions?
* Can complementary transformer layers improve detection performance?

---

## Methodology

### Activation Extraction

Residual-stream activations are extracted from selected transformer layers using last-token pooling.

Experiments include:

* **Llama-3.1-8B-Instruct**

  * Primary layer: Layer 15

* **Llama-3.3-70B-Instruct**

  * Primary layer: Layer 33

The resulting activation vectors represent the model's internal state for each prompt and are used as classifier inputs.

---

### Linear Probing

Linear probes are trained on residual-stream activations to identify directions associated with truthfulness and deception.

A linear classifier learns a separating direction

[
w \in \mathbb{R}^{D}
]

where (D) is the LLM residual-stream dimension.

These learned directions provide both:

* a conventional classification baseline
* candidate feature vectors for the CAM classifier

---

## CAM / Winner-Take-All Classifier

Instead of using a single classification vector, the CAM classifier stores a bank of learned feature vectors:

[
F = {f_1, f_2, ..., f_K}
]

Each incoming activation (x) is compared against every stored feature.

A similarity score is computed using the dot product:

[
s_i = x^\top f_i
]

Each feature also receives a learned scalar weight:

[
s_i = W_i(x^\top f_i)
]

The most responsive feature can then be selected using a **winner-take-all (WTA)** operation.

This architecture allows multiple independently learned activation directions to participate in classification while remaining compatible with a CAM-style similarity-search architecture.

---

## Feature Banks

Experiments compare several approaches to constructing the CAM feature bank.

### Domain-Specific Features

Feature vectors are trained independently on individual deception or truthfulness domains.

### Domain-General Features

Features from several domains are combined into a single feature bank intended to capture more general representations of truthfulness and deception.

### Random-Vector Control

Random vectors are placed into CAM and trained using the same downstream weighting procedure to determine whether performance comes from meaningful learned activation directions rather than classifier capacity alone.

---

## Cross-Domain Evaluation

The classifier is evaluated using cross-domain AUROC matrices.

Each experiment trains features on one dataset or domain and evaluates them across several others.

This allows the project to measure:

* within-domain performance
* cross-domain transfer
* domain-specific behavior
* general deception representations

Evaluation focuses primarily on **AUROC**, with classification accuracy used as an additional metric.

---

## Dimensionality Reduction

Large LLM residual streams contain thousands of dimensions, making direct CAM deployment expensive.

To address this, the project evaluates an **unsupervised linear projection to 128 dimensions**.

The projection is designed to preserve the dot products between:

* residual-stream activation vectors
* stored CAM feature vectors

Because CAM classification depends directly on these similarities, preserving their dot products helps maintain classifier behavior after dimensionality reduction.

Importantly, the projection is **unsupervised** and does not use class labels.

Experiments compare:

* Full-dimensional activations
* 128-dimensional projected activations

---

## Quantization

The project also studies aggressive low-bit quantization of CAM representations.

Experiments use:

* **Learned Step Size Quantization (LSQ)**
* **Quantization-Aware Training (QAT)**
* primarily **3-bit quantization**

The objective is to determine how far CAM rows and classifier parameters can be compressed while maintaining useful AUROC.

Ablation experiments compare:

* No compression
* Projection only
* Quantization only
* Projection + quantization

---

## Multi-Layer Monitoring

Different transformer layers may encode complementary information.

The project therefore explores classifiers that monitor several transformer layers simultaneously.

The motivation is that:

> When one layer fails to distinguish an example correctly, another layer may contain a stronger representation of the relevant behavior.

Strategies explored include:

* Single-layer baseline
* Greedy layer selection
* Accumulative layer combinations
* Weighted combinations of multiple layers

---

## Ablation Studies

The repository contains experiments investigating:

* CAM bank size
* Number of unique WTA winners
* Domain-specific vs. domain-general features
* Random feature vectors
* Dot-product vs. Euclidean similarity
* Projection dimension
* Low-bit quantization
* Projection vs. quantization performance loss
* Multi-layer feature combinations
* Cross-domain transfer
* Multi-class CAM classification

These experiments are intended to isolate which components of the system are responsible for classifier performance.

---

## Models

Primary models used in this project include:

### Meta Llama 3.1 8B Instruct

Used for:

* rapid experimentation
* classifier development
* feature-bank experiments
* multi-layer analysis
* ablation studies

### Meta Llama 3.3 70B Instruct

Used to evaluate whether the approach scales to substantially larger language models.

Because the full BF16 model is approximately 140 GB, memory-efficient model loading and activation extraction techniques are used for the 70B experiments.

---

## Technologies

The project uses:

* Python
* PyTorch
* Hugging Face Transformers
* Llama 3.1 / Llama 3.3
* NNsight
* NumPy
* Pandas
* scikit-learn
* HDF5 / h5py
* Matplotlib
* FAISS
* CUDA
* Quantization-Aware Training
* Learned Step Size Quantization
* Linear probing
* Residual-stream activation analysis

---

## Evaluation

The primary evaluation metric is **Area Under the Receiver Operating Characteristic Curve (AUROC)**.

Cross-domain experiments are visualized using heatmaps in which:

* rows represent the domain used to construct or train the classifier
* columns represent the evaluation dataset
* each cell contains the resulting AUROC

Additional analyses include:

* accuracy
* unique CAM winners
* cosine similarity distributions
* feature utilization
* compression-induced AUROC loss

---

## Example Experimental Pipeline

```text
Prompt
  ↓
Llama
  ↓
Intermediate Transformer Layer
  ↓
Residual Stream Activation
  ↓
Optional 128-D Projection
  ↓
Optional Low-Bit Quantization
  ↓
CAM Feature Bank
  ↓
Similarity Search / Winner-Take-All
  ↓
Learned Feature Weights
  ↓
Truthful / Deceptive Prediction
```

---

## Research Motivation

Most mechanistic interpretability methods are evaluated primarily as software techniques. However, deploying activation-based monitoring systems at inference time introduces substantial computational and memory overhead.

This project investigates whether interpretability-derived feature directions can instead be mapped onto **content-addressable memory**, where similarity comparisons can be performed efficiently in parallel.

The broader goal is to connect:

**LLM interpretability → model monitoring → efficient hardware implementation**

and explore whether model-internal representations can support practical, low-overhead AI safety systems.

