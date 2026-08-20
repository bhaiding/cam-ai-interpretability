from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import random

import numpy as np
import torch


DEFAULT_MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_LAYER_INDEX = 33
DEFAULT_HIDDEN_STATE_MODE = "post_layer"
POSITIVE_LABEL_NAME = "deceptive_or_false"

DEFAULT_DOMAINS = [
    "definitional",
    "empirical",
    "fictional",
    "logical",
    "ethical",
    "sycophancy",
    "exp_inverted",
    "roleplaying",
    "insider_trading",
    "sandbagging",
]

PREFERRED_DOMAINS = DEFAULT_DOMAINS + ["truthfulqa", "fever"]

DATASET_SPECS = {
    "definitional": ["definitional", "definition"],
    "empirical": ["empirical", "evidential", "facts", "factual"],
    "fictional": ["fictional", "fiction"],
    "logical": ["logical", "logic"],
    "ethical": ["ethical", "ethics"],
    "sycophancy": ["sycophancy", "sycophantic"],
    "exp_inverted": ["exp_inverted", "expectation", "inverted", "expectation_inverted"],
    "roleplaying": ["roleplaying", "roleplay"],
    "insider_trading": ["insider", "trading"],
    "sandbagging": ["sandbagging", "sandbag"],
    "alpaca": ["alpaca"],
}

# The uploaded notebook explicitly flips empirical labels after heuristic parsing.
FLIP_LABEL_DOMAINS = {"empirical"}


@dataclass
class PathsConfig:
    project_root: Path = field(default_factory=Path.cwd)
    truth_spec_dir: Optional[Path] = None
    output_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).expanduser().resolve()
        if self.truth_spec_dir is None:
            self.truth_spec_dir = self.project_root / "truth_spec"
        else:
            self.truth_spec_dir = Path(self.truth_spec_dir).expanduser().resolve()
        if self.output_dir is None:
            self.output_dir = self.project_root / "results" / "llama70b_L33"
        else:
            self.output_dir = Path(self.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> Path:
        p = self.output_dir / "activation_cache"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def artifact_dir(self) -> Path:
        p = self.output_dir / "artifacts"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def hamming_artifact_dir(self) -> Path:
        p = self.output_dir / "hamming_artifacts"
        p.mkdir(parents=True, exist_ok=True)
        return p


@dataclass
class ModelConfig:
    model_id: str = DEFAULT_MODEL_ID
    layer_index: int = DEFAULT_LAYER_INDEX
    hidden_state_mode: str = DEFAULT_HIDDEN_STATE_MODE
    device_map: str = "auto"
    dtype: torch.dtype = torch.bfloat16
    attn_implementation: str = "sdpa"


@dataclass
class ActivationConfig:
    batch_size: int = 1
    cache_dtype: str = "float16"
    train_on: str = "means"
    max_tokens_per_example: int = 64
    shard_size: int = 256
    use_mmap: bool = True
    force_reextract: bool = False
    clear_cuda_cache_every_n_batches: int = 1

    @property
    def extract_token_activations(self) -> bool:
        return self.train_on == "tokens"


@dataclass
class SplitConfig:
    test_fraction: float = 0.20
    val_fraction_of_train: float = 0.20
    random_seed: int = 0


@dataclass
class ProbeConfig:
    n_directions_single_domain: int = 4
    n_directions_combined: int = 10
    logreg_alpha: float = 1e-4
    use_standard_scaler: bool = False
    remove_general_before_single_domain: bool = False
    positive_label_name: str = POSITIVE_LABEL_NAME


@dataclass
class ProjectedWTAConfig:
    use_projection: bool = True
    projection_dim: int = 128
    projection_method: str = "cam_row_subspace"
    normalize_after_projection: bool = False
    projection_svd_rcond: float = 1e-6
    projection_pad_random_orthogonal: bool = True
    use_lsq_qat: bool = True
    lsq_bit_precision: int = 3
    lsq_signed: bool = True
    lsq_initial_scale: Optional[float] = None
    lsq_eps: float = 1e-8
    eval_distance_mode: str = "euclidean"
    epochs: int = 300
    lr: float = 3e-2
    weight_decay: float = 1e-4
    batch_size: int = 512
    eval_batch_size: int = 512
    random_seed: int = 0


@dataclass
class HammingConfig:
    methods: tuple[str, ...] = ("sign", "learned_matrix")
    weighted_rows: bool = True
    fixed_epochs: int = 150
    learned_epochs: int = 30
    early_stopping_patience: int = 6
    fixed_eval_every: int = 5
    learned_eval_every: int = 2
    readout_lr: float = 3e-2
    matrix_lr: float = 1e-4
    readout_weight_decay: float = 1e-4
    matrix_weight_decay: float = 1e-6
    fixed_batch_size: int = 512
    learned_batch_size: int = 32
    fixed_eval_batch_size: int = 2048
    learned_eval_batch_size: int = 64
    identity_reg: float = 1e-4
    bit_balance_reg: float = 1e-3
    ste_temperature: float = 1.0
    threshold_selection_metric: str = "accuracy"
    save_learned_hash_matrices: bool = False
    random_seed: int = 0
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    use_bf16: bool = field(default_factory=lambda: torch.cuda.is_available())


@dataclass
class ExperimentConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    activations: ActivationConfig = field(default_factory=ActivationConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    projected_wta: ProjectedWTAConfig = field(default_factory=ProjectedWTAConfig)
    hamming: HammingConfig = field(default_factory=HammingConfig)
    domains: list[str] = field(default_factory=lambda: list(DEFAULT_DOMAINS))
    max_examples_per_domain: Optional[int] = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
