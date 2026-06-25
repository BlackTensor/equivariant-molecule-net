"""Global configuration for EquiDrug.

Central place for the random seed, device auto-detection, and all
hyperparameters. Import from here everywhere so runs stay reproducible
and consistent across the three models (Vanilla GNN, EGNN, SE(3)-Transformer).

This module has NO heavy dependencies beyond torch/numpy so it is safe to
import early (e.g. from verify_env.py or any training script).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # keep config importable even before torch is installed
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED: int = 42


def set_seed(seed: int = SEED, deterministic: bool = True) -> None:
    """Seed every RNG we touch so results are reproducible.

    Seeds Python's `random`, NumPy, and (if available) PyTorch on both CPU
    and CUDA. Call this once at the start of any script that trains or
    evaluates a model.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            # Favor reproducibility over the last few % of speed.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device() -> "str":
    """Auto-detect compute device: CUDA if present, else CPU.

    Local development/smoke-testing returns 'cpu'; Colab's free GPU
    returns 'cuda'. Returns a string so the module imports even when torch
    is not installed yet.
    """
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE: str = get_device()


# --------------------------------------------------------------------------- #
# Hyperparameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DataConfig:
    """Dataset loading / splitting (see Phase 1)."""

    subset_size: int = 50_000      # QM9 subset per CLAUDE.md free-cost guardrail
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1         # train + val + test must sum to 1.0
    # Prediction targets (QM9 column names handled in qm9_loader.py).
    targets: tuple[str, ...] = ("homo_lumo_gap", "dipole_moment")


@dataclass(frozen=True)
class ModelConfig:
    """Shared architecture sizes; kept identical across models for fairness."""

    hidden_dim: int = 128
    num_layers: int = 4
    out_dim: int = 2               # must match len(DataConfig.targets)
    dropout: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    """Optimization / training loop settings."""

    epochs: int = 100
    batch_size: int = 96
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 10.0
    # Mixed precision on Colab GPU (free-cost guardrail); ignored on CPU.
    use_amp: bool = True
    num_workers: int = 2
    # Paths (relative to repo root).
    weights_dir: str = "weights"
    log_dir: str = "weights"       # CSV metric logs live alongside weights


@dataclass(frozen=True)
class RotationConfig:
    """Rotation stress test / ESS settings (see Phase 4)."""

    num_rotations: int = 24        # every 15 degrees, 0..345


@dataclass(frozen=True)
class Config:
    """Top-level config bundling all sections plus global settings."""

    seed: int = SEED
    device: str = field(default_factory=get_device)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    rotation: RotationConfig = field(default_factory=RotationConfig)


# Convenient ready-made instance for imports: `from src.train.config import CONFIG`
CONFIG = Config()


if __name__ == "__main__":
    set_seed()
    print(f"Seed:   {CONFIG.seed}")
    print(f"Device: {CONFIG.device}")
    print(f"Data:   {CONFIG.data}")
    print(f"Model:  {CONFIG.model}")
    print(f"Train:  {CONFIG.train}")
    print(f"Rotate: {CONFIG.rotation}")
    # Sanity: splits sum to 1.0 and out_dim matches number of targets.
    d = CONFIG.data
    assert abs(d.train_frac + d.val_frac + d.test_frac - 1.0) < 1e-9, "splits must sum to 1"
    assert CONFIG.model.out_dim == len(d.targets), "out_dim must match number of targets"
    print("\nConfig self-check passed.")
