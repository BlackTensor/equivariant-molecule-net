"""Train ONE model (Vanilla GNN, EGNN, or SE(3)-Transformer) on QM9.

Task 3.1: given a model name, run the training loop, log MAE/RMSE per epoch
to a CSV, and save the best-val-MAE weights. Reused as-is by
``notebooks/train_colab.ipynb`` (task 3.2) on Colab's free GPU — locally this
should only ever be smoke-tested with a tiny ``--epochs``/``--subset-size``,
never run to completion (see CLAUDE.md Phase 3 stop rule).

Usage:

    python -m src.train.train --model egnn
    python -m src.train.train --model vanilla --epochs 1 --subset-size 64  # smoke test
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.train.config import CONFIG, set_seed  # noqa: E402
from src.data.qm9_loader import get_qm9_splits, TargetNormalizer, select_targets  # noqa: E402
from src.models.vanilla_gnn import build_vanilla_gnn  # noqa: E402
from src.models.egnn import build_egnn  # noqa: E402
from src.models.se3_transformer import build_se3_transformer  # noqa: E402

MODEL_BUILDERS = {
    "vanilla": build_vanilla_gnn,
    "egnn": build_egnn,
    "se3": build_se3_transformer,
}


def forward_model(model_name: str, model: torch.nn.Module, batch) -> torch.Tensor:
    """Dispatch to the right forward signature (each model takes different args)."""
    if model_name == "vanilla":
        return model(batch.x, batch.edge_index, batch.batch)
    # egnn and se3 both use pos; se3 ignores edge_index/edge_attr internally.
    return model(batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch)


def run_epoch(
    model_name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    normalizer: TargetNormalizer,
    device: str,
    optimizer: torch.optim.Optimizer | None,
    max_steps: int | None = None,
) -> tuple[float, float]:
    """One pass over ``loader``. Trains if ``optimizer`` is given, else evaluates.

    Returns (MAE, RMSE) in physical units (denormalized), averaged over samples.
    """
    is_train = optimizer is not None
    model.train(is_train)

    abs_err_sum = torch.zeros(CONFIG.model.out_dim)
    sq_err_sum = torch.zeros(CONFIG.model.out_dim)
    n_samples = 0

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        batch = batch.to(device)
        target = normalizer.normalize(select_targets(batch.y))

        with torch.set_grad_enabled(is_train):
            pred = forward_model(model_name, model, batch)
            loss = F.mse_loss(pred, target)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.train.grad_clip)
            optimizer.step()

        with torch.no_grad():
            pred_phys = normalizer.denormalize(pred)
            target_phys = normalizer.denormalize(target)
            err = pred_phys - target_phys
            abs_err_sum += err.abs().sum(dim=0).cpu()
            sq_err_sum += (err ** 2).sum(dim=0).cpu()
            n_samples += err.size(0)

    mae = (abs_err_sum / n_samples).mean().item()
    rmse = (sq_err_sum / n_samples).sqrt().mean().item()
    return mae, rmse


def train(
    model_name: str,
    epochs: int | None = None,
    subset_size: int | None = None,
    max_steps: int | None = None,
) -> str:
    """Train ``model_name`` and return the path of the saved best-weights file."""
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"model must be one of {list(MODEL_BUILDERS)}, got {model_name!r}")

    set_seed(CONFIG.seed)
    device = CONFIG.device
    epochs = CONFIG.train.epochs if epochs is None else epochs

    splits = get_qm9_splits(subset_size=subset_size)
    normalizer = TargetNormalizer.from_dataset(splits.train)
    # Move normalizer stats to the training device -- batches are moved to
    # `device` below, and normalize()/denormalize() combine them elementwise.
    normalizer = TargetNormalizer(
        mean=normalizer.mean.to(device),
        std=normalizer.std.to(device),
        targets=normalizer.targets,
    )

    train_loader = DataLoader(splits.train, batch_size=CONFIG.train.batch_size, shuffle=True)
    val_loader = DataLoader(splits.val, batch_size=CONFIG.train.batch_size, shuffle=False)

    model = MODEL_BUILDERS[model_name]().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CONFIG.train.lr, weight_decay=CONFIG.train.weight_decay
    )

    os.makedirs(CONFIG.train.weights_dir, exist_ok=True)
    os.makedirs(CONFIG.train.log_dir, exist_ok=True)
    weights_path = os.path.join(CONFIG.train.weights_dir, f"{model_name}_best.pt")
    log_path = os.path.join(CONFIG.train.log_dir, f"{model_name}_train_log.csv")

    best_val_mae = float("inf")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_mae", "train_rmse", "val_mae", "val_rmse", "seconds"])

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_mae, train_rmse = run_epoch(
                model_name, model, train_loader, normalizer, device, optimizer, max_steps
            )
            val_mae, val_rmse = run_epoch(
                model_name, model, val_loader, normalizer, device, None, max_steps
            )
            elapsed = time.time() - t0

            writer.writerow([epoch, train_mae, train_rmse, val_mae, val_rmse, elapsed])
            f.flush()
            print(
                f"[{model_name}] epoch {epoch}/{epochs}  "
                f"train_mae={train_mae:.4f} train_rmse={train_rmse:.4f}  "
                f"val_mae={val_mae:.4f} val_rmse={val_rmse:.4f}  ({elapsed:.1f}s)"
            )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(
                    {
                        "model_name": model_name,
                        "model_state": model.state_dict(),
                        "epoch": epoch,
                        "val_mae": val_mae,
                        "normalizer_mean": normalizer.mean,
                        "normalizer_std": normalizer.std,
                        "targets": normalizer.targets,
                    },
                    weights_path,
                )

    return weights_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one EquiDrug model on QM9.")
    parser.add_argument("--model", required=True, choices=list(MODEL_BUILDERS))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap batches per epoch (smoke testing only; omit for real training).",
    )
    args = parser.parse_args()

    weights_path = train(
        args.model,
        epochs=args.epochs,
        subset_size=args.subset_size,
        max_steps=args.max_steps,
    )
    print(f"\nBest weights saved to: {weights_path}")


if __name__ == "__main__":
    main()
