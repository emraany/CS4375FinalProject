"""Entry point. Call run(model, dataset) from a Colab cell."""

from __future__ import annotations

import os
from typing import Dict

import torch

from data.loaders import get_dataloaders
from evaluation.evaluator import evaluate, save_curves
from models import build_model
from training.trainer import train
from utils.seed import set_seed


def run(
    model_name: str,
    dataset_name: str,
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 0.001,
    seed: int = 42,
    val_split: float = 0.1,
    num_workers: int = 2,
) -> Dict:
    """Train and evaluate `model_name` on `dataset_name`, writing artifacts to results/."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, info = get_dataloaders(
        dataset_name,
        batch_size=batch_size,
        val_split=val_split,
        seed=seed,
        num_workers=num_workers,
    )

    model = build_model(
        model_name,
        in_channels=info["in_channels"],
        num_classes=info["num_classes"],
    ).to(device)

    run_tag = f"{model_name.lower()}_{dataset_name.lower()}"
    results_dir = os.path.join("results", run_tag)
    log_dir = os.path.join("runs", run_tag)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    ckpt_path = os.path.join(results_dir, "weights.pt")

    print(f"Training {model_name} on {dataset_name} for {epochs} epochs")
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        log_dir=log_dir,
        ckpt_path=ckpt_path,
    )

    print("Reloading best weights for test-set evaluation")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    metrics = evaluate(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=info["class_names"],
        out_dir=results_dir,
        history=history,
    )
    save_curves(history, results_dir)

    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "test_accuracy": metrics["test_accuracy"],
        "results_dir": results_dir,
        "log_dir": log_dir,
    }
    print(f"Done. Test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Artifacts: {results_dir}")
    return summary
