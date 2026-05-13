"""Evaluation utilities: test metrics, confusion matrix, curve plots."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader


@torch.no_grad()
def _collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
    out_dir: str,
    history: Dict[str, List[float]] | None = None,
) -> Dict:
    """Run test inference and write confusion matrix and metrics.json. Returns the metrics dict."""
    os.makedirs(out_dir, exist_ok=True)

    preds, targets = _collect_predictions(model, test_loader, device)
    labels = list(range(len(class_names)))

    accuracy = float(accuracy_score(targets, preds))

    per_class_p, per_class_r, per_class_f, per_class_support = (
        precision_recall_fscore_support(
            targets, preds, labels=labels, average=None, zero_division=0,
        )
    )
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        targets, preds, labels=labels, average="macro", zero_division=0,
    )
    weighted_p, weighted_r, weighted_f, _ = precision_recall_fscore_support(
        targets, preds, labels=labels, average="weighted", zero_division=0,
    )

    per_class = {
        name: {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1": float(per_class_f[i]),
            "support": int(per_class_support[i]),
        }
        for i, name in enumerate(class_names)
    }

    metrics = {
        "test_accuracy": accuracy,
        "macro": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1": float(macro_f),
        },
        "weighted": {
            "precision": float(weighted_p),
            "recall": float(weighted_r),
            "f1": float(weighted_f),
        },
        "per_class": per_class,
    }
    if history is not None and len(history.get("train_loss", [])) > 0:
        metrics["final_epoch"] = {
            "train_loss": history["train_loss"][-1],
            "train_acc": history["train_acc"][-1],
            "val_loss": history["val_loss"][-1],
            "val_acc": history["val_acc"][-1],
            "best_val_acc": max(history["val_acc"]),
        }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        targets, preds,
        display_labels=class_names,
        ax=ax,
        xticks_rotation=45,
        colorbar=False,
    )
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    return metrics


def save_curves(history: Dict[str, List[float]], out_dir: str) -> None:
    """Save loss and accuracy curves as PNGs."""
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, history["train_loss"], label="train")
    ax.plot(epochs, history["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, history["train_acc"], label="train")
    ax.plot(epochs, history["val_acc"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training and validation accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_curve.png"), dpi=150)
    plt.close(fig)
