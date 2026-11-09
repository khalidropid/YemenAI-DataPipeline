from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


def metrics(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int = 5) -> dict:
    """Compute accuracy, macro metrics and a confusion matrix without sklearn."""
    truth = np.asarray(list(y_true), dtype=np.int64)
    pred = np.asarray(list(y_pred), dtype=np.int64)
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same length.")
    if truth.size == 0:
        raise ValueError("Cannot calculate metrics for an empty target array.")

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for actual, guessed in zip(truth, pred):
        if 0 <= actual < num_classes and 0 <= guessed < num_classes:
            cm[actual, guessed] += 1

    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(tp), where=(precision + recall) > 0,
    )
    total = int(cm.sum())
    report = {}
    for i in range(num_classes):
        report[str(i)] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1-score": float(f1[i]),
            "support": int(support[i]),
        }
    report["macro avg"] = {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1-score": float(f1.mean()),
        "support": total,
    }
    weighted = np.average(f1, weights=support) if support.sum() else 0.0
    report["weighted avg"] = {
        "precision": float(np.average(precision, weights=support)) if support.sum() else 0.0,
        "recall": float(np.average(recall, weights=support)) if support.sum() else 0.0,
        "f1-score": float(weighted),
        "support": total,
    }
    return {
        "accuracy": float(tp.sum() / max(total, 1)),
        "balanced_accuracy": float(recall.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    count = 0
    all_targets, all_predictions = [], []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(features)
            loss = criterion(logits, targets)

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            batch_count = targets.size(0)
            loss_sum += float(loss.detach().item()) * batch_count
            count += batch_count
            all_targets.extend(targets.detach().cpu().tolist())
            all_predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())

    if count == 0:
        raise ValueError("The data loader produced no batches.")
    return loss_sum / count, metrics(all_targets, all_predictions)


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    learning_rate: float,
    device: str | torch.device,
    output_dir: str | Path,
    class_weights: torch.Tensor | None = None,
):
    """Train with AdamW, validation macro-F1 checkpoint selection and early stop."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    model.to(device)

    weights = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_f1 = -1.0
    best_epoch = 0
    patience, stale_epochs = 8, 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_scores = _run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_scores = _run_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(val_scores["macro_f1"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_scores["accuracy"],
            "train_macro_f1": train_scores["macro_f1"],
            "validation_loss": val_loss,
            "validation_accuracy": val_scores["accuracy"],
            "validation_balanced_accuracy": val_scores["balanced_accuracy"],
            "validation_macro_f1": val_scores["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_f1={train_scores['macro_f1']:.4f} | "
            f"val_loss={val_loss:.4f} val_f1={val_scores['macro_f1']:.4f}"
        )

        if val_scores["macro_f1"] > best_f1:
            best_f1 = val_scores["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_validation_macro_f1": best_f1,
                    "validation_metrics": val_scores,
                },
                output_dir / "best.pt",
            )
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            print(f"Early stopping: no validation macro-F1 improvement for {patience} epochs.")
            break

    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"Best validation macro-F1: {best_f1:.4f} at epoch {best_epoch}")
    return history
