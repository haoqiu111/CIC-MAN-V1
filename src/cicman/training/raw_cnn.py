"""Training utilities for the Raw 1D-CNN baseline."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from cicman.data.dataset import WindowIndexDataset
from cicman.evaluation.metrics import accuracy, balanced_accuracy, confusion_matrix, macro_f1
from cicman.evaluation.shortcuts import apply_label_shortcut
from cicman.models.raw_cnn import build_raw_cnn


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limit_dataset(dataset, max_items: int | None):
    if max_items is None or max_items <= 0 or max_items >= len(dataset):
        return dataset
    import torch

    return torch.utils.data.Subset(dataset, list(range(max_items)))


def make_loader(index_csv: Path, batch_size: int, shuffle: bool, num_workers: int, max_items: int | None):
    import torch

    dataset = WindowIndexDataset(index_csv)
    dataset = limit_dataset(dataset, max_items)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device: str,
    train: bool,
    *,
    shortcut_mode: str = "none",
    shortcut_amplitude: float = 1.0,
    num_classes: int | None = None,
) -> dict[str, object]:
    import torch

    model.train(train)
    losses = []
    all_true: list[int] = []
    all_pred: list[int] = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        x = apply_label_shortcut(
            x,
            y,
            mode=shortcut_mode,
            num_classes=num_classes,
            amplitude=shortcut_amplitude,
        )
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        pred = logits.argmax(dim=1).detach().cpu().numpy().tolist()
        true = y.detach().cpu().numpy().tolist()
        all_true.extend(true)
        all_pred.extend(pred)

    num_classes = int(max(all_true + all_pred) + 1) if all_true or all_pred else 1
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": accuracy(all_true, all_pred),
        "macro_f1": macro_f1(all_true, all_pred, num_classes),
        "balanced_accuracy": balanced_accuracy(all_true, all_pred, num_classes),
        "num_samples": len(all_true),
        "confusion_matrix": confusion_matrix(all_true, all_pred, num_classes).tolist(),
    }


def train_raw_cnn(
    *,
    train_index: Path,
    val_index: Path,
    test_index: Path,
    output_dir: Path,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    num_workers: int,
    seed: int,
    max_train_items: int | None = None,
    max_eval_items: int | None = None,
    shortcut_train_mode: str = "none",
    shortcut_val_mode: str = "none",
    shortcut_test_mode: str = "none",
    shortcut_amplitude: float = 1.0,
) -> dict[str, object]:
    import torch

    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = make_loader(train_index, batch_size, True, num_workers, max_train_items)
    val_loader = make_loader(val_index, batch_size, False, num_workers, max_eval_items)
    test_loader = make_loader(test_index, batch_size, False, num_workers, max_eval_items)

    model = build_raw_cnn(num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val = -1.0
    best_path = output_dir / "best.pt"
    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            shortcut_mode=shortcut_train_mode,
            shortcut_amplitude=shortcut_amplitude,
            num_classes=num_classes,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False,
            shortcut_mode=shortcut_val_mode,
            shortcut_amplitude=shortcut_amplitude,
            num_classes=num_classes,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if val_metrics["macro_f1"] > best_val:
            best_val = float(val_metrics["macro_f1"])
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "num_classes": num_classes,
                    "config": {
                        "shortcut_train_mode": shortcut_train_mode,
                        "shortcut_val_mode": shortcut_val_mode,
                        "shortcut_test_mode": shortcut_test_mode,
                        "shortcut_amplitude": shortcut_amplitude,
                    },
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        optimizer,
        device,
        train=False,
        shortcut_mode=shortcut_test_mode,
        shortcut_amplitude=shortcut_amplitude,
        num_classes=num_classes,
    )
    result = {
        "best_checkpoint": str(best_path),
        "best_epoch": checkpoint["epoch"],
        "best_val_metrics": checkpoint["val_metrics"],
        "test_metrics": test_metrics,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
