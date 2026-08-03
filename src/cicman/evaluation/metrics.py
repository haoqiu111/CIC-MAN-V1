"""Small metric helpers without sklearn dependency."""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        if 0 <= int(true) < num_classes and 0 <= int(pred) < num_classes:
            matrix[int(true), int(pred)] += 1
    return matrix


def accuracy(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def macro_f1(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray, num_classes: int) -> float:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    f1_values = []
    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_values.append(f1)
    return float(np.mean(f1_values))


def balanced_accuracy(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray, num_classes: int) -> float:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    recalls = []
    for cls in range(num_classes):
        denom = cm[cls, :].sum()
        recalls.append(cm[cls, cls] / denom if denom else 0.0)
    return float(np.mean(recalls))

