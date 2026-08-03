"""Canonical recording aggregation shared by both papers.

Official rule
-------------
1. Each window casts one vote using the argmax class.
2. A unique vote winner is the recording prediction.
3. If vote counts tie, compare cumulative window probabilities only among
   the tied classes.
4. If cumulative probabilities also tie exactly, choose the smallest class
   id.  This final fallback is deterministic and independent of window order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

PROTOCOL_VERSION = "1.0.0"
OFFICIAL_EPOCH = "last"
OFFICIAL_AGGREGATION = "majority_probability_tiebreak"


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    cm = np.asarray(cm, dtype=np.int64)
    values: list[float] = []
    for c in range(cm.shape[0]):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum()) - tp
        fn = int(cm[c, :].sum()) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(values))


def resolve_majority_vote(vote_counts: Sequence[int], probability_sums: Sequence[float]) -> tuple[int, bool, bool]:
    """Return prediction, whether votes tied, and whether probability tie-break was decisive."""
    votes = np.asarray(vote_counts, dtype=np.int64)
    scores = np.asarray(probability_sums, dtype=np.float64)
    if votes.ndim != 1 or scores.shape != votes.shape or len(votes) == 0:
        raise ValueError("vote_counts and probability_sums must be non-empty 1-D arrays with equal shape")
    tied = np.flatnonzero(votes == votes.max())
    if len(tied) == 1:
        return int(tied[0]), False, False
    tied_scores = scores[tied]
    score_winners = tied[np.flatnonzero(np.isclose(tied_scores, tied_scores.max(), rtol=0.0, atol=1e-12))]
    return int(score_winners.min()), True, len(score_winners) < len(tied)


@dataclass
class _Record:
    label: int
    votes: np.ndarray
    probability_sums: np.ndarray
    windows: int = 0


class RecordingAccumulator:
    """Accumulate window probabilities and score recordings under the official rule."""

    def __init__(self, num_classes: int):
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        self.num_classes = int(num_classes)
        self._records: dict[str, _Record] = {}

    def add(self, recording_id: str, label: int, probabilities: Sequence[float]) -> None:
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.shape != (self.num_classes,):
            raise ValueError(f"expected probability vector shape {(self.num_classes,)}, got {probs.shape}")
        if not np.all(np.isfinite(probs)):
            raise ValueError("probabilities contain non-finite values")
        label = int(label)
        if not 0 <= label < self.num_classes:
            raise ValueError(f"label {label} outside [0, {self.num_classes})")
        record = self._records.get(str(recording_id))
        if record is None:
            record = _Record(label, np.zeros(self.num_classes, dtype=np.int64), np.zeros(self.num_classes))
            self._records[str(recording_id)] = record
        elif record.label != label:
            raise ValueError(f"recording {recording_id!r} has conflicting labels {record.label} and {label}")
        record.votes[int(np.argmax(probs))] += 1
        record.probability_sums += probs
        record.windows += 1

    def add_many(self, recording_id: str, label: int, probabilities: np.ndarray) -> None:
        matrix = np.asarray(probabilities, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.num_classes:
            raise ValueError(f"expected probability matrix with {self.num_classes} columns, got {matrix.shape}")
        for row in matrix:
            self.add(recording_id, label, row)

    def predictions(self, aggregation: str = OFFICIAL_AGGREGATION) -> tuple[list[int], list[int], dict[str, int]]:
        if aggregation not in {OFFICIAL_AGGREGATION, "probability_sum"}:
            raise ValueError(f"unknown recording aggregation: {aggregation}")
        y_true: list[int] = []
        y_pred: list[int] = []
        ties = probability_decisions = 0
        for recording_id in sorted(self._records):
            record = self._records[recording_id]
            if record.windows == 0:
                continue
            if aggregation == "probability_sum":
                prediction = int(np.argmax(record.probability_sums))
            else:
                prediction, tied, probability_decisive = resolve_majority_vote(record.votes, record.probability_sums)
                ties += int(tied)
                probability_decisions += int(probability_decisive)
            y_true.append(record.label)
            y_pred.append(prediction)
        return y_true, y_pred, {"vote_ties": ties, "probability_tiebreaks": probability_decisions}

    def metrics(self, aggregation: str = OFFICIAL_AGGREGATION) -> dict[str, object]:
        y_true, y_pred, tie_info = self.predictions(aggregation)
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for truth, pred in zip(y_true, y_pred):
            cm[truth, pred] += 1
        return {
            "recording_accuracy": float(np.trace(cm) / max(int(cm.sum()), 1)),
            "recording_macro_f1": macro_f1_from_confusion(cm),
            "recording_confusion_matrix": cm.tolist(),
            "num_recordings": int(cm.sum()),
            "aggregation": aggregation,
            "protocol_version": PROTOCOL_VERSION,
            **tie_info,
        }


def aggregate_rows(
    rows: Sequence[Mapping[str, object]],
    probabilities: np.ndarray,
    *,
    num_classes: int,
    label_key: str = "label",
) -> RecordingAccumulator:
    matrix = np.asarray(probabilities, dtype=np.float64)
    if len(rows) != len(matrix):
        raise ValueError("rows and probabilities must have equal length")
    accumulator = RecordingAccumulator(num_classes)
    for row, probs in zip(rows, matrix):
        recording_id = f"{row['dataset_id']}::{row['recording_id']}"
        accumulator.add(recording_id, int(row[label_key]), probs)
    return accumulator
