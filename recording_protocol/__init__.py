"""Shared recording-level evaluation protocol for paper 1 and paper 2."""

from .aggregation import (
    OFFICIAL_AGGREGATION,
    OFFICIAL_EPOCH,
    PROTOCOL_VERSION,
    RecordingAccumulator,
    aggregate_rows,
    macro_f1_from_confusion,
    resolve_majority_vote,
)

__all__ = [
    "OFFICIAL_AGGREGATION",
    "OFFICIAL_EPOCH",
    "PROTOCOL_VERSION",
    "RecordingAccumulator",
    "aggregate_rows",
    "macro_f1_from_confusion",
    "resolve_majority_vote",
]
