from __future__ import annotations

import itertools

import numpy as np

from recording_protocol import RecordingAccumulator, resolve_majority_vote


def test_unique_majority_ignores_probability_sum() -> None:
    pred, tied, decisive = resolve_majority_vote([3, 2, 0], [0.1, 100.0, 0.0])
    assert (pred, tied, decisive) == (0, False, False)


def test_probability_tiebreak_is_permutation_invariant() -> None:
    windows = [np.array([0.9, 0.1]), np.array([0.4, 0.6]), np.array([0.2, 0.8]), np.array([0.7, 0.3])]
    predictions = set()
    for order in itertools.permutations(windows):
        acc = RecordingAccumulator(2)
        for probs in order:
            acc.add("recording", 1, probs)
        predictions.add(acc.predictions()[1][0])
    assert predictions == {0}


def test_exact_probability_tie_uses_smallest_class_id() -> None:
    pred, tied, decisive = resolve_majority_vote([2, 2, 0], [1.0, 1.0, 0.0])
    assert (pred, tied, decisive) == (0, True, False)
