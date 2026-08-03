from recording_protocol import OFFICIAL_AGGREGATION, RecordingAccumulator, resolve_majority_vote


def test_probability_breaks_a_vote_tie() -> None:
    prediction, vote_tie, probability_decisive = resolve_majority_vote(
        [1, 1, 0], [0.7, 1.2, 0.1]
    )
    assert prediction == 1
    assert vote_tie is True
    assert probability_decisive is True


def test_exact_tie_uses_smallest_class_id() -> None:
    prediction, vote_tie, probability_decisive = resolve_majority_vote(
        [1, 1, 0], [1.0, 1.0, 0.0]
    )
    assert prediction == 0
    assert vote_tie is True
    assert probability_decisive is False


def test_accumulator_reports_official_protocol() -> None:
    accumulator = RecordingAccumulator(3)
    accumulator.add("rec-a", 1, [0.1, 0.8, 0.1])
    accumulator.add("rec-a", 1, [0.1, 0.7, 0.2])
    metrics = accumulator.metrics()
    assert metrics["aggregation"] == OFFICIAL_AGGREGATION
    assert metrics["num_recordings"] == 1
    assert metrics["recording_accuracy"] == 1.0
