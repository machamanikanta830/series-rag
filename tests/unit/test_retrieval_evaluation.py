"""Tests for small, deterministic retrieval-ranking metrics."""

import pytest

from app.evaluation.retrieval import (
    RetrievalEvaluationCase,
    RetrievalEvaluationResult,
    evaluate_case,
    evaluate_cases,
    summarize_results,
)
from app.models import Chunk, SearchResult


def _result(chunk_id: str, score: float) -> SearchResult:
    """Create a deterministic immutable retrieval result."""
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="document-1",
            source_name="source.txt",
            text=f"Text for {chunk_id}",
            chunk_index=0,
            start_word=0,
            end_word=3,
            metadata={"topic": "test"},
        ),
        score=score,
    )


class FakeRetriever:
    """Return predetermined results while recording evaluation requests."""

    def __init__(self, results_by_query: dict[str, list[SearchResult]]) -> None:
        self._results_by_query = results_by_query
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return the configured ranked result list without modifying it."""
        self.calls.append((query, top_k))
        return self._results_by_query[query]


def test_expected_chunk_at_rank_one_has_perfect_single_case_metrics() -> None:
    """The first result produces every hit metric and reciprocal rank of one."""
    case = RetrievalEvaluationCase("query", ("expected",), top_k=3)
    retriever = FakeRetriever({"query": [_result("expected", 0.9)]})

    result = evaluate_case(retriever, case)  # type: ignore[arg-type]

    assert retriever.calls == [("query", 3)]
    assert result.has_hit
    assert result.first_relevant_rank == 1
    assert result.hit_at_1
    assert result.hit_at_3
    assert result.hit_at_k
    assert result.reciprocal_rank == 1.0


def test_expected_chunk_at_rank_two_has_reciprocal_rank_one_half() -> None:
    """A relevant result in the top three is not a hit at one."""
    case = RetrievalEvaluationCase("query", ("expected",), top_k=3)
    retriever = FakeRetriever(
        {"query": [_result("other", 0.9), _result("expected", 0.7)]}
    )

    result = evaluate_case(retriever, case)  # type: ignore[arg-type]

    assert result.first_relevant_rank == 2
    assert not result.hit_at_1
    assert result.hit_at_3
    assert result.hit_at_k
    assert result.reciprocal_rank == 0.5


def test_absent_expected_chunk_has_no_hit_and_zero_reciprocal_rank() -> None:
    """No matching chunk produces zero-valued ranking metrics."""
    case = RetrievalEvaluationCase("query", ("expected",), top_k=3)
    retriever = FakeRetriever({"query": [_result("other", 0.9)]})

    result = evaluate_case(retriever, case)  # type: ignore[arg-type]

    assert not result.has_hit
    assert result.first_relevant_rank is None
    assert not result.hit_at_1
    assert not result.hit_at_3
    assert not result.hit_at_k
    assert result.reciprocal_rank == 0.0


def test_multiple_acceptable_chunk_ids_use_the_first_matching_rank() -> None:
    """Any expected ID is relevant, with the earliest one defining the metrics."""
    case = RetrievalEvaluationCase(
        "query",
        ("first-acceptable", "second-acceptable"),
        3,
    )
    retriever = FakeRetriever(
        {
            "query": [
                _result("other", 0.9),
                _result("second-acceptable", 0.8),
                _result("first-acceptable", 0.7),
            ]
        }
    )

    result = evaluate_case(retriever, case)  # type: ignore[arg-type]

    assert result.first_relevant_rank == 2
    assert result.reciprocal_rank == 0.5


def test_evaluation_preserves_the_retriever_result_order() -> None:
    """Evaluation reads ranks without sorting or mutating the result list."""
    results = [_result("second", 0.8), _result("expected", 0.7)]
    case = RetrievalEvaluationCase("query", ("expected",), top_k=2)
    retriever = FakeRetriever({"query": results})

    evaluate_case(retriever, case)  # type: ignore[arg-type]

    assert [result.chunk.chunk_id for result in results] == ["second", "expected"]


def test_aggregate_summary_calculates_hit_rates_and_mean_reciprocal_rank() -> None:
    """Aggregate metrics average one rank-one hit, one rank-two hit, and one miss."""
    cases = [
        RetrievalEvaluationCase("first", ("expected-1",), 3),
        RetrievalEvaluationCase("second", ("expected-2",), 3),
        RetrievalEvaluationCase("third", ("expected-3",), 3),
    ]
    retriever = FakeRetriever(
        {
            "first": [_result("expected-1", 0.9)],
            "second": [_result("other", 0.9), _result("expected-2", 0.8)],
            "third": [_result("other", 0.9)],
        }
    )

    summary = evaluate_cases(retriever, cases)  # type: ignore[arg-type]

    assert summary.case_count == 3
    assert summary.hit_rate_at_1 == pytest.approx(1 / 3)
    assert summary.hit_rate_at_3 == pytest.approx(2 / 3)
    assert summary.hit_rate_at_k == pytest.approx(2 / 3)
    assert summary.mean_reciprocal_rank == pytest.approx(0.5)


def test_empty_evaluation_case_inputs_are_rejected() -> None:
    """A case needs an expected ID, and an aggregate needs at least one case."""
    with pytest.raises(ValueError, match="expected_chunk_ids must not be empty"):
        RetrievalEvaluationCase("query", (), top_k=3)
    with pytest.raises(ValueError, match="evaluation cases must not be empty"):
        evaluate_cases(FakeRetriever({}), [])  # type: ignore[arg-type]


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_case_top_k_is_rejected(top_k: int) -> None:
    """Each evaluation case must request at least one retrieved result."""
    with pytest.raises(ValueError, match="greater than zero"):
        RetrievalEvaluationCase("query", ("expected",), top_k)


def test_empty_summary_results_are_rejected() -> None:
    """Aggregate metrics are undefined without evaluated cases."""
    with pytest.raises(ValueError, match="evaluation results must not be empty"):
        summarize_results([])


def test_hit_at_k_uses_the_case_limit() -> None:
    """A rank-two match is a hit when the case asks for two results."""
    case = RetrievalEvaluationCase("query", ("expected",), top_k=2)
    retriever = FakeRetriever(
        {"query": [_result("other", 0.9), _result("expected", 0.8)]}
    )

    result = evaluate_case(retriever, case)  # type: ignore[arg-type]

    assert result.hit_at_k


def test_summary_results_are_immutable_data() -> None:
    """The result model remains a frozen value object for callers to inspect."""
    case = RetrievalEvaluationCase("query", ("expected",), top_k=1)
    result = RetrievalEvaluationResult(
        case=case,
        has_hit=True,
        first_relevant_rank=1,
        hit_at_1=True,
        hit_at_3=True,
        hit_at_k=True,
        reciprocal_rank=1.0,
    )

    assert result.case is case
