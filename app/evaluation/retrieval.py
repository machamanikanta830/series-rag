"""Evaluate whether a retriever ranks expected chunks near the top."""

from dataclasses import dataclass

from app.retrieval.retriever import SemanticRetriever


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationCase:
    """One query and the chunk IDs considered relevant to it."""

    query: str
    expected_chunk_ids: tuple[str, ...]
    top_k: int

    def __post_init__(self) -> None:
        """Validate the smallest useful, immutable retrieval evaluation case."""
        if not isinstance(self.query, str):
            raise TypeError("evaluation query must be a string")
        if not self.query.strip():
            raise ValueError("evaluation query must not be empty")
        expected_chunk_ids = tuple(self.expected_chunk_ids)
        if not expected_chunk_ids:
            raise ValueError("expected_chunk_ids must not be empty")
        if any(
            not isinstance(chunk_id, str) or not chunk_id
            for chunk_id in expected_chunk_ids
        ):
            raise ValueError("expected_chunk_ids must contain non-empty strings")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        object.__setattr__(self, "expected_chunk_ids", expected_chunk_ids)


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationResult:
    """Ranking metrics for one retrieval evaluation case."""

    case: RetrievalEvaluationCase
    has_hit: bool
    first_relevant_rank: int | None
    hit_at_1: bool
    hit_at_3: bool
    hit_at_k: bool
    reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationSummary:
    """Aggregate hit rates and reciprocal rank across evaluation cases."""

    case_count: int
    hit_rate_at_1: float
    hit_rate_at_3: float
    hit_rate_at_k: float
    mean_reciprocal_rank: float


def evaluate_case(
    retriever: SemanticRetriever,
    case: RetrievalEvaluationCase,
) -> RetrievalEvaluationResult:
    """Evaluate the first relevant chunk rank in one retriever result list."""
    results = retriever.retrieve(case.query, top_k=case.top_k)
    expected_chunk_ids = set(case.expected_chunk_ids)
    first_relevant_rank = next(
        (
            rank
            for rank, result in enumerate(results, start=1)
            if result.chunk.chunk_id in expected_chunk_ids
        ),
        None,
    )
    has_hit = first_relevant_rank is not None
    reciprocal_rank = 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank

    return RetrievalEvaluationResult(
        case=case,
        has_hit=has_hit,
        first_relevant_rank=first_relevant_rank,
        hit_at_1=first_relevant_rank == 1,
        hit_at_3=first_relevant_rank is not None and first_relevant_rank <= 3,
        hit_at_k=first_relevant_rank is not None and first_relevant_rank <= case.top_k,
        reciprocal_rank=reciprocal_rank,
    )


def evaluate_cases(
    retriever: SemanticRetriever,
    cases: list[RetrievalEvaluationCase],
) -> RetrievalEvaluationSummary:
    """Run evaluation cases and return their aggregate ranking metrics."""
    if not cases:
        raise ValueError("evaluation cases must not be empty")

    results = [evaluate_case(retriever, case) for case in cases]
    return summarize_results(results)


def summarize_results(
    results: list[RetrievalEvaluationResult],
) -> RetrievalEvaluationSummary:
    """Calculate aggregate metrics from already evaluated cases."""
    if not results:
        raise ValueError("evaluation results must not be empty")

    case_count = len(results)
    return RetrievalEvaluationSummary(
        case_count=case_count,
        hit_rate_at_1=sum(result.hit_at_1 for result in results) / case_count,
        hit_rate_at_3=sum(result.hit_at_3 for result in results) / case_count,
        hit_rate_at_k=sum(result.hit_at_k for result in results) / case_count,
        mean_reciprocal_rank=sum(result.reciprocal_rank for result in results)
        / case_count,
    )
