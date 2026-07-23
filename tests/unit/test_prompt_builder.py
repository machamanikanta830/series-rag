"""Tests for deterministic, source-grounded prompt construction."""

import pytest

from app.prompts.builder import PromptBuilder


def test_builder_constructs_a_source_grounded_prompt() -> None:
    """A valid question and context produce the documented plain-text layout."""
    prompt = PromptBuilder().build(
        "Who secures cloud infrastructure?",
        "[Source: aws.md | Chunk 0]\n\nAWS secures the infrastructure.",
    )

    assert prompt == (
        "You are a source-grounded assistant.\n\n"
        "Instructions:\n"
        "- Answer only from the provided context.\n"
        "- Do not use outside knowledge.\n"
        "- Cite the relevant source labels from the context.\n"
        "- If the context is insufficient to support an answer, say so clearly.\n\n"
        "Context:\n"
        "[Source: aws.md | Chunk 0]\n\nAWS secures the infrastructure.\n\n"
        "Question:\nWho secures cloud infrastructure?\n\nAnswer:"
    )


def test_builder_output_is_deterministic() -> None:
    """The same validated inputs always produce exactly the same prompt text."""
    builder = PromptBuilder()

    first_prompt = builder.build("What is IAM?", "[Source: notes.md | Chunk 1]\n\nIAM.")
    second_prompt = builder.build(
        "What is IAM?", "[Source: notes.md | Chunk 1]\n\nIAM."
    )

    assert first_prompt == second_prompt


def test_builder_trims_outer_whitespace_from_question_and_context() -> None:
    """Only leading and trailing whitespace is normalized for each input."""
    prompt = PromptBuilder().build(
        "  What is shared responsibility?  ",
        "\n [Source: aws.md | Chunk 0]\n\nAWS text. \n",
    )

    assert "Question:\nWhat is shared responsibility?\n\nAnswer:" in prompt
    assert "Context:\n[Source: aws.md | Chunk 0]\n\nAWS text.\n\nQuestion:" in prompt


def test_builder_preserves_source_labels_in_context() -> None:
    """Attribution labels remain available for a future model to cite."""
    context = (
        "[Source: aws-notes.md | Chunk 2]\n\nAWS text.\n\n---\n\n"
        "[Source: security.md | Chunk 4]\n\nSecurity text."
    )

    prompt = PromptBuilder().build("What is protected?", context)

    assert "[Source: aws-notes.md | Chunk 2]" in prompt
    assert "[Source: security.md | Chunk 4]" in prompt


def test_builder_includes_explicit_grounding_instructions() -> None:
    """The template constrains future answer generation to supplied evidence."""
    prompt = PromptBuilder().build(
        "Question?", "[Source: source.md | Chunk 0]\n\nText."
    )

    assert "Answer only from the provided context." in prompt
    assert "Do not use outside knowledge." in prompt


def test_builder_includes_insufficient_context_instruction() -> None:
    """The template states what a future model should do when unsupported."""
    prompt = PromptBuilder().build(
        "Question?", "[Source: source.md | Chunk 0]\n\nText."
    )

    assert (
        "If the context is insufficient to support an answer, say so clearly." in prompt
    )


def test_builder_includes_citation_instruction() -> None:
    """The template explicitly asks a future model to use source labels."""
    prompt = PromptBuilder().build(
        "Question?", "[Source: source.md | Chunk 0]\n\nText."
    )

    assert "Cite the relevant source labels from the context." in prompt


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_builder_rejects_empty_or_whitespace_only_questions(question: str) -> None:
    """A prompt cannot be built without a meaningful user question."""
    with pytest.raises(ValueError, match="question must not be empty"):
        PromptBuilder().build(question, "[Source: source.md | Chunk 0]\n\nText.")


def test_builder_rejects_non_string_questions() -> None:
    """Question validation rejects values that cannot be safely normalized."""
    with pytest.raises(TypeError, match="question must be a string"):
        PromptBuilder().build(42, "[Source: source.md | Chunk 0]\n\nText.")  # type: ignore[arg-type]


@pytest.mark.parametrize("context", ["", "   ", "\n\t"])
def test_builder_rejects_empty_or_whitespace_only_context(context: str) -> None:
    """Grounding context must contain meaningful source text."""
    with pytest.raises(ValueError, match="context must not be empty"):
        PromptBuilder().build("Question?", context)


def test_builder_rejects_non_string_context() -> None:
    """Context validation rejects values that cannot preserve attribution."""
    with pytest.raises(TypeError, match="context must be a string"):
        PromptBuilder().build("Question?", ["context"])  # type: ignore[arg-type]
