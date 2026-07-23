"""Tests for provider-neutral, deterministic text generation."""

import pytest

from app.generation.fake import FakeGenerationProvider


def test_fake_provider_returns_its_configured_response() -> None:
    """A valid prompt produces the fixed plain-text response."""
    provider = FakeGenerationProvider("AWS secures the cloud infrastructure.")

    response = provider.generate("What does AWS secure?")

    assert response == "AWS secures the cloud infrastructure."


def test_fake_provider_response_is_deterministic() -> None:
    """The same fake configuration returns identical text for every call."""
    provider = FakeGenerationProvider("Fixed response.")

    first_response = provider.generate("First prompt")
    second_response = provider.generate("Second prompt")

    assert first_response == "Fixed response."
    assert second_response == "Fixed response."


def test_fake_provider_preserves_the_received_prompt_exactly() -> None:
    """Prompt formatting is left intact for a future provider implementation."""
    provider = FakeGenerationProvider("Response.")
    prompt = "  Context:\n[Source: notes.md | Chunk 0]\n\nQuestion: Why?  "

    provider.generate(prompt)

    assert provider.received_prompts == [prompt]


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t"])
def test_fake_provider_rejects_empty_or_whitespace_only_prompts(prompt: str) -> None:
    """Generation needs a prompt containing visible text."""
    provider = FakeGenerationProvider("Response.")

    with pytest.raises(ValueError, match="prompt must not be empty"):
        provider.generate(prompt)

    assert provider.received_prompts == []


def test_fake_provider_rejects_non_string_prompts() -> None:
    """Only a plain string can be passed to the provider-neutral interface."""
    provider = FakeGenerationProvider("Response.")

    with pytest.raises(TypeError, match="prompt must be a string"):
        provider.generate(42)  # type: ignore[arg-type]

    assert provider.received_prompts == []


def test_fake_provider_records_repeated_calls_independently() -> None:
    """Each call is recorded once without reusing or overwriting prior prompts."""
    provider = FakeGenerationProvider("Response.")

    provider.generate("First prompt")
    provider.generate("Second prompt")

    assert provider.received_prompts == ["First prompt", "Second prompt"]


def test_fake_provider_instances_do_not_share_received_prompts() -> None:
    """The fake has no global state between independently created providers."""
    first_provider = FakeGenerationProvider("First response.")
    second_provider = FakeGenerationProvider("Second response.")

    first_provider.generate("First prompt")

    assert first_provider.received_prompts == ["First prompt"]
    assert second_provider.received_prompts == []


def test_fake_provider_rejects_non_string_configured_responses() -> None:
    """Configured output must remain compatible with the plain-text contract."""
    with pytest.raises(TypeError, match="response must be a string"):
        FakeGenerationProvider(42)  # type: ignore[arg-type]
