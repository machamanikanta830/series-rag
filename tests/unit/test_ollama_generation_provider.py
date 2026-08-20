"""Tests for the explicit, non-streaming Ollama HTTP generation provider."""

import json
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import pytest

from app.generation.ollama import OllamaGenerationError, OllamaGenerationProvider


class FakeHttpResponse:
    """Provide a context-managed HTTP response body without a real network call."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "FakeHttpResponse":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None

    def read(self) -> bytes:
        """Return the configured raw HTTP response body."""
        return self._body


def test_provider_sends_one_non_streaming_request_and_returns_generated_text() -> None:
    """The provider sends only Ollama's required fields to its generate endpoint."""
    provider = OllamaGenerationProvider(
        model="test-model",
        base_url="http://ollama.test:11434",
        timeout_seconds=12.5,
    )
    response = FakeHttpResponse(b'{"response": "Generated text."}')

    with patch("app.generation.ollama.urlopen", return_value=response) as open_url:
        generated_text = provider.generate("Completed prompt")

    request = open_url.call_args.args[0]
    assert request.full_url == "http://ollama.test:11434/api/generate"
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data) == {
        "model": "test-model",
        "prompt": "Completed prompt",
        "stream": False,
    }
    assert open_url.call_args.kwargs == {"timeout": 12.5}
    assert generated_text == "Generated text."


def test_provider_preserves_prompt_text_and_generated_text() -> None:
    """The adapter does not trim valid prompt or generated-response contents."""
    provider = OllamaGenerationProvider("test-model")
    response = FakeHttpResponse(b'{"response": "  Generated text.  "}')
    prompt = "  Completed prompt with source labels.  "

    with patch("app.generation.ollama.urlopen", return_value=response) as open_url:
        generated_text = provider.generate(prompt)

    request = open_url.call_args.args[0]
    assert json.loads(request.data)["prompt"] == prompt
    assert generated_text == "  Generated text.  "


def test_readiness_lists_models_without_generating_or_pulling() -> None:
    """The read-only tags endpoint confirms the configured model is available."""
    provider = OllamaGenerationProvider(
        model="test-model",
        base_url="http://ollama.test:11434",
        timeout_seconds=4.0,
    )
    response = FakeHttpResponse(
        b'{"models": [{"name": "test-model:latest", "model": "test-model"}]}'
    )

    with patch("app.generation.ollama.urlopen", return_value=response) as open_url:
        is_ready = provider.is_ready()

    request = open_url.call_args.args[0]
    assert request.full_url == "http://ollama.test:11434/api/tags"
    assert request.get_method() == "GET"
    assert request.data is None
    assert open_url.call_args.kwargs == {"timeout": 4.0}
    assert is_ready is True


def test_readiness_is_false_when_configured_model_is_missing() -> None:
    """A reachable Ollama service is insufficient without the selected model."""
    provider = OllamaGenerationProvider("expected-model")
    response = FakeHttpResponse(b'{"models": [{"name": "another-model:latest"}]}')

    with patch("app.generation.ollama.urlopen", return_value=response):
        assert provider.is_ready() is False


def test_readiness_is_false_when_ollama_is_unavailable() -> None:
    """Connection failures become a boolean state rather than generation errors."""
    provider = OllamaGenerationProvider("test-model")

    with patch(
        "app.generation.ollama.urlopen",
        side_effect=URLError("connection refused"),
    ):
        assert provider.is_ready() is False


@pytest.mark.parametrize("model", ["", "   "])
def test_provider_rejects_empty_model_names(model: str) -> None:
    """A local generation request needs a named Ollama model."""
    with pytest.raises(ValueError, match="model must not be empty"):
        OllamaGenerationProvider(model)


@pytest.mark.parametrize("base_url", ["", "   "])
def test_provider_rejects_empty_base_urls(base_url: str) -> None:
    """A local generation request needs a configured Ollama service URL."""
    with pytest.raises(ValueError, match="base_url must not be empty"):
        OllamaGenerationProvider("test-model", base_url=base_url)


@pytest.mark.parametrize("timeout_seconds", [0, -1, float("inf"), float("nan")])
def test_provider_rejects_invalid_timeouts(timeout_seconds: float) -> None:
    """Ollama requests require a finite positive timeout."""
    with pytest.raises(ValueError, match="timeout_seconds must be greater than zero"):
        OllamaGenerationProvider("test-model", timeout_seconds=timeout_seconds)


def test_provider_rejects_non_numeric_timeout() -> None:
    """Timeouts must be numbers so they can be passed to the HTTP client."""
    with pytest.raises(TypeError, match="timeout_seconds must be a number"):
        OllamaGenerationProvider("test-model", timeout_seconds="60")  # type: ignore[arg-type]


def test_provider_wraps_connection_failures() -> None:
    """Connection errors explain that the local Ollama service is unavailable."""
    provider = OllamaGenerationProvider("test-model", base_url="http://ollama.test")

    with (
        patch(
            "app.generation.ollama.urlopen",
            side_effect=URLError("connection refused"),
        ),
        pytest.raises(OllamaGenerationError, match="Unable to connect to Ollama"),
    ):
        provider.generate("Completed prompt")


@pytest.mark.parametrize("error", [TimeoutError(), URLError(TimeoutError())])
def test_provider_wraps_timeouts(error: TimeoutError | URLError) -> None:
    """Both timeout forms are translated into one clear provider-level error."""
    provider = OllamaGenerationProvider("test-model", timeout_seconds=3.0)

    with (
        patch("app.generation.ollama.urlopen", side_effect=error),
        pytest.raises(OllamaGenerationError, match="timed out after 3.0 seconds"),
    ):
        provider.generate("Completed prompt")


def test_provider_wraps_non_success_http_responses() -> None:
    """HTTP failures expose the status without leaking urllib types to callers."""
    provider = OllamaGenerationProvider("test-model")
    error = HTTPError(
        "http://localhost:11434/api/generate",
        503,
        "Service Unavailable",
        hdrs=None,
        fp=None,
    )

    with (
        patch("app.generation.ollama.urlopen", side_effect=error),
        pytest.raises(OllamaGenerationError, match="HTTP status 503"),
    ):
        provider.generate("Completed prompt")


def test_provider_rejects_invalid_json_responses() -> None:
    """Malformed response bytes cannot be treated as generated text."""
    provider = OllamaGenerationProvider("test-model")

    with (
        patch(
            "app.generation.ollama.urlopen",
            return_value=FakeHttpResponse(b"not JSON"),
        ),
        pytest.raises(OllamaGenerationError, match="invalid JSON"),
    ):
        provider.generate("Completed prompt")


@pytest.mark.parametrize(
    "response_body",
    [b"{}", b'{"response": null}', b'{"response": 42}'],
)
def test_provider_rejects_missing_response_text(response_body: bytes) -> None:
    """Ollama JSON must include a string response value."""
    provider = OllamaGenerationProvider("test-model")

    with (
        patch(
            "app.generation.ollama.urlopen",
            return_value=FakeHttpResponse(response_body),
        ),
        pytest.raises(OllamaGenerationError, match="did not include generated text"),
    ):
        provider.generate("Completed prompt")


@pytest.mark.parametrize("response_text", ["", "   ", "\n\t"])
def test_provider_rejects_empty_generated_responses(response_text: str) -> None:
    """A syntactically valid but blank Ollama response is not useful output."""
    provider = OllamaGenerationProvider("test-model")
    response_body = json.dumps({"response": response_text}).encode("utf-8")

    with (
        patch(
            "app.generation.ollama.urlopen",
            return_value=FakeHttpResponse(response_body),
        ),
        pytest.raises(OllamaGenerationError, match="empty generated response"),
    ):
        provider.generate("Completed prompt")
