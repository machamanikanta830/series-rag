"""Generate plain text through Ollama's local non-streaming HTTP endpoint."""

import json
from math import isfinite
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.generation.base import GenerationProvider

_GENERATE_PATH = "/api/generate"
_TAGS_PATH = "/api/tags"


class OllamaGenerationError(RuntimeError):
    """A clear application-level failure while requesting local Ollama text."""


class OllamaGenerationProvider(GenerationProvider):
    """Send completed prompts to one locally running Ollama model."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 60.0,
    ) -> None:
        """Store validated connection settings without making a network request."""
        self._model = _validate_non_empty_string(model, "model")
        self._base_url = _validate_non_empty_string(base_url, "base_url").rstrip("/")
        self._timeout_seconds = _validate_timeout(timeout_seconds)

    def is_ready(self) -> bool:
        """Return whether Ollama can list the configured model without generating."""
        request = Request(
            url=f"{self._base_url}{_TAGS_PATH}",
            method="GET",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                response_body = response.read()
        except (HTTPError, TimeoutError, URLError):
            return False

        return _ollama_response_has_model(response_body, self._model)

    def _generate(self, prompt: str) -> str:
        """Request one non-streaming Ollama completion for a validated prompt."""
        request_body = json.dumps(
            {"model": self._model, "prompt": prompt, "stream": False}
        ).encode("utf-8")
        request = Request(
            url=f"{self._base_url}{_GENERATE_PATH}",
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                response_body = response.read()
        except HTTPError as error:
            raise OllamaGenerationError(
                f"Ollama generation request failed with HTTP status {error.code}."
            ) from error
        except TimeoutError as error:
            raise OllamaGenerationError(
                "Ollama generation request timed out after "
                f"{self._timeout_seconds} seconds."
            ) from error
        except URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise OllamaGenerationError(
                    "Ollama generation request timed out after "
                    f"{self._timeout_seconds} seconds."
                ) from error
            raise OllamaGenerationError(
                f"Unable to connect to Ollama at {self._base_url}. "
                "Ensure the local Ollama service is running."
            ) from error

        return _parse_response(response_body)


def _validate_non_empty_string(value: str, name: str) -> str:
    """Return trimmed configuration text after rejecting blank values."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")

    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError(f"{name} must not be empty")

    return normalized_value


def _validate_timeout(timeout_seconds: float) -> float:
    """Return a finite, positive timeout value as a float."""
    is_number = isinstance(timeout_seconds, (int, float))
    if isinstance(timeout_seconds, bool) or not is_number:
        raise TypeError("timeout_seconds must be a number")
    if not isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")

    return float(timeout_seconds)


def _parse_response(response_body: bytes) -> str:
    """Extract one non-empty plain-text response from Ollama JSON bytes."""
    try:
        response_data = json.loads(response_body)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise OllamaGenerationError(
            "Ollama returned an invalid JSON response."
        ) from error

    if not isinstance(response_data, dict):
        raise OllamaGenerationError("Ollama returned JSON that is not an object.")

    generated_text = response_data.get("response")
    if not isinstance(generated_text, str):
        raise OllamaGenerationError("Ollama response did not include generated text.")
    if not generated_text.strip():
        raise OllamaGenerationError("Ollama returned an empty generated response.")

    return generated_text


def _ollama_response_has_model(response_body: bytes, model: str) -> bool:
    """Check the read-only tags response for the configured model name."""
    try:
        response_data = json.loads(response_body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False

    if not isinstance(response_data, dict):
        return False
    models = response_data.get("models")
    if not isinstance(models, list):
        return False

    accepted_names = {model, f"{model}:latest"}
    for model_data in models:
        if not isinstance(model_data, dict):
            continue
        reported_names = (model_data.get("name"), model_data.get("model"))
        if any(name in accepted_names for name in reported_names):
            return True
    return False
