"""Define the minimal interface for generating text from a completed prompt."""

from abc import ABC, abstractmethod


class GenerationProvider(ABC):
    """Generate plain text without exposing provider-specific request types."""

    def generate(self, prompt: str) -> str:
        """Validate one prompt and return the provider's generated plain text."""
        validated_prompt = validate_prompt(prompt)
        generated_text = self._generate(validated_prompt)

        if not isinstance(generated_text, str):
            raise TypeError("generation providers must return a string")

        return generated_text

    @abstractmethod
    def _generate(self, prompt: str) -> str:
        """Generate text for one validated prompt without changing its contents."""


def validate_prompt(prompt: str) -> str:
    """Return a meaningful string prompt without changing its formatting."""
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    return prompt
