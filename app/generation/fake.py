"""A deterministic generation provider for tests and local demonstrations."""

from dataclasses import dataclass, field

from app.generation.base import GenerationProvider


@dataclass(slots=True)
class FakeGenerationProvider(GenerationProvider):
    """Return one configured response while recording prompts per instance."""

    response: str
    received_prompts: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Ensure the configured fake response is always plain text."""
        if not isinstance(self.response, str):
            raise TypeError("response must be a string")

    def _generate(self, prompt: str) -> str:
        """Record one validated prompt and return the fixed response unchanged."""
        self.received_prompts.append(prompt)
        return self.response
