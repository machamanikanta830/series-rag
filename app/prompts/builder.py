"""Build deterministic, source-grounded prompts from context and questions."""

_PROMPT_PREFIX = """You are a source-grounded assistant.

Instructions:
- Answer only from the provided context.
- Do not use outside knowledge.
- Cite the relevant source labels from the context.
- If the context is insufficient to support an answer, say so clearly.

Context:
"""


class PromptBuilder:
    """Combine validated context and a question into a plain-text prompt."""

    def build(self, question: str, context: str) -> str:
        """Return a deterministic prompt without making a model request."""
        normalized_question = _validate_and_strip(question, "question")
        normalized_context = _validate_and_strip(context, "context")

        return (
            f"{_PROMPT_PREFIX}{normalized_context}\n\n"
            f"Question:\n{normalized_question}\n\nAnswer:"
        )


def _validate_and_strip(value: str, name: str) -> str:
    """Reject non-string or blank prompt inputs and trim outer whitespace."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")

    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError(f"{name} must not be empty")

    return normalized_value
