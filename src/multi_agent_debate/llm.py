from __future__ import annotations

import asyncio

from litellm import acompletion


class TemporaryLLMError(RuntimeError):
    """Raised when the provider is temporarily unavailable."""


class ProviderSetupError(RuntimeError):
    """Raised when a local or remote provider needs setup before use."""


class LLMClient:
    def __init__(self, model: str, temperature: float = 0.4, max_tokens: int = 1200) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def complete(self, messages: list[dict[str, str]]) -> str:
        if self.model == "mock/debate-dev":
            return self._mock_complete(messages)

        content_parts: list[str] = []
        working_messages = list(messages)
        for _ in range(3):
            content, finish_reason = await self._complete_once(working_messages)
            if content:
                content_parts.append(content)
            if not self._needs_continuation(content, finish_reason):
                break
            working_messages = messages + [
                {"role": "assistant", "content": "\n\n".join(content_parts)},
                {
                    "role": "user",
                    "content": (
                        "Continue exactly where you stopped. Do not restart, summarize, or "
                        "repeat previous text. Finish the remaining sections."
                    ),
                },
            ]

        return "\n\n".join(part for part in content_parts if part).strip()

    async def _complete_once(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = await acompletion(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                choice = response.choices[0]
                content = choice.message.content
                finish_reason = getattr(choice, "finish_reason", "") or ""
                return (content.strip() if content else "", str(finish_reason))
            except Exception as exc:
                last_error = exc
                if self._is_provider_setup_error(exc):
                    break
                if not self._is_temporary_error(exc) or attempt == 2:
                    break
                await asyncio.sleep(2**attempt)

        if last_error and self._is_provider_setup_error(last_error):
            raise ProviderSetupError(
                f"{self.model} could not be reached. If this is an Ollama model, install "
                "Ollama, start the Ollama server, and pull the selected model."
            ) from last_error
        if last_error and self._is_temporary_error(last_error):
            raise TemporaryLLMError(
                f"{self.model} is temporarily unavailable or overloaded. "
                "Try again shortly, reduce the number of agents/rounds, or switch models."
            ) from last_error
        raise last_error or RuntimeError("LLM call failed without an exception.")

    @staticmethod
    def _needs_continuation(content: str, finish_reason: str) -> bool:
        normalized_reason = finish_reason.lower()
        if normalized_reason in {"length", "max_tokens", "max_output_tokens"}:
            return True
        if not content:
            return False
        stripped = content.rstrip()
        if stripped.endswith(("...", "…")):
            return True
        if stripped[-1:] not in {".", "!", "?", ")", "]", "`"}:
            return True
        headings = ["## Final Answer", "## Accepted Critiques", "## Change Log"]
        return "## Final Answer" in content and not any(
            heading in content for heading in headings[1:]
        )

    @staticmethod
    def _is_temporary_error(exc: Exception) -> bool:
        message = str(exc).lower()
        temporary_signals = [
            "503",
            "unavailable",
            "high demand",
            "temporarily",
            "rate limit",
            "overloaded",
        ]
        return any(signal in message for signal in temporary_signals)

    @staticmethod
    def _is_provider_setup_error(exc: Exception) -> bool:
        message = str(exc).lower()
        setup_signals = [
            "connection refused",
            "connection error",
            "could not connect",
            "failed to establish a new connection",
            "ollama",
            "localhost:11434",
        ]
        return any(signal in message for signal in setup_signals)

    @staticmethod
    def _mock_complete(messages: list[dict[str, str]]) -> str:
        user_message = next(
            (message["content"] for message in reversed(messages) if message["role"] == "user"),
            "",
        )
        excerpt = user_message.strip().splitlines()[-1] if user_message.strip() else "the query"
        return (
            "Mock debate response for development.\n\n"
            f"- Focus: {excerpt[:180]}\n"
            "- Analyst: clarify the goal, constraints, and assumptions.\n"
            "- Skeptic: check for missing evidence, tradeoffs, and failure modes.\n"
            "- Synthesizer: merge the strongest points into a concise answer.\n\n"
            "Final mock answer: this placeholder lets you test UI, storage, and debate flow "
            "without spending provider quota."
        )
