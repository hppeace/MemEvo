from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from openai import AsyncOpenAI

_USAGE_FIELDS = (
    "calls",
    "missing_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
)


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ChatResponse:
    content: str


class TokenUsageLedger:
    """Token usage accumulator for one model."""

    def __init__(self) -> None:
        self._usage = _empty_usage()

    def record(self, usage: Any) -> None:
        self._usage["calls"] += 1
        if usage is None:
            self._usage["missing_calls"] += 1
            return

        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = getattr(usage, "total_tokens", None)

        self._usage["input_tokens"] += input_tokens
        self._usage["output_tokens"] += output_tokens
        self._usage["total_tokens"] += (
            int(total_tokens)
            if total_tokens is not None
            else input_tokens + output_tokens
        )

    def summary(self) -> dict[str, int]:
        return dict(self._usage)


class _OpenAIClient:
    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._usage = TokenUsageLedger()

    def usage_summary(self) -> dict[str, int]:
        return self._usage.summary()

    async def close(self) -> None:
        await self._client.close()


class OpenAICompatLLM(_OpenAIClient):
    """Minimal OpenAI-compatible chat client."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(api_key, base_url)
        self._model = model
        self._temperature = temperature

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        **extra: Any,
    ) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "model": model or self._model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": self._temperature if temperature is None else temperature,
            **extra,
        }
        response = await self._client.chat.completions.create(**kwargs)
        self._usage.record(getattr(response, "usage", None))
        return ChatResponse(content=response.choices[0].message.content or "")


class OpenAIEmbedder(_OpenAIClient):
    """Minimal OpenAI-compatible embedding client."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        super().__init__(api_key, base_url)
        self._model = model

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self._model, input=list(texts)
        )
        self._usage.record(getattr(response, "usage", None))
        return [item.embedding for item in response.data]


ModelClient = OpenAICompatLLM | OpenAIEmbedder


class ModelPool:
    """Named model clients with shared experiment-stage accounting."""

    def __init__(self, clients: dict[str, ModelClient]) -> None:
        self._clients = clients

    def llm(self, name: str) -> OpenAICompatLLM:
        client = self._clients.get(name)
        if client is None:
            raise KeyError(f"LLM model '{name}' is not configured")
        return cast(OpenAICompatLLM, client)

    def embedder(self, name: str) -> OpenAIEmbedder:
        client = self._clients.get(name)
        if client is None:
            raise KeyError(f"Embedding model '{name}' is not configured")
        return cast(OpenAIEmbedder, client)

    def usage_summary(self) -> dict[str, dict[str, int]]:
        total = _empty_usage()
        usage_by_model = {}
        for name, client in sorted(self._clients.items()):
            usage = client.usage_summary()
            usage_by_model[name] = usage
            for field in _USAGE_FIELDS:
                total[field] += usage[field]
        return {"total": total, **usage_by_model}

    async def close(self) -> None:
        for client in self._clients.values():
            await client.close()


def _empty_usage() -> dict[str, int]:
    return dict.fromkeys(_USAGE_FIELDS, 0)
