from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, ContextManager, Iterator, cast

from openai import AsyncOpenAI


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ChatResponse:
    content: str


class TokenUsageLedger:
    """Small per-stage token usage accumulator."""

    def __init__(self) -> None:
        self._stage = "default"
        self._stages: dict[str, dict[str, int]] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        previous = self._stage
        self._stage = name
        self._stages.setdefault(name, _empty_usage())
        try:
            yield
        finally:
            self._stage = previous

    def record(self, usage: Any) -> None:
        row = self._stages.setdefault(self._stage, _empty_usage())
        row["calls"] += 1
        if usage is None:
            row["missing_calls"] += 1
            return

        input_tokens = _optional_int(getattr(usage, "prompt_tokens", None))
        output_tokens = _optional_int(getattr(usage, "completion_tokens", None))
        total_tokens = _optional_int(getattr(usage, "total_tokens", None))
        if total_tokens is None:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)

        row["input_tokens"] += input_tokens or 0
        row["output_tokens"] += output_tokens or 0
        row["total_tokens"] += total_tokens or 0

    def summary(self) -> dict[str, Any]:
        stages = {stage: dict(values) for stage, values in sorted(self._stages.items())}
        total = _empty_usage()
        for values in stages.values():
            for key in total:
                total[key] += int(values.get(key, 0))
        return {"total": total, "stages": stages}


class OpenAICompatLLM:
    """Minimal OpenAI-compatible chat client."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature
        self._usage = TokenUsageLedger()

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
        }
        kwargs.update(extra)
        response = await self._client.chat.completions.create(**kwargs)
        self._usage.record(getattr(response, "usage", None))
        return ChatResponse(content=response.choices[0].message.content or "")

    def stage(self, name: str) -> ContextManager[None]:
        return self._usage.stage(name)

    def usage_summary(self) -> dict[str, Any]:
        return self._usage.summary()

    async def close(self) -> None:
        await self._client.close()


class OpenAIEmbedder:
    """Minimal OpenAI-compatible embedding client."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._usage = TokenUsageLedger()

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self._model, input=list(texts)
        )
        self._usage.record(getattr(response, "usage", None))
        return [item.embedding for item in response.data]

    def stage(self, name: str) -> ContextManager[None]:
        return self._usage.stage(name)

    def usage_summary(self) -> dict[str, Any]:
        return self._usage.summary()

    async def close(self) -> None:
        await self._client.close()


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

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        with ExitStack() as stack:
            for client in self._clients.values():
                stack.enter_context(client.stage(name))
            yield

    def usage_summary(self) -> dict[str, object]:
        return {
            name: client.usage_summary()
            for name, client in sorted(self._clients.items())
        }

    async def close(self) -> None:
        for client in self._clients.values():
            await client.close()


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _empty_usage() -> dict[str, int]:
    return {
        "calls": 0,
        "missing_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
