from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Iterator

from openai import AsyncOpenAI
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatResponse(BaseModel):
    content: str


class TokenUsageLedger:
    """Small per-stage token usage accumulator for experiment accounting."""

    def __init__(self) -> None:
        self._stage = "default"
        self._stages: dict[str, dict[str, int]] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        previous = self._stage
        self._stage = name
        try:
            yield
        finally:
            self._stage = previous

    def record(self, usage: Any) -> None:
        row = self._stages.setdefault(
            self._stage,
            {
                "calls": 0,
                "missing_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        )
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
        total = {
            "calls": 0,
            "missing_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        for values in stages.values():
            for key in total:
                total[key] += int(values.get(key, 0))
        return {"total": total, "stages": stages}
    

class OpenAICompatLLM:
    """Minimal OpenAI-compatible client."""

    def __init__(self, api_key: str, base_url: str, model: str, temperature: float = 0.0) -> None:
        self._client = AsyncOpenAI({"api_key": api_key, "base_url": base_url})
        self._model = model
        self._temperature = temperature
        self._usage = TokenUsageLedger()

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        response_format: type[BaseModel] | None = None,
        **extra: Any,
    ) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "model": model or self._model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "temperature": self._temperature if temperature is None else temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = {"type": "json_object"}
        kwargs.update(extra)
        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        self._usage.record(getattr(resp, "usage", None))
        return ChatResponse(content=choice.message.content or "")

    def stage(self, name: str) -> Iterator[None]:
        return self._usage.stage(name)

    def usage_summary(self) -> dict[str, Any]:
        return self._usage.summary()


class OpenAIEmbedder:
    """Minimal embedding client."""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self._client = AsyncOpenAI({"api_key": api_key, "base_url": base_url})
        self._model = model
        self._usage = TokenUsageLedger()

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(
            model=self._model, input=list(texts)
        )
        self._usage.record(getattr(resp, "usage", None))
        return [item.embedding for item in resp.data]

    def stage(self, name: str) -> Iterator[None]:
        return self._usage.stage(name)

    def usage_summary(self) -> dict[str, Any]:
        return self._usage.summary()


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)