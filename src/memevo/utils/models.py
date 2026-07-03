import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from openai import AsyncOpenAI

_FIELDS = ("calls", "missing_calls", "input_tokens", "output_tokens", "total_tokens")
_STAGE: ContextVar[str] = ContextVar("stage", default="unscoped")


def _empty() -> dict[str, int]:
    return dict.fromkeys(_FIELDS, 0)


class Usage:
    """Token usage grouped by configured model and benchmark stage."""

    def __init__(self, model_names: Iterable[str] = ()) -> None:
        self._models = {name: _empty() for name in model_names}
        self._stages: dict[str, dict[str, int]] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self._stages.setdefault(name, _empty())
        token = _STAGE.set(name)
        try:
            yield
        finally:
            _STAGE.reset(token)

    def record(self, model: str, usage: Any) -> None:
        rows = (
            self._models.setdefault(model, _empty()),
            self._stages.setdefault(_STAGE.get(), _empty()),
        )
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(
            getattr(usage, "total_tokens", 0) or input_tokens + output_tokens
        )
        for row in rows:
            row["calls"] += 1
            if usage is None:
                row["missing_calls"] += 1
                continue
            row["input_tokens"] += input_tokens
            row["output_tokens"] += output_tokens
            row["total_tokens"] += total_tokens

    def summary(self) -> dict[str, Any]:
        total = {
            field: sum(row[field] for row in self._models.values()) for field in _FIELDS
        }
        return {
            "total": total,
            **{name: dict(row) for name, row in sorted(self._models.items())},
            "stages": {name: dict(row) for name, row in sorted(self._stages.items())},
        }


class _Model:
    def __init__(
        self,
        name: str,
        config: Mapping[str, Any],
        usage: Usage,
    ) -> None:
        self.name = name
        self.model = str(config["model"])
        self.options = dict(config.get("options", {}))
        self.usage = usage
        self.client = AsyncOpenAI(
            api_key=os.environ[str(config["api_key_env"])],
            base_url=config.get("base_url"),
        )

    async def close(self) -> None:
        await self.client.close()


class LLM(_Model):
    """Small async client for OpenAI-compatible chat APIs."""

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        **options: Any,
    ) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            **(self.options | options),
        )
        self.usage.record(self.name, response.usage)
        return response.choices[0].message.content or ""


class Embedder(_Model):
    """Small async client for OpenAI-compatible embedding APIs."""

    async def embed(
        self,
        texts: Sequence[str],
        **options: Any,
    ) -> list[list[float]]:
        response = await self.client.embeddings.create(
            model=self.model,
            input=list(texts),
            **(self.options | options),
        )
        self.usage.record(self.name, response.usage)
        return [
            item.embedding
            for item in sorted(response.data, key=lambda item: item.index)
        ]
