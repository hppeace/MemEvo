from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from memevo.algorithms.base import BaseAlgorithm
from memevo.algorithms.full_context.prompt import prepare_answer_prompt
from memevo.utils.models import ChatMessage, OpenAICompatLLM


class FullContext(BaseAlgorithm):
    """Baseline that sends the complete conversation to the answer model."""

    def __init__(self, answer_llm: OpenAICompatLLM, working_dir: Path) -> None:
        self._answer_llm = answer_llm
        self._working_dir = working_dir
        self._working_dir.mkdir(parents=True, exist_ok=True)

    async def ingest(self, conv_index: int, messages: Sequence[Any]) -> None:
        memory_file = self._memory_file(conv_index)
        memory_file.parent.mkdir(parents=True, exist_ok=True)
        payload = [_to_dict(message) for message in messages]
        memory_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    async def retrieve(
        self, conv_index: int, question: str = ""
    ) -> list[dict[str, Any]]:
        memory_file = self._memory_file(conv_index)
        if not memory_file.exists():
            raise FileNotFoundError(
                f"Memory not found for conversation {conv_index}; call ingest first"
            )
        return json.loads(memory_file.read_text(encoding="utf-8"))

    async def answer(self, question: str, memory: Any) -> str:
        response = await self._answer_llm.chat(
            [
                ChatMessage(
                    role="system",
                    content="Answer questions using only the supplied conversation.",
                ),
                ChatMessage(
                    role="user",
                    content=prepare_answer_prompt(memory, question),
                ),
            ]
        )
        return response.content.strip()

    def reset_all(self) -> None:
        shutil.rmtree(self._working_dir, ignore_errors=True)
        self._working_dir.mkdir(parents=True, exist_ok=True)

    def _memory_file(self, conv_index: int) -> Path:
        return self._working_dir / f"conv_{conv_index}" / "memory.json"


def _to_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    raise TypeError(f"Unsupported message type: {type(value).__name__}")
