import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from memevo.algorithms.full_context.prompt import prepare_answer_prompt
from memevo.utils.models import LLM


class FullContext:
    """Baseline that sends the complete conversation to the answer model."""

    def __init__(self, answer_llm: LLM, working_dir: Path) -> None:
        self._answer_llm = answer_llm
        self._working_dir = working_dir

    async def ingest(self, conv_index: int, conversation: Any) -> None:
        memory_file = self._memory_file(conv_index)
        memory_file.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            asdict(message) if is_dataclass(message) else message
            for message in conversation.messages
        ]
        memory_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    async def retrieve(self, conv_index: int, question: str) -> list[dict[str, Any]]:
        memory_file = self._memory_file(conv_index)
        return json.loads(memory_file.read_text(encoding="utf-8"))

    async def answer(self, question: str, memory: Any) -> str:
        response = await self._answer_llm.chat(
            [
                {
                    "role": "system",
                    "content": "Answer questions using only the supplied conversation.",
                },
                {
                    "role": "user",
                    "content": prepare_answer_prompt(memory, question),
                },
            ]
        )
        return response.strip()

    def reset_all(self) -> None:
        shutil.rmtree(self._working_dir, ignore_errors=True)
        self._working_dir.mkdir(parents=True, exist_ok=True)

    async def close(self) -> None:
        await self._answer_llm.close()

    def _memory_file(self, conv_index: int) -> Path:
        return self._working_dir / f"conv_{conv_index}" / "memory.json"
