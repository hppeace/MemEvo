from __future__ import annotations

from pathlib import Path
from typing import Any

from algorithms.base import Algorithm
from pydantic import BaseModel


from utils.clients import OpenAICompatLLM, ChatMessage

class FullContext(Algorithm):
    def __init__(self, answer_llm_config: dict, working_dir: Path):
        self._answer_llm = OpenAICompatLLM(**answer_llm_config)
        self._working_dir = working_dir
        working_dir.mkdir(parents=True, exist_ok=True)

    async def ingest(self, conv_index: int, message: BaseModel) -> None:
        conv_dir = self._working_dir / f"conv_{conv_index}"
        conv_dir.mkdir(parents=True, exist_ok=True)

        memory_file = conv_dir / f"memory.json"
        memory_file.write_text(message.model_dump_json(indent=2), encoding="utf-8")

    async def retrieve(self, conv_index: int) -> Any:
        memory_file = self._working_dir / f"conv_{conv_index}" / "memory.json"
        if not memory_file.exists():
            raise FileNotFoundError(f"Memory file not found for conversation {conv_index}")
        data = memory_file.read_text(encoding="utf-8")
        return data

    async def answer(self, conv_index: int, query: BaseModel) -> Any:
        """
        Provide an answer based on the memory.
        """
        answer = await self.retrieve(conv_index)
        answer_prompt = self._build_answer_prompt(answer, query.question)
        response = await self._answer_llm.chat(
            [
                ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(role="user", content=answer_prompt),
            ]
        )
        return response.content
    
    def reset_all(self) -> None:
        """
        Reset all memories to their initial state.
        """

        pass


        