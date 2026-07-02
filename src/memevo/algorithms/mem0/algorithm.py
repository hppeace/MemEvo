from __future__ import annotations

import copy
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mem0 import AsyncMemory

from memevo.algorithms.base import BaseAlgorithm
from memevo.algorithms.mem0.prompt import prepare_answer_prompt
from memevo.utils.models import ChatMessage, OpenAICompatLLM


class Mem0(BaseAlgorithm):
    """Mem0 OSS adapter for conversational-memory experiments."""

    def __init__(
        self,
        answer_llm: OpenAICompatLLM,
        working_dir: Path,
        memory_config: Mapping[str, Any],
        top_k: int = 200,
        cutoff: int = 10,
        rerank: bool = False,
    ) -> None:
        working_dir.mkdir(parents=True, exist_ok=True)
        config = copy.deepcopy(dict(memory_config))
        config.setdefault("history_db_path", str(working_dir / "history.db"))
        config.setdefault(
            "vector_store",
            {
                "provider": "qdrant",
                "config": {
                    "path": str(working_dir / "qdrant"),
                    "collection_name": "memevo_mem0",
                },
            },
        )
        self._memory = AsyncMemory.from_config(_resolve_env(config))
        self._answer_llm = answer_llm
        self._top_k = top_k
        self._cutoff = cutoff
        self._rerank = rerank
        self.reset_all()

    async def ingest(self, conv_index: int, conversation: Any) -> None:
        user_id = self._user_id(conv_index)
        sessions = sorted(
            conversation.sessions, key=lambda item: item.messages[0].timestamp_ms
        )
        if sessions:
            self._reference_dates[conv_index] = sessions[-1].session_datetime
        for session in sessions:
            for message in session.messages:
                await self._memory.add(
                    messages=[
                        {
                            "role": (
                                "user"
                                if message.speaker == conversation.speaker_a
                                else "assistant"
                            ),
                            "content": f"{message.speaker}: {message.text}",
                        }
                    ],
                    user_id=user_id,
                )

    async def retrieve(self, conv_index: int, question: str) -> dict[str, Any]:
        response = await self._memory.search(
            question,
            filters={"user_id": self._user_id(conv_index)},
            top_k=self._top_k,
            rerank=self._rerank,
        )
        results = (
            response.get("results", []) if isinstance(response, dict) else response
        )
        return {
            "results": sorted(
                results, key=lambda item: item.get("score", 0), reverse=True
            ),
            "reference_date": self._reference_dates.get(conv_index),
        }

    async def answer(self, question: str, memory: Any) -> str:
        memories = memory["results"][: self._cutoff]
        response = await self._answer_llm.chat(
            [
                ChatMessage(
                    role="user",
                    content=prepare_answer_prompt(
                        memories,
                        question,
                        memory["reference_date"],
                    ),
                )
            ]
        )
        answer = response.content.strip()
        return answer.rsplit("ANSWER:", 1)[-1].strip()

    def reset_all(self) -> None:
        self._run_id = uuid.uuid4().hex[:8]
        self._reference_dates: dict[int, str] = {}

    def _user_id(self, conv_index: int) -> str:
        return f"locomo_{conv_index}_{self._run_id}"


def _resolve_env(value: Any) -> Any:
    if isinstance(value, dict):
        resolved = {}
        for key, item in value.items():
            if key.endswith("_env"):
                env_name = str(item)
                env_value = os.getenv(env_name)
                if not env_value:
                    raise ValueError(f"Environment variable {env_name} is required")
                resolved[key.removesuffix("_env")] = env_value
            else:
                resolved[key] = _resolve_env(item)
        return resolved
    if isinstance(value, list):
        return [_resolve_env(item) for item in value]
    return value
