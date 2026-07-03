import asyncio
import json
import logging
import os
import uuid
import warnings
from collections.abc import Coroutine, Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("MEM0_TELEMETRY", "false")

from mem0 import AsyncMemory
from mem0.memory.utils import extract_json, remove_code_blocks

from memevo.algorithms.mem0.prompt import prepare_answer_prompt
from memevo.utils.models import Embedder, LLM
from memevo.utils.progress import progress

logging.getLogger("mem0.vector_stores.qdrant").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message="Payload indexes have no effect in the local Qdrant.*",
)


class Mem0:
    """Mem0 OSS adapter for conversational-memory experiments."""

    def __init__(
        self,
        answer_llm: LLM,
        memory_llm: LLM,
        embedder: Embedder,
        working_dir: Path,
        memory_config: Mapping[str, Any],
        top_k: int = 200,
        cutoff: int = 10,
        rerank: bool = False,
        embedding_dims: int = 1536,
    ) -> None:
        working_dir.mkdir(parents=True, exist_ok=True)
        config = dict(memory_config)
        config["llm"] = {
            "provider": "openai",
            "config": {
                "model": "unused",
                "api_key": "unused",
            },
        }
        config["embedder"] = {
            "provider": "openai",
            "config": {
                "model": "unused",
                "embedding_dims": embedding_dims,
                "api_key": "unused",
            },
        }
        config.setdefault("history_db_path", str(working_dir / "history.db"))
        config.setdefault(
            "vector_store",
            {
                "provider": "qdrant",
                "config": {
                    "path": str(working_dir / "qdrant"),
                    "collection_name": "memevo_mem0",
                    "embedding_model_dims": embedding_dims,
                },
            },
        )
        loop = asyncio.get_running_loop()
        self._memory = AsyncMemory.from_config(_resolve_env(config))
        self._memory.llm = _Mem0LLM(memory_llm, loop)
        self._memory.embedding_model = _Mem0Embedder(embedder, loop)
        self._answer_llm = answer_llm
        self._clients = (answer_llm, memory_llm, embedder)
        self._top_k = top_k
        self._cutoff = cutoff
        self._rerank = rerank

    async def ingest(self, conv_index: int, conversation: Any) -> None:
        user_id = self._user_id(conv_index)
        sessions = sorted(
            conversation.sessions, key=lambda item: item.messages[0].timestamp_ms
        )
        if sessions:
            self._reference_dates[conv_index] = sessions[-1].session_datetime
        total = sum(len(session.messages) for session in sessions)
        with progress("Ingest", total, "Turn") as bar:
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
                    bar.update()

    async def retrieve(self, conv_index: int, question: str) -> dict[str, Any]:
        response = await self._memory.search(
            question,
            filters={"user_id": self._user_id(conv_index)},
            top_k=self._top_k,
            rerank=self._rerank,
        )
        results = response["results"]
        return {
            "results": sorted(results, key=lambda item: item["score"], reverse=True),
            "reference_date": self._reference_dates[conv_index],
        }

    async def answer(self, question: str, memory: Any) -> str:
        memories = memory["results"][: self._cutoff]
        response = await self._answer_llm.chat(
            [
                {
                    "role": "user",
                    "content": prepare_answer_prompt(
                        memories,
                        question,
                        memory["reference_date"],
                    ),
                }
            ]
        )
        answer = response.strip()
        return answer.rsplit("ANSWER:", 1)[-1].strip()

    def reset_all(self) -> None:
        self._run_id = uuid.uuid4().hex[:8]
        self._reference_dates: dict[int, str] = {}

    def _user_id(self, conv_index: int) -> str:
        return f"locomo_{conv_index}_{self._run_id}"

    async def close(self) -> None:
        for client in self._clients:
            await client.close()


class _Mem0LLM:
    def __init__(self, client: LLM, loop: asyncio.AbstractEventLoop) -> None:
        self._client = client
        self._loop = loop

    def generate_response(self, messages: list[dict[str, str]], **options: Any) -> str:
        for _ in range(3):
            response = _run(self._loop, self._client.chat(messages, **options))
            if options.get("response_format", {}).get(
                "type"
            ) != "json_object" or _valid_memory_json(response):
                return response
        return '{"memory": []}'


class _Mem0Embedder:
    def __init__(self, client: Embedder, loop: asyncio.AbstractEventLoop) -> None:
        self._client = client
        self._loop = loop

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        return self.embed_batch([text], memory_action)[0]

    def embed_batch(
        self, texts: Sequence[str], memory_action: str | None = None
    ) -> list[list[float]]:
        texts = [text.replace("\n", " ") for text in texts]
        return [
            vector
            for start in range(0, len(texts), 100)
            for vector in _run(
                self._loop,
                self._client.embed(texts[start : start + 100]),
            )
        ]


def _run[T](
    loop: asyncio.AbstractEventLoop,
    coroutine: Coroutine[Any, Any, T],
) -> T:
    return asyncio.run_coroutine_threadsafe(coroutine, loop).result()


def _valid_memory_json(response: str) -> bool:
    try:
        data = json.loads(extract_json(remove_code_blocks(response)), strict=False)
        memory = data.get("memory", []) if isinstance(data, dict) else None
        return isinstance(memory, list) and all(
            isinstance(item, dict) for item in memory
        )
    except (json.JSONDecodeError, TypeError):
        return False


def _resolve_env(value: Any) -> Any:
    if isinstance(value, dict):
        resolved = {}
        for key, item in value.items():
            if key.endswith("_env"):
                resolved[key.removesuffix("_env")] = os.environ[str(item)]
            else:
                resolved[key] = _resolve_env(item)
        return resolved
    if isinstance(value, list):
        return [_resolve_env(item) for item in value]
    return value
