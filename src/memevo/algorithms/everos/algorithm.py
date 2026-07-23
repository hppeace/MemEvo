import asyncio
import importlib
import json
import logging
import os
import shutil
from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from time import monotonic
from typing import Any

from everos.component.llm import ChatMessage, ChatResponse
from everos.core.lifespan import LifespanProvider, MetricsLifespanProvider
from everos.core.persistence import MemoryRoot
from everos.entrypoints.api.app import create_app
from everos.entrypoints.api.lifespans import (
    LLMLifespanProvider,
    LanceDBLifespanProvider,
    OmeLifespanProvider,
    SqliteLifespanProvider,
)
from everos.memory.cascade import CascadeOrchestrator
from everos.memory.search import SearchRequest
from everos.service import memorize, search
from fastapi import FastAPI

from memevo.algorithms.everos.prompt import (
    extract_final_answer,
    prepare_answer_prompt,
)
from memevo.algorithms.everos.rerank import Qwen3DashScopeRerankProvider
from memevo.utils.models import Embedder, LLM
from memevo.utils.progress import progress

_OME_CONFIG = """[strategies.extract_foresight]
enabled = false

[strategies.extract_user_profile]
enabled = false
"""
_OME_STRATEGIES = (
    "extract_atomic_facts",
    "extract_foresight",
    "extract_agent_case",
    "trigger_skill_clustering",
    "extract_agent_skill",
    "trigger_profile_clustering",
    "extract_user_profile",
    "reflect_episodes",
)


class EverOS:
    """In-process EverOS adapter aligned with its official LoCoMo benchmark."""

    def __init__(
        self,
        answer_llm: LLM,
        memory_llm: LLM,
        embedder: Embedder,
        working_dir: Path,
        rerank_config: Mapping[str, Any],
        *,
        method: str = "agentic",
        top_k: int = 10,
        eval_owner: str = "speaker_a",
        batch_size: int = 25,
        answer_timeout: float = 300.0,
        answer_max_retries: int = 5,
        agentic_json_max_retries: int = 3,
        ready_timeout: float = 7200.0,
        log_level: str = "ERROR",
    ) -> None:
        if eval_owner not in {"speaker_a", "speaker_b"}:
            raise ValueError("eval_owner must be 'speaker_a' or 'speaker_b'")
        log_level = log_level.upper()
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid EverOS log level: {log_level!r}")
        if agentic_json_max_retries < 0:
            raise ValueError("agentic_json_max_retries must be non-negative")

        self._answer_llm = answer_llm
        self._memory_llm = memory_llm
        self._embedder = embedder
        self._clients = (answer_llm, memory_llm, embedder)
        self._working_dir = working_dir
        self._rerank_config = dict(rerank_config)
        self._method = method
        self._top_k = top_k
        self._eval_owner = eval_owner
        self._batch_size = batch_size
        self._answer_timeout = answer_timeout
        self._answer_max_retries = answer_max_retries
        self._agentic_json_max_retries = agentic_json_max_retries
        self._ready_timeout = ready_timeout
        self._log_level = log_level
        self._startup_lock = asyncio.Lock()
        self._speakers: dict[int, tuple[str, str]] = {}

        self._app: FastAPI | None = None
        self._lifespan: AbstractAsyncContextManager[None] | None = None
        self._environment: dict[str, str | None] = {}
        self._singleton_state: dict[str, Any] = {}

    async def ingest(self, conv_index: int, conversation: Any) -> None:
        await self._ensure_started()
        sessions = sorted(
            conversation.sessions, key=lambda item: item.messages[0].timestamp_ms
        )
        speaker_b = next(
            (
                message.speaker
                for session in sessions
                for message in session.messages
                if message.speaker != conversation.speaker_a
            ),
            conversation.speaker_a,
        )
        self._speakers[conv_index] = (conversation.speaker_a, speaker_b)

        total = sum(len(session.messages) for session in sessions)
        with progress("Ingest", total, "Turn") as bar:
            for session_index, session in enumerate(sessions, start=1):
                session_id = f"locomo_conv{conv_index}_s{session_index}"
                messages = [
                    {
                        "sender_id": f"{message.speaker.lower()}_conv{conv_index}",
                        "sender_name": message.speaker,
                        "role": "user",
                        "timestamp": message.timestamp_ms,
                        "content": [{"type": "text", "text": message.text}],
                    }
                    for message in session.messages
                ]
                for start in range(0, len(messages), self._batch_size):
                    batch = messages[start : start + self._batch_size]
                    await memorize(self._payload(session_id, batch))
                    bar.update(len(batch))
                await memorize(self._payload(session_id, []), is_final=True)

        await self._wait_ready()

    async def retrieve(self, conv_index: int, question: str) -> dict[str, Any]:
        await self._ensure_started()
        speakers = self._speakers[conv_index]
        owner = speakers[0] if self._eval_owner == "speaker_a" else speakers[1]
        request = SearchRequest(
            query=question,
            method=self._method,
            top_k=self._top_k,
            user_id=f"{owner.lower()}_conv{conv_index}",
            app_id="locomo_benchmark",
            project_id="memevo",
        )
        response = await self._search(request)
        return {
            "episodes": [
                item.model_dump(mode="json") for item in response.data.episodes
            ],
            "profiles": [
                item.model_dump(mode="json") for item in response.data.profiles
            ],
            "speakers": speakers,
        }

    async def _search(self, request: SearchRequest) -> Any:
        for attempt in range(self._agentic_json_max_retries + 1):
            try:
                return await search(request)
            except ValueError as exc:
                retryable = self._method == "agentic" and (
                    isinstance(exc, json.JSONDecodeError)
                    or str(exc).startswith("No JSON object found in LLM response:")
                )
                if not retryable or attempt == self._agentic_json_max_retries:
                    raise
                await asyncio.sleep(2**attempt)
        raise RuntimeError("unreachable Agentic search retry state")

    async def answer(self, question: str, memory: dict[str, Any]) -> str:
        prompt = prepare_answer_prompt(memory, question)
        for attempt in range(self._answer_max_retries):
            try:
                response = await self._answer_llm.chat(
                    [{"role": "user", "content": prompt}],
                    timeout=self._answer_timeout,
                )
                answer = extract_final_answer(response)
                if answer:
                    return answer
            except Exception:
                if attempt == self._answer_max_retries - 1:
                    raise
            if attempt < self._answer_max_retries - 1:
                await asyncio.sleep(2**attempt)
        raise RuntimeError("EverOS answer model returned an empty response")

    def reset_all(self) -> None:
        if self._lifespan is not None:
            raise RuntimeError("cannot reset EverOS while it is running")
        shutil.rmtree(self._working_dir, ignore_errors=True)
        self._working_dir.mkdir(parents=True, exist_ok=True)
        self._speakers.clear()

    async def close(self) -> None:
        try:
            if self._lifespan is not None:
                await self._lifespan.__aexit__(None, None, None)
                self._lifespan = None
                self._app = None
        finally:
            self._restore_singletons()
            self._restore_environment()
            for client in self._clients:
                await client.close()

    def _payload(
        self, session_id: str, messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "app_id": "locomo_benchmark",
            "project_id": "memevo",
            "messages": messages,
        }

    async def _ensure_started(self) -> None:
        if self._lifespan is not None:
            return
        async with self._startup_lock:
            if self._lifespan is not None:
                return

            self._working_dir.mkdir(parents=True, exist_ok=True)
            (self._working_dir / "ome.toml").write_text(_OME_CONFIG, encoding="utf-8")
            from everos.core.observability.logging import configure_logging

            configure_logging(self._log_level)
            jieba = importlib.import_module("jieba")
            jieba.setLogLevel(getattr(logging, self._log_level))
            self._configure_environment()
            self._install_singletons()

            providers: list[LifespanProvider] = [
                MetricsLifespanProvider(),
                LLMLifespanProvider(),
                SqliteLifespanProvider(),
                LanceDBLifespanProvider(),
                _SharedEmbeddingCascade(self._embedder),
                OmeLifespanProvider(),
            ]
            self._app = create_app(lifespan_providers=providers)
            lifespan = self._app.router.lifespan_context(self._app)
            try:
                await lifespan.__aenter__()
            except BaseException:
                self._restore_singletons()
                self._restore_environment()
                raise
            self._lifespan = lifespan

    async def _wait_ready(self) -> None:
        if self._app is None:
            raise RuntimeError("EverOS is not running")
        cascade = self._app.state.lifespan_data["cascade"]
        ome = self._app.state.lifespan_data["ome"]
        deadline = monotonic() + self._ready_timeout
        stable = 0

        while monotonic() < deadline:
            processed = await cascade.sync_once()
            remaining = max(deadline - monotonic(), 0.001)
            idle = await ome.wait_idle(timeout=min(remaining, 1.0))
            summary = await cascade.queue_summary()
            if summary.failed_retryable or summary.failed_permanent:
                raise RuntimeError(
                    "EverOS cascade failed: "
                    f"retryable={summary.failed_retryable}, "
                    f"permanent={summary.failed_permanent}"
                )
            if processed == 0 and idle and summary.pending == 0:
                stable += 1
                if stable == 2:
                    await _raise_for_ome_failures(ome)
                    return
            else:
                stable = 0
        raise TimeoutError(f"EverOS did not become ready in {self._ready_timeout}s")

    def _configure_environment(self) -> None:
        values = {
            "EVEROS_ROOT": str(self._working_dir),
            "EVEROS_MEMORIZE__MODE": "chat",
            **_rerank_environment(self._rerank_config),
        }
        self._environment = {name: os.environ.get(name) for name in values}
        os.environ.update(values)

        from everos.config import load_settings

        load_settings.cache_clear()

    def _restore_environment(self) -> None:
        for name, value in self._environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self._environment.clear()

        from everos.config import load_settings

        load_settings.cache_clear()

    def _install_singletons(self) -> None:
        embedding_module = importlib.import_module(
            "everos.component.embedding.accessor"
        )
        llm_module = importlib.import_module("everos.component.llm.client")
        search_module = importlib.import_module("everos.service.search")
        llm_adapter = _SharedLLM(self._memory_llm)
        embedding_adapter = _SharedEmbedder(self._embedder)

        names = {
            "embedding": (embedding_module, "_embedder"),
            "llm": (llm_module, "_llm_client"),
            "search_embedding": (search_module, "_embedding"),
            "search_embedding_resolved": (search_module, "_embedding_resolved"),
            "search_llm": (search_module, "_llm_client"),
            "search_llm_resolved": (search_module, "_llm_resolved"),
            "search_reranker": (search_module, "_reranker"),
            "search_rerank_resolved": (search_module, "_rerank_resolved"),
        }
        self._singleton_state = {
            key: (module, name, getattr(module, name))
            for key, (module, name) in names.items()
        }
        embedding_module._embedder = embedding_adapter
        llm_module._llm_client = llm_adapter
        search_module._embedding = embedding_adapter
        search_module._embedding_resolved = True
        search_module._llm_client = llm_adapter
        search_module._llm_resolved = True
        if reranker := _adapter_reranker(self._rerank_config):
            search_module._reranker = reranker
            search_module._rerank_resolved = True

    def _restore_singletons(self) -> None:
        for module, name, value in self._singleton_state.values():
            setattr(module, name, value)
        self._singleton_state.clear()

        memorize_module = importlib.import_module("everos.service.memorize")
        search_module = importlib.import_module("everos.service.search")
        for name in (
            "_episode_writer",
            "_prompt_loader",
            "_user_pipeline",
            "_agent_pipeline",
            "_ome_engine",
        ):
            setattr(memorize_module, name, None)
        search_module._manager = None


class _SharedLLM:
    def __init__(self, client: LLM) -> None:
        self._client = client

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> ChatResponse:
        options = dict(extra)
        if response_format is not None:
            options["response_format"] = dict(response_format)
        response = await self._client.chat(
            [message.model_dump(exclude_none=True) for message in messages],
            **options,
        )
        return ChatResponse(
            content=response,
            model=self._client.model,
            usage=None,
            finish_reason=None,
            raw=None,
        )


class _SharedEmbedder:
    dim = 1024

    def __init__(
        self,
        client: Embedder,
        batch_size: int = 10,
        max_concurrent: int = 5,
    ) -> None:
        self._client = client
        self._batch_size = batch_size
        self._slots = asyncio.Semaphore(max_concurrent)

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        chunks = [
            list(texts[start : start + self._batch_size])
            for start in range(0, len(texts), self._batch_size)
        ]
        results = await asyncio.gather(*(self._embed_chunk(chunk) for chunk in chunks))
        vectors = [vector for chunk in results for vector in chunk]
        if any(len(vector) < self.dim for vector in vectors):
            raise ValueError("EverOS requires embeddings with at least 1024 dimensions")
        return [vector[: self.dim] for vector in vectors]

    async def _embed_chunk(self, texts: list[str]) -> list[list[float]]:
        async with self._slots:
            return await self._client.embed(texts)


class _SharedEmbeddingCascade(LifespanProvider):
    def __init__(self, embedder: Embedder) -> None:
        super().__init__(name="cascade", order=12)
        self._embedder = _SharedEmbedder(embedder)
        self._orchestrator: CascadeOrchestrator | None = None

    async def startup(self, app: FastAPI) -> CascadeOrchestrator:
        from everos.component.tokenizer import build_tokenizer

        memory_root = MemoryRoot.default()
        memory_root.ensure()
        self._orchestrator = CascadeOrchestrator(
            memory_root=memory_root,
            embedder=self._embedder,
            tokenizer=build_tokenizer(),
        )
        await self._orchestrator.start()
        return self._orchestrator

    async def shutdown(self, app: FastAPI) -> None:
        if self._orchestrator is not None:
            await self._orchestrator.stop()
            self._orchestrator = None


def _rerank_environment(config: Mapping[str, Any]) -> dict[str, str]:
    required = ("provider", "model", "base_url", "api_key_env")
    missing = [name for name in required if not config.get(name)]
    if missing:
        raise ValueError(f"models.rerank is missing: {', '.join(missing)}")
    values = {
        "EVEROS_RERANK__PROVIDER": str(config["provider"]),
        "EVEROS_RERANK__MODEL": str(config["model"]),
        "EVEROS_RERANK__BASE_URL": str(config["base_url"]),
        "EVEROS_RERANK__API_KEY": os.environ[str(config["api_key_env"])],
    }
    for name, value in dict(config.get("options", {})).items():
        values[f"EVEROS_RERANK__{name.upper()}"] = str(value)
    return values


def _adapter_reranker(
    config: Mapping[str, Any],
) -> Qwen3DashScopeRerankProvider | None:
    if config.get("provider") != "dashscope" or config.get("model") != "qwen3-rerank":
        return None

    options = dict(config.get("options", {}))
    api_key_env = str(config["api_key_env"])
    return Qwen3DashScopeRerankProvider(
        model=str(config["model"]),
        api_key=os.environ[api_key_env],
        base_url=str(config["base_url"]),
        timeout=float(options.get("timeout_seconds", 30.0)),
        max_retries=int(options.get("max_retries", 3)),
        batch_size=int(options.get("batch_size", 10)),
        max_concurrent=int(options.get("max_concurrent", 5)),
    )


async def _raise_for_ome_failures(ome: Any) -> None:
    failed: list[str] = []
    for strategy in _OME_STRATEGIES:
        # EverOS keeps one record per attempt and returns newest records first.
        # A FAILED attempt is retryable and remains in history after a later
        # attempt succeeds, so only the newest attempt for each event is final.
        records = await ome.list_runs(strategy, limit=1000)
        latest_by_event: dict[str, Any] = {}
        for record in records:
            event_key = record.event_id or record.run_id
            latest_by_event.setdefault(event_key, record)

        for record in latest_by_event.values():
            status = str(record.status)
            if status in {"failed", "dead_letter", "crashed"}:
                failed.append(
                    f"{record.strategy_name}:{status}"
                    f"(event_id={record.event_id}, attempt={record.attempt})"
                )
    if failed:
        raise RuntimeError(f"EverOS OME failed: {', '.join(failed)}")
