import asyncio
import importlib
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import memevo.algorithms.everos.algorithm as everos_algorithm
from everos.component.llm import ChatMessage
from memevo.algorithms.everos.algorithm import EverOS, _SharedEmbedder, _SharedLLM
from memevo.algorithms.everos.rerank import Qwen3DashScopeRerankProvider


@dataclass
class Message:
    speaker: str
    text: str
    timestamp_ms: int


class FakeLLM:
    model = "fake-model"

    def __init__(
        self,
        response: str = "response",
        configured_options: dict[str, Any] | None = None,
    ) -> None:
        self.response = response
        self.configured_options = configured_options or {}
        self.calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []

    async def chat(self, messages: list[dict[str, Any]], **options: Any) -> str:
        self.calls.append((messages, self.configured_options | options))
        return self.response


class FakeEmbedder:
    def __init__(self, dimensions: int = 1025) -> None:
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(index)] * self.dimensions for index, _ in enumerate(texts)]


def test_everos_ingest_matches_official_locomo_shape(monkeypatch: Any) -> None:
    calls: list[tuple[dict[str, Any], bool]] = []

    async def memorize(payload: dict[str, Any], *, is_final: bool = False) -> None:
        calls.append((payload, is_final))

    async def no_op() -> None:
        pass

    monkeypatch.setattr(everos_algorithm, "memorize", memorize)
    algorithm = object.__new__(EverOS)
    algorithm._batch_size = 1
    algorithm._speakers = {}
    algorithm._ensure_started = no_op
    algorithm._wait_ready = no_op
    conversation = SimpleNamespace(
        speaker_a="Caroline",
        sessions=[
            SimpleNamespace(
                messages=[
                    Message("Caroline", "Hello", 1_000),
                    Message("Melanie", "Hi", 31_000),
                ]
            )
        ],
    )

    asyncio.run(algorithm.ingest(2, conversation))

    assert algorithm._speakers == {2: ("Caroline", "Melanie")}
    assert calls == [
        (
            {
                "session_id": "locomo_conv2_s1",
                "app_id": "locomo_benchmark",
                "project_id": "memevo",
                "messages": [
                    {
                        "sender_id": "caroline_conv2",
                        "sender_name": "Caroline",
                        "role": "user",
                        "timestamp": 1_000,
                        "content": [{"type": "text", "text": "Hello"}],
                    }
                ],
            },
            False,
        ),
        (
            {
                "session_id": "locomo_conv2_s1",
                "app_id": "locomo_benchmark",
                "project_id": "memevo",
                "messages": [
                    {
                        "sender_id": "melanie_conv2",
                        "sender_name": "Melanie",
                        "role": "user",
                        "timestamp": 31_000,
                        "content": [{"type": "text", "text": "Hi"}],
                    }
                ],
            },
            False,
        ),
        (
            {
                "session_id": "locomo_conv2_s1",
                "app_id": "locomo_benchmark",
                "project_id": "memevo",
                "messages": [],
            },
            True,
        ),
    ]


def test_everos_retrieves_agentic_top_10_for_speaker_a(monkeypatch: Any) -> None:
    captured: list[Any] = []

    class Item:
        def __init__(self, value: dict[str, Any]) -> None:
            self.value = value

        def model_dump(self, *, mode: str) -> dict[str, Any]:
            assert mode == "json"
            return self.value

    async def search(request: Any) -> Any:
        captured.append(request)
        return SimpleNamespace(
            data=SimpleNamespace(
                episodes=[Item({"episode": "remembered"})], profiles=[]
            )
        )

    async def no_op() -> None:
        pass

    monkeypatch.setattr(everos_algorithm, "search", search)
    algorithm = object.__new__(EverOS)
    algorithm._ensure_started = no_op
    algorithm._speakers = {0: ("Caroline", "Melanie")}
    algorithm._eval_owner = "speaker_a"
    algorithm._method = "agentic"
    algorithm._top_k = 10
    algorithm._agentic_json_max_retries = 3

    result = asyncio.run(algorithm.retrieve(0, "Question?"))

    request = captured[0]
    assert request.query == "Question?"
    assert request.method.value == "agentic"
    assert request.top_k == 10
    assert request.user_id == "caroline_conv0"
    assert request.app_id == "locomo_benchmark"
    assert request.project_id == "memevo"
    assert result == {
        "episodes": [{"episode": "remembered"}],
        "profiles": [],
        "speakers": ("Caroline", "Melanie"),
    }


def test_everos_retries_agentic_json_errors(monkeypatch: Any) -> None:
    calls = 0
    delays: list[int] = []

    async def search(request: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise json.JSONDecodeError("Expecting value", "{}", 1)
        return "found"

    async def sleep(delay: int) -> None:
        delays.append(delay)

    monkeypatch.setattr(everos_algorithm, "search", search)
    monkeypatch.setattr(everos_algorithm.asyncio, "sleep", sleep)
    algorithm = object.__new__(EverOS)
    algorithm._method = "agentic"
    algorithm._agentic_json_max_retries = 3

    result = asyncio.run(algorithm._search(SimpleNamespace()))

    assert result == "found"
    assert calls == 3
    assert delays == [1, 2]


def test_everos_does_not_retry_unrelated_search_errors(monkeypatch: Any) -> None:
    calls = 0

    async def search(request: Any) -> Any:
        nonlocal calls
        calls += 1
        raise ValueError("invalid search configuration")

    monkeypatch.setattr(everos_algorithm, "search", search)
    algorithm = object.__new__(EverOS)
    algorithm._method = "agentic"
    algorithm._agentic_json_max_retries = 3

    try:
        asyncio.run(algorithm._search(SimpleNamespace()))
    except ValueError as exc:
        assert str(exc) == "invalid search configuration"
    else:
        raise AssertionError("expected unrelated search error")

    assert calls == 1


def test_everos_raises_after_agentic_json_retries_are_exhausted(
    monkeypatch: Any,
) -> None:
    calls = 0
    delays: list[int] = []

    async def search(request: Any) -> Any:
        nonlocal calls
        calls += 1
        raise ValueError("No JSON object found in LLM response: 'invalid'")

    async def sleep(delay: int) -> None:
        delays.append(delay)

    monkeypatch.setattr(everos_algorithm, "search", search)
    monkeypatch.setattr(everos_algorithm.asyncio, "sleep", sleep)
    algorithm = object.__new__(EverOS)
    algorithm._method = "agentic"
    algorithm._agentic_json_max_retries = 2

    try:
        asyncio.run(algorithm._search(SimpleNamespace()))
    except ValueError as exc:
        assert str(exc) == "No JSON object found in LLM response: 'invalid'"
    else:
        raise AssertionError("expected exhausted Agentic JSON error")

    assert calls == 3
    assert delays == [1, 2]


def test_everos_answer_uses_prompt_and_configured_temperature() -> None:
    llm = FakeLLM(
        "reasoning\n## STEP 7: FINAL ANSWER\nDone",
        configured_options={"temperature": 0.6, "max_tokens": 4096},
    )
    algorithm = object.__new__(EverOS)
    algorithm._answer_llm = llm
    algorithm._answer_timeout = 300.0
    algorithm._answer_max_retries = 5
    memory = {
        "episodes": [{"subject": "A trip", "episode": "Went to Hawaii"}],
        "profiles": [],
        "speakers": ("Caroline", "Melanie"),
    }

    answer = asyncio.run(algorithm.answer("Where?", memory))

    assert answer == "Done"
    messages, options = llm.calls[0]
    assert (
        "Episodes memories for conversation between Caroline and Melanie"
        in (messages[0]["content"])
    )
    assert "A trip: Went to Hawaii" in messages[0]["content"]
    assert options == {
        "temperature": 0.6,
        "max_tokens": 4096,
        "timeout": 300.0,
    }


def test_everos_shared_model_adapters_delegate_and_truncate() -> None:
    async def run() -> None:
        llm = FakeLLM(
            "memory",
            configured_options={"temperature": 0.6, "max_tokens": 4096},
        )
        embedder = FakeEmbedder()
        response = await _SharedLLM(llm).chat(
            [ChatMessage(role="user", content="remember")],
            temperature=0.1,
            max_tokens=123,
            response_format={"type": "json_object"},
        )
        vectors = await _SharedEmbedder(embedder).embed_batch(
            [str(index) for index in range(11)]
        )

        assert response.content == "memory"
        assert llm.calls == [
            (
                [{"role": "user", "content": "remember"}],
                {
                    "temperature": 0.6,
                    "max_tokens": 4096,
                    "response_format": {"type": "json_object"},
                },
            )
        ]
        assert embedder.calls == [
            [str(index) for index in range(10)],
            ["10"],
        ]
        assert [len(vector) for vector in vectors] == [1024] * 11

    asyncio.run(run())


def test_qwen3_dashscope_reranker_uses_compatible_api_shape(
    monkeypatch: Any,
) -> None:
    provider = Qwen3DashScopeRerankProvider(
        model="qwen3-rerank",
        api_key="secret",
        base_url="https://example.test/compatible-api/v1/reranks",
        batch_size=2,
        max_concurrent=1,
    )
    payloads: list[dict[str, Any]] = []

    async def request(payload: dict[str, Any]) -> dict[str, Any]:
        payloads.append(payload)
        scores = [0.2, 0.9] if len(payload["documents"]) == 2 else [0.5]
        return {
            "results": [
                {"index": index, "relevance_score": score}
                for index, score in enumerate(scores)
            ]
        }

    monkeypatch.setattr(provider, "_request", request)
    results = asyncio.run(
        provider.rerank("query", ["a", "b", "c"], instruction="find facts")
    )

    assert payloads == [
        {
            "model": "qwen3-rerank",
            "query": "query",
            "documents": ["a", "b"],
            "top_n": 2,
            "instruct": "find facts",
        },
        {
            "model": "qwen3-rerank",
            "query": "query",
            "documents": ["c"],
            "top_n": 1,
            "instruct": "find facts",
        },
    ]
    assert [(result.index, result.score) for result in results] == [
        (1, 0.9),
        (2, 0.5),
        (0, 0.2),
    ]


def test_everos_injects_qwen3_reranker(monkeypatch: Any) -> None:
    embedding_module = importlib.import_module("everos.component.embedding.accessor")
    search_module = importlib.import_module("everos.service.search")

    algorithm = object.__new__(EverOS)
    algorithm._memory_llm = FakeLLM()
    algorithm._embedder = FakeEmbedder()
    algorithm._singleton_state = {}
    algorithm._rerank_config = {
        "provider": "dashscope",
        "model": "qwen3-rerank",
        "base_url": "https://example.test/compatible-api/v1/reranks",
        "api_key_env": "TEST_RERANK_API_KEY",
        "options": {"batch_size": 7, "max_concurrent": 2},
    }
    monkeypatch.setenv("TEST_RERANK_API_KEY", "secret")

    algorithm._install_singletons()
    try:
        assert isinstance(embedding_module._embedder, _SharedEmbedder)
        assert isinstance(search_module._reranker, Qwen3DashScopeRerankProvider)
        assert search_module._rerank_resolved is True
        assert search_module._reranker._batch_size == 7
    finally:
        algorithm._restore_singletons()
