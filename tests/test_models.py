import asyncio
from types import SimpleNamespace
from typing import Any

from memevo.algorithms.mem0.algorithm import _Mem0LLM
from memevo.utils.models import Embedder, LLM, Usage


class FakeAPI:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=self)
        self.embeddings = self

    async def create(self, **options: Any) -> Any:
        self.calls.append(options)
        if "messages" in options:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=2,
                    total_tokens=12,
                ),
            )
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(index)])
                for index, _ in enumerate(options["input"])
            ],
            usage=SimpleNamespace(
                prompt_tokens=len(options["input"]),
                completion_tokens=0,
                total_tokens=len(options["input"]),
            ),
        )

    async def close(self) -> None:
        pass


def test_models_forward_options_and_track_usage(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("API_KEY", "key")

    async def run() -> dict[str, Any]:
        usage = Usage(["memory", "embedding", "unused"])
        llm = LLM(
            "memory",
            {
                "model": "chat",
                "api_key_env": "API_KEY",
                "options": {"temperature": 0.1},
            },
            usage,
        )
        embedder = Embedder(
            "embedding",
            {
                "model": "embed",
                "api_key_env": "API_KEY",
                "options": {"dimensions": 3},
            },
            usage,
        )
        llm.client = FakeAPI()
        embedder.client = FakeAPI()

        with usage.stage("ingest"):
            assert (
                await llm.chat(
                    [{"role": "user", "content": "remember"}],
                    response_format={"type": "json_object"},
                )
                == "ok"
            )
            assert await embedder.embed(["first", "second"]) == [[0.0], [1.0]]

        assert llm.client.calls[0]["temperature"] == 0.1
        assert llm.client.calls[0]["response_format"] == {"type": "json_object"}
        assert embedder.client.calls[0]["dimensions"] == 3
        await llm.close()
        await embedder.close()
        return usage.summary()

    summary = asyncio.run(run())
    assert summary["total"]["total_tokens"] == 14
    assert summary["memory"]["total_tokens"] == 12
    assert summary["embedding"]["total_tokens"] == 2
    assert summary["unused"]["calls"] == 0
    assert summary["stages"]["ingest"]["calls"] == 2


def test_each_llm_uses_its_own_sampling_top_k(monkeypatch: Any) -> None:
    monkeypatch.setenv("API_KEY", "key")

    async def run() -> None:
        usage = Usage(["memory", "answer", "judge"])
        clients: dict[str, FakeAPI] = {}
        models: list[LLM] = []
        for name, top_k in (("memory", 11), ("answer", 22), ("judge", 33)):
            llm = LLM(
                name,
                {
                    "model": name,
                    "api_key_env": "API_KEY",
                    "options": {"extra_body": {"top_k": top_k}},
                },
                usage,
            )
            client = FakeAPI()
            llm.client = client
            clients[name] = client
            models.append(llm)

        for llm in models:
            await llm.chat([{"role": "user", "content": "test"}])
            await llm.close()

        assert clients["memory"].calls[0]["extra_body"]["top_k"] == 11
        assert clients["answer"].calls[0]["extra_body"]["top_k"] == 22
        assert clients["judge"].calls[0]["extra_body"]["top_k"] == 33

    asyncio.run(run())


def test_mem0_thread_bridge_keeps_stage(monkeypatch: Any) -> None:
    monkeypatch.setenv("API_KEY", "key")

    async def run() -> dict[str, Any]:
        usage = Usage(["memory"])
        llm = LLM(
            "memory",
            {"model": "chat", "api_key_env": "API_KEY"},
            usage,
        )
        llm.client = FakeAPI()
        with usage.stage("ingest"):
            await asyncio.to_thread(
                _Mem0LLM(llm, asyncio.get_running_loop()).generate_response,
                [{"role": "user", "content": "remember"}],
            )
        await llm.close()
        return usage.summary()

    assert asyncio.run(run())["stages"]["ingest"]["calls"] == 1
