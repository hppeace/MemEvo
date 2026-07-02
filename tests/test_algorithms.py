import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from memevo.algorithms.full_context import FullContext
from memevo.algorithms.mem0 import Mem0
from memevo.datasets.locomo import judge


@dataclass
class Message:
    speaker: str
    text: str
    timestamp_ms: int


class FakeMemory:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.search_calls: list[tuple[str, dict[str, Any]]] = []
        self.search_results: list[dict[str, Any]] = []

    async def add(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)

    async def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.search_calls.append((query, kwargs))
        return {"results": self.search_results}


class FakeLLM:
    def __init__(self) -> None:
        self.prompt = ""

    async def chat(self, messages: list[Any]) -> Any:
        self.prompt = messages[0].content
        return SimpleNamespace(content="ANSWER: Done")


class FakeJudge:
    def __init__(self) -> None:
        self.prompt = ""
        self.options: dict[str, Any] = {}

    async def chat(self, messages: list[Any], **options: Any) -> Any:
        self.prompt = messages[1].content
        self.options = options
        return SimpleNamespace(content='{"reasoning": "matches", "label": "CORRECT"}')


def test_mem0_ingests_official_locomo_message_shape() -> None:
    conversation = SimpleNamespace(
        speaker_a="Caroline",
        sessions=[
            SimpleNamespace(
                session_datetime="1:00 pm on 8 May, 2023",
                messages=[
                    Message("Caroline", "Hello", 1_000),
                    Message("Melanie", "Hi", 31_000),
                ],
            )
        ],
    )
    algorithm = object.__new__(Mem0)
    memory = FakeMemory()
    algorithm._memory = memory
    algorithm._run_id = "test"
    algorithm._reference_dates = {}

    asyncio.run(algorithm.ingest(0, conversation))

    assert algorithm._reference_dates == {0: "1:00 pm on 8 May, 2023"}
    assert memory.calls == [
        {
            "messages": [{"role": "user", "content": "Caroline: Hello"}],
            "user_id": "locomo_0_test",
        },
        {
            "messages": [{"role": "assistant", "content": "Melanie: Hi"}],
            "user_id": "locomo_0_test",
        },
    ]


def test_mem0_retrieves_200_and_answers_with_top_10() -> None:
    memory = FakeMemory()
    memory.search_results = [
        {
            "memory": f"Memory {index}",
            "score": index,
            "created_at": f"2023-05-{12 - index:02d}T00:00:00Z",
        }
        for index in range(12)
    ]
    llm = FakeLLM()
    algorithm = object.__new__(Mem0)
    algorithm._memory = memory
    algorithm._answer_llm = llm
    algorithm._run_id = "test"
    algorithm._reference_dates = {0: "1:00 pm on 8 May, 2023"}
    algorithm._top_k = 200
    algorithm._cutoff = 10
    algorithm._rerank = False

    retrieved = asyncio.run(algorithm.retrieve(0, "Question?"))
    answer = asyncio.run(algorithm.answer("Question?", retrieved))

    assert memory.search_calls == [
        (
            "Question?",
            {
                "filters": {"user_id": "locomo_0_test"},
                "top_k": 200,
                "rerank": False,
            },
        )
    ]
    assert [item["score"] for item in retrieved["results"]] == list(reversed(range(12)))
    assert answer == "Done"
    assert "Memory 11" in llm.prompt
    assert "Memory 2" in llm.prompt
    assert "Memory 1\n" not in llm.prompt
    assert "Memory 0\n" not in llm.prompt
    assert "1:00 pm on 8 May, 2023" in llm.prompt


def test_full_context_accepts_complete_conversation(tmp_path: Path) -> None:
    algorithm = FullContext(None, tmp_path)
    conversation = SimpleNamespace(messages=[{"speaker": "Caroline", "text": "Hello"}])

    asyncio.run(algorithm.ingest(0, conversation))

    assert asyncio.run(algorithm.retrieve(0)) == [
        {"speaker": "Caroline", "text": "Hello"}
    ]


def test_locomo_judge_matches_official_category_3_behavior(tmp_path: Path) -> None:
    answers_path = tmp_path / "answers.json"
    output_path = tmp_path / "evaluation.json"
    answers_path.write_text(
        json.dumps(
            {
                "qa_results": [
                    {
                        "question": "Question?",
                        "answer": "Primary answer; extra explanation",
                        "response": "Primary answer",
                        "category": 3,
                    }
                ]
            }
        )
    )
    llm = FakeJudge()

    metrics = asyncio.run(judge(llm, answers_path, output_path, concurrency=1))

    assert metrics == {"total": 1, "correct": 1, "accuracy": 1.0}
    assert "Gold answer: Primary answer\n" in llm.prompt
    assert "extra explanation" not in llm.prompt
    assert llm.options == {"response_format": {"type": "json_object"}}
