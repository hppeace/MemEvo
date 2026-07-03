import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import memevo.utils.runner as runner
from memevo.utils.models import Usage

TOKENS = SimpleNamespace(
    prompt_tokens=1,
    completion_tokens=1,
    total_tokens=2,
)


def test_runner_only_orchestrates_stages_and_results(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    question = SimpleNamespace(
        question="Question?",
        answer="Answer",
        category=1,
        evidence=["D1:1"],
    )

    class Dataset:
        indices = [0]

        def __init__(self, usage: Usage) -> None:
            self.usage = usage

        def load(self, conv_index: int) -> Any:
            return SimpleNamespace(qa=[question])

        def questions(self, conversation: Any) -> list[Any]:
            return conversation.qa

        def question_text(self, item: Any) -> str:
            return item.question

        def result(self, conv_index: int, item: Any, response: str) -> dict[str, Any]:
            return {
                "conv_index": conv_index,
                "question": item.question,
                "answer": item.answer,
                "response": response,
                "category": item.category,
                "evidence": item.evidence,
            }

        async def evaluate(
            self,
            answers_path: Path,
            output_path: Path,
            concurrency: int,
        ) -> dict[str, int]:
            self.usage.record("judge", TOKENS)
            metrics = {"total": 1, "correct": 1}
            output_path.write_text(json.dumps({"metrics": metrics}))
            return metrics

        async def close(self) -> None:
            pass

    class Algorithm:
        def __init__(self, usage: Usage) -> None:
            self.usage = usage

        def reset_all(self) -> None:
            pass

        async def ingest(self, conv_index: int, conversation: Any) -> None:
            self.usage.record("worker", TOKENS)

        async def retrieve(self, conv_index: int, query: str) -> str:
            self.usage.record("worker", TOKENS)
            return "memory"

        async def answer(self, query: str, memory: str) -> str:
            self.usage.record("worker", TOKENS)
            return "response"

        async def close(self) -> None:
            pass

    def import_component(name: str) -> Any:
        if ".datasets." in name:
            return SimpleNamespace(
                create=lambda settings, models, usage: Dataset(usage)
            )
        return SimpleNamespace(
            create=lambda settings, models, usage, path: Algorithm(usage)
        )

    monkeypatch.setattr(runner, "import_module", import_component)

    metrics = asyncio.run(
        runner.run_benchmark(
            {
                "run": {
                    "name": "test",
                    "output_dir": str(tmp_path),
                    "concurrency": 2,
                },
                "dataset": {"name": "dataset"},
                "algorithm": {"name": "algorithm"},
                "models": {"worker": {}, "judge": {}},
            }
        )
    )

    answers = json.loads((tmp_path / "answers.json").read_text())
    usage = json.loads((tmp_path / "usage.json").read_text())["model_usage"]
    assert metrics == {"total": 1, "correct": 1}
    assert answers["qa_results"][0]["response"] == "response"
    assert usage["total"]["calls"] == 4
    assert usage["worker"]["calls"] == 3
    assert usage["judge"]["calls"] == 1
    assert usage["stages"]["ingest"]["calls"] == 1
    assert usage["stages"]["retrieve"]["calls"] == 1
    assert usage["stages"]["answer"]["calls"] == 1
    assert usage["stages"]["judge"]["calls"] == 1
