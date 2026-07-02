import asyncio
import json
from contextlib import nullcontext

import pytest

from memevo.utils import runner
from memevo.utils.runner import (
    BenchmarkConfig,
    ConfigError,
    ModelConfig,
    load_config,
    parse_indices,
    run_benchmark,
)


def test_parse_indices_supports_ranges_and_deduplication():
    assert parse_indices("0-2, 2, 5") == [0, 1, 2, 5]
    assert parse_indices([3, 1, 3]) == [1, 3]


def test_parse_indices_rejects_reversed_range():
    with pytest.raises(ConfigError, match="Invalid conversation range"):
        parse_indices("3-1")


def test_load_config_resolves_models_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ANSWER_KEY", "answer-secret")
    monkeypatch.setenv("JUDGE_KEY", "judge-secret")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[run]
name = "test"
output_dir = "runs/test"

[dataset]
name = "locomo"
path = "data/test.json"
conv_indices = "0-1"
exclude_category = 5

[algorithm]
name = "full_context"

[models.answer]
model = "answer-model"
api_key_env = "ANSWER_KEY"

[models.judge]
model = "judge-model"
api_key_env = "JUDGE_KEY"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.conv_indices == [0, 1]
    assert config.models["answer"].api_key == "answer-secret"
    assert config.dataset_settings["path"] == "data/test.json"
    assert config.algorithm_name == "full_context"
    assert config.concurrency == 32


def test_run_benchmark_end_to_end_without_network(tmp_path, monkeypatch):
    data_path = tmp_path / "locomo.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "conversation": {
                        "speaker_a": "Alice",
                        "speaker_b": "Bob",
                        "session_1_date_time": "10:00 AM on 1 January, 2024",
                        "session_1": [
                            {
                                "dia_id": "d1",
                                "speaker": "Alice",
                                "text": "I live in Paris.",
                            }
                        ],
                    },
                    "qa": [
                        {
                            "question": "Where does Alice live?",
                            "answer": "Paris",
                            "evidence": ["d1"],
                            "category": 1,
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    class FakeLLM:
        def __init__(self, model, **_):
            self.model = model
            self.stages = set()

        async def chat(self, _messages):
            content = (
                '{"reasoning": "matches", "label": "CORRECT"}'
                if self.model == "judge"
                else "Paris"
            )
            return type("Response", (), {"content": content})()

        def stage(self, name):
            self.stages.add(name)
            return nullcontext()

        def usage_summary(self):
            return {
                "total": {},
                "stages": {name: {} for name in sorted(self.stages)},
            }

        async def close(self):
            pass

    monkeypatch.setattr(runner, "OpenAICompatLLM", FakeLLM)
    output_dir = tmp_path / "run"
    config = BenchmarkConfig(
        name="integration",
        output_dir=output_dir,
        conv_indices=[0],
        dataset_name="locomo",
        dataset_settings={"path": str(data_path), "exclude_category": 5},
        algorithm_name="full_context",
        algorithm_settings={},
        models={
            "answer": ModelConfig(model="answer", api_key="unused"),
            "judge": ModelConfig(model="judge", api_key="unused"),
        },
    )

    metrics = asyncio.run(run_benchmark(config))

    assert metrics == {"total": 1, "correct": 1, "accuracy": 1.0}
    evaluation = json.loads((output_dir / "evaluation.json").read_text())
    assert evaluation["results"][0]["result"] == "CORRECT"
    answers = json.loads((output_dir / "answers.json").read_text())
    assert set(answers["model_usage"]["answer"]["stages"]) == {
        "ingest",
        "retrieve",
        "answer",
        "judge",
    }
