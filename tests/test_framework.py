import asyncio
from pathlib import Path
from typing import Any

from memevo.algorithms.full_context import FullContext, create as create_algorithm
from memevo.datasets.locomo import (
    LocomoDataset,
    LocomoQuestion,
    create as create_dataset,
)
from memevo.utils.models import Usage


def test_components_load_from_config_name(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("API_KEY", "key")
    models = {
        "answer": {"model": "answer", "api_key_env": "API_KEY"},
        "judge": {"model": "judge", "api_key_env": "API_KEY"},
    }
    usage = Usage(list(models))

    algorithm = create_algorithm(
        {"name": "full_context"},
        models,
        usage,
        tmp_path,
    )
    dataset = create_dataset(
        {"name": "locomo", "path": "data.json", "conv_indices": [0]},
        models,
        usage,
    )

    assert isinstance(algorithm, FullContext)
    assert isinstance(dataset, LocomoDataset)
    asyncio.run(algorithm.close())
    asyncio.run(dataset.close())


def test_dataset_result_preserves_locomo_output_shape() -> None:
    dataset = object.__new__(LocomoDataset)
    question = LocomoQuestion(
        question="Where?",
        answer="At home",
        evidence=["D1:1"],
        category=1,
    )

    assert dataset.result(3, question, "Home") == {
        "conv_index": 3,
        "question": "Where?",
        "answer": "At home",
        "response": "Home",
        "category": 1,
        "evidence": ["D1:1"],
    }
