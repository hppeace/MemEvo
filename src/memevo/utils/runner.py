from __future__ import annotations

import argparse
import asyncio
import os
import tomllib
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm.auto import tqdm

from memevo.algorithms import create_algorithm
from memevo.datasets import create_dataset
from memevo.utils.models import ModelPool, OpenAICompatLLM, OpenAIEmbedder
from memevo.utils.utils import PROGRESS_FORMAT, gather_limited, write_json


@dataclass(frozen=True)
class ModelConfig:
    model: str
    api_key: str
    kind: str = "llm"
    base_url: str | None = None
    temperature: float = 0.0


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    output_dir: Path
    conv_indices: list[int]
    dataset_name: str
    dataset_settings: dict[str, Any]
    algorithm_name: str
    algorithm_settings: dict[str, Any]
    models: dict[str, ModelConfig]
    concurrency: int = 32


class ConfigError(ValueError):
    """Raised when a benchmark configuration is incomplete or invalid."""


def parse_indices(value: str | Sequence[int]) -> list[int]:
    """Parse values such as ``0-3,7`` into sorted, unique indices."""
    if not isinstance(value, str):
        return sorted(set(int(item) for item in value))

    indices: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            indices.add(int(part))
            continue
        start_text, end_text = part.split("-", maxsplit=1)
        start, end = int(start_text), int(end_text)
        if start > end:
            raise ConfigError(f"Invalid conversation range: {part}")
        indices.update(range(start, end + 1))
    if not indices:
        raise ConfigError("dataset.conv_indices must not be empty")
    return sorted(indices)


def load_config(path: Path) -> BenchmarkConfig:
    load_dotenv()
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    run = _section(payload, "run")
    dataset = dict(_section(payload, "dataset"))
    algorithm = dict(_section(payload, "algorithm"))
    model_sections = payload.get("models", algorithm.pop("models", None))
    if not isinstance(model_sections, Mapping):
        raise ConfigError("Missing or invalid [models] section")

    dataset_name = str(dataset.pop("name", "locomo"))
    conv_indices = parse_indices(_required(dataset, "conv_indices"))
    dataset.pop("conv_indices")
    algorithm_name = str(algorithm.pop("name", "full_context"))
    models = {
        name: _model_config(_mapping(settings, f"models.{name}"), name)
        for name, settings in model_sections.items()
    }
    concurrency = int(run.get("concurrency", 32))
    if concurrency < 1:
        raise ConfigError("run.concurrency must be at least 1")
    return BenchmarkConfig(
        name=str(run.get("name", path.stem)),
        output_dir=Path(_required(run, "output_dir")),
        conv_indices=conv_indices,
        dataset_name=dataset_name,
        dataset_settings=dataset,
        algorithm_name=algorithm_name,
        algorithm_settings=algorithm,
        models=models,
        concurrency=concurrency,
    )


async def run_benchmark(config: BenchmarkConfig) -> dict[str, float | int]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    models = _create_models(config.models)
    dataset = create_dataset(config.dataset_name, config.dataset_settings)
    algorithm = create_algorithm(
        config.algorithm_name,
        models,
        config.output_dir / "memory",
        config.algorithm_settings,
    )
    algorithm.reset_all()
    answers_path = config.output_dir / "answers.json"
    evaluation_path = config.output_dir / "evaluation.json"
    usage_path = config.output_dir / "usage.json"
    results: list[dict[str, Any]] = []

    try:
        _write_answers(answers_path, config, results)
        _write_usage(usage_path, config, models)
        for conv_index in config.conv_indices:
            conversation = dataset.load(conv_index)
            with tqdm(
                total=1,
                desc=f"Ingest Conversation {conv_index}",
                unit="Step",
                ncols=100,
                bar_format=PROGRESS_FORMAT,
            ) as progress:
                with models.stage("ingest"):
                    await algorithm.ingest(conv_index, conversation.messages)
                progress.update()

            with tqdm(
                total=len(conversation.qa),
                desc=f"Retrieve Conversation {conv_index}",
                unit="Question",
                ncols=100,
                bar_format=PROGRESS_FORMAT,
            ) as progress:
                with models.stage("retrieve"):
                    memories = await gather_limited(
                        (
                            _tracked(
                                algorithm.retrieve(conv_index, item.question),
                                progress,
                            )
                            for item in conversation.qa
                        ),
                        config.concurrency,
                    )

            with tqdm(
                total=len(conversation.qa),
                desc=f"Answer Conversation {conv_index}",
                unit="Question",
                ncols=100,
                bar_format=PROGRESS_FORMAT,
            ) as progress:
                with models.stage("answer"):
                    responses = await gather_limited(
                        (
                            _tracked(
                                algorithm.answer(item.question, memory),
                                progress,
                            )
                            for item, memory in zip(
                                conversation.qa, memories, strict=True
                            )
                        ),
                        config.concurrency,
                    )

            for item, response in zip(conversation.qa, responses, strict=True):
                results.append(
                    {
                        "conv_index": conv_index,
                        "question": item.question,
                        "answer": item.answer,
                        "response": response,
                        "category": item.category,
                        "evidence": item.evidence,
                    }
                )
            _write_answers(answers_path, config, results)
            _write_usage(usage_path, config, models)

        with models.stage("judge"):
            metrics = await dataset.evaluate(
                models,
                answers_path,
                evaluation_path,
                config.concurrency,
            )
        _write_usage(usage_path, config, models)
        return metrics
    finally:
        _write_usage(usage_path, config, models)
        await models.close()


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a MemEvo memory benchmark")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    asyncio.run(run_benchmark(load_config(args.config)))
    return 0


def _create_models(configs: Mapping[str, ModelConfig]) -> ModelPool:
    clients = {}
    for name, config in configs.items():
        common = {
            "api_key": config.api_key,
            "model": config.model,
            "base_url": config.base_url,
        }
        if config.kind == "llm":
            clients[name] = OpenAICompatLLM(**common, temperature=config.temperature)
        elif config.kind == "embedding":
            clients[name] = OpenAIEmbedder(**common)
        else:
            raise ConfigError(f"Unknown model type '{config.kind}' for {name}")
    return ModelPool(clients)


def _write_answers(
    path: Path,
    config: BenchmarkConfig,
    results: list[dict[str, Any]],
) -> None:
    write_json(
        path,
        {
            "run_name": config.name,
            "qa_results": results,
        },
    )


def _write_usage(
    path: Path,
    config: BenchmarkConfig,
    models: ModelPool,
) -> None:
    write_json(
        path,
        {
            "run_name": config.name,
            "model_usage": models.usage_summary(),
        },
    )


async def _tracked(awaitable: Awaitable[Any], progress: Any) -> Any:
    try:
        return await awaitable
    finally:
        progress.update()


def _model_config(section: Mapping[str, Any], name: str) -> ModelConfig:
    env_name = str(_required(section, "api_key_env"))
    api_key = os.getenv(env_name)
    if not api_key:
        raise ConfigError(f"Environment variable {env_name} is required for {name}")
    base_url = section.get("base_url")
    return ModelConfig(
        model=str(_required(section, "model")),
        api_key=api_key,
        kind=str(section.get("type", "llm")),
        base_url=str(base_url) if base_url else None,
        temperature=float(section.get("temperature", 0.0)),
    )


def _section(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _mapping(payload.get(key), key)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"Missing or invalid [{name}] section")
    return value


def _required(payload: Mapping[str, Any], key: str) -> Any:
    value = payload.get(key)
    if value is None or value == "":
        raise ConfigError(f"Missing required setting: {key}")
    return value
