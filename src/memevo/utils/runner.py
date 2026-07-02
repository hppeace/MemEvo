from __future__ import annotations

import argparse
import asyncio
import os
import tomllib
from collections.abc import Awaitable, Iterable, Mapping, Sequence
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
    run = _mapping(payload.get("run"), "run")
    dataset = dict(_mapping(payload.get("dataset"), "dataset"))
    algorithm = dict(_mapping(payload.get("algorithm"), "algorithm"))
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
        _write_output(answers_path, config.name, qa_results=results)
        _write_output(
            usage_path,
            config.name,
            model_usage=models.usage_summary(),
        )
        for conv_index in config.conv_indices:
            conversation = dataset.load(conv_index)
            with _progress(f"Ingest Conversation {conv_index}", 1, "Step") as progress:
                await algorithm.ingest(conv_index, conversation)
                progress.update()

            memories = await _gather_with_progress(
                f"Retrieve Conversation {conv_index}",
                (
                    algorithm.retrieve(conv_index, item.question)
                    for item in conversation.qa
                ),
                len(conversation.qa),
                config.concurrency,
            )
            responses = await _gather_with_progress(
                f"Answer Conversation {conv_index}",
                (
                    algorithm.answer(item.question, memory)
                    for item, memory in zip(conversation.qa, memories, strict=True)
                ),
                len(conversation.qa),
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
            _write_output(answers_path, config.name, qa_results=results)
            _write_output(
                usage_path,
                config.name,
                model_usage=models.usage_summary(),
            )

        metrics = await dataset.evaluate(
            models,
            answers_path,
            evaluation_path,
            config.concurrency,
        )
        return metrics
    finally:
        _write_output(
            usage_path,
            config.name,
            model_usage=models.usage_summary(),
        )
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


def _write_output(path: Path, run_name: str, **data: Any) -> None:
    write_json(path, {"run_name": run_name, **data})


def _progress(description: str, total: int, unit: str) -> tqdm:
    return tqdm(
        total=total,
        desc=description,
        unit=unit,
        ncols=100,
        bar_format=PROGRESS_FORMAT,
    )


async def _gather_with_progress[T](
    description: str,
    awaitables: Iterable[Awaitable[T]],
    total: int,
    concurrency: int,
) -> list[T]:
    with _progress(description, total, "Question") as progress:

        async def tracked(awaitable: Awaitable[T]) -> T:
            try:
                return await awaitable
            finally:
                progress.update()

        return await gather_limited(
            (tracked(awaitable) for awaitable in awaitables),
            concurrency,
        )


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


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"Missing or invalid [{name}] section")
    return value


def _required(payload: Mapping[str, Any], key: str) -> Any:
    value = payload.get(key)
    if value is None or value == "":
        raise ConfigError(f"Missing required setting: {key}")
    return value
