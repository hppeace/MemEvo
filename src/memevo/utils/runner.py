import argparse
import asyncio
import json
import tomllib
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from memevo.utils.models import Usage
from memevo.utils.progress import gather, progress


async def run_benchmark(config: Mapping[str, Any]) -> dict[str, float | int]:
    run = config["run"]
    models = config["models"]
    output_dir = Path(run["output_dir"])
    concurrency = int(run["concurrency"])
    output_dir.mkdir(parents=True, exist_ok=True)

    usage = Usage(models)
    dataset_settings = config["dataset"]
    dataset = import_module(f"memevo.datasets.{dataset_settings['name']}").create(
        dataset_settings, models, usage
    )
    algorithm_settings = config["algorithm"]
    algorithm = import_module(f"memevo.algorithms.{algorithm_settings['name']}").create(
        algorithm_settings,
        models,
        usage,
        output_dir / "memory",
    )
    algorithm.reset_all()

    answers_path = output_dir / "answers.json"
    evaluation_path = output_dir / "evaluation.json"
    usage_path = output_dir / "usage.json"
    results: list[dict[str, Any]] = []

    try:
        _write_json(answers_path, {"run_name": run["name"], "qa_results": results})
        for conv_index in dataset.indices:
            print(f"Conversation {conv_index} :", flush=True)
            conversation = dataset.load(conv_index)
            questions = dataset.questions(conversation)
            queries = [dataset.question_text(item) for item in questions]

            with progress("Ingest", 1, "Step") as ingest_bar:
                with usage.stage("ingest"):
                    await algorithm.ingest(conv_index, conversation)
                ingest_bar.update()

            with usage.stage("retrieve"):
                memories = await gather(
                    "Retrieve",
                    (algorithm.retrieve(conv_index, query) for query in queries),
                    concurrency,
                )
            with usage.stage("answer"):
                responses = await gather(
                    "Answer",
                    (
                        algorithm.answer(query, memory)
                        for query, memory in zip(queries, memories, strict=True)
                    ),
                    concurrency,
                )

            results.extend(
                dataset.result(conv_index, question, response)
                for question, response in zip(questions, responses, strict=True)
            )
            _write_json(
                answers_path,
                {"run_name": run["name"], "qa_results": results},
            )
            _write_json(
                usage_path,
                {"run_name": run["name"], "model_usage": usage.summary()},
            )

        print("\nEvaluation :", flush=True)
        with usage.stage("judge"):
            metrics = await dataset.evaluate(
                answers_path,
                evaluation_path,
                concurrency,
            )
        return metrics
    finally:
        _write_json(
            usage_path,
            {"run_name": run["name"], "model_usage": usage.summary()},
        )
        await algorithm.close()
        await dataset.close()


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    load_dotenv()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    asyncio.run(run_benchmark(config))
    return 0


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
