from collections.abc import Mapping
from pathlib import Path
from typing import Any

from memevo.algorithms.everos.algorithm import EverOS
from memevo.utils.models import Embedder, LLM, Usage


def create(
    settings: Mapping[str, Any],
    models: Mapping[str, Any],
    usage: Usage,
    working_dir: Path,
) -> EverOS:
    answer_llm = LLM("answer", models["answer"], usage)
    memory_llm = LLM("memory", models["memory"], usage)
    embedder = Embedder("embedding", models["embedding"], usage)
    answer_llm.client = answer_llm.client.with_options(max_retries=1)
    memory_llm.client = memory_llm.client.with_options(timeout=60.0)
    embedder.client = embedder.client.with_options(timeout=30.0, max_retries=3)
    return EverOS(
        answer_llm,
        memory_llm,
        embedder,
        working_dir,
        models["rerank"],
        method=str(settings.get("method", "agentic")),
        top_k=int(settings.get("top_k", 10)),
        eval_owner=str(settings.get("eval_owner", "speaker_a")),
        batch_size=int(settings.get("batch_size", 25)),
        search_concurrency=int(settings.get("search_concurrency", 5)),
        answer_max_tokens=int(settings.get("answer_max_tokens", 32768)),
        answer_timeout=float(settings.get("answer_timeout", 300.0)),
        answer_max_retries=int(settings.get("answer_max_retries", 5)),
        ready_timeout=float(settings.get("ready_timeout", 7200.0)),
    )
