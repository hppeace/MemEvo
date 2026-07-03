from collections.abc import Mapping
from pathlib import Path
from typing import Any

from memevo.algorithms.mem0.algorithm import Mem0
from memevo.utils.models import Embedder, LLM, Usage


def create(
    settings: Mapping[str, Any],
    models: Mapping[str, Any],
    usage: Usage,
    working_dir: Path,
) -> Mem0:
    config = settings.get("config", {})
    return Mem0(
        LLM("answer", models["answer"], usage),
        LLM("memory", models["memory"], usage),
        Embedder("embedding", models["embedding"], usage),
        working_dir,
        config,
        top_k=int(settings.get("top_k", 200)),
        cutoff=int(settings.get("cutoff", 10)),
        rerank=bool(settings.get("rerank", False)),
        embedding_dims=int(settings.get("embedding_dims", 1536)),
    )
