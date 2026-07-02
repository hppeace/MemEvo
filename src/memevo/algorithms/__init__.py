"""Memory algorithm registry."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from memevo.algorithms.base import BaseAlgorithm
from memevo.algorithms.full_context import FullContext
from memevo.utils.models import ModelPool

AlgorithmFactory = Callable[[ModelPool, Path, Mapping[str, Any]], BaseAlgorithm]
_ALGORITHMS: dict[str, AlgorithmFactory] = {}


def register_algorithm(name: str, factory: AlgorithmFactory) -> None:
    _ALGORITHMS[name] = factory


def create_algorithm(
    name: str,
    models: ModelPool,
    working_dir: Path,
    settings: Mapping[str, Any],
) -> BaseAlgorithm:
    factory = _ALGORITHMS.get(name)
    if factory is None:
        choices = ", ".join(sorted(_ALGORITHMS))
        raise ValueError(f"Unknown algorithm '{name}'. Available: {choices}") from None
    return factory(models, working_dir, settings)


def _create_mem0(
    models: ModelPool,
    working_dir: Path,
    settings: Mapping[str, Any],
) -> BaseAlgorithm:
    from memevo.algorithms.mem0 import Mem0

    config = settings.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError("algorithm.config must be a TOML table")
    return Mem0(
        models.llm("answer"),
        working_dir,
        config,
        top_k=int(settings.get("top_k", 200)),
        cutoff=int(settings.get("cutoff", 10)),
        rerank=bool(settings.get("rerank", False)),
    )


register_algorithm(
    "full_context",
    lambda models, working_dir, _: FullContext(models.llm("answer"), working_dir),
)
register_algorithm("mem0", _create_mem0)

__all__ = ["BaseAlgorithm", "create_algorithm", "register_algorithm"]
