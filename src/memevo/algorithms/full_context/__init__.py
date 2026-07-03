from collections.abc import Mapping
from pathlib import Path
from typing import Any

from memevo.algorithms.full_context.algorithm import FullContext
from memevo.utils.models import LLM, Usage


def create(
    settings: Mapping[str, Any],
    models: Mapping[str, Any],
    usage: Usage,
    working_dir: Path,
) -> FullContext:
    return FullContext(LLM("answer", models["answer"], usage), working_dir)
