"""Benchmark dataset registry."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from memevo.datasets.base import BaseDataset
from memevo.datasets.locomo import LocomoDataset

DatasetFactory = Callable[[Mapping[str, Any]], BaseDataset]
_DATASETS: dict[str, DatasetFactory] = {}


def register_dataset(name: str, factory: DatasetFactory) -> None:
    _DATASETS[name] = factory


def create_dataset(name: str, settings: Mapping[str, Any]) -> BaseDataset:
    factory = _DATASETS.get(name)
    if factory is None:
        choices = ", ".join(sorted(_DATASETS))
        raise ValueError(f"Unknown dataset '{name}'. Available: {choices}") from None
    return factory(settings)


def _create_locomo(settings: Mapping[str, Any]) -> LocomoDataset:
    exclude = settings.get("exclude_category", 5)
    return LocomoDataset(
        path=Path(str(settings["path"])),
        exclude_category=None if exclude is None else int(exclude),
    )


register_dataset("locomo", _create_locomo)

__all__ = ["BaseDataset", "create_dataset", "register_dataset"]
