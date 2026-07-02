from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from memevo.utils.models import ModelPool


class BaseDataset(ABC):
    @abstractmethod
    def load(self, conv_index: int) -> Any:
        """Load one benchmark conversation."""

    @abstractmethod
    async def evaluate(
        self,
        models: ModelPool,
        answers_path: Path,
        output_path: Path,
        concurrency: int = 32,
    ) -> dict[str, float | int]:
        """Evaluate saved answers and return aggregate metrics."""
