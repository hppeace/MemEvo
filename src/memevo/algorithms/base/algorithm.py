from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any


class BaseAlgorithm(ABC):
    @abstractmethod
    async def ingest(self, conv_index: int, messages: Sequence[Any]) -> None:
        """Store all messages from one conversation."""

    @abstractmethod
    async def retrieve(self, conv_index: int, question: str) -> Any:
        """Retrieve question-relevant memory."""

    @abstractmethod
    async def answer(self, question: str, memory: Any) -> str:
        """Answer a question using retrieved memory."""

    @abstractmethod
    def reset_all(self) -> None:
        """Remove all stored memories."""
