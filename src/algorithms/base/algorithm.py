from typing import Any
from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):

    @abstractmethod
    async def ingest(self, *args, **kwargs) -> Any:
        """
        Ingest data into the memory.
        """
        pass
    
    @abstractmethod
    async def retrieve(self, *args, **kwargs) -> Any:
        """
        Retrieve data from the memory.
        """
        pass

    @abstractmethod
    async def answer(self, *args, **kwargs) -> Any:
        """
        Provide an answer based on the memory.
        """
        pass

    @abstractmethod
    def reset_all(self) -> None:
        """
        Reset all memories to their initial state.
        """
        pass