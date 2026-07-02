import asyncio
import json
from collections.abc import Awaitable, Iterable
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

PROGRESS_FORMAT = (
    "{desc:<24}: {percentage:3.0f}%|{bar}| "
    "{n_fmt:>3}/{total_fmt:<3} [{elapsed}<{remaining}, {rate_fmt:>16}]"
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


async def gather_limited(
    awaitables: Iterable[Awaitable[T]], limit: int = 32
) -> list[T]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    semaphore = asyncio.Semaphore(limit)

    async def run(awaitable: Awaitable[T]) -> T:
        async with semaphore:
            return await awaitable

    return await asyncio.gather(*(run(awaitable) for awaitable in awaitables))
