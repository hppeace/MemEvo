import asyncio
from collections.abc import Awaitable, Iterable

from tqdm.auto import tqdm

FORMAT = (
    "{desc:<10}: {percentage:3.0f}%|{bar}| "
    "{n_fmt:>3}/{total_fmt:<3} [{elapsed}<{remaining}, {rate_fmt:>15}]"
)


def progress(description: str, total: int, unit: str) -> tqdm:
    return tqdm(
        total=total,
        desc=description,
        unit=unit,
        ncols=100,
        bar_format=FORMAT,
    )


async def gather[T](
    description: str,
    awaitables: Iterable[Awaitable[T]],
    concurrency: int,
) -> list[T]:
    items = list(awaitables)
    semaphore = asyncio.Semaphore(concurrency)
    with progress(description, len(items), "Query") as bar:

        async def tracked(item: Awaitable[T]) -> T:
            async with semaphore:
                try:
                    return await item
                finally:
                    bar.update()

        return await asyncio.gather(*(tracked(item) for item in items))
