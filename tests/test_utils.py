import asyncio

from memevo.utils.utils import gather_limited


def test_gather_limited_preserves_order_and_limit():
    active = 0
    peak = 0

    async def worker(value):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return value * 2

    results = asyncio.run(
        gather_limited((worker(value) for value in range(8)), limit=3)
    )

    assert results == [value * 2 for value in range(8)]
    assert peak == 3
