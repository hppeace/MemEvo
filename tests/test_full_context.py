import asyncio
from dataclasses import dataclass

from memevo.algorithms.full_context import FullContext


@dataclass(frozen=True)
class Message:
    speaker: str
    text: str
    timestamp_ms: int


class FakeLLM:
    def __init__(self) -> None:
        self.prompt = ""

    async def chat(self, messages):
        self.prompt = messages[-1].content
        return type("Response", (), {"content": " Paris. "})()


def test_full_context_round_trip(tmp_path):
    llm = FakeLLM()
    algorithm = FullContext(llm, tmp_path / "memory")
    messages = [Message("Alice", "I moved to Paris.", 1000)]

    async def exercise():
        await algorithm.ingest(2, messages)
        memory = await algorithm.retrieve(2, "Where did Alice move?")
        assert memory == [
            {"speaker": "Alice", "text": "I moved to Paris.", "timestamp_ms": 1000}
        ]
        assert await algorithm.answer("Where did Alice move?", memory) == "Paris."

    asyncio.run(exercise())
    assert "Alice: I moved to Paris." in llm.prompt
    assert "[12:00 am on 1 January, 1970]" in llm.prompt
    algorithm.reset_all()
    assert not (tmp_path / "memory" / "conv_2").exists()


def test_retrieve_requires_ingest(tmp_path):
    algorithm = FullContext(FakeLLM(), tmp_path)

    async def exercise():
        try:
            await algorithm.retrieve(9)
        except FileNotFoundError as error:
            assert "call ingest first" in str(error)
        else:
            raise AssertionError("retrieve should fail when no memory exists")

    asyncio.run(exercise())
