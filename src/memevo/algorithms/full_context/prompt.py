from datetime import UTC, datetime

ANSWER_PROMPT = """Use the conversation below to answer the question.
Preserve names, numbers, specific dates, places, and other specific details. If the
answer is absent, say that the conversation does not provide it.

Conversation:
{context}

Question: {question}

Return only a concise answer."""


def prepare_answer_prompt(memory: list[dict], question: str) -> str:
    context = "\n".join(
        f"[{_format_timestamp(item['timestamp_ms'])}] {item['speaker']}: {item['text']}"
        for item in memory
    )
    return ANSWER_PROMPT.format(context=context, question=question)


def _format_timestamp(timestamp_ms: int) -> str:
    timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
    hour = timestamp.hour % 12 or 12
    period = "am" if timestamp.hour < 12 else "pm"
    return (
        f"{hour}:{timestamp.minute:02d} {period} on "
        f"{timestamp.day} {timestamp:%B}, {timestamp.year}"
    )
