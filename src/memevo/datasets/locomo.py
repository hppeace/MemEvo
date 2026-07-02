"""LoCoMo dataset"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tqdm.auto import tqdm

from memevo.datasets.base import BaseDataset
from memevo.utils.models import ChatMessage, ModelPool, OpenAICompatLLM
from memevo.utils.utils import PROGRESS_FORMAT, gather_limited, write_json


@dataclass(frozen=True)
class LocomoMessage:
    dia_id: str
    speaker: str
    text: str
    timestamp_ms: int


@dataclass(frozen=True)
class LocomoSession:
    session_idx: int
    session_datetime: str
    messages: list[LocomoMessage]


@dataclass(frozen=True)
class LocomoQuestion:
    question: str
    answer: str
    evidence: list[str]
    category: int


@dataclass(frozen=True)
class LocomoConversation:
    conv_index: int
    speaker_a: str
    speaker_b: str
    sessions: list[LocomoSession]
    qa: list[LocomoQuestion]

    @property
    def messages(self) -> list[LocomoMessage]:
        return [message for session in self.sessions for message in session.messages]


class LocomoDataset(BaseDataset):
    def __init__(self, path: Path, exclude_category: int | None = 5) -> None:
        self.path = path
        self.exclude_category = exclude_category

    def load(self, conv_index: int) -> LocomoConversation:
        return load_locomo(self.path, conv_index, self.exclude_category)

    async def evaluate(
        self,
        models: ModelPool,
        answers_path: Path,
        output_path: Path,
        concurrency: int = 32,
    ) -> dict[str, float | int]:
        return await judge(
            models.llm("judge"),
            answers_path,
            output_path,
            concurrency,
        )


def parse_session_timestamp(ts_str: str) -> int:
    dt = datetime.strptime(ts_str.strip(), "%I:%M %p on %d %B, %Y")
    return int(dt.replace(tzinfo=UTC).timestamp() * 1000)


def load_locomo(
    data_path: Path, conv_index: int, exclude_category: int | None = 5
) -> LocomoConversation:
    dataset = json.loads(data_path.read_text(encoding="utf-8"))
    if not 0 <= conv_index < len(dataset):
        raise ValueError(
            f"conv_index {conv_index} out of range; dataset has {len(dataset)} conversations"
        )

    conv = dataset[conv_index]
    conversation = conv["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]

    sessions: list[LocomoSession] = []
    session_idx = 1
    while True:
        session_key = f"session_{session_idx}"
        dt_key = f"session_{session_idx}_date_time"
        if dt_key not in conversation:
            break
        if session_key in conversation:
            base_ts_ms = parse_session_timestamp(conversation[dt_key])
            msgs: list[LocomoMessage] = []
            for i, msg in enumerate(conversation[session_key]):
                text = str(msg.get("text", ""))

                # Add image description if available, copy from Mem0-Benchmarks
                query = str(msg.get("query", ""))
                caption = str(msg.get("blip_caption", ""))
                if query and caption:
                    image = (
                        f"[Sharing image - query: {query}. The image shows: {caption}]"
                    )
                elif query:
                    image = f"[Sharing image - query for: {query}]"
                elif caption:
                    image = f"[Sharing image that shows: {caption}]"
                else:
                    image = ""
                text = " ".join(part for part in (text, image) if part)

                if not text:
                    continue
                msgs.append(
                    LocomoMessage(
                        dia_id=str(msg["dia_id"]),
                        speaker=str(msg["speaker"]),
                        text=text,
                        timestamp_ms=base_ts_ms + i * 30000,
                    )
                )
            if msgs:
                sessions.append(
                    LocomoSession(
                        session_idx=session_idx,
                        session_datetime=conversation[dt_key],
                        messages=msgs,
                    )
                )
        session_idx += 1

    qa = [
        LocomoQuestion(
            question=str(item["question"]),
            answer=str(item.get("answer", "")),
            evidence=[str(value) for value in item.get("evidence", [])],
            category=int(item["category"]),
        )
        for item in conv.get("qa", [])
        if exclude_category is None or int(item["category"]) != exclude_category
    ]
    return LocomoConversation(
        conv_index=conv_index,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        sessions=sessions,
        qa=qa,
    )


# Copy from Mem0-Benchmarks and remove the evidence part
# https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/locomo/prompts.py

_JUDGE_SYSTEM_PROMPT = "You are evaluating conversational AI memory recall. Return JSON only with the format requested."

_JUDGE_PROMPT = """Label the generated answer as CORRECT or WRONG.

## Rules

1. **PARTIAL CREDIT**: If the generated answer includes AT LEAST ONE correct item from the gold answer's list, mark CORRECT. Getting 1 out of 2, 2 out of 4, etc. is always acceptable. Only mark WRONG if NONE of the gold answer items appear.

2. **PARAPHRASES COUNT**: Same concept in different words is CORRECT. "Chocolate raspberry tart" = "chocolate cake with raspberries". "Shelter meal service" = "volunteering at a homeless shelter". Emotions and sentiments in the same positive/negative family count as paraphrases: "proud" = "fulfilled" = "accomplished"; "huge success" = "relieved" = "thrilled" (all express positive achievement). Judge semantic meaning, not exact wording.

3. **EXTRA DETAIL IS FINE**: A longer answer that includes the gold answer's key facts plus additional information is CORRECT. Never penalize for being more detailed or specific. If the generated answer adds extra descriptive details beyond the gold answer while still referencing the same core entity or concept, mark CORRECT.

4. **DATE TOLERANCE**: Dates within 14 days of each other are CORRECT. Durations within 50% are CORRECT (e.g., "5 months" matches "six months"; "19 days" matches "two weeks"). Relative dates ("few days before November") match specific dates in the same window. A specific date (e.g., "February 2020") that is consistent with a vague reference (e.g., "a few years ago" relative to 2023) is CORRECT. Converting "last year" to the actual year (e.g., "2022" when conversations are in 2023) is CORRECT.

5. **SEMANTIC OVERLAP**: Judge whether the generated answer addresses the same topic and captures the core idea of the gold answer. Different wording, phrasing, or level of detail should not result in WRONG if the underlying concept matches. For EMOTIONS and FEELINGS questions, answers expressing sentiments in the same valence (positive/negative) about the same event are CORRECT — do not require the exact same emotion word.

6. **SAME REFERENT**: If the generated answer mentions or references the same named entity, character, person, or concept as the gold answer, mark CORRECT — even if the generated answer provides a different physical description or includes additional details. The key question is: does the generated answer identify the same core entity? If yes, it is CORRECT.

7. **FOCUS ON KNOWLEDGE, NOT WORDING**: The goal is to assess whether the system recalled the right fact. Minor differences in specificity, phrasing, or scope should not result in WRONG. Only mark WRONG when the generated answer demonstrates a genuinely different or incorrect understanding.

## ONLY mark WRONG if:
- The generated answer contains ZERO correct items from the gold answer
- The answer addresses a completely different topic

## Question
Question: {question}
Gold answer: {answer}
Generated answer: {response}

Return JSON with "reasoning" (one sentence) and "label" (CORRECT or WRONG). Do NOT include both labels."""


async def judge(
    judge_llm: OpenAICompatLLM,
    answers_path: Path,
    output_path: Path,
    concurrency: int = 32,
) -> dict[str, float | int]:
    payload = json.loads(answers_path.read_text(encoding="utf-8"))
    questions = payload["qa_results"]

    async def judge_one(qa: dict) -> dict:
        try:
            answer = str(qa.get("answer", ""))
            if qa.get("category") == 3:
                answer = answer.split(";", maxsplit=1)[0].strip()
            prompt = _JUDGE_PROMPT.format(
                question=qa["question"],
                answer=answer,
                response=qa.get("response", ""),
            )
            response = await judge_llm.chat(
                [
                    ChatMessage(role="system", content=_JUDGE_SYSTEM_PROMPT),
                    ChatMessage(role="user", content=prompt),
                ],
                response_format={"type": "json_object"},
            )
            label = _parse_label(response.content)
            return {**qa, "result": label, "judge_raw": response.content}
        finally:
            progress.update()

    with tqdm(
        total=len(questions),
        desc="Judge",
        unit="Question",
        ncols=100,
        bar_format=PROGRESS_FORMAT,
    ) as progress:
        results = await gather_limited(
            (judge_one(qa) for qa in questions),
            concurrency,
        )

    total = len(results)
    correct = sum(item["result"] == "CORRECT" for item in results)
    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
    }
    write_json(
        output_path,
        {
            "metrics": metrics,
            "results": results,
        },
    )
    return metrics


def _parse_label(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            label = str(data.get("label", "")).upper()
            if label in {"CORRECT", "WRONG"}:
                return label
        except json.JSONDecodeError:
            pass
    upper = text.upper()
    if "CORRECT" in upper and "WRONG" not in upper:
        return "CORRECT"
    return "WRONG"
