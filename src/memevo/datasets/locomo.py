"""LoCoMo dataset"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memevo.utils.models import LLM, Usage
from memevo.utils.progress import gather


@dataclass(frozen=True)
class LocomoMessage:
    speaker: str
    text: str
    timestamp_ms: int


@dataclass(frozen=True)
class LocomoSession:
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
    speaker_a: str
    sessions: list[LocomoSession]
    qa: list[LocomoQuestion]

    @property
    def messages(self) -> list[LocomoMessage]:
        return [message for session in self.sessions for message in session.messages]


class LocomoDataset:
    def __init__(
        self,
        path: Path,
        indices: Sequence[int],
        judge_llm: LLM,
        exclude_category: int | None = 5,
    ) -> None:
        self.path = path
        self.indices = indices
        self.judge_llm = judge_llm
        self.exclude_category = exclude_category

    def load(self, conv_index: int) -> LocomoConversation:
        return load_locomo(self.path, conv_index, self.exclude_category)

    def questions(self, conversation: LocomoConversation) -> list[LocomoQuestion]:
        return conversation.qa

    def question_text(self, question: LocomoQuestion) -> str:
        return question.question

    def result(
        self,
        conv_index: int,
        question: LocomoQuestion,
        response: str,
    ) -> dict[str, Any]:
        return {
            "conv_index": conv_index,
            "question": question.question,
            "answer": question.answer,
            "response": response,
            "category": question.category,
            "evidence": question.evidence,
        }

    async def evaluate(
        self,
        answers_path: Path,
        output_path: Path,
        concurrency: int = 32,
    ) -> dict[str, float | int]:
        return await judge(
            self.judge_llm,
            answers_path,
            output_path,
            concurrency,
        )

    async def close(self) -> None:
        await self.judge_llm.close()


def create(
    settings: Mapping[str, Any],
    models: Mapping[str, Any],
    usage: Usage,
) -> LocomoDataset:
    exclude = settings.get("exclude_category", 5)
    return LocomoDataset(
        path=Path(str(settings["path"])),
        indices=list(settings["conv_indices"]),
        judge_llm=LLM("judge", models["judge"], usage),
        exclude_category=None if exclude is None else int(exclude),
    )


def load_locomo(
    data_path: Path, conv_index: int, exclude_category: int | None = 5
) -> LocomoConversation:
    dataset = json.loads(data_path.read_text(encoding="utf-8"))
    conv = dataset[conv_index]
    conversation = conv["conversation"]
    speaker_a = conversation["speaker_a"]

    sessions: list[LocomoSession] = []
    session_idx = 1
    while True:
        session_key = f"session_{session_idx}"
        dt_key = f"session_{session_idx}_date_time"
        if dt_key not in conversation:
            break
        if session_key in conversation:
            session_time = datetime.strptime(
                conversation[dt_key].strip(),
                "%I:%M %p on %d %B, %Y",
            )
            base_ts_ms = int(session_time.replace(tzinfo=UTC).timestamp() * 1000)
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
                        speaker=str(msg["speaker"]),
                        text=text,
                        timestamp_ms=base_ts_ms + i * 30000,
                    )
                )
            if msgs:
                sessions.append(
                    LocomoSession(
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
        speaker_a=speaker_a,
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
    judge_llm: LLM,
    answers_path: Path,
    output_path: Path,
    concurrency: int = 32,
) -> dict[str, float | int]:
    payload = json.loads(answers_path.read_text(encoding="utf-8"))
    questions = payload["qa_results"]

    async def judge_one(qa: dict) -> dict:
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
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        label = _parse_label(response)
        return {**qa, "result": label, "judge_raw": response}

    results = await gather("Judge", (judge_one(qa) for qa in questions), concurrency)

    total = len(results)
    correct = sum(item["result"] == "CORRECT" for item in results)
    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
    }
    output_path.write_text(
        json.dumps(
            {"metrics": metrics, "results": results},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
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
