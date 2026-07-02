"""LoCoMo dataset"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


from utils.utils import write_json
from utils.clients import OpenAICompatLLM, ChatMessage

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


def parse_session_timestamp(ts_str: str) -> int:
    dt = datetime.strptime(ts_str.strip(), "%I:%M %p on %d %B, %Y")
    return int(dt.replace(tzinfo=UTC).timestamp() * 1000)


def load_locomo(data_path: Path, conv_index: int) -> LocomoConversation:
    dataset = json.loads(data_path.read_text(encoding="utf-8"))
    if conv_index >= len(dataset):
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
                text = msg.get("text")
                if not text:
                    continue
                msgs.append(
                    LocomoMessage(
                        dia_id=str(msg["dia_id"]),
                        speaker=str(msg["speaker"]),
                        text=str(text),
                        timestamp_ms=base_ts_ms + i * 30000,
                    )
                )
            if msgs:
                sessions.append(LocomoSession(session_idx=session_idx, messages=msgs))
        session_idx += 1

    qa = [q for q in conv.get("qa", []) if q.get("category") != 5]
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
Question: {{question}}
Gold answer: {{answer}}
Generated answer: {{response}}

Return JSON with "reasoning" (one sentence) and "label" (CORRECT or WRONG). Do NOT include both labels."""


# TODO parllelize the judge function to speed up the evaluation process
async def judge(judge_llm: OpenAICompatLLM, answers_path: Path, output_path: Path) -> None:
    payload = json.loads(answers_path.read_text(encoding="utf-8"))
    results = []
    correct = 0
    for qa in payload["qa_results"]:
        prompt = _JUDGE_PROMPT.format(
            question=qa["question"],
            answer=qa.get("answer", ""),
            response=qa.get("response", ""),
        )
        with judge_llm.usage_stage("Judge"):
            resp = await judge_llm.chat(
                [
                    ChatMessage(role="system", content=_JUDGE_SYSTEM_PROMPT),
                    ChatMessage(role="user", content=prompt),
                ]
            )
        label = _parse_label(resp.content)
        correct += int(label == "CORRECT")
        results.append({**qa, "result": label, "judge_raw": resp.content})

    total = len(results)
    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
    }
    write_json(
        output_path,
        {
            "metrics": metrics,
            "judge_llm_usage": judge_llm.usage_summary(),
            "results": results,
        },
    )


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