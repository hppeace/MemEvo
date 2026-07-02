import json
from pathlib import Path

import pytest

from memevo.datasets.locomo import _parse_label, load_locomo


def _write_dataset(path: Path) -> None:
    payload = [
        {
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1_date_time": "10:00 AM on 1 January, 2024",
                "session_1": [
                    {"dia_id": "d1", "speaker": "Alice", "text": "Hello"},
                    {"dia_id": "d2", "speaker": "Bob", "text": "Hi"},
                ],
            },
            "qa": [
                {
                    "question": "Who greeted Bob?",
                    "answer": "Alice",
                    "evidence": ["d1"],
                    "category": 1,
                },
                {
                    "question": "Adversarial?",
                    "answer": "",
                    "evidence": [],
                    "category": 5,
                },
            ],
        }
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_locomo_parses_messages_and_filters_questions(tmp_path):
    data_path = tmp_path / "locomo.json"
    _write_dataset(data_path)

    conversation = load_locomo(data_path, 0)

    assert conversation.speaker_a == "Alice"
    assert conversation.sessions[0].session_datetime.startswith("10:00 AM")
    assert [message.dia_id for message in conversation.messages] == ["d1", "d2"]
    assert (
        conversation.messages[1].timestamp_ms - conversation.messages[0].timestamp_ms
        == 30_000
    )
    assert [question.category for question in conversation.qa] == [1]


def test_load_locomo_rejects_negative_index(tmp_path):
    data_path = tmp_path / "locomo.json"
    _write_dataset(data_path)

    with pytest.raises(ValueError, match="out of range"):
        load_locomo(data_path, -1)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"reasoning": "matches", "label": "CORRECT"}', "CORRECT"),
        ("The answer is CORRECT.", "CORRECT"),
        ('{"label": "WRONG"}', "WRONG"),
        ("unclear response", "WRONG"),
    ],
)
def test_parse_label(response, expected):
    assert _parse_label(response) == expected
