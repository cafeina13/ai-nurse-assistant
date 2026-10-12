"""Text bank: varied Turkish wordings for each (category, urgency, topic, speaker) slot.

Banks are built once and committed, so generating a ward stays offline and repeatable:
  data/textbank/train/gemini/  written by Gemini (training data)
  data/textbank/test/claude/   written by Claude (test data; a different model, so a
                               classifier cannot score well by learning one generator's style)
  data/textbank/test/hand/     written by hand

Each bank holds one JSONL file per category; each line is
  {"text", "category", "urgency", "topic", "speaker", "source", "prompt_version"}.
`urgency` there is the urgency of the topic's cell, which is what the wording expresses;
a call's label can still differ through patient context (hard-case rule 6).
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from backend.models import Patient
from wardgen.config import Config, Topic

TEXTBANK_DIR = Path(__file__).resolve().parent.parent / "textbank"
SPEAKERS = ("patient", "companion")

Slot = tuple[str, str, str, str]  # category, urgency, topic, speaker


def slots(config: Config) -> list[Slot]:
    return [(*topic.key, speaker) for topic in config.topics for speaker in SPEAKERS]


def load_bank(bank_dir: Path) -> dict[Slot, list[str]]:
    bank: dict[Slot, list[str]] = defaultdict(list)
    for path in sorted(bank_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    slot = (row["category"], row["urgency"], row["topic"], row["speaker"])
                    bank[slot].append(row["text"])
    return bank


def append_rows(bank_dir: Path, rows: list[dict]) -> None:
    """Appends rows to their category's file."""
    bank_dir.mkdir(parents=True, exist_ok=True)
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    for category, category_rows in by_category.items():
        with (bank_dir / f"{category}.jsonl").open("a", encoding="utf-8", newline="\n") as f:
            for row in category_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    """Key for spotting duplicates: case, Turkish characters and punctuation ignored."""
    table = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    text = text.translate(table).lower()
    return " ".join("".join(ch for ch in text if ch.isalnum() or ch.isspace()).split())


class BankText:
    """Picks a wording for each call from a text bank; falls back to the topic's example."""

    def __init__(self, bank_dir: Path, seed: int):
        self.bank = load_bank(bank_dir)
        self.rng = random.Random(seed)
        if not self.bank:
            raise ValueError(f"text bank {bank_dir} is empty")

    def text_for(self, topic: Topic, speaker: str, patient: Patient) -> str:
        options = self.bank.get((*topic.key, speaker))
        return self.rng.choice(options) if options else topic.example
