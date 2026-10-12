"""Text bank: varied Turkish wordings for each (intent, wording urgency, speaker) cell.

Banks are built once and committed, so generating a ward stays offline and repeatable:
  data/textbank/train/gemini/  written by Gemini (training data)
  data/textbank/test/claude/   written by Claude (test data; a different model, so a
                               classifier cannot score well by learning one generator's style)
  data/textbank/test/hand/     written by hand

Each bank holds one JSONL file per intent; each line is
  {"text", "intent", "urgency", "speaker", "source", "prompt_version"}.
`urgency` there is the urgency the WORDING expresses; see wording_urgencies().
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from backend.models import Patient
from wardgen.config import Intent

TEXTBANK_DIR = Path(__file__).resolve().parent.parent / "textbank"
SPEAKERS = ("patient", "companion")


def wording_urgencies(intent: Intent) -> list[str]:
    """The urgencies a request's words can express for this intent.

    routine_assisted wording is always routine: whether the patient can walk alone is not
    in the words, so the label (soon for assisted patients) needs patient context.
    """
    match intent.urgency:
        case "soon_plus":
            return ["soon", "critical"]
        case "routine_assisted":
            return ["routine"]
        case urgency:
            return [urgency]


def cells(intents: dict[str, Intent]) -> list[tuple[str, str, str]]:
    return [
        (intent.id, urgency, speaker)
        for intent in intents.values()
        for urgency in wording_urgencies(intent)
        for speaker in SPEAKERS
    ]


def load_bank(bank_dir: Path) -> dict[tuple[str, str, str], list[str]]:
    bank: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for path in sorted(bank_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    bank[(row["intent"], row["urgency"], row["speaker"])].append(row["text"])
    return bank


def append_rows(bank_dir: Path, intent_id: str, rows: list[dict]) -> None:
    bank_dir.mkdir(parents=True, exist_ok=True)
    with (bank_dir / f"{intent_id}.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    """Key for spotting duplicates: case, Turkish characters and punctuation ignored."""
    table = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    text = text.translate(table).lower()
    return " ".join("".join(ch for ch in text if ch.isalnum() or ch.isspace()).split())


class BankText:
    """Picks a wording for each call from a text bank; falls back to the intent's example."""

    def __init__(self, bank_dir: Path, seed: int):
        self.bank = load_bank(bank_dir)
        self.rng = random.Random(seed)
        if not self.bank:
            raise ValueError(f"text bank {bank_dir} is empty")

    def text_for(self, intent: Intent, urgency: str, speaker: str, patient: Patient) -> str:
        wording = urgency if urgency in wording_urgencies(intent) else wording_urgencies(intent)[0]
        options = self.bank.get((intent.id, wording, speaker))
        return self.rng.choice(options) if options else intent.example
