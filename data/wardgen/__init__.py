"""Synthetic ward generator (Module 0)."""

import random
from datetime import datetime
from pathlib import Path

from backend.db import init_db, make_engine, session_factory
from wardgen.calls import generate_calls
from wardgen.config import Config, load_config
from wardgen.text import ExampleText, TextGenerator
from wardgen.textbank import BankText
from wardgen.ward import generate_ward

DEFAULT_START = datetime(2026, 1, 5, 8, 0)  # a Monday, start of the day shift


def generate(db_path: str | Path, *, seed: int, days: int = 1,
             start: datetime = DEFAULT_START, config: Config | None = None,
             textbank: Path | None = None) -> int:
    """Writes a new ward and its scripted calls to an empty database. Returns the call count.

    Without a text bank, every call uses its intent's example phrase.
    """
    config = config or load_config()
    text: TextGenerator = BankText(textbank, seed) if textbank else ExampleText()
    engine = make_engine(db_path)
    init_db(engine)
    rng = random.Random(seed)
    with session_factory(engine).begin() as session:
        generate_ward(session, config, rng, start, days)
        return generate_calls(session, config, rng, text, start, days * 24)
