import asyncio
from datetime import timedelta

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from backend.db import make_engine, session_factory
from backend.eventlog import EventType
from backend.models import Event, Patient, Request, ScenarioCall, Shift
from backend.replay import Replay
from wardgen import DEFAULT_START, generate
from wardgen.calls import call_rate_per_day
from wardgen.config import load_config


@pytest.fixture
def ward_db(tmp_path):
    path = tmp_path / "ward.db"
    generate(path, seed=7, days=1)
    return path


def sessions(path):
    return session_factory(make_engine(path))


def test_config_matches_plan_doc():
    config = load_config()
    assert len(config.categories) == 11
    assert len(config.intents) == 47
    assert {i.category for i in config.intents.values()} == set(config.categories)


def test_same_seed_gives_same_ward(tmp_path):
    def snapshot(path):
        generate(path, seed=3, days=1)
        with sessions(path)() as s:
            calls = s.scalars(select(ScenarioCall).order_by(ScenarioCall.id)).all()
            return [(c.at, c.patient_id, c.intent, c.urgency, c.text) for c in calls]

    assert snapshot(tmp_path / "a.db") == snapshot(tmp_path / "b.db")


def test_call_volume_follows_configured_rates(tmp_path):
    path = tmp_path / "week.db"
    count = generate(path, seed=11, days=7)
    calls_cfg = load_config().ward["calls"]
    with sessions(path)() as s:
        expected = 7 * sum(call_rate_per_day(p, calls_cfg) for p in s.scalars(select(Patient)))
    assert abs(count - expected) / expected < 0.1


def test_calls_respect_patient_attributes(ward_db):
    with sessions(ward_db)() as s:
        for call in s.scalars(select(ScenarioCall)):
            patient = s.get(Patient, call.patient_id)
            if call.intent in {"pump_alarm", "iv_finished", "iv_site_problem"}:
                assert patient.has_iv
            if call.intent == "bedpan_urinal":
                assert patient.mobility == "bedbound"
            if call.intent in {"toilet_help", "get_up_walk"}:
                assert patient.mobility != "bedbound"
                assert call.urgency == ("routine" if patient.mobility == "independent" else "soon")


def test_shifts_cover_the_whole_period(ward_db):
    with sessions(ward_db)() as s:
        shifts = s.scalars(select(Shift).order_by(Shift.starts_at)).all()
    assert shifts[0].starts_at == DEFAULT_START
    for earlier, later in zip(shifts, shifts[1:]):
        assert earlier.ends_at == later.starts_at
    assert shifts[-1].ends_at == DEFAULT_START + timedelta(days=1)


def test_event_log_is_append_only(ward_db):
    asyncio.run(Replay(sessions(ward_db)).run())
    engine = make_engine(ward_db)
    for statement in ("UPDATE events SET type = 'x'", "DELETE FROM events"):
        with pytest.raises(IntegrityError, match="append-only"):
            with engine.begin() as conn:
                conn.execute(text(statement))


def test_replay_creates_one_request_per_call_in_order(ward_db):
    seen = []
    replay = Replay(sessions(ward_db))
    replay.subscribe(seen.append)
    created = asyncio.run(replay.run())

    with sessions(ward_db)() as s:
        n_calls = len(s.scalars(select(ScenarioCall)).all())
        requests = s.scalars(select(Request)).all()
        events = s.scalars(select(Event).order_by(Event.id)).all()

    assert created == n_calls == len(requests)
    assert [e.id for e in seen] == [e.id for e in events]
    times = [e.at for e in events]
    assert times == sorted(times)
    assert events[0].type == EventType.REPLAY_STARTED
    assert events[-1].type == EventType.REPLAY_FINISHED
    # The system must never see the generator's labels.
    for e in events:
        assert "intent" not in e.details and "urgency" not in e.details


def test_replay_refuses_to_run_twice(ward_db):
    asyncio.run(Replay(sessions(ward_db)).run())
    with pytest.raises(RuntimeError, match="already been replayed"):
        asyncio.run(Replay(sessions(ward_db)).run())


def test_textbank_cells_and_lookup(tmp_path):
    from wardgen.textbank import BankText, append_rows, cells

    config = load_config()
    # soon_plus intents have two wordings (soon, critical); every cell has two speakers.
    soon_plus = sum(1 for i in config.intents.values() if i.urgency == "soon_plus")
    assert len(cells(config.intents)) == 2 * (len(config.intents) + soon_plus)

    append_rows(tmp_path, "water", [
        {"text": "su verir misiniz", "intent": "water", "urgency": "routine",
         "speaker": "patient", "source": "test", "prompt_version": 1},
    ])
    bank = BankText(tmp_path, seed=1)
    water, fell = config.intents["water"], config.intents["fell"]
    assert bank.text_for(water, "routine", "patient", None) == "su verir misiniz"
    # Empty cells fall back to the intent's example phrase.
    assert bank.text_for(fell, "critical", "companion", None) == fell.example


def test_generate_with_textbank(tmp_path):
    from wardgen.textbank import append_rows

    bank_dir = tmp_path / "bank"
    append_rows(bank_dir, "water", [
        {"text": "bi bardak su alabilir miyim", "intent": "water", "urgency": "routine",
         "speaker": speaker, "source": "test", "prompt_version": 1}
        for speaker in ("patient", "companion")
    ])
    generate(tmp_path / "w.db", seed=5, days=2, textbank=bank_dir)
    with sessions(tmp_path / "w.db")() as s:
        water_texts = {c.text for c in s.scalars(select(ScenarioCall))
                       if c.intent == "water" and c.channel == "text"}
    assert water_texts == {"bi bardak su alabilir miyim"}


def test_claude_test_bank_covers_every_cell():
    from wardgen.textbank import TEXTBANK_DIR, cells, load_bank

    bank = load_bank(TEXTBANK_DIR / "test" / "claude")
    assert {k: len(v) for k, v in bank.items()} == {c: 4 for c in cells(load_config().intents)}
