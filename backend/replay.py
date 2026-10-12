"""Plays a generated ward on a simulated clock, as if it were happening now.

Each scripted call becomes a Request and a request.created event at its simulated time;
each shift start becomes a shift.started event. Listeners (later: the nurse board over
WebSockets) receive every event as it is written.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from backend import eventlog
from backend.eventlog import EventType
from backend.models import Event, Patient, Request, ScenarioCall, Shift

Listener = Callable[[Event], Awaitable[None] | None]


class SimClock:
    """Simulated time running `speed` times faster than real time; None means no waiting."""

    def __init__(self, start: datetime, speed: float | None):
        self.start = start
        self.speed = speed
        self._real_start = time.monotonic()
        self._latest = start

    def now(self) -> datetime:
        if self.speed is None:
            return self._latest
        elapsed = (time.monotonic() - self._real_start) * self.speed
        return self.start + timedelta(seconds=elapsed)

    async def wait_until(self, at: datetime) -> None:
        if self.speed is not None:
            delay = (at - self.now()).total_seconds() / self.speed
            if delay > 0:
                await asyncio.sleep(delay)
        self._latest = max(self._latest, at)


@dataclass(frozen=True)
class _Step:
    at: datetime
    order: int  # shift starts before calls at the same instant
    item: Shift | ScenarioCall


class Replay:
    def __init__(self, sessions: sessionmaker[Session], speed: float | None = None):
        self.sessions = sessions
        self.speed = speed
        self.listeners: list[Listener] = []

    def subscribe(self, listener: Listener) -> None:
        self.listeners.append(listener)

    async def run(self, until: datetime | None = None) -> int:
        """Plays the whole script (or up to `until`). Returns the number of requests created."""
        steps = self._load_steps(until)
        if not steps:
            return 0
        clock = SimClock(steps[0].at, self.speed)
        await self._emit(steps[0].at, EventType.REPLAY_STARTED, "system",
                         details={"speed": self.speed, "steps": len(steps)})

        created = 0
        for step in steps:
            await clock.wait_until(step.at)
            if isinstance(step.item, Shift):
                await self._start_shift(step.item)
            else:
                await self._create_request(step.item)
                created += 1

        await self._emit(clock.now(), EventType.REPLAY_FINISHED, "system",
                         details={"requests": created})
        return created

    def _load_steps(self, until: datetime | None) -> list[_Step]:
        with self.sessions() as session:
            if session.scalar(select(func.count()).select_from(Request)):
                raise RuntimeError(
                    "this database has already been replayed; generate a fresh one "
                    "(the event log is append-only, so a replay cannot be undone)"
                )
            shifts = session.scalars(select(Shift)).all()
            calls = session.scalars(select(ScenarioCall)).all()
        steps = [_Step(s.starts_at, 0, s) for s in shifts]
        steps += [_Step(c.at, 1, c) for c in calls]
        if until is not None:
            steps = [s for s in steps if s.at < until]
        return sorted(steps, key=lambda s: (s.at, s.order, s.item.id))

    async def _start_shift(self, shift: Shift) -> None:
        with self.sessions.begin() as session:
            assignments = session.get(Shift, shift.id).assignments
            rooms_by_nurse: dict[int, list[int]] = {}
            for a in assignments:
                rooms_by_nurse.setdefault(a.nurse_id, []).append(a.room_id)
            event = eventlog.append(
                session, at=shift.starts_at, type=EventType.SHIFT_STARTED, actor="system",
                details={"shift_id": shift.id, "name": shift.name,
                         "ends_at": shift.ends_at.isoformat(),
                         "rooms_by_nurse": {str(k): v for k, v in rooms_by_nurse.items()}},
            )
        await self._notify(event)

    async def _create_request(self, call: ScenarioCall) -> None:
        with self.sessions.begin() as session:
            patient = session.get(Patient, call.patient_id)
            request = Request(
                created_at=call.at, patient_id=patient.id, room_id=patient.room_id,
                speaker=call.speaker, channel=call.channel, text=call.text,
                scenario_call_id=call.id,
            )
            session.add(request)
            session.flush()
            # Only what the system could know: no intent or urgency from the script.
            event = eventlog.append(
                session, at=call.at, type=EventType.REQUEST_CREATED,
                actor=f"{call.speaker}:{patient.id}", room_id=patient.room_id,
                patient_id=patient.id,
                details={"request_id": request.id, "room": patient.room.number,
                         "channel": call.channel, "text": call.text},
            )
        await self._notify(event)

    async def _emit(self, at: datetime, type: str, actor: str, details: dict) -> None:
        with self.sessions.begin() as session:
            event = eventlog.append(session, at=at, type=type, actor=actor, details=details)
        await self._notify(event)

    async def _notify(self, event: Event) -> None:
        for listener in self.listeners:
            result = listener(event)
            if asyncio.iscoroutine(result):
                await result


def main() -> None:
    """python -m backend.replay data/generated/ward.db --speed 3600"""
    import argparse

    from backend.db import make_engine, session_factory

    parser = argparse.ArgumentParser(description="Replay a generated ward.")
    parser.add_argument("db", help="SQLite file made by `python -m wardgen`")
    parser.add_argument("--speed", type=float, default=3600.0,
                        help="simulated seconds per real second (3600 = one hour per second); "
                             "0 = as fast as possible")
    args = parser.parse_args()

    def show(event: Event) -> None:
        d = event.details
        match event.type:
            case EventType.REQUEST_CREATED:
                line = f"room {d['room']}  {event.actor:<13} {d['text']}"
            case EventType.SHIFT_STARTED:
                line = f"--- {d['name']} shift starts, nurses on duty: {len(d['rooms_by_nurse'])}"
            case _:
                line = f"--- {event.type} {d}"
        print(f"{event.at:%a %H:%M}  {line}")

    replay = Replay(session_factory(make_engine(args.db)), speed=args.speed or None)
    replay.subscribe(show)
    asyncio.run(replay.run())


if __name__ == "__main__":
    main()
