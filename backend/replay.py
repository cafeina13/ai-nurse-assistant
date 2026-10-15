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
    """Simulated time running `speed` times faster than real time; None means no waiting.

    It can change speed, pause and resume while running, so a demo can be slowed down,
    held, or frozen when the shift ends.
    """

    def __init__(self, start: datetime, speed: float | None):
        self.speed = speed
        self.paused = False
        self._anchor = start  # simulated time at the last rebase
        self._real_anchor = time.monotonic()
        self._latest = start

    def now(self) -> datetime:
        if self.speed is None:
            return self._latest
        if self.paused:
            return self._anchor
        elapsed = (time.monotonic() - self._real_anchor) * self.speed
        return self._anchor + timedelta(seconds=elapsed)

    def _rebase(self) -> None:
        self._anchor = self.now()
        self._latest = max(self._latest, self._anchor)
        self._real_anchor = time.monotonic()

    def set_speed(self, speed: float | None) -> None:
        self._rebase()
        self.speed = speed

    def pause(self) -> None:
        self._rebase()
        self.paused = True

    def resume(self) -> None:
        self._real_anchor = time.monotonic()
        self.paused = False

    async def wait_until(self, at: datetime) -> None:
        # Short sleeps, re-checked, so a pause or a speed change takes effect at once.
        while self.speed is not None:
            if self.paused:
                await asyncio.sleep(0.2)
                continue
            delay = (at - self.now()).total_seconds() / self.speed
            if delay <= 0:
                break
            await asyncio.sleep(min(delay, 0.5))
        self._latest = max(self._latest, at)


@dataclass(frozen=True)
class _Step:
    at: datetime
    order: int  # shift starts before calls at the same instant
    item: Shift | ScenarioCall


class Replay:
    def __init__(self, sessions: sessionmaker[Session], speed: float | None = None,
                 clock: SimClock | None = None):
        self.sessions = sessions
        self.speed = speed
        self.listeners: list[Listener] = []
        # Pass a clock to share one simulated time with the rest of an app (the demo does).
        self.clock: SimClock | None = clock

    def subscribe(self, listener: Listener) -> None:
        self.listeners.append(listener)

    async def run(self, until: datetime | None = None) -> int:
        """Plays the whole script (or up to `until`). Returns the number of requests created."""
        steps = self._load_steps(until)
        if not steps:
            return 0
        clock = self.clock = self.clock or SimClock(steps[0].at, self.speed)
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
            # Only scripted calls count: requests typed live (e.g. on the demo's patient page)
            # before the replay starts must not block it.
            replayed = select(func.count()).select_from(Request).where(Request.scenario_call_id.is_not(None))
            if session.scalar(replayed):
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
            # Only what the system could know: no topic or label from the script.
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
