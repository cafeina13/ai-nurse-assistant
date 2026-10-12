"""Writing to and reading from the append-only event log.

There is deliberately no update or delete here. A correction is a new event.
"""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models import Event


class EventType:
    SHIFT_STARTED = "shift.started"
    REQUEST_CREATED = "request.created"
    REQUEST_CLAIMED = "request.claimed"
    REQUEST_COMPLETED = "request.completed"
    REQUEST_REASSIGNED = "request.reassigned"
    REPLAY_STARTED = "replay.started"
    REPLAY_FINISHED = "replay.finished"


def append(
    session: Session,
    *,
    at: datetime,
    type: str,
    actor: str,
    room_id: int | None = None,
    patient_id: int | None = None,
    details: dict | None = None,
) -> Event:
    event = Event(
        at=at,
        type=type,
        actor=actor,
        room_id=room_id,
        patient_id=patient_id,
        details=details or {},
    )
    session.add(event)
    return event


def events_between(session: Session, start: datetime, end: datetime) -> list[Event]:
    """Events with start <= at < end, in the order they happened."""
    query = (
        select(Event)
        .where(Event.at >= start, Event.at < end)
        .order_by(Event.at, Event.id)
    )
    return list(session.scalars(query))
