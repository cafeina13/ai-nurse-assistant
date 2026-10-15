"""The nurse board: triages each new request and keeps the live queue.

Every change is written to the append-only event log first; the board's in-memory cards
are only a view of it. Order: urgency first, then waiting time (Module 3).
"""

from collections.abc import Callable
from datetime import datetime

from sqlalchemy.orm import Session, sessionmaker

from backend import eventlog
from backend.eventlog import EventType
from backend.models import Event, Nurse, Patient, Request
from triage.pipeline import URGENCY_RANK, Triage

Card = dict


class Board:
    def __init__(self, sessions: sessionmaker[Session], triage: Triage, now: Callable[[], datetime]):
        self.sessions = sessions
        self.triage = triage
        self.now = now
        self.cards: dict[int, Card] = {}

    def create_request(self, patient_id: int, text: str, channel: str, speaker: str = "patient") -> Card:
        """A request from a bedside device: stored and logged exactly like a replayed one."""
        with self.sessions.begin() as session:
            patient = session.get(Patient, patient_id)
            if patient is None:
                raise KeyError(f"no patient {patient_id}")
            at = self.now()
            request = Request(created_at=at, patient_id=patient.id, room_id=patient.room_id,
                              speaker=speaker, channel=channel, text=text)
            session.add(request)
            session.flush()
            event = eventlog.append(
                session, at=at, type=EventType.REQUEST_CREATED, actor=f"{speaker}:{patient.id}",
                room_id=patient.room_id, patient_id=patient.id,
                details={"request_id": request.id, "room": patient.room.number,
                         "channel": channel, "text": text})
        return self.on_created(event)

    def on_created(self, event: Event) -> Card:
        """Triages a request.created event (from a device or the replay) into a card."""
        request_id = event.details["request_id"]
        with self.sessions.begin() as session:
            request = session.get(Request, request_id)
            patient = session.get(Patient, request.patient_id)
            result = self.triage.run(request.text, request.channel, patient.mobility)
            eventlog.append(
                session, at=event.at, type=EventType.REQUEST_TRIAGED, actor="system",
                room_id=request.room_id, patient_id=patient.id,
                details={"request_id": request_id, "cell": result.cell, "confidence": result.confidence,
                         "reasons": result.reasons, "uncertain": result.uncertain,
                         "interrupt": result.interrupt, "version": result.version})
            card = {
                "id": request_id,
                "room": patient.room.number,
                "bed": patient.bed,
                "patient": patient.name.split()[0],
                "speaker": request.speaker,
                "channel": request.channel,
                "text": request.text,
                "created_at": request.created_at.isoformat(),
                "cell": result.cell,
                "category": result.cell.split("/")[0],
                "urgency": result.urgency,
                "confidence": result.confidence,
                "reasons": result.reasons,
                "uncertain": result.uncertain,
                "interrupt": result.interrupt,
                # Critical only on a weak model guess (no rule, no protocol line): it keeps its place
                # at the top, but the board says "possible critical" rather than claiming certainty.
                "tentative": (result.urgency == "critical" and (result.confidence or 0) < 0.5
                              and not any(r.startswith("rule:") for r in result.reasons)),
                "status": "open",
                "claimed_by": None,
            }
        self.cards[request_id] = card
        return card

    def claim(self, request_id: int, nurse_id: int) -> Card:
        card = self._open_card(request_id)
        with self.sessions.begin() as session:
            nurse = session.get(Nurse, nurse_id)
            if nurse is None:
                raise KeyError(f"no nurse {nurse_id}")
            self._log(session, EventType.REQUEST_CLAIMED, card, f"nurse:{nurse_id}", {})
            card.update(status="claimed", claimed_by=nurse.name)
        return card

    def complete(self, request_id: int, nurse_id: int, really_critical: bool | None = None) -> Card:
        """Closes a card. For a critical one the nurse says whether it really was critical,
        which is how false-alarm rates get measured in real use."""
        card = self._open_card(request_id)
        details = {}
        if card["urgency"] == "critical":
            if really_critical is None:
                raise ValueError("completing a critical request needs really_critical")
            details["really_critical"] = really_critical
        with self.sessions.begin() as session:
            self._log(session, EventType.REQUEST_COMPLETED, card, f"nurse:{nurse_id}", details)
        card["status"] = "done"
        del self.cards[request_id]
        return card

    def queue(self) -> list[Card]:
        return sorted(self.cards.values(),
                      key=lambda c: (-URGENCY_RANK[c["urgency"]], c["created_at"]))

    def nurses(self) -> list[dict]:
        with self.sessions() as session:
            return [{"id": n.id, "name": n.name, "role": n.role}
                    for n in session.query(Nurse).order_by(Nurse.id)]

    def patients(self) -> list[dict]:
        with self.sessions() as session:
            return [{"id": p.id, "name": p.name, "room": p.room.number, "bed": p.bed,
                     "mobility": p.mobility}
                    for p in session.query(Patient).order_by(Patient.room_id, Patient.bed)]

    def _open_card(self, request_id: int) -> Card:
        card = self.cards.get(request_id)
        if card is None:
            raise KeyError(f"request {request_id} is not open")
        return card

    def _log(self, session: Session, type: str, card: Card, actor: str, details: dict) -> None:
        request = session.get(Request, card["id"])
        eventlog.append(session, at=self.now(), type=type, actor=actor, room_id=request.room_id,
                        patient_id=request.patient_id, details={"request_id": card["id"], **details})
