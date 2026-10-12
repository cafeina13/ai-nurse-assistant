"""Database models for the ward: people, places, orders, requests and the event log.

Times are naive datetimes in ward-local simulated time.
"""

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Room(Base):
    __tablename__ = "rooms"

    id: Mapped[int] = mapped_column(primary_key=True)
    number: Mapped[str] = mapped_column(String(8), unique=True)
    beds: Mapped[int] = mapped_column(Integer)

    patients: Mapped[list["Patient"]] = relationship(back_populates="room")


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    age: Mapped[int] = mapped_column(Integer)
    sex: Mapped[str] = mapped_column(String(1))  # "F" or "M"
    diagnosis_group: Mapped[str] = mapped_column(String(40))
    mobility: Mapped[str] = mapped_column(String(20))  # independent | needs_assistance | bedbound
    has_iv: Mapped[bool] = mapped_column(Boolean)
    has_companion: Mapped[bool] = mapped_column(Boolean)
    room_id: Mapped[int] = mapped_column(ForeignKey("rooms.id"))
    bed: Mapped[int] = mapped_column(Integer)
    admitted_at: Mapped[datetime] = mapped_column(DateTime)
    expected_discharge_at: Mapped[datetime] = mapped_column(DateTime)

    room: Mapped[Room] = relationship(back_populates="patients")
    medication_orders: Mapped[list["MedicationOrder"]] = relationship(back_populates="patient")


class Nurse(Base):
    __tablename__ = "nurses"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    role: Mapped[str] = mapped_column(String(20))  # nurse | head_nurse


class Shift(Base):
    __tablename__ = "shifts"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(20))
    starts_at: Mapped[datetime] = mapped_column(DateTime)
    ends_at: Mapped[datetime] = mapped_column(DateTime)

    assignments: Mapped[list["ShiftAssignment"]] = relationship(back_populates="shift")


class ShiftAssignment(Base):
    """Which nurse covers which room during a shift."""

    __tablename__ = "shift_assignments"

    id: Mapped[int] = mapped_column(primary_key=True)
    shift_id: Mapped[int] = mapped_column(ForeignKey("shifts.id"))
    nurse_id: Mapped[int] = mapped_column(ForeignKey("nurses.id"))
    room_id: Mapped[int] = mapped_column(ForeignKey("rooms.id"))

    shift: Mapped[Shift] = relationship(back_populates="assignments")


class MedicationOrder(Base):
    """A doctor's order. The system shows timing context from it; it never suggests doses."""

    __tablename__ = "medication_orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    drug: Mapped[str] = mapped_column(String(80))
    dose: Mapped[str] = mapped_column(String(40))
    route: Mapped[str] = mapped_column(String(10))
    # Scheduled times of day as "HH:MM", empty for as-needed (PRN) orders.
    times: Mapped[list[str]] = mapped_column(JSON, default=list)
    prn: Mapped[bool] = mapped_column(Boolean, default=False)
    min_interval_hours: Mapped[int | None] = mapped_column(Integer, nullable=True)

    patient: Mapped[Patient] = relationship(back_populates="medication_orders")


class ScenarioCall(Base):
    """A call scripted by the generator, waiting to happen during replay.

    Holds the ground truth (category and urgency, i.e. the grid cell) used only for
    evaluation; the running system sees calls as Requests and must never read these labels.
    """

    __tablename__ = "scenario_calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    at: Mapped[datetime] = mapped_column(DateTime, index=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    speaker: Mapped[str] = mapped_column(String(20))  # patient | companion
    channel: Mapped[str] = mapped_column(String(20))  # text | button
    text: Mapped[str] = mapped_column(Text)
    topic: Mapped[str] = mapped_column(String(40))
    category: Mapped[str] = mapped_column(String(40))
    urgency: Mapped[str] = mapped_column(String(20))  # critical | soon | routine


class Request(Base):
    """A call as the system received it."""

    __tablename__ = "requests"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    room_id: Mapped[int] = mapped_column(ForeignKey("rooms.id"))
    speaker: Mapped[str] = mapped_column(String(20))
    channel: Mapped[str] = mapped_column(String(20))
    text: Mapped[str] = mapped_column(Text)
    # Links back to the script for evaluation; null for live (non-replayed) requests.
    scenario_call_id: Mapped[int | None] = mapped_column(
        ForeignKey("scenario_calls.id"), nullable=True, unique=True
    )


class Event(Base):
    """Append-only log of everything that happens. Corrections are new events.

    Database triggers (see db.py) reject UPDATE and DELETE on this table.
    """

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime, index=True)
    type: Mapped[str] = mapped_column(String(40), index=True)
    actor: Mapped[str] = mapped_column(String(40))  # "system", "patient:12", "nurse:3"
    room_id: Mapped[int | None] = mapped_column(ForeignKey("rooms.id"), nullable=True)
    patient_id: Mapped[int | None] = mapped_column(ForeignKey("patients.id"), nullable=True)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
