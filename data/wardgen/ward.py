"""Generates the ward itself: rooms, patients, nurses, shifts and medication orders."""

import random
from datetime import datetime, time, timedelta

from sqlalchemy.orm import Session

from backend.models import MedicationOrder, Nurse, Patient, Room, Shift, ShiftAssignment
from wardgen.config import Config

# Synthetic names only; any match with a real person is coincidence.
FEMALE_NAMES = ["Ayşe", "Fatma", "Emine", "Hatice", "Zeynep", "Elif", "Meryem", "Şerife",
                "Sultan", "Hanife", "Gülsüm", "Esra", "Merve", "Büşra", "Seda", "Derya"]
MALE_NAMES = ["Mehmet", "Mustafa", "Ahmet", "Ali", "Hüseyin", "Hasan", "İbrahim", "İsmail",
              "Osman", "Yusuf", "Murat", "Ömer", "Ramazan", "Halil", "Kemal", "Emre"]
SURNAMES = ["Yılmaz", "Kaya", "Demir", "Şahin", "Çelik", "Yıldız", "Yıldırım", "Öztürk",
            "Aydın", "Özdemir", "Arslan", "Doğan", "Kılıç", "Aslan", "Çetin", "Kara",
            "Koç", "Kurt", "Özkan", "Şimşek"]


def age_band(age: int) -> str:
    if age < 65:
        return "under_65"
    if age < 80:
        return "65_to_79"
    return "80_plus"


def generate_ward(session: Session, config: Config, rng: random.Random,
                  start: datetime, days: int) -> None:
    ward = config.ward
    rooms = _make_rooms(session, ward)
    _make_patients(session, config, rng, rooms, start)
    _make_shifts(session, ward, rng, rooms, start, days)
    session.flush()


def _make_rooms(session: Session, ward: dict) -> list[Room]:
    rooms = [
        Room(number=str(101 + i), beds=ward["ward"]["beds_per_room"])
        for i in range(ward["ward"]["rooms"])
    ]
    session.add_all(rooms)
    session.flush()
    return rooms


def _make_patients(session: Session, config: Config, rng: random.Random,
                   rooms: list[Room], start: datetime) -> None:
    ward = config.ward
    p = ward["patients"]
    beds = [(room, bed) for room in rooms for bed in range(1, room.beds + 1)]
    occupied = rng.sample(beds, round(len(beds) * ward["ward"]["occupancy"]))
    occupied.sort(key=lambda rb: (rb[0].number, rb[1]))
    # Turkish wards usually keep rooms single-sex.
    room_sex = {room.id: rng.choice("FM") for room in rooms}
    groups = ward["diagnosis_groups"]

    for room, bed in occupied:
        sex = room_sex[room.id]
        age = round(min(p["age"]["max"], max(p["age"]["min"],
                                             rng.gauss(p["age"]["mean"], p["age"]["sd"]))))
        band = age_band(age)
        mobility = _weighted_choice(rng, p["mobility"][band])
        group = rng.choices(groups, weights=[g["weight"] for g in groups])[0]
        stay = rng.randint(p["length_of_stay_days"]["min"], p["length_of_stay_days"]["max"])
        admitted = start - timedelta(days=rng.uniform(0, stay - 1), hours=rng.uniform(0, 12))
        first = rng.choice(FEMALE_NAMES if sex == "F" else MALE_NAMES)

        patient = Patient(
            name=f"{first} {rng.choice(SURNAMES)}",
            age=age,
            sex=sex,
            diagnosis_group=group["id"],
            mobility=mobility,
            has_iv=rng.random() < group["iv_probability"],
            has_companion=rng.random() < p["companion_probability"][band],
            room=room,
            bed=bed,
            admitted_at=admitted,
            expected_discharge_at=admitted + timedelta(days=stay),
        )
        session.add(patient)
        for med in group["medications"]:
            patient.medication_orders.append(MedicationOrder(
                drug=med["drug"], dose=med["dose"], route=med["route"], times=list(med["times"]),
            ))
        for med in ward["prn_medications"]:
            patient.medication_orders.append(MedicationOrder(
                drug=med["drug"], dose=med["dose"], route=med["route"], times=[], prn=True,
                min_interval_hours=med["min_interval_hours"],
            ))


def _make_shifts(session: Session, ward: dict, rng: random.Random, rooms: list[Room],
                 start: datetime, days: int) -> None:
    shift_specs = ward["shifts"]
    # Enough nurses that nobody works two shifts in a row.
    pool_size = 2 * sum(s["nurses"] for s in shift_specs)
    nurses = [Nurse(name=_nurse_name(rng), role="nurse") for _ in range(pool_size)]
    head = Nurse(name=_nurse_name(rng), role="head_nurse")
    session.add_all([*nurses, head])
    session.flush()

    next_nurse = 0
    for day in range(days):
        date = start.date() + timedelta(days=day)
        for spec in shift_specs:
            begins = datetime.combine(date, time.fromisoformat(spec["start"]))
            shift = Shift(name=spec["name"], starts_at=begins,
                          ends_at=begins + timedelta(hours=spec["hours"]))
            session.add(shift)
            on_duty = [nurses[(next_nurse + i) % pool_size] for i in range(spec["nurses"])]
            next_nurse += spec["nurses"]
            # Contiguous blocks of rooms per nurse, so each walks a short stretch of corridor.
            for i, room in enumerate(rooms):
                nurse = on_duty[i * len(on_duty) // len(rooms)]
                shift.assignments.append(ShiftAssignment(nurse_id=nurse.id, room_id=room.id))


def _nurse_name(rng: random.Random) -> str:
    first = rng.choice(FEMALE_NAMES + MALE_NAMES)
    return f"{first} {rng.choice(SURNAMES)}"


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]
