"""Scripts the calls a ward makes over a period, as ScenarioCalls for replay."""

import math
import random
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models import Patient, ScenarioCall
from wardgen.config import Config, Topic
from wardgen.text import TextGenerator, button_text


def generate_calls(session: Session, config: Config, rng: random.Random, text: TextGenerator,
                   start: datetime, hours: int) -> int:
    """Adds every call for the period to the session and returns how many there are."""
    calls_cfg = config.ward["calls"]
    profile = calls_cfg["hourly_profile"]
    profile_mean = sum(profile) / len(profile)
    patients = list(session.scalars(select(Patient).order_by(Patient.id)))

    count = 0
    for patient in patients:
        per_hour = call_rate_per_day(patient, calls_cfg) / 24
        for h in range(hours):
            hour_start = start + timedelta(hours=h)
            expected = per_hour * profile[hour_start.hour] / profile_mean
            for _ in range(_poisson(rng, expected)):
                at = hour_start + timedelta(seconds=rng.randrange(3600))
                session.add(_make_call(config, rng, text, patient, at))
                count += 1
    return count


def call_rate_per_day(patient: Patient, calls_cfg: dict) -> float:
    modifiers = calls_cfg["rate_modifiers"]
    rate = calls_cfg["base_rate_per_patient_per_day"]
    rate *= modifiers.get(patient.mobility, 1.0)
    if patient.age >= 80:
        rate *= modifiers["age_80_plus"]
    return rate


def topic_weights(patient: Patient, config: Config, hour: int) -> list[tuple[Topic, float]]:
    """Every topic this patient could call about at this hour, with its relative chance."""
    calls_cfg = config.ward["calls"]
    group = next(g for g in config.ward["diagnosis_groups"] if g["id"] == patient.diagnosis_group)
    boosts = group.get("topic_boosts", {})
    active_hours = calls_cfg.get("active_hours", {})

    weighted = []
    for topic in config.topics:
        if not _meets_requirements(topic, patient):
            continue
        if topic.id in active_hours and hour not in active_hours[topic.id]:
            continue
        weight = topic.weight * boosts.get(topic.id, 1.0)
        if topic.cell.category == "physical_help" and patient.mobility != "independent":
            weight *= calls_cfg["physical_help_boost_assisted"]
        weighted.append((topic, weight))
    return weighted


def label_urgency(topic: Topic, patient: Patient) -> str:
    """The urgency of the call's label: its cell's, unless patient context raises it."""
    # Hard-case rule 6: help for a patient who cannot walk alone is soon (fall risk).
    if topic.context == "soon_if_assisted" and patient.mobility != "independent":
        return "soon"
    return topic.cell.urgency


def _make_call(config: Config, rng: random.Random, text: TextGenerator,
               patient: Patient, at: datetime) -> ScenarioCall:
    calls_cfg = config.ward["calls"]
    weighted = topic_weights(patient, config, at.hour)
    topic = rng.choices([t for t, _ in weighted], weights=[w for _, w in weighted])[0]
    speaker = ("companion" if patient.has_companion and rng.random() < calls_cfg["companion_speaks"]
               else "patient")

    if rng.random() < calls_cfg["button_share"].get(topic.id, 0.0):
        channel, words = "button", button_text(topic.id)
    else:
        channel, words = "text", text.text_for(topic, speaker, patient)

    return ScenarioCall(
        at=at, patient_id=patient.id, speaker=speaker, channel=channel, text=words,
        topic=topic.id, category=topic.cell.category, urgency=label_urgency(topic, patient),
    )


def _meets_requirements(topic: Topic, patient: Patient) -> bool:
    for requirement in topic.requires:
        if requirement == "has_iv" and not patient.has_iv:
            return False
        if requirement == "bedbound" and patient.mobility != "bedbound":
            return False
        if requirement == "not_bedbound" and patient.mobility == "bedbound":
            return False
    return True


def _poisson(rng: random.Random, mean: float) -> int:
    # Knuth's method; fine for the small means of a per-hour call count.
    limit, k, product = math.exp(-mean), 0, rng.random()
    while product > limit:
        k += 1
        product *= rng.random()
    return k
