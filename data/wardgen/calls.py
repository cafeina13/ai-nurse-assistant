"""Scripts the calls a ward makes over a period, as ScenarioCalls for replay."""

import math
import random
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models import Patient, ScenarioCall
from wardgen.config import Config, Intent
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


def intent_weights(patient: Patient, config: Config, hour: int) -> dict[str, float]:
    """Relative chance of each intent for this patient at this hour of day."""
    calls_cfg = config.ward["calls"]
    group = next(g for g in config.ward["diagnosis_groups"] if g["id"] == patient.diagnosis_group)
    boosts = group.get("intent_boosts", {})
    active_hours = calls_cfg.get("active_hours", {})

    weights = {}
    for intent in config.intents.values():
        if not _meets_requirements(intent, patient):
            continue
        if intent.id in active_hours and hour not in active_hours[intent.id]:
            continue
        weight = intent.weight * boosts.get(intent.id, 1.0)
        if intent.category == "physical_help" and patient.mobility != "independent":
            weight *= calls_cfg["physical_help_boost_assisted"]
        weights[intent.id] = weight
    return weights


def resolve_urgency(intent: Intent, patient: Patient, rng: random.Random,
                    soon_plus_critical: float) -> str:
    """The urgency this call's wording should express (its ground-truth label)."""
    match intent.urgency:
        case "soon_plus":
            return "critical" if rng.random() < soon_plus_critical else "soon"
        case "routine_assisted":
            # Labeling rule 6: help for a patient who cannot walk alone is soon (fall risk).
            return "routine" if patient.mobility == "independent" else "soon"
        case urgency:
            return urgency


def _make_call(config: Config, rng: random.Random, text: TextGenerator,
               patient: Patient, at: datetime) -> ScenarioCall:
    calls_cfg = config.ward["calls"]
    weights = intent_weights(patient, config, at.hour)
    intent = config.intents[rng.choices(list(weights), weights=list(weights.values()))[0]]
    urgency = resolve_urgency(intent, patient, rng, calls_cfg["soon_plus_critical"])
    speaker = ("companion" if patient.has_companion and rng.random() < calls_cfg["companion_speaks"]
               else "patient")

    if rng.random() < calls_cfg["button_share"].get(intent.id, 0.0):
        channel, words = "button", button_text(intent.id)
    else:
        channel, words = "text", text.text_for(intent, urgency, speaker, patient)

    return ScenarioCall(
        at=at, patient_id=patient.id, speaker=speaker, channel=channel, text=words,
        intent=intent.id, category=intent.category, urgency=urgency,
    )


def _meets_requirements(intent: Intent, patient: Patient) -> bool:
    for requirement in intent.requires:
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
