"""Loads the YAML configs in data/config and checks they agree with each other."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

URGENCY_RULES = {"critical", "soon", "routine", "soon_plus", "routine_assisted"}
REQUIREMENTS = {"has_iv", "bedbound", "not_bedbound"}


@dataclass(frozen=True)
class Intent:
    id: str
    category: str
    urgency: str
    weight: float
    example: str
    requires: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Config:
    categories: dict[str, dict]
    intents: dict[str, Intent]
    ward: dict


def load_config(config_dir: Path = CONFIG_DIR) -> Config:
    categories = {
        c["id"]: c for c in _read(config_dir / "categories.yaml")["categories"]
    }
    intents = {}
    for raw in _read(config_dir / "intents.yaml")["intents"]:
        intent = Intent(
            id=raw["id"],
            category=raw["category"],
            urgency=raw["urgency"],
            weight=float(raw["weight"]),
            example=raw["example"],
            requires=tuple(raw.get("requires", ())),
        )
        _check_intent(intent, categories, intents)
        intents[intent.id] = intent
    ward = _read(config_dir / "ward.yaml")
    _check_ward(ward, intents)
    return Config(categories=categories, intents=intents, ward=ward)


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _check_intent(intent: Intent, categories: dict, seen: dict) -> None:
    if intent.id in seen:
        raise ValueError(f"duplicate intent {intent.id!r}")
    if intent.category not in categories:
        raise ValueError(f"intent {intent.id!r} has unknown category {intent.category!r}")
    if intent.urgency not in URGENCY_RULES:
        raise ValueError(f"intent {intent.id!r} has unknown urgency {intent.urgency!r}")
    unknown = set(intent.requires) - REQUIREMENTS
    if unknown:
        raise ValueError(f"intent {intent.id!r} has unknown requirements {sorted(unknown)}")
    if intent.weight < 0:
        raise ValueError(f"intent {intent.id!r} has a negative weight")


def _check_ward(ward: dict, intents: dict) -> None:
    if len(ward["calls"]["hourly_profile"]) != 24:
        raise ValueError("calls.hourly_profile must have 24 values")
    for group in ward["diagnosis_groups"]:
        for intent_id in group.get("intent_boosts", {}):
            if intent_id not in intents:
                raise ValueError(
                    f"diagnosis group {group['id']!r} boosts unknown intent {intent_id!r}"
                )
    for key in ("button_share", "active_hours"):
        for intent_id in ward["calls"].get(key, {}):
            if intent_id not in intents:
                raise ValueError(f"calls.{key} names unknown intent {intent_id!r}")
