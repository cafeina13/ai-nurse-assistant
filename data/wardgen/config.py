"""Loads the YAML configs in data/config and checks they agree with each other."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

URGENCIES = ("critical", "soon", "routine")
REQUIREMENTS = {"has_iv", "bedbound", "not_bedbound"}
CONTEXTS = {"soon_if_assisted"}


@dataclass(frozen=True)
class Cell:
    """One label: a category at one urgency level."""

    category: str
    urgency: str
    means: str

    @property
    def label(self) -> str:
        return f"{self.category}/{self.urgency}"


@dataclass(frozen=True)
class Topic:
    """An example of what a cell's requests are about; steers the generator only."""

    id: str
    cell: Cell
    weight: float
    example: str
    requires: tuple[str, ...] = field(default_factory=tuple)
    context: str | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.cell.category, self.cell.urgency, self.id)


@dataclass(frozen=True)
class Config:
    categories: dict[str, dict]
    cells: dict[str, Cell]  # by label
    topics: list[Topic]
    ward: dict

    @property
    def topic_ids(self) -> set[str]:
        return {t.id for t in self.topics}


def load_config(config_dir: Path = CONFIG_DIR) -> Config:
    categories, cells, topics = {}, {}, []
    for raw_category in _read(config_dir / "grid.yaml")["categories"]:
        category = raw_category["id"]
        if category in categories:
            raise ValueError(f"duplicate category {category!r}")
        categories[category] = {k: v for k, v in raw_category.items() if k != "cells"}
        for urgency, raw_cell in raw_category["cells"].items():
            if urgency not in URGENCIES:
                raise ValueError(f"{category}: unknown urgency {urgency!r}")
            cell = Cell(category, urgency, raw_cell["means"])
            cells[cell.label] = cell
            for raw in raw_cell["topics"]:
                topics.append(_make_topic(raw, cell))

    keys = [t.key for t in topics]
    duplicates = {k for k in keys if keys.count(k) > 1}
    if duplicates:
        raise ValueError(f"topics listed twice in one cell: {sorted(duplicates)}")
    ward = _read(config_dir / "ward.yaml")
    config = Config(categories=categories, cells=cells, topics=topics, ward=ward)
    _check_ward(config)
    return config


def _make_topic(raw: dict, cell: Cell) -> Topic:
    topic = Topic(
        id=raw["id"],
        cell=cell,
        weight=float(raw["weight"]),
        example=raw["example"],
        requires=tuple(raw.get("requires", ())),
        context=raw.get("context"),
    )
    where = f"{cell.label}/{topic.id}"
    if set(topic.requires) - REQUIREMENTS:
        raise ValueError(f"{where}: unknown requirements {topic.requires}")
    if topic.context is not None and topic.context not in CONTEXTS:
        raise ValueError(f"{where}: unknown context {topic.context!r}")
    if topic.context == "soon_if_assisted" and cell.urgency != "routine":
        raise ValueError(f"{where}: soon_if_assisted only makes sense in a routine cell")
    if topic.weight < 0:
        raise ValueError(f"{where}: negative weight")
    return topic


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _check_ward(config: Config) -> None:
    ward = config.ward
    if len(ward["calls"]["hourly_profile"]) != 24:
        raise ValueError("calls.hourly_profile must have 24 values")
    known = config.topic_ids
    for group in ward["diagnosis_groups"]:
        for topic_id in group.get("topic_boosts", {}):
            if topic_id not in known:
                raise ValueError(f"diagnosis group {group['id']!r} boosts unknown topic {topic_id!r}")
    for key in ("button_share", "active_hours"):
        for topic_id in ward["calls"].get(key, {}):
            if topic_id not in known:
                raise ValueError(f"calls.{key} names unknown topic {topic_id!r}")
    # A soon_if_assisted topic is relabeled soon, so that cell must exist.
    for topic in config.topics:
        if topic.context == "soon_if_assisted" and f"{topic.cell.category}/soon" not in config.cells:
            raise ValueError(f"{topic.cell.category} has no soon cell for {topic.id!r}")
