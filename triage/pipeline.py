"""The triage pipeline: one request in, one grid cell out, with reasons.

    button? -> fixed cell
    else    -> model predicts a cell -> safety rules may raise to a critical cell
    then    -> patient context (rule 6) -> uncertainty and interrupt flags

The system orders and flags; it never decides care (core principle). Every result carries
the reasons a nurse sees and the version that produced it.
"""

from dataclasses import dataclass, field
from typing import Protocol

from triage.rules import Rule, RuleHit, find_hits, load_rules

URGENCY_RANK = {"routine": 0, "soon": 1, "critical": 2}

# Quick buttons skip the AI. "Tuvalet" can still become soon through patient context.
BUTTON_CELLS = {
    "[Su]": "comfort_supplies/routine",
    "[Tuvalet]": "physical_help/routine",
    "[Ağrı]": "pain/soon",
}
# Topics whose routine wording becomes soon for a patient who cannot walk alone (rule 6).
# The pipeline sees no topic, so it applies the rule to the whole Physical help/Routine cell.
CONTEXT_CELL = "physical_help/routine"


class CellModel(Protocol):
    name: str

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        """For each text, a probability per cell label ("category/urgency")."""


@dataclass(frozen=True)
class Settings:
    uncertain_below: float = 0.5  # model confidence under this flags the card for the nurse
    interrupt_above: float = 0.9  # model confidence for a critical to interrupt on its own


@dataclass
class TriageResult:
    cell: str
    confidence: float | None  # the model's probability for its cell; None for buttons
    reasons: list[str] = field(default_factory=list)
    uncertain: bool = False
    interrupt: bool = False
    version: str = ""

    @property
    def urgency(self) -> str:
        return self.cell.split("/")[1]


class Triage:
    def __init__(self, model: CellModel, rules: list[Rule] | None = None,
                 settings: Settings = Settings()):
        self.model = model
        self.rules = load_rules() if rules is None else rules
        self.settings = settings
        self.version = f"rules:{len(self.rules)}+model:{model.name}"

    def run(self, text: str, channel: str = "text", mobility: str | None = None) -> TriageResult:
        return self.run_many([text], [channel], [mobility])[0]

    def run_many(self, texts: list[str], channels: list[str] | None = None,
                 mobilities: list[str | None] | None = None) -> list[TriageResult]:
        channels = channels or ["text"] * len(texts)
        mobilities = mobilities or [None] * len(texts)
        free_text = [i for i, ch in enumerate(channels) if ch != "button"]
        probas = dict(zip(free_text, self.model.predict_proba([texts[i] for i in free_text])))

        results = []
        for i, text in enumerate(texts):
            if channels[i] == "button":
                result = TriageResult(BUTTON_CELLS[text], None, [f"button: {text}"])
            else:
                result = self._from_text(text, probas[i])
            self._apply_context(result, mobilities[i])
            result.version = self.version
            results.append(result)
        return results

    def _from_text(self, text: str, proba: dict[str, float]) -> TriageResult:
        model_cell, confidence = max(proba.items(), key=lambda kv: kv[1])
        model_critical = model_cell.endswith("/critical")
        result = TriageResult(model_cell, confidence, [f"model: {model_cell} ({confidence:.2f})"])
        hits = find_hits(text, self.rules)
        if hits:
            # The rule wins outright: its critical cell replaces the model's (hard rule 2).
            hit = _best_hit(hits, model_cell)
            result.cell = hit.cell
            result.reasons.append(f"rule: {hit.pattern}")
        result.uncertain = confidence < self.settings.uncertain_below and not hits
        # Alarm principle: every critical tops the queue, but only a confident one interrupts:
        # the rule and the model agree, or the model alone is very sure.
        result.interrupt = model_critical and (
            bool(hits) or confidence >= self.settings.interrupt_above)
        return result

    @staticmethod
    def _apply_context(result: TriageResult, mobility: str | None) -> None:
        if result.cell == CONTEXT_CELL and mobility not in (None, "independent"):
            result.cell = "physical_help/soon"
            result.reasons.append("context: cannot walk alone")


def _best_hit(hits: list[RuleHit], model_cell: str) -> RuleHit:
    """Prefer the rule whose cell the model also chose, else the first in config order."""
    return next((h for h in hits if h.cell == model_cell), hits[0])
