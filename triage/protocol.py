"""Protocol engine: extract facts, then decide the cell with an ordered, readable table.

A different kind of approach from the classifiers: no training, no vectors. Facts and the
protocol live in data/config/protocol.yaml, where a nurse can read and correct them; every
decision names the facts and the protocol line behind it.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from triage.normalize import fold
from triage.rules import compile_pattern

PROTOCOL_PATH = Path(__file__).resolve().parent.parent / "data" / "config" / "protocol.yaml"

# Confidence the pipeline sees: an explicit protocol line is certain about what it matched;
# the fallback line ("nothing recognized") is not.
LINE_CONFIDENCE, FALLBACK_CONFIDENCE = 0.95, 0.3


@dataclass(frozen=True)
class Line:
    number: int
    cell: str
    all: tuple[str, ...] = ()
    any: tuple[str, ...] = ()
    none: tuple[str, ...] = ()

    def holds(self, facts: set[str]) -> bool:
        return (all(f in facts for f in self.all)
                and (not self.any or any(f in facts for f in self.any))
                and not any(f in facts for f in self.none))

    def describe(self) -> str:
        parts = []
        if self.all:
            parts.append(" + ".join(self.all))
        if self.any:
            parts.append("(" + " / ".join(self.any) + ")")
        if self.none:
            parts.append("not " + " / ".join(self.none))
        return f"line {self.number}: " + (", ".join(parts) or "nothing recognized")


@dataclass(frozen=True)
class Decision:
    cell: str
    line: Line
    facts: frozenset[str]


class Protocol:
    def __init__(self, path: Path = PROTOCOL_PATH):
        with path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.facts: dict[str, list[re.Pattern]] = {
            name: [compile_pattern(p) for p in patterns] for name, patterns in config["facts"].items()}
        self.lines = [Line(i + 1, raw["cell"], tuple(raw.get("all", ())), tuple(raw.get("any", ())),
                           tuple(raw.get("none", ())))
                      for i, raw in enumerate(config["protocol"])]
        unknown = {f for line in self.lines for f in line.all + line.any + line.none} - set(self.facts)
        if unknown:
            raise ValueError(f"protocol uses undefined facts: {sorted(unknown)}")
        if self.lines[-1].all or self.lines[-1].any or self.lines[-1].none:
            raise ValueError("the last protocol line must be the unconditional fallback")

    def extract(self, text: str) -> set[str]:
        folded = fold(text)
        return {name for name, patterns in self.facts.items() if any(p.search(folded) for p in patterns)}

    def decide(self, text: str) -> Decision:
        facts = self.extract(text)
        line = next(line for line in self.lines if line.holds(facts))
        return Decision(line.cell, line, frozenset(facts))


class ProtocolModel:
    """The protocol behind the same interface as the classifiers, so the pipeline, the
    evaluation and the comparison treat it the same way."""

    name = "protocol"

    def __init__(self, protocol: Protocol | None = None):
        self.protocol = protocol or Protocol()
        self.settings = {"protocol": str(PROTOCOL_PATH.name), "lines": len(self.protocol.lines)}

    def fit(self, texts: list[str], cells: list[str]) -> "ProtocolModel":
        return self  # nothing to learn: the knowledge is written down

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        out = []
        for text in texts:
            d = self.protocol.decide(text)
            fallback = d.line is self.protocol.lines[-1]
            out.append({d.cell: FALLBACK_CONFIDENCE if fallback else LINE_CONFIDENCE})
        return out

    def explain(self, text: str) -> str:
        d = self.protocol.decide(text)
        return f"{d.line.describe()} [{', '.join(sorted(d.facts)) or 'no facts'}]"


URGENCY_RANK = {"routine": 0, "soon": 1, "critical": 2}
# Confidence for a critical only the protocol found: the card tops the queue, but stays below
# the pipeline's interrupt threshold (alarm principle: order generously, interrupt carefully).
PROTOCOL_ONLY_CONFIDENCE = 0.6


class HybridModel:
    """A learned classifier and the protocol together; `mode` says how they combine.

    raise     the higher urgency of the two wins (the protocol can only raise, like safety rules)
    critical  the classifier decides, but a protocol critical line overrides it
    protocol  the protocol decides when a specific line matches, the classifier otherwise
    """

    MODES = ("raise", "critical", "protocol")

    def __init__(self, classifier, mode: str = "raise", protocol: Protocol | None = None):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.classifier, self.mode = classifier, mode
        self.protocol = ProtocolModel(protocol)
        self.name = f"hybrid-{mode}({classifier.name}+protocol)"
        self.settings = {"mode": mode, "classifier": classifier.settings, "protocol": self.protocol.settings}

    def fit(self, texts, cells):
        self.classifier.fit(texts, cells)
        return self

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        out = []
        for text, learned in zip(texts, self.classifier.predict_proba(texts)):
            model_cell = max(learned, key=learned.get)
            decision = self.protocol.protocol.decide(text)
            specific = decision.line is not self.protocol.protocol.lines[-1]
            p_cell = decision.cell
            if self.mode == "raise":
                take = specific and URGENCY_RANK[_urgency(p_cell)] > URGENCY_RANK[_urgency(model_cell)]
            elif self.mode == "critical":
                take = specific and _urgency(p_cell) == "critical" and _urgency(model_cell) != "critical"
            else:
                take = specific and p_cell != model_cell
            if take:
                confidence = (PROTOCOL_ONLY_CONFIDENCE if _urgency(p_cell) == "critical"
                              else max(learned.get(p_cell, 0.0), 0.5))
                out.append({p_cell: confidence, model_cell: min(learned[model_cell], confidence - 0.01)})
            else:
                out.append(learned)
        return out

    def explain(self, text: str) -> str:
        return self.protocol.explain(text)


def _urgency(cell: str) -> str:
    return cell.split("/")[1]
