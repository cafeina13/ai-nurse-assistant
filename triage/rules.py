"""The safety layer: red-flag phrases that raise a request to a critical cell.

Rules only ever raise. Patterns live in data/config/safety_phrases.yaml, written in normal
Turkish; see that file for the syntax.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from triage.normalize import fold

PHRASES_PATH = Path(__file__).resolve().parent.parent / "data" / "config" / "safety_phrases.yaml"
_GAP = r"(?:\s+\w+){0,4}"


@dataclass(frozen=True)
class Rule:
    cell: str  # e.g. "breathing/critical"
    pattern: str  # as written in the config
    regex: re.Pattern
    not_after: tuple[str, ...] = ()  # roots that must not come right before the match
    unless: tuple[str, ...] = ()  # roots that, anywhere in the text, stop the rule
    unless_in_match: tuple[str, ...] = ()  # roots that stop the rule when inside the matched words


@dataclass(frozen=True)
class RuleHit:
    cell: str
    pattern: str
    matched: str  # the folded words that matched


def compile_pattern(pattern: str) -> re.Pattern:
    """Turns "göğs(üm/ü) (ağrı/sıkış) ... baş" into a regex over folded text."""
    parts = []
    for token in fold(pattern, keep="()/.").split():
        if token == "...":
            parts.append(_GAP)
            continue
        # A group that starts the word is a required choice: "(birden/aniden)", "(dudak/dil)".
        # A group after a stem lists suffix variants and is optional: "düştü(m)", "nefes(im/i)".
        token_regex = re.sub(
            r"\(([^()]*)\)",
            lambda m: ("(?:" + "|".join(re.escape(alt) for alt in m.group(1).split("/")) + ")"
                       + ("" if m.start() == 0 else "?")),
            token,
        )
        parts.append(r"\s+" + token_regex + r"\w*")
    # Each word must start at a word boundary; a word may continue past the pattern (suffixes).
    regex = "".join(parts).removeprefix(r"\s+")
    return re.compile(r"\b" + regex)


def load_rules(path: Path = PHRASES_PATH) -> list[Rule]:
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    rules = []
    for cell, entries in config["rules"].items():
        if not cell.endswith("/critical"):
            raise ValueError(f"safety rules may only target critical cells, not {cell!r}")
        for entry in entries:
            rules.append(Rule(
                cell=cell,
                pattern=entry["pattern"],
                regex=compile_pattern(entry["pattern"]),
                not_after=tuple(fold(w) for w in entry.get("not_after", ())),
                unless=tuple(fold(w) for w in entry.get("unless", ())),
                unless_in_match=tuple(fold(w) for w in entry.get("unless_in_match", ())),
            ))
    return rules


def find_hits(text: str, rules: list[Rule]) -> list[RuleHit]:
    """Every rule that fires on the text, in config order."""
    folded = fold(text)
    words = folded.split()
    hits = []
    for rule in rules:
        if rule.unless and any(w.startswith(rule.unless) for w in words):
            continue
        for match in rule.regex.finditer(folded):
            before = folded[:match.start()].split()
            if before and rule.not_after and before[-1].startswith(rule.not_after):
                continue
            if rule.unless_in_match and any(w.startswith(rule.unless_in_match)
                                            for w in match.group(0).split()):
                continue
            hits.append(RuleHit(rule.cell, rule.pattern, match.group(0)))
            break
    return hits
