"""Train a cell model on the Gemini bank and evaluate the triage pipeline.

    python -m triage.evaluate                # report on the held-back 20% of the training bank
    python -m triage.evaluate --final        # also the Claude test bank and the tone sets

Critical recall comes first, as the plan requires. Every run is saved to experiments/ with
its data fingerprint, settings and scores so results can be reproduced.
"""

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from triage.models import make_model
from triage.pipeline import URGENCY_RANK, Triage
from triage.rules import find_hits, load_rules
from wardgen.textbank import TEXTBANK_DIR

# TRIAGE_EXPERIMENTS overrides it where the code folder is read-only (Kaggle).
EXPERIMENTS_DIR = Path(os.environ.get("TRIAGE_EXPERIMENTS",
                                      Path(__file__).resolve().parent.parent / "experiments"))


@dataclass(frozen=True)
class Example:
    text: str
    cell: str
    slot: tuple
    tone: str | None = None

    @property
    def urgency(self) -> str:
        return self.cell.split("/")[1]


def load_examples(bank: str) -> list[Example]:
    examples = []
    for path in sorted((TEXTBANK_DIR / bank).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            examples.append(Example(
                text=r["text"], cell=f"{r['category']}/{r['urgency']}",
                slot=(r["category"], r["urgency"], r["topic"], r["speaker"]), tone=r.get("tone"),
            ))
    return examples


def split_dev(examples: list[Example], frac: float = 0.2, seed: int = 0):
    """Holds back `frac` of every slot, so the dev set covers every topic and speaker."""
    by_slot = defaultdict(list)
    for ex in examples:
        by_slot[ex.slot].append(ex)
    rng = random.Random(seed)
    train, dev = [], []
    for slot in sorted(by_slot):
        items = by_slot[slot][:]
        rng.shuffle(items)
        k = round(len(items) * frac)
        dev += items[:k]
        train += items[k:]
    return train, dev


def evaluate(triage: Triage, examples: list[Example]) -> dict:
    texts = [ex.text for ex in examples]
    results = triage.run_many(texts)
    model_cells = [max(p, key=p.get) for p in triage.model.predict_proba(texts)]
    rule_hit = [bool(find_hits(t, triage.rules)) for t in texts]

    crit = [i for i, ex in enumerate(examples) if ex.urgency == "critical"]
    other = [i for i, ex in enumerate(examples) if ex.urgency != "critical"]
    pred_rank = [URGENCY_RANK[r.urgency] for r in results]
    true_rank = [URGENCY_RANK[ex.urgency] for ex in examples]

    def share(idx, cond):
        return round(sum(cond(i) for i in idx) / len(idx), 4) if idx else None

    per_cell = defaultdict(lambda: [0, 0])
    for ex, r in zip(examples, results):
        per_cell[ex.cell][1] += 1
        per_cell[ex.cell][0] += r.cell == ex.cell
    confusion = Counter((ex.urgency, r.urgency) for ex, r in zip(examples, results))

    return {
        "n": len(examples),
        "critical_recall": {
            "combined": share(crit, lambda i: results[i].urgency == "critical"),
            "model_only": share(crit, lambda i: model_cells[i].endswith("/critical")),
            "rules_only": share(crit, lambda i: rule_hit[i]),
        },
        "under_triage_rate": share(range(len(examples)), lambda i: pred_rank[i] < true_rank[i]),
        "over_triage_rate": share(range(len(examples)), lambda i: pred_rank[i] > true_rank[i]),
        "false_critical_rate": share(other, lambda i: results[i].urgency == "critical"),
        "interrupt": {
            "critical_interrupted": share(crit, lambda i: results[i].interrupt),
            "false_interrupt_rate": share(other, lambda i: results[i].interrupt),
        },
        "cell_accuracy": share(range(len(examples)), lambda i: results[i].cell == examples[i].cell),
        "cell_accuracy_model_only": share(range(len(examples)),
                                          lambda i: model_cells[i] == examples[i].cell),
        "urgency_accuracy": share(range(len(examples)), lambda i: pred_rank[i] == true_rank[i]),
        "urgency_confusion": {f"{t}->{p}": n for (t, p), n in sorted(confusion.items())},
        "per_cell_recall": {c: round(k / n, 4) for c, (k, n) in sorted(per_cell.items())},
        "missed_critical": [
            {"text": examples[i].text, "true": examples[i].cell, "pred": results[i].cell}
            for i in crit if results[i].urgency != "critical"
        ],
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def print_report(title: str, m: dict, show: int) -> None:
    cr = m["critical_recall"]
    print(f"\n== {title} ({m['n']} texts)")
    print(f"Critical recall    combined {_pct(cr['combined'])}   model only {_pct(cr['model_only'])}"
          f"   rules only {_pct(cr['rules_only'])}")
    print(f"Under-triage {_pct(m['under_triage_rate'])}   over-triage {_pct(m['over_triage_rate'])}"
          f"   non-critical flagged critical {_pct(m['false_critical_rate'])}")
    print(f"Interrupts: criticals interrupting {_pct(m['interrupt']['critical_interrupted'])}"
          f"   false interrupts {_pct(m['interrupt']['false_interrupt_rate'])}")
    print(f"Cell accuracy {_pct(m['cell_accuracy'])} (model only {_pct(m['cell_accuracy_model_only'])})"
          f"   urgency accuracy {_pct(m['urgency_accuracy'])}")
    print("Urgency (true->predicted): " + ", ".join(f"{k} {v}" for k, v in m["urgency_confusion"].items()))
    weakest = sorted(m["per_cell_recall"].items(), key=lambda kv: kv[1])[:6]
    print("Weakest cells: " + ", ".join(f"{c} {r:.0%}" for c, r in weakest))
    if m["missed_critical"]:
        print(f"Missed criticals ({len(m['missed_critical'])}):")
        for miss in m["missed_critical"][:show]:
            print(f"  {miss['true']} -> {miss['pred']}: {miss['text']}")


def fingerprint(bank: str) -> str:
    h = hashlib.sha256()
    for path in sorted((TEXTBANK_DIR / bank).glob("*.jsonl")):
        h.update(path.read_bytes())
    return h.hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--final", action="store_true",
                        help="also evaluate on the test banks; only once settings are fixed")
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--model", default="tfidf", help='"tfidf" or "emb:<hugging face id>"')
    parser.add_argument("--show", type=int, default=10, help="missed criticals to print")
    parser.add_argument("--with-tone-bank", action="store_true",
                        help="also train on train/gemini_tone (calm criticals, panicky non-criticals)")
    args = parser.parse_args()

    train, dev = split_dev(load_examples("train/gemini"))
    if args.with_tone_bank:
        # Training only: the dev split stays the plain bank, so dev numbers stay comparable.
        train += load_examples("train/gemini_tone")
    model = make_model(args.model, C=args.C).fit([ex.text for ex in train], [ex.cell for ex in train])
    triage = Triage(model, load_rules())

    reports = {"dev (20% of train/gemini)": evaluate(triage, dev)}
    if args.final:
        tone = load_examples("test/tone")
        reports["test/claude"] = evaluate(triage, load_examples("test/claude"))
        reports["test/tone: calm criticals"] = evaluate(triage, [e for e in tone if e.urgency == "critical"])
        reports["test/tone: panicky non-criticals"] = evaluate(triage, [e for e in tone if e.urgency != "critical"])
        # Written by a fresh sub-agent that never saw our errors, rules or protocol: the clean test.
        if (TEXTBANK_DIR / "test" / "fresh").exists():
            fresh = load_examples("test/fresh")
            reports["test/fresh"] = evaluate(triage, fresh)
            for tone_name in ("calm", "neutral", "urgent"):
                reports[f"test/fresh: {tone_name}"] = evaluate(triage, [e for e in fresh if e.tone == tone_name])
    for title, metrics in reports.items():
        print_report(title, metrics, args.show)

    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    record = {
        "time": stamp, "model": model.name, "settings": {**model.settings, "tone_bank": args.with_tone_bank},
        "pipeline": triage.version,
        "data": {b: fingerprint(b) for b in ("train/gemini", "train/gemini_tone", "test/claude", "test/tone", "test/fresh")},
        "split": {"dev_fraction": 0.2, "seed": 0, "train": len(train), "dev": len(dev)},
        "reports": reports,
    }
    out = EXPERIMENTS_DIR / f"{stamp}_{model.name}{'_tone' if args.with_tone_bank else ''}.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
