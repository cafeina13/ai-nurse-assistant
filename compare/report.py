"""Head to head: our engine vs the competitors, on the same test texts and the same scale.

    python -m compare.report data/generated/compare
    python -m compare.report data/generated/compare --with-tone-bank   # ours also trained on tone texts

Every engine is scored on urgency (critical / soon / routine), the one scale all of them
produce. Dousery answers "LEVEL: advice"; its levels are mapped two ways (ACİL as critical,
and ACİL as soon) so the result's dependence on that choice is visible.

Caveat to report: the grid, the labels and the test set are ours. The test texts were written
by a different model than our training texts, but the label definitions are still home ground.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from triage.evaluate import load_examples
from triage.models import TfidfModel
from triage.normalize import fold
from triage.pipeline import URGENCY_RANK, Triage

DOUSERY_MAPS = {
    "ACİL = critical": {"cok acil": "critical", "acil": "critical", "orta": "soon",
                        "normal": "routine", "dusuk": "routine", "acil degil": "routine"},
    "ACİL = soon": {"cok acil": "critical", "acil": "soon", "orta": "soon",
                    "normal": "routine", "dusuk": "routine", "acil degil": "routine"},
}


def dousery_level(raw: str) -> str:
    """Its level from the start of the answer, folded: "ÇOK ACİL: Hemen 112 arayın." -> "cok acil".
    It sometimes writes a sentence instead ("Normal bir durum…"), so the first words decide."""
    head = fold(raw.split(":", 1)[0] if ":" in raw else raw)
    for level in ("cok acil", "cok onemli", "acil degil", "acil", "orta", "normal", "dusuk"):
        if head.startswith(level):
            return "cok acil" if level == "cok onemli" else level
    return head[:20]


def llama_cell(raw: str) -> tuple[str | None, str | None]:
    try:
        answer = json.loads(raw)
        return answer.get("category"), answer.get("urgency")
    except (json.JSONDecodeError, AttributeError):
        return None, None


def score(rows: list[dict], predicted: list[str | None]) -> dict:
    crit = [i for i, r in enumerate(rows) if r["urgency"] == "critical"]
    other = [i for i, r in enumerate(rows) if r["urgency"] != "critical"]

    def share(idx, cond):
        return sum(cond(i) for i in idx) / len(idx) if idx else None

    rank = lambda u: URGENCY_RANK.get(u, -1)  # unparsed answers count as below routine
    return {
        "n": len(rows),
        "crit_recall": share(crit, lambda i: predicted[i] == "critical"),
        "crit_to_routine": share(crit, lambda i: predicted[i] in ("routine", None)),
        "false_critical": share(other, lambda i: predicted[i] == "critical"),
        "under_triage": share(range(len(rows)), lambda i: rank(predicted[i]) < rank(rows[i]["urgency"])),
        "urgency_acc": share(range(len(rows)), lambda i: predicted[i] == rows[i]["urgency"]),
        "unparsed": share(range(len(rows)), lambda i: predicted[i] is None),
    }


def _pct(x):
    return "   n/a" if x is None else f"{x:6.1%}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, help="folder with dousery.jsonl and llama31.jsonl")
    parser.add_argument("--with-tone-bank", action="store_true",
                        help="train ours on train/gemini plus train/gemini_tone")
    args = parser.parse_args()

    engines = {}
    for name in ("dousery", "llama31"):
        path = args.results / f"{name}.jsonl"
        if path.exists():
            engines[name] = {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    ids = sorted(set.intersection(*(set(e) for e in engines.values()))) if engines else []
    rows = [next(iter(engines.values()))[i] for i in ids]
    print(f"{len(rows)} test texts answered by every engine")

    train = load_examples("train/gemini")
    if args.with_tone_bank:
        train += load_examples("train/gemini_tone")
    ours = Triage(TfidfModel().fit([e.text for e in train], [e.cell for e in train]))
    ours_results = ours.run_many([r["text"] for r in rows])
    predictions = {f"ours (TF-IDF + rules{', tone bank' if args.with_tone_bank else ''})":
                   [r.urgency for r in ours_results]}

    if "llama31" in engines:
        predictions["Llama 3.1 8B zero-shot"] = [llama_cell(engines["llama31"][i]["raw"])[1] for i in ids]
    if "dousery" in engines:
        levels = [dousery_level(engines["dousery"][i]["raw"]) for i in ids]
        print("Dousery levels seen:", dict(Counter(levels).most_common()))
        for label, mapping in DOUSERY_MAPS.items():
            predictions[f"Dousery ({label})"] = [mapping.get(level) for level in levels]

    subsets = {
        "test/claude": [k for k, r in enumerate(rows) if r["bank"] == "test/claude"],
        "calm criticals": [k for k, r in enumerate(rows) if r.get("tone") == "calm_critical"],
        "panicky non-criticals": [k for k, r in enumerate(rows) if r.get("tone") == "panicky_noncritical"],
    }
    for title, idx in subsets.items():
        if not idx:
            continue
        print(f"\n== {title} ({len(idx)} texts)")
        print(f"  {'engine':<40} {'crit rec':>8} {'crit->rout':>10} {'false crit':>10} "
              f"{'under-tri':>9} {'urg acc':>7} {'unparsed':>8}")
        for name, pred in predictions.items():
            s = score([rows[k] for k in idx], [pred[k] for k in idx])
            print(f"  {name:<40} {_pct(s['crit_recall']):>8} {_pct(s['crit_to_routine']):>10} "
                  f"{_pct(s['false_critical']):>10} {_pct(s['under_triage']):>9} "
                  f"{_pct(s['urgency_acc']):>7} {_pct(s['unparsed']):>8}")

    if "llama31" in engines:
        cells = [llama_cell(engines["llama31"][i]["raw"]) for i in ids]
        exact = sum(f"{c}/{u}" == f"{r['category']}/{r['urgency']}" for (c, u), r in zip(cells, rows))
        mine = sum(o.cell == f"{r['category']}/{r['urgency']}" for o, r in zip(ours_results, rows))
        print(f"\nFull cell (category + urgency) accuracy: ours {mine / len(rows):.1%}, "
              f"Llama 3.1 zero-shot {exact / len(rows):.1%}")


if __name__ == "__main__":
    main()
