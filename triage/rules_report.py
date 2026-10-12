"""How the safety rules do on a text bank: critical recall and false alarms.

    python -m triage.rules_report                        # training bank (for tuning)
    python -m triage.rules_report --bank test/claude --final

Tune rules on the training bank only. The test bank is for the final, honest number.
"""

import argparse
import sys
from collections import Counter, defaultdict

from triage.rules import find_hits, load_rules
from wardgen.textbank import TEXTBANK_DIR, load_bank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bank", default="train/gemini")
    parser.add_argument("--final", action="store_true",
                        help="required for a test bank: confirms tuning is finished")
    parser.add_argument("--show", type=int, default=5, help="examples to print per problem")
    args = parser.parse_args()
    if args.bank.startswith("test/") and not args.final:
        sys.exit("Refusing to report on a test bank without --final: tune on the training bank.")

    rules = load_rules()
    bank = load_bank(TEXTBANK_DIR / args.bank)
    ruled_cells = {r.cell for r in rules}

    caught, caught_any, total = Counter(), Counter(), Counter()
    misses, false_alarms, wrong_cell = defaultdict(list), defaultdict(list), []
    for (category, urgency, topic, _speaker), texts in bank.items():
        cell = f"{category}/{urgency}"
        for text in texts:
            hits = find_hits(text, rules)
            if cell in ruled_cells:
                total[cell] += 1
                caught_any[cell] += bool(hits)
                if any(h.cell == cell for h in hits):
                    caught[cell] += 1
                else:
                    misses[f"{cell}/{topic}"].append(text)
                    if hits:
                        wrong_cell.append((cell, hits[0].cell, text))
            elif hits:
                false_alarms[f"{cell}/{topic}"].append((hits[0].pattern, text))

    n_texts = sum(len(t) for t in bank.values())
    print(f"{args.bank}: {n_texts} texts, {len(rules)} rules\n")
    print("Critical recall by cell: same-cell rule fired / any critical rule fired")
    for cell in sorted(total):
        t = total[cell]
        print(f"  {cell:<24} {caught[cell]:>4}/{t:<4} {100 * caught[cell] / t:5.1f}%"
              f"   {caught_any[cell]:>4}/{t:<4} {100 * caught_any[cell] / t:5.1f}%")
    all_caught, all_total = sum(caught.values()), sum(total.values())
    all_any = sum(caught_any.values())
    print(f"  {'all ruled cells':<24} {all_caught:>4}/{all_total:<4} {100 * all_caught / all_total:5.1f}%"
          f"   {all_any:>4}/{all_total:<4} {100 * all_any / all_total:5.1f}%")

    n_false = sum(len(v) for v in false_alarms.values())
    n_other = n_texts - all_total
    print(f"\nFalse alarms on non-critical texts: {n_false}/{n_other} ({100 * n_false / n_other:.1f}%)")
    for slot, items in sorted(false_alarms.items(), key=lambda kv: -len(kv[1])):
        print(f"  {slot:<44} {len(items):>3}")
        for pattern, text in items[:args.show]:
            print(f"      [{pattern}] {text}")

    print(f"\nCritical texts caught by a rule for a different critical cell: {len(wrong_cell)}")
    for cell, hit_cell, text in wrong_cell[:args.show]:
        print(f"  {cell} -> {hit_cell}: {text}")

    print("\nMissed critical texts by topic")
    for slot, texts in sorted(misses.items(), key=lambda kv: -len(kv[1])):
        print(f"  {slot:<44} {len(texts):>3}")
        for text in texts[:args.show]:
            print(f"      {text}")


if __name__ == "__main__":
    main()
