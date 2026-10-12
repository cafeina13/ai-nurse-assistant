"""python -m wardgen data/generated/ward.db --seed 42 --days 1"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from sqlalchemy import select

from backend.db import make_engine, session_factory
from backend.models import Patient, ScenarioCall
from wardgen import generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic ward and its calls.")
    parser.add_argument("db", type=Path, help="SQLite file to create")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--textbank", type=Path,
                        help="text bank directory, e.g. data/textbank/train/gemini "
                             "(default: one example phrase per intent)")
    parser.add_argument("--force", action="store_true", help="replace the file if it exists")
    args = parser.parse_args()

    if args.db.exists():
        if not args.force:
            sys.exit(f"{args.db} exists; pass --force to replace it")
        args.db.unlink()
    args.db.parent.mkdir(parents=True, exist_ok=True)

    count = generate(args.db, seed=args.seed, days=args.days, textbank=args.textbank)
    print(f"Wrote {args.db}: {count} calls over {args.days} day(s), seed {args.seed}\n")
    print_summary(args.db)


def print_summary(db: Path) -> None:
    with session_factory(make_engine(db))() as session:
        patients = list(session.scalars(select(Patient)))
        calls = list(session.scalars(select(ScenarioCall).order_by(ScenarioCall.at)))

    mobility = Counter(p.mobility for p in patients)
    print(f"Patients: {len(patients)}  "
          + "  ".join(f"{k}={v}" for k, v in sorted(mobility.items())))
    print(f"With IV: {sum(p.has_iv for p in patients)}  "
          f"With companion: {sum(p.has_companion for p in patients)}\n")

    _print_counts("Urgency", Counter(c.urgency for c in calls), len(calls))
    _print_counts("Category", Counter(c.category for c in calls), len(calls))
    _print_counts("Speaker", Counter(c.speaker for c in calls), len(calls))
    _print_counts("Channel", Counter(c.channel for c in calls), len(calls))

    by_hour = Counter(c.at.hour for c in calls)
    peak = max(by_hour.values(), default=1)
    print("Calls by hour of day")
    for hour in range(24):
        n = by_hour.get(hour, 0)
        print(f"  {hour:02d}:00 {'#' * round(30 * n / peak):<30} {n}")


def _print_counts(title: str, counts: Counter, total: int) -> None:
    print(title)
    for key, n in counts.most_common():
        print(f"  {key:<18} {n:>5}  {100 * n / total:5.1f}%")
    print()


if __name__ == "__main__":
    main()
