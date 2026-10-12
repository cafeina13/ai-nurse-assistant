# AI Nurse Assistant

A Turkish-language assistant that triages patient calls, protects nurses from interruptions
and drafts shift handovers. It supports nurses and never makes clinical decisions.
Synthetic data only.

## Setup

```sh
uv sync
```

## Module 0: synthetic ward

Generate a ward and a day of calls, then replay it on a simulated clock:

```sh
uv run python -m wardgen data/generated/ward.db --seed 42 --days 1
uv run python -m backend.replay data/generated/ward.db --speed 3600   # one simulated hour per second
```

Add `--textbank data/textbank/train/gemini` to use varied request texts instead of one
example phrase per intent.

Text banks (`data/textbank/`) hold Turkish wordings per intent, urgency and speaker:

- `train/gemini/`: 4,160 texts written by Gemini, for training. Rebuild or extend with
  `uv run python -m wardgen.gemini_fill` (needs `GEMINI_API_KEY` in `.env`; resumable).
- `test/claude/`: 416 texts written by Claude, for testing. A different model than training,
  so a classifier cannot score well by learning one generator's style.
- `test/hand/`: hand-written test cases, including needs that are deliberately not in the
  intent list.

Every number in `data/config/ward.yaml` is a labeled assumption. Intents and categories live in
`data/config/intents.yaml` and `data/config/categories.yaml`.

## Tests

```sh
uv run pytest
```
