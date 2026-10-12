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

Every number in `data/config/ward.yaml` is a labeled assumption. Intents and categories live in
`data/config/intents.yaml` and `data/config/categories.yaml`.

## Tests

```sh
uv run pytest
```
