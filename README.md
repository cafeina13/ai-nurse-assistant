# AI Nurse Assistant

A triage engine for Turkish ward patient requests. A patient types or says a short request at
the bedside ("nefes alamıyorum", "su alabilir miyim"); the engine sorts it into one of 28 cells
(category x urgency) so the nurse board shows emergencies first, each with a readable reason.
It sorts and flags; it never diagnoses, never advises patients and never asks them follow-up
questions. Synthetic data only; not validated by nurses.

**Results** (fresh test set, 420 texts written blind to the design):

| Engine | Critical recall | False criticals |
|---|---:|---:|
| This engine (TF-IDF + safety rules + protocol) | **96.1%** | 3.8% |
| Llama 3.1 8B, zero-shot | 61.8% | 3.1% |
| Dousery (Turkish medical triage model) | calls ~80% of all requests urgent | |

Spoken by synthetic voices and transcribed with Whisper: 95.1% critical recall in clean audio,
94.6% with ward noise, 80.4% in hard audio.

- [docs/results.md](docs/results.md): every number, with the experiment record or command behind it
- [docs/report.md](docs/report.md): the full write-up (methods, discussion, limitations)

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
example phrase per topic.

Text banks (`data/textbank/`) hold Turkish wordings per grid cell, topic and speaker:

- `train/gemini/`: 5,600 texts written by Gemini, for training. Rebuild or extend with
  `uv run python -m wardgen.gemini_fill` (needs `GEMINI_API_KEY` in `.env`; resumable).
- `test/claude/`: 560 texts written by Claude, for testing. A different model than training,
  so a classifier cannot score well by learning one generator's style.
- `train/gemini_tone/`: 956 calm criticals and panicky non-criticals, so tone stops
  standing in for urgency.
- `test/tone/`: the same idea as a test (54 texts).
- `test/fresh/`: 420 texts written by a separate model instance that saw only the slot
  specification, never our rules, protocol or errors. The final test set.

Every number in `data/config/ward.yaml` is a labeled assumption. The request grid (categories, urgency
levels and example topics per cell) lives in `data/config/grid.yaml`.

## Demo: patient page and live nurse board

```sh
uv run uvicorn backend.app:app --port 8000
```

- Nurse board: http://localhost:8000/board
- Patient page: http://localhost:8000/patient (open it in a second window or on a phone)

Each start builds a fresh synthetic ward. "Vardiyayı oynat" on the board replays a 24-hour
shift through the same triage pipeline (×600 = 10 simulated minutes per second), using the
Claude-written test texts the model never trained on. Critical requests top the queue; only
confident ones beep. Completing a critical asks whether it really was critical.

## Evaluation

```sh
uv run python -m triage.evaluate --final --with-tone-bank --model hybrid-critical:tfidf
uv run python -m compare.report data/generated/compare_fresh --with-tone-bank --model hybrid-critical:tfidf
```

The second command needs the competitors' answers, which a Kaggle run downloads to
`data/generated/` (not in the repo). GPU work (embeddings, BERTurk, competitors, voice) runs
on Kaggle: `triage.kaggle_triage`,
`compare.kaggle_compare`, `voice.kaggle_job`.

## Tests

```sh
uv run pytest
```
