# Results: a Turkish triage engine for ward patient requests

This is the backbone of the course report ("AI in Healthcare"). Every number below names the
record it comes from, so it can be checked and reproduced. Dates: all runs 2026-10-15.

**The task.** A patient (or companion) on a ward presses a button or types a short Turkish
request. The engine puts it in one *cell* of a grid (category x urgency, e.g.
`breathing/critical`) so the nurse board can order the queue. The engine orders and flags; it
never makes a clinical decision, never diagnoses, and never asks the patient follow-up
questions.

**The headline.** On a test set written independently of our development (the *fresh* set,
420 texts), our hybrid engine catches **96.1%** of critical requests with **3.8%** false
criticals. The two existing engines we could run score 61.8% (Llama 3.1 8B, zero-shot) and
"everything is urgent" (Dousery, a Turkish medical model that calls about 80% of all requests
ACİL). Read this with the caveats in section 6: the grid, the labels and the data are ours.

---

## 1. Methods

### 1.1 The grid (`data/config/grid.yaml`)
12 categories (breathing, pain, neuro, other_symptom, fall_injury, physical_help,
comfort_supplies, medication, equipment, information, emotional, unclear) and the urgency
levels each can have, 28 cells in all, with 70 example topics that steer the generator. Topic
weights are an assumption shaped by the request ranking in Tzeng 2010 (bathroom help > pain
medication > IV/pump problems > personal help > repositioning). **Not validated by a nurse.**

### 1.2 Data (all synthetic, approved by the professor)

| Bank | Texts | Written by | Use |
|---|---:|---|---|
| `train/gemini` | 5,600 | Gemini, filling grid slots (`data/wardgen/gemini_fill.py`) | training; 20% per slot held out as **dev** |
| `train/gemini_tone` | 956 | Gemini with `--tone calm` (320 calm criticals) / `--tone panicky` (636 panicky non-criticals) | training only |
| `test/claude` | 560 | Claude (a different model family) | test |
| `test/tone` | 54 | Claude: 28 calm criticals, 26 panicky non-criticals | test of the tone shortcut |
| `test/fresh` | 420 | a clean Claude sub-agent that saw only the slot spec (no rules, protocol, errors), one calm / neutral / urgent text per slot | **final test** |

Slots cover speaker (patient / companion), register, spelling (with and without Turkish
letters, typos) and patient context (e.g. cannot walk alone).

### 1.3 The engine (`triage/`)
Layers, in order (`triage/pipeline.py`):

1. **Buttons**: a pressed button (e.g. "Ağrı", "Tuvalet") gives the category directly.
2. **Model**: a classifier gives the cell with a confidence.
3. **Safety rules** (`data/config/safety_phrases.yaml`, 51 rules): red-flag phrases ("nefes
   alamıyorum", "bayıldı", "kan kusuyor") force critical. Rules always win, and only raise.
   Options `not_after`, `unless`, `unless_in_match` stop false hits such as negations.
4. **Patient context**: a routine toilet request from a patient who cannot walk alone becomes
   soon.
5. **Flags**: `uncertain` (low confidence, shown as "Emin değil — kontrol edin") and
   `interrupt` (only a confident critical plays a sound: *order generously, interrupt
   carefully*).

Text is folded first (`triage/normalize.py`: lower case, Turkish letters to ASCII), so
"nefes alamıyorum" and "nefes alamiyom" meet.

**Models tried** (`triage/models.py`, `make_model`):
- **TF-IDF**: character 2-5 + word 1-2 grams, logistic regression, C = 10.
- **Frozen embeddings** + logistic regression: multilingual-e5-base / large, bge-m3,
  trmteb/turkish-embedding-model (chosen from TR-MTEB).
- **Protocol** (`triage/protocol.py`, `data/config/protocol.yaml`): no training. 59 facts
  (regex patterns, e.g. `vomit`, `dark_vomit`, `breath`) and 47 ordered lines
  ("vomit + (blood / dark_vomit) -> other_symptom/critical"); the first line that holds
  decides. A nurse can read and correct it, and every decision names its line and facts, which
  the board shows in Turkish ("protokol: kusma + koyu kusmuk").
- **Hybrid** = classifier + protocol, four ways of combining them (`HybridModel`):
  `raise` (higher urgency wins), `critical` (the classifier decides, but a protocol critical
  line overrides it), `protocol` (the protocol decides whenever a specific line matches),
  `both` (`critical`, plus a weak model critical yields to a non-critical protocol line).
  A critical found only by the protocol gets confidence 0.6: it tops the queue but does not
  interrupt.

The mode was chosen on **dev only**; the test sets were run once with the chosen mode.

### 1.4 Competitors (`compare/`)
Run on Kaggle (T4 x2) on the same texts (`compare/prompts.py`):
- **Dousery** (`dousery/turkish-medical-triage-llama3-gguf`, `unsloth.Q4_K_M.gguf`): a Turkish
  medical triage model. It gets the raw message with the system prompt from its model card and
  answers "LEVEL: advice". Its levels are mapped two ways (ACİL = critical, ACİL = soon), so the
  result's dependence on that choice is visible (`compare/report.py`).
- **Llama 3.1 8B Instruct**, zero-shot. Its (English) system prompt gives what a human labeler
  gets: every cell's meaning, example topics and the hard-case rules; it answers JSON
  `{category, urgency}`.

All engines are scored on the one scale they all produce: urgency (critical / soon / routine).

### 1.5 Voice (`voice/`)
No real recordings were available, so test texts were spoken by TTS voices (Drifting, Piper
dfki, Piper "Merve" (a voice the author trained), MMS-tur) and degraded three ways (`clean`,
`ward` = ward noise, `hard` = noise + reverb + low bandwidth; `voice/audio.py`). They were
transcribed with faster-whisper on Kaggle, and triage ran on the original text and on the
transcript, so the drop between the two is what speech to text costs (`voice/report.py`).

### 1.6 Metrics
**Critical recall comes first** (a missed critical is the worst error). Also: false critical
rate (non-criticals flagged critical), under-triage (any level too low), critical -> routine,
false interrupts, urgency accuracy, cell accuracy.

---

## 2. Text results

### 2.1 Development history (dev = 20% of `train/gemini`, n = 1,120)

| Step | Crit. recall | False crit. | Cell acc. | Record |
|---|---:|---:|---:|---|
| TF-IDF + rules | 94.5% | 3.8% | 88.1% | `experiments/20261015-154418_*` |
| ... rules alone | 55.5% | | | same |
| TF-IDF + rules + tone bank | 94.1% | 5.5% | 86.7% | `20261015-222913_*_tone` |
| Protocol alone, first draft | 64.3% | 4.8% | 62.3% | `20261015-222528_protocol` |
| Protocol alone, revised | 73.9% | 5.5% | 68.5% | `20261015-222734_protocol` |
| Hybrid `raise` | 96.0% | 7.4% | 82.3% | `20261015-222825_*` |
| **Hybrid `critical`** (chosen) | **96.0%** | 7.4% | **85.6%** | `20261015-222833_*` |
| Hybrid `protocol` | 82.0% | 5.7% | 74.4% | `20261015-222840_*` |
| Hybrid `both` (rejected) | 92.7% | 6.6% | 85.2% | `20261015-230541_*` |

`critical` and `raise` tie on critical recall; `critical` keeps the classifier's cell for
everything else, so cell accuracy is 3 points higher. `both` bought under one point fewer
false alarms for 3.3 points of critical recall, so it was rejected.

### 2.2 Embeddings vs TF-IDF (all with the tone bank, `20261015-1820*` and `-1822*`)

| Model | Dev crit. | Claude test crit. | Claude false crit. | Claude cell acc. |
|---|---:|---:|---:|---:|
| TF-IDF | 94.1% | 86.8% | 8.5% | 73.4% |
| turkish-embedding-model | 94.8% | 86.8% | 6.4% | 76.1% |
| bge-m3 | 94.1% | 85.3% | 5.9% | 73.4% |
| multilingual-e5-large | 93.4% | 83.8% | 8.7% | 72.1% |
| multilingual-e5-base | 91.9% | 86.8% | 10.1% | 66.2% |

Frozen embeddings only **tie** TF-IDF on critical recall. They miss *different* texts,
though (only 6 of about 40 missed criticals were shared), which points to ensembles as future
work.

### 2.3 The tone shortcut and the tone bank
Our first model had learned "panicky wording = critical". Without tone training, a calm
"annemin dudakları morardı" was easy to miss, and an excited "ACİL battaniye lazım!" was
flagged critical.

| `test/tone` | TF-IDF, no tone bank | TF-IDF + tone bank | Hybrid + tone bank |
|---|---:|---:|---:|
| Calm criticals: critical recall (n = 28) | 82.1% (model alone 64.3%) | 89.3% | **100%** |
| Panicky non-criticals: false critical (n = 26) | 26.9% | **0%** | 0% |

Records: `20261015-210340` (no tone bank) vs `20261015-210349` / `-222922`. The price: false
criticals on the Claude test rose from 5.7% to 8.5%.

### 2.4 The contamination problem, and the fresh set
The protocol's first draft was written **after** seeing which `test/claude` criticals every
model missed ("kahve telvesi gibi kusmuk", "simsiyah dışkı", fever + confusion, ...). Its
`test/claude` score (hybrid 94.8% vs TF-IDF 86.8%, `20261015-222922`) is therefore
optimistic. To get an honest number, a fresh sub-agent wrote `test/fresh` from the slot spec
alone, without ever seeing our rules, protocol, errors or test texts. **Nothing was tuned on
it.**

| `test/fresh` (n = 420), `evaluate.py` | TF-IDF + rules | Hybrid `critical` + rules |
|---|---:|---:|
| Critical recall | 89.2% | **96.1%** |
| ... calm / neutral / urgent | 85.3 / 85.3 / 97.1% | 94.1 / 94.1 / 100% |
| False critical | 3.5% | 4.7% |
| Under-triage | 6.2% | 4.5% |
| False interrupts | 0% | 0% |
| Cell accuracy | 86.4% | 87.1% |

Records: `20261015-224242` (TF-IDF), `20261015-224251` (hybrid). The gain held on texts
nobody had seen: about 7 points of critical recall for about 1 point of false criticals. So the
protocol adds real knowledge, not just patches for known test errors.

### 2.5 Head to head with existing engines (the main result)
Urgency scale, `test/fresh`, all 420 texts answered by every engine. Ours is trained on all of
`train/gemini` + `train/gemini_tone` (no dev hold-out), which is why its numbers differ
slightly from 2.4 (false critical 3.8% here vs 4.7% there). Reproduce:
`python -m compare.report data/generated/compare_fresh --with-tone-bank --model hybrid-critical:tfidf`.

| Engine | Crit. recall | False crit. | Under-triage | Urgency acc. | Cell acc. |
|---|---:|---:|---:|---:|---:|
| **Ours** (hybrid `critical` + rules, tone bank) | **96.1%** | 3.8% | **5.0%** | **90.5%** | **87.1%** |
| Llama 3.1 8B zero-shot | 61.8% | 3.1% | 10.2% | 73.1% | 58.3% |
| Dousery, ACİL = critical | 100% | 74.8% | 1.2% | 42.1% | n/a |
| Dousery, ACİL = soon | 2.9% | 0% | 24.8% | 50.7% | n/a |

By tone (critical recall, ours vs Llama): calm 94.1% vs 52.9%, neutral 94.1% vs 58.8%,
urgent 100% vs 73.5%. Llama's misses track tone: a calmly stated emergency is the case it
loses most.

**Dousery** called 337 of 420 requests ACİL, 78 NORMAL and only 3 ÇOK ACİL. It does not sort:
whichever mapping you choose, it either floods the board with criticals or misses almost all
of them. It also writes diagnoses and tells patients to call 112. On the earlier 614-text run,
it suggested a diagnosis in 298 answers and said to call 112 in 393. Both break our core
principle (sort, never decide), and both are unsafe for a patient-facing device.

Earlier run on `test/claude` + `test/tone` (614 texts, TF-IDF + rules + tone bank, before the
protocol existed): ours 86.0% critical recall / 7.3% false critical, Llama 52.9% / 5.0%,
full-cell accuracy 76.7% vs 46.3% (`data/generated/compare`).

---

## 3. Voice results

Speech to text ran on Kaggle T4s (faster-whisper); the laptop only re-scores the saved
transcripts, so changing the triage engine needs no new speech run. Runs 1 and 2 read
`test/claude` + `test/tone` aloud (614 texts); run 3 reads `test/fresh` (420 texts). Records:
`data/generated/voice_eval/` (gitignored), scored with `python -m voice.report`.

Runs 1 and 2 were scored with the early engine (TF-IDF + rules, typed texts only). Section
3.1 re-scores run 2 with the final engine; **section 3.2 (the fresh set) is the voice result
to quote.**

**Run 1** (large-v3, 3,684 clips = 614 texts x 2 voices x 3 conditions):

| Condition | WER | Crit. recall, text -> voice | Crit. -> routine |
|---|---:|---:|---:|
| clean | 21.6% | 85.4% -> 80.2% | 5.5% |
| ward | 27.5% | 85.4% -> 76.8% | 5.2% |
| hard | 51.2% | 85.4% -> 57.9% | 18.6% |

The "Merve" voice (a noisy home-trained Piper voice) was far worse (clean WER 54%), which shows
how much the result depends on speech quality. On noise, Whisper hallucinates YouTube
endings ("İzlediğiniz için teşekkür ederim"); a filter catches these (`voice/checks.py`).

**Run 2, STT presets** (3,222 clips per preset):

| Preset | Hard WER | Hard crit. recall | Hard crit. -> routine |
|---|---:|---:|---:|
| large-v3 + Syntia decoding | 53.6% | 56.5% | 18.6% |
| **large-v3 + Syntia decoding + ward hint** | **50.7%** | **63.9%** | **13.7%** |
| turbo + Syntia decoding | 54.8% | 58.6% | 21.1% |

The hint (a short ward vocabulary prompt) helps most in the hard condition, and it adds no
false criticals. It is the preset to use for voice (live voice in the demo is skipped: it
would run Whisper on the laptop).

**The "never routine when in doubt" floor** (a doubtful transcript is never ranked below
soon): critical -> routine falls to about 1-2% in every condition, but about a third of
routine requests rise to soon (clean 32.8%, hard 50.3%). That trade-off is for a nurse to
judge, not for us.

### 3.1 The final engine on voice
Run 2 transcripts, preset large-v3 + Syntia + hint (3,222 clips). Each engine runs on the
original text and on the transcript, so text -> voice is what speech to text costs.

| Engine | Clean crit. | Ward crit. | Hard crit. | Hard crit. -> routine | False crit. clean / hard |
|---|---:|---:|---:|---:|---:|
| TF-IDF + rules (early) | 85.3 -> 80.4% | 85.3 -> 75.8% | 85.3 -> 63.9% | 13.7% | 6.3 / 9.6% |
| TF-IDF + rules + tone bank | 87.0 -> 83.2% | 87.0 -> 80.0% | 87.0 -> 69.5% | 12.6% | 8.2 / 14.2% |
| **Hybrid `critical` + tone bank** (demo) | 96.1 -> **91.6%** | 96.1 -> **89.1%** | 96.1 -> **75.8%** | 11.6% | 9.3 / 15.1% |
| ... + checks + floor | 96.1 -> 91.6% | 96.1 -> 89.1% | 96.1 -> 75.4% | **1.4%** | 9.3 / 14.7% |

Reproduce: `python -m voice.report data/generated/voice_eval/stt/transcripts.jsonl --stt
large-v3-syntia-hint --with-tone-bank --model hybrid-critical:tfidf [--checks --floor listen]`.

- The hybrid's gain survives speech to text: about 11 points of critical recall on clean and
  ward speech, 12 on hard. The *drop* caused by speech to text stays about the same (clean
  about 4.5, ward about 7, hard about 20 points), so the engine does not fix bad audio; it
  just starts higher.
- The price is false criticals on noisy speech (hard 9.6% -> 15.1%): garbled transcripts
  trip protocol facts more often.
- Calm criticals by voice: 95.7% clean, 91.5% ward, 72.3% hard (early engine: 76.6 / 72.3 /
  59.6%).
- With the floor, a critical almost never reaches routine (0.7-1.4%), at the same cost as
  before: 29-45% of routine requests rise to soon.
- **Caveat:** the protocol's first draft was written after seeing `test/claude` errors (2.4),
  and these clips are `test/claude` texts. The *absolute* voice numbers are therefore
  optimistic; the text -> voice drop is the more trustworthy figure. Section 3.2 is the clean
  check.

### 3.2 Voice on the fresh set (the number to quote)
Run 3: `test/fresh` read aloud by the same test voices, same three conditions, preset
large-v3 + Syntia + hint: 2,520 clips (420 texts x 2 voices x 3 conditions), 4 workers on
2 T4s, about 18 minutes (`data/generated/voice_eval/fresh/`). Nothing was tuned on it.

| Engine | Clean crit. | Ward crit. | Hard crit. | Hard crit. -> routine | False crit. clean / ward / hard |
|---|---:|---:|---:|---:|---:|
| TF-IDF + rules (early) | 88.2 -> 87.7% | 88.2 -> 85.3% | 88.2 -> 68.1% | 16.7% | 4.9 / 4.7 / 8.2% |
| **Hybrid `critical` + tone bank** | 96.1 -> **95.1%** | 96.1 -> **94.6%** | 96.1 -> **80.4%** | 11.8% | 6.0 / 5.8 / 10.1% |
| ... + checks + floor | 96.1 -> 95.1% | 96.1 -> 94.6% | 96.1 -> 79.9% | **0.0%** | 6.0 / 5.8 / 9.6% |

WER: clean 16.2%, ward 21.3%, hard 46.4%. Reproduce: `python -m voice.report
data/generated/voice_eval/fresh/transcripts.jsonl --with-tone-bank --model
hybrid-critical:tfidf [--checks --floor listen]`.

- **The hybrid's lead holds by voice and is not an artefact of the biased set:** 7-12
  points over the early engine in every condition.
- Speech to text costs little in clean and ward speech (1-1.5 points) and a lot in hard
  audio (about 16 points). The fresh texts are easier to hear than `test/claude` (clean WER
  16% vs 20%), so this drop is smaller than in 3.1.
- **Tone matters for speech recognition too.** Calm texts are heard almost perfectly (clean
  WER 4.5%); urgent ones are not (clean 31.9%, hard 68.0%), since shouted capitals, "!!" and
  broken spelling become garbled speech. Hard-condition critical recall: calm 86.8%, urgent
  76.5%. So the requests most likely to be real emergencies are the hardest to hear.
- With the floor, **no** critical reached routine in any condition; 20-42% of routine requests
  rose to soon.
- **The worst error is a dropped negation**, and no check catches it:
  "babam nefes alamıyor" was heard as "babam nefes alıyor" ("is breathing") with high
  confidence, so it was ranked soon and not flagged for listening. A typed request cannot fail
  this way. This is the main argument for attaching the recording to every voice card.

---

## 4. Negative results (kept on purpose)

- **Frozen embeddings did not beat TF-IDF** (2.2), not even the Turkish-specific model.
- **Hybrid `protocol` mode** (the protocol decides whenever it can): dev critical recall falls
  to 82.0%. The protocol is a good *net*, not a good *judge*.
- **Hybrid `both` mode** was rejected (2.1).
- **Training on transcripts** barely helped. It cut hard-condition critical -> routine from
  18.6% to 14.3% and did little else.
- **STT confidence alone** flags few of the confident mishearings. Adding triage uncertainty
  catches more, but then 35-60% of clips get a "listen" flag, which is too often to be useful.
- **The tone bank** raised false criticals on `test/claude` (5.7% -> 8.5%) while it fixed
  the tone shortcut.

---

## 5. The demo (`backend/`, `frontend/`)

A replayed ward shift (`uv run uvicorn backend.app:app --port 8000`, then `/board` and
`/patient`). The engine is hybrid `critical` + TF-IDF trained on `train/gemini` +
`train/gemini_tone`. The board orders by urgency, then waiting time. Each card shows its
reasons in Turkish ("kural: ...", "model %91", "protokol: kusma + koyu kusmuk"). A critical
based only on a weak model guess is shown as "Olası kritik" rather than "Kritik", and only
confident criticals beep. When completing a critical, the nurse says whether it really was
critical. That answer goes to an append-only event log, which is how false alarms would be
measured in real use.

---

## 6. Caveats (must be in the report)

1. **Home ground.** The grid, the label definitions and all data are ours. The competitors
   were not built for our grid. Llama got the grid definitions in its prompt; Dousery was used as
   published, with its own levels mapped onto ours. A fairer comparison needs a dataset
   neither side wrote.
2. **Synthetic text only.** No real patient wrote any of it. Fresh-set texts are by a
   different model from the training texts, but language models share habits.
3. **No clinical validation.** No nurse has checked the labels, the grid or the protocol yet.
   The plan is to ask a nursing student to label about 100 texts as a gold standard.
4. **Voice is TTS, not people.** Real patients (weak, elderly, accented, in pain) will be
   harder to hear than these voices. Quote the fresh-set voice numbers (3.2), not 3.1.
5. **Small subsets.** Each fresh-set tone has 34 criticals, so one text is about 3 points.
6. **The protocol was partly written after seeing test errors**; that is why the fresh set
   exists (2.4). Its `test/claude` numbers should not be quoted as unbiased.

---

## 7. Related work

- **Enlitic** (radiology worklist triage): sorts imaging studies so the urgent ones are read
  first, and never gives the diagnosis. It is a precedent for the "sort, never decide" design.
- **PMR-Bench** (Gatto et al.): a public English benchmark for triage of asynchronous patient
  portal messages, framed as pairwise ranking ("which of the two is more urgent"), with 1,569
  unique messages. It is the closest in task, but English, outpatient and asynchronous.
- **Few-shot LLM triage of portal messages** (arXiv 2605.15680, 2026): TF-IDF and BioBERT
  baselines vs six prompted LLMs at 0, 4 and 12 shots. GPT-4o-mini with 12 shots had the best
  urgent-or-higher recall (0.984). It shows what prompting can reach with examples; our
  Llama comparison is zero-shot.
- **TR-MTEB**: the Turkish embedding benchmark we used to pick embedding models.
- **Tzeng 2010**: the ranking of ward call-light request types used to weight topics.
- No public **Turkish** patient-request urgency dataset was found, which is why we built our own
  (professor approved: own synthetic data is fine, but beating a published dataset's results
  would otherwise be required).

---

## 8. Next

- Nurse-student labels for about 100 texts (gold standard; also checks the grid).
- Fine-tuned BERTurk, and an ensemble of TF-IDF and embeddings (they miss different texts).
