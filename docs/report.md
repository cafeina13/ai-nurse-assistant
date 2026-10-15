# Sorting, Not Deciding: A Turkish Triage Engine for Ward Patient Requests

*Course project, AI in Healthcare. [Author name, student number, date]*

## Abstract

A ward call button tells the nurse that someone needs them, but not why. We built an engine
that reads a short Turkish request, typed or spoken at the bedside, and sorts it into one of 28
cells (category x urgency) so the nurse board can put emergencies first. The engine never
diagnoses, never asks the patient follow-up questions, and never decides care: it only orders
and flags. It combines a TF-IDF classifier, 51 hand-written safety rules and a readable
47-line protocol. Because no public Turkish dataset exists for this task, we generated our own
synthetic data (approved by the course instructor) and kept a final test set written blind to
our design. On that set (420 texts), the engine catches 96.1% of critical requests with 3.8%
false criticals. Two existing engines we could run on the same texts did worse: Llama 3.1 8B
(zero-shot) caught 61.8% of criticals, and Dousery, a published Turkish medical triage model,
called about 80% of all requests urgent. When the requests were spoken by synthetic voices
and transcribed with Whisper, critical recall stayed at 95.1% in clean audio and 94.6% with
ward noise, and fell to 80.4% in hard audio. All labels, data and the grid are ours, and no
nurse has validated them yet, so these are results on home ground.

---

## 1. Introduction

On a hospital ward, a patient who needs something presses a call button. The light comes on
at the nurse station, and a nurse walks to the room to find out what is wrong. The call
carries no content: a request for water and a patient who cannot breathe look the same until
someone arrives. When several calls come at once, the order in which they are answered is
decided by whoever is free and whichever light they see first.

If the request carried a few words ("nefes alamıyorum", "su alabilir miyim"), the calls could
be sorted before anyone walks. That is the task of this project: a **triage engine** that reads
a short Turkish request and places it in a grid of categories and urgency levels, so a nurse
board can show emergencies first and give the nurse a reason for each position.

Two design principles shaped everything that follows:

1. **Sort, never decide.** The engine orders and flags. It does not diagnose, does not give
   medical advice to patients, and does not replace the nurse's judgement. A nurse sees every
   request and can reorder anything.
2. **Never question the patient.** Asking follow-up questions is the nurse's job, and an
   automated dialogue slows a patient down at the worst moment. The engine works with what it
   is given.

The main metric is **critical recall**: the share of truly critical requests the engine puts
at the top. A missed critical is the worst error; a false critical costs a walk. False
criticals still matter, because a board where everything is red sorts nothing.

The contributions are:

- A grid of 12 request categories and 28 category-urgency cells for a Turkish internal medicine
  ward, with labelling rules for hard cases (Section 3.1).
- Synthetic Turkish training and test data, including a *tone bank* that breaks the shortcut
  "panicky wording means critical" and a *fresh* test set written blind to our design
  (Section 3.2).
- A hybrid engine: classifier + safety rules + an ordered, nurse-readable protocol, where
  every decision names its reason (Section 3.3).
- A head-to-head comparison with two existing engines on the same texts (Section 4.3).
- A voice evaluation using synthetic speech, ward noise and Whisper, which measures what speech
  recognition costs the triage (Section 4.4).
- A working demo: a replayed ward shift with a nurse board and a patient tablet page
  (Section 5).

---

## 2. Related work

**Triage of patient messages.** The closest task in the literature is triage of asynchronous
patient portal messages in outpatient care. PMR-Bench (Gatto et al.) is a public English
benchmark of 1,569 unique portal messages, framed as pairwise ranking: which of two messages is
more urgent. A 2026 preprint (arXiv 2605.15680) compared TF-IDF and BioBERT baselines with six
prompted LLMs at 0, 4 and 12 shots. GPT-4o-mini with 12 shots reached the best
urgent-or-higher recall (0.984). Both works are in English, about outpatient messages that can
wait hours, and not about a ward where a request may need an answer within minutes.

**Sorting without deciding.** Commercial radiology triage (for example Enlitic) reorders the
radiologist's worklist so likely-urgent studies are read first. It does not report the finding
itself. Our engine follows the same pattern for ward requests: it changes the order, not the
decision.

**Ward requests.** Tzeng (2010) ranked the reasons patients use call lights, with bathroom
help, pain medication, IV and pump problems, personal help and repositioning at the top. We
used that ranking to weight topics in our synthetic data.

**Turkish language resources.** TR-MTEB is a Turkish text-embedding benchmark. We used it to
choose the embedding models we compared with TF-IDF. We found **no public Turkish dataset of
patient requests with urgency labels**. Dousery (`dousery/turkish-medical-triage-llama3-gguf`)
is a published Turkish medical triage model, and it is one of our two competitors.

Because no public dataset fits the task, we built our own. The course instructor's rule was
that using another study's public dataset means beating its published results, while using our
own synthetic data is acceptable.

---

## 3. Methods

### 3.1 The request grid

A request is labelled with one *cell*: a category and an urgency level (critical, soon or
routine). There are 12 categories: breathing, pain, neuro (consciousness), other symptom,
fall or injury, physical help, comfort and supplies, medication, equipment, information,
emotional, and unclear. Not every category has every urgency (a request for supplies is never
critical), which gives 28 cells. Each cell has a one-line meaning (for example
*breathing/critical: cannot breathe, or breathlessness getting rapidly worse*) and example
topics.

Hard-case rules make the labels consistent. Examples: label the current state, not the topic
(pain that has passed is routine); vague distress with no stated need is at least *soon*; with
several needs in one message, the most urgent decides; a relative speaking for the patient is
labelled like the patient; a routine toilet request from a patient who cannot walk alone becomes
*soon*.

The grid was written by the author and **has not been validated by a nurse**.

### 3.2 Data

All texts are synthetic, generated by large language models from a slot specification: a cell,
a topic, a speaker (patient or companion), a register, a spelling style (with and without
Turkish letters, typos, chat spelling such as "alamiyom") and a patient context.

| Set | Texts | Written by | Use |
|---|---:|---|---|
| Training | 5,600 | Gemini | training; 20% of every slot held out as a development set |
| Tone bank | 956 | Gemini: 320 calm criticals, 636 panicky non-criticals | training only |
| Claude test | 560 | Claude (a different model family) | test |
| Tone test | 54 | Claude: 28 calm criticals, 26 panicky non-criticals | tests the tone shortcut |
| **Fresh test** | 420 | a separate Claude instance that saw only the slot specification | **final test** |

Two of these sets exist because of problems found along the way.

**The tone bank.** The first model learned a shortcut: panicky wording means critical. A calm
"annemin dudakları morardı" (my mother's lips have turned blue) was easy to miss, and an excited
"ACİL battaniye lazım!" (URGENT, I need a blanket!) was flagged critical. The tone bank adds
calm emergencies and panicky trivia to the training data, so tone and urgency are no longer
correlated.

**The fresh test set.** The protocol (Section 3.3) was first drafted after we had seen which
Claude-test criticals every model missed. Its score on that set is therefore optimistic. To get
an honest number, a separate model instance, which never saw our rules, protocol, errors or test
texts, wrote one calm, one neutral and one urgent text for each slot. Nothing was tuned on this
set; it was run once with the final engine. **All headline numbers in this report come from it.**

### 3.3 The engine

The engine runs in layers. Text is first folded (lower case, Turkish letters to ASCII), so
"nefes alamıyorum" and "nefes alamiyom" meet.

1. **Buttons.** If the patient pressed a category button (for example "Ağrı", pain), it gives
   the category.
2. **Classifier.** A TF-IDF model (character 2-5-grams and word 1-2-grams, logistic regression)
   predicts the cell with a confidence.
3. **Protocol.** A readable table in a YAML file: 59 *facts* (word patterns such as *vomit*,
   *dark vomit*, *breath*) and 47 ordered *lines* such as
   `vomit + (blood or dark vomit) -> other_symptom/critical`. The first line that holds decides.
   In the chosen **hybrid "critical" mode**, the classifier decides the cell, but a protocol
   critical line overrides a non-critical classifier answer. The protocol is a safety net
   written in a form a nurse can read and correct.
4. **Safety rules.** 51 red-flag phrases ("nefes alamıyorum", "bayıldı", "kan kusuyor") force
   critical. Rules only raise urgency, never lower it, and always win. Exceptions stop false
   hits: the fall rule "annem ... düştü" (my mother fell) does not fire on "annemin ateşi
   düştü" (my mother's fever went down).
5. **Patient context.** For example, the toilet rule above for patients who cannot walk alone.
6. **Flags.** *Uncertain* (low confidence: the card says "Emin değil — kontrol edin", check
   this one) and *interrupt* (only a confident critical makes a sound). The principle is
   *order generously, interrupt carefully*: a doubtful critical goes to the top of the board but
   does not ring.

Every decision carries its reasons, shown on the board in Turkish: "kural: nefes alamıyorum"
(rule), "model %91", "protokol: kusma + koyu kusmuk" (protocol: vomiting + dark vomit).

**Model selection.** We also tried frozen sentence embeddings (multilingual-e5-base and large,
bge-m3, and turkish-embedding-model chosen from TR-MTEB) with logistic regression, the protocol
alone, and four ways of combining classifier and protocol. All choices were made on the
development set only.

### 3.4 Competitors

Both competitors ran on a Kaggle GPU (2x T4) on exactly the same texts.

- **Dousery**, a Turkish medical triage model (Llama 3 based, 4-bit GGUF). It received the raw
  request with the system prompt from its model card, as published. It answers with a level
  and advice ("ACİL: ..."). Since its levels do not match ours, we mapped them two ways: ACİL as
  critical, and ACİL as soon.
- **Llama 3.1 8B Instruct**, zero-shot. Its system prompt contained what a human labeller would
  get: every cell's meaning, example topics and the hard-case rules. It answered in JSON with a
  category and an urgency.

All engines were scored on urgency (critical, soon, routine), the one scale all three produce.

### 3.5 Voice

We had no real recordings, so the test texts were spoken by text-to-speech voices (Drifting
TTS speakers, Piper dfki, a Piper voice trained by the author, and MMS) and degraded three
ways: *clean*, *ward* (ward noise and background talk) and *hard* (noise, reverberation and
low bandwidth). Each text was read by two voices, in all three conditions. faster-whisper
(large-v3, with decoding settings from earlier Turkish work and a short ward-vocabulary hint)
transcribed them on Kaggle. Triage then ran on both the original text and the transcript, so the
difference is what speech recognition costs.

A *never routine when in doubt* floor was also tested: if the transcript is doubtful (low
speech-recognition confidence, an empty or implausible transcript, or an uncertain triage), the
request is never ranked below *soon*.

### 3.6 Metrics

Critical recall (critical requests ranked critical) comes first. We also report false
criticals (non-critical requests ranked critical), under-triage (any level too low), critical
to routine (the most dangerous error), urgency accuracy, cell accuracy, and for voice the word
error rate (WER).

---

## 4. Results

### 4.1 Building the engine (development set)

| Step | Critical recall | False criticals | Cell accuracy |
|---|---:|---:|---:|
| Safety rules alone | 55.5% | | |
| TF-IDF + rules | 94.5% | 3.8% | 88.1% |
| TF-IDF + rules + tone bank | 94.1% | 5.5% | 86.7% |
| Protocol alone | 73.9% | 5.5% | 68.5% |
| Hybrid, "raise" (higher urgency wins) | 96.0% | 7.4% | 82.3% |
| **Hybrid, "critical"** (chosen) | **96.0%** | 7.4% | **85.6%** |
| Hybrid, "protocol" (protocol decides when it can) | 82.0% | 5.7% | 74.4% |
| Hybrid, "both" | 92.7% | 6.6% | 85.2% |

Rules alone catch only about half the criticals: people describe emergencies in more ways than
a phrase list can hold. The protocol alone is weaker than the classifier, and letting it decide
everything hurts. Used as a net for criticals only, it adds about two points of critical recall
on the development set. "Raise" ties on recall but changes more cells, so "critical" was chosen.

Frozen embeddings did not beat TF-IDF. The best, turkish-embedding-model, tied it on critical
recall (94.8% development, 86.8% Claude test). The models missed *different* texts, though:
of about 40 missed criticals, only 6 were shared, which suggests an ensemble as future work.

### 4.2 The tone shortcut

| Tone test | Without tone bank | With tone bank | Final hybrid |
|---|---:|---:|---:|
| Calm criticals caught (n = 28) | 82.1% | 89.3% | **100%** |
| Panicky non-criticals flagged critical (n = 26) | 26.9% | **0%** | 0% |

Without the tone bank, the classifier alone caught only 64.3% of calm criticals; rules rescued
some of them. The tone bank removed the shortcut, at a price: false criticals on the Claude test
rose from 5.7% to 8.5%.

### 4.3 Final test and head-to-head comparison

On the fresh set, the hybrid raised critical recall from 89.2% (TF-IDF + rules) to **96.1%**,
for about one point more false criticals, with no false interrupts. The gain survived on texts
nobody had seen, so the protocol adds real knowledge rather than patches for known errors.

All three engines on the fresh set (420 texts, urgency scale):

| Engine | Critical recall | False criticals | Under-triage | Urgency accuracy | Cell accuracy |
|---|---:|---:|---:|---:|---:|
| **Ours** (hybrid + rules + tone bank) | **96.1%** | 3.8% | **5.0%** | **90.5%** | **87.1%** |
| Llama 3.1 8B, zero-shot | 61.8% | 3.1% | 10.2% | 73.1% | 58.3% |
| Dousery, ACİL = critical | 100% | 74.8% | 1.2% | 42.1% | |
| Dousery, ACİL = soon | 2.9% | 0% | 24.8% | 50.7% | |

(Our engine here is trained on the whole training set, without the development hold-out. With
the hold-out, false criticals are 4.7%.)

**Llama 3.1** is cautious but misses many emergencies, and most often the calm ones: critical
recall 52.9% on calm texts, 58.8% neutral, 73.5% urgent, against ours at 94.1%, 94.1% and 100%.
This is the tone shortcut again, in a model we did not train.

**Dousery** does not sort. It called 337 of 420 requests ACİL, 78 NORMAL and 3 ÇOK ACİL. Mapped
one way, it marks three in four non-critical requests as critical; mapped the other way, it
misses almost every emergency. It also gives diagnoses and tells patients to call 112: on an
earlier 614-text run, it suggested a diagnosis in 298 answers and said to call 112 in 393. On a
ward, where the nurse is a few metres away, that advice is wrong. And for a patient-facing
device, a diagnosis breaks the first design principle.

### 4.4 Voice

The fresh set read aloud (2,520 clips), critical recall from text to speech:

| Engine | Clean | Ward noise | Hard |
|---|---:|---:|---:|
| TF-IDF + rules (early engine) | 88.2 -> 87.7% | 88.2 -> 85.3% | 88.2 -> 68.1% |
| **Final hybrid** | 96.1 -> **95.1%** | 96.1 -> **94.6%** | 96.1 -> **80.4%** |

WER was 16.2% in clean audio, 21.3% with ward noise and 46.4% in hard audio. Speech recognition
costs little in clean and ward audio (1 to 1.5 points) and about 16 points in hard audio. False
criticals rose with noise (6.0%, 5.8%, 10.1%), because garbled transcripts trip protocol facts
more often. The hybrid's lead over the early engine holds in every condition.

With the "never routine when in doubt" floor, **no critical request was ranked routine in any
condition**. The cost is that 20 to 42% of routine requests were raised to *soon*. Whether that
trade is worth it is a question for nurses, not for us.

Three observations about speech recognition:

- **Hallucinations.** On noise, Whisper sometimes writes video-subtitle phrases ("İzlediğiniz
  için teşekkür ederim", thank you for watching). A filter rejects these, and a rejected
  transcript becomes *unclear/soon*, like the old call button: someone called, we could not
  tell why.
- **Choice of decoding.** On an earlier test, the ward-vocabulary hint raised hard-audio
  critical recall from 56.5% to 63.9% with no extra false criticals. The faster *turbo* model was
  slightly worse.
- **Dropped negations are the most dangerous error.** "babam nefes alamıyor" (my father
  *cannot* breathe) was heard as "babam nefes alıyor" (my father *is* breathing), with high
  confidence, so it was ranked *soon* and not flagged. A typed request cannot fail this way.
  This is the strongest argument for attaching the recording to every voice card.

### 4.5 Negative results

- Frozen embeddings, including a Turkish-specific model, did not beat TF-IDF.
- Letting the protocol decide whenever it could reduced critical recall to 82.0%. The protocol
  is a good net but a poor judge.
- Allowing a weak classifier critical to be lowered by the protocol ("both" mode) cost 3.3
  points of critical recall for under one point fewer false alarms.
- Training on Whisper transcripts barely helped.
- Speech-recognition confidence alone missed most confident mishearings. Combined with triage
  uncertainty, it flagged 35 to 60% of clips, too many for a "listen to this" flag to stay
  useful.
- We first concluded that urgent requests are the hardest to hear. That was a confound: the
  rule that assigned voices gave every urgent fresh text the weakest voice. Within a single
  voice family, urgent texts were not harder to hear than neutral ones. The claim was withdrawn
  and the assignment fixed.

---

## 5. Demo

The engine runs inside a small web application (FastAPI with plain HTML and JavaScript). A
replayed ward shift sends scripted requests at a chosen speed, and the replay can be paused and
resumed. The **nurse board** orders cards by urgency, then waiting time. Each card shows room,
bed, patient, the request, a Turkish reason line and badges: *Kritik*, *Olası kritik* (a
critical based only on a weak classifier guess, which keeps its place at the top but does not
claim certainty), *Emin değil* and *Refakatçi* (companion). Only confident criticals beep. A
nurse can claim a card and complete it; when completing a critical, the nurse answers whether it
really was critical. The **patient page** is a bedside tablet with category buttons and a text
box.

Every event (request created, triaged, claimed, completed) is written to an append-only log
that the database protects from edits. The board is only a view of that log. In real use, the
nurses' "really critical?" answers in the log would measure the false-alarm rate on real
requests, which is the measurement this project cannot make with synthetic data.

---

## 6. Discussion

**What worked.** A plain classifier with a readable safety layer beat both a general 8B LLM and
a published medical model on this task. The likely reason is that the task is narrow and
well-defined: 28 cells with explicit rules, where a model trained on thousands of examples of
exactly this task has an advantage over one prompted to do it. The protocol mattered less for
its numbers (about seven points of critical recall on the fresh set) than for its form: a nurse
can read "vomit + dark vomit -> critical", disagree, and change it, which no learned model allows.

**What tone taught us.** Both our first model and Llama 3.1 treated calm wording as a sign of
low urgency. Real patients who are seriously ill are often quiet, so this is a dangerous bias,
and it is invisible unless you test for it. The tone bank fixed it in our model; the zero-shot
LLM kept it.

**Contamination and how we handled it.** The protocol was first written after seeing test
errors, which made its Claude-test score optimistic. Instead of reporting that score, we wrote a
new test set blind to the design and reported only that. The gain held, smaller than on the
contaminated set but real. We think this step is the most important methodological choice in
the project.

**Why not just use an LLM?** A larger LLM with few-shot examples might do much better than our
zero-shot baseline (the 2026 preprint reached 0.984 recall with GPT-4o-mini and 12 examples).
But a ward device that sends patient words to a cloud API raises privacy and availability
questions, and Dousery shows a second risk: a medical LLM's habit of advising and diagnosing is
hard to switch off. Our engine runs on an ordinary laptop CPU, with no GPU or network, and cannot say anything it
was not written to say.

**Voice.** Speech recognition works in clean and noisy-ward audio and fails in hard audio. The
floor removes the worst outcome (a critical ranked routine) at the cost of many routine requests
raised to soon. The negation error shows that voice needs a human safeguard: the recording.

---

## 7. Limitations

1. **Home ground.** The grid, the labels and all data are ours. The competitors were not built
   for our grid. Llama got the grid definitions in its prompt; Dousery was used as published,
   with its levels mapped onto ours. A fair comparison needs data neither side wrote.
2. **Synthetic text only.** No patient wrote any of these texts. The fresh set comes from a
   different model instance than the training data, but language models share habits, and real
   requests will be messier.
3. **No clinical validation.** No nurse has checked the grid, the labels or the protocol. A
   nursing student labelling about 100 texts would be the first step.
4. **Synthetic voices.** Real patients (elderly, weak, accented, in pain) will be harder to
   hear than these voices.
5. **Small subsets.** Each tone group in the fresh set has 34 critical texts, so one text moves
   a percentage by about three points.
6. **No real deployment.** We did not measure what matters most: whether sorting requests
   changes response times or outcomes on a real ward.

---

## 8. Conclusion and future work

A small, explainable engine can sort Turkish ward requests well enough to put 96% of
emergencies at the top of a nurse's board, on data written blind to its design, and it does so
more safely than the two existing engines we could test. The results hold on home ground only.

Next steps, in order of value: nurse validation of the grid and about 100 labels; a voice check
that flags transcripts containing the positive form of a critical phrase ("nefes alıyor") for
the nurse to listen to; a fine-tuned Turkish BERT model and an ensemble of TF-IDF and embeddings,
which miss different texts; and, eventually, real requests from a real ward.

---

## Use of AI tools

All training and test texts were generated by large language models (Gemini and Claude), as
described in Section 3.2. AI coding assistance was used to build the software. [Adjust to the
course's disclosure rules.]

## References

[Complete these with full author lists, titles and venues before submission.]

- Gatto et al. PMR-Bench: a benchmark for triage of patient portal messages.
- arXiv 2605.15680 (2026). Few-shot LLM triage of patient portal messages.
- TR-MTEB: a Turkish text embedding benchmark.
- Tzeng, H.-M. (2010). Study of patients' reasons for using call lights.
- Enlitic. Radiology worklist triage (product).
- Dousery. turkish-medical-triage-llama3-gguf. Hugging Face model card.
- Meta. Llama 3.1 8B Instruct. Model card.
- Radford et al. (2023). Robust speech recognition via large-scale weak supervision (Whisper).

## Appendix: reproducing the results

Every number in this report is in `docs/results.md`, with the experiment record or the command
that produced it. Code: `triage/` (engine and evaluation), `compare/` (competitors),
`voice/` (speech evaluation), `backend/` and `frontend/` (demo).
