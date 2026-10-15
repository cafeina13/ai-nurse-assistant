"""Fills the training text bank with Gemini-written Turkish requests.

    python -m wardgen.gemini_fill --target 40           # fill every slot to 40 texts
    python -m wardgen.gemini_fill --dry-run              # print one prompt, call nothing
    python -m wardgen.gemini_fill --tone calm --target 10     # calm criticals (train/gemini_tone)
    python -m wardgen.gemini_fill --tone panicky --target 6   # panicky non-criticals

Resumable: every batch is saved as it arrives, so after a rate limit or a stopped run,
running it again continues where it left off. Reads GEMINI_API_KEY from the environment
or from a .env file in the repo root. Synthetic data only; Google's free tier may use
prompts to improve its products.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from wardgen.config import Config, Topic, load_config
from wardgen.textbank import TEXTBANK_DIR, append_rows, load_bank, normalize, slots

PROMPT_VERSION = 4
DEFAULT_MODEL = "gemini-3.5-flash-lite"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
REPO_ROOT = Path(__file__).resolve().parents[2]

URGENCY_GUIDE = {
    "critical": "The situation is acute and alarming right now; the words sound urgent or panicked.",
    "soon": "The person needs a nurse within minutes; uncomfortable or worried, not panicked.",
    "routine": "Calm, no time pressure: a request, a question, or a mild or passing symptom.",
}
SPEAKER_GUIDE = {
    "patient": "The patient writes about themselves (first person).",
    "companion": ("A family member staying with the patient writes about them, mostly as "
                  "annem, babam, eşim, oğlum, kızım, ninem, dedem, kayınvalidem. Families talk "
                  "plainly ('annem su istiyor'); never formal phrasing like 'refakatçisi olduğum "
                  "hasta' or 'su talebinde bulundu'."),
}
# Tone sets break the shortcut "panic words mean critical" that the default prompt teaches
# (critical messages there are asked to sound panicked). They replace "How it should sound".
TONE_GUIDE = {
    "calm": ("Calm, matter-of-fact, even polite or apologetic, as people who do not want to "
             "bother the staff often write. No urgency words (acil, çabuk, hemen, yetişin, koşun), "
             "no exclamation marks, no capitals. The situation itself is still exactly as serious "
             "as described: understate the tone, never the facts."),
    "panicky": ("Hurried and alarmed: urgency words (acil, çabuk, hemen, yetişin), exclamation "
                "marks or capitals, impatience. The need itself stays ordinary and not dangerous: "
                "exaggerate the tone, never the facts."),
}

# Every batch mixes these, so no cell ends up in a single style.
STYLE_MIX = """- about a third: hurried phone typing, lowercase, missing Turkish characters
  (agri, cok, nefes alamiyorum), small typos, little punctuation
- about a quarter: elderly or colloquial speech (kızım, evladım, yavrum, bi, geliyo)
- some short and blunt (two or three words), some longer and explanatory
- some polite, some impatient or annoyed"""


def build_prompt(config: Config, topic: Topic, speaker: str, n: int, avoid: list[str],
                 tone: str | None = None) -> str:
    cell = topic.cell
    sound = TONE_GUIDE[tone] if tone else URGENCY_GUIDE[cell.urgency]
    category = config.categories[cell.category]["name"]
    avoid_block = "\n".join(f"- {t}" for t in avoid) or "- (none yet)"
    return f"""You write synthetic test data for a hospital nurse-call assistant in Turkey.
On an internal medicine ward, people type short messages to the nurse station.

Write {n} different messages in Turkish.

Kind of request: {category}, urgency {cell.urgency}: {cell.means.lower()}.
What the message is about: {topic.id.replace("_", " ")} (example: "{topic.example}")
How it should sound: {sound}
Who writes: {SPEAKER_GUIDE[speaker]}

Rules:
- Natural, everyday Turkish as real patients and families type it; 2 to 25 words each.
- Keep to this one need. Do not add other symptoms or requests.
- Every message must make sense; no odd metaphors.
- Vary wording, sentence shape and openings; do not start every message the same way.
- Mix styles across the messages:
{STYLE_MIX}
- Only the request itself: no greetings to the reader, no emoji, no medical advice,
  no names of real people or hospitals.
- Do not repeat or lightly rephrase these existing messages:
{avoid_block}

Return a JSON array of {n} strings."""


def load_tone_bank(bank_dir: Path, tone: str) -> dict:
    bank: dict = {}
    for path in sorted(bank_dir.glob("*.jsonl")) if bank_dir.exists() else []:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("tone") == tone:
                key = (row["category"], row["urgency"], row["topic"], row["speaker"])
                bank.setdefault(key, []).append(row["text"])
    return bank


def call_gemini(prompt: str, model: str, api_key: str) -> list[str]:
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 1.0,
            "response_mime_type": "application/json",
            "response_schema": {"type": "ARRAY", "items": {"type": "STRING"}},
        },
    }
    request = urllib.request.Request(
        API_URL.format(model=model),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    text = payload["candidates"][0]["content"]["parts"][0]["text"]
    return [t.strip() for t in json.loads(text) if isinstance(t, str) and t.strip()]


def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    env_file = REPO_ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "GEMINI_API_KEY":
                key = value.strip().strip('"').strip("'")
    if not key:
        sys.exit("Set GEMINI_API_KEY in the environment or in a .env file (never commit it).")
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill the training text bank with Gemini.")
    parser.add_argument("--target", type=int, default=40, help="texts per slot")
    parser.add_argument("--batch", type=int, default=20, help="texts asked for per call")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--min-interval", type=float, default=7.0,
                        help="seconds between calls, to stay under a per-minute limit")
    parser.add_argument("--topics", help="comma-separated topic ids (default: all)")
    parser.add_argument("--out", type=Path, help="default: train/gemini, or train/gemini_tone with --tone")
    parser.add_argument("--tone", choices=["calm", "panicky"],
                        help="calm: only critical topics, worded calmly; panicky: only non-critical, worded urgently")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config()
    topics = {t.key: t for t in config.topics}
    wanted = set(args.topics.split(",")) if args.topics else None
    all_slots = [s for s in slots(config) if wanted is None or s[2] in wanted]
    if args.tone == "calm":
        # Vague-but-alarmed is defined by its alarm, so it has no calm form.
        all_slots = [s for s in all_slots if s[1] == "critical" and s[0] != "unclear"]
    elif args.tone == "panicky":
        all_slots = [s for s in all_slots if s[1] != "critical"]
    if args.out is None:
        args.out = TEXTBANK_DIR / "train" / ("gemini_tone" if args.tone else "gemini")
    # Tone texts share a folder, so a slot counts only texts of the requested tone.
    bank_tone = load_tone_bank(args.out, args.tone) if args.tone else None
    bank = bank_tone if args.tone else (load_bank(args.out) if args.out.exists() else {})
    todo = [s for s in all_slots if len(bank.get(s, [])) < args.target]
    print(f"{len(todo)} of {len(all_slots)} slots need texts (target {args.target}).")

    if args.dry_run:
        if todo:
            *key, speaker = todo[0]
            print(build_prompt(config, topics[tuple(key)], speaker, args.batch,
                               bank.get(todo[0], [])[-15:], args.tone))
        return

    key = api_key()
    for slot in todo:
        category, urgency, topic_id, speaker = slot
        topic = topics[(category, urgency, topic_id)]
        existing = list(bank.get(slot, []))
        seen = {normalize(t) for t in existing}
        attempts = 0
        while len(existing) < args.target and attempts < 6:
            attempts += 1
            n = min(args.batch, args.target - len(existing) + 5)  # a few spare for duplicates
            prompt = build_prompt(config, topic, speaker, n, existing[-15:], args.tone)
            texts = _call_with_backoff(prompt, args.model, key)
            if texts is None:
                print("\nStopping: the API keeps refusing (daily quota?). Run again later to resume.")
                return
            fresh = []
            for t in texts:
                if normalize(t) not in seen and len(existing) + len(fresh) < args.target:
                    seen.add(normalize(t))
                    fresh.append(t)
            append_rows(args.out, [
                {"text": t, "category": category, "urgency": urgency, "topic": topic_id,
                 "speaker": speaker, "source": args.model, "prompt_version": PROMPT_VERSION,
                 **({"tone": args.tone} if args.tone else {})}
                for t in fresh
            ])
            existing += fresh
            print(f"{category}/{urgency}/{topic_id:<24} {speaker:<9} +{len(fresh):>2} -> {len(existing)}",
                  flush=True)
            time.sleep(args.min_interval)
    print("Done.")


def _call_with_backoff(prompt: str, model: str, key: str) -> list[str] | None:
    """Retries rate limits and server errors with growing waits; None if it keeps failing."""
    wait = 30.0
    for _ in range(6):
        try:
            return call_gemini(prompt, model, key)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            if e.code not in (429, 500, 503):
                sys.exit(f"Gemini API error {e.code}: {detail}")
            print(f"  {e.code}, waiting {wait:.0f}s ...")
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            print(f"  {type(e).__name__}, waiting {wait:.0f}s ...")
        time.sleep(wait)
        wait = min(wait * 2, 600)
    return None


if __name__ == "__main__":
    main()
