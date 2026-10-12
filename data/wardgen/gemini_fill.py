"""Fills the training text bank with Gemini-written Turkish requests.

    python -m wardgen.gemini_fill --target 40           # fill every cell to 40 texts
    python -m wardgen.gemini_fill --dry-run              # print one prompt, call nothing

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

from wardgen.config import Intent, load_config
from wardgen.textbank import TEXTBANK_DIR, append_rows, cells, load_bank, normalize

PROMPT_VERSION = 3
DEFAULT_MODEL = "gemini-3.5-flash-lite"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
REPO_ROOT = Path(__file__).resolve().parents[2]

URGENCY_GUIDE = {
    "critical": "The situation is acute and alarming right now; the words sound urgent or panicked.",
    "soon": "The person needs a nurse within minutes; uncomfortable or worried, not panicked.",
    "routine": "A calm request for comfort, information or supplies; no time pressure.",
}
SPEAKER_GUIDE = {
    "patient": "The patient writes about themselves (first person).",
    "companion": ("A family member staying with the patient writes about them, mostly as "
                  "annem, babam, eşim, oğlum, kızım, ninem, dedem, kayınvalidem. Families talk "
                  "plainly ('annem su istiyor'); never formal phrasing like 'refakatçisi olduğum "
                  "hasta' or 'su talebinde bulundu'."),
}
# Every batch mixes these, so no cell ends up in a single style.
STYLE_MIX = """- about a third: hurried phone typing, lowercase, missing Turkish characters
  (agri, cok, nefes alamiyorum), small typos, little punctuation
- about a quarter: elderly or colloquial speech (kızım, evladım, yavrum, bi, geliyo)
- some short and blunt (two or three words), some longer and explanatory
- some polite, some impatient or annoyed"""


def build_prompt(intent: Intent, urgency: str, speaker: str, n: int,
                 avoid: list[str]) -> str:
    avoid_block = "\n".join(f"- {t}" for t in avoid) or "- (none yet)"
    return f"""You write synthetic test data for a hospital nurse-call assistant in Turkey.
On an internal medicine ward, people type short messages to the nurse station.

Write {n} different messages in Turkish.

What the message is about: {intent.id.replace("_", " ")} (example: "{intent.example}")
Urgency the wording should express: {urgency}. {URGENCY_GUIDE[urgency]}
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
    parser.add_argument("--target", type=int, default=40, help="texts per cell")
    parser.add_argument("--batch", type=int, default=20, help="texts asked for per call")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--min-interval", type=float, default=7.0,
                        help="seconds between calls, to stay under a per-minute limit")
    parser.add_argument("--intents", help="comma-separated intent ids (default: all)")
    parser.add_argument("--out", type=Path, default=TEXTBANK_DIR / "train" / "gemini")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    intents = load_config().intents
    if args.intents:
        intents = {k: intents[k] for k in args.intents.split(",")}
    bank = load_bank(args.out) if args.out.exists() else {}
    todo = [c for c in cells(intents) if len(bank.get(c, [])) < args.target]
    print(f"{len(todo)} of {len(cells(intents))} cells need texts (target {args.target}).")

    if args.dry_run:
        if todo:
            intent_id, urgency, speaker = todo[0]
            print(build_prompt(intents[intent_id], urgency, speaker, args.batch,
                               bank.get(todo[0], [])[-15:]))
        return

    key = api_key()
    for cell in todo:
        intent_id, urgency, speaker = cell
        existing = list(bank.get(cell, []))
        seen = {normalize(t) for t in existing}
        attempts = 0
        while len(existing) < args.target and attempts < 6:
            attempts += 1
            n = min(args.batch, args.target - len(existing) + 5)  # a few spare for duplicates
            prompt = build_prompt(intents[intent_id], urgency, speaker, n, existing[-15:])
            texts = _call_with_backoff(prompt, args.model, key)
            if texts is None:
                print("\nStopping: the API keeps refusing (daily quota?). Run again later to resume.")
                return
            fresh = []
            for t in texts:
                if normalize(t) not in seen and len(existing) + len(fresh) < args.target:
                    seen.add(normalize(t))
                    fresh.append(t)
            append_rows(args.out, intent_id, [
                {"text": t, "intent": intent_id, "urgency": urgency, "speaker": speaker,
                 "source": args.model, "prompt_version": PROMPT_VERSION}
                for t in fresh
            ])
            existing += fresh
            print(f"{intent_id:<22} {urgency:<8} {speaker:<9} +{len(fresh):>2} -> {len(existing)}")
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
