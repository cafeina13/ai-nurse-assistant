"""Throws out transcripts the audio cannot have contained.

Adapted from the user's Syntia bot (assistant/stt.py check_transcript), plus phrases Whisper
is known to invent from noise. A rejected transcript means "did not understand": the request
becomes Unclear/Soon with the recording flagged, never a guess.
"""

import re

from triage.normalize import fold
from voice.stt import HINT_PHRASES

MAX_COMPRESSION_RATIO = 2.4  # Whisper's own "suspiciously repetitive" line
MAX_WORDS_PER_SECOND = 4.5  # fast Turkish speech is about 3-4 words a second

# Whisper learned these from video subtitles and outros; seen in this project's runs.
HALLUCINATIONS = [
    "izlediginiz icin tesekkur",
    "altyazi m k",
    "abone olmayi unutmayin",
    "bir sonraki videoda gorusmek uzere",
    "altyazi",
]


def reject_reason(text: str, seconds: float, compression_ratio: float | None = None) -> str | None:
    """A reason to reject the transcript, or None if it is believable for the audio."""
    words = re.findall(r"\w+", text)
    if not words:
        return None
    if compression_ratio is not None and compression_ratio > MAX_COMPRESSION_RATIO:
        return "repetitive"
    if len(words) > max(3.0, MAX_WORDS_PER_SECOND * seconds):
        return f"{len(words)} words cannot fit in {seconds:.1f}s"
    folded = fold(text)
    for phrase in HALLUCINATIONS:
        if phrase in folded:
            return f"known hallucination: {phrase}"
    # Whole words only: "su" must not count inside "susadım".
    echoed = sum(1 for p in HINT_PHRASES[1:] if re.search(rf"\b{re.escape(fold(p))}\b", folded))
    if echoed >= 3:
        return "recited the vocabulary hint"
    return None
