"""Turns a scripted call (intent, urgency, speaker) into the words the system receives.

The plan calls for LLM-generated Turkish with deliberate variety (elderly speech, typos,
family members speaking, vague requests). Until that is wired in, ExampleText returns the
intent's example phrase, which is enough to generate and replay a ward.
"""

from typing import Protocol

from backend.models import Patient
from wardgen.config import Intent

BUTTON_LABELS = {"water": "Su", "toilet_help": "Tuvalet", "wants_painkiller": "Ağrı"}


class TextGenerator(Protocol):
    def text_for(self, intent: Intent, urgency: str, speaker: str, patient: Patient) -> str: ...


class ExampleText:
    """Placeholder: one fixed phrase per intent."""

    def text_for(self, intent: Intent, urgency: str, speaker: str, patient: Patient) -> str:
        return intent.example


def button_text(intent_id: str) -> str:
    return f"[{BUTTON_LABELS[intent_id]}]"
