"""Turns a scripted call (topic, speaker) into the words the system receives.

ExampleText returns the topic's example phrase; BankText (textbank.py) picks varied
wordings from a text bank.
"""

from typing import Protocol

from backend.models import Patient
from wardgen.config import Topic

BUTTON_LABELS = {"water": "Su", "toilet_help": "Tuvalet", "wants_painkiller": "Ağrı"}


class TextGenerator(Protocol):
    def text_for(self, topic: Topic, speaker: str, patient: Patient) -> str: ...


class ExampleText:
    """Placeholder: one fixed phrase per topic."""

    def text_for(self, topic: Topic, speaker: str, patient: Patient) -> str:
        return topic.example


def button_text(topic_id: str) -> str:
    return f"[{BUTTON_LABELS[topic_id]}]"
