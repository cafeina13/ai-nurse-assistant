"""Speech to text with faster-whisper, keeping the confidence values the nurse board needs."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Transcript:
    text: str
    avg_logprob: float | None  # mean over segments; closer to 0 means more confident
    no_speech_prob: float | None  # max over segments


class WhisperSTT:
    def __init__(self, model: str = "large-v3", device: str = "cuda", compute_type: str = "float16"):
        from faster_whisper import WhisperModel

        self.name = f"faster-whisper-{model}-{compute_type}"
        self.model = WhisperModel(model, device=device, compute_type=compute_type)

    def __call__(self, audio_16k: np.ndarray) -> Transcript:
        segments, _info = self.model.transcribe(
            audio_16k, language="tr", beam_size=5, condition_on_previous_text=False)
        segments = list(segments)
        if not segments:
            return Transcript("", None, None)
        return Transcript(
            text=" ".join(s.text.strip() for s in segments).strip(),
            avg_logprob=float(np.mean([s.avg_logprob for s in segments])),
            no_speech_prob=float(max(s.no_speech_prob for s in segments)),
        )
