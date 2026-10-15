"""Speech to text with faster-whisper, keeping the confidence values the nurse board needs.

Decoding presets compare the plain default against the settings the user's Syntia bot
arrived at for Turkish voice commands (assistant/stt.py there): one greedy pass at
temperature 0, no repeated 3-token phrases, a token cap and the VAD filter, optionally
with a vocabulary hint.
"""

from dataclasses import dataclass

import numpy as np

# Ward vocabulary for the hint, written from the request grid's topics, never from test audio
# (Syntia's 44% -> 12% gain came from a hint written after hearing the clip: optimistic).
HINT_PHRASES = [
    "Hemşire hanım", "nefes alamıyorum", "göğsüm ağrıyor", "ağrı kesici", "serum bitti",
    "serum makinesi ötüyor", "sürgü", "tuvalete gitmem lazım", "düştüm", "kanıyor",
    "bayıldı", "ateşim var", "midem bulanıyor", "su", "battaniye", "ilacım", "doktor",
]
HINT = HINT_PHRASES[0] + ", " + ", ".join(HINT_PHRASES[1:]) + "."

_SYNTIA = {"beam_size": 1, "temperature": 0.0, "no_repeat_ngram_size": 3,
           "max_new_tokens": 96, "vad_filter": True}
PRESETS = {
    "large-v3": ("large-v3", {"beam_size": 5}),  # the first evaluation run
    "large-v3-syntia": ("large-v3", _SYNTIA),
    "large-v3-syntia-hint": ("large-v3", {**_SYNTIA, "initial_prompt": HINT}),
    "turbo-syntia": ("large-v3-turbo", _SYNTIA),
}


@dataclass
class Transcript:
    text: str
    avg_logprob: float | None  # mean over segments; closer to 0 means more confident
    no_speech_prob: float | None  # max over segments
    compression_ratio: float | None = None  # max over segments; high means repetitive


class WhisperSTT:
    def __init__(self, model: str = "large-v3", device: str = "cuda", compute_type: str = "float16",
                 options: dict | None = None, name: str | None = None, loaded=None):
        from faster_whisper import WhisperModel

        self.name = name or f"faster-whisper-{model}-{compute_type}"
        self.options = {"beam_size": 5} if options is None else options
        self.model = loaded or WhisperModel(model, device=device, compute_type=compute_type)

    @classmethod
    def presets(cls, names: list[str], device: str = "cuda") -> list["WhisperSTT"]:
        """One transcriber per preset; presets sharing a model share one copy in memory."""
        from faster_whisper import WhisperModel

        models, out = {}, []
        for name in names:
            model, options = PRESETS[name]
            if model not in models:
                models[model] = WhisperModel(model, device=device, compute_type="float16")
            out.append(cls(model, device, options=options, name=name, loaded=models[model]))
        return out

    def __call__(self, audio_16k: np.ndarray) -> Transcript:
        segments, _info = self.model.transcribe(
            audio_16k, language="tr", condition_on_previous_text=False, **self.options)
        segments = list(segments)
        if not segments:
            return Transcript("", None, None, None)
        return Transcript(
            text=" ".join(s.text.strip() for s in segments).strip(),
            avg_logprob=float(np.mean([s.avg_logprob for s in segments])),
            no_speech_prob=float(max(s.no_speech_prob for s in segments)),
            compression_ratio=float(max(s.compression_ratio for s in segments)),
        )
