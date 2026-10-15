"""Turkish TTS voices behind one interface: text in, (float32 audio, sample rate) out.

Engines (all local):
  drifting  Vyvo Drifting TTS v3.2, many speaker ids (CC-BY-4.0)
  piper     Piper voices from .onnx files: the official tr_TR-dfki-medium and the
            user's own fine-tuned tr_TR-merve-medium (internal use only, see its consent note)
  mms       facebook/mms-tts-tur (CC-BY-NC)
"""

from pathlib import Path

import numpy as np


def speakable(text: str) -> str:
    """Piper-style VITS voices babble after the last word unless the text ends with
    punctuation (found in the user's TTS experiments), so always end with one."""
    text = " ".join(text.split())
    return text if text[-1:] in ".!?" else text + "."


class PiperTTS:
    def __init__(self, name: str, onnx_path: Path, use_cuda: bool = False):
        from piper import PiperVoice

        self.name = name
        self.voice = PiperVoice.load(str(onnx_path), use_cuda=use_cuda)

    def __call__(self, text: str) -> tuple[np.ndarray, int]:
        chunks = list(self.voice.synthesize(speakable(text)))
        audio = np.concatenate([np.frombuffer(c.audio_int16_bytes, dtype=np.int16) for c in chunks])
        return audio.astype(np.float32) / 32768.0, chunks[0].sample_rate


class DriftingTTS:
    RATE = 24000

    def __init__(self, device: str = "cuda"):
        from drifting_tts.synthesize import Synthesizer

        self.tts = Synthesizer.from_pretrained("v3.2", device)

    def voice(self, speaker: int):
        def speak(text: str) -> tuple[np.ndarray, int]:
            wav, _info = self.tts(speakable(text), speaker=speaker, cfg_scale=2.0,
                                  temperature=0.3, seed=0)
            return wav.detach().cpu().numpy().astype(np.float32).reshape(-1), self.RATE

        speak.name = f"drifting-{speaker}"
        return speak


class MmsTTS:
    name = "mms-tur"

    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import AutoTokenizer, VitsModel

        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tur")
        self.model = VitsModel.from_pretrained("facebook/mms-tts-tur").to(device).eval()

    def __call__(self, text: str) -> tuple[np.ndarray, int]:
        inputs = self.tokenizer(speakable(text), return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            wav = self.model(**inputs).waveform[0]
        return wav.cpu().numpy().astype(np.float32), self.model.config.sampling_rate


def pick_drifting_speakers(speakers_csv: Path, n: int = 16) -> list[int]:
    """Spreads speakers across pitch (f0) and the model's own intelligibility (wer), so the
    set covers low and high voices and easy and harder ones, not just the clearest few."""
    import csv

    with speakers_csv.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if not r["voice"]]  # skip the named voices
    rows.sort(key=lambda r: float(r["f0_hz"]))
    bands = np.array_split(np.arange(len(rows)), n // 2)
    chosen = []
    for band in bands:
        in_band = sorted((rows[i] for i in band), key=lambda r: float(r["wer"]))
        chosen += [int(in_band[len(in_band) // 4]["speaker"]), int(in_band[-len(in_band) // 4]["speaker"])]
    return chosen
