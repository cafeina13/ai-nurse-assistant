"""TTS -> ward conditions -> speech to text, for every test text. Runs on a GPU (Kaggle).

    python -m voice.run_eval --banks DIR --out transcripts.jsonl [--merve PATH.onnx]

Each test text is spoken by one Drifting speaker and one other voice (Piper dfki, MMS or
Merve, in turn), then transcribed under each condition in voice.audio.CONDITIONS. Triage
scoring happens later, on the laptop (voice.report), so this stays a pure audio job.
"""

import argparse
import json
import re
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np

from voice.audio import CONDITIONS, STT_RATE, apply_condition, resample, room_hum
from voice.stt import WhisperSTT
from voice.tts import DriftingTTS, MmsTTS, PiperTTS, pick_drifting_speakers

HF = "https://huggingface.co"
DFKI = f"{HF}/rhasspy/piper-voices/resolve/main/tr/tr_TR/dfki/medium/tr_TR-dfki-medium.onnx"
SPEAKERS_CSV = f"{HF}/Vyvo/drifting-tts-tr/resolve/main/speakers.csv"

# Background chatter for the ward conditions: neutral speech, never test texts, so nothing
# from one request can leak into another's transcript.
BABBLE = [
    "Bugün hava çok güzel, öğleden sonra biraz yürüyüş yapalım.",
    "Haberlerde yarın yağmur yağacağını söylediler.",
    "Kardeşim akşam gelecek, ona da yemek getirsin.",
    "Maç kaç kaç bitti, sen izledin mi?",
    "Torunum bu sene üniversiteyi kazandı, çok sevindik.",
    "Otoparkta yer bulmak çok zor oldu bugün.",
    "Annemler bayramda köye gidecekmiş.",
    "Bu dizinin yeni bölümü ne zaman çıkıyor acaba?",
    "Kantinden bir çay alıp geliyorum hemen.",
    "Telefonun şarjı bitmek üzere, kabloyu verir misin?",
    "Komşumuzun kızı geçen hafta evlendi.",
    "Market kapanmadan ekmek almam lazım.",
    "Otobüs bugün yine geç geldi.",
    "Bahçedeki domatesler bu yıl çok güzel oldu.",
    "Pazartesi işe başlıyorum, biraz heyecanlıyım.",
    "Yeğenim yarın askerden dönüyor.",
    "Bu yıl tatile gidemedik, belki seneye.",
    "Radyoda eski şarkılar çalıyor, çok hoşuma gidiyor.",
    "Arabanın lastiği patlamış, tamirciye götüreceğiz.",
    "Akşam yemeğinde mercimek çorbası vardı.",
]
_TR_LETTERS = set("çğıöşüÇĞİÖŞÜ")


def spoken_form(text: str) -> str:
    """What a person would say: speech has no missing Turkish letters or "gelinnn" typos.
    Texts typed entirely in ASCII get their letters restored before TTS."""
    text = re.sub(r"(\w)\1{2,}", r"\1", text)
    if _TR_LETTERS & set(text):
        return text
    try:
        from turkish.deasciifier import Deasciifier
    except ImportError:
        return text
    return Deasciifier(text).convert_to_turkish()


def download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def load_texts(banks_dir: Path) -> list[dict]:
    rows = []
    for bank in ("claude", "tone"):
        for path in sorted((banks_dir / bank).glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                row["bank"] = f"test/{bank}"
                rows.append(row)
    for i, row in enumerate(rows):
        row["id"] = i
    return rows


def load_voices(work: Path, merve: Path | None, device: str) -> tuple[list, list]:
    """Returns (drifting voices, other voices); an engine that fails to load is skipped."""
    drifting, others = [], []
    try:
        engine = DriftingTTS(device)
        speakers = pick_drifting_speakers(download(SPEAKERS_CSV, work / "speakers.csv"))
        drifting = [engine.voice(s) for s in speakers]
        print(f"drifting: {len(drifting)} speakers {speakers}", flush=True)
    except Exception:
        traceback.print_exc()
    def load_dfki():
        download(DFKI + ".json", work / "dfki.onnx.json")  # Piper reads the config beside the model
        return PiperTTS("piper-dfki", download(DFKI, work / "dfki.onnx"))

    loaders = [("piper-dfki", load_dfki), ("mms-tur", lambda: MmsTTS(device))]
    if merve is not None:
        loaders.append(("piper-merve", lambda: PiperTTS("piper-merve", merve)))
    for name, load in loaders:
        try:
            others.append(load())
            print(f"loaded {name}", flush=True)
        except Exception:
            traceback.print_exc()
    if not drifting and not others:
        raise SystemExit("no TTS voice could be loaded")
    return drifting, others


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--banks", type=Path, required=True, help="folder with claude/ and tone/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--merve", type=Path, help="tr_TR-merve-medium.onnx (with its .json)")
    parser.add_argument("--stt-model", default="large-v3")
    parser.add_argument("--samples", type=Path, help="folder to save a few example wavs")
    parser.add_argument("--limit", type=int, help="only the first N texts (smoke test)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import soundfile as sf

    # Downloads go to a temp folder, not next to the output: Kaggle returns everything in
    # its output folder, and a 60 MB voice file is not a result.
    work = Path(tempfile.gettempdir()) / "voice_eval_cache"
    texts = load_texts(args.banks)[:args.limit]
    drifting, others = load_voices(work, args.merve, args.device)
    stt = WhisperSTT(args.stt_model, device=args.device)
    rng = np.random.default_rng(0)
    babble_pool = []
    for i, sentence in enumerate(BABBLE):
        voice = (drifting + others)[i % len(drifting + others)]
        try:
            babble_pool.append(resample(*voice(sentence)))
        except Exception as e:
            print(f"babble failed {voice.name}: {e}", flush=True)
    if args.samples:
        args.samples.mkdir(parents=True, exist_ok=True)

    started, done = time.time(), 0
    with args.out.open("w", encoding="utf-8") as out:
        for row in texts:
            voices = []
            if drifting:
                voices.append(drifting[row["id"] % len(drifting)])
            if others:
                voices.append(others[row["id"] % len(others)])
            spoken = spoken_form(row["text"])
            for voice in voices:
                try:
                    audio, rate = voice(spoken)
                except Exception as e:
                    print(f"tts failed {voice.name} #{row['id']}: {e}", flush=True)
                    continue
                speech = resample(audio, rate)
                for condition in CONDITIONS:
                    babble = babble_pool[rng.integers(len(babble_pool))] if babble_pool else room_hum(STT_RATE, rng)
                    heard = apply_condition(speech, condition, babble, rng)
                    t = stt(heard)
                    out.write(json.dumps({
                        "id": row["id"], "bank": row["bank"], "text": row["text"], "spoken": spoken,
                        "category": row["category"], "urgency": row["urgency"], "topic": row["topic"],
                        "speaker": row["speaker"], "tone": row.get("tone"), "voice": voice.name,
                        "condition": condition, "transcript": t.text, "avg_logprob": t.avg_logprob,
                        "no_speech_prob": t.no_speech_prob, "seconds": round(len(speech) / STT_RATE, 2),
                        "stt": stt.name,
                    }, ensure_ascii=False) + "\n")
                    if args.samples and row["id"] < 6:
                        sf.write(args.samples / f"{row['id']}_{voice.name}_{condition}.wav", heard, STT_RATE)
            done += 1
            if done % 25 == 0:
                out.flush()
                rate_s = (time.time() - started) / done
                print(f"{done}/{len(texts)} texts, {rate_s:.1f} s/text, "
                      f"~{rate_s * (len(texts) - done) / 60:.0f} min left", flush=True)
    print(f"done: {done} texts in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
