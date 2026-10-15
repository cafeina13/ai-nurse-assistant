"""Scores a voice evaluation on the laptop: transcription errors and what they cost triage.

    python -m voice.report data/generated/voice_eval/transcripts.jsonl

Triage runs twice on every clip, on the original text and on the transcript, so the drop
between them is what speech to text costs. Critical recall comes first, as the plan requires.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from triage.evaluate import load_examples
from triage.models import TfidfModel
from triage.normalize import fold
from triage.pipeline import Triage


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """(edit distance in words, reference length), on folded text."""
    ref, hyp = fold(reference).split(), fold(hypothesis).split()
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1], len(ref)


def summarize(rows: list[dict]) -> dict:
    errors = sum(r["word_errors"] for r in rows)
    words = sum(r["ref_words"] for r in rows)
    crit = [r for r in rows if r["urgency"] == "critical"]
    other = [r for r in rows if r["urgency"] != "critical"]

    def share(items, key):
        return sum(key(r) for r in items) / len(items) if items else None

    return {
        "clips": len(rows),
        "wer": errors / words if words else None,
        "empty": share(rows, lambda r: not r["transcript"].strip()),
        "crit_recall_text": share(crit, lambda r: r["text_urgency"] == "critical"),
        "crit_recall_voice": share(crit, lambda r: r["voice_urgency"] == "critical"),
        "lost_in_stt": share(crit, lambda r: r["text_urgency"] == "critical" != r["voice_urgency"]),
        "false_critical_voice": share(other, lambda r: r["voice_urgency"] == "critical"),
        "flagged_listen": share(rows, lambda r: r["listen"]),
        "listen_on_lost": share([r for r in crit if r["text_urgency"] == "critical" != r["voice_urgency"]],
                                lambda r: r["listen"]),
    }


def _pct(x):
    return "  n/a" if x is None else f"{x:5.1%}"


def print_table(title: str, groups: dict[str, list[dict]]) -> None:
    print(f"\n{title}")
    print(f"  {'':<22} {'clips':>5} {'WER':>6} {'empty':>6} {'crit text':>9} {'crit voice':>10} "
          f"{'lost':>6} {'false crit':>10} {'listen':>6}")
    for name, rows in groups.items():
        s = summarize(rows)
        print(f"  {name:<22} {s['clips']:>5} {_pct(s['wer'])} {_pct(s['empty'])} {_pct(s['crit_recall_text']):>9} "
              f"{_pct(s['crit_recall_voice']):>10} {_pct(s['lost_in_stt'])} {_pct(s['false_critical_voice']):>10} "
              f"{_pct(s['flagged_listen'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("transcripts", type=Path)
    parser.add_argument("--listen-logprob", type=float, default=-0.6,
                        help="flag 'listen to the recording' when Whisper's avg_logprob is below this")
    parser.add_argument("--show", type=int, default=15, help="criticals lost in STT to print")
    parser.add_argument("--train-transcripts", type=Path,
                        help="transcripts of the training bank (voice.run_eval --set train); the model "
                             "then trains on typed texts plus these, to learn what Whisper mishears")
    args = parser.parse_args()

    clips = [json.loads(line) for line in args.transcripts.read_text(encoding="utf-8").splitlines() if line]
    train = load_examples("train/gemini")
    texts, cells = [e.text for e in train], [e.cell for e in train]
    if args.train_transcripts:
        heard = [json.loads(line) for line in args.train_transcripts.read_text(encoding="utf-8").splitlines() if line]
        heard = [h for h in heard if h["transcript"].strip()]
        texts += [h["transcript"] for h in heard]
        cells += [f"{h['category']}/{h['urgency']}" for h in heard]
        print(f"Model trained on {len(train)} typed texts + {len(heard)} training transcripts")
    else:
        print(f"Model trained on {len(train)} typed texts only")
    triage = Triage(TfidfModel().fit(texts, cells))
    by_text = triage.run_many([c["text"] for c in clips])
    by_voice = triage.run_many([c["transcript"] or "" for c in clips])

    for clip, t, v in zip(clips, by_text, by_voice):
        # Measured against what was spoken (letters restored), not the typed original.
        clip["word_errors"], clip["ref_words"] = word_errors(clip.get("spoken", clip["text"]), clip["transcript"])
        clip["text_urgency"], clip["voice_urgency"] = t.urgency, v.urgency
        clip["voice_cell"] = v.cell
        low_confidence = clip["avg_logprob"] is None or clip["avg_logprob"] < args.listen_logprob
        clip["listen"] = low_confidence or not clip["transcript"].strip() or v.uncertain

    print(f"{len(clips)} clips, {len({c['id'] for c in clips})} texts, STT {clips[0]['stt']}")
    groups = defaultdict(list)
    for c in clips:
        groups[c["condition"]].append(c)
    print_table("By condition", dict(sorted(groups.items())))
    groups = defaultdict(list)
    for c in clips:
        family = "drifting" if c["voice"].startswith("drifting") else c["voice"]
        groups[f"{family} / {c['condition']}"].append(c)
    print_table("By voice and condition", dict(sorted(groups.items())))
    tone = [c for c in clips if c.get("tone")]
    if tone:
        groups = defaultdict(list)
        for c in tone:
            groups[f"{c['tone']} / {c['condition']}"].append(c)
        print_table("Tone sets", dict(sorted(groups.items())))

    lost = [c for c in clips if c["urgency"] == "critical" and c["text_urgency"] == "critical" != c["voice_urgency"]]
    print(f"\nCriticals caught from text but lost after speech to text: {len(lost)}")
    for c in lost[:args.show]:
        flag = "LISTEN" if c["listen"] else "      "
        print(f"  {flag} [{c['voice']}/{c['condition']}] {c['text']}  ->  {c['transcript']!r} ({c['voice_cell']})")


if __name__ == "__main__":
    main()
