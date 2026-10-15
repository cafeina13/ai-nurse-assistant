"""Scores a voice evaluation on the laptop: transcription errors and what they cost triage.

    python -m voice.report data/generated/voice_eval/transcripts.jsonl
    python -m voice.report ... --model hybrid-critical:tfidf --with-tone-bank   # the demo's engine

Triage runs twice on every clip, on the original text and on the transcript, so the drop
between them is what speech to text costs. Critical recall comes first, as the plan requires.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from triage.evaluate import load_examples
from triage.models import make_model
from triage.normalize import fold
from triage.pipeline import Triage
from voice.checks import reject_reason


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
        "crit_to_routine": share(crit, lambda r: r["voice_urgency"] == "routine"),
        "false_critical_voice": share(other, lambda r: r["voice_urgency"] == "critical"),
        "routine_as_soon": share([r for r in rows if r["urgency"] == "routine"],
                                 lambda r: r["voice_urgency"] == "soon"),
        "flagged_listen": share(rows, lambda r: r["listen"]),
        "listen_on_lost": share([r for r in crit if r["text_urgency"] == "critical" != r["voice_urgency"]],
                                lambda r: r["listen"]),
    }


def _pct(x):
    return "  n/a" if x is None else f"{x:5.1%}"


def print_table(title: str, groups: dict[str, list[dict]]) -> None:
    print(f"\n{title}")
    print(f"  {'':<22} {'clips':>5} {'WER':>6} {'empty':>6} {'crit text':>9} {'crit voice':>10} "
          f"{'lost':>6} {'->routine':>9} {'false crit':>10} {'rout->soon':>10} {'listen':>6} {'listen|lost':>11}")
    for name, rows in groups.items():
        s = summarize(rows)
        print(f"  {name:<22} {s['clips']:>5} {_pct(s['wer'])} {_pct(s['empty'])} {_pct(s['crit_recall_text']):>9} "
              f"{_pct(s['crit_recall_voice']):>10} {_pct(s['lost_in_stt'])} {_pct(s['crit_to_routine']):>9} "
              f"{_pct(s['false_critical_voice']):>10} {_pct(s['routine_as_soon']):>10} {_pct(s['flagged_listen'])} {_pct(s['listen_on_lost']):>11}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("transcripts", type=Path)
    parser.add_argument("--listen-logprob", type=float, default=-0.6,
                        help="flag 'listen to the recording' when Whisper's avg_logprob is below this")
    parser.add_argument("--show", type=int, default=15, help="criticals lost in STT to print")
    parser.add_argument("--train-transcripts", type=Path,
                        help="transcripts of the training bank (voice.run_eval --set train); the model "
                             "then trains on typed texts plus these, to learn what Whisper mishears")
    parser.add_argument("--model", default="tfidf", help='triage model: "tfidf", "hybrid-critical:tfidf", ...')
    parser.add_argument("--with-tone-bank", action="store_true",
                        help="train on train/gemini plus train/gemini_tone")
    parser.add_argument("--stt", help="only clips from this STT preset (the run's 'stt' field)")
    parser.add_argument("--checks", action="store_true",
                        help="reject unbelievable transcripts (voice.checks): they become Unclear/Soon")
    parser.add_argument("--floor", choices=["listen", "stt"],
                        help="never routine when in doubt: 'listen' lifts every listen-flagged request "
                             "to soon; 'stt' only those where speech to text itself is in doubt")
    args = parser.parse_args()

    clips = [json.loads(line) for line in args.transcripts.read_text(encoding="utf-8").splitlines() if line]
    presets = sorted({c["stt"] for c in clips})
    if args.stt:
        clips = [c for c in clips if c["stt"] == args.stt]
    elif len(presets) > 1:
        raise SystemExit(f"several STT presets in this file, pick one with --stt: {presets}")
    for c in clips:
        c["rejected"] = (reject_reason(c["transcript"], c["seconds"], c.get("compression_ratio"))
                         if args.checks else None)
    train = load_examples("train/gemini")
    if args.with_tone_bank:
        train += load_examples("train/gemini_tone")
    texts, cells = [e.text for e in train], [e.cell for e in train]
    if args.train_transcripts:
        heard = [json.loads(line) for line in args.train_transcripts.read_text(encoding="utf-8").splitlines() if line]
        heard = [h for h in heard if h["transcript"].strip()]
        texts += [h["transcript"] for h in heard]
        cells += [f"{h['category']}/{h['urgency']}" for h in heard]
        print(f"Model trained on {len(train)} typed texts + {len(heard)} training transcripts")
    else:
        print(f"Model trained on {len(train)} typed texts only")
    print(f"Triage model: {args.model}")
    triage = Triage(make_model(args.model).fit(texts, cells))
    by_text = triage.run_many([c["text"] for c in clips])
    by_voice = triage.run_many([c["transcript"] or "" for c in clips])

    for clip, t, v in zip(clips, by_text, by_voice):
        # Measured against what was spoken (letters restored), not the typed original.
        clip["word_errors"], clip["ref_words"] = word_errors(clip.get("spoken", clip["text"]), clip["transcript"])
        clip["text_urgency"] = t.urgency
        heard_nothing = not clip["transcript"].strip() or clip["rejected"]
        if heard_nothing:
            # Like the old call button: someone called, we could not tell why.
            clip["voice_urgency"], clip["voice_cell"] = "soon", "unclear/soon"
        else:
            clip["voice_urgency"], clip["voice_cell"] = v.urgency, v.cell
        low_confidence = clip["avg_logprob"] is None or clip["avg_logprob"] < args.listen_logprob
        clip["listen"] = bool(low_confidence or heard_nothing or v.uncertain)
        doubt = clip["listen"] if args.floor == "listen" else bool(low_confidence or heard_nothing)
        if args.floor and doubt and clip["voice_urgency"] == "routine":
            clip["voice_urgency"] = "soon"

    rejected = sum(bool(c["rejected"]) for c in clips)
    print(f"{len(clips)} clips, {len({c['id'] for c in clips})} texts, STT {clips[0]['stt']}"
          f"{f', checks rejected {rejected}' if args.checks else ''}{f', floor on {args.floor} doubt' if args.floor else ''}")
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
