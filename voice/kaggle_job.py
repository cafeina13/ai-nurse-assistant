"""Runs the voice evaluation on a Kaggle GPU, from the laptop, with the Kaggle CLI.

    python -m voice.kaggle_job --limit 10      # smoke test: 10 texts
    python -m voice.kaggle_job                 # full run
    python -m voice.kaggle_job --download-only # fetch results of the last run
    python -m voice.kaggle_job --set train     # speak the training bank (for transcript training)
    python -m voice.kaggle_job --set fresh --stt-presets large-v3-syntia-hint   # the blind test set
    python -m voice.kaggle_job --tag stt --stt-presets large-v3-syntia,turbo-syntia

Uploads two private datasets (the code with the test texts, and the Merve voice), pushes a
GPU script that runs voice.run_eval, waits for it, and downloads transcripts.jsonl and a
few sample wavs to data/generated/voice_eval/. Needs `kaggle auth login` once.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STAGE = REPO / "data" / "generated" / "kaggle"
RESULTS = REPO / "data" / "generated" / "voice_eval"
MERVE_DIR = Path(r"\\wsl.localhost\Ubuntu\home\cafeina\TTS\deploy\step_15000")

RUN_SCRIPT = '''\
import glob, os, subprocess, sys
from pathlib import Path

LIMIT = {limit!r}
SET = {set!r}
PRESETS = {presets!r}
WORKERS_PER_GPU = {workers_per_gpu!r}

def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

sh('pip install -q faster-whisper piper-tts soundfile '
   '"drifting-tts[bigvgan] @ git+https://github.com/kadirnar/drifting-tts" '
   'git+https://github.com/emres/turkish-deasciifier')
code = Path(next(glob.iglob("/kaggle/input/**/voice/run_eval.py", recursive=True))).parent.parent
merve = next(glob.iglob("/kaggle/input/**/tr_TR-merve-medium.onnx", recursive=True), None)
gpus = len(subprocess.check_output(["nvidia-smi", "-L"], text=True).strip().splitlines())
workers = gpus * WORKERS_PER_GPU
print(f"{{gpus}} GPU(s), {{workers}} workers", flush=True)

# Every worker runs the same code on every Nth text, pinned to one GPU. Two per GPU keep it
# busy while the other worker does CPU work (TTS on CPU, noise mixing).
work = Path("/kaggle/working")
procs = []
for i in range(workers):
    cmd = [sys.executable, "-m", "voice.run_eval", "--banks", str(code / "banks"),
           "--out", str(work / f"part{{i}}.jsonl"), "--set", SET,
           "--shard", str(i), "--shards", str(workers)]
    if i == 0:
        cmd += ["--samples", str(work / "samples")]
    if merve:
        cmd += ["--merve", merve]
    if LIMIT:
        cmd += ["--limit", str(LIMIT)]
    if PRESETS:
        cmd += ["--stt-presets", PRESETS]
    env = {{**os.environ, "PYTHONPATH": str(code), "CUDA_VISIBLE_DEVICES": str(i % gpus)}}
    log = open(work / f"worker{{i}}.log", "w")
    procs.append((i, subprocess.Popen(cmd, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT), log))

failed = []
for i, proc, log in procs:
    proc.wait()
    log.close()
    print(f"worker {{i}} exit {{proc.returncode}}", flush=True)
    if proc.returncode != 0:
        failed.append(i)
for i in failed:
    print(f"--- worker {{i}} log (tail) ---")
    print((work / f"worker{{i}}.log").read_text()[-3000:])
if failed:
    sys.exit(f"workers failed: {{failed}}")

with open(work / "transcripts.jsonl", "w", encoding="utf-8") as out:
    for i in range(workers):
        part = work / f"part{{i}}.jsonl"
        out.write(part.read_text(encoding="utf-8"))
        part.unlink()
print("merged", flush=True)
'''


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    print("$ kaggle " + " ".join(args), flush=True)
    # UTF-8 so the CLI does not crash writing Turkish text with the Windows default codec.
    return subprocess.run(["kaggle", *args], check=check, text=True, capture_output=True,
                          encoding="utf-8", errors="replace",
                          env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})


def username() -> str:
    out = kaggle("config", "view").stdout
    for line in out.splitlines():
        if line.strip().startswith("- username:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit("could not read the Kaggle username; run `kaggle auth login`")


def publish_dataset(folder: Path, slug: str, title: str, user: str) -> str:
    """Creates the private dataset, or adds a new version if it already exists."""
    ref = f"{user}/{slug}"
    (folder / "dataset-metadata.json").write_text(json.dumps(
        {"title": title, "id": ref, "licenses": [{"name": "other"}]}), encoding="utf-8")
    if kaggle("datasets", "status", ref, check=False).returncode == 0:
        result = kaggle("datasets", "version", "-p", str(folder), "-m", "update", "-r", "zip", check=False)
    else:
        result = kaggle("datasets", "create", "-p", str(folder), "-r", "zip", check=False)
    print(result.stdout.strip() or result.stderr.strip())
    for _ in range(60):  # Kaggle processes uploads before kernels can mount them
        status = kaggle("datasets", "status", ref, check=False).stdout.strip()
        if "ready" in status:
            return ref
        time.sleep(10)
    raise SystemExit(f"dataset {ref} not ready: {status}")


def stage_code() -> Path:
    folder = STAGE / "code"
    shutil.rmtree(folder, ignore_errors=True)
    shutil.copytree(REPO / "voice", folder / "voice", ignore=shutil.ignore_patterns("__pycache__"))
    banks = REPO / "data" / "textbank"
    for bank in ("test/claude", "test/tone", "test/fresh", "train/gemini"):
        shutil.copytree(banks / bank, folder / "banks" / bank.split("/")[1])
    return folder


def stage_merve() -> Path | None:
    if not (MERVE_DIR / "tr_TR-merve-medium.onnx").exists():
        print(f"Merve voice not found at {MERVE_DIR}; running without it")
        return None
    folder = STAGE / "merve"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("tr_TR-merve-medium.onnx", "tr_TR-merve-medium.onnx.json"):
        target = folder / name
        if not target.exists():
            shutil.copy2(MERVE_DIR / name, target)
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="only the first N texts (smoke test)")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--skip-merve-upload", action="store_true", help="the voice is already up")
    parser.add_argument("--set", choices=["test", "train", "fresh"], default="test")
    parser.add_argument("--stt-presets", help="voice.stt.PRESETS names, comma-separated")
    parser.add_argument("--tag", help="name for a separate notebook and results folder")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    args = parser.parse_args()

    user = username()
    # Separate notebooks, so a training run never overwrites the test results.
    tag = args.tag or (args.set if args.set != "test" else None)
    # Kaggle turns "_" into "-" in notebook ids; asking for the raw name later finds nothing.
    slug = f"ai-nurse-voice-{tag.replace('_', '-')}" if tag else "ai-nurse-voice-eval"
    kernel_ref = f"{user}/{slug}"
    results = RESULTS / tag if tag else RESULTS
    if not args.download_only:
        sources = [publish_dataset(stage_code(), "ai-nurse-voice-code", "ai-nurse-voice-code", user)]
        merve = None if args.skip_merve_upload else stage_merve()
        if merve:
            sources.append(publish_dataset(merve, "merve-tts-voice", "merve-tts-voice", user))
        elif args.skip_merve_upload:
            sources.append(f"{user}/merve-tts-voice")

        kernel = STAGE / f"kernel-{tag or 'eval'}"
        kernel.mkdir(parents=True, exist_ok=True)
        (kernel / "run.py").write_text(RUN_SCRIPT.format(limit=args.limit, set=args.set, presets=args.stt_presets,
                          workers_per_gpu=args.workers_per_gpu), encoding="utf-8")
        (kernel / "kernel-metadata.json").write_text(json.dumps({
            "id": kernel_ref, "title": slug, "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": "true",
            "enable_gpu": "true", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": sources, "competition_sources": [], "kernel_sources": [],
            "model_sources": [],
        }, indent=1), encoding="utf-8")
        result = kaggle("kernels", "push", "-p", str(kernel), "--accelerator", "NvidiaTeslaT4", check=False)
        print(result.stdout.strip() or result.stderr.strip())
        if result.returncode != 0:
            sys.exit(1)

        while True:
            check = kaggle("kernels", "status", kernel_ref, check=False)
            if check.returncode != 0:  # never poll forever on a notebook that cannot be found
                sys.exit(f"status check failed: {check.stderr.strip() or check.stdout.strip()}")
            status = check.stdout.strip()
            print(time.strftime("%H:%M:%S"), status, flush=True)
            if any(word in status.lower() for word in ("complete", "error", "cancel")):
                break
            time.sleep(60)

    results.mkdir(parents=True, exist_ok=True)
    result = kaggle("kernels", "output", kernel_ref, "-p", str(results), "-o",
                    "--file-pattern", r"(transcripts\.jsonl|\.log|\.wav)$", check=False)
    print(result.stdout.strip()[-2000:] or result.stderr.strip())
    print(f"Results in {results}")


if __name__ == "__main__":
    main()
