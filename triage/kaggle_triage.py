"""Evaluates triage models on a Kaggle T4x2, from the laptop; only small reports come back.

    python -m triage.kaggle_triage --models emb:intfloat/multilingual-e5-base,emb:BAAI/bge-m3

Each model runs `python -m triage.evaluate --final --with-tone-bank --model <spec>` on Kaggle;
models are spread over both GPUs. The experiment records land in experiments/, like local runs.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from voice.kaggle_job import kaggle, publish_dataset, username

REPO = Path(__file__).resolve().parent.parent
STAGE = REPO / "data" / "generated" / "kaggle" / "triage"
SLUG = "ai-nurse-triage-models"
DEFAULT_MODELS = ",".join([
    "emb:intfloat/multilingual-e5-base",
    "emb:intfloat/multilingual-e5-large",
    "emb:BAAI/bge-m3",
    "emb:trmteb/turkish-embedding-model",
])

RUN_SCRIPT = '''\
import glob, os, subprocess, sys, threading
from pathlib import Path

MODELS = {models!r}
EXTRA = {extra!r}

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"], check=True)
code = Path(next(glob.iglob("/kaggle/input/**/triage/evaluate.py", recursive=True))).parent.parent
gpus = len(subprocess.check_output(["nvidia-smi", "-L"], text=True).strip().splitlines())
work = Path("/kaggle/working")
env_base = {{**os.environ, "PYTHONPATH": os.pathsep.join([str(code), str(code / "data")]),
            "TRIAGE_EXPERIMENTS": str(work / "experiments")}}

def run_queue(gpu, specs, failures):
    # Models on one GPU run one after another; the two GPUs run in parallel.
    for spec in specs:
        name = spec.replace("/", "_").replace(":", "_")
        cmd = [sys.executable, "-m", "triage.evaluate", "--final", "--model", spec, "--show", "0"] + EXTRA
        with open(work / f"{{name}}.log", "w") as log:
            rc = subprocess.run(cmd, cwd=work, env={{**env_base, "CUDA_VISIBLE_DEVICES": str(gpu)}},
                                stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{{spec}} on GPU {{gpu}}: exit {{rc}}", flush=True)
        if rc:
            failures.append(spec)

failures = []
threads = [threading.Thread(target=run_queue, args=(g, MODELS[g::gpus], failures)) for g in range(gpus)]
for t in threads: t.start()
for t in threads: t.join()
if failures:
    sys.exit(f"failed: {{failures}}")
'''


def stage_code() -> Path:
    folder = STAGE / "code"
    shutil.rmtree(folder, ignore_errors=True)
    ignore = shutil.ignore_patterns("__pycache__")
    for package in ("triage", "backend"):
        shutil.copytree(REPO / package, folder / package, ignore=ignore)
    shutil.copytree(REPO / "data" / "wardgen", folder / "data" / "wardgen", ignore=ignore)
    shutil.copytree(REPO / "data" / "config", folder / "data" / "config")
    for bank in ("train/gemini", "train/gemini_tone", "test/claude", "test/tone"):
        shutil.copytree(REPO / "data" / "textbank" / bank, folder / "data" / "textbank" / bank)
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--no-tone-bank", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    user = username()
    kernel_ref = f"{user}/{SLUG}"
    results = REPO / "data" / "generated" / "triage_models"
    if not args.download_only:
        source = publish_dataset(stage_code(), "ai-nurse-triage-code", "ai-nurse-triage-code", user)
        kernel = STAGE / "kernel"
        kernel.mkdir(parents=True, exist_ok=True)
        extra = [] if args.no_tone_bank else ["--with-tone-bank"]
        (kernel / "run.py").write_text(RUN_SCRIPT.format(models=args.models.split(","), extra=extra),
                                       encoding="utf-8")
        (kernel / "kernel-metadata.json").write_text(json.dumps({
            "id": kernel_ref, "title": SLUG, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
            "enable_internet": "true", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [source],
            "competition_sources": [], "kernel_sources": [], "model_sources": [],
        }, indent=1), encoding="utf-8")
        result = kaggle("kernels", "push", "-p", str(kernel), "--accelerator", "NvidiaTeslaT4", check=False)
        print(result.stdout.strip() or result.stderr.strip())
        if result.returncode != 0:
            sys.exit(1)
        while True:
            status = kaggle("kernels", "status", kernel_ref, check=False).stdout.strip()
            print(time.strftime("%H:%M:%S"), status, flush=True)
            if any(word in status.lower() for word in ("complete", "error", "cancel")):
                break
            time.sleep(60)

    results.mkdir(parents=True, exist_ok=True)
    result = kaggle("kernels", "output", kernel_ref, "-p", str(results), "-o",
                    "--file-pattern", r"\.(json|log)$", check=False)
    print(result.stdout.strip()[-1500:] or result.stderr.strip())
    copied = 0
    for path in results.rglob("*.json"):
        if path.parent.name == "experiments" or path.name.startswith("2"):
            target = REPO / "experiments" / path.name
            if not target.exists():
                shutil.copy2(path, target)
                copied += 1
    print(f"Results in {results}; {copied} experiment record(s) copied to experiments/")


if __name__ == "__main__":
    main()
