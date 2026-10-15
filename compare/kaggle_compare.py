"""Runs the competitor engines on a Kaggle T4x2, one engine per GPU, from the laptop.

    python -m compare.kaggle_compare --limit 20    # smoke test: see answers and real cost
    python -m compare.kaggle_compare               # full run on all test texts

Results land in data/generated/compare/<engine>.jsonl. Needs `kaggle auth login` once.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from voice.kaggle_job import kaggle, publish_dataset, username

REPO = Path(__file__).resolve().parent.parent
STAGE = REPO / "data" / "generated" / "kaggle" / "compare"
RESULTS = REPO / "data" / "generated" / "compare"
SLUG = "ai-nurse-compare"

RUN_SCRIPT = '''\
import glob, os, subprocess, sys
from pathlib import Path

LIMIT = {limit!r}
MAX_TOKENS = {max_tokens!r}
ENGINES = {engines!r}

def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

# Prebuilt CUDA wheel: no compiling on Kaggle's clock.
sh("pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124")
code = Path(next(glob.iglob("/kaggle/input/**/compare/run_competitors.py", recursive=True))).parent.parent
gpus = len(subprocess.check_output(["nvidia-smi", "-L"], text=True).strip().splitlines())
work = Path("/kaggle/working")
procs = []
for i, engine in enumerate(ENGINES):
    cmd = [sys.executable, "-m", "compare.run_competitors", "--engine", engine,
           "--banks", str(code / "banks"), "--grid", str(code / "data" / "config" / "grid.yaml"),
           "--out", str(work / f"{{engine}}.jsonl"), "--max-tokens", str(MAX_TOKENS)]
    if LIMIT:
        cmd += ["--limit", str(LIMIT)]
    env = {{**os.environ, "PYTHONPATH": str(code), "CUDA_VISIBLE_DEVICES": str(i % gpus)}}
    log = open(work / f"{{engine}}.log", "w")
    procs.append((engine, subprocess.Popen(cmd, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT), log))
failed = []
for engine, proc, log in procs:
    proc.wait()
    log.close()
    print(f"{{engine}} exit {{proc.returncode}}", flush=True)
    if proc.returncode != 0:
        failed.append(engine)
        print((work / f"{{engine}}.log").read_text()[-3000:])
if failed:
    sys.exit(f"engines failed: {{failed}}")
'''


def stage_code() -> Path:
    folder = STAGE / "code"
    shutil.rmtree(folder, ignore_errors=True)
    shutil.copytree(REPO / "compare", folder / "compare", ignore=shutil.ignore_patterns("__pycache__"))
    (folder / "data" / "config").mkdir(parents=True)
    shutil.copy2(REPO / "data" / "config" / "grid.yaml", folder / "data" / "config" / "grid.yaml")
    for bank in ("claude", "tone", "fresh"):
        if (REPO / "data" / "textbank" / "test" / bank).exists():
            shutil.copytree(REPO / "data" / "textbank" / "test" / bank, folder / "banks" / bank)
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="only the first N texts (smoke test)")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--engines", default="dousery,llama31")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    user = username()
    kernel_ref = f"{user}/{SLUG}"
    if not args.download_only:
        source = publish_dataset(stage_code(), "ai-nurse-compare-code", "ai-nurse-compare-code", user)
        kernel = STAGE / "kernel"
        kernel.mkdir(parents=True, exist_ok=True)
        (kernel / "run.py").write_text(RUN_SCRIPT.format(
            limit=args.limit, max_tokens=args.max_tokens, engines=args.engines.split(",")), encoding="utf-8")
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

    RESULTS.mkdir(parents=True, exist_ok=True)
    result = kaggle("kernels", "output", kernel_ref, "-p", str(RESULTS), "-o",
                    "--file-pattern", r"\.(jsonl|log)$", check=False)
    print(result.stdout.strip()[-1500:] or result.stderr.strip())
    print(f"Results in {RESULTS}")


if __name__ == "__main__":
    main()
