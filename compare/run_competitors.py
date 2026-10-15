"""Runs a competitor triage engine on the test texts and records its raw answers (GPU, Kaggle).

    python -m compare.run_competitors --engine dousery --banks DIR --out dousery.jsonl
    python -m compare.run_competitors --engine llama31 --banks DIR --out llama31.jsonl

Only raw answers are stored; parsing them into urgency levels happens later on the laptop
(compare.report), so a parsing mistake never costs another GPU run. Temperature 0 for
reproducible answers.
"""

import argparse
import json
import time
from pathlib import Path

from compare.prompts import dousery_prompt, load_grid, zero_shot_schema, zero_shot_system

ENGINES = {
    # Existing Turkish triage engine: LLaMA 3 8B + LoRA, fine-tuned on synthetic Turkish
    # emergency data; answers in free text (triage level, possible diagnosis, advice).
    "dousery": ("dousery/turkish-medical-triage-llama3-gguf", "unsloth.Q4_K_M.gguf"),
    # The same model family, untuned, given our grid as instructions (zero-shot).
    "llama31": ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
}


def load_texts(banks: Path) -> list[dict]:
    rows = []
    for bank in ("claude", "tone", "fresh"):
        if not (banks / bank).exists():
            continue
        for path in sorted((banks / bank).glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                row["bank"] = f"test/{bank}"
                rows.append(row)
    for i, row in enumerate(rows):
        row["id"] = i
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", choices=list(ENGINES), required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--grid", type=Path, help="grid.yaml (for the zero-shot prompt)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=300, help="dousery answers in free text")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    repo, filename = ENGINES[args.engine]
    model_path = hf_hub_download(repo, filename)
    llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=4096, seed=0, verbose=False)
    texts = load_texts(args.banks)[:args.limit]

    if args.engine == "llama31":
        grid = load_grid(args.grid) if args.grid else load_grid()
        system, schema = zero_shot_system(grid), zero_shot_schema(grid)

    started = time.time()
    with args.out.open("w", encoding="utf-8") as out:
        for n, row in enumerate(texts, 1):
            t0 = time.time()
            if args.engine == "dousery":
                result = llm.create_completion(dousery_prompt(row["text"]), max_tokens=args.max_tokens,
                                               temperature=0.0, stop=["<|im_end|>"])
                raw = result["choices"][0]["text"]
            else:
                result = llm.create_chat_completion(
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": row["text"]}],
                    response_format={"type": "json_object", "schema": schema},
                    temperature=0.0, max_tokens=60)
                raw = result["choices"][0]["message"]["content"]
            out.write(json.dumps({
                "id": row["id"], "bank": row["bank"], "text": row["text"], "category": row["category"],
                "urgency": row["urgency"], "topic": row["topic"], "tone": row.get("tone"),
                "engine": args.engine, "raw": raw, "tokens": result["usage"]["completion_tokens"],
                "seconds": round(time.time() - t0, 2),
            }, ensure_ascii=False) + "\n")
            if n % 25 == 0:
                out.flush()
                rate = (time.time() - started) / n
                print(f"{args.engine}: {n}/{len(texts)}, {rate:.1f} s/text, ~{rate * (len(texts) - n) / 60:.0f} min left",
                      flush=True)
    print(f"{args.engine}: done {len(texts)} texts in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
