"""Prompts for the competitor engines, built from the same grid our engine is trained on.

The zero-shot LLM gets exactly what a human labeler gets: every cell's meaning plus the
hard-case rules, so it is not handicapped by knowing less about the task.
"""

from pathlib import Path

import yaml

GRID = Path(__file__).resolve().parent.parent / "data" / "config" / "grid.yaml"

DOUSERY_SYSTEM = "Sen tıbbi aciliyet değerlendirmesi yapan bir asistansın."  # from its model card


def dousery_prompt(text: str) -> str:
    """The ChatML format shown on the model card (dousery/turkish-medical-triage-llama3-gguf)."""
    return (f"<|im_start|>system\n{DOUSERY_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")


def load_grid(path: Path = GRID) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)["categories"]


def zero_shot_system(grid: list[dict]) -> str:
    lines = [
        "You triage messages that hospital patients or their relatives type to the nurse station "
        "on a Turkish internal medicine ward. Read the Turkish message and choose exactly one "
        "category and one urgency level from the table below. You never diagnose; you only sort "
        "the request so a nurse knows what it is and how soon to go.",
        "",
        "Categories and the urgency levels each can have:",
    ]
    for cat in grid:
        lines.append(f"- {cat['id']} ({cat['name']}; trip: {cat['trip']})")
        for urgency, cell in cat["cells"].items():
            topics = ", ".join(t["id"].replace("_", " ") for t in cell["topics"])
            lines.append(f"    - {urgency}: {cell['means']} (e.g. {topics})")
    lines += [
        "",
        "Rules:",
        "1. When in doubt, choose the higher urgency. A false alarm costs a walk; a miss can cost a life.",
        "2. Label the current state, not the topic: pain that has passed is routine.",
        "3. Vague distress with no stated need is at least soon (category unclear).",
        "4. Several needs in one message: the most urgent need decides category and urgency.",
        "5. A relative speaking for the patient is labeled the same as the patient.",
        "6. Use only a category/urgency pair listed above.",
        "",
        'Answer with JSON only: {"category": "<id>", "urgency": "critical|soon|routine"}.',
    ]
    return "\n".join(lines)


def zero_shot_schema(grid: list[dict]) -> dict:
    return {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": [c["id"] for c in grid]},
            "urgency": {"type": "string", "enum": ["critical", "soon", "routine"]},
        },
        "required": ["category", "urgency"],
    }
