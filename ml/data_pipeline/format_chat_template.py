"""Merge MedQA-derived and synthetic NICE-derived pairs into one Llama-3
chat-formatted training file.

Matching Llama-3's own chat template (rather than a custom format) matters
because the model already learned to recognise this turn structure during
pretraining. Loss masking (assistant-only tokens) happens downstream in
ml/training/sft_train.py, not here.
"""

import json
import re
from pathlib import Path

# Keep in sync with the production agent's system prompt (configs/domain/
# medical.yaml) so the model isn't trained against a different prompt than
# the one it's served behind at inference time.
SYSTEM_PROMPT = "You are a clinical decision support assistant trained on NICE guidelines."

MEDQA_PATH = Path("data/raw/medqa/medqa_normalised.jsonl")
SYNTHETIC_PATH = Path("data/synthetic/qa_pairs.jsonl")
OUTPUT_PATH = Path("data/processed/train.jsonl")


def to_chat_format(question: str, answer: str) -> dict:
    """Wrap a (question, answer) pair into {"messages": [system, user, assistant]}."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


# MCQ stems almost always phrase the ask as "Which of the following ..."; swapping
# that for "What" turns the stem into an open question without touching the
# clinical scenario, so no options list is needed in the prompt.
_MCQ_STEM = re.compile(r"which of the following", re.IGNORECASE)


def medqa_to_pair(example: dict) -> tuple[str, str]:
    """Convert a normalised MedQA record into a free-text (question, answer) pair."""
    question = _MCQ_STEM.sub(
        lambda m: "What" if m.group(0)[0].isupper() else "what",
        example["clinical_scenario"].strip(),
        count=1,
    )

    answer = example["correct_answer"].strip()
    if not answer.endswith("."):
        answer += "."
    # rationale is null in the current MedQA export, but future sources may fill it.
    if example.get("rationale"):
        answer = f"{answer} {example['rationale'].strip()}"
    return question, answer


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    pairs = []
    for example in load_jsonl(MEDQA_PATH):
        pairs.append(to_chat_format(*medqa_to_pair(example)))
    for qa in load_jsonl(SYNTHETIC_PATH):
        pairs.append(to_chat_format(qa["instruction"], qa["response"]))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")


if __name__ == "__main__":
    main()
