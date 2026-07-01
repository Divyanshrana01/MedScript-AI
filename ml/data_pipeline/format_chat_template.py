"""Merge MedQA-derived pairs and synthetic NICE-derived pairs, apply the
Llama-3 chat template, and produce the loss mask so only assistant-turn
tokens contribute to the SFT gradient (system/user tokens are set to -100).

Output is a single JSONL training file consumed by push_to_hub.py.
"""

import json
from pathlib import Path

SYSTEM_PROMPT = "You are a clinical decision support assistant trained on NICE guidelines."

MEDQA_PATH = Path("data/raw/medqa/medqa_normalised.jsonl")
SYNTHETIC_PATH = Path("data/synthetic/qa_pairs.jsonl")
OUTPUT_PATH = Path("data/processed/train.jsonl")


def to_chat_format(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def medqa_to_pair(example: dict) -> dict:
    # TODO: convert the MedQA MCQ + rationale into a free-text
    # question/answer pair before passing to to_chat_format.
    raise NotImplementedError


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    pairs = []
    for example in load_jsonl(MEDQA_PATH):
        pairs.append(to_chat_format(*medqa_to_pair(example)))
    for qa in load_jsonl(SYNTHETIC_PATH):
        pairs.append(to_chat_format(qa["question"], qa["answer"]))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")


if __name__ == "__main__":
    main()
