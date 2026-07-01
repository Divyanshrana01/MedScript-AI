"""Merge MedQA-derived and synthetic NICE-derived pairs into one Llama-3
chat-formatted training file.

Matching Llama-3's own chat template (rather than a custom format) matters
because the model already learned to recognise this turn structure during
pretraining. Loss masking (assistant-only tokens) happens downstream in
ml/training/sft_train.py, not here.
"""

import json
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


def medqa_to_pair(example: dict) -> dict:
    """Convert a normalised MedQA record into a free-text (question, answer) pair.

    TODO: rephrase the MCQ stem as an open question and compose the answer
    from correct_answer + rationale as prose, not "the answer is B".
    """
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
