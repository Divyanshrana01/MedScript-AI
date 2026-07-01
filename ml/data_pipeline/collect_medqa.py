"""Pull MedQA (USMLE) from HuggingFace and normalise it to an intermediate
question / correct_answer / rationale format, ready for conversion to
free-text instruction pairs in format_chat_template.py.

Retained for clinical reasoning patterns (differential diagnosis reasoning
transfers across systems even though US and NHS guidelines differ) -- not
used as a source of UK-specific facts.
"""

import json
from pathlib import Path

from datasets import load_dataset

OUTPUT_PATH = Path("data/raw/medqa/medqa_normalised.jsonl")


def normalise(example: dict) -> dict:
    # TODO: map the bigbio/med_qa schema (question, options, answer_idx, ...)
    # to {"question": ..., "correct_answer": ..., "rationale": ...}.
    raise NotImplementedError


def main() -> None:
    dataset = load_dataset("bigbio/med_qa", split="train")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for example in dataset:
            f.write(json.dumps(normalise(example)) + "\n")


if __name__ == "__main__":
    main()
