"""Pull MedQA from HuggingFace and normalise it to question/answer/rationale.

Retained only for differential-diagnosis reasoning patterns, not as a
source of clinical facts (those come from NICE) -- MedQA is US-context.
"""

import json
from pathlib import Path

from datasets import load_dataset

OUTPUT_PATH = Path("data/raw/medqa/medqa_normalised.jsonl")


def normalise(example: dict) -> dict:
    """Map a bigbio/med_qa record to {"question", "correct_answer", "rationale"}.

    TODO: implement once the exact dataset schema has been inspected.
    """
    raise NotImplementedError


def main() -> None:
    dataset = load_dataset("bigbio/med_qa", split="train")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for example in dataset:
            f.write(json.dumps(normalise(example)) + "\n")


if __name__ == "__main__":
    main()
