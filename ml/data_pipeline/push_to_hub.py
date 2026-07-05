"""Split train/validation, write a dataset card, and push the processed
dataset to the HuggingFace Hub with a version tag.

Training runs reference a specific commit of this dataset, so runs stay
reproducible even as new data versions are added later.
"""

import io
import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

PROCESSED_PATH = Path("data/processed/train.jsonl")
HUB_REPO = "Divyansh619/medscriptai-sft"
VAL_FRACTION = 0.05
SEED = 42

DATASET_CARD = """---
license: cc-by-4.0
---
# MedScriptAI SFT Dataset

UK/NHS-context clinical instruction-response pairs, synthesised from NICE
guidelines plus MedQA-derived reasoning pairs. Formatted with the Llama-3
chat template for QLoRA SFT.
"""


def load_records() -> list[dict]:
    return [json.loads(line) for line in PROCESSED_PATH.read_text().splitlines() if line]


def main() -> None:
    """Split, wrap, and push the dataset to the HuggingFace Hub.

    Requires an authenticated HF session (HF_TOKEN, see .env.example) with
    write access to HUB_REPO.
    """
    records = load_records()
    # train.jsonl is ordered MedQA-then-synthetic; shuffle so the validation
    # split isn't drawn from a single source.
    random.Random(SEED).shuffle(records)
    split_idx = int(len(records) * (1 - VAL_FRACTION))
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(records[:split_idx]),
            "validation": Dataset.from_list(records[split_idx:]),
        }
    )
    # Private: the synthetic split is derived from NICE guideline text, whose
    # licence doesn't clearly permit public redistribution.
    dataset.push_to_hub(HUB_REPO, private=True)
    HfApi().upload_file(
        path_or_fileobj=io.BytesIO(DATASET_CARD.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=HUB_REPO,
        repo_type="dataset",
    )


if __name__ == "__main__":
    main()
