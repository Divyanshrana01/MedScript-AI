"""Split train/validation, write a dataset card, and push the processed
dataset to the HuggingFace Hub with a version tag.

Training runs reference a specific commit of this dataset, so runs stay
reproducible even as new data versions are added later.
"""

import json
from pathlib import Path

from datasets import Dataset, DatasetDict

PROCESSED_PATH = Path("data/processed/train.jsonl")
HUB_REPO = "your-hf-username/medscriptai-sft"  # TODO: replace with the real repo id
VAL_FRACTION = 0.05

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
    split_idx = int(len(records) * (1 - VAL_FRACTION))
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(records[:split_idx]),
            "validation": Dataset.from_list(records[split_idx:]),
        }
    )
    dataset.push_to_hub(HUB_REPO)
    # TODO: write DATASET_CARD via huggingface_hub.upload_file(..., path_in_repo="README.md")


if __name__ == "__main__":
    main()
