"""
Download the MedQA (USMLE) dataset from Hugging Face and convert it into
MedScriptAI's unified dataset schema.

The normalized dataset serves as an intermediate representation before:
    1. Synthetic instruction generation
    2. Chat template formatting
    3. QLoRA fine-tuning

Output format:
    JSON Lines (.jsonl)

Each line represents one clinical question.
"""

import json
from pathlib import Path

from datasets import load_dataset

# ---------------------------------------------------------------------
# Dataset Configuration
# ---------------------------------------------------------------------

# Hugging Face dataset repository
DATASET_REPO = "GBaker/MedQA-USMLE-4-options-hf"

# Destination of the normalized dataset
OUTPUT_PATH = Path("data/raw/medqa/medqa_normalised.jsonl")


def normalise(example: dict) -> dict:
    """
    Convert a single raw MedQA example into the MedScriptAI dataset schema.

    The original MedQA dataset stores the clinical scenario, question,
    answer options, and correct label using dataset-specific field names
    (sent1, sent2, ending0-3, label).

    This function converts those fields into a cleaner, standardized
    representation that will be shared across all MedScriptAI datasets.

    Args:
        example:
            A single MedQA sample from the Hugging Face dataset.

    Returns:
        dict:
            A normalized MedScriptAI record.
    """

    # -------------------------------------------------------------
    # Collect all answer choices into a single list.
    #
    # Original dataset:
    #   ending0
    #   ending1
    #   ending2
    #   ending3
    #
    # Becomes:
    #   options = [...]
    # -------------------------------------------------------------
    options = [
        example["ending0"],
        example["ending1"],
        example["ending2"],
        example["ending3"],
    ]

    # Index (0-3) of the correct option.
    correct_index = int(example["label"])

    # -------------------------------------------------------------
    # Return the unified MedScriptAI schema.
    #
    # This schema is intentionally independent of MedQA so future
    # datasets (NICE, PLAB, PubMedQA, etc.) can share the same format.
    # -------------------------------------------------------------
    return {

        # =========================================================
        # Identification
        # =========================================================
        "id": example["id"],

        # =========================================================
        # Clinical Question
        # =========================================================
        # Patient presentation / case description.
        "clinical_scenario": example["sent1"].strip(),

        # The actual clinical question being asked.
        "question": example["sent2"].strip(),

        # Convenience field used by downstream stages.
        # Avoids repeatedly concatenating scenario + question.
        "prompt": (
            f"{example['sent1'].strip()}\n\n"
            f"{example['sent2'].strip()}"
        ),

        # =========================================================
        # Multiple Choice Information
        # =========================================================
        "options": options,

        # Index of the correct answer.
        "correct_option_index": correct_index,

        # Human-readable correct answer.
        "correct_answer": options[correct_index],

        # =========================================================
        # Explanation
        # =========================================================
        # MedQA does not provide rationales.
        # This field is reserved for future synthetic generation.
        "rationale": None,

        # =========================================================
        # Dataset Metadata
        # =========================================================
        "metadata": {

            # Dataset origin
            "source": "MedQA",

            # Hugging Face repository
            "dataset": DATASET_REPO,

            # Medical terminology follows US conventions.
            "country": "US",

            # Primary reasoning capability represented by this sample.
            "task_type": "clinical_reasoning",

            # Reserved for future enrichment.
            "specialty": None,
            "difficulty": None,
            "tags": [],
        },
    }


def main() -> None:
    """
    Download the MedQA training split, normalize every sample,
    and write the result as a JSONL dataset.
    """

    # Download the dataset from Hugging Face.
    dataset = load_dataset(DATASET_REPO, split="train")

    # Create the output directory if it does not already exist.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Write one JSON object per line.
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:

        for example in dataset:
            normalized_example = normalise(example)

            file.write(
                json.dumps(
                    normalized_example,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Successfully wrote {len(dataset)} samples to:")
    print(f"  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()