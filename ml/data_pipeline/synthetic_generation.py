"""distilabel pipeline: NICE sections -> synthetic instruction/response pairs
via a teacher model (GPT-4o / Claude Sonnet).

Steps: seed extraction -> instruction generation -> response generation ->
quality filtering (HelpSteer2 score + MinHash LSH dedup at 0.85 similarity).

Terminology rules come from the domain config so UK/BNF terms are enforced
at generation time instead of corrected afterward.

Run on a small slice first (~10 guidelines) to catch format bugs before
spending teacher-model API budget on the full corpus.
"""

import json
from pathlib import Path

from configs import load_domain_config

domain = load_domain_config()
rules = domain["terminology_rules"]

TERM_INSTRUCTIONS = "\n".join(
    f"- Always use '{r['use']}', never '{r['not']}'." for r in rules
)

SYSTEM_PROMPT = f"""You are generating clinical QA pairs for a UK NHS context.
Use {domain['guideline_authority']} guidelines as the authority.
Terminology rules:
{TERM_INSTRUCTIONS}
Every answer must cite the relevant guideline."""

SEED_DIR = Path("data/raw/nice")
OUTPUT_PATH = Path("data/synthetic/qa_pairs.jsonl")


def load_seeds(limit: int | None = None) -> list[dict]:
    """Load NICE section JSON files (collect_nice.py output) as generation seeds."""
    seeds = [json.loads(p.read_text()) for p in sorted(SEED_DIR.glob("*.json"))]
    return seeds[:limit] if limit else seeds


def build_pipeline():
    """Build the distilabel Pipeline (seed -> instruction -> response -> filter).

    TODO: wire up the actual distilabel steps described in the module docstring.
    """
    raise NotImplementedError


def main(limit: int | None = 50) -> None:
    """Generate synthetic QA pairs from a slice of NICE seeds (default: 50)."""
    seeds = load_seeds(limit=limit)
    pipeline = build_pipeline()
    results = pipeline.run(seeds)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
