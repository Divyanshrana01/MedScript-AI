"""distilabel pipeline: NICE sections -> synthetic instruction/response pairs
via a teacher model (GPT-4o).

Steps: seed extraction -> instruction generation -> response generation ->
quality filtering (LLM-judge rating + MinHash dedup at 0.85 similarity).

Terminology rules come from the domain config so UK/BNF terms are enforced
at generation time instead of corrected afterward.

Judge LLM is gpt-4o-mini, not the gpt-4o teacher -- a smaller model grading
a larger one's output is a cheaper, reasonably independent quality signal.

Run on a small slice first (~10 guidelines) to catch format bugs before
spending teacher-model API budget on the full corpus.
"""

import json
import os
import random
import sys

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

# Put repo root on the path so `configs` imports whether run as a script or
# re-imported by distilabel's spawned worker processes.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import nltk
from dotenv import load_dotenv
from distilabel.models import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, MinHashDedup, Step, StepInput
from distilabel.steps.tasks import TextGeneration, UltraFeedback

from configs import load_domain_config

if TYPE_CHECKING:
    from distilabel.typing import StepColumns, StepOutput

load_dotenv()

# MinHashDedup's word tokenizer needs this; cached locally after first run.
nltk.download("punkt_tab", quiet=True)

domain = load_domain_config()
rules = domain["terminology_rules"]

TERM_INSTRUCTIONS = "\n".join(
    f"- Always use '{r['use']}', never '{r['not']}'." for r in rules
)

QUESTION_SYSTEM_PROMPT = """You are writing exam-style clinical questions for a
UK doctor. Given a NICE guideline excerpt and a clinical aspect to focus on,
write ONE realistic question a clinician would ask that this excerpt directly
answers, focused on that aspect. If the excerpt does not cover the aspect, ask
about the excerpt's main clinical point instead. Output only the question, no
preamble."""

# One question per (seed, aspect) so a single guideline section yields several
# distinct pairs -- diagnosis, treatment, counselling etc. -- instead of one
# generic question. Aspects are sampled per seed with a fixed RNG seed so the
# expansion is reproducible.
ASPECTS = [
    "diagnosis and clinical assessment",
    "treatment and clinical management",
    "medication safety and contraindications",
    "follow-up, monitoring and referral decisions",
    "patient education and counselling",
    "investigations",
    "emergency or urgent management",
]
PAIRS_PER_SEED = 3

SYSTEM_PROMPT = f"""You are generating clinical QA pairs for a UK NHS context.
Use {domain['guideline_authority']} guidelines as the authority.
Terminology rules:
{TERM_INSTRUCTIONS}
Every answer must cite the relevant guideline."""

SEED_DIR = Path("data/raw/nice")
OUTPUT_PATH = Path("data/synthetic/qa_pairs.jsonl")

TEACHER_MODEL = "gpt-4o"
JUDGE_MODEL = "gpt-4o-mini"
PROMPT_VERSION = "v2"  # bump whenever the prompts or aspect list change

MINHASH_THRESHOLD = 0.85
MIN_QUALITY_RATING = 4  # UltraFeedback overall-rating is 1-5
MIN_RESPONSE_CHARS = 150
# A response counts as truncated unless it ends like a finished sentence or
# citation, e.g. "... (CG113, section 1.1)" or "... [2011]".
COMPLETE_ENDINGS = (".", "!", "?", ")", "]", '"')

# The org is OpenAI tier 1: 30k tokens/min on gpt-4o, shared by the question
# and answer steps running concurrently. Answer prompts embed the full
# guideline excerpt (up to ~5k tokens), so answers go one at a time; once
# retries exhaust on a 429 the step silently yields None and the row is lost.
QUESTION_BATCH_SIZE = 2
ANSWER_BATCH_SIZE = 1
JUDGE_BATCH_SIZE = 5  # gpt-4o-mini has its own, much higher TPM limit


def load_seeds(limit: int | None = None) -> list[dict]:
    """Load NICE section JSON files (collect_nice.py output) as generation seeds."""
    seeds = [json.loads(p.read_text()) for p in sorted(SEED_DIR.glob("*.json"))]
    return seeds[:limit] if limit else seeds


def expand_seeds(seeds: list[dict]) -> list[dict]:
    """Fan each seed out to PAIRS_PER_SEED rows, one per sampled clinical aspect."""
    rng = random.Random(42)
    expanded = []
    for seed in seeds:
        for aspect in rng.sample(ASPECTS, k=PAIRS_PER_SEED):
            row = dict(seed)
            row["aspect"] = aspect
            row["question_prompt"] = f"Aspect: {aspect}\n\nExcerpt:\n{seed['text']}"
            expanded.append(row)
    return expanded


def _rating_of(row: dict) -> int | None:
    """Pull the single quality rating out of a row, tolerating the judge
    returning None or an empty list when its output couldn't be parsed.
    """
    ratings = row.get("ratings")
    if not ratings:
        return None
    return ratings[0]


def _passes_quality(row: dict) -> bool:
    """A row survives if MinHash kept it, its rating met the threshold, and the
    response is complete, long enough, and actually cites the guideline.
    """
    rating = _rating_of(row)
    generations = row.get("generations")
    if not (
        row.get("keep_row_after_minhash_filtering")
        and generations
        and generations[0]
        and rating is not None
        and rating >= MIN_QUALITY_RATING
    ):
        return False

    response = generations[0].strip()
    cites_guideline = (
        row["guideline_id"].upper() in response.upper() or "NICE" in response.upper()
    )
    return (
        len(response) >= MIN_RESPONSE_CHARS
        and response.endswith(COMPLETE_ENDINGS)
        and cites_guideline
    )


class BuildAnswerPrompt(Step):
    """Combine the generated question with its source excerpt into one grounded
    prompt -- keeps the answer step citing the actual passage instead of the
    model's own general knowledge.
    """

    @property
    def inputs(self) -> "StepColumns":
        return ["guideline_id", "title", "section", "text", "clinical_question"]

    @property
    def outputs(self) -> "StepColumns":
        return ["instruction"]

    def process(self, inputs: StepInput) -> "StepOutput":
        for row in inputs:
            row["instruction"] = (
                f"Guideline: {row['title']} ({row['guideline_id'].upper()}), "
                f"section {row['section']}\n\n"
                f"Excerpt:\n{row['text']}\n\n"
                f"Question: {row['clinical_question']}\n\n"
                "Answer using only the excerpt above."
            )
        yield inputs


class FilterLowQuality(Step):
    """Drop rows that failed MinHash dedup or scored below MIN_QUALITY_RATING.

    MinHashDedup only flags duplicates (keep_row_after_minhash_filtering);
    it doesn't remove them, so this is the step that actually drops rows.
    """

    @property
    def inputs(self) -> "StepColumns":
        return ["keep_row_after_minhash_filtering", "ratings", "guideline_id"]

    @property
    def outputs(self) -> "StepColumns":
        return []

    def process(self, inputs: StepInput) -> "StepOutput":
        kept = [row for row in inputs if _passes_quality(row)]
        yield kept


def build_pipeline(seeds: list[dict]) -> Pipeline:
    """Build the distilabel Pipeline: seed -> question -> answer -> judge -> dedup -> filter."""
    # max_retries raised from the default 6: 429 backoff waits are ~1s, and a
    # saturated 30k TPM window can take longer than 6 retries to clear.
    teacher = OpenAILLM(
        model=TEACHER_MODEL,
        max_retries=12,
        generation_kwargs={"max_new_tokens": 1024},
    )

    # gpt-4o-mini judges gpt-4o's output -- a smaller, cheaper model on the
    # same key. Not fully independent, but the free-tier Groq/Gemini
    # alternatives rate-limit at ~10 req/min, a bottleneck for 600+ rows.
    judge = OpenAILLM(model=JUDGE_MODEL)

    with Pipeline(name="nice-synthetic-qa") as pipeline:
        load_seeds_step = LoadDataFromDicts(name="load_seeds", data=seeds)

        generate_question = TextGeneration(
            name="generate_question",
            llm=teacher,
            system_prompt=QUESTION_SYSTEM_PROMPT,
            input_mappings={"instruction": "question_prompt"},
            output_mappings={"generation": "clinical_question"},
            input_batch_size=QUESTION_BATCH_SIZE,
        )

        build_answer_prompt = BuildAnswerPrompt(name="build_answer_prompt")

        generate_answer = TextGeneration(
            name="generate_answer",
            llm=teacher,
            system_prompt=SYSTEM_PROMPT,
            num_generations=1,
            group_generations=True,
            output_mappings={"generation": "generations"},
            input_batch_size=ANSWER_BATCH_SIZE,
        )

        judge_step = UltraFeedback(
            name="judge",
            aspect="overall-rating",
            llm=judge,
            input_batch_size=JUDGE_BATCH_SIZE,
        )

        dedup = MinHashDedup(
            name="dedup",
            threshold=MINHASH_THRESHOLD,
            input_mappings={"text": "clinical_question"},
        )

        filter_step = FilterLowQuality(name="filter_low_quality")

        (
            load_seeds_step
            >> generate_question
            >> build_answer_prompt
            >> generate_answer
            >> judge_step
            >> dedup
            >> filter_step
        )

    return pipeline


def main(limit: int | None = 50) -> None:
    """Generate synthetic QA pairs from a slice of NICE seeds (default: 50)."""
    seeds = expand_seeds(load_seeds(limit=limit))
    pipeline = build_pipeline(seeds)
    # use_cache=False: the cache-resume path crashes multiprocess pool startup
    # on Python 3.12.0 ("OSError: handle is closed"). Revisit after a Python
    # upgrade -- resume support is worth having back.
    distiset = pipeline.run(use_cache=False)

    if not distiset:
        print("Pipeline produced no output (all rows filtered or a step failed).")
        return

    leaf_dataset = next(iter(distiset.values()))
    rows = leaf_dataset["default"] if "default" in leaf_dataset else next(iter(leaf_dataset.values()))

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            f.write(json.dumps({
                "id": f"nice_{row['guideline_id']}_{i:05d}",
                "instruction": row["clinical_question"],
                "response": row["generations"][0],
                "evidence": row["text"],
                "metadata": {
                    "source": "NICE",
                    "guideline_id": row["guideline_id"],
                    "title": row["title"],
                    "section": row["section"],
                    "task_type": row["aspect"],
                    "quality_rating": _rating_of(row),
                    "teacher_model": TEACHER_MODEL,
                    "judge_model": JUDGE_MODEL,
                    "prompt_version": PROMPT_VERSION,
                    "generated_at": generated_at,
                },
            }, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} synthetic QA pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
