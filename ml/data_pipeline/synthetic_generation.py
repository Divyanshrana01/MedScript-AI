"""distilabel pipeline: NICE sections -> synthetic instruction/response pairs
via a teacher model.

Two sequential phases instead of one DAG so the question and answer steps
never share the teacher's tokens-per-minute window at the same time (the
concurrent-window saturation is what caused repeated 429 crashes on tier-1):

  Phase 1: seeds missing a question  -> generate_question -> questions bank
  Phase 2: banked questions missing an answer -> answer -> judge -> filter
           -> appended to the output dataset

Both phases are resume-aware: work already in the bank or the output file is
never regenerated, so a crash or rerun costs nothing extra. Rows whose
generation failed (429 outlasting retries -> None) are dropped by a guard
step before they can reach distilabel's router, which crashes on None-filled
batches.

Teacher/judge default to OpenAI (gpt-4o / gpt-4o-mini). Set
TEACHER_PROVIDER=gemini (and optionally JUDGE_PROVIDER) to run against
Gemini's OpenAI-compatible endpoint on the free tier instead -- needs
GEMINI_API_KEY in .env.

Run on a small slice first (main(limit=10)) to catch format bugs before
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
BANK_PATH = Path("data/synthetic/questions_bank_v2.jsonl")

# Providers are env-switchable so the same pipeline can run on OpenAI or on
# Gemini's OpenAI-compatible endpoint (free tier) when budget is tight.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODELS = {
    "openai": {"teacher": "gpt-4o", "judge": "gpt-4o-mini"},
    "gemini": {"teacher": "gemini-2.5-flash", "judge": "gemini-2.5-flash"},
}
TEACHER_PROVIDER = os.getenv("TEACHER_PROVIDER", "openai")
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", TEACHER_PROVIDER)
TEACHER_MODEL = os.getenv("TEACHER_MODEL", DEFAULT_MODELS[TEACHER_PROVIDER]["teacher"])
JUDGE_MODEL = os.getenv("JUDGE_MODEL", DEFAULT_MODELS[JUDGE_PROVIDER]["judge"])
PROMPT_VERSION = "v2"  # bump whenever the prompts or aspect list change

MINHASH_THRESHOLD = 0.85
MIN_QUALITY_RATING = 4  # UltraFeedback overall-rating is 1-5
MIN_RESPONSE_CHARS = 150
# A response counts as truncated unless it ends like a finished sentence or
# citation, e.g. "... (CG113, section 1.1)" or "... [2011]".
COMPLETE_ENDINGS = (".", "!", "?", ")", "]", '"')

# The org is OpenAI tier 1: 30k tokens/min on gpt-4o. Answer prompts embed the
# full guideline excerpt (up to ~5k tokens), so teacher calls go one at a time.
QUESTION_BATCH_SIZE = 1
ANSWER_BATCH_SIZE = 1
JUDGE_BATCH_SIZE = 5  # the judge model has its own, much higher rate limit


def _make_llm(provider: str, model: str, max_new_tokens: int | None = None) -> OpenAILLM:
    """OpenAILLM against either OpenAI or Gemini's OpenAI-compatible endpoint.

    max_retries raised from the default 6: 429 backoff waits are short, and a
    saturated TPM window can take longer than 6 retries to clear.
    """
    kwargs: dict = {"model": model, "max_retries": 12}
    if max_new_tokens:
        kwargs["generation_kwargs"] = {"max_new_tokens": max_new_tokens}
    if provider == "gemini":
        kwargs["base_url"] = GEMINI_BASE_URL
        kwargs["api_key"] = os.environ["GEMINI_API_KEY"]
    return OpenAILLM(**kwargs)


def _key(row: dict) -> tuple[str, str, str]:
    return (row["guideline_id"], row["section"], row["aspect"])


def load_seeds(limit: int | None = None) -> list[dict]:
    """Load NICE section JSON files (collect_nice.py output) as generation seeds."""
    seeds = [json.loads(p.read_text()) for p in sorted(SEED_DIR.glob("*.json"))]
    return seeds[:limit] if limit else seeds


def expand_seeds(seeds: list[dict]) -> list[dict]:
    """Fan each seed out to PAIRS_PER_SEED rows, one per sampled clinical aspect.

    The RNG stream is consumed per seed in file order, so the seed->aspect
    assignment is stable across runs -- resume filtering relies on that.
    """
    rng = random.Random(42)
    expanded = []
    for seed in seeds:
        for aspect in rng.sample(ASPECTS, k=PAIRS_PER_SEED):
            row = dict(seed)
            row["aspect"] = aspect
            row["question_prompt"] = f"Aspect: {aspect}\n\nExcerpt:\n{seed['text']}"
            expanded.append(row)
    return expanded


def load_bank() -> dict[tuple, dict]:
    """Load the questions bank: every question ever generated, keyed by seed."""
    if not BANK_PATH.exists():
        return {}
    bank = {}
    with open(BANK_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            bank[_key(row)] = row
    return bank


def append_bank(rows: list[dict]) -> None:
    BANK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BANK_PATH, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done() -> tuple[set[tuple], int]:
    """Keys already answered in the output file, plus its row count (for ids)."""
    if not OUTPUT_PATH.exists():
        return set(), 0
    done, count = set(), 0
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)["metadata"]
            done.add((m["guideline_id"], m["section"], m["task_type"]))
            count += 1
    return done, count


def build_instruction(row: dict) -> str:
    """Combine the question with its source excerpt into one grounded prompt --
    keeps the answer citing the actual passage instead of general knowledge.
    """
    return (
        f"Guideline: {row['title']} ({row['guideline_id'].upper()}), "
        f"section {row['section']}\n\n"
        f"Excerpt:\n{row['text']}\n\n"
        f"Question: {row['clinical_question']}\n\n"
        "Answer using only the excerpt above."
    )


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


class DropFailedGenerations(Step):
    """Drop rows whose generation came back None/empty (a 429 that outlasted
    retries). Without this, distilabel's router crashes on None-filled batches
    ("list indices must be integers or slices, not str") and kills the run.
    """

    @property
    def inputs(self) -> "StepColumns":
        return ["generations"]

    @property
    def outputs(self) -> "StepColumns":
        return []

    def process(self, inputs: StepInput) -> "StepOutput":
        kept = [r for r in inputs if r.get("generations") and r["generations"][0]]
        yield kept


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


def _distiset_rows(distiset) -> list[dict]:
    if not distiset:
        return []
    leaf = next(iter(distiset.values()))
    dataset = leaf["default"] if "default" in leaf else next(iter(leaf.values()))
    return list(dataset)


def run_question_phase(todo: list[dict]) -> list[dict]:
    """Generate questions for seeds not yet in the bank; returns bank rows."""
    with Pipeline(name="nice-synthetic-questions") as pipeline:
        load = LoadDataFromDicts(name="load_seeds", data=todo)
        generate_question = TextGeneration(
            name="generate_question",
            llm=_make_llm(TEACHER_PROVIDER, TEACHER_MODEL),
            system_prompt=QUESTION_SYSTEM_PROMPT,
            input_mappings={"instruction": "question_prompt"},
            output_mappings={"generation": "clinical_question"},
            input_batch_size=QUESTION_BATCH_SIZE,
        )
        load >> generate_question

    # use_cache=False: the cache-resume path crashes multiprocess pool startup
    # on Python 3.12.0 ("OSError: handle is closed"). The bank file is our
    # resume mechanism instead.
    rows = _distiset_rows(pipeline.run(use_cache=False))
    banked = []
    for r in rows:
        question = (r.get("clinical_question") or "").strip()
        if not question:
            continue  # failed generation: leave un-banked so a rerun retries it
        banked.append({
            "guideline_id": r["guideline_id"],
            "title": r["title"],
            "section": r["section"],
            "text": r["text"],
            "aspect": r["aspect"],
            "clinical_question": question,
            "question_model": TEACHER_MODEL,
            "salvaged_from": "generated",
        })
    return banked


def run_answer_phase(todo: list[dict]) -> list[dict]:
    """Answer + judge + filter banked questions; returns quality-passing rows."""
    for row in todo:
        row["instruction"] = build_instruction(row)

    with Pipeline(name="nice-synthetic-answers") as pipeline:
        load = LoadDataFromDicts(name="load_questions", data=todo)
        generate_answer = TextGeneration(
            name="generate_answer",
            llm=_make_llm(TEACHER_PROVIDER, TEACHER_MODEL, max_new_tokens=1024),
            system_prompt=SYSTEM_PROMPT,
            num_generations=1,
            group_generations=True,
            output_mappings={"generation": "generations"},
            input_batch_size=ANSWER_BATCH_SIZE,
        )
        drop_failed = DropFailedGenerations(name="drop_failed")
        judge_step = UltraFeedback(
            name="judge",
            aspect="overall-rating",
            llm=_make_llm(JUDGE_PROVIDER, JUDGE_MODEL),
            input_batch_size=JUDGE_BATCH_SIZE,
        )
        dedup = MinHashDedup(
            name="dedup",
            threshold=MINHASH_THRESHOLD,
            input_mappings={"text": "clinical_question"},
        )
        filter_step = FilterLowQuality(name="filter_low_quality")

        load >> generate_answer >> drop_failed >> judge_step >> dedup >> filter_step

    return _distiset_rows(pipeline.run(use_cache=False))


def append_output(rows: list[dict], start_index: int) -> None:
    """Append new QA pairs to the dataset; existing rows are never rewritten."""
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        for i, row in enumerate(rows, start=start_index):
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
                    "question_model": row.get("question_model", TEACHER_MODEL),
                    "judge_model": JUDGE_MODEL,
                    "prompt_version": PROMPT_VERSION,
                    "generated_at": generated_at,
                },
            }, ensure_ascii=False) + "\n")


def main(limit: int | None = None) -> None:
    """Generate synthetic QA pairs from NICE seeds, resuming past work."""
    seeds = expand_seeds(load_seeds(limit=limit))
    bank = load_bank()
    done, next_index = load_done()
    print(
        f"providers: teacher={TEACHER_PROVIDER}/{TEACHER_MODEL} "
        f"judge={JUDGE_PROVIDER}/{JUDGE_MODEL}"
    )
    print(f"seeds expanded: {len(seeds)} | banked questions: {len(bank)} | done rows: {len(done)}")

    todo_questions = [s for s in seeds if _key(s) not in bank and _key(s) not in done]
    if todo_questions:
        print(f"phase 1: generating {len(todo_questions)} questions")
        new_bank_rows = run_question_phase(todo_questions)
        append_bank(new_bank_rows)
        bank.update({_key(r): r for r in new_bank_rows})
        print(f"phase 1: banked {len(new_bank_rows)} new questions")
    else:
        print("phase 1: nothing to do, all questions banked")

    todo_answers = [dict(row) for key, row in bank.items() if key not in done]
    if not todo_answers:
        print("phase 2: nothing to do, all banked questions answered")
        return
    print(f"phase 2: answering {len(todo_answers)} questions")
    kept = run_answer_phase(todo_answers)
    append_output(kept, start_index=next_index)
    print(
        f"phase 2: {len(kept)}/{len(todo_answers)} passed quality; "
        f"dataset now {next_index + len(kept)} rows at {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main(limit=None)
