"""Generate fresh safety-eval questions for eval_dpo.py and push the eval split.

The synthetic pipeline used 3 of 7 clinical aspects per NICE section, so every
section still has unused aspects. Questions generated from an unused
(section, safety-aspect) combo are grounded in the same corpus but were never
part of SFT or DPO training -- the earlier eval reused prompts whose gold
answers SFT had memorised, which made the SFT baseline unbeatable by
construction.

CPU-only, runs locally: needs data/raw/nice/*.json (local corpus), plus
OPENAI_API_KEY and HF_TOKEN in .env.
"""

import json
import random
from pathlib import Path

import yaml
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

cfg = yaml.safe_load(open("configs/training/dpo_config.yaml"))

SYSTEM_PROMPT = "You are a clinical decision support assistant trained on NICE guidelines."

# Same brief as the synthetic pipeline's question phase, so eval questions look
# like the training distribution -- just from combos training never touched.
QUESTION_SYSTEM_PROMPT = """You are writing exam-style clinical questions for a
UK doctor. Given a NICE guideline excerpt and a clinical aspect to focus on,
write ONE realistic question a clinician would ask that this excerpt directly
answers, focused on that aspect. If the excerpt does not cover the aspect, ask
about the excerpt's main clinical point instead. Output only the question, no
preamble."""

SEED_DIR = Path("data/raw/nice")
SAFETY_ASPECTS = set(cfg["safety_aspects"])

client = OpenAI()


def used_combos() -> set[tuple[str, str, str]]:
    """(guideline_id, section, aspect) triples already used by the synthetic set."""
    rows = load_dataset(cfg["nice_qa_repo"], split="train")
    return {(r["metadata"]["guideline_id"], r["metadata"]["section"], r["metadata"]["task_type"]) for r in rows}


def candidate_combos() -> list[dict]:
    """Section x safety-aspect combos the synthetic pipeline never used."""
    used = used_combos()
    candidates = []
    for path in sorted(SEED_DIR.glob("*.json")):
        seed = json.loads(path.read_text(encoding="utf-8"))
        for aspect in sorted(SAFETY_ASPECTS):
            if (seed["guideline_id"], seed["section"], aspect) not in used:
                candidates.append({**seed, "aspect": aspect})
    return candidates


def generate_question(seed: dict) -> str:
    resp = client.chat.completions.create(
        model=cfg["judge_model"],
        messages=[
            {"role": "system", "content": QUESTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Aspect: {seed['aspect']}\n\nExcerpt:\n{seed['text']}"},
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def main() -> None:
    candidates = candidate_combos()
    picked = random.Random(42).sample(candidates, k=min(cfg["eval_size"], len(candidates)))
    print(f"unused safety combos: {len(candidates)} | generating {len(picked)} questions")

    records = []
    for i, seed in enumerate(picked, 1):
        question = generate_question(seed)
        if not question:
            continue
        records.append({
            "id": f"eval_{seed['guideline_id']}_{seed['section']}_{i:03d}",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        })
        if i % 10 == 0:
            print(f"{i}/{len(picked)}")

    print(f"pushing {len(records)} eval questions -> {cfg['dpo_eval_repo']} (split=eval)")
    Dataset.from_list(records).push_to_hub(cfg["dpo_eval_repo"], split="eval", private=True)


if __name__ == "__main__":
    main()
