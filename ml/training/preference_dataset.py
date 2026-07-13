"""Build the DPO preference set: (prompt, chosen, rejected) triples.

chosen is the existing gpt-4o gold answer from the synthetic NICE pairs;
rejected is the SFT adapter sampled hot on the same prompt (on-policy, so DPO's
gradient learns preference rather than model mismatch). A judge then keeps only
pairs where chosen clearly beats rejected, dropping ties where the SFT sample is
already as good as the gold -- those carry no usable preference signal.

Restricted to safety-sensitive prompts (emergency, medication safety, referral)
per the Week 3 scope. Runs in a Colab/Kaggle GPU session (needs the SFT model to
sample); the judge calls OpenAI, so set OPENAI_API_KEY alongside HF_TOKEN.

Resume-aware: every kept/rejected prompt id is recorded, so a crashed or rerun
job never re-samples or re-judges a prompt it already handled.
"""

from unsloth import FastLanguageModel

import json
import os
from pathlib import Path

import yaml
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

cfg = yaml.safe_load(open("configs/training/dpo_config.yaml"))

# Match the system prompt the model was SFT-trained and is served behind, so the
# sampled rejected response comes from the same conditioning as production.
SYSTEM_PROMPT = "You are a clinical decision support assistant trained on NICE guidelines."

PAIRS_PATH = Path(cfg["synthetic_pairs_path"])
OUTPUT_PATH = Path("data/synthetic/dpo_pairs.jsonl")
SAFETY_ASPECTS = set(cfg["safety_aspects"])

JUDGE_SYSTEM_PROMPT = """You are grading two answers to a UK clinical question
for safety and correctness against NICE guidance. Rate each answer 1-10 on
clinical safety, correctness, and appropriate escalation/referral. Reply with
ONLY strict JSON: {"chosen": <int>, "rejected": <int>}. No prose."""

client = OpenAI()


def _iter_qa_rows() -> list[dict]:
    """NICE synthetic QA rows: from the private hub dataset, else the local file.

    Kaggle clones don't carry the gitignored local jsonl, so the hub is the real
    source; the local path only backs dev runs on this machine.
    """
    try:
        return list(load_dataset(cfg["nice_qa_repo"], split="train"))
    except Exception as exc:
        print(f"hub load failed ({exc}); falling back to {PAIRS_PATH}")
        return [json.loads(l) for l in PAIRS_PATH.read_text().splitlines() if l]


def load_safety_prompts() -> list[dict]:
    """QA rows whose task_type is a safety-sensitive aspect.

    Each yields a prompt id (for resume), the question, and the gpt-4o gold
    answer used as the DPO 'chosen'.
    """
    rows = []
    for qa in _iter_qa_rows():
        if qa["metadata"]["task_type"] not in SAFETY_ASPECTS:
            continue
        rows.append({"id": qa["id"], "question": qa["instruction"], "chosen": qa["response"]})
    return rows


def load_done() -> set[str]:
    """Prompt ids already emitted, so reruns skip them."""
    if not OUTPUT_PATH.exists():
        return set()
    return {json.loads(line)["id"] for line in OUTPUT_PATH.read_text().splitlines() if line}


def sample_rejected(model, tokenizer, question: str) -> str:
    """Sample one hot, on-policy response from the SFT model for `question`."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        do_sample=True,
        temperature=cfg["sample_temperature"],
        max_new_tokens=cfg["sample_max_new_tokens"],
    )
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def judge_keeps(question: str, chosen: str, rejected: str) -> bool:
    """True if the judge scores chosen above rejected by at least judge_margin.

    A parse failure or a non-positive margin drops the pair -- better a smaller,
    clean preference set than a noisy one.
    """
    user = (
        f"Question:\n{question}\n\n"
        f"Answer CHOSEN:\n{chosen}\n\nAnswer REJECTED:\n{rejected}"
    )
    resp = client.chat.completions.create(
        model=cfg["judge_model"],
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    try:
        scores = json.loads(resp.choices[0].message.content)
        return int(scores["chosen"]) - int(scores["rejected"]) >= cfg["judge_margin"]
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return False


def to_preference_record(prompt_id: str, question: str, chosen: str, rejected: str) -> dict:
    """TRL conversational preference format: explicit prompt, per-side completions."""
    return {
        "id": prompt_id,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


def main() -> None:
    """Sample rejected answers, judge-gate against gold, append kept triples."""
    prompts = load_safety_prompts()
    done = load_done()
    todo = [p for p in prompts if p["id"] not in done]
    print(f"safety prompts: {len(prompts)} | already done: {len(done)} | todo: {len(todo)}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["sft_adapter_repo"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        for i, p in enumerate(todo, 1):
            rejected = sample_rejected(model, tokenizer, p["question"])
            if not rejected or not judge_keeps(p["question"], p["chosen"], rejected):
                continue
            record = to_preference_record(p["id"], p["question"], p["chosen"], rejected)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # survive a mid-run crash; the file is the resume state
            kept += 1
            if i % 20 == 0:
                print(f"{i}/{len(todo)} processed, {kept} kept")

    records = [json.loads(line) for line in OUTPUT_PATH.read_text().splitlines() if line]
    print(f"preference set: {len(records)} pairs -> pushing to {cfg['dpo_dataset_repo']}")
    # Private, like the SFT dataset: derived from NICE guideline text.
    Dataset.from_list(records).push_to_hub(cfg["dpo_dataset_repo"], private=True)


if __name__ == "__main__":
    main()
