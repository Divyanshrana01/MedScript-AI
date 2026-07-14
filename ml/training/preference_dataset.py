"""Build the DPO preference set: fully on-policy (prompt, chosen, rejected).

For each safety-sensitive prompt, sample n_samples from the SFT adapter in one
batched generate, have a judge score each 1-10, and pair the best (chosen)
against the worst (rejected) -- keeping the pair only when the margin is >=
pair_margin, so every surviving pair carries real contrast. Both sides come
from the policy itself: earlier gold-as-chosen builds pushed probability away
from the model's own outputs without a reachable target (the gold was far
off-policy) and made the model worse with every step.

Runs in a Colab/Kaggle GPU session (needs the SFT model to sample); the judge
calls OpenAI, so set OPENAI_API_KEY alongside HF_TOKEN.

Resume-aware: every handled prompt id is recorded in the output jsonl, so a
crashed or rerun job never re-samples or re-judges a prompt it already did.
"""

from unsloth import FastLanguageModel

import json
import os
from pathlib import Path

import yaml
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download
from openai import OpenAI

load_dotenv()

cfg = yaml.safe_load(open("configs/training/dpo_config.yaml"))


def prefetch(adapter_repo: str) -> None:
    """Warm the HF cache for an adapter and its recorded base model.

    transformers' own file-existence probe intermittently 404s unsloth's
    mirror repos from Kaggle ("does not appear to have a file named
    model.safetensors" for a repo that has one); huggingface_hub's downloader
    is reliable, and with a warm cache transformers resolves files locally.
    """
    snapshot_download(adapter_repo)
    with open(hf_hub_download(adapter_repo, "adapter_config.json")) as f:
        base = json.load(f)["base_model_name_or_path"]
    try:
        snapshot_download(base)
    except Exception:
        pass  # base is a local dir (e.g. a merged model) -- nothing to fetch

# Match the system prompt the model was SFT-trained and is served behind, so
# sampling happens under the same conditioning as production.
SYSTEM_PROMPT = "You are a clinical decision support assistant trained on NICE guidelines."

PAIRS_PATH = Path(cfg["synthetic_pairs_path"])
OUTPUT_PATH = Path("data/synthetic/dpo_pairs_onpolicy.jsonl")
SAFETY_ASPECTS = set(cfg["safety_aspects"])

JUDGE_SYSTEM_PROMPT = """You are grading candidate answers to a UK clinical
question on clinical safety, correctness, and appropriate escalation/referral
against NICE guidance. Score each answer 1-10. Reply with ONLY strict JSON:
{"scores": [<int>, ...]} -- one score per answer, in the order given."""

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
    """QA rows whose task_type is a safety-sensitive aspect (id + question only)."""
    rows = []
    for qa in _iter_qa_rows():
        if qa["metadata"]["task_type"] not in SAFETY_ASPECTS:
            continue
        rows.append({"id": qa["id"], "question": qa["instruction"]})
    return rows


def load_done() -> set[str]:
    """Prompt ids already handled (kept or dropped), so reruns skip them."""
    if not OUTPUT_PATH.exists():
        return set()
    return {json.loads(line)["id"] for line in OUTPUT_PATH.read_text().splitlines() if line}


def sample_candidates(model, tokenizer, question: str) -> list[str]:
    """One batched generate returning n_samples on-policy candidate answers."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=cfg["sample_temperature"],
        max_new_tokens=cfg["sample_max_new_tokens"],
        num_return_sequences=cfg["n_samples"],
    )
    prompt_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(out[prompt_len:], skip_special_tokens=True).strip() for out in outputs
    ]


def judge_scores(question: str, candidates: list[str]) -> list[int] | None:
    """Score every candidate in one judge call; None if the reply can't be parsed."""
    numbered = "\n\n".join(f"Answer {i + 1}:\n{c}" for i, c in enumerate(candidates))
    resp = client.chat.completions.create(
        model=cfg["judge_model"],
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question:\n{question}\n\n{numbered}"},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    try:
        scores = json.loads(resp.choices[0].message.content)["scores"]
        if len(scores) != len(candidates):
            return None
        return [int(s) for s in scores]
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def to_record(prompt_id: str, question: str, chosen: str, rejected: str, margin: int) -> dict:
    """TRL conversational preference format; kept=False rows are resume markers."""
    return {
        "id": prompt_id,
        "kept": True,
        "margin": margin,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


def main() -> None:
    """Sample, judge, pair best-vs-worst, append kept pairs, push the train split."""
    prompts = load_safety_prompts()
    done = load_done()
    todo = [p for p in prompts if p["id"] not in done]
    print(f"safety prompts: {len(prompts)} | already done: {len(done)} | todo: {len(todo)}")

    prefetch(cfg["sft_adapter_repo"])
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
            candidates = sample_candidates(model, tokenizer, p["question"])
            scores = judge_scores(p["question"], candidates)
            if scores is None:
                continue  # unparsable judge reply: leave un-done so a rerun retries
            best, worst = max(range(len(scores)), key=scores.__getitem__), min(
                range(len(scores)), key=scores.__getitem__
            )
            margin = scores[best] - scores[worst]
            if margin >= cfg["pair_margin"] and candidates[best] and candidates[worst]:
                record = to_record(
                    p["id"], p["question"], candidates[best], candidates[worst], margin
                )
                kept += 1
            else:
                # low contrast: record the id so reruns skip it, but train won't see it
                record = {"id": p["id"], "kept": False, "margin": margin}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # survive a mid-run crash; the file is the resume state
            if i % 20 == 0:
                print(f"{i}/{len(todo)} processed, {kept} kept this run")

    records = [json.loads(line) for line in OUTPUT_PATH.read_text().splitlines() if line]
    pairs = [
        {k: r[k] for k in ("id", "prompt", "chosen", "rejected")} for r in records if r["kept"]
    ]
    print(f"preference set: {len(pairs)} pairs from {len(records)} prompts -> pushing train split")
    # Private, like the SFT dataset: derived from NICE guideline text. The eval
    # split is pushed separately by ml/evaluation/gen_eval_questions.py.
    Dataset.from_list(pairs).push_to_hub(cfg["dpo_dataset_repo"], split="train", private=True)


if __name__ == "__main__":
    main()
