# MedScriptAI

A Clinical Decision Support System combining a QLoRA/DPO fine-tuned Llama-3.1-8B, hybrid BM25 + dense RAG over NICE/NHS clinical guidelines, and a LangGraph multi-agent backend (Supervisor, Clinical QA, Literature Retrieval).

Built for a UK/NHS clinical context (NICE guidelines, BNF terminology).

> Status: Data pipeline + QLoRA SFT complete. DPO scaffolding written, training run pending. Architecture diagram and demo link will land here as later stages complete.

## Progress

- **Data pipeline** — 10,768-row SFT dataset (590 synthetic NICE QA pairs + 10,178 MedQA, 95/5 train/val split) live on HF Hub: [`Divyansh619/medscriptai-sft`](https://huggingface.co/datasets/Divyansh619/medscriptai-sft) (private).
- **SFT (QLoRA)** — `meta-llama/Llama-3.1-8B-Instruct`, LoRA rank 16, 2 epochs, trained via Unsloth + TRL on Kaggle T4. `train_loss: 1.067`. Adapter on HF Hub: [`Divyansh619/medscriptai-sft-adapter`](https://huggingface.co/Divyansh619/medscriptai-sft-adapter) (private).
- **Eval** (held-out validation split): `rougeL: 0.2315`, `bertscore_f1: 0.9003`.
- **DPO (Week 3)** — preference set built from safety-sensitive NICE prompts (gpt-4o gold as `chosen`, SFT hot-sampled `rejected`, judge-gated), then DPO on top of the SFT adapter via TRL's `DPOTrainer`. Scripts (`preference_dataset.py`, `dpo_train.py`, `eval_dpo.py`) ready; Kaggle training run + win-rate/safety-rate eval pending.

## Repository layout

```
configs/        # domain + training + RAG configuration (YAML, no hardcoded prompts/values in code)
ml/             # data pipeline, training (QLoRA + DPO), evaluation
backend/        # FastAPI + LangGraph agent service (added in later weeks)
frontend/       # React + Tailwind client (added in later weeks)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
cp .env.example .env   # fill in HF_TOKEN, OPENAI_API_KEY, etc.
```
