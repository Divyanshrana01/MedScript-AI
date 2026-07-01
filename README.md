# MedScriptAI

A Clinical Decision Support System combining a QLoRA/DPO fine-tuned Llama-3.1-8B, hybrid BM25 + dense RAG over NICE/NHS clinical guidelines, and a LangGraph multi-agent backend (Supervisor, Clinical QA, Literature Retrieval).

Built for a UK/NHS clinical context (NICE guidelines, BNF terminology) as an MSc AI portfolio project.

> Status: Week 1 — repo scaffolding + data pipeline in progress. Architecture diagram, evaluation results, and demo link will land here as the corresponding build weeks complete.

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
