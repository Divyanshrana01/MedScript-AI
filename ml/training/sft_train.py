"""QLoRA SFT on the pushed dataset (Divyansh619/medscriptai-sft), via Unsloth + TRL.

Meant to run inside a Colab/Kaggle GPU session, not the local venv: clone the
repo there, `pip install unsloth trl peft bitsandbytes datasets mlflow
python-dotenv` fresh (see Implementation Workflow doc, Chapter 3), then run
this from the repo root so the relative config path resolves.

NOTE: MLFLOW_TRACKING_URI defaults to http://localhost:5000, which is only
reachable from this machine, not a Colab VM. Until a hosted/tunnelled MLflow
is set up, point MLFLOW_TRACKING_URI at something the notebook can reach
(e.g. a Databricks Community Edition tracking server) before running there.

Run a tiny sanity pass first (small `epochs`/dataset slice on a free Kaggle
P100) before the full run on paid Colab Pro compute -- a broken config caught
free costs nothing; the same break two hours into a paid A100 run doesn't.
"""

from unsloth import FastLanguageModel

import os

import mlflow
import yaml
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from mlflow_utils import start_run

cfg = yaml.safe_load(open("configs/training/sft_config.yaml"))

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=cfg["base_model"],
    max_seq_length=cfg["max_seq_length"],
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=cfg["lora_rank"],
    lora_alpha=cfg["lora_alpha"],
    target_modules=cfg["target_modules"],
    lora_dropout=cfg["lora_dropout"],
)

train_dataset = load_dataset(cfg["dataset_repo"], split="train")
eval_dataset = load_dataset(cfg["dataset_repo"], split="validation")


def formatting_prompts_func(examples):
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in examples["messages"]
    ]
    return {"text": texts}


train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

with start_run(cfg):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir=cfg["output_dir"],
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["grad_accum"],
            learning_rate=cfg["learning_rate"],
            num_train_epochs=cfg["epochs"],
            optim="paged_adamw_8bit",
            eval_strategy="epoch",
            save_strategy="steps",
            save_steps=cfg["save_steps"],
            save_total_limit=cfg["save_total_limit"],
        ),
    )
    last_checkpoint = (
        get_last_checkpoint(cfg["output_dir"]) if os.path.isdir(cfg["output_dir"]) else None
    )
    trainer.train(resume_from_checkpoint=last_checkpoint)
    model.save_pretrained(cfg["output_dir"])
    mlflow.log_artifacts(cfg["output_dir"])
