"""DPO on top of the SFT adapter, via Unsloth + TRL's DPOTrainer.

Runs in a Colab/Kaggle GPU session after preference_dataset.py has pushed the
preference set (see sft_train.py's header for the environment/MLflow notes --
the same apply here). Kaggle T4 x2, not P100 (current wheels dropped Pascal
kernels).

Making the SFT adapter the frozen reference: TRL's PEFT integration uses the
model with its adapter disabled as the implicit reference, which would be the
raw base. So the SFT adapter is merged into the base first (16bit merge --
unsloth's merged_4bit path raises, and merged_4bit_forced is buggy -- then
requantized to 4bit on reload); a fresh LoRA on top is the trainable policy,
and the disabled-adapter reference is then the SFT model, not the base.

Checkpoints every save_steps and auto-resumes -- ephemeral notebook sessions
disconnect, and DPO shouldn't restart from zero when they do.
"""

from unsloth import FastLanguageModel, PatchDPOTrainer

PatchDPOTrainer()  # must run before DPOTrainer is constructed

import json
import os

import mlflow
import yaml
from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOConfig, DPOTrainer

from mlflow_utils import start_run

cfg = yaml.safe_load(open("configs/training/dpo_config.yaml"))


def prefetch(adapter_repo: str) -> None:
    """Warm the HF cache for an adapter and its recorded base model.

    transformers' own file-existence probe intermittently 404s unsloth's
    mirror repos from Kaggle; huggingface_hub's downloader is reliable, and
    with a warm cache transformers resolves files locally.
    """
    snapshot_download(adapter_repo)
    with open(hf_hub_download(adapter_repo, "adapter_config.json")) as f:
        base = json.load(f)["base_model_name_or_path"]
    try:
        snapshot_download(base)
    except Exception:
        pass  # base is a local dir (e.g. a merged model) -- nothing to fetch

# Merge SFT into a frozen base, then reload it 4-bit as the DPO base. After this
# the SFT weights live in the (frozen) base, so the fresh LoRA below is the only
# trainable part and the adapter-disabled reference is the SFT model.
prefetch(cfg["sft_adapter_repo"])
sft_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=cfg["sft_adapter_repo"],
    max_seq_length=cfg["max_seq_length"],
    load_in_4bit=True,
)
sft_model.save_pretrained_merged(cfg["merged_sft_dir"], tokenizer, save_method="merged_16bit")
del sft_model

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=cfg["merged_sft_dir"],
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

# The 'id' column is only for resume in preference_dataset.py; DPOTrainer wants
# just prompt/chosen/rejected.
train_dataset = load_dataset(cfg["dpo_dataset_repo"], split="train").remove_columns("id")

with start_run(cfg):
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT: adapter-disabled model (the merged SFT) is the reference
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=DPOConfig(
            output_dir=cfg["output_dir"],
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["grad_accum"],
            learning_rate=cfg["learning_rate"],
            num_train_epochs=cfg["epochs"],
            beta=cfg["beta"],
            rpo_alpha=cfg["rpo_alpha"],
            max_length=cfg["max_length"],
            max_prompt_length=cfg["max_prompt_length"],
            optim="paged_adamw_8bit",
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
    model.push_to_hub(cfg["adapter_hub_repo"], private=True)
    tokenizer.push_to_hub(cfg["adapter_hub_repo"], private=True)
