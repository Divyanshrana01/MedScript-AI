"""Score the trained SFT adapter against the held-out validation split.

Runs in the same Colab/Kaggle session as sft_train.py, right after training
(needs the fine-tuned model loaded to generate predictions), not locally --
unlike the rest of ml/evaluation/, which can stay CPU-only. Logs to the same
"fine-tuning" MLflow experiment as sft_train.py so the SFT run and its eval
numbers sit together.

Extra deps beyond training (install alongside unsloth/trl in Colab):
pip install evaluate rouge_score bert_score
"""

import os

import mlflow
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from evaluate import load as load_metric
from unsloth import FastLanguageModel

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

cfg = yaml.safe_load(open("configs/training/sft_config.yaml"))

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=cfg["output_dir"],
    max_seq_length=cfg["max_seq_length"],
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

eval_dataset = load_dataset(cfg["dataset_repo"], split="validation")

rouge = load_metric("rouge")
bertscore = load_metric("bertscore")


def generate(messages: list[dict]) -> str:
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=800)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def main() -> None:
    predictions = [generate(row["messages"]) for row in eval_dataset]
    references = [row["messages"][-1]["content"] for row in eval_dataset]

    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bertscore_scores = bertscore.compute(predictions=predictions, references=references, lang="en")

    mlflow.set_experiment(cfg["mlflow_experiment"])
    with mlflow.start_run():
        mlflow.log_metric("rougeL", rouge_scores["rougeL"])
        mlflow.log_metric("bertscore_f1", sum(bertscore_scores["f1"]) / len(bertscore_scores["f1"]))


if __name__ == "__main__":
    main()
