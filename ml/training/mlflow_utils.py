"""Shared MLflow setup for sft_train.py and (Week 3) dpo_train.py.

Centralised so both scripts log to the same tracking URI and experiment
naming, and so switching trackers later (local file store -> hosted MLflow)
is a one-file change.
"""

import os
from contextlib import contextmanager

import mlflow
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))


@contextmanager
def start_run(cfg: dict):
    """Set the experiment by name (created on first use) and log cfg as params.

    Using set_experiment(name) instead of start_run(experiment_id=...) avoids
    needing a numeric MLflow experiment id in the YAML config -- the config
    only ever names the experiment, MLflow resolves/creates the id.
    """
    mlflow.set_experiment(cfg["mlflow_experiment"])
    with mlflow.start_run() as run:
        mlflow.log_params(cfg)
        yield run
