import os

import pytest
from transformers import TrainingArguments, DistilBertConfig

from src.train.config import Config


@pytest.fixture
def config():
    return Config(os.path.join(os.path.dirname(__file__), "test_config.yaml"))


def test_training_arguments(config):
    training_arguments = config.get_training_arguments()
    expected = TrainingArguments(
        output_dir="training_output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2.0e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
    )
    assert training_arguments == expected


def test_model_config(config):
    model_config = config.get_model_config()
    expected = DistilBertConfig(hidden_dim=768, num_labels=3, dropout=0.1)
    assert model_config == expected


def test_dataset_config(config):
    dataset_config = config.get_dataset_config()
    expected = {
        "path_to_train_dataset": "datasets/dataset1/train.csv",
        "path_to_eval_dataset": "datasets/dataset1/eval.csv",
    }
    assert dataset_config == expected
