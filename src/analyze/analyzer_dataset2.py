import os

import numpy as np
import yaml
from torch import cuda
from transformers import DistilBertTokenizer

from src.analyze.common import get_model
from src.predict.predict import get_predictions
from typing import List


def get_final_predictions(commit_messages: List[str], path_to_model: str):
    """
    Produces list with predictions from original messages using trained model.

    :param commit_messages: List with original messages.
    :param path_to_model: Path to pretrained model.
    :return: Two dimensional numpy array (len(commit_messages) x 3) with integer values.
    """
    model = get_model(os.path.join("src", "train", "train_config.yaml"), path_to_model)

    device = "cuda" if cuda.is_available() else "cpu"
    model.to(device)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    predictions = get_predictions(model, tokenizer, commit_messages)
    return (np.array(predictions) >= 0.5).astype(int)


if __name__ == "__main__":
    with open(os.path.join("src", "analyze", "analyze.yaml")) as file:
        config = yaml.safe_load(file)
    config = config["dataset2"]
    with open(config["path_to_dataset"]) as file:
        commit_messages = [x.strip() for x in file]
    predictions = get_final_predictions(commit_messages, config["path_to_model"])
    corrective_count, adaptive_count, perfective_count = 0, 0, 0
    labels_count = [0, 0, 0, 0]
    total_count = {
        (0, 0, 0): 0,
        (0, 0, 1): 0,
        (0, 1, 0): 0,
        (0, 1, 1): 0,
        (1, 0, 0): 0,
        (1, 0, 1): 0,
        (1, 1, 0): 0,
        (1, 1, 1): 0,
    }
    for label in predictions:
        corrective_count += label[0]
        adaptive_count += label[1]
        perfective_count += label[2]
        labels_count[sum(label)] += 1
        total_count[tuple(label)] += 1

    print(corrective_count)
    print(adaptive_count)
    print(perfective_count)
    print(labels_count)
    print(total_count)
