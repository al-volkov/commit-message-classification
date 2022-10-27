import os

import numpy as np
import torch
import yaml
from torch import cuda
from torch.utils.data import DataLoader
from src.analyze.common import get_model
from src.metrics.compute_basic_metrics import compute_basic_metrics
from src.train.commit_dataset import CommitDataset
from src.train.distilbert_classifier import DistilBertClassifier


def get_dataloader(path: str) -> DataLoader:
    """
    Creates CommitDataset from .csv file and produces DataLoader from it.

    :param path: Path to .csv file with dataset.
    """
    test_dataset = CommitDataset(path=path)
    test_params = {
        "batch_size": 8,
        "shuffle": True,
    }

    return DataLoader(test_dataset, **test_params)


def get_comparable(dataloader: DataLoader, model: DistilBertClassifier):
    """Returns predictions and targets

    This function is used for creating two arrays: predictions and targets. It runs through the dataset and for each
    index stores two values: predictions for commit text and actual labels.

    :param dataloader: DataLoader created from test dataset.
    :param model: Trained model.
    """
    device = "cuda" if cuda.is_available() else "cpu"
    model.eval()
    targets = []
    predictions = []
    with torch.no_grad():
        for _, data in enumerate(dataloader, 0):
            input_ids = data["input_ids"].to(device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(device, dtype=torch.long)
            target = data["labels"].to(device, dtype=torch.float)
            prediction = model(input_ids=input_ids, attention_mask=attention_mask)
            targets.extend(target.cpu().detach().numpy().tolist())
            predictions.extend(torch.sigmoid(prediction).cpu().detach().numpy().tolist())
    return np.array(predictions) >= 0.5, targets


if __name__ == "__main__":
    with open(os.path.join("src", "analyze", "analyze.yaml")) as file:
        config = yaml.safe_load(file)
    config = config["dataset1"]
    model = get_model(os.path.join("src", "train", "train_config.yaml"), config["path_to_model"])
    dataloader = get_dataloader(config["path_to_test_dataset"])
    predictions, targets = get_comparable(dataloader, model)
    result = compute_basic_metrics(predictions, targets)
    print(result)
    for key in result:
        print(f"{key} = {result[key]}")
