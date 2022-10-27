import torch
from torch import cuda

from src.train.config import Config
from src.train.distilbert_classifier import DistilBertClassifier


def get_model(path_to_config: str, path_to_model: str) -> DistilBertClassifier:
    """

    Returns DistilBertClassifier created from .pth file and configuration file.
    :param path_to_config: Path to train_config.yaml (or other config, that can be parsed by Config class).
    :param path_to_model: Path to .pth file.
    :return: Ready to use model.
    """
    config = Config(path_to_config).get_model_config()
    model = DistilBertClassifier(config)
    device = "cuda" if cuda.is_available() else "cpu"
    model.to(device)
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device(device)))
    model.to(device)

    return model
