import os

import torch
from transformers import DistilBertTokenizer
from src.train.commit_dataset import CommitDataset
from src.train.config import Config
from src.train.distilbert_classifier import DistilBertClassifier
from src.train.model_trainer import ModelTrainer
from torch import cuda


class Model:
    """
    Main class for this classifier. It gets the config using Config, and then saves all the necessary elements.
    """

    def __init__(self):
        self.device = "cuda" if cuda.is_available() else "cpu"
        path = os.path.join("src", "train", "train_config.yaml")
        self.config = Config(path)
        self.model = DistilBertClassifier(self.config.get_model_config())
        self.model.to(self.device)
        self.training_arguments = self.config.get_training_arguments()
        dataset_config = self.config.get_dataset_config()
        self.train_dataset = CommitDataset(path=dataset_config["path_to_train_dataset"])
        self.eval_dataset = CommitDataset(path=dataset_config["path_to_eval_dataset"])
        self.trainer = ModelTrainer(
            model=self.model,
            args=self.config.get_training_arguments(),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=DistilBertTokenizer.from_pretrained("distilbert-base-uncased"),
        )

    def train(self) -> None:
        """
        Starts training process.
        """
        self.trainer.train()

    def save_model(self, path) -> None:
        """
        Saves the model to the specified path, so it can be used later.
        """
        torch.save(self.model.state_dict(), path)
