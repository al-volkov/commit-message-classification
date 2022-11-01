import yaml
from transformers import TrainingArguments, DistilBertConfig


class Config:
    """
    Class for parsing training config.
    """

    def __init__(self, path_to_config: str) -> None:
        with open(path_to_config) as file:
            config = yaml.safe_load(file)
        self.config = config

    def get_training_arguments(self) -> TrainingArguments:
        """
        Parses the config and returns a TrainingArguments instance created from the received data.
        """
        config = self.config["training_arguments"]
        return TrainingArguments(
            output_dir=config["output_dir"],
            evaluation_strategy=config["evaluation_strategy"],
            save_strategy=config["save_strategy"],
            learning_rate=config["learning_rate"],
            per_device_train_batch_size=config["per_device_train_batch_size"],
            per_device_eval_batch_size=config["per_device_eval_batch_size"],
            num_train_epochs=config["num_train_epochs"],
            weight_decay=config["weight_decay"],
            load_best_model_at_end=config["load_best_model_at_end"],
        )

    def get_model_config(self) -> DistilBertConfig:
        """
        Parses the config and returns a DistilBertConfig instance created from the received data.
        """
        config = self.config["model_config"]
        return DistilBertConfig(
            hidden_dim=config["hidden_dim"], num_labels=config["num_labels"], dropout=config["dropout"]
        )

    def get_dataset_config(self) -> dict:
        """

        :return: Dictionary that contains paths to datasets.
        """
        return self.config["dataset"]
