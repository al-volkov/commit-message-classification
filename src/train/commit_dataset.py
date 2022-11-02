import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import DistilBertTokenizer
from typing import Optional


class CommitDataset(Dataset):
    """
    Subclass of torch.utils.data.Dataset, that used for creating datasets from .csv files. The .csv file is supposed to contain only the message text and three label columns.
    """

    def __init__(
        self, path: str, text_column_name="text", tokenizer: Optional[DistilBertTokenizer] = None, max_length=512
    ):
        """
        :param path: Path to csv file with data.
        :param text_column_name: Name of column that contains text of messages.
        :param tokenizer: Tokenized that will be used to tokenize text of messages. If None, DistilBertTokenizer.from("distilbert-base-uncased") will be used.
        :param max_length: Max length of sentence.
        """
        if not tokenizer:
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        df = pd.read_csv(path, index_col=0).sample(frac=1, random_state=42).reset_index(drop=True)
        self.commits = df[text_column_name].to_numpy()
        self.labels = df[["Corrective", "Adaptive", "Perfective"]].astype(float).to_numpy()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index: int) -> dict:
        tokenized_commit = self.tokenizer(
            self.commits[index],
            None,
            max_length=self.max_length,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = torch.from_numpy(self.labels[index])
        return {
            "input_ids": tokenized_commit["input_ids"][0],
            "attention_mask": tokenized_commit["attention_mask"][0],
            "labels": labels,
        }

    def get_labels(self):
        """
        :return: Labels from this dataset.
        """
        return self.labels

    def get_text(self):
        """
        :return: Text (commit messages) from this dataset.
        """
        return self.commits

    def split(self, lengths):
        return random_split(self, lengths)

    def __len__(self):
        return len(self.commits)
