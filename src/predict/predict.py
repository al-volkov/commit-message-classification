import numpy as np
import torch
from torch import cuda
from transformers import DistilBertTokenizer

from src.train.distilbert_classifier import DistilBertClassifier
from typing import List


def get_predictions(model: DistilBertClassifier, tokenizer: DistilBertTokenizer, arr: List[str], max_length=512):
    """

    Takes model, tokenizer for this model and max_length parameter and produces predictions for string in array.

    :param model: Pretrained model.
    :param tokenizer: Pretrained tokenizer.
    :param arr: List with commit messages.
    :param max_length: Max length of sentence.
    :return: List with predictions for each message.
    """
    device = "cuda" if cuda.is_available() else "cpu"
    model.eval()
    predictions = []
    with torch.no_grad():
        for text in arr:
            tokenized_text = tokenizer(
                text,
                None,
                max_length=max_length,
                add_special_tokens=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = tokenized_text["input_ids"].to(device, dtype=torch.long)
            attention_mask = tokenized_text["attention_mask"].to(device, dtype=torch.long)
            prediction = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions.extend(torch.sigmoid(prediction).cpu().detach().numpy().tolist())

    return predictions
