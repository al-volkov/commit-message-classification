from torch import nn
from transformers import DistilBertModel


class DistilBertClassifier(nn.Module):
    """
    Subclass of nn.Module (base class for pytorch models). Based on DistilBert. Here one more layer is added with dropout and ReLLU, and on the last layer there are 3 neurons.
    """

    def __init__(self, config):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        last_hidden_state = self.distilbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        output = self.pre_classifier(last_hidden_state[:, 0])
        output = nn.ReLU()(output)
        output = self.dropout(output)
        return self.classifier(output)
