from transformers import Trainer
from torch.nn import BCEWithLogitsLoss


class ModelTrainer(Trainer):
    """
    Subclass of transformers.Trainer with overwritten loss function (Because Trainer can't be used in our case without this modification.).
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = BCEWithLogitsLoss()(outputs, labels)
        return (loss, outputs) if return_outputs else loss
