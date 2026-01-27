import torch
from transformers import Trainer

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, num_labels=2, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.num_labels = num_labels

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")
        # Asegurarse de que los pesos estén en el mismo dispositivo que los logits
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        else:
            weight = None
            
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
