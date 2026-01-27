import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def compute_class_weights(labels, num_labels=2):
    labels = np.array(labels, dtype=int)
    counts = np.bincount(labels, minlength=num_labels)
    total = counts.sum()
    if total == 0:
        return torch.ones(num_labels)
    weights = total / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float)
