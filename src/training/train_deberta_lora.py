from pathlib import Path
import os
import sys
import json
import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    AutoTokenizer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DebertaV2Tokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "preprocessing", "processed")
MODEL_NAME = os.environ.get("MODEL_NAME", "microsoft/deberta-v3-base")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "..", "models", "deberta_lora")


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
    weights = total / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, num_labels=2, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.num_labels = num_labels

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main(data_dir, output_dir, model_name="microsoft/deberta-v3-base", num_labels=2, num_train_epochs=3):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = load_from_disk(data_dir / "train")
    val_ds = load_from_disk(data_dir / "val")
    train_ds.set_format(type="torch")
    val_ds.set_format(type="torch")

    class_weights = compute_class_weights(train_ds["labels"], num_labels=num_labels)

    try:
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.config.id2label = {0: "FAKE", 1: "REAL"}
    model.config.label2id = {"FAKE": 0, "REAL": 1}

    lora_applied = False
    try:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=["query_proj", "key_proj", "value_proj"],
        )
        model = get_peft_model(model, lora_config)
        lora_applied = True
    except Exception:
        lora_applied = False

    try:
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=100,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=num_train_epochs,
            learning_rate=2e-5,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            warmup_ratio=0.1,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            seed=42,
            report_to="none"
        )
    except TypeError:
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=100,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=num_train_epochs,
            learning_rate=2e-5,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            warmup_ratio=0.1,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            seed=42,
            report_to="none"
        )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        num_labels=num_labels,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    metrics = trainer.evaluate()
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if lora_applied:
        try:
            model.save_pretrained(output_dir)
        except Exception:
            pass
    else:
        trainer.save_model(output_dir)


if __name__ == "__main__":
    main(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        model_name=MODEL_NAME,
        num_labels=2,
        num_train_epochs=3
    )
