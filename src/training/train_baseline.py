import os
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "preprocessing", "processed")
MODEL_NAME = os.environ.get("MODEL_NAME", "roberta-base")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "..", "models", "baseline")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    print("🔹 Cargando datasets procesados")
    train_ds = load_from_disk(os.path.join(DATA_DIR, "train"))
    val_ds = load_from_disk(os.path.join(DATA_DIR, "val"))

    print("🔹 Cargando modelo:", MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    print("🚀 Entrenando modelo baseline")
    trainer.train()

    print("📊 Evaluación final")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("💾 Guardando modelo")
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()
