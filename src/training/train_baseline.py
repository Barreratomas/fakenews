from pathlib import Path
import os
import json
import numpy as np
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
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

def main(data_dir, output_dir, model_name="roberta-base", num_labels=2, num_train_epochs=3):
    """
    Entrena el modelo baseline usando datasets procesados.

    Args:
        data_dir (str or Path): Carpeta con datasets procesados (train/val).
        output_dir (str or Path): Carpeta donde guardar modelos y métricas.
        model_name (str): Nombre del modelo preentrenado de HuggingFace.
        num_labels (int): Número de clases.
        num_train_epochs (int): Número de épocas.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔹 Cargando datasets procesados")
    train_ds = load_from_disk(data_dir / "train")
    val_ds = load_from_disk(data_dir / "val")

    train_ds.set_format(type="torch")
    val_ds.set_format(type="torch")

    print("🔹 Cargando modelo:", model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

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
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            seed=42,
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

    # Guardar métricas
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"💾 Modelo y métricas guardados en {output_dir}")
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        model_name=MODEL_NAME,
        num_labels=2,
        num_train_epochs=3
    )
