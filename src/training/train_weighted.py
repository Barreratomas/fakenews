import os
import sys
import json
import numpy as np

# Deshabilitar WandB para evitar bloqueos en Kaggle
os.environ["WANDB_DISABLED"] = "true"

from pathlib import Path
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from src.training.utils.metrics import compute_metrics, compute_class_weights
from src.training.utils.trainer_utils import WeightedTrainer
from src.config import (
    TRAIN_DATA_DIR, 
    VAL_DATA_DIR, 
    WEIGHTED_MODEL_DIR, 
    DEFAULT_BASE_MODEL_NAME
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main(
    data_dir=None, 
    output_dir=None, 
    model_name=DEFAULT_BASE_MODEL_NAME, 
    num_labels=2, 
    num_train_epochs=10
):
    train_dir = Path(data_dir) / "train" if data_dir else TRAIN_DATA_DIR
    val_dir = Path(data_dir) / "val" if data_dir else VAL_DATA_DIR
    output_dir = Path(output_dir) if output_dir else WEIGHTED_MODEL_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Cargando datasets desde {train_dir} y {val_dir}")
    train_ds = load_from_disk(str(train_dir))
    val_ds = load_from_disk(str(val_dir))

    label_col = "label" if "label" in train_ds.features else "labels"
    invalid = [l for l in train_ds[label_col] if l not in (0, 1)]
    if invalid:
        raise ValueError(f"Etiquetas fuera de rango detectadas: {set(invalid)}. Se requieren valores 0/1.")

    train_ds.set_format(type="torch")
    val_ds.set_format(type="torch")

    # Calcular pesos
    logger.info("Calculando pesos de clases...")
    train_labels = np.array(train_ds[label_col])
    class_weights = compute_class_weights(train_labels, num_labels=num_labels)
    logger.info(f"Pesos calculados: {class_weights}")

    logger.info(f"Cargando modelo base: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Forzar uso de GPU si está disponible
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info(f"Modelo movido a GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("GPU no detectada. Entrenando en CPU.")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=2,
        dataloader_num_workers=4,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        num_labels=num_labels,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    logger.info("Iniciando entrenamiento ponderado...")
    trainer.train()

    logger.info("Evaluando modelo...")
    metrics = trainer.evaluate()
    logger.info(f"Métricas finales: {metrics}")

    trainer.save_model(str(output_dir))
    
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(metrics, f)
        
    logger.info(f"Modelo guardado en {output_dir}")

if __name__ == "__main__":
    main()
