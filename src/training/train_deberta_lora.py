import os
import sys
import numpy as np

# Deshabilitar WandB para evitar bloqueos en Kaggle
os.environ["WANDB_DISABLED"] = "true"

import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    AutoTokenizer,
    DebertaV2Tokenizer
)
from peft import LoraConfig, get_peft_model, TaskType

from src.config import (
    TRAIN_DATA_DIR,
    MODELS_DIR,
    DEFAULT_BASE_MODEL_NAME
)
from src.utils.logger import get_logger
from src.training.utils.metrics import compute_metrics, compute_class_weights
from src.training.utils.trainer_utils import WeightedTrainer

logger = get_logger(__name__)

def main(
    data_dir: Path = TRAIN_DATA_DIR, 
    output_dir: Path = MODELS_DIR / "deberta_lora", 
    model_name: str = "microsoft/deberta-v3-base", 
    num_labels: int = 2, 
    num_train_epochs: int = 3
):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Cargando datos desde {data_dir}")
    try:
        train_ds = load_from_disk(str(data_dir / "train"))
        val_ds = load_from_disk(str(data_dir / "val"))
    except FileNotFoundError:
        logger.error(f"No se encontraron datos en {data_dir}. Ejecuta el preprocesamiento primero.")
        sys.exit(1)

    train_ds.set_format(type="torch")
    val_ds.set_format(type="torch")

    logger.info("Calculando pesos de clase...")
    class_weights = compute_class_weights(train_ds["labels"], num_labels=num_labels)
    logger.info(f"Pesos de clase: {class_weights}")

    logger.info(f"Cargando tokenizer: {model_name}")
    try:
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.warning(f"Error cargando DebertaV2Tokenizer: {e}. Intentando AutoTokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    logger.info(f"Cargando modelo base: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.config.id2label = {0: "FAKE", 1: "REAL"}
    model.config.label2id = {"FAKE": 0, "REAL": 1}

    logger.info("Aplicando LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["query_proj", "key_proj", "value_proj"],
        modules_to_save=["pooler", "classifier"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=4e-5,  # Slightly higher for LoRA
        per_device_train_batch_size=16, # Increased batch size
        per_device_eval_batch_size=16,
        num_train_epochs=10, # Increased epochs
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2, # Keep only best 2 checkpoints
        push_to_hub=False,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        num_labels=num_labels,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # Stop if no improvement for 3 epochs
    )

    logger.info("Iniciando entrenamiento...")
    trainer.train()
    
    logger.info(f"Guardando modelo en {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

if __name__ == "__main__":
    main()
