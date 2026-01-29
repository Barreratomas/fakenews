import os
import json
import numpy as np

# Deshabilitar WandB para evitar bloqueos en Kaggle
os.environ["WANDB_DISABLED"] = "true"

from pathlib import Path
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, EarlyStoppingCallback
from src.training.utils.metrics import compute_metrics
from src.config import (
    TRAIN_DATA_DIR, 
    VAL_DATA_DIR, 
    BASELINE_MODEL_DIR, 
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
    """
    Entrena el modelo baseline usando datasets procesados.
    """
    # Si no se pasan argumentos, usar defaults de config
    train_dir = Path(data_dir) / "train" if data_dir else TRAIN_DATA_DIR
    val_dir = Path(data_dir) / "val" if data_dir else VAL_DATA_DIR
    output_dir = Path(output_dir) if output_dir else BASELINE_MODEL_DIR
    
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

    logger.info(f"Cargando modelo base y tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Forzar uso de GPU si está disponible
    import torch
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info(f"Modelo movido a GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("GPU no detectada. Entrenando en CPU.")

    try:
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=100,
            learning_rate=1e-5,  # Professional setting for stability
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=num_train_epochs,
            gradient_accumulation_steps=2,
            dataloader_num_workers=4,
            weight_decay=0.1,  # Strong regularization
            warmup_ratio=0.1,  # Smooth start
            lr_scheduler_type="cosine",  # Optimal convergence
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=2
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,  # Pasar tokenizer para que se guarde automáticamente
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        logger.info("Iniciando entrenamiento...")
        trainer.train()

        logger.info("Evaluando modelo...")
        metrics = trainer.evaluate()
        logger.info(f"Métricas finales: {metrics}")

        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir)) # Asegurar guardado explícito
        
        # Guardar métricas
        with open(output_dir / "eval_results.json", "w") as f:
            json.dump(metrics, f)
            
        logger.info(f"Modelo y tokenizer guardados en {output_dir}")

    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}")
        raise

if __name__ == "__main__":
    main()
