import os
import sys
import argparse
import json
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
    DebertaV2Tokenizer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
from src.config import (
    TRAIN_DATA_DIR,
    VAL_DATA_DIR,
    MODELS_DIR,
    DEFAULT_BASE_MODEL_NAME
)
from src.utils.logger import get_logger
from src.training.utils.metrics import compute_metrics, compute_class_weights
from src.training.utils.trainer_utils import WeightedTrainer

logger = get_logger(__name__)

def load_best_params(params_path: Path):
    if params_path.exists():
        logger.info(f"Cargando mejores hiperparámetros desde {params_path}")
        with open(params_path, "r") as f:
            return json.load(f)
    return None

def main(
    data_dir: Path = None, 
    output_dir: Path = MODELS_DIR / "deberta_lora", 
    model_name: str = "microsoft/mdeberta-v3-base", 
    num_labels: int = 2, 
    num_train_epochs: int = 10,
    optimize: bool = True,
    n_trials: int = 10
):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # === 1. Gestión de Hiperparámetros ===
    params_path = MODELS_DIR / "best_hyperparameters.json"
    best_params = {}

   
    if params_path.exists():
        # Intentar cargar si existen
        loaded = load_best_params(params_path)
        if loaded:
            best_params = loaded
            print(f"Usando hiperparámetros guardados: {best_params}")
    
    # Valores por defecto (si no están en best_params)
    learning_rate = best_params.get("learning_rate", 1e-4)
    per_device_train_batch_size = best_params.get("per_device_train_batch_size", 16)
    weight_decay = best_params.get("weight_decay", 0.01)
    lora_r = best_params.get("lora_r", 16)
    lora_alpha = best_params.get("lora_alpha", 32)
    lora_dropout = best_params.get("lora_dropout", 0.1)

    # CRÍTICO: Si Optuna optimizó epochs, usar ese valor para evitar mismatch
    if "num_train_epochs" in best_params:
        optimized_epochs = int(best_params["num_train_epochs"])
        if optimized_epochs != num_train_epochs:
            logger.info(f"🔄 Sobrescribiendo epochs ({num_train_epochs}) con valor optimizado: {optimized_epochs}")
            print(f"🔄 Sobrescribiendo epochs ({num_train_epochs}) con valor optimizado: {optimized_epochs}")
            num_train_epochs = optimized_epochs
    elif num_train_epochs > 5:
        logger.warning(f"⚠️ CUIDADO: Usando {num_train_epochs} épocas. Si usa parámetros optimizados para menos épocas, puede causar colapso.")

    # === 2. Carga de Datos ===
    train_dir = Path(data_dir) / "train" if data_dir else TRAIN_DATA_DIR
    val_dir = Path(data_dir) / "val" if data_dir else VAL_DATA_DIR

    print(f"Cargando datos desde {train_dir} y {val_dir}")
    try:
        train_ds = load_from_disk(str(train_dir))
        val_ds = load_from_disk(str(val_dir))
    except FileNotFoundError:
        print(f"No se encontraron datos en {train_dir} o {val_dir}. Ejecuta el preprocesamiento primero.")
        sys.exit(1)

    label_col = "label" if "label" in train_ds.features else "labels"
    train_ds.set_format(type="torch")
    val_ds.set_format(type="torch")

    print("Calculando pesos de clase...")
    class_weights = compute_class_weights(np.array(train_ds[label_col]), num_labels=num_labels)
    
    # === 3. Tokenizer y Modelo ===
    print(f"Cargando tokenizer: {model_name}")
    try:
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error cargando DebertaV2Tokenizer: {e}. Intentando AutoTokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print(f"Cargando modelo base: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"Modelo movido a GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("GPU no detectada. El entrenamiento será extremadamente lento en CPU.")

    # === 4. Configuración LoRA Dinámica ===
    print(f"Aplicando LoRA con r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["query_proj", "key_proj", "value_proj", "dense"],
        modules_to_save=["pooler", "classifier"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # === 5. Argumentos de Entrenamiento Dinámicos ===
    # Ajustar gradient accumulation para mantener un batch efectivo total de ~16
    # Si batch=8 -> steps=2 (8*2=16)
    # Si batch=4 -> steps=4 (4*4=16)
    target_batch_size = 16
    grad_acc_steps = max(1, target_batch_size // per_device_train_batch_size)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=16, # Reducido de 32 a 16 para ahorrar VRAM
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=grad_acc_steps,
        dataloader_num_workers=4,
        weight_decay=weight_decay,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        push_to_hub=False,
        fp16=torch.cuda.is_available(),
        # group_by_length=True,  # DESHABILITADO: Puede causar inestabilidad en datasets pequeños/desbalanceados
        gradient_checkpointing=True, 
        seed=42,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        num_labels=num_labels,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    logger.info("Iniciando entrenamiento final...")
    print("Iniciando entrenamiento final...")
    trainer.train()
    
    logger.info(f"Guardando modelo en {output_dir}")
    print(f"Guardando modelo en {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar DeBERTa LoRA con opción de optimización")
    parser.add_argument("--optimize", action="store_true", help="Ejecutar optimización de hiperparámetros antes de entrenar")
    parser.add_argument("--n_trials", type=int, default=10, help="Número de pruebas para Optuna (si optimize=True)")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas de entrenamiento")
    
    args = parser.parse_args()
    
    main(
        optimize=args.optimize,
        n_trials=args.n_trials,
        num_train_epochs=args.epochs
    )
