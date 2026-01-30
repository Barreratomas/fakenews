import os
import sys
import optuna
import torch
import json
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    AutoTokenizer,
    DebertaV2Tokenizer,
    DataCollatorWithPadding,
    TrainerCallback
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

# Desactivar wandb para la búsqueda
os.environ["WANDB_DISABLED"] = "true"

class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Filtramos logs internos si es necesario, pero imprimimos lo principal
        if logs and state.is_local_process_zero:
            # logs suele contener loss, learning_rate, epoch, etc.
            # Imprimimos en una sola línea para no saturar, pero visible
            print(f"    👉 [Epoch {logs.get('epoch', '?'):.2f} | Step {state.global_step}] {logs}")

def objective(trial):
    print(f"🟢 [Trial {trial.number}] Iniciando prueba de hiperparámetros...")
    
    # 1. Definir Espacio de Búsqueda
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [4, 8])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
    lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])
    lora_dropout = trial.suggest_float("lora_dropout", 0.05, 0.2)
    
    print(f"📋 [Trial {trial.number}] Configuración: LR={learning_rate:.2e}, Batch={per_device_train_batch_size}, LoRA R={lora_r}")
    
    # 2. Cargar Datos
    train_ds = load_from_disk(str(TRAIN_DATA_DIR))
    val_ds = load_from_disk(str(VAL_DATA_DIR))
    print(f"📊 [Trial {trial.number}] Datos cargados. Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Para acelerar la búsqueda, usamos un subset (activado para eficiencia)
    # Usamos 10% o max 4000 muestras para que sea rápido
    subset_size_train = min(len(train_ds), 4000)
    subset_size_val = min(len(val_ds), 500)
    
    logger.info(f"✂ Usando subset para optimización: {subset_size_train} train, {subset_size_val} val")
    print(f"✂ Usando subset para optimización: {subset_size_train} train, {subset_size_val} val")
    
    train_ds = train_ds.shuffle(seed=42).select(range(subset_size_train))
    val_ds = val_ds.shuffle(seed=42).select(range(subset_size_val))

    train_ds.set_format(type="torch")
    val_ds.set_format(type="torch")
    
    # Pesos de clase
    label_col = "label" if "label" in train_ds.features else "labels"
    class_weights = compute_class_weights(np.array(train_ds[label_col]), num_labels=2)

    # 3. Modelo y Tokenizer
    model_name = "microsoft/mdeberta-v3-base"
    try:
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Configurar LoRA con parámetros del trial
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
    
    if torch.cuda.is_available():
        model = model.cuda()

    # 4. Argumentos de Entrenamiento
    # Ajustar gradient accumulation para batch efectivo constante de 16
    target_batch = 16
    grad_acc_steps = max(1, target_batch // per_device_train_batch_size)

    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / f"optuna_trial_{trial.number}"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=16, # Reducido de 32 a 16
        num_train_epochs=3, # Pocas épocas para la búsqueda
        weight_decay=weight_decay,
        gradient_accumulation_steps=grad_acc_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        group_by_length=True,
        gradient_checkpointing=True, # CRÍTICO para evitar OOM durante optimización
        report_to="none",
        disable_tqdm=True,
        logging_strategy="steps",
        logging_steps=100
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        num_labels=2,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1), PrinterCallback()] 
    )

    logger.info(f"🚀 [Trial {trial.number}] Comenzando entrenamiento...")
    try:
        trainer.train()
        metrics = trainer.evaluate()
        f1_score = metrics["eval_f1"]
        logger.info(f"✅ [Trial {trial.number}] Finalizado. F1 Score: {f1_score:.4f}")
        return f1_score
    except Exception as e:
        logger.error(f"❌ [Trial {trial.number}] Falló con error: {str(e)}")
        # Retornamos un valor muy bajo para que Optuna descarte esta configuración
        return 0.0
    
def run_hyperparameter_optimization(n_trials=10, save_path=None):
    """
    Ejecuta la optimización de hiperparámetros y retorna los mejores parámetros.
    """
    logger.info("Iniciando Optimización de Hiperparámetros con Optuna...")
    print("Iniciando Optimización de Hiperparámetros con Optuna...")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    logger.info("Búsqueda completada.")
    logger.info(f"Mejores parámetros: {study.best_params}")
    logger.info(f"Mejor F1 Score: {study.best_value}")
    print("Búsqueda completada.")
    print(f"Mejores parámetros: {study.best_params}")
    print(f"Mejor F1 Score: {study.best_value}")
    
    if save_path:
        with open(save_path, "w") as f:
            json.dump(study.best_params, f, indent=4)
        logger.info(f"Mejores parámetros guardados en {save_path}")
            
    return study.best_params

def main():
    # Por defecto guarda en models/best_hyperparameters.json
    save_path = MODELS_DIR / "best_hyperparameters.json"
    run_hyperparameter_optimization(n_trials=10, save_path=save_path)

if __name__ == "__main__":
    main()
