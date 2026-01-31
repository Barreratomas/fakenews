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

# Integración Optuna con Transformers para Pruning eficiente
from optuna.integration import PyTorchLightningPruningCallback
# Nota: HuggingFace Trainer tiene integración nativa, pero lo haremos explícito en el loop o usaremos un callback custom si es necesario.
# En este caso, usaremos el pruning nativo de Optuna a través de reportes manuales en el callback.

logger = get_logger(__name__)

# Desactivar wandb para evitar spam de proyectos
os.environ["WANDB_DISABLED"] = "true"

class OptunaPruningCallback(TrainerCallback):
    """
    Callback para reportar métricas intermedias a Optuna y permitir Pruning (poda) de trials malos.
    """
    def __init__(self, trial: optuna.Trial, metric_name: str = "eval_f1"):
        self.trial = trial
        self.metric_name = metric_name

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            current_score = metrics.get(self.metric_name)
            if current_score is not None:
                # Reportar valor actual a Optuna
                self.trial.report(current_score, step=state.epoch)
                
                # Verificar si el trial debe ser podado
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

def objective(trial):
    trial_num = trial.number
    logger.info(f"🧪 [Trial {trial_num}] Iniciando configuración...")
    
    # -------------------------------------------------------------------------
    # 1. Definir Espacio de Búsqueda (Refinado y Profesional)
    # -------------------------------------------------------------------------
    # Rango de LR más conservador para Fine-Tuning de Transformers
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
    
    # Batch size: Preferimos 8 o 16 para estabilidad.
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16])
    
    # Weight decay estándar
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)
    
    # LoRA params
    lora_r = trial.suggest_categorical("lora_r", [16, 32, 64])
    lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])
    lora_dropout = trial.suggest_float("lora_dropout", 0.05, 0.15)
    
    # CRÍTICO: Optimizar también el número de épocas para evitar mismatch.
    # El modelo final usará exactamente estas épocas.
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 6)

    logger.info(f"📋 [Trial {trial_num}] Config: LR={learning_rate:.2e}, Batch={per_device_train_batch_size}, R={lora_r}, Epochs={num_train_epochs}")
    
    # -------------------------------------------------------------------------
    # 2. Cargar Datos y Estrategia de Muestreo
    # -------------------------------------------------------------------------
    try:
        train_ds = load_from_disk(str(TRAIN_DATA_DIR))
        val_ds = load_from_disk(str(VAL_DATA_DIR))
    except Exception as e:
        logger.error(f"❌ Error fatal cargando datos: {e}")
        raise e

    # ESTRATEGIA DE SUBSET:
    # Para optimización profesional pero rápida, usamos un subset representativo pero más grande que 4k si es posible.
    # Usaremos el 20% del dataset o 5000 muestras, lo que sea mayor, para tener significancia estadística.
    total_train = len(train_ds)
    subset_size_train = min(total_train, 6000) # Aumentado de 4000 a 6000 para mayor robustez
    subset_size_val = min(len(val_ds), 1000)   # Aumentado de 500 a 1000
    
    # Mezcla determinística para consistencia entre trials
    train_ds = train_ds.shuffle(seed=42).select(range(subset_size_train))
    val_ds = val_ds.shuffle(seed=42).select(range(subset_size_val))
    
    logger.info(f"✂ [Trial {trial_num}] Subset activo: {len(train_ds)} train, {len(val_ds)} val")

    train_ds.set_format(type="torch")
    val_ds.set_format(type="torch")
    
    # Pesos de clase (Recalculados sobre el subset para precisión local)
    label_col = "label" if "label" in train_ds.features else "labels"
    labels_array = np.array(train_ds[label_col])
    class_weights = compute_class_weights(labels_array, num_labels=2)
    
    # -------------------------------------------------------------------------
    # 3. Modelo y Tokenizer
    # -------------------------------------------------------------------------
    model_name = "microsoft/mdeberta-v3-base"
    try:
        # Intentar cargar tokenizer rápido si es posible
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Configurar LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["query_proj", "key_proj", "value_proj", "dense"],
        modules_to_save=["pooler", "classifier"], # CRÍTICO: Guardar cabezales de clasificación
    )
    model = get_peft_model(model, lora_config)
    
    if torch.cuda.is_available():
        model = model.cuda()

    # -------------------------------------------------------------------------
    # 4. Entrenamiento
    # -------------------------------------------------------------------------
    # Ajuste de Gradient Accumulation para mantener batch efectivo
    target_effective_batch = 16
    grad_acc_steps = max(1, target_effective_batch // per_device_train_batch_size)

    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / f"optuna_trial_{trial_num}"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=16,
        num_train_epochs=num_train_epochs, # Usamos el valor optimizado
        weight_decay=weight_decay,
        gradient_accumulation_steps=grad_acc_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        group_by_length=True,
        gradient_checkpointing=True, 
        report_to="none",
        disable_tqdm=True,
        logging_strategy="epoch", # Menos ruido en logs
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
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2), # Patience un poco más relajado
            OptunaPruningCallback(trial, metric_name="eval_f1") # Pruning activado
        ] 
    )

    try:
        trainer.train()
        metrics = trainer.evaluate()
        f1_score = metrics["eval_f1"]
        logger.info(f"✅ [Trial {trial_num}] Resultado: F1={f1_score:.4f}")
        return f1_score
    except optuna.exceptions.TrialPruned:
        logger.info(f"✂ [Trial {trial_num}] Podado (Pruned) por bajo rendimiento.")
        raise
    except Exception as e:
        logger.error(f"❌ [Trial {trial_num}] Falló: {e}")
        return 0.0

def run_hyperparameter_optimization(n_trials=15, save_path=None):
    """
    Ejecuta la optimización usando TPE Sampler y Median Pruner para eficiencia.
    """
    logger.info("🚀 Iniciando Optimización de Hiperparámetros Profesional...")
    
    # MedianPruner descarta trials que estén peor que la mediana de trials anteriores en el mismo paso.
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
    
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        study_name="fake_news_optimization"
    )
    
    study.optimize(objective, n_trials=n_trials)
    
    logger.info("🎉 Búsqueda completada.")
    logger.info(f"🏆 Mejores parámetros: {study.best_params}")
    logger.info(f"🏆 Mejor F1 Score: {study.best_value}")
    
    if save_path:
        # Guardar metadatos adicionales
        final_config = study.best_params.copy()
        final_config["_optimization_metadata"] = {
            "best_f1": study.best_value,
            "n_trials": n_trials,
            "strategy": "Optuna TPE + MedianPruner"
        }
        
        with open(save_path, "w") as f:
            json.dump(final_config, f, indent=4)
        logger.info(f"💾 Configuración guardada en {save_path}")
            
    return study.best_params

def main():
    save_path = MODELS_DIR / "best_hyperparameters.json"
    # Aumentamos trials a 15 ya que el Pruner hará que los malos sean rápidos
    run_hyperparameter_optimization(n_trials=15, save_path=save_path)

if __name__ == "__main__":
    main()
