import os
import sys
import argparse
import json
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    DebertaV2Tokenizer,
    DataCollatorWithPadding
)
from peft import PeftModel, PeftConfig

from src.config import (
    TEST_DATA_DIR,
    MODELS_DIR,
    DEFAULT_BASE_MODEL_NAME
)
from src.utils.logger import get_logger
from src.training.utils.metrics import compute_metrics

logger = get_logger(__name__)

def main(
    model_dir: Path = MODELS_DIR / "deberta_lora",
    data_dir: Path = TEST_DATA_DIR,
    output_file: Path = None
):
    if not model_dir.exists():
        logger.error(f"El directorio del modelo no existe: {model_dir}")
        sys.exit(1)
        
    if not data_dir.exists():
        logger.error(f"El directorio de datos no existe: {data_dir}")
        sys.exit(1)

    # Validar archivos críticos del modelo antes de intentar cargar
    adapter_config_path = model_dir / "adapter_config.json"
    adapter_model_path = model_dir / "adapter_model.safetensors"
    adapter_model_bin = model_dir / "adapter_model.bin"
    
    if not adapter_config_path.exists():
        logger.error(f"❌ Falta adapter_config.json en {model_dir}")
        logger.error("Asegúrate de haber entrenado el modelo o de tener los archivos en la ruta correcta.")
        sys.exit(1)
        
    if not adapter_model_path.exists() and not adapter_model_bin.exists():
        logger.error(f"❌ Falta adapter_model.safetensors (o .bin) en {model_dir}")
        logger.info(f"Contenido de {model_dir}:")
        for f in model_dir.iterdir():
            logger.info(f" - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
        sys.exit(1)

    # 1. Cargar Configuración del Adaptador para saber el modelo base
    try:
        peft_config = PeftConfig.from_pretrained(str(model_dir))
        base_model_name = peft_config.base_model_name_or_path
    except Exception as e:
        logger.warning(f"No se pudo cargar config de PEFT desde {model_dir}: {e}")
        logger.warning(f"Usando modelo base por defecto: {DEFAULT_BASE_MODEL_NAME}")
        base_model_name = DEFAULT_BASE_MODEL_NAME

    # 2. Tokenizer
    logger.info(f"Cargando tokenizer desde {model_dir} o base {base_model_name}")
    try:
        # Intentar cargar el tokenizer guardado en el directorio del modelo
        tokenizer = DebertaV2Tokenizer.from_pretrained(str(model_dir))
    except:
        try:
            tokenizer = DebertaV2Tokenizer.from_pretrained(base_model_name)
        except:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    # 3. Datos
    logger.info(f"Cargando datos de test desde {data_dir}")
    test_ds = load_from_disk(str(data_dir))
    
    # Asegurar nombres de columnas
    label_col = "label" if "label" in test_ds.features else "labels"
    test_ds.set_format(type="torch")
    
    # 4. Modelo
    logger.info(f"Cargando modelo base: {base_model_name}")
    # Inferir número de etiquetas
    num_labels = 2 
    if hasattr(test_ds.features[label_col], "num_classes"):
        num_labels = test_ds.features[label_col].num_classes
        
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, 
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    
    logger.info(f"Cargando adaptadores LoRA desde {model_dir}")
    model = PeftModel.from_pretrained(base_model, str(model_dir))
    
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info(f"Modelo en GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("GPU no detectada. La evaluación será lenta.")
    
    # 5. Evaluación
    training_args = TrainingArguments(
        output_dir=str(model_dir), # Solo para logs temporales
        per_device_eval_batch_size=16,
        report_to="none",
        fp16=torch.cuda.is_available()
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )
    
    logger.info("Iniciando evaluación...")
    test_results = trainer.evaluate(test_ds)
    
    print(f"\n📊 RESULTADOS FINALES (TEST SET):")
    print(f"   F1 Score: {test_results.get('eval_f1', 'N/A')}")
    print(f"   Accuracy: {test_results.get('eval_accuracy', 'N/A')}")
    print(f"   Loss:     {test_results.get('eval_loss', 'N/A')}")
    
    # Guardar resultados
    if output_file is None:
        output_file = model_dir / "test_results.json"
        
    with open(output_file, "w") as f:
        json.dump(test_results, f, indent=4)
    print(f"✔ Resultados guardados en {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluar modelo DeBERTa LoRA")
    parser.add_argument("--model_dir", type=str, default=str(MODELS_DIR / "deberta_lora"), help="Directorio del modelo entrenado")
    parser.add_argument("--data_dir", type=str, default=str(TEST_DATA_DIR), help="Directorio de datos de test")
    
    args = parser.parse_args()
    
    main(
        model_dir=Path(args.model_dir),
        data_dir=Path(args.data_dir)
    )
