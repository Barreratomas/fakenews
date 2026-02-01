import os
import sys

# Añadir el directorio raíz al PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(src_dir)

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
    DEFAULT_BASE_MODEL_NAME,
    DEFAULT_NUM_LABELS,
    DEFAULT_BATCH_SIZE
)
from src.utils.logger import get_logger
from src.training.utils.metrics import compute_metrics

logger = get_logger(__name__)

def main(
    model_dir: Path = MODELS_DIR / "deberta_lora",
    data_dir: Path = TEST_DATA_DIR,
    output_file: Path = None
):
    # validar existencia de directorios
    if not model_dir.exists():
        logger.error(f"EL DIRECTORIO DEL MODELO NO EXISTE: {model_dir}")
        sys.exit(1)
        
    if not data_dir.exists():
        logger.error(f"EL DIRECTORIO DE DATOS NO EXISTE: {data_dir}")
        sys.exit(1)
        


    if model_dir.exists():
        logger.info(f"CONTENIDO DE {model_dir}:")
        files = list(model_dir.iterdir())
        if not files:
            logger.warning("EL DIRECTORIO ESTÁ VACÍO.")
        for f in files:
            logger.info(f"   - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        logger.error(f"EL DIRECTORIO {model_dir} NO EXISTE.")
        sys.exit(1)

    # Validar archivos críticos del modelo antes de intentar cargar
    adapter_config_path = model_dir / "adapter_config.json"
    adapter_model_path = model_dir / "adapter_model.safetensors"
    adapter_model_bin = model_dir / "adapter_model.bin"
    
    logger.info("Buscando archivos de configuración y pesos...")
    if adapter_config_path.exists():
        logger.info(f"ENCONTRADO: {adapter_config_path.name}")
    else:
        logger.error(f"FALTA: {adapter_config_path.name}")

    if adapter_model_path.exists():
        logger.info(f"ENCONTRADO: {adapter_model_path.name}")
    elif adapter_model_bin.exists():
        logger.info(f"ENCONTRADO: {adapter_model_bin.name}")
    else:
        logger.error(f"FALTA: adapter_model.safetensors O adapter_model.bin")

    if not adapter_config_path.exists():
        logger.error(f"FALTA: adapter_config.json EN {model_dir}")
        logger.error("Asegúrate de haber entrenado el modelo o de tener los archivos en la ruta correcta.")
        sys.exit(1)
        
    if not adapter_model_path.exists() and not adapter_model_bin.exists():
        logger.error(f"FALTA: adapter_model.safetensors (o .bin) EN {model_dir}")
        sys.exit(1)

    # cargar Configuración del Adaptador para saber el modelo base
    try:
        peft_config = PeftConfig.from_pretrained(str(model_dir))
        base_model_name = peft_config.base_model_name_or_path
    except Exception as e:
        logger.warning(f"NO SE PUDO CARGAR CONFIG DE PEFT DESDE {model_dir}: {e}")
        logger.warning(f"USANDO MODELO BASE POR DEFECTO: {DEFAULT_BASE_MODEL_NAME}")
        base_model_name = DEFAULT_BASE_MODEL_NAME

    # tokenizer
    logger.info(f"CARGANDO TOKENIZER DESDE {model_dir} O BASE {base_model_name}")
    try:
        # Intentar cargar el tokenizer guardado en el directorio del modelo
        tokenizer = DebertaV2Tokenizer.from_pretrained(str(model_dir))
    except:
        try:
            tokenizer = DebertaV2Tokenizer.from_pretrained(base_model_name)
        except:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    # datos
    logger.info(f"CARGANDO DATOS DE TEST DESDE {data_dir}")
    test_ds = load_from_disk(str(data_dir))
    
    # Asegurar nombres de columnas
    label_col = "label" if "label" in test_ds.features else "labels"
    test_ds.set_format(type="torch")
    
    # modelo
    logger.info(f"CARGANDO MODELO BASE: {base_model_name}")
    # Inferir número de etiquetas
    num_labels = DEFAULT_NUM_LABELS 
    if hasattr(test_ds.features[label_col], "num_classes"):
        num_labels = test_ds.features[label_col].num_classes
        
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, 
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    
    # adaptadores LoRA
    logger.info(f"CARGANDO ADAPTADORES LoRA DESDE {model_dir}")
    model = PeftModel.from_pretrained(base_model, str(model_dir))
    
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info(f"MODELO EN GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("GPU NO DETECTADA. LA EVALUACIÓN SERÁ LENTA.")
    
    # evaluación
    logger.info(f"EVALUACIÓN DEL MODELO EN {data_dir}")
    # Usar un directorio temporal para logs de evaluación, no el directorio del modelo (que puede ser read-only)
    eval_output_dir = Path("eval_logs")
    training_args = TrainingArguments(
        output_dir=str(eval_output_dir), 
        per_device_eval_batch_size=DEFAULT_BATCH_SIZE,
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
    
    # evaluar
    logger.info("INICIANDO EVALUACIÓN...")
    test_results = trainer.evaluate(test_ds)
    
    logger.info(f"\n RESULTADOS FINALES (TEST SET):")
    logger.info(f"F1 Score: {test_results.get('eval_f1', 'N/A')}")
    logger.info(f"Accuracy: {test_results.get('eval_accuracy', 'N/A')}")
    logger.info(f"Loss:     {test_results.get('eval_loss', 'N/A')}")
    
    # guardar resultados
    if output_file is None:
        # Guardar en el directorio actual por defecto para evitar errores de Read-only filesystem
        output_file = Path("test_results.json")
        
    with open(output_file, "w") as f:
        json.dump(test_results, f, indent=4)
    logger.info(f"RESULTADOS GUARDADOS EN {output_file.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluar modelo DeBERTa LoRA")
    parser.add_argument("--model_dir", type=str, default=str(MODELS_DIR / "deberta_lora"), help="Directorio del modelo entrenado")
    parser.add_argument("--data_dir", type=str, default=str(TEST_DATA_DIR), help="Directorio de datos de test")
    parser.add_argument("--output_file", type=str, default=None, help="Archivo donde guardar los resultados (JSON)")
    
    args = parser.parse_args()
    
    main(
        model_dir=Path(args.model_dir),
        data_dir=Path(args.data_dir),
        output_file=Path(args.output_file) if args.output_file else None
    )
