from typing import Optional, Tuple, List, Any, Dict, cast
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os
import json
import torch
from peft import PeftModel
from src.config import DEFAULT_MODEL_DIR, DEFAULT_BASE_MODEL_NAME
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CACHED_DIR: Optional[str] = None
_CACHED_CLF = None
_CACHED_TOKENIZER = None
_CACHED_MODEL = None

def default_model_dir() -> str:
    return os.environ.get("MODEL_DIR", str(DEFAULT_MODEL_DIR))

def load_text_clf_pipeline(model_dir: Optional[str] = None) -> Tuple[Any, AutoTokenizer, AutoModelForSequenceClassification]:
    global _CACHED_DIR, _CACHED_CLF, _CACHED_TOKENIZER, _CACHED_MODEL
    
    model_dir = model_dir or default_model_dir()
    
    # Cache hit
    if _CACHED_CLF is not None and _CACHED_DIR == model_dir:
        return _CACHED_CLF, _CACHED_TOKENIZER, _CACHED_MODEL
    
    # Intentar cargar el modelo solicitado primero
    try:
        return _load_specific_model(model_dir)
    except Exception as e:
        logger.error(f"FALLO CRÍTICO cargando modelo principal {model_dir}: {e}")
        raise e

def _load_specific_model(model_dir: str):
    global _CACHED_DIR, _CACHED_CLF, _CACHED_TOKENIZER, _CACHED_MODEL
    
    logger.info(f"Cargando modelo desde: {model_dir}")
    
    # Detectar LoRA (adapter_config.json)
    adapter_cfg = os.path.join(model_dir, "adapter_config.json")
    if os.path.isfile(adapter_cfg):
        logger.info("Detectado adaptador LoRA.")
        with open(adapter_cfg, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            
        base_name = cfg.get("base_model_name_or_path", DEFAULT_BASE_MODEL_NAME)
        logger.info(f"Cargando modelo base: {base_name}")
        
        # Intentar cargar tokenizer localmente primero
        try:
            logger.info(f"Intentando cargar tokenizer desde {model_dir}")
            tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        except Exception:
            logger.warning(f"No se encontró tokenizer en {model_dir}, cargando desde base: {base_name}")
            try:
                from transformers import DebertaV2TokenizerFast
                tokenizer = DebertaV2TokenizerFast.from_pretrained(base_name, use_fast=True)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=True)
            
        model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
        
        logger.info("Aplicando pesos LoRA...")
        model = PeftModel.from_pretrained(model, model_dir)
            
    else:
        # Modelo estándar (full fine-tuning o base)
        logger.info("Cargando modelo estándar (sin LoRA detectado explícitamente).")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
        except Exception:
            logger.warning(f"No se pudo cargar tokenizer de {model_dir}, usando base.")
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL_NAME)
            
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Crear pipeline
    clf = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        truncation=True
    )
    
    # Actualizar cache solo si es exitoso
    _CACHED_DIR = model_dir
    _CACHED_CLF = clf
    _CACHED_TOKENIZER = tokenizer
    _CACHED_MODEL = model
    
    logger.info(f"Pipeline cargado exitosamente desde {model_dir}")
    return _CACHED_CLF, _CACHED_TOKENIZER, _CACHED_MODEL

def label_names_from_config(model: AutoModelForSequenceClassification) -> List[str]:
    # Intentar leer id2label
    id2label = getattr(model.config, "id2label", None)
    if isinstance(id2label, dict) and len(id2label) >= 2:
        # Extraer labels en orden 0, 1
        labels = [id2label.get(0, "FAKE"), id2label.get(1, "REAL")]
        # Si son genéricos LABEL_0/LABEL_1, forzamos FAKE/REAL
        if labels[0] == "LABEL_0" and labels[1] == "LABEL_1":
            return ["FAKE", "REAL"]
        return labels
    return ["FAKE", "REAL"]

def _label_map(idx: int) -> str:
    return f"LABEL_{idx}"

def model_predict(text: str, model_dir: Optional[str] = None) -> Dict:
    logger.info("🔎 DEBUG: Entrando a model_predict...")
    clf, tokenizer, model = load_text_clf_pipeline(model_dir)
    
    # Debugging Tokenization
    logger.info("🔎 DEBUG: Tokenizando texto para inspección...")
    tokens = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
    input_ids = tokens["input_ids"][0].tolist()
    decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    logger.info(f"🔎 DEBUG: Primeros 20 tokens: {decoded_tokens[:20]}")
    logger.info(f"🔎 DEBUG: Total tokens generados: {len(input_ids)}")

    logger.info("🔎 DEBUG: Ejecutando inferencia en pipeline...")
    raw_scores = clf(text)[0]
    logger.info(f"🔎 DEBUG: Scores crudos del pipeline: {raw_scores}")

    scores = cast(List[Dict[str, float]], raw_scores)
    idx = int(max(range(len(scores)), key=lambda i: scores[i]["score"]))
    conf = float(scores[idx]["score"])
    
    class_names = label_names_from_config(model)
    logger.info(f"🔎 DEBUG: Nombres de clases detectados en config: {class_names}")
    
    label = class_names[idx] if 0 <= idx < len(class_names) else _label_map(idx)
    logger.info(f"🔎 DEBUG: Etiqueta final seleccionada: {label} (idx={idx}) con confianza {conf:.4f}")
    
    return {"label": label, "confidence": round(conf, 4)}
