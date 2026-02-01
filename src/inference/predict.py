from typing import Dict, Optional, List, cast, Tuple, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os
import json
import torch
from peft import PeftModel
from src.config import DEFAULT_MODEL_DIR, DEFAULT_BASE_MODEL_NAME
from src.preprocessing.clean_text import clean_text
from src.extraction.article_extractor import extract_article_from_url, ArticleExtractionError
from src.utils.logger import get_logger

from src.config import (
    FAKE_THRESHOLD_HIGH, 
    FAKE_THRESHOLD_LOW,
    TOKENIZER_MAX_LENGTH,
    DEFAULT_EXPLANATION_METHOD
)

logger = get_logger(__name__)

# Cache de modelos
_CACHED_DIR: Optional[str] = None
_CACHED_CLF = None
_CACHED_TOKENIZER = None
_CACHED_MODEL = None

def default_model_dir() -> str:
    return os.environ.get("MODEL_DIR", str(DEFAULT_MODEL_DIR))

def load_text_clf_pipeline(model_dir: Optional[str] = None) -> Tuple[Any, AutoTokenizer, AutoModelForSequenceClassification]:
    global _CACHED_DIR, _CACHED_CLF, _CACHED_TOKENIZER, _CACHED_MODEL
    
    model_dir = model_dir or default_model_dir()
    
    # Acierto de caché
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
    return "FAKE" if idx == 0 else "REAL"


def model_predict(text: str, model_dir: Optional[str] = None) -> Dict:
    logger.info(" DEBUG: Entrando a model_predict...")
    clf, tokenizer, model = load_text_clf_pipeline(model_dir)

    # Depuración de Tokenización
    logger.info(" DEBUG: Tokenizando texto para inspección...")
    tokens = tokenizer(text, truncation=True, max_length=TOKENIZER_MAX_LENGTH, return_tensors="pt")
    input_ids = tokens["input_ids"][0].tolist()
    decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    logger.info(f" DEBUG: Primeros 20 tokens: {decoded_tokens[:20]}")
    logger.info(f" DEBUG: Total tokens generados: {len(input_ids)}")

    logger.info(" DEBUG: Ejecutando inferencia en pipeline...")
    raw_scores = clf(text)[0]
    logger.info(f" DEBUG: Scores crudos del pipeline: {raw_scores}")

    scores = cast(List[Dict[str, float]], raw_scores)
    
    # === Lógica de Umbral Estricto con Zona Gris ===
    # Rango 0.00 - 0.50: REAL (Seguro)
    # Rango 0.50 - 0.95: AMBIGUOUS (Sospechoso, requiere RAG)
    # Rango 0.95 - 1.00: FAKE (Seguro)
    
    fake_score = 0.0
    real_score = 0.0
    
    # Extraer scores por etiqueta
    for item in scores:
        if item["label"] == "LABEL_0": # FAKE
            fake_score = item["score"]
        elif item["label"] == "LABEL_1": # REAL
            real_score = item["score"]
            
    
    if fake_score >= FAKE_THRESHOLD_HIGH:
        label = "FAKE"
        conf = fake_score
    elif fake_score > FAKE_THRESHOLD_LOW:
        label = "AMBIGUOUS"
        conf = fake_score
    else:
        label = "REAL"
        conf = real_score
        
    logger.info(f"DEBUG: Scores -> Fake: {fake_score:.4f} | Real: {real_score:.4f}")
    logger.info(f"DEBUG: Etiqueta final seleccionada: {label} con confianza {conf:.4f}")

    return {"label": label, "confidence": round(conf, 4)}


def generate_explanation(text: str, method: str = DEFAULT_EXPLANATION_METHOD, model_dir: Optional[str] = None) -> Dict:
    """
    Genera una explicación para el texto usando el método indicado.
    Soporta: "lime", "shap", "attention", "integrated_gradients".
    Aplica truncado básico para textos muy largos.
    """
    from src.explainability.explainers import (
        explain_with_lime,
        explain_with_shap,
        explain_with_attention,
    )

    clf, tokenizer, model = load_text_clf_pipeline(model_dir)
    enc = tokenizer(text, truncation=True, max_length=TOKENIZER_MAX_LENGTH, return_offsets_mapping=True)
    offsets = enc.get("offset_mapping", None)
    if offsets:
        last_end = 0
        for off in offsets:
            if off is not None:
                last_end = max(last_end, off[1])
        text = text[:last_end] if last_end > 0 else text
    
    logger.info(f"Generando explicación con método: {method}")
    if method == "lime":
        return explain_with_lime(text, model_dir=model_dir, clf=clf, tokenizer=tokenizer, model=model)
    if method == "shap":
        return explain_with_shap(text, model_dir=model_dir, clf=clf, tokenizer=tokenizer, model=model)
    if method == "integrated_gradients":
        logger.warning("Método 'integrated_gradients' no implementado nativamente. Usando 'attention' como fallback rápido.")
        return explain_with_attention(text, model_dir=model_dir, clf=clf, tokenizer=tokenizer, model=model)
    if method == "attention":
        return explain_with_attention(text, model_dir=model_dir, clf=clf, tokenizer=tokenizer, model=model)
    
    logger.warning(f"Método de explicación '{method}' desconocido. Usando 'attention' por defecto.")
    return explain_with_attention(text, model_dir=model_dir, clf=clf, tokenizer=tokenizer, model=model)


def predict(input_data: Dict, method: str = "lime", model_dir: Optional[str] = None) -> Dict:
    """
    Orquesta la predicción y la explicación.
    Entradas:
      - input_data: {"type": "text"|"url", "content": "..."}
      - method: "lime" | "shap" | "attention"
    Salida:
      - {"label", "confidence", "explanation": {"top_words","top_word_scores","sentence_contributions"}, "extracted_title"?}
    """
    content_type = str(input_data.get("type", "text")).lower()
    content = str(input_data.get("content", "")).strip()
    extracted_title = ""

    logger.info(f" DEBUG: Iniciando predicción para contenido de tipo '{content_type}'")
    
    if content_type == "url":
        try:
            logger.info(f" DEBUG: Extrayendo artículo desde URL: {content}")
            extracted = extract_article_from_url(content)
        except ArticleExtractionError as e:
            logger.error(f"DEBUG: Error de extracción ({e.stage}): {e}")
            return {"type": content_type, "content": content, "error_stage": e.stage, "error": str(e)}
        except Exception as e:
            logger.error(f"DEBUG: Error desconocido en extracción: {e}")
            return {"type": content_type, "content": content, "error_stage": "unknown", "error": str(e)}
        extracted_title = extracted.get("title", "")
        text = extracted.get("text", "")
        logger.info(f"DEBUG: Artículo extraído exitosamente. Título: '{extracted_title}' | Longitud texto: {len(text)} caracteres")
        if len(text) < 200:
             logger.warning(f"DEBUG: Texto extraído muy corto: '{text}'")
    else:
        text = content
        logger.info(f" DEBUG: Procesando texto directo. Longitud: {len(text)} caracteres")

    # Validar texto procesable
    if not text or not text.strip():
        logger.warning("DEBUG: Texto vacío o no procesable")
        return {
            "type": content_type,
            "content": content,
            "error_stage": "empty_text",
            "error": "Texto vacío o no procesable"
        }

    logger.info("DEBUG: Limpiando texto...")
    text = clean_text(text)
    logger.info(f"DEBUG: Texto limpio (primeros 100 chars): '{text[:100]}...'")

    logger.info("DEBUG: Llamando a model_predict...")
    result = model_predict(text, model_dir=model_dir)
    logger.info(f"DEBUG: Resultado crudo de model_predict: {result}")

    logger.info(f"DEBUG: Generando explicación (método: {method})...")
    explanation = generate_explanation(text, method=method, model_dir=model_dir)
    out = {
        "label": result["label"],
        "confidence": result["confidence"],
        "explanation": {
            "top_words": explanation.get("top_words", []),
            "top_word_scores": explanation.get("top_word_scores", []),
            "sentence_contributions": explanation.get("sentence_contributions", [])
        }
    }
    if extracted_title:
        out["extracted_title"] = extracted_title
    
    logger.info(f"Predicción completada: {out['label']} ({out['confidence']})")
    return out
