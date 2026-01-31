from typing import Dict, Optional, List, cast
from src.inference.model_utils import load_text_clf_pipeline, label_names_from_config
from src.preprocessing.clean_text import clean_text
from src.explainability.explainers import (
    explain_with_lime,
    explain_with_shap,
    explain_with_attention,
)
from src.extraction.article_extractor import extract_article_from_url, ArticleExtractionError
from src.utils.logger import get_logger

logger = get_logger(__name__)

def _label_map(idx: int) -> str:
    return "FAKE" if idx == 0 else "REAL"


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


def generate_explanation(text: str, method: str = "lime", model_dir: Optional[str] = None) -> Dict:
    """
    Genera una explicación para el texto usando el método indicado.
    Soporta: "lime", "shap", "attention".
    Aplica truncado básico para textos muy largos.
    """
    clf, tokenizer, model = load_text_clf_pipeline(model_dir)
    enc = tokenizer(text, truncation=True, max_length=512, return_offsets_mapping=True)
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
    if method == "attention":
        return explain_with_attention(text, model_dir=model_dir, clf=clf, tokenizer=tokenizer, model=model)
    return explain_with_lime(text, model_dir=model_dir, clf=clf, tokenizer=tokenizer, model=model)


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

    logger.info(f"🔎 DEBUG: Iniciando predicción para contenido de tipo '{content_type}'")
    
    if content_type == "url":
        try:
            logger.info(f"🔎 DEBUG: Extrayendo artículo desde URL: {content}")
            extracted = extract_article_from_url(content)
        except ArticleExtractionError as e:
            logger.error(f"❌ DEBUG: Error de extracción ({e.stage}): {e}")
            return {"type": content_type, "content": content, "error_stage": e.stage, "error": str(e)}
        except Exception as e:
            logger.error(f"❌ DEBUG: Error desconocido en extracción: {e}")
            return {"type": content_type, "content": content, "error_stage": "unknown", "error": str(e)}
        extracted_title = extracted.get("title", "")
        text = extracted.get("text", "")
        logger.info(f"🔎 DEBUG: Artículo extraído exitosamente. Título: '{extracted_title}' | Longitud texto: {len(text)} caracteres")
        if len(text) < 200:
             logger.warning(f"⚠️ DEBUG: Texto extraído muy corto: '{text}'")
    else:
        text = content
        logger.info(f"🔎 DEBUG: Procesando texto directo. Longitud: {len(text)} caracteres")

    # Validar texto procesable
    if not text or not text.strip():
        logger.warning("❌ DEBUG: Texto vacío o no procesable")
        return {
            "type": content_type,
            "content": content,
            "error_stage": "empty_text",
            "error": "Texto vacío o no procesable"
        }

    logger.info("🔎 DEBUG: Limpiando texto...")
    text = clean_text(text)
    logger.info(f"🔎 DEBUG: Texto limpio (primeros 100 chars): '{text[:100]}...'")

    logger.info("🔎 DEBUG: Llamando a model_predict...")
    result = model_predict(text, model_dir=model_dir)
    logger.info(f"🔎 DEBUG: Resultado crudo de model_predict: {result}")

    logger.info(f"🔎 DEBUG: Generando explicación (método: {method})...")
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
