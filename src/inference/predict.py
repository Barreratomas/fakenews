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
    clf, tokenizer, model = load_text_clf_pipeline(model_dir)
    scores = cast(List[Dict[str, float]], clf(text)[0])
    idx = int(max(range(len(scores)), key=lambda i: scores[i]["score"]))
    conf = float(scores[idx]["score"])
    class_names = label_names_from_config(model)
    label = class_names[idx] if 0 <= idx < len(class_names) else _label_map(idx)
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

    logger.info(f"Iniciando predicción. Tipo: {content_type}")

    if content_type == "url":
        try:
            extracted = extract_article_from_url(content)
        except ArticleExtractionError as e:
            logger.error(f"Error de extracción ({e.stage}): {e}")
            return {"type": content_type, "content": content, "error_stage": e.stage, "error": str(e)}
        except Exception as e:
            logger.error(f"Error desconocido en extracción: {e}")
            return {"type": content_type, "content": content, "error_stage": "unknown", "error": str(e)}
        extracted_title = extracted.get("title", "")
        text = extracted.get("text", "")
    else:
        text = content
    # Validar texto procesable
    if not text or not text.strip():
        logger.warning("Texto vacío o no procesable")
        return {
            "type": content_type,
            "content": content,
            "error_stage": "empty_text",
            "error": "Texto vacío o no procesable"
        }

    text = clean_text(text)
    result = model_predict(text, model_dir=model_dir)
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
