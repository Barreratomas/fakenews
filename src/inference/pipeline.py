from typing import Dict, Any
from src.extraction.article_extractor import extract_article_from_url
from src.inference.predict import model_predict
from src.rag.rag_pipeline import rag_fact_check
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_inference(input_type: str, content: str) -> Dict[str, Any]:
    extracted_title = None
    text = ""

    # 1️⃣ Obtener texto
    if input_type == "url":
        try:
            logger.info(f"Extrayendo artículo desde URL: {content}")
            article = extract_article_from_url(content)
            text = article["text"]
            extracted_title = article.get("title")
        except Exception as e:
            logger.error(f"Error en extracción: {e}")
            return {
                "error_stage": "extraction",
                "error": str(e)
            }
    else:
        text = content

    if not text or not text.strip():
        logger.warning("Texto vacío recibido.")
        return {
            "error_stage": "empty_text",
            "error": "Texto vacío"
        }

    # 2️⃣ Clasificación fake / real
    try:
        logger.info("Ejecutando predicción del modelo...")
        clf_result = model_predict(text)
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        return {
            "error_stage": "prediction",
            "error": str(e)
        }

    # 3️⃣ RAG (Fact-Checking Asistido)
    try:
        logger.info("Ejecutando Fact-Checking (RAG)...")
        rag_result = rag_fact_check(text)
    except Exception as e:
        logger.error(f"Error en RAG: {e}")
        rag_result = {}

    # Mapeo a la respuesta final
    # Corregido: claves coinciden con rag_pipeline.py ('analysis', 'sources')
    explanation_text = rag_result.get("analysis", "")
    if not explanation_text:
        explanation_text = "No se pudo generar explicación."

    return {
        "label": clf_result["label"],
        "confidence": clf_result["confidence"],
        "explanation": explanation_text,
        "extracted_title": extracted_title,
        "rag_analysis": rag_result.get("analysis"),
        "retrieved_sources": rag_result.get("sources", [])
    }
