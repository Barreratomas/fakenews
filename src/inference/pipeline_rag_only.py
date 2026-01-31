from typing import Dict, Any
from src.extraction.article_extractor import extract_article_from_url, ArticleExtractionError
# from src.inference.predict import model_predict, generate_explanation
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
        except ArticleExtractionError as e:
            logger.error(f"Error de extracción ({e.stage}): {e}")
            return {
                "error_stage": e.stage, # Propagate specific stage (paywall, http_error, etc)
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Error en extracción: {e}")
            return {
                "error_stage": "extraction_unknown",
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
    logger.info("⚠️ MODELO DESHABILITADO (Modo RAG-Only). Usando dummy result.")
    clf_result = {"label": "RAG_ONLY", "confidence": 0.0}

    # 3️⃣ RAG (Fact-Checking Asistido)
    try:
        logger.info("Ejecutando Fact-Checking (RAG)...")
        rag_result = rag_fact_check(text)
    except Exception as e:
        logger.error(f"Error crítico en RAG: {e}")
        # Enforce mandatory RAG: Fail the entire inference if RAG fails
        return {
            "error_stage": "rag_error",
            "error": str(e)
        }

    # Mapeo a la respuesta final
    # Corregido: claves coinciden con rag_pipeline.py ('analysis', 'sources')
    explanation_text = rag_result.get("analysis", "")
    if not explanation_text:
        explanation_text = "No se pudo generar explicación."

    # 4️⃣ Explicación del modelo (Keywords)
    model_explanation = {}
    # try:
    #     logger.info("Generando explicación del modelo (Keywords)...")
    #     model_explanation = generate_explanation(text, method="attention")
    # except Exception as e:
    #     logger.warning(f"No se pudo generar explicación del modelo: {e}")

    # 5️⃣ Resolución de Conflictos (DeBERTa vs RAG)
    rag_analysis = rag_result.get("analysis", "").lower()
    
    # Heurística simple para detectar la postura del RAG basándonos en palabras clave del análisis
    # (En un futuro ideal, el RAG debería devolver un label explícito, pero por ahora analizamos el texto)
    rag_verdict = "UNCERTAIN"
    if any(k in rag_analysis for k in ["contradicted", "false", "incorrect", "unsupported", "fake", "hoax"]):
        rag_verdict = "FAKE"
    elif any(k in rag_analysis for k in ["supported", "true", "correct", "confirmed", "accurate"]):
        rag_verdict = "REAL"
        
    # En modo RAG-Only, el veredicto final es el del RAG
    model_label = rag_verdict if rag_verdict != "UNCERTAIN" else "RAG_UNCERTAIN"
    final_verdict = model_label
    verdict_message = "Decisión basada puramente en RAG (Modelo deshabilitado)"
    
    # Lógica antigua de conflicto (comentada para modo RAG-Only)
    # model_label = clf_result["label"].upper()  # FAKE | REAL
    # if model_label == "REAL" and rag_verdict == "FAKE":
    #     final_verdict = "WARNING_DISPUTED"
    #     verdict_message = "⚠️ ALERTA: Texto parece real, pero los hechos lo contradicen (Desinformación Sofisticada)"
    # elif model_label == "FAKE" and rag_verdict == "REAL":
    #     final_verdict = "WARNING_SENSATIONALIST"
    #     verdict_message = "⚠️ ALERTA: Hechos reales, pero estilo sensacionalista/clickbait"
    # elif rag_verdict == "UNCERTAIN":
    #      verdict_message = "RAG no pudo verificar (Falta información)"

    return {
        "label": final_verdict, # Usamos el veredicto final como label principal
        "confidence": 1.0 if rag_verdict != "UNCERTAIN" else 0.0, # Confianza simulada
        "explanation": explanation_text,
        "extracted_title": extracted_title,
        "rag_analysis": rag_result.get("analysis"),
        "retrieved_sources": rag_result.get("sources", []),
        "model_explanation": model_explanation,
        "text": text,
        "conflict_resolution": {
            "rag_verdict": rag_verdict,
            "final_verdict": final_verdict,
            "message": verdict_message
        }
    }
