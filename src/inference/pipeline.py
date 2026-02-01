from typing import Dict, Any, Tuple
from src.extraction.article_extractor import extract_article_from_url, ArticleExtractionError
from src.inference.predict import model_predict, generate_explanation
from src.rag.rag_pipeline import rag_fact_check
from src.utils.logger import get_logger
from src.config import (
    DEFAULT_EXPLANATION_METHOD,
    DEFAULT_MODEL_DIR
)

logger = get_logger(__name__)

def _resolve_conflict(model_label: str, rag_verdict: str) -> Tuple[str, str]:
    """
    Resuelve conflictos entre la predicción del modelo (DeBERTa) y la verificación RAG
    utilizando una tabla de decisiones determinista.
    
    Args:
        model_label: Etiqueta del modelo ('FAKE', 'REAL', 'AMBIGUOUS')
        rag_verdict: Veredicto del RAG ('FAKE', 'REAL', 'UNCERTAIN')
        
    Returns:
        Tuple[str, str]: (veredicto_final, mensaje_explicativo)
    """
    # Mapa de resolución de conflictos: (Model, RAG) -> (Final, Mensaje)
    conflict_map = {
        # Casos de Ambigüedad del Modelo
        ("AMBIGUOUS", "FAKE"): ("FAKE", "CONFIRMADO: El análisis de contenido y la verificación externa indican falsedad."),
        ("AMBIGUOUS", "REAL"): ("REAL", "VERIFICADO: La información ha sido corroborada por fuentes externas confiables."),
        ("AMBIGUOUS", "UNCERTAIN"): ("POSSIBLE_FAKE", "NO VERIFICADO: El contenido presenta características sospechosas sin evidencia externa concluyente."),
        
        # Casos de Contradicción Directa (High Confidence vs RAG)
        ("REAL", "FAKE"): ("WARNING_DISPUTED", "DESINFORMACIÓN SOFISTICADA: El texto es coherente pero contiene hechos refutados por fuentes externas."),
        ("FAKE", "REAL"): ("WARNING_SENSATIONALIST", "SENSACIONALISTA: Los hechos son correctos, pero presentados con un estilo engañoso o alarmista."),
    }
    
    # Intentar resolución específica
    if (model_label, rag_verdict) in conflict_map:
        return conflict_map[(model_label, rag_verdict)]
        
    # Comportamiento por defecto (Consistente o Fallback)
    final_verdict = model_label
    message = "Análisis consistente."
    
    # Si RAG falla en verificar (y no es un caso AMBIGUOUS ya cubierto), reportarlo
    if rag_verdict == "UNCERTAIN":
         message = "Verificación externa no disponible o inconclusa (Falta información)."
         
    return final_verdict, message

def run_inference(input_type: str, content: str) -> Dict[str, Any]:
    extracted_title = None
    text = ""
    
    # Obtener texto
    if input_type == "url":
        try:
            logger.info(f"Extrayendo artículo desde URL: {content}")
            article = extract_article_from_url(content)
            text = article["text"]
            extracted_title = article.get("title")
            # Si no se extrajo título, usar parte del texto
            if not extracted_title:
                extracted_title = text[:50].strip()
                
        # Manejar errores de extracción
        except ArticleExtractionError as e:
            logger.error(f"Error de extracción ({e.stage}): {e}")
            return {
                "error_stage": e.stage, 
                "error": str(e)
            }
        # Manejar errores generales
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

    # Clasificación fake / real
    try:
        logger.info("Ejecutando predicción del modelo...")
        clf_result = model_predict(text)
    # Manejar errores en predicción
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        return {
            "error_stage": "prediction",
            "error": str(e)
        }

    # RAG (Fact-Checking Asistido)
    try:
        logger.info("Ejecutando Fact-Checking (RAG)...")
        rag_result = rag_fact_check(text)
    # Manejar errores en RAG    
    except Exception as e:
        logger.error(f"Error en RAG: {e}")
        # En caso de error, devuelve error para que la UI lo maneje
        return {
            "error": str(e),
            "error_stage": "rag_error"
        }

    # Mapeo a la respuesta final
    explanation_text = rag_result.get("analysis", "")
    if not explanation_text:
        explanation_text = "No se pudo generar explicación."

    # Explicación del modelo (Keywords)
    model_explanation = generate_explanation(text, method=DEFAULT_EXPLANATION_METHOD, model_dir=DEFAULT_MODEL_DIR)

    # Resolución de Conflictos (DeBERTa vs RAG)
    model_label = clf_result["label"].upper()  # FAKE o REAL
    
    
    
    # Si no existe (versiones anteriores), fallback a UNCERTAIN
    rag_verdict = rag_result.get("verdict", "UNCERTAIN")
        
    # === Lógica de Resolución Híbrida (Refactorizada) ===
    final_verdict, verdict_message = _resolve_conflict(model_label, rag_verdict)

    return {
        "label": final_verdict, # Usamos el veredicto final como label principal para la UI
        "confidence": clf_result["confidence"],
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
