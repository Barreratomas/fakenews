import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import gradio as gr
from src.inference.pipeline import run_inference
from src.ui.formatters import format_model_explanation, format_sources_dataframe, format_extracted_text
from src.ui.input_validators import validate_input, handle_pipeline_error
from src.ui.html_generators import generate_label_html
from src.ui.gradio_config import create_gradio_interface

def predict_fn(input_option, text_input, url_input):
    """Función principal de predicción que coordina el análisis de noticias."""
    content = url_input if input_option == "URL" else text_input
    input_type_key = "url" if input_option == "URL" else "text"
    
    # Validación de entrada
    is_valid, error_message = validate_input(input_option, content)
    if not is_valid:
        return (error_message, "")
    
    try:
        # Llamada al pipeline unificado
        result = run_inference(input_type_key, content)
        
        # Manejo de errores del pipeline
        if "error" in result:
            return handle_pipeline_error(result)
            
    except Exception as e:
        return (f"<h3 style='color:red'>Error interno: {str(e)}</h3>", "")
    
    # Extracción de resultados
    label = result.get("label", "UNKNOWN")
    confidence = result.get("confidence", 0.0)
    rag_analysis = result.get("rag_analysis")
    extracted_title = result.get("extracted_title", "")
    sources = result.get("retrieved_sources", [])
    model_expl = result.get("model_explanation", {})
    full_text = result.get("text", "")
    conflict = result.get("conflict_resolution", {})
    
    # Generación de HTML para el veredicto
    conflict_msg = conflict.get("message", "")
    label_html = generate_label_html(label, confidence, conflict_msg)
    
    # Formateo de resultados (solo lo necesario para mostrar)
    extracted_display = format_extracted_text(input_option, extracted_title, full_text)
    
    return (
        label_html,
        extracted_display
    )

def main():
    """Función principal que crea y lanza la interfaz de Gradio."""
    demo = create_gradio_interface(predict_fn)
    demo.launch(share=False, theme=gr.themes.Soft())

if __name__ == "__main__":
    main()