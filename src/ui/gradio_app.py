import sys
import os
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import gradio as gr
from src.inference.pipeline import run_inference
import pandas as pd

def format_model_explanation(expl_dict):
    if not expl_dict:
        return "No disponible"
    
    top_words = expl_dict.get("top_words", [])
    scores = expl_dict.get("top_word_scores", [])
    
    if not top_words:
        return "No se encontraron palabras clave relevantes."
    
    lines = []
    lines.append("### Palabras más influyentes en la decisión:")
    for w, s in zip(top_words, scores):
        lines.append(f"- **{w}**: {s:.4f}")
        
    return "\n".join(lines)

def predict_fn(input_option, text_input, url_input):
    content = url_input if input_option == "URL" else text_input
    input_type_key = "url" if input_option == "URL" else "text"
    
    #  Validación explícita
    if not content or not content.strip():
        return ("<h3 style='color:red'>Error: El contenido no puede estar vacío.</h3>", "", "", pd.DataFrame(), "")
    
    if input_option == "Texto" and len(content.strip()) < 50:
        return ("<h3 style='color:orange'>Advertencia: Texto demasiado corto (mínimo 50 caracteres).</h3>", "", "", pd.DataFrame(), "")
        
    if input_option == "URL" and not content.lower().startswith("http"):
        return ("<h3 style='color:red'>Error: URL inválida (debe comenzar con http/https).</h3>", "", "", pd.DataFrame(), "")

    try:
        # Llamada al pipeline unificado
        result = run_inference(input_type_key, content)
        
        #  Diferenciar errores del pipeline
        if "error" in result:
            stage = result.get("error_stage", "unknown")
            error_msg = result.get("error", "Error desconocido")
            
            # Mapeo de errores a mensajes amigables
            friendly_msg = f"Error en etapa: {stage}"
            if stage == "paywall":
                friendly_msg = "<b>Paywall detectado:</b> No pudimos acceder al contenido completo de la noticia."
            elif stage == "invalid_url":
                friendly_msg = "<b>URL Inválida:</b> Verifica que el enlace sea correcto."
            elif stage == "connection_error":
                friendly_msg = "<b>Error de conexión:</b> No pudimos conectar con el sitio web."
            elif stage == "http_error":
                friendly_msg = f"<b>Error del servidor:</b> El sitio devolvió un error ({error_msg})."
            elif stage == "empty_article":
                friendly_msg = "<b>Artículo vacío:</b> No se encontró texto relevante en la página."
            elif stage == "too_short":
                friendly_msg = "<b>Texto insuficiente:</b> El artículo es demasiado corto para ser analizado."
            else:
                friendly_msg = f"<b>Error de procesamiento ({stage}):</b> {error_msg}"

            return (
                f"<div style='padding: 20px; background-color: #fff5f5; border: 1px solid red; border-radius: 8px; color: red;'><h3>{friendly_msg}</h3></div>", 
                "", 
                "", 
                pd.DataFrame(), 
                ""
            )
            
    except Exception as e:
        return (
            f"<h3 style='color:red'>Error interno: {str(e)}</h3>", 
            "", 
            "", 
            pd.DataFrame(), 
            ""
        )
    
    # Extract results
    label = result.get("label", "UNKNOWN")
    confidence = result.get("confidence", 0.0)
    rag_analysis = result.get("rag_analysis")
    extracted_title = result.get("extracted_title", "")
    sources = result.get("retrieved_sources", [])
    model_expl = result.get("model_explanation", {})
    full_text = result.get("text", "")
    conflict = result.get("conflict_resolution", {})
    
    # Formatting Label & Confidence ( D. Barra de confianza + E. Veredicto de Conflicto)
    
    # Definir colores y estilos según el veredicto
    if label == "REAL":
        color = "green"
        bg_color = "#e6fffa" # Light Green
    elif label == "FAKE":
        color = "red"
        bg_color = "#fff5f5" # Light Red
    elif label == "POSSIBLE_FAKE":
        color = "#dd6b20" # Orange
        bg_color = "#fffaf0" # Light Orange
    elif label.startswith("WARNING"):
        color = "#d69e2e" # Yellow/Dark Gold
        bg_color = "#fffff0" # Light Yellow
    else:
        color = "gray"
        bg_color = "#f7fafc"

    # Bloque de Alerta de Conflicto
    conflict_html = ""
    # final_verdict ya es igual a label en la nueva lógica del pipeline, pero mantenemos compatibilidad
    conflict_msg = conflict.get("message", "")
    
    # Si hay mensaje de conflicto o advertencia, lo mostramos
    if conflict_msg and (label.startswith("WARNING") or label == "POSSIBLE_FAKE" or label == "AMBIGUOUS"):
         conflict_html = f"""
        <div style="margin-top: 15px; padding: 15px; background-color: #fffaf0; border-left: 5px solid {color}; border-radius: 4px;">
            <h4 style="color: {color}; margin: 0;">Análisis Híbrido:</h4>
            <p style="margin: 5px 0 0 0; color: #2d3748;">{conflict_msg}</p>
        </div>
        """
    
    label_html = f"""
    <div style="text-align: center; padding: 20px; background-color: {bg_color}; border-radius: 10px; border: 2px solid {color};">
        <h1 style="color: {color}; margin: 0; font-size: 2.5em;">{label}</h1>
        <h3 style="color: #4a5568; margin: 10px 0 5px 0;">Confianza: {confidence:.2%} <span style="font-size: 0.8em; color: #718096;">(Modelo + RAG)</span></h3>
        
        <!-- Barra de Progreso Visual -->
        <div style="width: 100%; background-color: #e2e8f0; border-radius: 9999px; height: 12px; margin-top: 10px; overflow: hidden;">
            <div style="width: {confidence*100}%; background-color: {color}; height: 100%; border-radius: 9999px; transition: width 0.5s ease;"></div>
        </div>
        
        {conflict_html}
    </div>
    """
    
    # Formatting Extracted Info
    extracted_display = ""
    if input_option == "URL":
        extracted_display = f"### {extracted_title}\n\n{full_text[:1000]}..."
        if len(full_text) > 1000:
            extracted_display += " (truncado)"
    else:
        extracted_display = full_text[:1000] + "..." if len(full_text) > 1000 else full_text

    # Formatting Model Explanation
    expl_text = format_model_explanation(model_expl)
    
    # Formatting RAG
    rag_text = rag_analysis if rag_analysis else "No se pudo generar análisis comparativo."
    
    # Formatting Sources
    if sources:
        df_data = []
        for s in sources:
            df_data.append({
                "Fuente": s.get("source", "Desconocida"),
                "Score": round(s.get("score", 0), 4),
                "Fragmento": s.get("text", "")[:80] + "..."
            })
        sources_df = pd.DataFrame(df_data)
    else:
        sources_df = pd.DataFrame(columns=["Fuente", "Score", "Fragmento"])
        
    return (
        label_html,
        extracted_display,
        expl_text,
        sources_df,
        rag_text
    )

# UI Layout
with gr.Blocks(title="Detector de Fake News", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Detector de Fake News IA
        Ingrese el texto de una noticia o una URL para analizar su veracidad.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_type = gr.Radio(
                ["Texto", "URL"], 
                label="Tipo de entrada", 
                value="Texto"
            )
            
            text_input = gr.Textbox(
                label="Texto de la noticia", 
                placeholder="Pega aquí el contenido completo de la noticia...", 
                lines=8,
                visible=True
            )
            
            url_input = gr.Textbox(
                label="URL de la noticia", 
                placeholder="https://ejemplo.com/noticia", 
                visible=False
            )
            
            btn_analyze = gr.Button("Analizar Noticia", variant="primary", size="lg")
            
    # Sección de Resultados (Output)
    with gr.Row():
        # Columna Izquierda: Veredicto
        with gr.Column(scale=1):
            label_output = gr.HTML(label="Veredicto")
            extracted_output = gr.Textbox(label="Texto Analizado", lines=10, interactive=False)
            
        # Columna Derecha: Detalles
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("Explicación Modelo"):
                    expl_output = gr.Markdown()
                    
                with gr.TabItem("Fact-Checking (RAG)"):
                    txt_rag = gr.Markdown(label="Análisis LLM")
                    df_sources = gr.Dataframe(
                        headers=["Fuente", "Score", "Fragmento"],
                        label="Fuentes Recuperadas"
                    )

    # Eventos de visibilidad
    def toggle_inputs(choice):
        if choice == "Texto":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)
        
    input_type.change(toggle_inputs, input_type, [text_input, url_input])
    
    # Evento de análisis
    btn_analyze.click(
        fn=predict_fn,
        inputs=[input_type, text_input, url_input],
        outputs=[label_output, extracted_output, expl_output, df_sources, txt_rag]
    )

if __name__ == "__main__":
    demo.launch(share=False)
