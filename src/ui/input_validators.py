import pandas as pd

def validate_input(input_option, content):
    """Valida el contenido de entrada según el tipo (texto o URL)."""
    if not content or not content.strip():
        return False, "<h3 style='color:red'>Error: El contenido no puede estar vacío.</h3>"
    
    if input_option == "Texto" and len(content.strip()) < 50:
        return False, "<h3 style='color:orange'>Advertencia: Texto demasiado corto (mínimo 50 caracteres).</h3>"
    
    if input_option == "URL" and not content.lower().startswith("http"):
        return False, "<h3 style='color:red'>Error: URL inválida (debe comenzar con http/https).</h3>"
    
    return True, None

def get_error_message(stage, error_msg):
    """Mapea errores del pipeline a mensajes amigables para el usuario."""
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
    
    return f"<div style='padding: 20px; background-color: #fff5f5; border: 1px solid red; border-radius: 8px; color: red;'><h3>{friendly_msg}</h3></div>"

def handle_pipeline_error(result):
    """Procesa errores del pipeline y retorna la respuesta adecuada."""
    stage = result.get("error_stage", "unknown")
    error_msg = result.get("error", "Error desconocido")
    
    error_html = get_error_message(stage, error_msg)
    
    return (error_html, "")