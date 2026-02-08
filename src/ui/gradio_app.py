import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import gradio as gr
import requests
import json
from src.config import API_HOST, API_PORT
from src.ui.formatters import format_model_explanation, format_sources_dataframe, format_extracted_text
from src.ui.input_validators import validate_input, handle_pipeline_error
from src.ui.html_generators import generate_label_html
from src.ui.gradio_config import create_gradio_interface, CSS

# Configuración de conexión API
API_URL = f"http://127.0.0.1:{API_PORT}/predict"  
WS_URL = f"ws://127.0.0.1:{API_PORT}/ws/monitor"  



JS_WEBSOCKET_CLIENT = f"""
<script>
    (function() {{
        // Obtener o generar Session ID único
        let sessionId = sessionStorage.getItem("custom_session_id");
        if (!sessionId) {{
            sessionId = Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
            sessionStorage.setItem("custom_session_id", sessionId);
        }}
        
        const wsBaseUrl = "{WS_URL}";
        const wsUrl = wsBaseUrl + "/" + sessionId;
        
        let socket;
        let reconnectInterval = 5000;

        function connect() {{
            console.log("Intentando conectar al WebSocket de monitoreo...", wsUrl);
            socket = new WebSocket(wsUrl);

            socket.onopen = function(e) {{

                // Enviar ping periódico para mantener viva la conexión
                setInterval(() => {{
                    if (socket.readyState === WebSocket.OPEN) {{
                        socket.send("ping");
                    }}
                }}, 30000);
            }};

            socket.onmessage = function(event) {{
                // Ignorar pongs silenciosamente
            }};

            socket.onclose = function(event) {{
                if (event.wasClean) {{
                    console.log("");
                }} else {{
                    console.log("");
                    setTimeout(connect, reconnectInterval);
                }}
            }};

            socket.onerror = function(error) {{
                console.log("");
            }};
            
            // Detectar cierre de página explícito
            window.addEventListener('beforeunload', function () {{
                if (socket.readyState === WebSocket.OPEN) {{
                    socket.close();
                }}
            }});
        }}

        // Iniciar conexión al cargar
        if (document.readyState === 'complete') {{
            connect();
        }} else {{
            window.addEventListener('load', connect);
        }}
    }})();
</script>
"""

def predict_fn(input_option, text_input, url_input, session_id=None):
    """Función principal de predicción que consume la API del backend."""
    content = url_input if input_option == "URL" else text_input
    input_type_key = "url" if input_option == "URL" else "text"
    
    # Validación de entrada local
    is_valid, error_message = validate_input(input_option, content)
    if not is_valid:
        return (error_message, "")
    
    try:
        # Preparar payload para la API
        payload = {
            "type": input_type_key,
            "content": content,
            "session_id": session_id
        }
        
        # Llamada a la API
        response = requests.post(API_URL, json=payload, timeout=120)
        
        # Verificar si la API devolvió error HTTP
        if response.status_code != 200:
            return (f"<h3 style='color:red'>Error API ({response.status_code}): {response.text}</h3>", "")
            
        result = response.json()
   
        
        # Manejo de errores del pipeline (devueltos en el JSON)
        if "error" in result:
            return handle_pipeline_error(result)
            
    except requests.exceptions.ConnectionError:
        return (f"<h3 style='color:red'>Error de conexión: No se pudo conectar con el Backend ({API_URL}). Asegúrate de que la API esté corriendo.</h3>", "")
    except Exception as e:
        return (f"<h3 style='color:red'>Error interno UI: {str(e)}</h3>", "")
    
    # Extracción de resultados
    label = result.get("label", "UNKNOWN")
    confidence = result.get("confidence", 0.0)
    extracted_title = result.get("extracted_title", "")
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
    # Creamos la interfaz
    demo = create_gradio_interface(predict_fn)
    
    # Lanzamos pasando el script JS en 'head' y el CSS personalizado
    demo.launch(share=False, theme=gr.themes.Soft(), css=CSS, head=JS_WEBSOCKET_CLIENT)
    
if __name__ == "__main__":
    main()