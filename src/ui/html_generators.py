def get_label_styles(label):
    """Retorna los colores y estilos CSS según el tipo de etiqueta."""
    if label == "REAL":
        return {
            "color": "green",
            "bg_color": "#e6fffa",  # Light Green
            "border_color": "green"
        }
    elif label == "FAKE":
        return {
            "color": "red", 
            "bg_color": "#fff5f5",  # Light Red
            "border_color": "red"
        }
    elif label == "POSIBLE FALSO":
        return {
            "color": "#dd6b20",  # Orange
            "bg_color": "#fffaf0",  # Light Orange
            "border_color": "#dd6b20"
        }
    elif label.startswith("WARNING") or label.startswith("ADVERTENCIA"):
        return {
            "color": "#d69e2e",  # Yellow/Dark Gold
            "bg_color": "#fffff0",  # Light Yellow
            "border_color": "#d69e2e"
        }
    else:
        return {
            "color": "gray",
            "bg_color": "#f7fafc",
            "border_color": "gray"
        }

def generate_conflict_html(conflict_msg, color):
    """Genera el HTML para el bloque de alerta de conflicto."""
    if not conflict_msg:
        return ""
    
    return f"""
    <div style="margin-top: 15px; padding: 15px; background-color: #fffaf0; border-left: 5px solid {color}; border-radius: 4px;">
        <h4 style="color: {color}; margin: 0;">Análisis Híbrido:</h4>
        <p style="margin: 5px 0 0 0; color: #2d3748;">{conflict_msg}</p>
    </div>
    """

def generate_label_html(label, confidence, conflict_msg=""):
    """Genera el HTML completo para mostrar el veredicto con estilos."""
    styles = get_label_styles(label)
    conflict_html = generate_conflict_html(conflict_msg, styles["color"])
    
    return f"""
    <div style="text-align: center; padding: 20px; background-color: {styles['bg_color']}; border-radius: 10px; border: 2px solid {styles['border_color']};">
        <h1 style="color: {styles['color']}; margin: 0; font-size: 2.5em;">{label}</h1>
        <h3 style="color: #4a5568; margin: 10px 0 5px 0;">Confianza: {confidence:.2%} <span style="font-size: 0.8em; color: #718096;">(Modelo + RAG)</span></h3>
        
        <!-- Barra de Progreso Visual -->
        <div style="width: 100%; background-color: #e2e8f0; border-radius: 9999px; height: 12px; margin-top: 10px; overflow: hidden;">
            <div style="width: {confidence*100}%; background-color: {styles['color']}; height: 100%; border-radius: 9999px; transition: width 0.5s ease;"></div>
        </div>
        
        {conflict_html}
    </div>
    """