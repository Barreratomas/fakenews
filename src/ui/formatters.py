import pandas as pd

def format_model_explanation(expl_dict):
    """Formatea la explicación del modelo para mostrar palabras clave y sus scores."""
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

def format_sources_dataframe(sources):
    """Convierte las fuentes recuperadas en un DataFrame para mostrar en la UI."""
    if sources:
        df_data = []
        for s in sources:
            df_data.append({
                "Fuente": s.get("source", "Desconocida"),
                "Score": round(s.get("score", 0), 4),
                "Fragmento": s.get("text", "")[:80] + "..."
            })
        return pd.DataFrame(df_data)
    else:
        return pd.DataFrame(columns=["Fuente", "Score", "Fragmento"])

def format_extracted_text(input_option, extracted_title, full_text):
    """Formatea el texto extraído para mostrar en la UI."""
    if input_option == "URL":
        extracted_display = f"### {extracted_title}\n\n{full_text[:1000]}..."
        if len(full_text) > 1000:
            extracted_display += " (truncado)"
    else:
        extracted_display = full_text[:1000] + "..." if len(full_text) > 1000 else full_text
    
    return extracted_display