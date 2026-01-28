import re
import html

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    # Decodificar entidades HTML (&amp; -> &)
    text = html.unescape(text)
    
    # Eliminar etiquetas HTML
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Eliminar URLs (http/https/www)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Normalizar espacios
    text = text.strip()
    text = " ".join(text.split())
    
    return text
