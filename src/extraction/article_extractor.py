from typing import Dict
from newspaper import Article, Config
import validators
import requests
from urllib.parse import urlparse, unquote
from src.config import PAYWALL_KEYWORDS, MIN_ARTICLE_LENGTH

class ArticleExtractionError(Exception):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


# Headers que imitan un navegador real para evitar bloqueos (403/307)
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}


def extract_text_from_fragment(url: str) -> str:
    """
    Intenta extraer texto desde el fragmento de URL (Text Fragments).
    Formato: #:~:text=[prefix-,]textStart[,textEnd][,-suffix]
    """
    try:
        parsed = urlparse(url)
        if not parsed.fragment:
            return ""
        
        # Buscar directiva :~:text=
        fragment = parsed.fragment
        if ":~:text=" not in fragment:
            return ""
            
        # Extraer el valor de text=
        # El formato es complejo, pero simplificamos tomando lo que sigue a text=
        # hasta el próximo parámetro (&) o fin de string.
        # Nota: un fragmento puede tener múltiples directivas, pero text= es la principal para contenido.
        directives = fragment.split(":~:")
        text_directive = None
        for d in directives:
            if d.startswith("text="):
                text_directive = d[5:] # remover 'text='
                break
        
        if not text_directive:
            return ""
            
        # Decodificar URL encoding
        # El standard soporta start,end. Separado por coma.
        # Simplemente decodificamos todo y lo unimos con espacios si hay comas,
        # aunque la coma indica rango, para fake news detection concatenar es mejor que nada.
        # Ojo: la coma en el texto real está codificada? 
        # Spec: text=start,end -> start y end son percent-encoded.
        
        parts = text_directive.split(",")
        decoded_parts = [unquote(p) for p in parts]
        
        # Limpieza básica de prefijos/sufijos (dash)
        # Si hay guiones, el spec dice que son contexto.
        # Simplificación: devolvemos todo el texto decodificado unido.
        return " ... ".join(decoded_parts)
        
    except Exception:
        return ""

def extract_article_from_url(
    url: str,
    min_length: int = MIN_ARTICLE_LENGTH,
    language: str = "es",
    timeout: int = 15
) -> Dict[str, str]:
    """
    Extrae título y texto de una noticia desde un URL.
    Lanza ArticleExtractionError con stage específico.
    """

    # Validación básica de URL
    if not validators.url(url):
        raise ArticleExtractionError("invalid_url", "URL inválida")
        
    # Intentar extraer desde fragmento primero (intención explícita del usuario)
    fragment_text = extract_text_from_fragment(url)
    if fragment_text and len(fragment_text) > 50: # Si hay un fragmento sustancial
        return {
            "title": "Texto resaltado (Fragmento URL)",
            "text": fragment_text
        }

    # Verificar que el sitio responde
    try:
        r = requests.get(url, timeout=timeout, headers=BROWSER_HEADERS)
        if r.status_code >= 400:
            raise ArticleExtractionError(
                "http_error",
                f"HTTP status {r.status_code}"
            )
        
        # Detectar protección anti-bot / redirección JS (ej. Sucuri, Cloudflare)
        if "Javascript is required" in r.text or "You are being redirected" in r.text:
            raise ArticleExtractionError(
                "antibot", 
                "El sitio requiere JavaScript o tiene protección anti-bot compleja."
            )
            
    except requests.RequestException as e:
        raise ArticleExtractionError("connection_error", str(e))

    # Extracción con newspaper
    # Configurar newspaper para usar el mismo User-Agent y headers
    config = Config()
    config.browser_user_agent = BROWSER_HEADERS["User-Agent"]
    config.headers = BROWSER_HEADERS
    config.request_timeout = timeout
    
    article = Article(url, language=language, config=config)

    try:
        # Usamos el HTML ya descargado para evitar doble request y errores 403
        # Sin embargo, si requests siguió redirects (e.g. 307), r.text tiene el contenido final.
        # A veces newspaper prefiere descargar por su cuenta si el input_html es complejo,
        # pero input_html es más seguro si ya pasamos la barrera con requests.
        article.download(input_html=r.text)
    except Exception as e:
        raise ArticleExtractionError("download", str(e))

    try:
        article.parse()
    except Exception as e:
        raise ArticleExtractionError("parse", str(e))

    text = article.text.strip()
    title = article.title or ""

    # Validaciones post-extracción
    if not text:
        raise ArticleExtractionError("empty_article", "No se pudo extraer texto")

    if len(text) < min_length:
        raise ArticleExtractionError(
            "too_short",
            f"Artículo demasiado corto ({len(text)} chars)"
        )

    # Heurística simple de paywall
    lower_text = text.lower()
    if any(k in lower_text for k in PAYWALL_KEYWORDS):
        raise ArticleExtractionError("paywall", "Posible paywall detectado")

    return {
        "title": title.strip(),
        "text": text
    }
