from typing import Dict
from newspaper import Article
import validators
import requests
from src.config import PAYWALL_KEYWORDS, MIN_ARTICLE_LENGTH

class ArticleExtractionError(Exception):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


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

    # Verificar que el sitio responde
    try:
        r = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0"
        })
        if r.status_code >= 400:
            raise ArticleExtractionError(
                "http_error",
                f"HTTP status {r.status_code}"
            )
    except requests.RequestException as e:
        raise ArticleExtractionError("connection_error", str(e))

    # Extracción con newspaper
    article = Article(url, language=language)

    try:
        # Usamos el HTML ya descargado para evitar doble request y errores 403
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
