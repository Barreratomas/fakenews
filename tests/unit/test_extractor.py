import pytest
from unittest.mock import patch, MagicMock
from src.extraction.article_extractor import extract_article_from_url, ArticleExtractionError

class TestArticleExtractor:
    
    @patch("src.extraction.article_extractor.requests.get")
    @patch("src.extraction.article_extractor.Article")
    def test_extract_success(self, mock_article_cls, mock_get):
        # 1. Mock de la respuesta HTTP
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>Contenido HTML dummy</html>"
        mock_get.return_value = mock_response

        # 2. Mock del objeto Article
        mock_instance = MagicMock()
        mock_instance.title = "Noticia Importante"
        # Usamos un texto largo para pasar la validación de longitud
        mock_instance.text = "Este es el contenido de la noticia. " * 10  
        mock_instance.publish_date = "2024-01-01"
        
        # Hacemos que la clase Article devuelva nuestra instancia falsa
        mock_article_cls.return_value = mock_instance
        
        # 3. Ejecutar la función
        result = extract_article_from_url("http://fake-news.com/article")
        
        # 4. Verificar resultados
        assert result["title"] == "Noticia Importante"
        assert len(result["text"]) > 150
        
        # Verificar llamadas
        mock_get.assert_called_once()
        mock_instance.download.assert_called_once()
        mock_instance.parse.assert_called_once()

    @patch("src.extraction.article_extractor.requests.get")
    @patch("src.extraction.article_extractor.Article")
    def test_extract_empty_text(self, mock_article_cls, mock_get):
        # Mock HTTP
        mock_get.return_value.status_code = 200
        
        # Simular artículo vacío
        mock_instance = MagicMock()
        mock_instance.text = ""  
        mock_article_cls.return_value = mock_instance
        
        # Verificar excepción
        with pytest.raises(ArticleExtractionError) as excinfo:
            extract_article_from_url("http://empty.com")
        
        assert "No se pudo extraer texto" in str(excinfo.value)
