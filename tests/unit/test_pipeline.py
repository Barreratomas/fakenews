import pytest
from unittest.mock import patch, MagicMock
from src.inference.pipeline import run_inference, _resolve_conflict

# TEST UNITARIO AVANZADO: Prueba el flujo lógico principal usando Mocks.
# NO carga modelos reales ni hace peticiones a internet.

class TestPipeline:
    
    # @patch sustituye las funciones reales por objetos falsos (Mocks) durante este test.
    # El orden de los argumentos en la función es inverso al orden de los decoradores.
    @patch("src.inference.pipeline.extract_article_from_url") # -> mock_extract
    @patch("src.inference.pipeline.model_predict")            # -> mock_predict
    @patch("src.inference.pipeline.rag_fact_check")           # -> mock_rag
    def test_run_inference_text_fake(self, mock_rag, mock_predict, mock_extract):
        """Prueba el flujo cuando el usuario ingresa TEXTO directo que resulta ser FALSO."""
        
        # 1. Configurar el comportamiento de los Mocks (Simulamos lo que harían las funciones reales)
        mock_predict.return_value = {
            "label": "FAKE",
            "confidence": 0.95,
            "explanation": "Fake explanation"
        }
        # Simulamos que RAG también dice que es falso
        mock_rag.return_value = {
            "verdict": "FAKE",
            "analysis": "RAG explanation", # Corregido: 'analysis' es la clave correcta en el código real
            "sources": []
        }
        
        # 2. Ejecutar la función real 'run_inference'
        # Internamente, esta función llamará a nuestros mocks en lugar de las funciones reales
        result = run_inference("text", "This is a fake text")
        
        # 3. Validar resultados (Aserciones)
        assert result["label"] == "FAKE"
        assert result["confidence"] == 0.95
        assert result["rag_analysis"] == "RAG explanation"
        
        # Verificar nuevo campo de resolución de conflictos
        assert "conflict_resolution" in result
        assert result["conflict_resolution"]["rag_verdict"] == "FAKE"
        assert result["conflict_resolution"]["final_verdict"] == "FAKE"
        
        # 4. Validar comportamiento
        # Como el input fue texto, NO se debió llamar al extractor de URL
        mock_extract.assert_not_called()

    @patch("src.inference.pipeline.extract_article_from_url")
    @patch("src.inference.pipeline.model_predict")
    @patch("src.inference.pipeline.rag_fact_check")
    def test_run_inference_url_real(self, mock_rag, mock_predict, mock_extract):
        """Prueba el flujo cuando el usuario ingresa una URL que resulta ser REAL."""
        
        # 1. Configurar Mocks
        # Simulamos que el extractor descarga exitosamente una noticia
        mock_extract.return_value = {
            "title": "Real News",
            "text": "This is a real news article content."
        }
        mock_predict.return_value = {
            "label": "REAL",
            "confidence": 0.99,
            "explanation": "Real explanation"
        }
        mock_rag.return_value = {
            "verdict": "REAL",
            "analysis": "Confirmed by sources",
            "sources": [{"url": "http://bbc.com"}]
        }
        
        # 2. Ejecutar con tipo "url"
        result = run_inference("url", "http://example.com/real")
        
        # 3. Validar
        assert result["label"] == "REAL"
        assert result["extracted_title"] == "Real News" # Debe venir del extractor
        assert len(result["retrieved_sources"]) == 1
        
        # 4. Validar que el extractor SÍ fue llamado esta vez
        mock_extract.assert_called_once()

    def test_resolve_conflict_logic(self):
        """Prueba pura de la lógica de decisión (Matriz de Confusión)."""
        # Aquí no necesitamos mocks porque _resolve_conflict es una función pura (sin efectos secundarios)
        
        # Caso: Modelo duda (AMBIGUOUS) pero RAG confirma que es FAKE -> Veredicto final FAKE
        verdict, msg = _resolve_conflict("AMBIGUOUS", "FAKE")
        assert verdict == "FAKE"
        assert "CONFIRMADO" in msg
        
        # Caso: Modelo dice REAL pero RAG dice FAKE -> Conflicto (ADVERTENCIA CONTROVERTIDA)
        verdict, msg = _resolve_conflict("REAL", "FAKE")
        assert verdict == "ADVERTENCIA CONTROVERTIDA"
        
        # Caso: Ambos coinciden -> Veredicto consistente
        verdict, msg = _resolve_conflict("REAL", "REAL")
        assert verdict == "REAL"
        assert msg == "Análisis consistente."

    @patch("src.inference.pipeline.model_predict")
    def test_cancellation_callback(self, mock_predict):
        """Prueba que el proceso se detiene si el usuario cancela (callback)."""
        
        mock_predict.return_value = {"label": "REAL", "confidence": 1.0}
        
        # Creamos un Mock que lanza una excepción cuando se le llama
        # Esto simula lo que hace el backend cuando detecta desconexión
        mock_cancel = MagicMock(side_effect=Exception("Cancelled"))
        
        # Verificamos que la excepción 'burbujea' hacia arriba
        with pytest.raises(Exception, match="Cancelled"):
            run_inference("text", "Some text", check_cancellation=mock_cancel)
            
        # Verificamos que el callback de cancelación fue consultado
        mock_cancel.assert_called()
