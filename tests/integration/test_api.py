from unittest.mock import patch, MagicMock
from src.schemas.response import PredictResponse

# TEST DE INTEGRACIÓN: Prueba los Endpoints HTTP de la API.
# Objetivo: Verificar que la API recibe peticiones JSON y responde correctamente.
# Nota: Seguimos mockeando la inferencia pesada (modelos) para que el test sea rápido.

def test_predict_text_success(client):
    """Verifica el endpoint POST /predict con texto simple."""
    
    # Preparamos una respuesta falsa perfecta para el pipeline
    mock_response = {
        "label": "FAKE",
        "confidence": 0.98,
        "explanation": "Test explanation",
        "extracted_title": "Test Title",
        "rag_analysis": "RAG result",
        "retrieved_sources": [],
        "model_explanation": None,
        "text": "Test content"
    }
    
    # Interceptamos la llamada interna a 'run_inference'
    with patch("src.api.main.run_inference", return_value=mock_response) as mock_run:
        # Simulamos una petición POST real desde un cliente
        response = client.post(
            "/predict",
            json={
                "type": "text", 
                "content": "This is a fake news", 
                "session_id": "test_session"
            }
        )
        
        # Validaciones de API
        assert response.status_code == 200 # Debe ser OK
        data = response.json()             # Convertimos respuesta a diccionario
        
        # Validamos que la API no rompió la estructura de datos
        assert data["label"] == "FAKE"
        assert data["confidence"] == 0.98
        
        # Confirmamos que la API realmente llamó al negocio
        mock_run.assert_called_once()

def test_predict_url_success(client):
    """Verifica el endpoint POST /predict con una URL."""
    
    mock_response = {
        "label": "REAL",
        "confidence": 0.95,
        "explanation": "Real news",
        "extracted_title": "Real Title",
        "rag_analysis": "RAG result",
        "retrieved_sources": [],
        "model_explanation": None,
        "text": "Extracted content"
    }
    
    with patch("src.api.main.run_inference", return_value=mock_response) as mock_run:
        response = client.post(
            "/predict",
            json={
                "type": "url", 
                "content": "http://example.com/news", 
                "session_id": "test_session_url"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "REAL"

def test_websocket_connection(client):
    """Verifica que el endpoint de WebSockets acepte conexiones y responda al Ping."""
    
    # client.websocket_connect abre una conexión persistente simulada
    with client.websocket_connect("/ws/monitor/session_123") as websocket:
        # Enviamos un mensaje de texto "ping" (como lo haría el frontend)
        websocket.send_text("ping")
        
        # Esperamos recibir respuesta
        data = websocket.receive_text()
        
        # La lógica del backend dice que si recibe "ping", responde "pong"
        assert data == "pong"
