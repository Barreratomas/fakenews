from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Dict, Any
import sys
import logging
import time
from fastapi.responses import JSONResponse
from pathlib import Path
from datetime import datetime


from src.inference.pipeline import run_inference
from src.schemas.request import PredictRequest
from src.schemas.response import PredictResponse
from src.config import API_HOST, API_PORT

# Configure Logger for Connection Monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ConnectionMonitor")

app = FastAPI(title="Fake News Detection API", version="1.0")

class ConnectionManager:
    """
    Gestor profesional de conexiones WebSocket para monitoreo en tiempo real.
    Mantiene un registro de clientes activos y notifica eventos de conexión/desconexión.
    """
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.last_disconnect_times: Dict[str, float] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        
        msg = f"Cliente conectado [{session_id}] desde {websocket.client.host}:{websocket.client.port} at {datetime.now().isoformat()}"
        logger.info(msg)
        print(msg, flush=True)

    def disconnect(self, session_id: str, websocket: WebSocket):
        # Solo eliminar de active_connections si es el socket actual
        if session_id in self.active_connections:
            if self.active_connections[session_id] == websocket:
                del self.active_connections[session_id]
        
        # Registrar SIEMPRE el tiempo de desconexión para invalidar tareas anteriores
        self.last_disconnect_times[session_id] = time.time()
            
        msg = f"Cliente desconectado [{session_id}] (Cierre/Reinicio) desde {websocket.client.host}:{websocket.client.port} at {datetime.now().isoformat()}"
        logger.info(msg)
        print(msg, flush=True)

    def should_cancel(self, session_id: str, task_start_time: float) -> bool:
        """
        Verifica si se debe cancelar una tarea basándose en tiempos.
        Si hubo una desconexión DESPUÉS de que la tarea inició, se debe cancelar.
        """
        if session_id not in self.last_disconnect_times:
            return False
        
        last_disconnect = self.last_disconnect_times[session_id]
        return last_disconnect > task_start_time

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/monitor/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    Endpoint WebSocket para monitoreo de estado de conexión ("Heartbeat").
    El frontend mantiene esta conexión abierta; si se cierra, el backend loguea el evento.
    """
    await manager.connect(websocket, session_id)
    try:
        while True:
            # Esperamos mensajes del cliente (ping/heartbeat) para mantener viva la conexión
            # y detectar desconexiones abruptas.
            data = await websocket.receive_text()
            # Opcional: Responder al ping para confirmar que el servidor sigue vivo
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(session_id, websocket)
    except Exception as e:
        logger.error(f"Error en conexión WebSocket: {e}")
        manager.disconnect(session_id, websocket)

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Endpoint to analyze news (text or URL) and detect if it's fake.
    """
    start_time = time.time()
    logger.info(f"Recibida solicitud de predicción. Session ID: {request.session_id}")
    try:
        # Define cancellation check callback
        def check_cancellation():
            if request.session_id:
                if manager.should_cancel(request.session_id, start_time):
                    msg = f"Tarea cancelada por desconexión del usuario [{request.session_id}]"
                    logger.warning(msg)
                    raise Exception(msg)
            else:
                pass
                # logger.info("🔍 Verificando cancelación: Sin Session ID")

        # Run inference using the existing pipeline with cancellation check
        result = run_inference(
            input_type=request.type, 
            content=request.content,
            check_cancellation=check_cancellation
        )
        
        # Check for errors returned by the pipeline
        if "error" in result:
             return JSONResponse(content=result, status_code=200)
        
        return result
        
    except Exception as e:
        # If the exception message indicates cancellation, we could return a specific status code
        # or just let it be 500. For now, logging it is important.
        if "Tarea cancelada" in str(e):
             logger.warning(f"Pipeline interrumpido: {e}")
             raise HTTPException(status_code=499, detail=str(e)) # 499 Client Closed Request (Nginx style)
        
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
