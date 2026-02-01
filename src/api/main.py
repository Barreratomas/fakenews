from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from src.schemas.request import PredictRequest
from src.schemas.response import PredictResponse
from src.inference.pipeline import run_inference
from src.utils.logger import get_logger
from src.config import API_HOST, API_PORT

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando API de Detección de Fake News...")
    yield
    logger.info("Apagando API...")

app = FastAPI(
    title="Fake News Detection API",
    version="1.0.0",
    description="API para detección de fake news a partir de texto o URL",
    lifespan=lifespan
)

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, request: Request):
    logger.info(f"Request recibido: {req.type} - {req.content[:50]}...")
    try:
        # run_inference gestiona la lógica de negocio y el registro interno
        result = run_inference(req.type, req.content)

        if "error" in result:
            logger.warning(f"Error en inferencia: {result}")
            # Si ocurre un error, FastAPI espera un código HTTP o una respuesta válida según el esquema.
            # Aunque PredictResponse tiene campos opcionales, si faltan datos requeridos (label/confidence), fallará.
            # Para errores fatales de procesamiento, lanzamos una excepción HTTP 400.
            raise HTTPException(status_code=400, detail=result)

        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error inesperado en endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
