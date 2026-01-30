import os
from pathlib import Path

# ==========================================
# Directorios Base
# ==========================================
SRC_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# ==========================================
# Directorios de Datos
# ==========================================
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_DIR = PROCESSED_DATA_DIR / "train"
VAL_DATA_DIR = PROCESSED_DATA_DIR / "val"
TEST_DATA_DIR = PROCESSED_DATA_DIR / "test"

# ==========================================
# Directorios de Modelos
# ==========================================
# Solo usamos DeBERTa + LoRA
DEBERTA_LORA_DIR = MODELS_DIR / "deberta_lora"
DEFAULT_MODEL_DIR = DEBERTA_LORA_DIR

# ==========================================
# Configuración de Modelos (HuggingFace)
# ==========================================
DEFAULT_BASE_MODEL_NAME = os.environ.get("MODEL_NAME", "microsoft/mdeberta-v3-base")
SENTENCE_TRANSFORMER_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ==========================================
# Configuración RAG
# ==========================================
RAG_INDEX_DIR = MODELS_DIR / "rag_index"
RAG_INDEX_PATH = RAG_INDEX_DIR / "faiss.index"
RAG_METADATA_PATH = RAG_INDEX_DIR / "metadata.json"
RAG_INFO_PATH = RAG_INDEX_DIR / "index_info.json"
RAG_TOP_K = 5
# Umbral sugerido para considerar "relevante" un documento (heurística)
RAG_SIMILARITY_THRESHOLD = 0.4
# Fuente de RAG: SIEMPRE "web" para garantizar acceso a internet
RAG_SOURCE = "web"

# ==========================================
# Configuración de Extracción
# ==========================================
MIN_ARTICLE_LENGTH = 150
PAYWALL_KEYWORDS = [
    "suscríbete",
    "suscripción",
    "subscribe",
    "sign up to continue",
    "inicia sesión"
]

# ==========================================
# Configuración API
# ==========================================
API_HOST = "0.0.0.0"
API_PORT = 8000
