import os
from pathlib import Path

# ==========================================
# Directorios Base
# ==========================================
SRC_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

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
# Configuración Global
# ==========================================
SEED = 42

# ==========================================
# Configuración de Modelos (HuggingFace)
# ==========================================
DEFAULT_BASE_MODEL_NAME = os.environ.get("MODEL_NAME", "microsoft/mdeberta-v3-base")
TOKENIZER_MAX_LENGTH = 512
DEFAULT_NUM_LABELS = 2
LABEL_MAP = {0: "FAKE", 1: "REAL"}

# ==========================================
# Configuración de Entrenamiento
# ==========================================
DEFAULT_TRAIN_EPOCHS = 10
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 0.01

# ==========================================
# Configuración de Inferencia
# ==========================================
# Umbrales de clasificación
FAKE_THRESHOLD_HIGH = 0.95  # >= 0.95 -> FAKE
FAKE_THRESHOLD_LOW = 0.50   # > 0.50 -> AMBIGUOUS

# Keywords para explicación de modelo (método por defecto)
DEFAULT_EXPLANATION_METHOD = "attention"

# ==========================================
# Configuración RAG
# ==========================================
RAG_INDEX_DIR = MODELS_DIR / "rag_index"
RAG_INDEX_PATH = RAG_INDEX_DIR / "faiss.index"
RAG_METADATA_PATH = RAG_INDEX_DIR / "metadata.json"
RAG_INFO_PATH = RAG_INDEX_DIR / "index_info.json"

RAG_LLM_NAME = "google/flan-t5-base"
SENTENCE_TRANSFORMER_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

RAG_TOP_K = 3
# Umbral sugerido para considerar "relevante" un documento (heurística)
RAG_SIMILARITY_THRESHOLD = 0.4
# Fuente de RAG: SIEMPRE "web" para garantizar acceso a internet
RAG_SOURCE = "web"

# Keywords para detectar veredicto de RAG
RAG_VERDICT_FAKE_KEYWORDS = ["contradicted", "false", "incorrect", "unsupported", "fake", "hoax"]
RAG_VERDICT_REAL_KEYWORDS = ["supported", "true", "correct", "confirmed", "accurate"]

# Parámetros de generación LLM
RAG_GEN_PARAMS = {
    "max_length": 200,
    "do_sample": False,
    "num_beams": 4,
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 1.2
}

# Límites de texto para RAG
RAG_CONTEXT_MAX_LENGTH = 500
RAG_CLAIM_MAX_LENGTH = 800
RAG_EMBEDDING_BATCH_SIZE = 64
RAG_INDEX_MIN_TEXT_LEN = 50

# Prompt Template
RAG_PROMPT_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Claim: {claim}\n\n"
    "Task: Compare the Claim with the Context. Does the Context confirm the SPECIFIC events and reasons described in the Claim?\n"
    "- If the Context describes the SAME event but with DIFFERENT reasons or details (e.g. claim says 'mind control', context says 'protests'), answer 'CONTRADICTED'.\n"
    "- If the Context confirms the main points, answer 'SUPPORTED'.\n"
    "- If the Context is unrelated, answer 'NOT ENOUGH INFO'.\n\n"
    "Answer:"
)

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
