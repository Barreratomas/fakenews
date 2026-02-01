import os
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")

def load_raw_data():
    """
    Carga Fake.csv y True.csv, asigna etiquetas y combina título+texto.
    Returns: pd.DataFrame unificado con columnas [text, label]
    """
    fake_path = os.path.join(RAW_DIR, "Fake.csv")
    true_path = os.path.join(RAW_DIR, "True.csv")

    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        raise FileNotFoundError(f"Faltan archivos raw: {fake_path} o {true_path}")

    logger.info("Cargando Fake.csv y True.csv...")
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    # Asignar etiquetas
    df_fake["label"] = 0
    df_true["label"] = 1

    # Combinar Title + Text para mejor contexto
    # Rellenar nulos para evitar errores
    df_fake["title"] = df_fake["title"].fillna("")
    df_fake["text"] = df_fake["text"].fillna("")
    df_true["title"] = df_true["title"].fillna("")
    df_true["text"] = df_true["text"].fillna("")

    df_fake["text"] = df_fake["title"] + " " + df_fake["text"]
    df_true["text"] = df_true["title"] + " " + df_true["text"]

    # Concatenar
    df = pd.concat([df_fake, df_true], ignore_index=True)
    
    # Mezclar aleatoriamente
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df[["text", "label"]]



def normalize_labels(series):
    def map_label(x):
        if pd.isna(x):
            return None
        v = str(x).strip().lower()
        if v in {"1", "true", "real", "verdadero"}:
            return 1
        if v in {"0", "false", "fake", "falso"}:
            return 0
        try:
            iv = int(v)
            return iv if iv in (0, 1) else None
        except Exception:
            return None

    return series.map(map_label)
