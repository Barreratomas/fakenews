import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")

def load_dataset(filename, column_mapping=None):
    file_path = os.path.join(RAW_DIR, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No existe el archivo: {file_path}")

    df = pd.read_csv(file_path)

    if column_mapping:
        df = df.rename(columns=column_mapping)

    df.columns = [c.lower() for c in df.columns]

    if "text" not in df.columns:
        raise ValueError(f"{filename} no tiene columna 'text'")

    return df


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
