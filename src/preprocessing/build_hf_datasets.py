import os
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from src.preprocessing.clean_text import clean_text
from src.data.load_datasets import load_raw_data
from src.utils.logger import get_logger
from src.config import (
    PROCESSED_DATA_DIR, 
    DEFAULT_BASE_MODEL_NAME, 
    TOKENIZER_MAX_LENGTH, 
    SEED
)

logger = get_logger(__name__)


def to_hf_dataset(df, tokenizer, with_labels=True):
    # Asumimos que el texto ya viene limpio desde main()
    texts = df["text"].astype(str).tolist()

    enc = tokenizer(
        texts,
        truncation=True,
        max_length=TOKENIZER_MAX_LENGTH
    )

    data = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"]
    }

    if with_labels:
        data["labels"] = df["label"].astype(int).tolist()

    return Dataset.from_dict(data)


def save_dataset(ds, name):
    out_dir = PROCESSED_DATA_DIR / name
    os.makedirs(out_dir, exist_ok=True)
    ds.save_to_disk(out_dir)
    logger.info(f"Dataset guardado en {out_dir}")


def main():
    logger.info(f"Cargando tokenizer: {DEFAULT_BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL_NAME)

    # 1. Cargar Datos Crudos Unificados (Fake + True)
    df = load_raw_data()
    logger.info(f"Total registros cargados: {len(df)}")

    # 2. Limpieza
    logger.info("Limpiando textos...")
    df["text"] = df["text"].map(clean_text)
    
    # 3. Deduplicación
    len_before = len(df)
    df = df.drop_duplicates(subset=["text"])
    logger.info(f"Eliminados {len_before - len(df)} duplicados exactos.")

    # 4. Split (80% Train, 10% Val, 10% Test)
    # Primero separamos Test (10%)
    # Train+Val (90%) / Test (10%)
    train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=SEED, stratify=df["label"])
    
    # Train (89% de 90% ~ 80% total) / Val (11% de 90% ~ 10% total)
    # 0.111 * 0.9 ~ 0.10
    train_df, val_df = train_test_split(train_val_df, test_size=0.1111, random_state=SEED, stratify=train_val_df["label"])

    logger.info(f"Distribución Final: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 6. Guardar Datasets
    logger.info("Guardando datasets procesados...")
    save_dataset(to_hf_dataset(train_df, tokenizer), "train")
    save_dataset(to_hf_dataset(val_df, tokenizer), "val")
    save_dataset(to_hf_dataset(test_df, tokenizer), "test")

    logger.info("Proceso completado exitosamente.")


if __name__ == "__main__":
    main()
