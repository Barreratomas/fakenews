import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from datasets import Dataset
from transformers import AutoTokenizer
from src.preprocessing.clean_text import clean_text
from src.preprocessing.split_data import split_train_val
from src.data.load_datasets import load_dataset, normalize_labels
from src.config import PROCESSED_DATA_DIR

TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME", "microsoft/deberta-v3-base")
MAX_LENGTH = 512


def to_hf_dataset(df, tokenizer, with_labels=True):
    # Asumimos que el texto ya viene limpio desde main()
    texts = df["text"].astype(str).tolist()

    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
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
    print(f"✔ Dataset guardado en {out_dir}")


def main():
    print(f"Cargando tokenizer: {TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # ---- TRAIN / VAL ----
    print("Procesando dataset de entrenamiento...")
    train_df = load_dataset("train.csv")
    
    # Limpieza
    train_df["text"] = train_df["text"].map(clean_text)
    
    # Deduplicación
    len_before = len(train_df)
    train_df = train_df.drop_duplicates(subset=["text"])
    print(f"⬇ Eliminados {len_before - len(train_df)} duplicados exactos en train.csv")

    train_df["label"] = normalize_labels(train_df["label"])
    train_df = train_df.dropna(subset=["label"])
    train_df = train_df[train_df["label"].isin([0, 1])].astype({"label": int})

    tr_df, val_df = split_train_val(train_df)

    save_dataset(to_hf_dataset(tr_df, tokenizer), "train")
    save_dataset(to_hf_dataset(val_df, tokenizer), "val")

    # ---- TEST ----
    test_df = load_dataset("test.csv")
    test_df["text"] = test_df["text"].map(clean_text)
    save_dataset(to_hf_dataset(test_df, tokenizer, with_labels=False), "test")

    # ---- ONLY FAKES ----
    onlyfakes = load_dataset("onlyfakes1000.csv")
    onlyfakes["text"] = onlyfakes["text"].map(clean_text)
    onlyfakes["label"] = 0
    save_dataset(to_hf_dataset(onlyfakes, tokenizer), "onlyfakes1000")

    # ---- ONLY TRUE ----
    onlytrue = load_dataset("onlytrue1000.csv")
    onlytrue["text"] = onlytrue["text"].map(clean_text)
    onlytrue["label"] = 1
    save_dataset(to_hf_dataset(onlytrue, tokenizer), "onlytrue1000")


if __name__ == "__main__":
    main()
