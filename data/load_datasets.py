import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset

def _base_dir():
    return os.path.dirname(os.path.abspath(__file__))

def load_dataset(filename, column_mapping=None):
    base_dir = _base_dir()
    file_path = os.path.join(base_dir, "raw", filename)
    if not os.path.exists(file_path):
        print(f"Error: archivo no encontrado en {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error al leer {filename}: {e}")
        return None
    if column_mapping:
        df = df.rename(columns=column_mapping)
    df.columns = [c.lower() for c in df.columns]
    if "text" not in df.columns:
        print(f"Error: falta columna 'text' en {filename}. Columnas: {df.columns.tolist()}")
        return None
    if "label" in df.columns:
        df = df.dropna(subset=["text", "label"])
    else:
        df = df.dropna(subset=["text"])
    return df

def normalize_label_series(s):
    def map_one(x):
        if pd.isna(x):
            return None
        v = str(x).strip().lower()
        if v in {"1", "true", "real", "verdadero"}:
            return 1
        if v in {"0", "false", "fake", "falso"}:
            return 0
        try:
            return int(v)
        except Exception:
            return None
    out = s.map(map_one)
    return out

def clean_text(x):
    if not isinstance(x, str):
        return ""
    x = x.strip()
    x = " ".join(x.split())
    return x

def split_train_val(df, test_size=0.2, random_state=42):
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["label"])

def to_hf_dataset(df, tokenizer, with_labels=True, max_length=256):
    texts = df["text"].astype(str).map(clean_text).tolist()
    enc = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)
    data = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    if with_labels and "label" in df.columns:
        data["labels"] = df["label"].astype(int).tolist()
    return Dataset.from_dict(data)

def save_dataset(ds, name):
    out_dir = os.path.join(_base_dir(), "processed", name)
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    ds.save_to_disk(out_dir)
    print(f"Guardado: {out_dir}")

if __name__ == "__main__":
    print("Paso 3: Preprocesamiento NLP")
    tokenizer_name = os.environ.get("TOKENIZER_NAME", "roberta-base")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Error cargando tokenizer {tokenizer_name}: {e}")
        raise

    print("Cargando train.csv")
    train_df = load_dataset("train.csv")
    if train_df is None:
        raise SystemExit(1)
    train_df["text"] = train_df["text"].astype(str).map(clean_text)
    train_df["label"] = normalize_label_series(train_df["label"])
    train_df = train_df.dropna(subset=["label"]).astype({"label": int})
    tr_df, val_df = split_train_val(train_df, test_size=0.2)
    print(f"Train: {len(tr_df)} | Val: {len(val_df)}")
    ds_train = to_hf_dataset(tr_df, tokenizer, with_labels=True)
    ds_val = to_hf_dataset(val_df, tokenizer, with_labels=True)
    save_dataset(ds_train, "train")
    save_dataset(ds_val, "val")

    print("Cargando test.csv")
    test_df = load_dataset("test.csv")
    if test_df is not None:
        test_df["text"] = test_df["text"].astype(str).map(clean_text)
        ds_test = to_hf_dataset(test_df, tokenizer, with_labels=False)
        save_dataset(ds_test, "test")

    print("Cargando fakes1000.csv")
    fakes_mixed = load_dataset("fakes1000.csv", column_mapping={"class": "label", "Text": "text"})
    if fakes_mixed is not None and "label" in fakes_mixed.columns:
        fakes_mixed["text"] = fakes_mixed["text"].astype(str).map(clean_text)
        fakes_mixed["label"] = normalize_label_series(fakes_mixed["label"])
        fakes_mixed = fakes_mixed.dropna(subset=["label"]).astype({"label": int})
        ds_fakes_mixed = to_hf_dataset(fakes_mixed, tokenizer, with_labels=True)
        save_dataset(ds_fakes_mixed, "fakes1000")

    print("Cargando onlyfakes1000.csv")
    onlyfakes = load_dataset("onlyfakes1000.csv")
    if onlyfakes is not None:
        onlyfakes["text"] = onlyfakes["text"].astype(str).map(clean_text)
        onlyfakes["label"] = 0
        ds_onlyfakes = to_hf_dataset(onlyfakes, tokenizer, with_labels=True)
        save_dataset(ds_onlyfakes, "onlyfakes1000")

    print("Cargando onlytrue1000.csv")
    onlytrue = load_dataset("onlytrue1000.csv")
    if onlytrue is not None:
        onlytrue["text"] = onlytrue["text"].astype(str).map(clean_text)
        onlytrue["label"] = 1
        ds_onlytrue = to_hf_dataset(onlytrue, tokenizer, with_labels=True)
        save_dataset(ds_onlytrue, "onlytrue1000")
