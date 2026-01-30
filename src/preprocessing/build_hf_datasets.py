import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from datasets import Dataset
from transformers import AutoTokenizer
from src.preprocessing.clean_text import clean_text
from src.data.load_datasets import load_raw_data
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, DEFAULT_BASE_MODEL_NAME

TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME", DEFAULT_BASE_MODEL_NAME)
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

    # 1. Cargar Datos Crudos Unificados (Fake + True)
    df = load_raw_data()
    print(f"Total registros cargados: {len(df)}")

    # 2. Cargar Datos Aumentados (si existen)
    aug_path = os.path.join(RAW_DATA_DIR, "augmented_train.csv")
    if os.path.exists(aug_path):
        print(f"Cargando datos aumentados desde {aug_path}...")
        try:
            aug_df = pd.read_csv(aug_path)
            # Asegurar columnas
            if "text" in aug_df.columns and "label" in aug_df.columns:
                df = pd.concat([df, aug_df[["text", "label"]]], ignore_index=True)
                print(f"✔ Añadidos {len(aug_df)} registros aumentados.")
            else:
                print("⚠ augmented_train.csv no tiene columnas 'text' y 'label'. Ignorando.")
        except Exception as e:
            print(f"⚠ Error cargando aumentados: {e}")

    # 3. Limpieza
    print("Limpiando textos...")
    df["text"] = df["text"].map(clean_text)
    
    # 4. Deduplicación
    len_before = len(df)
    df = df.drop_duplicates(subset=["text"])
    print(f"⬇ Eliminados {len_before - len(df)} duplicados exactos.")

    # 5. Split (80% Train, 10% Val, 10% Test)
    # Primero separamos Test (10%)
    from sklearn.model_selection import train_test_split
    
    # Train+Val (90%) / Test (10%)
    train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label"])
    
    # Train (89% de 90% ≈ 80% total) / Val (11% de 90% ≈ 10% total)
    # 0.111 * 0.9 ≈ 0.10
    train_df, val_df = train_test_split(train_val_df, test_size=0.1111, random_state=42, stratify=train_val_df["label"])

    print(f"Distribución Final: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 6. Guardar Datasets
    print("Guardando datasets procesados...")
    save_dataset(to_hf_dataset(train_df, tokenizer), "train")
    save_dataset(to_hf_dataset(val_df, tokenizer), "val")
    save_dataset(to_hf_dataset(test_df, tokenizer), "test")

    print("✔ Proceso completado exitosamente.")


if __name__ == "__main__":
    main()
