import os
import json
import shutil
import subprocess
import argparse
from pathlib import Path
from zipfile import ZipFile

# Configuración
KAGGLE_USER = "" # Se llenará dinámicamente desde kaggle.json
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = PROJECT_ROOT / "build_kaggle"
SRC_ZIP = BUILD_DIR / "fake_news_src.zip"
DATA_ZIP = BUILD_DIR / "fake_news_data.zip"

def check_kaggle_auth():
    """Verifica que kaggle.json exista y carga el usuario."""
    global KAGGLE_USER
    kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_config.exists():
        print(f"❌ Error: No se encontró {kaggle_config}")
        print("   -> Ve a https://www.kaggle.com/settings -> API -> Create New Token")
        print("   -> Guarda el archivo en ~/.kaggle/kaggle.json")
        exit(1)
    
    with open(kaggle_config) as f:
        data = json.load(f)
        KAGGLE_USER = data["username"]
    print(f"✅ Autenticado como: {KAGGLE_USER}")

def prepare_build_dir():
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir()

def zip_source_code():
    print("📦 Empaquetando código fuente (src/)...")
    with ZipFile(SRC_ZIP, 'w') as zipf:
        for root, dirs, files in os.walk(PROJECT_ROOT / "src"):
            for file in files:
                file_path = Path(root) / file
                # Mantener estructura relativa dentro del zip
                arcname = file_path.relative_to(PROJECT_ROOT)
                if "__pycache__" not in str(arcname):
                    zipf.write(file_path, arcname)
        
        # Agregar requirements.txt
        zipf.write(PROJECT_ROOT / "requirements.txt", "requirements.txt")
    print(f"   -> {SRC_ZIP.name} creado ({SRC_ZIP.stat().st_size / 1024:.2f} KB)")

def upload_dataset(zip_path, title, slug):
    """Sube o actualiza un dataset en Kaggle."""
    dataset_dir = BUILD_DIR / slug
    dataset_dir.mkdir(exist_ok=True)
    shutil.copy(zip_path, dataset_dir)
    
    # Crear metadata
    meta = {
        "title": title,
        "id": f"{KAGGLE_USER}/{slug}",
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(dataset_dir / "dataset-metadata.json", "w") as f:
        json.dump(meta, f)
    
    print(f"🚀 Subiendo dataset {slug}...")
    # Intentar crear, si falla, intentar versionar (actualizar)
    cmd_create = f"kaggle datasets create -p {dataset_dir} -u"
    cmd_update = f"kaggle datasets version -p {dataset_dir} -m 'Auto-update from IDE'"
    
    result = subprocess.run(cmd_create, shell=True, capture_output=True, text=True)
    if "already exists" in result.stderr or result.returncode != 0:
        print(f"   -> El dataset ya existe, actualizando versión...")
        subprocess.run(cmd_update, shell=True)
    else:
        print("   -> Dataset creado exitosamente.")

def create_training_kernel():
    print("📝 Generando Kernel de entrenamiento...")
    
    # Este es el código que correrá en Kaggle
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# 🚀 Fake News Detection - Auto Training\n", "Generado automáticamente desde VS Code."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Configurar entorno\n",
                    "import os\n",
                    "import sys\n",
                    "import shutil\n",
                    "\n",
                    "KAGGLE_INPUT_PATH = '/kaggle/input/fake-news-src'\n",
                    "print(f'Contenido de {KAGGLE_INPUT_PATH}:', os.listdir(KAGGLE_INPUT_PATH))\n",
                    "\n",
                    "# Estrategia de carga robusta: Kaggle a veces descomprime automático, a veces no.\n",
                    "if os.path.exists(os.path.join(KAGGLE_INPUT_PATH, 'src')):\n",
                    "    print('📂 Detectado código ya descomprimido. Copiando a /kaggle/working...')\n",
                    "    shutil.copytree(KAGGLE_INPUT_PATH, '.', dirs_exist_ok=True)\n",
                    "elif os.path.exists(os.path.join(KAGGLE_INPUT_PATH, 'fake_news_src.zip')):\n",
                    "    try:\n",
                    "        print('📦 Detectado ZIP. Intentando descomprimir...')\n",
                    "        shutil.unpack_archive(os.path.join(KAGGLE_INPUT_PATH, 'fake_news_src.zip'), '.')\n",
                    "    except Exception as e:\n",
                    "        print(f'⚠️ Error descomprimiendo ({e}). Intentando copiar como directorio...')\n",
                    "        shutil.copytree(KAGGLE_INPUT_PATH, '.', dirs_exist_ok=True)\n",
                    "else:\n",
                    "    print('⚠️ Estructura desconocida, copiando todo recursivamente...')\n",
                    "    shutil.copytree(KAGGLE_INPUT_PATH, '.', dirs_exist_ok=True)\n",
                    "\n",
                    "# Instalar dependencias\n",
                    "if os.path.exists('requirements.txt'):\n",
                    "    print('⬇️ Instalando dependencias...')\n",
                    "    !pip install -r requirements.txt -q\n",
                    "else:\n",
                    "    print('❌ Error CRÍTICO: No se encontró requirements.txt en el directorio de trabajo.')\n",
                    "    print('Archivos actuales:', os.listdir('.'))\n",
                    "\n",
                    "print('✅ Entorno listo.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 2. Entrenar Modelo Baseline (DistilBERT)\n", "Entrenamiento estándar sin pesos de clase."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!PYTHONPATH=. python src/training/train_baseline.py"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 3. Entrenar Modelo Weighted (DistilBERT + Pesos)\n", "Manejo de desbalance de clases usando pesos en la Loss."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!PYTHONPATH=. python src/training/train_weighted.py"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 4. Entrenar Modelo Avanzado (DeBERTa + LoRA)\n", "Modelo estado del arte con adaptación eficiente."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Usamos PYTHONPATH=. para asegurar que se encuentre el paquete 'src'\n",
                    "!PYTHONPATH=. python src/training/train_deberta_lora.py"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    kernel_slug = "fake-news-auto-train"
    kernel_dir = BUILD_DIR / "kernel"
    kernel_dir.mkdir(exist_ok=True)
    
    with open(kernel_dir / "train.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook_content, f)
        
    # Metadata del Kernel
    kernel_meta = {
        "id": f"{KAGGLE_USER}/{kernel_slug}",
        "title": "Fake News Auto Train",
        "code_file": "train.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [
            f"{KAGGLE_USER}/fake-news-src",
            f"{KAGGLE_USER}/fake-news-data" # Asumimos que los datos también están subidos
        ],
        "competition_sources": [],
        "kernel_sources": []
    }
    
    with open(kernel_dir / "kernel-metadata.json", "w") as f:
        json.dump(kernel_meta, f)
        
    print(f"🚀 Lanzando Kernel {kernel_slug}...")
    subprocess.run(f"kaggle kernels push -p {kernel_dir}", shell=True)
    print(f"\n✅ Entrenamiento iniciado! Monitorea aquí: https://www.kaggle.com/{KAGGLE_USER}/{kernel_slug}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="store_true", help="Subir también la carpeta data/ (tarda más)")
    args = parser.parse_args()

    check_kaggle_auth()
    prepare_build_dir()
    
    # 1. Subir Código
    zip_source_code()
    upload_dataset(SRC_ZIP, "Fake News Source Code", "fake-news-src")
    
    # 2. Subir Datos (Opcional)
    if args.data:
        print("📦 Empaquetando datos (esto puede tardar)...")
        shutil.make_archive(str(BUILD_DIR / "fake_news_data"), 'zip', PROJECT_ROOT / "data")
        upload_dataset(DATA_ZIP, "Fake News Data", "fake-news-data")
    
    # 3. Lanzar Kernel
    create_training_kernel()

if __name__ == "__main__":
    main()
