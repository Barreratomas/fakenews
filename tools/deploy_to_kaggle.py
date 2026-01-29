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
WORKSPACE_ZIP = BUILD_DIR / "fake_news_workspace.zip"
DATASET_SLUG = "fake-news-workspace"

# Carpetas a IGNORAR al empaquetar
IGNORE_DIRS = [
    "__pycache__", 
    ".git", 
    ".venv", 
    "venv", 
    "build_kaggle", 
    "models", # No subimos modelos entrenados para ahorrar espacio
    ".idea", 
    ".vscode",
    "wandb",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache"
]

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

def zip_workspace():
    print(f"📦 Empaquetando TODO el Workspace en un único Dataset...")
    print(f"   -> Root: {PROJECT_ROOT}")
    
    with ZipFile(WORKSPACE_ZIP, 'w') as zipf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Filtrar directorios ignorados in-place
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                file_path = Path(root) / file
                # Calcular ruta relativa al root del proyecto
                arcname = file_path.relative_to(PROJECT_ROOT)
                
                # Ignorar archivos específicos si es necesario
                if file == "kaggle.json": continue 

                zipf.write(file_path, arcname)
                
    print(f"   -> {WORKSPACE_ZIP.name} creado ({WORKSPACE_ZIP.stat().st_size / 1024 / 1024:.2f} MB)")

def upload_workspace_dataset():
    """Sube el workspace completo como un dataset."""
    dataset_dir = BUILD_DIR / DATASET_SLUG
    dataset_dir.mkdir(exist_ok=True)
    shutil.copy(WORKSPACE_ZIP, dataset_dir)
    
    # Crear metadata
    meta = {
        "title": "Fake News Workspace",
        "id": f"{KAGGLE_USER}/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(dataset_dir / "dataset-metadata.json", "w") as f:
        json.dump(meta, f)
    
    print(f"🚀 Subiendo dataset {DATASET_SLUG}...")
    
    cmd_create = f"kaggle datasets create -p {dataset_dir} -u"
    cmd_update = f"kaggle datasets version -p {dataset_dir} -m 'Auto-update from IDE' --dir-mode zip"
    
    # Intentar crear primero
    result = subprocess.run(cmd_create, shell=True, capture_output=True, text=True)
    
    if "already exists" in result.stderr or result.returncode != 0:
        print(f"   -> El dataset ya existe, actualizando versión...")
        # Usamos subprocess.run normal para ver el output de progreso de Kaggle
        subprocess.run(cmd_update, shell=True)
    else:
        print("   -> Dataset creado exitosamente.")

def create_training_kernel():
    print("📝 Generando Kernel de entrenamiento...")
    
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# 🚀 Fake News Detection - Auto Training (Unified Workspace)\n", 
                           "Entorno replicado exactamente desde el IDE local."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Restaurar Workspace\n",
                    "import os\n",
                    "import shutil\n",
                    "import sys\n",
                    "\n",
                    "# Ruta donde Kaggle monta el dataset\n",
                    f"INPUT_DIR = '/kaggle/input/{DATASET_SLUG}'\n",
                    "WORKING_DIR = '/kaggle/working'\n",
                    "\n",
                    "print('📂 Contenido del input:', os.listdir(INPUT_DIR))\n",
                    "\n",
                    "# Copiar todo al directorio de trabajo (para tener permisos de escritura)\n",
                    "print('� Restaurando estructura del proyecto en /kaggle/working...')\n",
                    "\n",
                    "# Kaggle a veces descomprime el zip, a veces no. Manejamos ambos casos.\n",
                    "zip_path = os.path.join(INPUT_DIR, 'fake_news_workspace.zip')\n",
                    "if os.path.exists(zip_path):\n",
                    "    print('   -> Descomprimiendo ZIP maestro...')\n",
                    "    shutil.unpack_archive(zip_path, WORKING_DIR)\n",
                    "else:\n",
                    "    print('   -> Copiando archivos planos...')\n",
                    "    # Si Kaggle ya lo descomprimió, copiamos recursivamente\n",
                    "    shutil.copytree(INPUT_DIR, WORKING_DIR, dirs_exist_ok=True)\n",
                    "\n",
                    "print('✅ Estructura restaurada:')\n",
                    "for root, dirs, files in os.walk(WORKING_DIR):\n",
                    "    level = root.replace(WORKING_DIR, '').count(os.sep)\n",
                    "    indent = ' ' * 4 * (level)\n",
                    "    print(f'{indent}{os.path.basename(root)}/')\n",
                    "    subindent = ' ' * 4 * (level + 1)\n",
                    "    # Mostrar solo algunos archivos para no saturar el log\n",
                    "    for f in files[:5]:\n",
                    "        print(f'{subindent}{f}')\n",
                    "    if len(files) > 5:\n",
                    "        print(f'{subindent}... ({len(files)-5} más)')\n",
                    "\n",
                    "# 2. Instalar Dependencias\n",
                    "if os.path.exists('requirements.txt'):\n",
                    "    print('⬇️ Instalando dependencias...')\n",
                    "    !pip install -r requirements.txt -q\n",
                    "else:\n",
                    "    print('❌ ALERTA: No se encontró requirements.txt')\n",
                    "\n",
                    "# 3. Verificar Datos Procesados\n",
                    "if os.path.exists('data/processed/train'):\n",
                    "    print('✅ Datos procesados detectados. Saltando build_hf_datasets.py')\n",
                    "else:\n",
                    "    print('⚠️ No hay datos procesados. Ejecutando preprocesamiento...')\n",
                    "    !PYTHONPATH=. python src/preprocessing/build_hf_datasets.py\n",
                    "\n",
                    "print('✅ Entorno listo para entrenar.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 2. Ejecutar Entrenamientos"]
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
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!PYTHONPATH=. python src/training/train_weighted.py"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
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
    
    kernel_slug = "fake-news-auto-train-unified"
    kernel_dir = BUILD_DIR / "kernel"
    kernel_dir.mkdir(exist_ok=True)
    
    with open(kernel_dir / "train.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook_content, f)
        
    # Metadata del Kernel
    kernel_meta = {
        "id": f"{KAGGLE_USER}/{kernel_slug}",
        "title": "Fake News Auto Train (Unified)",
        "code_file": "train.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [
            f"{KAGGLE_USER}/{DATASET_SLUG}"
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
    check_kaggle_auth()
    prepare_build_dir()
    
    # 1. Empaquetar TODO
    zip_workspace()
    
    # 2. Subir Dataset Único
    upload_workspace_dataset()
    
    # 3. Lanzar Kernel
    create_training_kernel()

if __name__ == "__main__":
    main()
