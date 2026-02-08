# Sistema de Detección de Fake News

## Descripción del Proyecto

**Fake News Detection System** es una plataforma avanzada de verificación de noticias que combina modelos de lenguaje modernos (Transformers) con técnicas de recuperación de información (RAG). El sistema permite analizar tanto texto plano como URLs de artículos, proporcionando un veredicto de veracidad fundamentado en análisis estilístico y verificación de hechos.

> **Nota**: Este proyecto está diseñado para ejecutarse en un entorno local de Python, aprovechando la potencia de librerías como PyTorch y Hugging Face Transformers.

## Características Principales

*   **Análisis Dual**: Procesa texto directo o extrae contenido automáticamente desde URLs.
*   **Modelo Híbrido**: Combina un clasificador neuronal (`mDeBERTa v3` con LoRA) para detectar patrones de escritura engañosos.
*   **Verificación de Hechos (RAG)**: Busca evidencia en tiempo real en internet para contrastar la información.
*   **Explicabilidad**: Resalta las palabras clave que influyeron en la decisión del modelo.
*   **Interfaz Gráfica**: UI intuitiva basada en Gradio para interactuar con el sistema.
*   **API REST**: Backend robusto en FastAPI para integraciones.

## Instalación y Ejecución Local

Sigue estos pasos para poner en marcha el sistema en tu máquina.

### Prerrequisitos
*   Python 3.10 o superior.
*   Git.

### 1. Clonar el Repositorio
```bash
git clone <url-del-repositorio>
cd fake_news
```

### 2. Configurar el Entorno Virtual
Es recomendable usar un entorno virtual para aislar las dependencias:

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno (Windows)
.\venv\Scripts\activate

# Activar entorno (Linux/Mac)
source venv/bin/activate
```

### 3. Instalar Dependencias
Instala todas las librerías necesarias listadas en `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la Aplicación
El proyecto incluye un script unificado que levanta tanto el Backend (API) como el Frontend (UI):

```bash
python run_app.py
```

Una vez iniciado, verás en la consola las direcciones de acceso:
*   **Interfaz de Usuario**: `http://localhost:7860`
*   **Documentación de la API**: `http://localhost:8000/docs`

## Arquitectura del Sistema

El flujo de información sigue este pipeline:

1.  **Entrada**: URL o Texto del usuario.
2.  **Extracción**: Si es URL, se descarga y limpia el contenido principal.
3.  **Inferencia (Modelo NLP)**: `mDeBERTa` analiza el estilo y semántica del texto.
4.  **Verificación (RAG)**: Se buscan noticias relacionadas en fuentes confiables y se comparan.
5.  **Resolución**: Un sistema de reglas pondera el análisis estilístico vs. la evidencia encontrada.
6.  **Salida**: Veredicto final (REAL/FAKE), confianza y explicación.

## Stack Tecnológico

**NLP & ML**
*   **Modelo**: mDeBERTa v3 (Multilingual) + LoRA (PEFT).
*   **Frameworks**: PyTorch, Hugging Face Transformers.
*   **RAG**: FAISS, Sentence Transformers.

**Ingeniería de Software**
*   **Backend**: FastAPI, Uvicorn.
*   **Frontend**: Gradio.
*   **Scraping**: newspaper3k, BeautifulSoup.
*   **Validación**: Pydantic.

## Documentación de Desarrollo

Para conocer en profundidad las decisiones técnicas, desde el entrenamiento del modelo hasta la arquitectura del software, consulta:

📄 [**Leer Historia del Desarrollo (DESARROLLO.md)**](./DESARROLLO.md)

## Licencia
Este proyecto es de código abierto y se distribuye bajo la licencia MIT.
