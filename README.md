# Sistema de Detección de Fake News

## Descripción del Proyecto

Fake News Detection System es un sistema de detección de noticias falsas basado en modelos de NLP modernos. Permite analizar tanto texto plano como URLs de artículos periodísticos, extrayendo automáticamente el contenido antes de su evaluación.

**Importante**: Cuando se ingresa una URL, el sistema extrae automáticamente el texto del artículo y lo analiza como texto plano.

## Cómo Funciona el Sistema

Arquitectura del sistema a alto nivel:

```
INPUT
 ├─ Texto ──────────────→ Clasificador Transformer → Predicción
 └─ URL → Scraper → Texto → Clasificador Transformer → Predicción
                                     ↓
                                RAG + LLM (Verificación)
```

El flujo de trabajo se compone de los siguientes pasos:

*   **Clasificación binaria (Fake / Real)**: Determinación de la veracidad del contenido basada en patrones aprendidos.
*   **Modelo base**: Uso de arquitectura Transformer Multilingüe (`mDeBERTa v3`) fine-tuned para la tarea.
*   **Eficiencia**: Implementación de LoRA (Low-Rank Adaptation) para optimización de recursos.
*   **RAG (Retrieval-Augmented Generation)**: Contraste de información con fuentes reales indexadas para validación contextual.
*   **Resolución de Conflictos**: Sistema híbrido que combina el análisis de estilo (DeBERTa) con la verificación de hechos (RAG) para emitir un veredicto final robusto.
*   **Explicabilidad**: Identificación y resaltado de palabras clave que influyeron en la predicción.

## Historia del Desarrollo

Si te interesa conocer el proceso técnico, las decisiones de arquitectura y los desafíos enfrentados durante la construcción de este sistema (desde modelos básicos hasta la solución híbrida final), consulta el documento detallado:

📄 [**Leer Historia del Desarrollo (DESARROLLO.md)**](./DESARROLLO.md)

## API Backend

El sistema expone un endpoint REST para integración. Es importante notar que la API y la interfaz de usuario comparten el mismo pipeline de inferencia unificado.

**Endpoint**: `POST /predict`

**Ejemplo de solicitud:**

```json
{
  "type": "url",
  "content": "https://news.site/article"
}
```

**Ejemplo de respuesta:**

```json
{
  "label": "FAKE",
  "confidence": 0.92,
  "extracted_title": "Título extraído de la noticia...",
  "explanation": "Palabras clave identificadas...",
  "rag_analysis": "Análisis comparativo con fuentes confiables..."
}
```

## Interfaz de Usuario (UI)

El proyecto incluye una interfaz gráfica desarrollada en Gradio que permite interactuar con el modelo de forma sencilla e intuitiva:

*   **Selector de entrada**: Permite alternar entre análisis de Texto y URL.
*   **Vista previa**: Muestra el texto extraído automáticamente desde la URL para validación del usuario.
*   **Resultado**: Visualización clara de la etiqueta (Fake/Real) y el nivel de confianza del modelo.
*   **Explicación del modelo**: Detalles sobre las palabras que más influyeron en la decisión.
*   **Análisis comparativo (RAG)**: Sección dedicada a mostrar fuentes similares y su consistencia con la noticia analizada.

## Stack Tecnológico

El proyecto utiliza un stack tecnológico moderno y modular:

**NLP & ML**
*   PyTorch
*   Hugging Face Transformers
*   mDeBERTa v3 (Multilingual) + LoRA (PEFT)

**Backend**
*   FastAPI
*   Pydantic

**RAG (Fact-Checking)**
*   FAISS
*   Sentence Transformers
*   FLAN-T5

**Scraping**
*   newspaper3k
*   BeautifulSoup
*   lxml

**UI & Deploy**
*   Gradio
*   Docker

## Limitaciones del Sistema

El sistema presenta las siguientes limitaciones conocidas:

*   **Naturaleza probabilística**: El sistema no garantiza veracidad absoluta, sino una estimación probabilística basada en patrones lingüísticos y semánticos aprendidos.
*   **Acceso a contenido**: Puede fallar en la extracción de artículos protegidos por paywalls estrictos o con estructuras HTML no estándar.
*   **Alcance del entrenamiento**: El modelo fue entrenado con texto plano; no verifica la reputación de la fuente en tiempo real ni metadatos externos.
*   **Dependencia del RAG**: La calidad del análisis comparativo depende directamente de la cobertura y calidad del índice de noticias reales utilizado.
*   **Escalabilidad**: El diseño actual prioriza la demostración técnica y la arquitectura limpia, no estando optimizado para procesamiento masivo concurrente en producción.

## Posibles Mejoras

Para futuras iteraciones del proyecto se contempla la implementación de:

*   Fine-tuning con datasets multilingües para ampliar el alcance global.
*   Verificación cruzada integrando APIs externas de fact-checking en tiempo real.
*   Módulos específicos para detección de clickbait y sensacionalismo.
*   Soporte nativo multi-idioma.
*   Despliegue optimizado con soporte GPU y procesamiento por lotes (batching).
*   Monitoreo continuo de drift del modelo para mantenimiento a largo plazo.

## Docker y Despliegue

El proyecto está contenerizado para facilitar su ejecución en cualquier entorno. El contenedor expone una UI interactiva vía Gradio.

**Construcción y ejecución:**

```bash
docker build -t fake-news-detector .
docker run -p 7860:7860 fake-news-detector
```

La interfaz estará disponible en `http://localhost:7860`.

## Notas Finales

Este proyecto fue desarrollado con fines educativos y de demostración técnica, priorizando una arquitectura limpia, buenas prácticas de desarrollo y extensibilidad del código.
