# Objetivo

Desarrollar un sistema de detección automática de fake news utilizando modelos de Natural Language Processing (NLP) basados en Transformers, capaz de analizar texto plano o noticias obtenidas desde URLs y clasificar su veracidad.

## Idioma
- Idioma inicial del sistema: Español
- (El modelo puede extenderse a otros idiomas en futuras versiones)


## Entradas del sistema
El sistema acepta dos tipos de entrada:

- Texto de una noticia:
* Artículo completo o fragmento relevante en texto plano.

- URL de una noticia:
* Enlace a un artículo periodístico publicado en la web.


## Salidas del sistema
Para cada entrada, el sistema devuelve:
- Clasificación binaria:
* "Fake" si la noticia es falsa.
* "True" si la noticia es verdadera.

- Explicación básica del resultado:
* Palabras, frases o patrones que influyeron en la predicción


## Aclaración clave sobre URLs
Cuando se ingresa una URL, el sistema extrae automáticamente el contenido del artículo antes de analizarlo.

- el modelo no se entrena con links, solo con texto
- La extracción del contenido se realiza únicamente en tiempo de inferencia.
- El texto extraído es el que se utiliza para:
    * Clasificación
    * Explicabilidad
38→    * Comparacion con otras noticias (RAG)
39→
40→### Módulo RAG (Fact-Checking Asistido)
41→El sistema incluye un sub-pipeline de Recuperación Aumentada (RAG) para contrastar noticias:
42→- Busca noticias similares en un corpus confiable (onlytrue1000.csv).
43→- Usa embeddings semánticos (Sentence Transformers) y FAISS.
44→- **Importante**: El "score" devuelto por FAISS representa **similitud semántica (coseno)**, no veracidad. Un score alto indica que hablan del mismo tema, no que la noticia sea cierta.
45→- Compara narrativas usando un LLM (Flan-T5) o un fallback heurístico si el modelo no está disponible.
46→
47→### Extracción de artículos desde URLs

El sistema soporta análisis directo desde enlaces de noticias.
Se implementa extracción robusta usando `newspaper3k` con manejo de errores:

- URL inválida
- Error de conexión
- Artículos cortos
- Paywalls
- Errores de parseo

La extracción se ejecuta únicamente en inferencia.


## Alcance del sistema
- Clasificación binaria (Fake / Real)
- Uso de modelos Transformer modernos
- Explicabilidad del resultado
- Soporte para URLs de noticias
- API y despliegue como producto


## Enfoque del proyecto
Este proyecto combina:
- Machine Learning aplicado

- Ingeniería de datos real

- Procesamiento de lenguaje natural

- Scraping responsable

- Arquitectura de producto
 
 ## RAG – Asistencia de verificación
 El sistema agrega un sub-pipeline paralelo de fact-checking asistido:
 - Corpus confiable: noticias reales en `data/raw/onlytrue1000.csv` (texto plano en español, sin duplicados).
 - Embeddings: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (dimensión estable, multilingüe).
 - Vector DB: FAISS local con persistencia en `models/rag_index/`.
 - Recuperación: top‑k vecinos por similitud semántica.
 - LLM comparador: `google/flan-t5-base` (con fallback heurístico si no está disponible).
 - Las fuentes recuperadas provienen de un corpus previamente curado de noticias reales.
 
 Flujo:
 1. Construir índice desde el corpus real (no es entrenamiento, es indexado).
 2. Para una noticia nueva (texto o URL ya extraído), generar embedding y consultar el índice.
 3. Preparar el contexto y comparar narrativas con LLM.
 4. Devolver salida explicativa (resumen comparativo, contradicciones, nivel de alineación).
 
 Limitaciones:
 - No garantiza veracidad absoluta ni reemplaza el criterio humano.
 - Puede fallar con paywalls o textos muy cortos.
 - Depende de la calidad y cobertura del corpus confiable.
 - Umbral conceptual: resultados con `score < 0.4` se consideran débilmente relacionados.
 - Alcance: el RAG no decide Fake/Real; aporta contexto comparativo y explicativo.

### API – Endpoint /predict

El sistema expone un único endpoint que acepta texto o URLs.
Cuando se ingresa una URL, el contenido se extrae automáticamente
antes del análisis.

La salida incluye:
- Clasificación fake / real
- Confianza del modelo
- Explicación basada en recuperación de fuentes reales (RAG)
