# Historia del Desarrollo: Sistema de Detección de Fake News

Este documento narra el proceso técnico y las decisiones de diseño tomadas durante la construcción del sistema, desde los primeros prototipos hasta la arquitectura híbrida final.

## 1. El Problema Inicial
El objetivo era crear un sistema capaz de distinguir entre noticias verdaderas y falsas. Inicialmente, el desafío parecía ser un problema clásico de **clasificación de texto**.

### Enfoque Temprano (Baseline)
- Se exploraron modelos simples (TF-IDF + Regresión Logística/SVM).
- **Limitación**: Estos modelos dependían demasiado de palabras clave específicas y no entendían el contexto semántico profundo ni la intencionalidad del texto.

## 2. La Transición a Transformers
Para capturar matices lingüísticos, migramos a arquitecturas basadas en Transformers.
- **Elección**: `microsoft/deberta-v3-base`.
- **Por qué DeBERTa**: Supera a BERT y RoBERTa gracias a su mecanismo de atención desenredada (disentangled attention), crucial para entender contextos complejos en artículos largos.

## 3. El Desafío del Idioma (Español)
El sistema debía funcionar en español. El modelo original de DeBERTa es principalmente inglés.
- **Solución**: Migración a `microsoft/mdeberta-v3-base` (Multilingual DeBERTa).
- **Impacto**: Mejora drástica en la detección de sintaxis y modismos propios de noticias en español.

## 4. Optimización de Recursos (LoRA)
El entrenamiento completo (Full Fine-Tuning) de un modelo Transformer es costoso en memoria GPU.
- **Implementación**: PEFT (Parameter-Efficient Fine-Tuning) con **LoRA (Low-Rank Adaptation)**.
- **Resultado**: Entrenamos solo el 0.6% de los parámetros totales, logrando un rendimiento comparable al entrenamiento completo pero consumiendo mucha menos memoria VRAM y permitiendo iteraciones rápidas.

## 5. El Problema de la "Alucinación" y Hechos Obsoletos
Un modelo de lenguaje solo "sabe" lo que aprendió durante su entrenamiento. Si aparece una noticia falsa sobre un evento de ayer, el modelo no tiene forma de verificarlo y puede equivocarse confiando solo en el estilo de escritura.
- **Solución**: Implementación de **RAG (Retrieval-Augmented Generation)**.
- **Cómo funciona**:
    1. El sistema busca en internet (Google Search) artículos recientes sobre el tema.
    2. Recupera el contenido relevante.
    3. Un LLM (Google FLAN-T5) compara la noticia del usuario con los hechos encontrados en internet.
    4. El sistema emite un veredicto basado en evidencia, no solo en estilo.

## 6. Arquitectura Híbrida y Resolución de Conflictos
¿Qué pasa si el modelo dice "FAKE" (por el estilo de escritura) pero el RAG dice "REAL" (porque los hechos son ciertos)?
- **Estrategia**: Implementamos una **Matriz de Resolución de Conflictos**.
- **Lógica**:
    - Si hay evidencia externa fuerte (RAG), esta tiene prioridad sobre el estilo.
    - Si el RAG no encuentra información (incertidumbre), confiamos en la predicción estilística del modelo DeBERTa.
    - Se generaron etiquetas profesionales como "DESINFORMACIÓN SOFISTICADA" (estilo real, hechos falsos) o "SENSACIONALISTA" (hechos reales, estilo engañoso).

## 7. Profesionalización y Despliegue
La etapa final se centró en la robustez y usabilidad:
- **API**: Backend rápido con FastAPI.
- **UI**: Interfaz gráfica con Gradio para demostraciones sencillas.
- **Docker**: Contenerización completa para garantizar que el sistema corra en cualquier máquina sin problemas de dependencias ("funciona en mi máquina").
- **Clean Code**: Refactorización, logs centralizados en español, y eliminación de deuda técnica.

## Conclusión Técnica
El sistema final no es solo un clasificador; es un **verificador de hechos asistido por IA**. Combina la intuición lingüística de los Transformers modernos con la capacidad de verificación de hechos en tiempo real del RAG, ofreciendo una solución mucho más robusta que los enfoques tradicionales.
