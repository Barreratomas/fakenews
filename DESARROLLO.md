# Documentación de Desarrollo e Ingeniería

Este documento detalla el ciclo de vida completo del proyecto, dividido en dos fases fundamentales: la creación del "cerebro" (Ingeniería de IA) y la construcción del "cuerpo" (Ingeniería de Software).

---

# Parte 1: Ingeniería de Inteligencia Artificial
*El proceso de investigación, datos y modelado para crear el núcleo predictivo.*

## 1. Definición del Problema y Datos
El primer desafío fue definir qué constituye una "Fake News" desde la perspectiva de un modelo.
*   **Dataset**: Se utilizó el dataset "Fake and Real News Dataset" (ISOT), que contiene miles de artículos etiquetados.
*   **Limpieza**: Se implementaron scripts de preprocesamiento para eliminar ruido (URLs, caracteres especiales, firmas de autores) que pudieran sesgar al modelo hacia patrones irrelevantes ("shortcuts").
*   **Desafío Multilingüe**: Aunque el dataset base es en inglés, el objetivo era soportar español. Se optó por modelos multilingües desde el inicio.

## 2. Selección de Arquitectura (Model Selection)
Se evaluaron varias arquitecturas de Transformers:
*   **BERT/RoBERTa**: Descartados por limitaciones en el manejo de contextos largos y dependencias complejas.
*   **Elección Final: mDeBERTa v3 (Microsoft Decoding-enhanced BERT with disentangled attention)**.
    *   *Por qué*: Su mecanismo de atención desenredada permite entender mejor la semántica profunda y la intencionalidad del texto, crucial para detectar desinformación sutil. Al ser la versión `m` (multilingual), ofrece soporte nativo para español e inglés.

## 3. Entrenamiento Eficiente (Fine-Tuning con LoRA)
El entrenamiento completo de un modelo de 200M+ parámetros es costoso. Adoptamos **LoRA (Low-Rank Adaptation)**:
*   **Técnica**: En lugar de reentrenar todos los pesos, se inyectan matrices de bajo rango en las capas de atención.
*   **Resultado**: Entrenamos solo el **0.6%** de los parámetros totales.
*   **Beneficio**: El modelo resultante pesa lo mismo que el original pero con "adaptadores" ligeros (~50MB), facilitando su almacenamiento y carga.

## 4. Implementación de RAG (Retrieval-Augmented Generation)
Detectamos que el modelo "alucinaba" o fallaba con noticias muy recientes no presentes en su entrenamiento.
*   **Solución**: Un módulo RAG que actúa como verificador de hechos.
*   **Funcionamiento**:
    1.  Toma el titular/texto de la noticia sospechosa.
    2.  Realiza una búsqueda en tiempo real en internet.
    3.  Recupera fragmentos de fuentes confiables.
    4.  Compara semánticamente la noticia de entrada con la evidencia encontrada.

---

# Parte 2: Ingeniería de Software
*La construcción de la plataforma robusta que hace utilizable a la IA.*

## 1. Arquitectura del Sistema
Se diseñó una arquitectura modular desacoplada para facilitar el mantenimiento:

```
[ Frontend (UI) ] <---> [ API Gateway (FastAPI) ] <---> [ Pipeline de Inferencia ]
                                                                 |
                                                     [ Modelo NLP ] + [ Módulo RAG ]
```

## 2. Desarrollo del Backend (FastAPI)
El núcleo del sistema es una API RESTful de alto rendimiento.
*   **Validación Estricta**: Uso de **Pydantic** para garantizar que los datos de entrada (URLs, textos) cumplan con los formatos esperados antes de llegar al modelo.
*   **Gestión de Errores**: Sistema centralizado de manejo de excepciones para devolver mensajes claros al cliente (ej. "Error al descargar la URL", "Texto demasiado corto").
*   **Asincronía**: Endpoints `async` para no bloquear el servidor durante operaciones de I/O (como el scraping de noticias).

## 3. Pipeline de Inferencia Unificado
Se creó una clase orquestadora (`Pipeline`) que encapsula toda la lógica compleja:
*   Coordina la limpieza del texto.
*   Invoca al modelo DeBERTa.
*   Activa el RAG si es necesario.
*   Ejecuta la **Matriz de Resolución de Conflictos**: Un sistema de reglas que decide el veredicto final cuando el modelo y el RAG discrepan (ej. el modelo dice "Real" por el estilo, pero el RAG encuentra pruebas de que es falso).

## 4. Interfaz de Usuario (Gradio)
Para democratizar el acceso al modelo, se desarrolló una UI interactiva:
*   **Experiencia de Usuario (UX)**: Diseño limpio que oculta la complejidad técnica.
*   **Feedback Visual**: Uso de colores semánticos (Rojo/Verde/Amarillo) y barras de progreso para indicar el estado del análisis.
*   **WebSockets**: Comunicación en tiempo real para mostrar logs y estados internos del proceso mientras el usuario espera.

## 5. Calidad y Testing
*   **Unit Tests**: Batería de pruebas para asegurar que cada componente (limpiador de texto, scraper, modelo) funcione aisladamente.
*   **Integration Tests**: Pruebas del flujo completo API -> Modelo -> Respuesta.
*   **Logging**: Sistema de logs rotativos para monitorear el comportamiento del sistema en producción sin saturar el disco.
