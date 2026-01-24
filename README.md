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
    * Comparacion con otras noticias (RAG)


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
