---
aliases: [Descripción la base de datos, contenido y cantidad de ele...]
tags:
  - grupo/g3_texto
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2024-09-13
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/BERT - GPT - Llama/Artículo CONAISI 2024/Preguntas y Respuestas/Descripción la base de datos, contenido y cantidad de elementos.docx"
tamanio_bytes: 17463
---

# Descripción la base de datos, contenido y cantidad de elementos

Ruta interna: `GIBD/BERT - GPT - Llama/Artículo CONAISI 2024/Preguntas y Respuestas/Descripción la base de datos, contenido y cantidad de elementos.docx`

---

Descripción de la base de datos, contenido, cantidad de elementos
En el presente trabajo, se han utilizado dos fuentes de datos principales que constituyen la base para el posterior análisis. Estas son un archivo de texto que contiene la reglamentación de la facultad y un archivo CSV que incluye una serie de consultas con sus respectivas respuestas esperadas.
Reglamentación de la Facultad
La primera base de datos corresponde a un archivo de texto que reúne un conjunto de normativas extraídas de distintos documentos institucionales. Este archivo contiene un total de 10008 documentos y 10200 artículos, los cuales abarcan diversas áreas de la gestión universitaria. Entre los temas que se abordan se incluyen la organización de concursos para cargos docentes, las funciones de los consejos de carrera, y los derechos y obligaciones de los estudiantes, entre otros.
Cada entrada en este archivo está estructurada de la siguiente manera:
Documento
Artículo
Capítulo
Contenido del artículo
Por ejemplo, un fragmento del archivo contiene lo siguiente:
Documento: 10007
Artículo: 10121
Capítulo: "DE LOS CONSEJOS DE CARRERA. Disposiciones transitorias"
Contenido: "Los representantes del Consejo de carrera durarán 2 (dos) años en sus funciones y desempeñarán las mismas ad-honorem."
Este tipo de formato se repite a lo largo del archivo, cubriendo una amplia variedad de aspectos relacionados con la organización universitaria.
Base de Consultas y Respuestas
La segunda base de datos es un archivo CSV que contiene un total de 300 consultas. Cada consulta está relacionada con aspectos específicos de la reglamentación mencionada anteriormente y está acompañada de una respuesta esperada que remite al número de artículo correspondiente en el archivo de reglamentación.
Las columnas correspondientes al archivo CSV son las siguientes:
idConsulta: Un identificador único para cada consulta.
Consulta: La pregunta planteada, vinculada a alguna normativa específica.
idRespuestaEsperada (idArticulo): El número del artículo en la reglamentación que proporciona la respuesta esperada.
A modo de ejemplo, las primeras entradas del archivo incluyen:
