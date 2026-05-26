---
aliases: [GIBD Texto 2025]
tags:
  - grupo/g3_texto
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2025-09-01
origen_zip: GIBD-20260521T205218Z-3-005.zip
ruta_interna: "GIBD/Texto 2025/GIBD Texto 2025.docx"
tamanio_bytes: 180617
---

# GIBD Texto 2025

Ruta interna: `GIBD/Texto 2025/GIBD Texto 2025.docx`

---

Asignación de Puntajes para la Evaluación de Consultas
Para mejorar la robustez del sistema de recuperación de información, se asignará un puntaje a cada artículo respecto a cada consulta formulada. Esto permitirá evaluar no sólo si el sistema devuelve la respuesta exacta, sino también si devuelve artículos que, aunque no sean la respuesta esperada, contienen información relacionada con la consulta.
Escala de Puntaje
Asignación de Puntajes
Selección de artículos candidatos
La selección de los artículos a evaluar para cada consulta fue realizada previamente mediante el uso de Gemini, el cual devolvió los 50 artículos más similares para cada consulta. Estos serán los únicos considerados en el proceso de evaluación, suponiendo que aquellos no incluidos no poseen ningún tipo de relación con la consulta. 
Evaluación de artículos 
Para cada consulta, se analizará manualmente el contenido de los 50 artículos mencionados anteriormente. En función del grado de relevancia que guarden con la consulta, se les asignará un puntaje según la escala previamente definida. Solo un artículo podrá recibir el puntaje 10, ya que es el que corresponde a la respuesta esperada, de acuerdo con la base de datos (Consultas). Uno o más artículos podrán recibir un puntaje de 5 si se considera que aportan información relevante o relacionada. Los artículos que no tengan relación alguna con la consulta recibirán puntaje 0 y no serán registrados.
Registro estructurado
Los resultados se almacenarán en una tabla con el formato: idConsulta, idArticulo, PuntajeAsignado.
Sólo se incluirán en esta tabla aquellos artículos que hayan recibido un puntaje distinto de 0. Por este motivo, cada consulta podrá aparecer múltiples veces en la tabla, una por cada artículo considerado relevante (con puntaje 10 o 5). 
Es importante tener en cuenta que la columna idRespuestaEsperada (nroArticulo) definida en la base de datos se refiere al número del artículo dentro del reglamento, y no al ID del mismo. Es este último el que se debe utilizar a la hora de registrar los resultados, el cual se puede ver especificado en el documento HTML.
Ejemplo
Luego de seleccionar la consulta específica (en este caso, “Consulta 1”) del documento HTML, nos encontraremos con una página donde se presenta:
La consulta en sí.
Un listado de los 50 artículos más similares para dicha consulta, ordenados de forma aleatoria.

Desde un principio sabremos qué artículo debe recibir puntaje 10. En este caso corresponde al artículo número 9, debido a que así figura en el archivo Consultas. Este documento contiene el listado completo de consultas junto con la respuesta esperada para cada una de ellas. 

Habiendo identificado el número de este artículo, se lo busca en el documento HTML para así determinar su ID correspondiente y agregarlo a la tabla de Puntaje de la forma que se puede ver en la imagen a continuación. 


Los 49 artículos restantes deberán ser evaluados manualmente, teniendo en cuenta los criterios mencionados en la escala de puntaje. 
Una vez finalizada la evaluación y habiendo agregado a la tabla el ID de los artículos correspondientes, con su respectivo puntaje, se tendrá algo como:

Cada uno tendrá asignado su propia hoja, donde se van a trabajar con los números de IdConsultas que le asignaron: 

Tener en cuenta
En la tabla de Puntaje:
Un único artículo tendrá puntaje 10 y es aquel cuyo número de artículo se incluye en la columna de idRespuestaEsperada(nroArticulo) del archivo de Consultas.
No se deben añadir aquellos artículos cuyo puntaje sea 0.




