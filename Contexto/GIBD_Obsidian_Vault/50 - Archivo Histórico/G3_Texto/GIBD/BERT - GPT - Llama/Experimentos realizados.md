---
aliases: [Experimentos realizados]
tags:
  - grupo/g3_texto
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2024-09-13
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/BERT - GPT - Llama/Artículo CONAISI 2024/Preguntas y Respuestas/Experimentos realizados.docx"
tamanio_bytes: 115126
---

# Experimentos realizados

Ruta interna: `GIBD/BERT - GPT - Llama/Artículo CONAISI 2024/Preguntas y Respuestas/Experimentos realizados.docx`

---

Breve descripción de los Experimentos
Los experimentos que hicimos fueron utilizando modelos de embeddings para poder, a partir de una pregunta, obtener el artículo asociado de la reglamentación. La respuesta es el artículo de forma textual como está en la reglamentación, y  lo que hicimos fue hacer varios experimentos. 
Los experimentos se pueden clasificar por las consultas utilizadas y por los modelos de embedding utilizados.
Probamos varias combinaciones de tipos de consultas con modelos de embedding para ver cuál combinación nos otorga los mejores resultados y cuáles los peores para ver cómo mejorar los resultados..

Modelos utilizados en los experimentos                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
En los experimentos se usan 2 tipos de modelos:

a. Modelos de Embeddings: los que generan los embeddings y traen las más similares (bge-m3, nomic-embed-text-v1, sentence-transformers-beto)

b. Modelos de generación de texto: son utilizados para poder generar preguntas similares (Llama 3.1, GPT 4o)

Listado de experimentos

Experimentos con el modelo de embedding bge-m3   
La mayoría de los experimentos fueron realizados con este modelo
10 Preguntas  (original + 9 similares generadas por Llama3.1 con Ollama)1
10 Preguntas  (original + 9 similares generadas por GPT 4o Supervisado)
10 Preguntas  (original + 9 similares generadas por GPT 4o Supervisado con las 10 preguntas ejecutadas de forma individual)
5 Preguntas (original + 4 similares generadas por Llama3.1 con Ollama)
5 Preguntas (original + 4 similares generadas por GPT 4o Supervisado)
5 Preguntas (original + 4 similares  generadas por GPT 4o Supervisado realizadas individualmente)
3 Preguntas (original + 2 similares generadas por Llama3.1 con Ollama)
3 Preguntas (original + 2 similares generadas por GPT 4o Supervisado)
3 Preguntas (original + 2 similares generadas por GPT 4o Supervisado realizadas individualmente)
Solo Pregunta Original de Consultas.csv

Experimentos con el modelo de embedding  nomic-embed-text-v1
10 Preguntas  (original + 9 similares generadas por GPT 4o Supervisado con las 10 preguntas ejecutadas de forma individual)
Solo Pregunta Original de Consultas.csv
Experimentos con el modelo de embedding  sentence-embeddings-BETO                 
10 Preguntas  (original + 9 similares generadas por GPT 4o Supervisado con las 10 preguntas ejecutadas de forma individual)
Solo Pregunta Original de Consultas.csv

Clasificación por consultas utilizadas:
a. Solo Consultas Originales: utilizando el archivo que compartió Andrés Pascal llamado "Consultas.csv",este archivo tiene un listado de 300 preguntas acerca de la reglamentación junto al número de artículo que la responde,  usamos las preguntas para validar la respuesta y para poder evaluar la precisión de los modelos de embeddings. Este archivo contiene el artículo que responde a la pregunta por lo que lo utilizamos para validar los resultados.

b. Con Consultas Similares (Llama 3.1): en este caso el archivo de consultas se utiliza para validar los resultados pero las preguntas similares son realizadas con modelo Llama 3.1, variando las preguntas generadas, (9, 4 y 2 preguntas similares generadas).

c. Con Consultas Similares (GPT 4o Supervisado): este caso es similar al anterior pero con un dataset armado a mano de preguntas similares elaboradas por GPT y  supervisadas por los integrantes del grupo para validar su elaboración. 

d. Consultas Similares Individuales (GPT 4o Supervisado): a diferencia del caso anterior que las preguntas similares se pasaban todas juntas, en este caso las preguntas son comparadas de forma individual y se calcula el promedio de los scores de cada respuesta para poder determinar así la precisión.

Clasificación por modelo de embedding:
Podemos clasificar los experimentos según cuál de los tres modelos de embedding fue utilizado
Experimentos con el modelo bge-m3: 
los experimentos con este modelo de embedding se realizaron con
Solo consultas originales de Consultas.csv
Con consultas similares generadas por Llama 3.1  con Ollama con 9, 4 y 2 consultas generadas.
Con consultas similares generadas por GPT4o supervisado con 9, 4 y 2 consultas generadas.
Con consultas similares individuales generadas por GPT4o supervisado con 9, 4 y 2 consultas generadas.
Experimentos con el modelo nomic-embed-text-v1: 
los experimentos con este modelo de embedding se realizaron con
Con consultas similares individuales generadas por GPT4o supervisado con 9 consultas generadas.
Solo consultas originales de Consultas.csv
Experimentos con el modelo sentence-embeddings-BETO : 
los experimentos con este modelo de embedding se realizaron con
Con consultas similares individuales generadas por GPT4o supervisado con 9 consultas generadas.
Solo consultas originales de Consultas.csv

Aclaración: Los experimentos con los modelos nomic-embed-text-v1 y BETO están limitados, por ahora, a pruebas con 9 consultas similares generadas por GPT 4o supervisado y consultas originales. Futuramente se realizarán pruebas con 4 y 2 consultas similares.

Finalidad de los experimentos
El objetivo de los experimentos es evaluar si el uso de múltiples preguntas similares incrementa la precisión de los modelos de embeddings en tareas de preguntas y respuestas (Q&A). En caso de no observar mejoras, también buscamos identificar posibles estrategias para optimizar la precisión de estos modelos.

Generación de las consultas similares
A continuación se especificará cómo se generaron las consultas similares utilizadas para los distintos experimentos:
10 Preguntas  (original + 9 similares generadas por Llama3.1 con Ollama): 
en este caso el archivo de consultas se utiliza para validar los resultados pero las preguntas similares son realizadas con modelo Llama 3.1 con Ollama, 
Se tomó una pregunta del listado original de 300 preguntas y se le pidió aLlama 3.1 con Ollama que genere 9 similares a la original con el siguiente prompt: 
“Eres un experto en generar preguntas similares en español, específicamente relacionadas con el reglamento de una facultad universitaria. Tu tarea es:

    1. Tomar esta pregunta original extraída del reglamento de una universidad: "{pregunta_original}"
    2. Generar EXACTAMENTE 9 preguntas similares enfatizando sinónimos para los sustantivos relevantes, manteniendo el contexto académico y administrativo de una universidad.
    3. Presentar el resultado en el siguiente formato estricto, sin espacios adicionales entre líneas:

    {pregunta_original}
    [Primera pregunta similar]
    [Segunda pregunta similar]
    [Tercera pregunta similar]
    [Cuarta pregunta similar]
    [Quinta pregunta similar]
    [Sexta pregunta similar]
    [Séptima pregunta similar]
    [Octava pregunta similar]
    [Novena pregunta similar]

    IMPORTANTE:
    - No incluyas NINGÚN texto adicional, comentario o aclaración.
    - Asegúrate de que haya EXACTAMENTE 10 preguntas en total (la original más 9 similares).
    - Mantén el significado y la intención de la pregunta original en todas las variaciones.
    - Todas las preguntas deben estar relacionadas con el contexto de reglamentos, procedimientos o políticas universitarias.
    - No dejes espacios adicionales entre las preguntas.
    Comienza ahora:”
10 Preguntas  (original + 9 similares generadas por GPT 4o Supervisado)
Este caso es muy similar al punto anterior, pero en este caso las 9 preguntas similares se generaron con el mismo prompt pero se crearon con el modelo GPT4 con supervisión humana para que todas las preguntas generadas sean válidas y sean exactamente 9.
Este caso es similar al anterior pero con un dataset armado a mano de preguntas similares elaboradas por gpt y supervisadas para validar su elaboración. 
10 Preguntas  (original + 9 similares generadas por GPT 4o Supervisado con las 10 preguntas ejecutadas de forma individual)
Similar a “original + 9 similares generadas por GPT 4o Supervisado” pero a diferencia de éste que las preguntas similares se pasaban todas juntas, en este caso las preguntas son comparadas de forma individual y se calcula el promedio de los scores de cada respuesta para poder determinar así la precisión.
5 Preguntas (original + 4 similares generadas por Llama3.1 con Ollama)
Este caso es muy similar  que “original + 9 similares generadas por Llama3.1 con Ollama” pero en este caso se modifica el prompt para que solo se generen 4 preguntas similares a la original.
5 Preguntas (original + 4 similares generadas por GPT 4o Supervisado)
Este caso es muy similar  que “original + 9 similares generadas por GPT 4o Supervisado” pero en este caso se modifica el prompt para que solo se generen 4 preguntas similares a la original y verificamos las consultas.
5 Preguntas (original + 4 similares  generadas por GPT 4o Supervisado realizadas individualmente)
Este caso es muy similar que “original + 9 similares generadas por GPT 4o Supervisado” pero en este caso se modifica el prompt para que solo se generen 4 preguntas similares a la original y verificamos las consultas. Además. Las preguntas son evaluadas individualmente.
3 Preguntas (original + 2 similares generadas por Llama3.1 con Ollama)
Este caso es muy similar que “original + 9 similares generadas por Llama3.1 con Ollama”, pero en este caso se modifica el prompt para que solo se generen 4 preguntas similares a la original.
3 Preguntas (original + 2 similares generadas por GPT 4o Supervisado)
Este caso es muy similar  que “original + 9  similares  generadas por GPT 4o Supervisado”  pero en este caso se modifica el prompt para que solo se generen 2 preguntas similares a la original y verificamos las consultas.
3 Preguntas (original + 2 similares generadas por GPT 4o Supervisado realizadas individualmente)
Este caso es muy similar  que “original + 9  similares  generadas por GPT 4o Supervisado realizadas individualmente” pero en este caso se modifica el prompt para que solo se generen 2 preguntas similares a la original y verificamos las consultas. Además. Las preguntas son evaluadas individualmente.





