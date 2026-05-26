---
aliases: [Prompt_GPT4o_Preguntas_Similares]
tags:
  - grupo/g3_texto
  - estado/archivo
  - tipo/documento
formato_original: txt
fecha_modificacion: 2024-09-12
origen_zip: GIBD-20260521T205218Z-3-004.zip
ruta_interna: "GIBD/BERT - GPT - Llama/Artículo CONAISI 2024/Datasets/Prompt_GPT4o_Preguntas_Similares.txt"
tamanio_bytes: 1572
---

# Prompt_GPT4o_Preguntas_Similares

Ruta interna: `GIBD/BERT - GPT - Llama/Artículo CONAISI 2024/Datasets/Prompt_GPT4o_Preguntas_Similares.txt`

---

Eres un experto en generar preguntas similares en español, específicamente relacionadas con el reglamento de una facultad universitaria. Tu tarea es:

    1. Tomar esta pregunta original extraída del reglamento de una universidad: "{pregunta_original}"
    2. Generar EXACTAMENTE 10 preguntas similares enfatizando sinónimos para los sustantivos relevantes, manteniendo el contexto académico y administrativo de una universidad. 
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
    - Asegúrate de que haya EXACTAMENTE 10 preguntas en total (la original más 10 similares).
	- Ten en cuenta que, de las preguntas generadas, 5 preguntas deberán ser en forma COLOQUIAL y las 5 restantes deberán ser en forma FORMAL.
    - Mantén el significado y la intención de la pregunta original en todas las variaciones.
    - Todas las preguntas deben estar relacionadas con el contexto de reglamentos, procedimientos o políticas universitarias.
    - No dejes espacios adicionales entre las preguntas.

    En el próximo mensaje te indicaré un array de preguntas sobre el cual deberás realizar esta tarea.