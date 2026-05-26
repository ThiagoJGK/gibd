---
aliases: [A_Vit-Large_CLS_DB-Logos_150_80]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2025-08-26
origen_zip: GIBD-20260521T205218Z-3-002.zip
ruta_interna: "GIBD/ViT/A_Vit-Large_CLS_DB-Logos_150_80.docx"
tamanio_bytes: 706213
---

# A_Vit-Large_CLS_DB-Logos_150_80

Ruta interna: `GIBD/ViT/A_Vit-Large_CLS_DB-Logos_150_80.docx`

---

La base de datos utilizada está compuesta por 150 imágenes de logotipos, mientras que el conjunto de consultas incluyó 80 imágenes. Para la búsqueda por similitud se empleó un modelo ViT preentrenado en ImageNet, obteniéndose los siguientes resultados:
De las 80 consultas, el sistema recuperó el resultado correcto dentro de las primeras tres posiciones en 57 ocasiones, lo que equivale a una tasa de éxito del 71,25% (top 3).

En 54 casos, el resultado esperado fue identificado en la primera posición, alcanzando una tasa de acierto del 67,5% (top 1).

La métrica de accuracy en el top 5 también se mantuvo en un 71,25%, coincidiendo con la registrada para el top 3.

En síntesis, con las condiciones actuales y aplicando representaciones CLS con similitud de coseno, el modelo presenta un 67,5% de precisión en top 1 y un 71,25% de precisión en top 3 y top 5.

Respecto al rendimiento, la generación de los embeddings base de la base de datos requirió 116,3 segundos (aproximadamente dos minutos), mientras que el tiempo promedio por consulta fue de 0,97 segundos.

Por último, resulta pertinente analizar los casos en los que el modelo logró identificar la imagen correcta, pero no la devolvió en la primera posición. Este fenómeno se presentó en 3 de las 80 consultas evaluadas.






















Stuttgart


Análisis:
	Podemos destacar la diferencia de fondo, en la bdd es inexistente, mientras que en la consulta tiene un color gris. Además un cambio en la tonalidad de rojo. La consulta posee un texto inexistente en la base de datos y no posee la línea roja que lo rodea. Además el espacio amarillo es diferente. Vemos que el escudo de la salernitana posee una forma similar, sin embargo son bastante diferentes en color, formas dentro del escudo y demás

Nacional

Análisis:
	En este caso observamos que en la bdd posee como un espacio claro hacia el medio, además posee un bordeado en blanco y sin fondo. La consulta posee fondo negro. No existen similitudes entre lo encontrado y lo consultado, de ningún tipo.


Hellas Verona


Análisis:
	La diferencia más notoria la encontramos en el fondo gris que posee la consulta además el escudo cambia de color y toma un color amarillo en vez del azul de la base de datos, aún así mantiene las formas.

