---
aliases: [Analisis de preguntas no encontradas]
tags:
  - grupo/g3_texto
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2024-09-13
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/BERT - GPT - Llama/Artículo CONAISI 2024/Preguntas y Respuestas/Analisis de preguntas no encontradas.docx"
tamanio_bytes: 1634538
---

# Analisis de preguntas no encontradas

Ruta interna: `GIBD/BERT - GPT - Llama/Artículo CONAISI 2024/Preguntas y Respuestas/Analisis de preguntas no encontradas.docx`

---


Preguntas no encontradas por consulta original:

Modelo BGE


Modelo BETO





Modelo NOMIC





Respuestas sin encontrar cada 2 modelos 


Respuestas sin encontrar en los 3 modelos 



Analisis individual de cada consulta no encontrada

CONSULTA 18:
Cuales son los causales de recusasión para algún aspirante?


TEXTO OBJETIVO:(Articulo 28)
Documento:8
Articulo: 28
Capitulo: V - DESIGNACIÓN DEL JURADO

Dentro de los cinco (5) días de designados los Jurados, se deberá notificar fehacientemente a sus integrantes y a los aspirantes. Posteriormente a dicha notificación se dará a publicidad, la nómina de los miembros que componen el Jurado en la cartelera mural de la Unidad Académica correspondiente y en su página electrónica, durante ocho (8) días.


Primer resultado de los 10 primeros articulos (BGE)
Documento:8
Articulo: 30
Capitulo: VI - RECUSACIÓN DE LOS MIEMBROS DEL JURADO

Causales de recusación: Serán causales de recusación: a) El parentesco por consanguinidad dentro del cuarto grado y segundo de afinidad entre Jurado y algún aspirante. b) Tener el Jurado o sus consanguíneos y/o afines, dentro de los grados establecidos en el inciso anterior sociedad o comunidad con alguno de los aspirantes, salvo que la Sociedad fuese anónima. c) Tener algún Jurado miembro pleito pendiente con el aspirante. d) Ser el Jurado o aspirante, recíprocamente, acreedor, deudor o fiador. e) Ser o haber sido el Jurado autor de denuncia o querella contra el aspirante, o denunciado o querellado por este ante los Tribunales de Justicia o Tribunal Académico con anterioridad a la designación del Jurado. f) Haber emitido el Jurado opinión, dictamen o recomendación prejuzgando acerca del resultado del concurso que se tramita. g) Tener el Jurado amistad íntima con alguno de los aspirantes o enemistad o resentimiento que se manifiesten por hechos conocidos en el momento de su designación. RESOLUCIÓN Nº289/08. h) Haber recibido el Jurado importantes beneficios del aspirante. i) Carecer el Jurado de versación reconocida en el área del conocimiento científico o técnico motivo del Concurso. j) Las mencionadas en el artículo 18°.


CONSULTA 99:
Qué tiene como objetivos la educación de posgrado?

TEXTO OBJETIVO:(Articulo 10152)
Documento:10008
Articulo: 10152
Capitulo: TÍTULO III: ESTRUCTURA DEL SISTEMA EDUCATIVO NACIONAL. CAPÍTULO V: EDUCACIÓN SUPERIOR

La organización y autorización de universidades alternativas, experimentales, de postrado, abiertas, a distancia, institutos universitarios tecnológicos, pedagógicos y otros creados libremente por iniciativa comunitaria, se regirá por una ley específica.


Primer resultado de los 10 primeros articulos (BGE)

Documento:10008
Articulo: 10154
Capitulo: TÍTULO III: ESTRUCTURA DEL SISTEMA EDUCATIVO NACIONAL. CAPÍTULO VI: EDUCACIÓN CUATERNARIA

El objetivo de la Educación de Posgrado es profundizar y actualizar la formación cultural, docente, científica, artística y tecnológica mediante la investigación, la reflexión crítica sobre la disciplina y el intercambio sobre los avances en las especialidades. (Expresión "cuaternaria" sustituida por expresión "de posgrado" por art. 86, inc. a) de la Ley Nº 24.521 B.O. 10/08/1995)


Para ambos casos el problema de que no se hayan encontrado fue un mal etiquetado en el número de articulo donde tendría que encontrarse la respuesta. Cuando se utiliza el modelo BGE, aparece el articulo correcto en la primer posición. 


Preguntas no encontradas por otros motivos (modelo bge)


En la consulta 198 se observa que la pregunta contiene más información que el cuerpo del artículo objetivo. Esto podría explicar el porqué el texto objetivo no se está encontrando dentro de los primeros diez resultados. Es posible que la pregunta esté aportando demasiado ruido en su información, aunque contenga palabras que se corresponden con el artículo que se debe encontrar.

CONSULTA 198:
Como es la emision de cualquier tipo de voto dentro de la Universidad?

TEXTO OBJETIVO:(Articulo 189)
Documento:11
Articulo: 189
Capitulo: DE LOS DIRECTORES DE CARRERA

La emisión del voto será secreta, obligatoria y directa.

Cuando se mira los resultados de
bge-m3_gpt4-similares_3_preguntas-individuales_combinadas-promedio



Observamos que de todas maneras al realizar un promedio de preguntas similares, sigue sin aparecer dentro de los primeros diez resultados 


Sin embargo para el modelo “Nomic”lo encuentra en la posición 1 y el modelo “Beto” lo encuentra en la posición 2 lo que descarta que sea un problema propio de la pregunta.