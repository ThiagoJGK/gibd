---
aliases: [Análisis de Sentimientos]
tags:
  - grupo/g3_texto
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2026-02-11
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/2025 - Analisis de Sentimientos/CinesArgentinos/Análisis de Sentimientos.docx"
tamanio_bytes: 74034
---

# Análisis de Sentimientos

Ruta interna: `GIBD/2025 - Analisis de Sentimientos/CinesArgentinos/Análisis de Sentimientos.docx`

---

Análisis de Sentimientos
Consigna
Vamos a hacer experimentos de "análisis de sentimientos", lo que significa que tomamos un texto escrito en el lenguaje natural y lo tenemos que clasificar como positivo, negativo o neutro.
La primera base de datos sobre la cual lo tenemos que probar es una que utilizamos en el 2021, extraída de comentarios de películas del sitio Cines Argentinos. La idea es utilizar BGE-M3 en lugar de BERT y luego comparar los resultados. 
Se deberá leer el artículo escrito en aquel momento y tratar de hacer lo mismo utilizando BGE-M3 en lugar de BERT. Primero sin ajuste fino y luego, reentrenando el modelo sobre una parte de la BD para hacer el ajuste fino. 
La BD hay que dividirla en tres partes: training (60%), Validación (10%), y el 30% restante para hacer los experimentos. Su estructura es la siguiente: 
idPelicula, comentario, puntuación (dada por el que escribió el comentario, de 0 a 5). 
Podemos tomar 0-2 como comentario negativo, 3 como neutro y 4-5 como positivo.
Resumen Artículo 
Introducción
El Análisis de Sentimientos o Minería de Opiniones estudia la interpretación automática de opiniones y sentimientos expresados mediante el lenguaje natural.
Este trabajo presenta la aplicación y comparación de distintas técnicas de aprendizaje automático, con el enfoque clásico de bolsa de palabras, contra técnicas más actuales como la utilización de embeddings con redes neuronales recurrentes y Transformers, también conocidos como Modelos de Lenguaje. El caso de estudio se realiza sobre los comentarios y valoraciones de usuarios acerca de películas extraídas del sitio www.cinesargentinos.com.ar. 
Marco Teórico
El Análisis de Sentimientos es el estudio computacional de opiniones, sentimientos y emociones expresadas a través de un texto. Utilizamos el término objeto para denotar la entidad destino que se ha comentado. A partir de esto, se pueden definir los siguientes conceptos:
Objeto: un objeto o es una entidad que está asociado a un par, o: (T, A), donde T es una jerarquía de componentes y A es un conjunto de atributos de o. Cada componente tiene su propio conjunto de componentes y atributos.
Opinión: una opinión sobre una característica f es una actitud, emoción o valoración positiva o negativa sobre esta.
Orientación de una opinión: la orientación de una opinión sobre una característica f indica si la opinión es positiva o negativa.
Nuestro problema es establecer si un documento expresa una opinión positiva o negativa de un objeto, aplicando diferentes técnicas de evaluación de opiniones sobre una misma base de datos, para analizar sus desempeños en forma comparativa.
Los métodos seleccionados para nuestro estudio son de aprendizaje supervisado, lo que significa que se requiere conocer la clase a la que pertenece la observación al momento de su entrenamiento. Los métodos son Naive Bayes, Random Forest, Regresión Logística y SVM con la representación clásica de bolsa de palabras; Redes Neuronales Recurrentes con el embedding Word2Vec y por último, para la arquitectura de Transformers se utilizó el modelo de lenguaje BETO, una versión en español del modelo original BERT.
BERT
A finales de 2017 Google presenta una nueva arquitectura denominada Transformer en la cual propone quitar las capas recurrentes y convolucionales de las redes utilizadas hasta el momento, a cambio de mecanismos o capas de atención. Estas capas de atención codifican las palabras en función de las demás palabras de la frase, permitiendo introducir información del contexto junto con la representación de cada palabra.
BERT (Bidirectional Encoder Representations from Transformers) es un Modelo de Lenguaje diseñado para entrenar representaciones bidireccionales profundas a partir de textos sin etiquetar, tomando en cuenta tanto el contexto izquierdo como derecho en todas las capas. Ha sido pre-entrenado mediante
aprendizaje no supervisado a partir de corpus de gran tamaño en idioma inglés. A diferencia de los modelos secuenciales o recurrentes tradicionales, la arquitectura de atención procesa toda la secuencia de entrada a la vez, permitiendo que todos los tokens de entrada se procesen en paralelo.
Para superar su limitación inicial de funcionamiento sólo para el inglés, han surgido versiones que soportan distintos lenguajes, o inclusive múltiples lenguajes en uno. Para el lenguaje español en particular, uno de los modelos más conocidos se llama BETO y tiene las mismas características
antes mencionadas de BERT, pero con la diferencia que el pre-entrenamiento se realizó con textos en español.
Experimentos Realizados
Conjunto de Datos
Este estudio fue realizado sobre una base de datos de comentarios extraídos del sitio web www.cinesagentinos.com.ar; los comentarios son reseñas de distintas películas que los usuarios aportan sin ninguna estructura definida, donde además se pondera la película con un puntaje de una a cinco estrellas. El lote de datos final fue de 52.309 comentarios de los cuales 36.661 fueron etiquetados como positivos (aproximadamente el 70%).
Métricas

Accuracy = (TP+TN) / (TP+FP+TN+FN)
Precission = TP / (TP+FP)
Recall = TP/(TP+FN)
F1_score = 2 * (Precision* Recall) / (Precision+Recall)
donde: TP=True Positive, TN=True Negative, FP=False Positive, FN=False Negative.
Descripción de los Experimentos
Con el fin de obtener el mejor modelo para cada uno de los algoritmos se realizó una búsqueda de hiperparámetros por medio del método Grid Search, entrenando modelos con distintos valores de los parámetros propios de cada algoritmo, quitando o dejando las “stop words” y con distintos tamaños del corpus de entrenamiento. Los hiperparámetros de ajuste para BETO fueron learning rate y batch size (número de muestras en cada iteración).
Resultados

El F1-score obtenido en este experimento fue de 88%. La tasa de learning rate óptima fue de 0,03, el batch size de 64, y la cantidad de palabras seleccionadas por comentario fue 150 (se toman las primeras 150 palabras del comentario y en caso de tener menos palabras, BERT completa con un carácter de relleno). Tampoco se quitaron las Stop Words.
Análisis de Resultados
Las dos técnicas más recientes obtuvieron los mejores resultados, aunque entre ellas no hay diferencias significativas en este caso. En cuanto al preproceso de los datos se observó que la eliminación de Stop Words no generó mejores resultados si no que, por el contrario, disminuyó su rendimiento.
Existen distintos trabajos de clasificación de comentarios de películas escritos en inglés, en donde utilizando BERT se obtuvieron como resultado entre un 85% y un 94% de Accuracy, mientras que en nuestro caso de estudio el valor alcanzado fue 83%, es decir, entre un 2% y un 11% menos. Esta diferencia puede tener varias causas: diferencias propias del lenguaje, pre-entrenamiento con un corpus de menor tamaño, diferencias en el nivel de informalidad del lenguaje coloquial utilizado, o incluso mejor ajuste de algunos hiperparámetros.
Comentarios Mal Clasificados
Para comparar los comentarios mal clasificados tomamos en cuenta solo los 2 mejores modelos obtenidos (LSTM+Word2vec y BETO). Del total de 10.462 comentarios del conjunto de testing, 1.992 fueron mal clasificados utilizando el primer algoritmo, mientras que con BETO fueron 1.815. Teniendo en cuenta que se utilizó el mismo conjunto de testing para los experimentos, se observó que 994 comentarios fueron mal clasificados por ambos algoritmos a la vez.
Analizando estos comentarios, encontramos al menos cinco posibles causas por las cuales el comentario no obtuvo la clasificación correcta:
Casos en los que, a pesar de que el comentario tiene una connotación positiva, la etiqueta original del mismo es negativa. Es decir, el autor del comentario escribió una opinión positiva de la película, pero la calificó negativamente.
Casos de comentarios calificados positivamente por el usuario, pero acompañado de un comentario con mensaje negativo.
Comentarios ambiguos, es decir, con cierto balance entre lo positivo y negativo. Por ejemplo, “superó mis expectativas, las escenas de susto un poco predecibles”.
Frases con sentido figurado, que probablemente no son aprendidas correctamente por el modelo: “se pasó en un suspiro”.
Negación y a veces doble o triple negación en la misma frase: es probable que los modelos tengan problemas cuando se invierte el sentido de una frase a través de la negación: “no es una película de la que te arrepientas de haber visto”.

Los primeros dos casos no están asociados a los modelos sino a los datos, y sólo son problemáticos cuando el entrenamiento se realiza sobre un corpus que posee una cantidad significativa de ellos.
Respecto a los comentarios ambiguos, una solución parcial que se presenta en distintos trabajos, es definir una tercera clase “neutral” para los casos en los cuales no está claro si el comentario es positivo o negativo. 
Las últimas dos causas son conocidas limitaciones de la mayoría de los modelos, ya que hasta el momento ningún modelo comprende realmente el significado del texto, sino que se basan en las relaciones de co-ocurrencia que encuentran entre las palabras.
2025
El objetivo es replicar el experimento de análisis de sentimientos que se hizo en 2021, pero esta vez utilizando un modelo de lenguaje más reciente (BGE-M3). La tarea es clasificar comentarios de películas como positivos, negativos o neutros basándose en la puntuación del 0 al 5 que les dieron los usuarios.
Teniendo esto, se comparará el desempeño de BGE-M3 con los resultados obtenidos por BETO en el estudio original. Para ello, se realizarán dos experimentos con BGE-M3: uno sin ajuste fino (fine-tuning) y otro con ajuste fino, lo que implica reentrenar el modelo con una parte del conjunto de datos.
Pasos a Seguir
Preparación de los datos
Lo primero es procesar la base de datos para clasificar a cada comentario según lo definido:
Negativo: puntuación de 0 a 2.
Neutro: puntuación de 3.
Positivo: puntuación de 4 a 5.
Una vez clasificados los datos, se deberá dividir la base de datos en tres conjuntos:
Training (60%): para entrenar el modelo en el experimento de ajuste fino.
Validación (10%): para evaluar el modelo durante el entrenamiento y afinar los hiperparámetros.
Testing (30%): para evaluar el rendimiento final de los modelos. Este conjunto de datos debe ser el mismo para ambos, tal como se hizo en el estudio original, para que la comparación sea justa.
BGE-M3 Sin Ajuste Fino
Se utilizará el modelo BGE-M3 tal como está, es decir, sin reentrenarlo con los datos de Training. Esto es posible debido a que el modelo está pre-entrenado para generar embeddings que capturan el significado semántico del texto.
Los pasos a seguir en este caso son:
Generar embeddings: se usa el modelo BGE-M3 para convertir cada comentario del conjunto de Training y Testing en un vector numérico (embedding).
Clasificación: una vez obtenidos los embeddings, se entrena un clasificador tradicional usando usando los embeddings del conjunto de entrenamiento. 
Evaluación: se evalúa el rendimiento del clasificador usando los embeddings del conjunto de testing. 
BGE-M3 Con Ajuste Fino
El ajuste fino (fine-tuning) es el proceso de tomar un modelo de lenguaje pre-entrenado (como BGE-M3) y re-entrenarlo en una tarea específica y con un conjunto de datos particular. 
Entonces, los pasos que a seguir en este caso son:
Ajuste fino del modelo: se usará el conjunto de datos de Training para reentrenar el modelo BGE-M3 para la clasificación de sentimientos. En este paso, el modelo aprende a ajustar sus pesos internos para mejorar su capacidad de clasificar los comentarios.
Ajuste de hiperparámetros: se usará el conjunto de Validación para optimizar los hiperparámetros del modelo, como el learning rate y el batch size, tal como se hizo con BETO en el artículo original.
Evaluación: con el modelo ajustado, se lo evaluará en el mismo conjunto de Testing usado en el primer experimento para calcular las métricas de rendimiento y comparar los resultados con los obtenidos por el modelo sin ajuste fino y por BETO.
Comparación de Resultados
El último paso es analizar los resultados obtenidos por los distintos experimentos y compararlos.










