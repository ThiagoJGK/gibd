---
aliases: [Pasos Investigación y Desarrollo]
tags:
  - grupo/general
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2023-05-09
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Papers/6 - Metric Learning/Pasos Investigación y Desarrollo.docx"
tamanio_bytes: 212152
---

# Pasos Investigación y Desarrollo

Ruta interna: `GIBD/Papers/6 - Metric Learning/Pasos Investigación y Desarrollo.docx`

---

5 de Mayo

Arribamos a la conclusión de que no estábamos trabajando para hallar una función de distancia, ya que esto se llama Function Metric Learning (FML).

Por lo tanto, vamos a trabajar en dos temas:
ML, para cerrar: utilizar el código de constelation para entrenamiento. INCEPTION necesita mínimo 75x75 mínimo, en lugar de 28x28.
Luego, intentar reemplazar el modelo inception por nuestro propio modelo (FedeAndrés). 
FML: crear una función de distancia para poder minimizar la pérdida (loss del entrenamiento).

___________________________________________________________________


29 de Noviembre
Terminamos de arreglar el generador de las imágenes y empezamos a trabajar con el descifrado de la fórmula que utilizan en el algoritmo de NPairLoss para conocer la función de pérdida.


Donde:
T significa transpuesta.
+ representa el valor positivo.
x es el ancla.
x + es el elemento positivo.
f i  es el vector característico del i-ésimo elemento negativo.
f T es la transpuesta del vector característico de x.
f + es el vector característico de x +.
N es la cantidad de clases.
exp significa valor absoluto.

Independientemente de conocer la función, se debe conocer su significado, para tener una base de lo esperado al definir un parámetro metric propio.


















22 de Noviembre

Se avanzó con el código de  traducción de triplet loss a NPairLoss, ya que no se estaba pudiendo lograr la generación del arreglo de dimensión n + 1, siendo n la cantidad de clases. Este arreglo posee el ancla, el valor positivo y las n - 1 clases que son negativas, por cada elemento evaluado.









15 de Noviembre

Logramos probar el algoritmo MultiClassLoss o NPLoss con la DB de UMCM para la extracción de características. Se probó para 8 Clases y 10 épocas. La primera época se ejecutó en aproximadamente 1020 segundos y a partir de la segunda el tiempo fué de menos de un minuto y más de 50 segundos.

Andrés nos pidió que probemos con las  203 clases el algoritmo NPairLoss. Por lo tanto, dejaremos de trabajar con la DB de Enfermedades en tejidos UMCM.

La Notebook quedó frenada en:
GIBD/CNN Marcas/Notebooks/Metric Learning/Replica.ipynb

Decidimos trabajar con la CNN utilizada en un principio por Fede Lederhos para la extracción de características, usando 203 clases. Y posteriormente aplicar la función de pérdida MultiClassLoss o NPairLoss. (Todo esto para el 13).
Tratando que la pérdida tienda lo más posible a cero.

El objetivo 2023 es utilizar MultiClassLoss o NPairLoss entrenando, para la generación de una medida de distancia del resultado obtenido ante una consulta.






8 de Noviembre 2022

Andrés nos pidió que analicemos el funcionamiento del algoritmo de: https://proceedings.neurips.cc/paper/2016/file/6b180037abbebea991d8b1232f8a8ca9-Paper.pdf

cuyo código está en:
https://git.code.tecnalia.com/comvis_public/piccolo/constellation_loss/-/tree/master/multiclass

Replicado en el archivo replica.ipynb que se encuentra en GIBD/CNN MArcas/Notebooks/ Metric Learning

Todo funciona pero investigamos el funcionamiento de
https://scikit-learn.org/stable/modules/manifold.html
2.2.9. t-distributed Stochastic Neighbor Embedding (t-SNE)


4 de Octubre 2022

El método constará en tomar objetos ancla (OA), y para cada una de ellas: 
Un objeto positivo (OP).
N objetos negativos (NP).

Para los OA, OP y NP trabajaremos con augmentación tanto de los positivos como de los negativos.

La augmentación será en dos pasos simples:

Valor A1 aleatorio entre un inicio y un fin (Ejemplo: 0 y 4), que se usará como la cantidad de transformaciones a aplicar.
Para cada uno de los A1 aplicar un aleatorio entre 1 y las posibles transformaciones (NA).

Luego, y teniendo el OA, el OP y los N NP se procede a entrenar.

Dato no menor, evitar la repetición de augmentaciones para evitar, por ejemplo, rotaciones excesivas o traslaciones que degeneran demasiado la imagen.

Hemos decidido comenzar con un límite de repetición de algoritmo de augmentación igual a 2 (RepA = 2).

Armar el Batch en lo posible para todas las imagenes, de esta manera se entrena al menos una vez cada imagen por iteración.

labels => [1,2,3, … , N] Donde N es el tamaño de Batch.
