---
aliases: [GIBD 2023-Club futbol-Iván y Choi]
tags:
  - grupo/general
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2023-08-28
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Transfer Learning y Fine Tuning/Pruebas 2023/GIBD 2023-Club futbol-Iván y Choi.docx"
tamanio_bytes: 1074370
---

# GIBD 2023-Club futbol-Iván y Choi

Ruta interna: `GIBD/Transfer Learning y Fine Tuning/Pruebas 2023/GIBD 2023-Club futbol-Iván y Choi.docx`

---

Proyecto GIBD 2023
Extracción de características sobre BD marcas, utilizando transfer learning y transfer learning+fine tuning, con modelos entrenados sobre ImageNet y alguna otra BD de elementos parecidos a las marcas.
Tareas
Notebook Nueva DENSENET CLEAN: Similitud_Marcas_CNN_DenseNet_Clean.ipynb - Colaboratory (google.com)
Notebook vieja:Notebook dónde se importa el modelo densenet


Opcion 1
Crear otra base de datos y cambiarla a 299 299 3

Opcion 2
Cargar la imagen de 224,224, 3 utilizar una función, para cambiar la imagen a 299,299,3


Resultados de tasa de aciertos 
Densenet

45% de aciertos
Inception v3

Tasa de acierto 37,5%




INFORMACION TRANSFER LEARNING+FINE TUNING
1-Información Iván:
Introducción al transfer learning:
Básicamente cuando creamos modelos de redes neuronales lo que hacemos es darle cómo entrada un cierto dataset para luego hacer predicciones con otro de los mismos, es decir, descargamos dos datasets, uno de entrenamiento o lo que comúnmente se le llama train, y uno de prueba que nos sirve para hacer predicciones y que convencionalmente llamamos test.
Entonces con este procedimiento nuestra red neuronal, con los pesos que ha ido obteniendo a lo largo de su entrenamiento está entrenada para hacer predicciones sobre un dataset específico, lo que nos permite el transfer learning es traernos ese y aprovecharlo al máximo posible para hacer entrenamiento y pruebas sobre otros datasets diferentes.

Cómo vemos en el gráfico anterior tenemos un dataset de entrenamiento al que llamamos DA que sirve para entrenar al modelo A que tiene una ciertas capas, este modelo digamos que ahora está “capacitado” para poder realizar tareas específicas sobre el dataset al que llamamos TA ahora bien, aplicando transfer learning podemos reutilizar este modelo al que vamos a llamar A’ que va a ser entrenado con el dataset de entrenamiento DB, tendrá las capas B y podrá realizar tareas sobre el dataset de prueba TB

Aplicación teórica:
En palabras sencillas lo que básicamente hace el transfer learning es tomar una red neuronal que por ejemplo ya le enseñamos a “ver” ciertas cosas y hacemos las modificaciones necesarias para que pueda ver otro tipo de cosas.

En nuestro caso vamos a construir un modelo (se adjunta notebook y paper en referencias) que aprenda y ver y a clasificar flores en 5 grupos distintos, tenemos cómo datos de entrada un conjunto de imágenes pero poca cantidad, para realizar un modelo al que entrenamos desde cero necesitaríamos tal vez un conjunto con varios miles o millones de datos. 

Funcionamiento del modelo:
Tenemos un conjunto de datos de imágenes de flores al que debemos de normalizar, es decir, dejar las imágenes con valores entre 0 y 1 para poder trabajar, además de esto, debemos aplicarle un resize para asegurarnos de tener todas las imágenes de igual tamaño

Ahora al momento de aplicar transfer learning lo que haremos será tomar el modelo que nos traemos desde otro lado, que venía pre creado para clasificar imágenes en 1000 categorías diferentes, ahora, lo adaptamos para nuestro problema que es clasificar flores en 5 categorías, con los pesos ya entrenados, todo esto lo realizaremos usando resnet. 

Agregaremos una capa lineal que conecta 512 que son los elementos que nos da resnet y pasaremos 5 categorías, esta capa si la entrenaremos desde cero, que es lo que indica el freeze en false. 

Básicamente lo que hace la notebook adjuntada en la parte de referencias es entrenar un modelo que clasifica las flores desde cero, originalmente no tenemos ni mil imágenes por cada categoría por lo que el resultado del entrenamiento no será el más adecuado, luego realiza el mismo entrenamiento pero con el modelo ya pre entrenado, y se podrá visualizar la diferencia, sólo con 5 épocas de entrenamiento hemos podido conseguir un 80% de validación

Principal diferencia entre Transfer Learning y Transfer Learning+fine tuning: 
Cómo mencionamos en los párrafos anteriores el objetivo principal del transfer learning es que el modelo pre-entrenado se usa cómo un punto de partida para entrenar un nuevo modelo específico para la nueva tarea mientras que el transfer learning con ajuste fino el modelo que pre-entrenamos se usa cómo punto de partida y luego se ajusta para adaptarlo a la nueva tarea ajustando sus parámetros.




Segunda parte “Reutilización de las features aprendidas”
El verdadero legado que imagenet le hizo al mundo en realidad no fue esa arquitectura con tanta cantidad de imágenes sino el hecho de que podemos reutilizar los modelos ya entrenados para aplicarlos a diferentes tareas.

Problema disparador: Suponemos que una empresa nos contrata para reconocer imágenes de autos, entonces para poder entrenar una red necesitamos mucha cantidad de imágenes para entrenar el modelo y otro tanto para poder hacer las predicciones que nos piden. 
Con la transferencia de conocimiento podemos simplemente agarrar un modelo entrenado, copiar las features y aplicarlo a una tarea que esté relacionado debido a qué para entrenar un modelo de gran magnitud desde cero necesitamos mucha cantidad de recursos computacionales. 

Hay dos teorías de por qué funciona el transfer learning, una de ellas es porque imagenet es una base de datos lo  suficientemente grande cómo para acaparar casi todas las aristas del problema de reconocimiento que puede haber. 
La segunda es que fueron entrenadas para 
Referencias:
Dataset y notebook de transfer learning

Arquitectura del modelo densenet:
Capa de convolución 1x1: Esta capa reduce la dimensionalidad de las características que ingresan al bloque.

Capa de convolución 3x3: Esta capa es la principal capa convolucional del bloque y procesa las características de entrada.

Capa de normalización por lotes (Batch Normalization): Esta capa normaliza las características de salida de la capa de convolución para acelerar el entrenamiento y reducir el riesgo de sobreajuste.

Capa de activación (ReLU): Esta capa introduce una no-linealidad en la red al aplicar una función ReLU a las características normalizadas.

Capa de agrupación: Esta capa reduce la dimensionalidad de las características para reducir el costo computacional.

Además, la arquitectura de DenseNet utiliza conexiones densamente conectadas entre bloques, lo que significa que todas las capas de un bloque están conectadas a todas las capas del bloque siguiente. Esto permite que las características de todas las capas se reutilicen de manera efectiva y se combinen en capas posteriores para mejorar la precisión y eficiencia del entrenamiento.

En cuanto a sus parámetros, DenseNet tiene una serie de hiperparámetros1 que pueden ajustarse durante el entrenamiento, como el número de bloques, el número de capas convolucionales por bloque y el tamaño del filtro de convolución. Además, también hay parámetros de entrenamiento como la tasa de aprendizaje y la regularización L2, que también pueden ajustarse para mejorar la precisión del modelo.

1 HiperParámetros: Son parámetros que no son aprendidos directamente por el modelo durante el entrenamiento, sino que son establecidos previamente por el diseñador del modelo o por el usuario. Son valores que controlan el comportamiento y la complejidad del modelo, y afectan el proceso de aprendizaje, la velocidad de convergencia, la precisión del modelo y el riesgo de sobreajuste.

Los hiperparámetros comunes incluyen la tasa de aprendizaje, la tasa de regularización, el tamaño del lote, el número de capas ocultas, el número de neuronas por capa, el tamaño del kernel de convolución, el número de filtros en cada capa, la tasa de abandono (dropout), el tipo de función de activación, entre otros.

Ausencia de la capa flatten en Densenet:
DenseNet no tiene una capa "flatten" explícita en su arquitectura. En lugar de eso, utiliza una capa de agrupación global (Global Average Pooling) al final de la red para reducir la dimensión de las características a un vector unidimensional, que luego se alimenta a una capa totalmente conectada (Fully Connected) para producir las salidas de la red.

La capa de agrupación global toma como entrada un tensor de características de tamaño N x H x W x C, donde N es el tamaño del lote, H y W son la altura y el ancho de la imagen, y C es el número de canales (o características). La capa calcula la media de las características a lo largo de las dimensiones H y W para cada canal, produciendo un tensor de salida de tamaño N x C.

Este tensor de salida se puede conectar directamente a una capa totalmente conectada para producir las salidas finales de la red, sin necesidad de una capa "flatten" explícita para aplanar las características en un vector unidimensional. Esta técnica de agrupación global es común en otras arquitecturas de redes neuronales convolucionales modernas y se ha demostrado que es efectiva para reducir la cantidad de parámetros y mejorar la generalización de la red.
2-Informacion Zoe:
Transfer Learning
Es usado en machine learning, es el reuso de un modelo pre entrenado para un nuevo problema. En transfer learning, la machine explota el conocimiento ganado de una tarea anterior para mejorar la generalización de otra. Por ejemplo, en el entrenamiento de un clasificador para predecir si una imagen contiene comida, se puede usar el conocimiento ganado durante el entrenamiento para reconocer bebidas.

What is Transfer Learning? - YouTube
Concepto
Transfer learning es una técnica de machine learning donde una inteligencia artificial que ha sido entrenada para realizar una tarea específica es reusada (reutilizada)  en cierto punto para otra tarea similar.
Transfer learning es utilizado a partir de un modelo de IA pre entrenado, esto reduce drásticamente el tiempo computacional en caso que el entrenamiento sea realizado desde cero.


TensorFlow.js: Crea tu propia &qout;Teachable Machine&qout; usando aprendizaje por transferencia con TensorFlow.js  |  Google Codelabs
se muestra cómo compilar una app web a partir de un lienzo en blanco y recrear el popular sitio web "Teachable Machine" de Google. El sitio web te permite crear una aplicación web funcional que cualquier usuario puede utilizar para reconocer un objeto personalizado con solo algunas imágenes de ejemplo de su cámara web. 


Transfer Learning+fine tuning

Como crear el modelo
Para extraer características de una base de datos de imágenes utilizando transfer learning y transfer learning + fine tuning, se puede seguir los siguientes pasos:

Cargar el modelo pre-entrenado en ImageNet: Utilice una biblioteca de aprendizaje profundo como TensorFlow o PyTorch para cargar un modelo pre-entrenado en ImageNet. Algunos ejemplos de modelos pre-entrenados populares incluyen VGG, ResNet, Inception, DenseNet, etc.

Preprocesamiento de las imágenes: Preprocese las imágenes de la base de datos según los requisitos del modelo pre-entrenado. Esto podría incluir el escalado de las imágenes, la normalización y la conversión al formato requerido por el modelo.

Transfer Learning: Utilice el modelo pre-entrenado como extractor de características. Pase cada imagen de la base de datos a través del modelo y registre las salidas de una o más capas del modelo como las características extraídas. Estas características se pueden utilizar para entrenar un modelo personalizado o para realizar una tarea específica, como la clasificación de imágenes.

Transfer Learning + Fine-tuning: Si la base de datos contiene imágenes similares a las imágenes en ImageNet, se puede realizar un ajuste fino del modelo. En este enfoque, se descongelan una o más capas del modelo pre-entrenado y se vuelven a entrenar en la nueva base de datos. El ajuste fino permite que el modelo pre-entrenado se adapte mejor a las características específicas de la nueva base de datos.

Entrenamiento del modelo personalizado: Si se desea entrenar un modelo personalizado utilizando las características extraídas en el paso 3, se pueden utilizar algoritmos de aprendizaje supervisado, como SVM o Random Forests. Estos algoritmos utilizan las características extraídas para entrenar un modelo personalizado para la tarea de clasificación de imágenes.

En resumen, utilizar transfer learning y transfer learning + fine tuning con modelos pre-entrenados en ImageNet es una forma eficaz de extraer características de una base de datos de imágenes y utilizarlas para entrenar un modelo personalizado o realizar una tarea específica, como la clasificación de imágenes.

INFO INCEPTION-ZOE
Inception es un modelo de red neuronal convolucional (CNN) desarrollado por Google en 2014. La arquitectura de Inception está diseñada para ser más eficiente en términos de cómputo y memoria que los modelos CNN anteriores, al mismo tiempo que mejora el rendimiento en tareas de clasificación de imágenes.

El modelo Inception utiliza una técnica conocida como "módulos de Inception", que son bloques de capas convolucionales que se ramifican en paralelo y luego se concatenan para formar una salida única. Estos módulos utilizan filtros de diferentes tamaños de ventana (por ejemplo, 1x1, 3x3, 5x5) para extraer características de diferentes escalas.

Inception también utiliza técnicas de regularización, como la normalización por lotes (batch normalization) y la eliminación de conexiones (dropout), para reducir el sobreajuste y mejorar la capacidad de generalización del modelo.

 Inception ha demostrado ser efectivo en una variedad de tareas de clasificación de imágenes, reconocimiento de objetos y segmentación semántica en conjuntos de datos grandes, como ImageNet y COCO.

Implementación de código inception
NOTEBOOK: Similitud_Marcas_CNN_Inception_V3 - Colaboratory (google.com)

Chat gpt
Para implementar Inception v5 en Google Colab, se puede utilizar TensorFlow 2.x y cargar el modelo pre-entrenado a través de la API de Keras.

Aquí hay un ejemplo de cómo cargar el modelo pre-entrenado de Inception v5 y hacer una predicción en una imagen en Google Colab:

Primero, se debe importar TensorFlow 2.x y cargar el modelo pre-entrenado de Inception v5 desde la API de Keras:

import tensorflow as tf

# Cargar el modelo pre-entrenado de Inception v5
model = tf.keras.applications.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)


Luego, se puede descargar una imagen de prueba y preprocesarla para que coincida con el tamaño de entrada del modelo (299x299 píxeles) y normalizar los valores de píxel:

import urllib.request
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# Descargar imagen de prueba
urllib.request.urlretrieve("https://www.python.org/static/img/python-logo.png", "python-logo.png")

# Cargar imagen de prueba y preprocesarla
img = Image.open('python-logo.png').resize((299, 299))
img_array = img_to_array(img)
img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
img_array = tf.expand_dims(img_array, axis=0)


Finalmente, se puede hacer una predicción en la imagen utilizando el modelo cargado:

# Hacer una predicción en la imagen de prueba
prediction = model.predict(img_array)

# Decodificar la predicción y mostrar las 5 clases principales
decoded_predictions = tf.keras.applications.inception_v3.decode_predictions(prediction, top=5)[0]
for i, (class_id, class_name, score) in enumerate(decoded_predictions):
    print(f"{i+1}. {class_name} ({score:.2f})")

RESULTADOS




Info Rest Net- Ivan

Densenet hacer el fine tuning.
Transferencia de aprendizaje y ajuste  |  TensorFlow Core

Conv5_block9_1_conv

Después de entrenar el modelo asignar la capa global 2d pooling a una variable.
 Volver a hacer las busquedas por similitud para ver el porcentaje de aciertos. 
Cargar las consultas antes de hacer las busquedas
