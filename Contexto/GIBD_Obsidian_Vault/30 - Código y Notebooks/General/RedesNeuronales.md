---
aliases: [RedesNeuronales]
tags:
  - grupo/general
  - estado/util
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2022-12-15
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Tareas Iván/RedesNeuronales.ipynb"
tamanio_bytes: 259295
---

# Notebook: RedesNeuronales.ipynb

Ruta interna: `GIBD/Tareas Iván/RedesNeuronales.ipynb`

---


**[Celda 1 - Código]**
```python

```

# Primera red neuronal(aprende a pasar grados C a Farenheit)


**[Celda 3 - Código]**
```python
#Importo las librerías
import tensorflow as tf
import numpy as np 

```


**[Celda 4 - Código]**
```python
#Ahora creo un arreglo con grados Celsius y otro con los fahrenheit
celsius = np.array([-40,-10,0,8,15,22,38], dtype=float)
fahrenheit = np.array([-40,14,32,46,59,72,100], dtype=float)

#cada numero de celsius se relaciona con uno de los fahrenheit, el primero con primero, segundo con segundo, etc.
```


**[Celda 5 - Código]**
```python
#Se crean las capas y la entradas ambas tienen una sola neurona por esto es que se pone1 
capa = tf.keras.layers.Dense(units=1,input_shape=[1])
#uso el modelo secuencial
modelo= tf.keras.Sequential([capa])

```


**[Celda 6 - Código]**
```python
#le doy 0.1 que es el grado de ajustes que tiene que ir haciendo red para poder aprender, si le damos un numero muy chico va a ser preciso, pero el aprendizaje 
#puede volverse lento, en cambio con un número mas grande el aprendizaje es más rápido pero poco preciso. Lo ideal es encontrar un balance entre los dos puntos.
#llamamos a este numero taza de aprendizaje
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)
#loss es la función de pérdida, usamos la media cuadrática esta funcón considera que poca cantidad de grandes errores es mucho peor
#que una cantidad más grande de errores pequeños 



```


**[Celda 7 - Código]**
```python
#preparo el entrenamiento
print("Comenzando el entrenamiento...")
historial=modelo.fit(celsius,fahrenheit,epochs=1000,verbose=False)
#Hay siete datos entonces una vuelta(epoch) sería revisar los 7 datos una sola vez , pongo mil vueltas
print("Modelo entrenado")

```


*Salida:*
```text
Comenzando el entrenamiento...
Modelo entrenado
```


**[Celda 8 - Código]**
```python
#Ahora antes de predecir vemos el resultado de la función de pérdida:
import matplotlib.pyplot as plt
plt.xlabel("vueltas")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
#ejecuto la función de pérdida, en el eje x van las vuetlas que fue dando para ser entrenado
#en el eje y pongo las magnitudes de pérdida van a ser la distancia de error que va a haber entre los resultados reales y los lanzados


```


*Salida:*
```text
[<matplotlib.lines.Line2D at 0x7f9840189350>]<Figure size 432x288 with 1 Axes>
```


**[Celda 9 - Código]**
```python
#análisis del gráfico: 
#se entiende que a las 500/600 vueltas ya alcanza como entrenamiento para el modelo no son necesarias 1000 
#Ahora se hace una predicción, 100 celsius son 212 F realmente, la red entrenada predijo un número aproximado
celsius = 100.0
print("Haciendo predicción")
resultado = modelo.predict([celsius])
print("El resultado es " +str(resultado) + "fahrenheit")
```


*Salida:*
```text
Haciendo predicción
El resultado es [[211.7441]]fahrenheit
```


**[Celda 10 - Código]**
```python
#ahora imprimo las variables internas del modelo, peso, sesgo 
print(capa.get_weights())
```


*Salida:*
```text
[array([[1.7981299]], dtype=float32), array([31.9311], dtype=float32)]
```

Puede verse que le asignó 1.79 al peso y al sesgo lo dejo en 31.9. Lo que hizo fue lo siguiente: 
Entrada( 100C * 1.798 ) + 31.9 = 211.74



**[Celda 12 - Código]**
```python

```

En el documento va a quedar el gráfico explicado de como funciona la red



**[Celda 14 - Código]**
```python

```

# Primer clasificador de imágenes

Importo las librerías para poder descargar el dataset de Zalando


**[Celda 17 - Código]**
```python
import tensorflow as tf
import tensorflow_datasets as tfds
```

Descargo el dataset que Zalando tiene subido en github


**[Celda 19 - Código]**
```python
datos, metadatos = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

```

imprimo el valor de los metadatos, donde puede verse que hay 60.000 datos para entrenamiento y 10.000 para prueba


**[Celda 21 - Código]**
```python
metadatos
```


*Salida:*
```text
tfds.core.DatasetInfo(
    name='fashion_mnist',
    full_name='fashion_mnist/3.0.1',
    description="""
    Fashion-MNIST is a dataset of Zalando's article images consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.
    """,
    homepage='https://github.com/zalandoresearch/fashion-mnist',
    data_path='~/tensorflow_datasets/fashion_mnist/3.0.1',
    file_format=tfrecord,
    download_size=29.45 MiB,
    dataset_size=36.42 MiB,
    features=FeaturesDict({
        'image': Image(shape=(28, 28, 1), dtype=tf.uint8),
        'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=10),
    }),
    supervised_keys=('image', 'label'),
    disable_shuffling=False,
    splits={
        'test': <SplitInfo num_examples=10000, num_shards=1>,
        'train': <SplitInfo num_examples=60000, num_shards=1>,
    },
    citation="""@article{DBLP:journals/corr/abs-1708-07747,
      author    = {Han Xiao and
                   Kashif Rasul and
                   Roland Vollgraf},
      title     = {Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning
                   Algorithms},
      journal   = {CoRR},
      volume    = {abs/1708.07747},
      year      = {2017},
      url       = {http://arxiv.org/abs/1708.07747},
      archivePrefix = {arXiv},
      eprint    = {1708.07747},
      timestamp = {Mon, 13 Aug 2018 16:47:27 +0200},
      biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1708-07747},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }""",
)
```

Agrupo cada conjunto de datos en una variable diferente para poder usarlos:


**[Celda 23 - Código]**
```python
datos_entrenamiento, datos_pruebas = datos['train'], datos['test']
```

Los metadatos además traen las categorías que existen en el dataset, los asigno a una variable para poder verlos


**[Celda 25 - Código]**
```python
nombres_clases = metadatos.features['label'].names
```


**[Celda 26 - Código]**
```python
nombres_clases

```


*Salida:*
```text
['T-shirt/top',
 'Trouser',
 'Pullover',
 'Dress',
 'Coat',
 'Sandal',
 'Shirt',
 'Sneaker',
 'Bag',
 'Ankle boot']
```

Podemos ver que el índice 0 es camiseta, 1=pantalón, etc.


Cuando se quiere entrenar una red, generalmente queremos pasar los datos a entradas entre 0 y 1, esto ayuda muchísimo a la velocidad en los entrenamientos, actualmente los valores de entrada son de 0..255, por los valores de los pixeles, voy a usar una función de normalización para esto:


**[Celda 29 - Código]**
```python
def normalizar(imagenes, etiquetas):
  imagenes = tf.cast(imagenes, tf.float32)
  imagenes /= 255 #en esta parte se pasa de 0..255 a 0..1
  return imagenes, etiquetas

```


**[Celda 30 - Código]**
```python
#Normalizo a continuación todos los datos de entrenamiento y prueba usando la función anterior
datos_entrenamiento = datos_entrenamiento.map(normalizar)
datos_pruebas = datos_pruebas.map(normalizar)

#Ahora agregamos los datos a memoria caché, se hace más rápido el entrenamiento ya que se guardan los datos en memoria y no en disco
datos_entrenamiento = datos_entrenamiento.cache()
datos_pruebas = datos_pruebas.cache()


```

Acá voy a mostrar la primer imagen de todo el conjunto de datos:


**[Celda 32 - Código]**
```python
for imagen, etiqueta in datos_entrenamiento.take(1):
  break
imagen = imagen.numpy().reshape((28,28)) #la función reshape me redimensiona las imágenes para que queden en pixeles de 28x28

import matplotlib.pyplot as plt

#Dibujo la imagen correspondiente: 
plt.figure()
plt.imshow(imagen, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

```


*Salida:*
```text
<Figure size 432x288 with 2 Axes>
```

Ahora imprimo las imágenes y muestra cual es la categoría correcta en la que viene etiquetada:


**[Celda 34 - Código]**
```python
plt.figure(figsize=(10,10))
for i, (imagen, etiqueta) in enumerate(datos_entrenamiento.take(25)):
  imagen = imagen.numpy().reshape((28,28))
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(imagen, cmap=plt.cm.binary)
  plt.xlabel(nombres_clases[etiqueta])
plt.show()
```


*Salida:*
```text
<Figure size 720x720 with 25 Axes>
```

Voy a definir la capa de entrada de forma manual, voy a usar la capa flatten a la que le paso como parametros 28,28 que son los pixeles de las imagenes y 1 es lo que hace flatten es aplastar a 1 dimension, blanco y negro es siempre 1, ahora agrego dos capas ocultas con 50 neuronas y capas densas, la capa de salida, y cada una de esas capas con la función de activación relu, la ultima, la capa de salida con 10 neuronas que son las 10 categorías que tiene el dataset, pero tiene la función de activación softmax, se usa siempre la función de activación softmax, lo que hace es asegurar que la suma de todas las neuronas de salida de 1, haciendo el trabajo de la siguiente manera: si al final el resultado es 0.8 sueter, 0 camisa, 0.1 bag, 0.1 pantalon, entonces el resultado de la predicción es el número mayor osea sueter:




**[Celda 36 - Código]**
```python
modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),  #el numero 1 corresponde al blanco y negro
    tf.keras.layers.Dense(50, activation=tf.nn.relu), 
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])
```

ahora compilo el modelo usando una función de perdida, con el algoritmo adam 


**[Celda 38 - Código]**
```python
modelo.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
    metrics=['accuracy']
)
```

Como son 70mil ejemplos entonces podemos dividir en lotes para poder ir ejecutando de a poco las imágenes, voy a definir en dos variables arriba entonces voy a tener los datos de prueba en una y los de entrenamiento en otra variable que la voy a usar para pasar como parámetros después.-


**[Celda 40 - Código]**
```python
numEntrenamiento=metadatos.splits["train"].num_examples
numPruebas=metadatos.splits["test"].num_examples
```

el shuffle es una función que hace de manera aleatoria las imagenes, además batch es otra función a la que le pasamos el tamaño de lote


**[Celda 42 - Código]**
```python
tamano_lote=32

datos_entrenamiento=datos_entrenamiento.repeat().shuffle(numEntrenamiento).batch(tamano_lote)
datos_pruebas=datos_pruebas.batch(tamano_lote)
```

Empieza el entrenamiento del modelo con la función fit, le pasamos las vueltas que queremos darle a los datos, y la función steps per epoch, cuando se ejecuta cada línea obtengo al final la precisión en cada vuelta, en general debemos obtener una precisión de alrededor del 88%


**[Celda 44 - Código]**
```python
import math

historial = modelo.fit(datos_entrenamiento, epochs = 4, steps_per_epoch= math.ceil(numEntrenamiento/tamano_lote))
```


*Salida:*
```text
Epoch 1/4
1875/1875 [==============================] - 9s 2ms/step - loss: 0.5193 - accuracy: 0.8164
Epoch 2/4
1875/1875 [==============================] - 5s 3ms/step - loss: 0.3869 - accuracy: 0.8601
Epoch 3/4
1875/1875 [==============================] - 4s 2ms/step - loss: 0.3494 - accuracy: 0.8721
Epoch 4/4
1875/1875 [==============================] - 4s 2ms/step - loss: 0.3262 - accuracy: 0.8794
```

A continuación vemos el resultado que arroja la funcióin de pérdida en cada vuelta (le llamamos época)


**[Celda 46 - Código]**
```python
plt.xlabel("Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])
```


*Salida:*
```text
[<matplotlib.lines.Line2D at 0x7f4d1917bd50>]<Figure size 432x288 with 1 Axes>
```

vemos que cuando se acerca al 3 (las epocas van del 0 al 4 entonces la magnitud de pérdidas tiende a 0). 
Ahora a contiunación traigo un código de mtplotlib para imprimir imágenes junto con la predicción que hace el modelo, marcando en azul cuando le acierta y en rojo cuando no:


**[Celda 48 - Código]**
```python
import numpy as np

for imagenes_prueba, etiquetas_prueba in datos_pruebas.take(1):
  imagenes_prueba = imagenes_prueba.numpy()
  etiquetas_prueba = etiquetas_prueba.numpy()
  predicciones = modelo.predict(imagenes_prueba)
  
def graficar_imagen(i, arr_predicciones, etiquetas_reales, imagenes):
  arr_predicciones, etiqueta_real, img = arr_predicciones[i], etiquetas_reales[i], imagenes[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img[...,0], cmap=plt.cm.binary)

  etiqueta_prediccion = np.argmax(arr_predicciones)
  if etiqueta_prediccion == etiqueta_real:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(nombres_clases[etiqueta_prediccion],
                                100*np.max(arr_predicciones),
                                nombres_clases[etiqueta_real]),
                                color=color)
  
def graficar_valor_arreglo(i, arr_predicciones, etiqueta_real):
  arr_predicciones, etiqueta_real = arr_predicciones[i], etiqueta_real[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  grafica = plt.bar(range(10), arr_predicciones, color="#777777")
  plt.ylim([0, 1]) 
  etiqueta_prediccion = np.argmax(arr_predicciones)
  
  grafica[etiqueta_prediccion].set_color('red')
  grafica[etiqueta_real].set_color('blue')
  
filas = 5
columnas = 5
num_imagenes = filas*columnas
plt.figure(figsize=(2*2*columnas, 2*filas))
for i in range(num_imagenes):
  plt.subplot(filas, 2*columnas, 2*i+1)
  graficar_imagen(i, predicciones, etiquetas_prueba, imagenes_prueba)
  plt.subplot(filas, 2*columnas, 2*i+2)
  graficar_valor_arreglo(i, predicciones, etiquetas_prueba)
```


*Salida:*
```text
<Figure size 1440x720 with 50 Axes>
```

A continuación dejo el código para probar cualquier imagen del dataset y ver la predicción:


**[Celda 50 - Código]**
```python
imagen = imagenes_prueba[8] #AL ser la variable imagenes_prueba solo tiene lo que se le puso en el bloque anterior
imagen = np.array([imagen])
prediccion = modelo.predict(imagen)

print("Prediccion: " + nombres_clases[np.argmax(prediccion[0])])
```


*Salida:*
```text
Prediccion: Sneaker
```

A continuación voy a exportar el modelo y usarlo con tensorflow.js


**[Celda 52 - Código]**
```python
#Exportacion del modelo a h5
modelo.save('modelo_exportado.h5')
```


**[Celda 53 - Código]**
```python
#Instalar tensorflowjs para convertir el h5 a un modelo que pueda cargar tensorflowjs en un explorador
!pip install tensorflowjs
```


*Salida:*
```text
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting tensorflowjs
  Downloading tensorflowjs-3.20.0-py3-none-any.whl (81 kB)
[K     |████████████████████████████████| 81 kB 3.3 MB/s 
[?25hRequirement already satisfied: protobuf<3.20,>=3.9.2 in /usr/local/lib/python3.7/dist-packages (from tensorflowjs) (3.17.3)
Requirement already satisfied: six<2,>=1.12.0 in /usr/local/lib/python3.7/dist-packages (from tensorflowjs) (1.15.0)
Collecting flax>=0.5.3
  Downloading flax-0.6.0-py3-none-any.whl (180 kB)
[K     |████████████████████████████████| 180 kB 20.0 MB/s 
[?25hRequirement already satisfied: tensorflow-hub<0.13,>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from tensorflowjs) (0.12.0)
Requirement already satisfied: tensorflow<3,>=2.1.0 in /usr/local/lib/python3.7/dist-packages (from tensorflowjs) (2.8.2+zzzcolab20220719082949)
Requirement already satisfied: jax>=0.3.16 in /usr/local/lib/python3.7/dist-packages (from tensorflowjs) (0.3.17)
Requirement already satisfied: importlib_resources>=5.9.0 in /usr/local/lib/python3.7/dist-packages (from tensorflowjs) (5.9.0)
Collecting packaging~=20.9
  Downloading packaging-20.9-py2.py3-none-any.whl (40 kB)
[K     |████████████████████████████████| 40 kB 7.5 MB/s 
[?25hRequirement already satisfied: matplotlib in /usr/local/lib/python3.7/dist-packages (from flax>=0.5.3->tensorflowjs) (3.2.2)
Requirement already satisfied: numpy>=1.12 in /usr/local/lib/python3.7/dist-packages (from flax>=0.5.3->tensorflowjs) (1.21.6)
Collecting rich~=11.1
  Downloading rich-11.2.0-py3-none-any.whl (217 kB)
[K     |████████████████████████████████| 217 kB 54.7 MB/s 
[?25hRequirement already satisfied: msgpack in /usr/local/lib/python3.7/dist-packages (from flax>=0.5.3->tensorflowjs) (1.0.4)
Collecting optax
  Downloading optax-0.1.3-py3-none-any.whl (145 kB)
[K     |████████████████████████████████| 145 kB 67.3 MB/s 
[?25hRequirement already satisfied: typing-extensions>=4.1.1 in /usr/local/lib/python3.7/dist-packages (from flax>=0.5.3->tensorflowjs) (4.1.1)
Requirement already satisfied: PyYAML>=5.4.1 in /usr/local/lib/python3.7/dist-packages (from flax>=0.5.3->tensorflowjs) (6.0)
Requirement already satisfied: zipp>=3.1.0 in /usr/local/lib/python3.7/dist-packages (from importlib_resources>=5.9.0->tensorflowjs) (3.8.1)
Requirement already satisfied: scipy>=1.5 in /usr/local/lib/python3.7/dist-packages (from jax>=0.3.16->tensorflowjs) (1.7.3)
Requirement already satisfied: opt-einsum in /usr/local/lib/python3.7/dist-packages (from jax>=0.3.16->tensorflowjs) (3.3.0)
Requirement already satisfied: absl-py in /usr/local/lib/python3.7/dist-packages (from jax>=0.3.16->tensorflowjs) (1.2.0)
Requirement already satisfied: etils[epath] in /usr/local/lib/python3.7/dist-packages (from jax>=0.3.16->tensorflowjs) (0.7.1)
Requirement already satisfied: pyparsing>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging~=20.9->tensorflowjs) (3.0.9)
Collecting commonmark<0.10.0,>=0.9.0
  Downloading commonmark-0.9.1-py2.py3-none-any.whl (51 kB)
[K     |████████████████████████████████| 51 kB 8.6 MB/s 
[?25hRequirement already satisfied: pygments<3.0.0,>=2.6.0 in /usr/local/lib/python3.7/dist-packages (from rich~=11.1->flax>=0.5.3->tensorflowjs) (2.6.1)
Collecting colorama<0.5.0,>=0.4.0
  Downloading colorama-0.4.5-py2.py3-none-any.whl (16 kB)
Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (1.14.1)
Requirement already satisfied: flatbuffers>=1.12 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (2.0.7)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (1.48.1)
Requirement already satisfied: keras<2.9,>=2.8.0rc0 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (2.8.0)
Requirement already satisfied: h5py>=2.9.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (3.1.0)
Requirement already satisfied: keras-preprocessing>=1.1.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (1.1.2)
Requirement already satisfied: tensorflow-estimator<2.9,>=2.8 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (2.8.0)
Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (1.6.3)
Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (0.2.0)
Requirement already satisfied: gast>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (0.5.3)
Requirement already satisfied: tensorboard<2.9,>=2.8 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (2.8.0)
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (0.26.0)
Requirement already satisfied: setuptools in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (57.4.0)
Requirement already satisfied: libclang>=9.0.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (14.0.6)
Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (1.1.0)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.7/dist-packages (from astunparse>=1.6.0->tensorflow<3,>=2.1.0->tensorflowjs) (0.37.1)
Requirement already satisfied: cached-property in /usr/local/lib/python3.7/dist-packages (from h5py>=2.9.0->tensorflow<3,>=2.1.0->tensorflowjs) (1.5.2)
Requirement already satisfied: google-auth<3,>=1.6.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (1.35.0)
Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (0.4.6)
Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (2.23.0)
Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (1.8.1)
Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (0.6.1)
Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (1.0.1)
Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (3.4.1)
Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (4.2.4)
Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (4.9)
Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (0.2.8)
Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (1.3.1)
Requirement already satisfied: importlib-metadata>=4.4 in /usr/local/lib/python3.7/dist-packages (from markdown>=2.6.8->tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (4.12.0)
Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.7/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (0.4.8)
Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (2022.6.15)
Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (2.10)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (1.24.3)
Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (3.2.0)
Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->flax>=0.5.3->tensorflowjs) (2.8.2)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib->flax>=0.5.3->tensorflowjs) (0.11.0)
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->flax>=0.5.3->tensorflowjs) (1.4.4)
Requirement already satisfied: jaxlib>=0.1.37 in /usr/local/lib/python3.7/dist-packages (from optax->flax>=0.5.3->tensorflowjs) (0.3.15+cuda11.cudnn805)
Collecting chex>=0.0.4
  Downloading chex-0.1.5-py3-none-any.whl (85 kB)
[K     |████████████████████████████████| 85 kB 4.2 MB/s 
[?25hRequirement already satisfied: toolz>=0.9.0 in /usr/local/lib/python3.7/dist-packages (from chex>=0.0.4->optax->flax>=0.5.3->tensorflowjs) (0.12.0)
Requirement already satisfied: dm-tree>=0.1.5 in /usr/local/lib/python3.7/dist-packages (from chex>=0.0.4->optax->flax>=0.5.3->tensorflowjs) (0.1.7)
Installing collected packages: commonmark, colorama, chex, rich, optax, packaging, flax, tensorflowjs
  Attempting uninstall: packaging
    Found existing installation: packaging 21.3
    Uninstalling packaging-21.3:
      Successfully uninstalled packaging-21.3
Successfully installed chex-0.1.5 colorama-0.4.5 commonmark-0.9.1 flax-0.6.0 optax-0.1.3 packaging-20.9 rich-11.2.0 tensorflowjs-3.20.0
```


**[Celda 54 - Código]**
```python
#Convertir el archivo h5 a formato de tensorflowjs
!mkdir tfjs_target_dir
!tensorflowjs_converter --input_format keras modelo_exportado.h5 tfjs_target_dir
```

Chequeo a continuación que se haya creado la carpeta


**[Celda 56 - Código]**
```python
!ls
```


*Salida:*
```text
'~'   modelo_exportado.h5   sample_data   tfjs_target_dir
```

Ahora imprimo el contenido de la carpeta:


**[Celda 58 - Código]**
```python
 !ls tfjs_target_dir
```


*Salida:*
```text
group1-shard1of1.bin  model.json
```

# Problema de clasificacion con dos circulos en eje x e y 


**[Celda 60 - Código]**
```python
import scipy as sc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
```


**[Celda 61 - Código]**
```python
 #Creacion del dataset
#make_circles? <- Sirve para que salga la documentacion de make_circles 
n = 500
p = 2 
#n es el numero de registros que tenemos y p son las caracteristicas que tenemos sobre cada registro
X , Y = make_circles(n_samples=n , factor = 0.5 , noise=0.05)
#factor de 0.5 es la distancia entre los dos circulos, si le dieramos 0 serian dos circulos perfectos
#noise es el ruido es decir la dispersion de puntos
#Si imprimiesemos el eje y seria unn vector binario, es decir, con 0 y 1, lo que significan estos es que si pertenece a un circulo o al otro, esto lo imprimimos en la linea de abajo
#Cuando la y vale cero imprimo un circulo celeste y cuando es 1 imprimo salmon 
plt.scatter(X[Y==0,0],X[Y==0,1],c="skyblue")
plt.scatter(X[Y==1,0],X[Y==1,1],c="salmon")
plt.axis("equal")
plt.show()
```


*Salida:*
```text
<Figure size 432x288 with 1 Axes>
```


**[Celda 62 - Código]**
```python
#clase de la capa de la red

class neural_layer():  
   #pasamos como parametros, numero de condiciones, numero de neuronas, funcion de activacion
   #esta es la funcion que inicializa nuestra capa neural layer
  def __init__(self,n_conn,n_neur, act_f):
    self.act_f = act_f #la funcion de act de la capa es igual al parametro que le pasamos
    self.b = np.random.rand(1,n_neur) * 2 - 1 #inicializamos aleatorios los parametros de la capa, nos devuelve un valor entre cero y uno, quiero que vaya de
    #-1 a 1 entonces es la multiplicacion y la resta
    self.w = np.random.rand(n_conn,n_neur) * 2 - 1 #w es una matriz con los parametros de numeros de conexiones y d neuronas
```


**[Celda 63 - Código]**
```python
#Funciones de activacion
#la f.a es la funcion por la cual se pasa la suma ponderada que se realiza en la neurona esta introduce en nuestra red neuronal no-linealidades que
#nos permite combinar muchas neuronas
#usamos funcion de activacion sigmoide, definiendo con la palabra reservada lambda  

#debemos calcular la derivada de la funcion sigmoide para esto definimos un par
# <- funcion de activacion en forma de s
sigm = (lambda x: 1 / (1 + np.e ** (-x)),  
        lambda x: x * (1-x))

_x = np.linspace(-5,5,100) #va de -5 a 5 en forma lineal y genera 100 valores
plt.plot(_x,sigm[0](_x))
#cuando queramos acceder a la funcion sigmoide accedemos a esta con el indice cero, si le pasamos el 1 es la derivada
```


*Salida:*
```text
[<matplotlib.lines.Line2D at 0x7f5f029f64f0>]<Figure size 432x288 with 1 Axes>
```


**[Celda 64 - Código]**
```python
#l0 = neural_layer(p, 4, sigm)
#l1 = neural_layer(4, 8, sigm)
# ... 
#Podriamos ir una por una definiendo las capas de la red neuronal pero usamos la funcion creada anteriormente 

#la ultima capa tiene solo una neurona de salida porq el resultado que queremos generar es binario, o es de una clase o es de otra 
topology = [p,4,8,16,8,4,1]
def create_nn(topology,act_f):
  #nn es neural network este es la estructura que sostenga toda la estructura de nuestra red neuronal
  nn=[]
  for l, layer in enumerate(topology[:-1]):
    nn.append(neural_layer(topology[l], topology[l+1],act_f)) #tenemos la primera capa de conexiones que pertenece al neural_layer en la posicion l, l es el indice que recorremos durante todo el bucle for, y tantas neuronaas como hayamos marcado en l+1

  return nn

create_nn(topology , sigm) #creamos una nueva red neuronal con el vector topology y la funcion de activacion sigmoide, tiene 6 capas ocultas correspondientes a cada valor del vector topology, con esto tenemos la estructura de datos que soporta nuestra red neuronal

```


*Salida:*
```text
[<__main__.neural_layer at 0x7f5f029a1b80>,
 <__main__.neural_layer at 0x7f5f02984070>,
 <__main__.neural_layer at 0x7f5f029a1ac0>,
 <__main__.neural_layer at 0x7f5f0cd39eb0>,
 <__main__.neural_layer at 0x7f5f0ccd6cd0>,
 <__main__.neural_layer at 0x7f5f029a6790>]
```


**[Celda 65 - Código]**
```python
#min 31:01
```
