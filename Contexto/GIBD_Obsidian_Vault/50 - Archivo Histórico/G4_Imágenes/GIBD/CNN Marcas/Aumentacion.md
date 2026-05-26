---
aliases: [Aumentacion]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2023-03-28
origen_zip: GIBD-20260521T205218Z-3-006.zip
ruta_interna: "GIBD/CNN Marcas/Notebooks/Metric Learning/Aumentacion.ipynb"
tamanio_bytes: 16093
---

# Notebook: Aumentacion.ipynb

Ruta interna: `GIBD/CNN Marcas/Notebooks/Metric Learning/Aumentacion.ipynb`

---


**[Celda 1 - Código]**
```python
# Imports

from keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D,Activation, Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from skimage.io import imshow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import tensorflow as tf
```


**[Celda 2 - Código]**
```python
#Definir funciones de transformacion de las imagenes para la augmentacion

from skimage import transform
import tensorflow.compat.v1 as tf1

tf1.disable_v2_behavior()


#FUNCIÓN "ROTATION" angulo -> entero entre 0 y 20 para que no sufra demasiados cabios
def rotar (imagen, angulo = 0):
    return transform.rotate(imagen, angle = angulo)

#FUNCIÓN "SCALE" escala -> Nº decimal entre 0.7 y 1 que devuelve la escala entre el 70% y 100% (sin cambios) 
def escalar (imagen, escala):
    if escala > 1:
        escala = 1
    aum = transform.rescale(imagen, scale = escala, mode = 'constant')
    filas = aum.shape[0]
    colum = aum.shape[1]
    aj_filas = (28-filas)//2
    aj_colum = (28-colum)//2
    tensor_aum = tf.image.pad_to_bounding_box(aum.reshape(filas,colum,1), aj_filas, aj_colum, 28, 28)
    return tensor_aum.eval(session = tf.compat.v1.Session())

#FUNCIÓN "CROP" dim_corte -> [n1,n2] n1 y n2 valores entre 20 y 27
def cortar (imagen, dim_corte):
    seed = np.random.randint(1234)
    tensor_crop = tf.image.random_crop(imagen, size = dim_corte, seed = seed)
    crop = tensor_crop.eval(session = tf.compat.v1.Session())
    filas = crop.shape[0]
    colum = crop.shape[1]
    aj_filas = (28-filas)//2
    aj_colum = (28-colum)//2
    tensor_aum = tf.image.pad_to_bounding_box(crop.reshape(filas,colum,1), aj_filas, aj_colum, 28, 28)
    return tensor_aum.eval(session = tf.compat.v1.Session())

#FUNCIÓN "GAUSSIAN NOISE" ruido -> Nº entre 0,01 y 0,1 para que no destruya
def gauss_ruido (imagen, ruido):
    noise = tf.random.normal(shape = tf.shape(imagen), mean = 0.0, stddev = ruido, dtype = tf.float64)
    tensor_aum = tf.add(imagen, noise)
    return tensor_aum.eval(session = tf.compat.v1.Session())

#FUNCIÓN "TRANSLATION" desp_fila -> Nº de filas a trsladar; desp_colum -> Nº de columnas a trsladar
def trasladar (imagen, desp_fila, desp_colum):
    translation = tf.image.pad_to_bounding_box(imagen.reshape(1,28,28,1), desp_fila, desp_colum, 28 + desp_fila, 28 + desp_colum)
    tensor_aum = tf.image.crop_to_bounding_box(translation, 0, 0, 28, 28)
    t = tensor_aum.eval(session=tf.compat.v1.Session())
    return t.reshape(28,28,1)
```


*Salida:*
```text
WARNING:tensorflow:From /usr/local/lib/python3.9/dist-packages/tensorflow/python/compat/v2_compat.py:107: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
```


**[Celda 3 - Código]**
```python
# Carga las imagenes de la DB para comenzar
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

directorio = '/content/sample_data/BDNormalizada(28x28)'
contenido = os.listdir(directorio)

imagenesBD = []
y_imagenesBD = []
for fichero in contenido:
    if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.jpg'):
        image = directorio + '/' + fichero
        image_gris = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        image_gris = image_gris / 255
        image_gris = 1 - image_gris
        imagenesBD.append(image_gris)
        n = fichero[:fichero.find('.')]
        y_imagenesBD.append(n)
print(y_imagenesBD)
```


**[Celda 4 - Código]**
```python
# Augmentación de una imagen determinada

import random

def augmentar_random(imagen):
  operador = random.randint(1,5)
  return {
      1: rotar(imagen, random.randint(0,20)),
      2: escalar(imagen, random.uniform(0.7,1)),
      3: cortar(imagen, [random.randint(21,23), random.randint(21,23)]),
      4: gauss_ruido(imagen,random.uniform(0.01,0.05)),
      5: trasladar(imagen, random.randint(1,5), random.randint(1,5))
  }.get(operador)


```


**[Celda 5 - Código]**
```python
# Generar Augmentación

import matplotlib.cm as cm
import matplotlib.pyplot as plt

imagen_original = imagenesBD[3]
imagen = imagen_original

fin = random.randint(2,5)


for i in range (1,fin):
  imagen = augmentar_random(imagen)

plt.imshow(imagen_original.reshape(28, 28), cmap=cm.Greys)
print(fin)

```


**[Celda 6 - Código]**
```python
# Imagen modificada
plt.imshow(imagen.reshape(28, 28), cmap=cm.Greys)
```


*Salida:*
```text
<matplotlib.image.AxesImage at 0x7f436692cdd0><Figure size 432x288 with 1 Axes>
```
