---
aliases: [AFLM_HWOBC_CNN_Siamesa]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2023-08-14
origen_zip: GIBD-20260521T205218Z-3-004.zip
ruta_interna: "GIBD/ARTICULOS NUESTROS/2023 - CONAIISI/Caracteres Chinos/AFLM_HWOBC_CNN_Siamesa.ipynb"
tamanio_bytes: 7971
---

# Notebook: AFLM_HWOBC_CNN_Siamesa.ipynb

Ruta interna: `GIBD/ARTICULOS NUESTROS/2023 - CONAIISI/Caracteres Chinos/AFLM_HWOBC_CNN_Siamesa.ipynb`

---

IMPORTS


**[Celda 2 - Código]**
```python
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
import matplotlib.cm as cm
import cv2
import math
import shutil
```

MODELO


**[Celda 4 - Código]**
```python
#Model

model = keras.Sequential()

model.add(tf.keras.layers.Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(32,kernel_size=3,activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))

model.add(tf.keras.layers.Conv2D(64,kernel_size=3,activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64,kernel_size=3,activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Dropout(0.4))
#model.add(tf.keras.layers.Dense(10, activation='softmax'))

#model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
```

SIAMESA



**[Celda 6 - Código]**
```python
# SIAMESA

# We have 2 inputs, 1 for each picture
left_input = Input((28,28,1))
right_input = Input((28,28,1))


# We will use 2 instances of 1 network for this task

# Connect each 'leg' of the network to each input
# Remember, they have the same weights
encoded_l = model(left_input)
encoded_r = model(right_input)

# Getting the L1 Distance between the 2 encodings
L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))

# Add the distance function to the network
L1_distance = L1_layer([encoded_l, encoded_r])

prediction = Dense(1,activation='sigmoid')(L1_distance)
siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

optimizer = Adam(0.001)

#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])
```

Conectar con Google



**[Celda 8 - Código]**
```python
# Conecto la carpeta Drive con las Imagenes
# Monta Google Drive en Colab
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

left_right_target_creator


**[Celda 10 - Código]**
```python
def left_right_target_creator(lista_distintos, lista_similares):
    left = []
    right = []
    target = []

    for par in lista_distintos:


        left.append(par[0])  # en vez de guardar las etiquetas, guardar las imagenes.
        right.append(par[1]) # en vez de guardar las etiquetas, guardar las imagenes.
        target.append(1.0)

    for par in lista_similares:

        left.append(par[0])
        right.append(par[1])
        target.append(0.0)


    return left, right, target
```

left_right_target (Test) creator


**[Celda 12 - Código]**
```python
def left_right_target_creator_test(lista_distintos, lista_similares):
    test_left = []
    test_right = []
    test_targets = []

    for _ in range(math.trunc(len(lista_distintos) * 0.20)):
        random_distinto = random.choice(lista_distintos)
        test_left.append(random_distinto[0])
        test_right.append(random_distinto[1])
        test_targets.append(1.0)

    for _ in range(math.trunc(len(lista_similares) * 0.20)):
        random_similar = random.choice(lista_similares)
        test_left.append(random_similar[0])
        test_right.append(random_similar[1])
        test_targets.append(0.0)

    test_left = np.squeeze(np.array(test_left))
    test_right = np.squeeze(np.array(test_right))
    test_targets = np.squeeze(np.array(test_targets))

    return test_left, test_right, test_targets
```

Entrenamiento


**[Celda 14 - Código]**
```python
#Prueba

siamese_net.fit([left_input,right_input], targets,
          batch_size=16,
          epochs=40,
          verbose=1,
          validation_data=([test_left,test_right],test_targets))
```

Carga de datos


**[Celda 16 - Código]**
```python
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

directorio = '/content/drive/MyDrive/Grupo Investigación /BD_CaracteresChinos'
contenido = os.listdir(directorio)

imagenesBD = []
y_imagenesBD = []
for fichero in contenido:
    if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.JPG'):
        image = directorio + '/' + fichero
        image_gris = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        image_gris = image_gris / 255
        image_gris = 1 - image_gris
        image_gris = image_gris.reshape(28, 28, 1)
        imagenesBD.append(image_gris)
        n = fichero[:fichero.find('.')]
        y_imagenesBD.append(n)
print(y_imagenesBD)
```

Consulta


**[Celda 18 - Código]**
```python
def nnk(actuales,nuevo,k=1):
    p = 0
    while (p<len(actuales) and (nuevo[0]>=actuales[p][0])):
        p += 1
    actuales.insert(p, nuevo)
    while len(actuales)>k:
        actuales.pop(len(actuales)-1)
```


**[Celda 19 - Código]**
```python
def nnk2etiquetas(nnklist, y_BD):
    l = []
    for par in nnklist:
        nuevo = [par[0], y_BD[par[1]]]
        l.append(nuevo)
    return l
```
