---
aliases: [Copia de ML -  Similitud-HWOBC_CNN_Siamesa_2023]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2024-06-19
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/CNN Marcas/Notebooks/Copia de ML -  Similitud-HWOBC_CNN_Siamesa_2023.ipynb"
tamanio_bytes: 55784
---

# Notebook: Copia de ML -  Similitud-HWOBC_CNN_Siamesa_2023.ipynb

Ruta interna: `GIBD/CNN Marcas/Notebooks/Copia de ML -  Similitud-HWOBC_CNN_Siamesa_2023.ipynb`

---

# Busqueda por Similitud de Marcas

## 1 Cargar todas las imagenes de la carpeta de Marcas


**[Celda 2 - Código]**
```python
# SIAMESA
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


**[Celda 3 - Código]**
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


**[Celda 4 - Código]**
```python
model.summary()
```

ARMO LAS LISTAS DE PARES DE IMAGENES


**[Celda 6 - Código]**
```python

```


**[Celda 7 - Código]**
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


**[Celda 8 - Código]**
```python
#convnet = Sequential([
 #   Conv2D(5,3, input_shape=(75,75,3)),
  #  Activation('relu'),
   # MaxPooling2D(),
   # Conv2D(5,3),
    #Activation('relu'),
    #MaxPooling2D(),
    #Conv2D(7,2),
    #Activation('relu'),
    #MaxPooling2D(),
    #Conv2D(7,2),
    #Activation('relu'),
    #Flatten(),
    #Dense(18),
    #Activation('sigmoid')
#])"""
print(x_train.shape)
```

PRUEBAS

Funcion que copia imagenes a otra carpeta


**[Celda 11 - Código]**
```python
def copiar_imagenes(carpeta_origen, carpeta_destino):
    # Carpeta_origen debe tener subcarpetas con imagenes
    # Obtener todas las subcarpetas dentro de la carpeta de origen
    subcarpetas = [f.path for f in os.scandir(carpeta_origen) if f.is_dir()]

    # Recorrer cada subcarpeta y copiar la primera imagen a la carpeta de destino
    for subcarpeta in subcarpetas:
        # Obtener todas las imágenes en la subcarpeta actual
        imagenes = [f.path for f in os.scandir(subcarpeta) if f.is_file() and f.name.endswith(('.jpg', '.jpeg', '.png'))]

        # Copiar la primera imagen si hay alguna
        if imagenes:
            imagen_a_copiar = imagenes[0]
            ruta_destino = os.path.join(carpeta_destino, os.path.basename(imagen_a_copiar))
            shutil.copyfile(imagen_a_copiar, ruta_destino)
```

1


**[Celda 13 - Código]**
```python
# Conecto la carpeta Drive con las Imagenes
# Monta Google Drive en Colab
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

2 Generador de pares similares por Carpeta



**[Celda 15 - Código]**
```python
import os
import random

def generar_pares_similares(ruta_carpeta, extension_valida, num_pares_similares_por_simbolo):

    imagenes = os.listdir(ruta_carpeta)
    nombre_imagenes = [imagen for imagen in imagenes if any(imagen.endswith(ext) for ext in extension_valida)]

    pares_imagenes = []

    for _ in range(num_pares_similares_por_simbolo):
        # Select a random image
        imagen_aleatoria = random.choice(nombre_imagenes)

        # Generate similar pairs for the selected image
        candidatas = [imagen for imagen in nombre_imagenes if imagen != imagen_aleatoria and not any(imagen == p[1] for p in pares_imagenes if p[0] == imagen_aleatoria)]
        for nombre2 in random.sample(candidatas, min(len(candidatas), num_pares_similares_por_simbolo)):
            pares_imagenes.append((imagen_aleatoria, nombre2))

    return pares_imagenes



```


**[Celda 16 - Código]**
```python
# Ejemplo de uso:
ruta_carpeta = '/content/drive/MyDrive/Grupo Investigación /Selección de Prueba/60A03'
extension_valida = ['.JPG']
num_pares_similares_por_simbolo = 5

pares_imagenes = generar_pares_similares(ruta_carpeta, extension_valida, num_pares_similares_por_simbolo)

print(pares_imagenes)
```

Ejemplo de Uso


**[Celda 18 - Código]**
```python
# Ejemplo de uso:
ruta_carpeta = '/content/drive/MyDrive/Grupo Investigación /Selección de Prueba/60A03'
extension_valida = ['.JPG']
num_pares_similares_por_simbolo = 5

pares_imagenes = generar_pares_similares(ruta_carpeta, extension_valida, num_pares_similares_por_simbolo)

for par in pares_imagenes :
  left_input.append(par[0])
  right_input.append(par[1])
  target.append(0)


```

3 Generador de pares similares completo



**[Celda 20 - Código]**
```python
def generar_pares_similares_completo(ruta_principal, extension_valida, num_pares_similares):
    pares_imagenes_total = []
    for root, _, files in os.walk(ruta_principal):
        # Recorremos todas las carpetas y archivos en la ruta principal
        nombre_imagenes = [os.path.join(root, file) for file in files if any(file.endswith(ext) for ext in extension_valida)]
        if nombre_imagenes:
            # Si hay imágenes en la carpeta actual, generamos pares similares
            pares_imagenes = generar_pares_similares(root, extension_valida, num_pares_similares_por_simbolo)
            pares_imagenes_total.extend(pares_imagenes)
    return pares_imagenes_total



```


**[Celda 21 - Código]**
```python
#-------------------- TEST ------------------------------------#
ruta_carpeta_principal = '/content/drive/MyDrive/Grupo Investigación /Selección de Prueba'
extension_valida = ['.JPG']
num_pares_similares_por_simbolo = 5

pares_imagenes_total = generar_pares_similares_completo(ruta_carpeta_principal, extension_valida, num_pares_similares_por_simbolo)

print(pares_imagenes_total)
```

3 Generador de pares distintos




**[Celda 23 - Código]**
```python
from google.colab import drive
import os
import random

def obtener_pares_imagenes(ruta_carpeta_principal, num_pares_similares):

    # Obtén la lista de carpetas en la carpeta principal
    carpetas = [nombre_carpeta for nombre_carpeta in os.listdir(ruta_carpeta_principal) if os.path.isdir(os.path.join(ruta_carpeta_principal, nombre_carpeta))]

    num_pares_similares_por_simbolo = 1

    if (num_pares_similares > len(carpetas)):
      num_pares_similares_por_simbolo = num_pares_similares//len(carpetas) # ver división entera



    # Obtén una lista de nombres de imágenes de todas las carpetas
    extensiones_validas = ['.JPG']
    nombre_imagenes = []
    for carpeta in carpetas:
        ruta_carpeta = os.path.join(ruta_carpeta_principal, carpeta)
        imagenes = [imagen for imagen in os.listdir(ruta_carpeta) if imagen.endswith(tuple(extensiones_validas))]
        nombre_imagenes.extend(imagenes)

    # Obtén pares aleatorios de imágenes
    pares_imagenes = []

    cant_pares = 0

    for nombre1 in nombre_imagenes:
        carpeta1 = random.choice(carpetas)
        for _ in range(num_pares_similares_por_simbolo):
            carpeta2 = carpeta1
            while carpeta2 == carpeta1:
                carpeta2 = random.choice(carpetas)
            nombre2 = random.choice(os.listdir(os.path.join(ruta_carpeta_principal, carpeta2)))
            if (cant_pares < num_pares_similares) :
              pares_imagenes.append((os.path.splitext(nombre1)[0] + os.path.splitext(nombre1)[1], os.path.splitext(nombre2)[0] + os.path.splitext(nombre2)[1]))
              cant_pares += 1


    return pares_imagenes

```

Mod


**[Celda 25 - Código]**
```python
def obtener_pares_imagenes_similares(ruta_carpeta_principal, num_pares_similares):

    # Obtén la lista de carpetas en la carpeta principal
    carpetas = [nombre_carpeta for nombre_carpeta in os.listdir(ruta_carpeta_principal) if os.path.isdir(os.path.join(ruta_carpeta_principal, nombre_carpeta))]


    # Obtén una lista de nombres de imágenes de todas las carpetas
    extensiones_validas = ['.JPG']
    pares_imagenes = []
    cant_pares = 0

    while cant_pares < num_pares_similares :
      for carpeta in carpetas:
          ruta_carpeta = os.path.join(ruta_carpeta_principal, carpeta)
          imagenes = [imagen for imagen in os.listdir(ruta_carpeta) if imagen.endswith(tuple(extensiones_validas))]
          img1 = random.choice(imagenes)
          img2 = random.choice(imagenes)
          while img1 == img2:
                img2 = random.choice(imagenes)
          if (cant_pares < num_pares_similares) :
            pares_imagenes.append((img1,img2))
            cant_pares +=1



    # Obtén pares aleatorios de imágenes

    '''     for nombre1 in nombre_imagenes:
        carpeta1 = random.choice(carpetas)
        for _ in range(num_pares_similares_por_simbolo):
            carpeta2 = carpeta1
            while carpeta2 == carpeta1:
                carpeta2 = random.choice(carpetas)
            nombre2 = random.choice(os.listdir(os.path.join(ruta_carpeta_principal, carpeta2)))
            if (cant_pares < num_pares_similares) :
              pares_imagenes.append((os.path.splitext(nombre1)[0] + os.path.splitext(nombre1)[1], os.path.splitext(nombre2)[0] + os.path.splitext(nombre2)[1]))
              cant_pares += 1 '''


    return pares_imagenes
```


**[Celda 26 - Código]**
```python
# Ejemplo de uso
ruta_carpeta_principal = '/content/drive/MyDrive/Grupo Investigación /Selección de Prueba'
num_pares_similares_por_simbolo = 5
pares_imagenes = obtener_pares_imagenes(ruta_carpeta_principal, num_pares_similares_por_simbolo)
print(pares_imagenes)
```

Muestro los pares complejos


**[Celda 28 - Código]**
```python
import cv2

import os

def mostrar_pares_imagenes_complejo(pares_imagenes, ruta_carpeta):

    for par in pares_imagenes:
        nombre1, nombre2 = par
        ruta_imagen1 = buscar_imagen(ruta_carpeta, nombre1)
        ruta_imagen2 = buscar_imagen(ruta_carpeta, nombre2)

        if ruta_imagen1 is None:
            print(f"No se encontró la imagen {nombre1} en la carpeta {ruta_carpeta}.")
        if ruta_imagen2 is None:
            print(f"No se encontró la imagen {nombre2} en la carpeta {ruta_carpeta}.")

        if ruta_imagen1 is not None and ruta_imagen2 is not None:
            image1 = cv2.imread(ruta_imagen1, cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread(ruta_imagen2, cv2.IMREAD_GRAYSCALE)
            image1 = 255 - image1
            image1 = image1 / 255
            image1 = 1 - image1
            image2 = 255 - image2
            image2 = image2 / 255
            image2 = 1 - image2
            image1 = image1.reshape(28, 28, 1)
            image2 = image2.reshape(28, 28, 1)
            plt.subplot(1, 2, 1)
            plt.imshow(image1.squeeze(), cmap='gray')
            plt.title(nombre1)
            plt.subplot(1, 2, 2)
            plt.imshow(image2.squeeze(), cmap='gray')
            plt.title(nombre2)
            plt.show()


```

4 Buscar Imagen



**[Celda 30 - Código]**
```python
import os

def buscar_imagen(ruta_carpeta_principal, nombre_imagen):
    # Obtén la lista de carpetas en la carpeta principal
    carpetas = [nombre_carpeta for nombre_carpeta in os.listdir(ruta_carpeta_principal) if os.path.isdir(os.path.join(ruta_carpeta_principal, nombre_carpeta))]

    # Busca la imagen en las subcarpetas
    for carpeta in carpetas:
        ruta_carpeta = os.path.join(ruta_carpeta_principal, carpeta)
        imagenes = [imagen for imagen in os.listdir(ruta_carpeta) if imagen.endswith(('.jpg', '.JPG'))]
        if nombre_imagen in imagenes:
            return os.path.join(ruta_carpeta, nombre_imagen)

    return None


```

Muestro los diferentes pares


**[Celda 32 - Código]**
```python
ruta_carpeta_principal = '/content/drive/MyDrive/Grupo Investigación /Selección de Prueba'
ruta_carpeta = '/content/drive/MyDrive/Grupo Investigación /Selección de Prueba/60A03'
num_pares = 5

pares_imagenes_distintos = obtener_pares_imagenes(ruta_carpeta_principal, num_pares)
print(pares_imagenes_distintos)
#mostrar_pares_imagenes_complejo(pares_imagenes_distintos,ruta_carpeta_principal) #Listo


pares_imagenes = generar_pares_similares_completo(ruta_carpeta_principal, extension_valida, num_pares)
mostrar_pares_imagenes_complejo(pares_imagenes,ruta_carpeta_principal)
print(pares_imagenes)

```

FIN DE PRUEBAS


**[Celda 34 - Código]**
```python
# SIAMESA
# Transformar las imagenes de 400x400 a 28x28 (Hecho)
# Se tienen 3875 simbolos distintos
# 5 pares similares por cada simbolo
# 19375 cantidad de pares similares y 19375 cantidad de pares distintos
# Para armar los distintos, por cada carpeta debo obtener un elemento de la carpeta y uno aleatorio
# pero de una carpeta random, formando un total de 5 pares. Y asi con cada carpeta



cant_img_entrenamiento = 45000

# First let's separate the dataset from 1 matrix to a list of matricies
image_list = np.split(x_train[:cant_img_entrenamiento],cant_img_entrenamiento)
label_list = np.split(y_train[:cant_img_entrenamiento],cant_img_entrenamiento)

left_input = []
right_input = []
targets = []

pares_similares = 0

#Number of pairs per image
pairs = 30
#Let's create the new dataset to train on
for i in range(len(label_list)):
    for _ in range(pairs):
        compare_to = i
        while compare_to == i: #Make sure it's not comparing to itself
            compare_to = random.randint(0, cant_img_entrenamiento - 1)
        left_input.append(image_list[i])
        right_input.append(image_list[compare_to])
        if label_list[i] == label_list[compare_to]:# They are the same
            targets.append(0.)
            pares_similares += 1
        else:# Not the same
            targets.append(1.)



print('Cantidad de pares similares: ', pares_similares)
print('Cantidad total de pares: ', len(label_list) * pairs)
left_input = np.squeeze(np.array(left_input))
right_input = np.squeeze(np.array(right_input))
targets = np.squeeze(np.array(targets))

iceimage = x_train[101]
test_left = []
test_right = []
test_targets = []

for i in range(y_train.shape[0]-cant_img_entrenamiento):
    test_left.append(iceimage)
    test_right.append(x_train[i+cant_img_entrenamiento])
    test_targets.append(y_train[i+cant_img_entrenamiento])

test_left = np.squeeze(np.array(test_left))
test_right = np.squeeze(np.array(test_right))
test_targets = np.squeeze(np.array(test_targets))
```


**[Celda 35 - Código]**
```python
#Carga de imagenes de marca aumentadas para entrenamiento

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

directorio = r'C:\Users\User\Documents\UTN\GIBD\Similitud de Marcas\BDNormalizada_R_Aumentada(28x28)'
contenido = os.listdir(directorio)

marcasBDA = []
y_marcasBDA = []
for fichero in contenido:
    if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.jpg'):
        image = directorio + '/' + fichero
        image_gris = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        image_gris = image_gris / 255
        image_gris = 1 - image_gris
        image_gris = image_gris.reshape(28, 28, 1)
        marcasBDA.append(image_gris)
        if (fichero.find('_') >= 0):
            n = fichero[:fichero.find('_')]
        else:
            n = fichero[:fichero.find('.')]
        y_marcasBDA.append(n)
print(y_marcasBDA)
```


**[Celda 36 - Código]**
```python
print(len(y_marcasBDA))
marcasBDA[1].shape
```


**[Celda 37 - Código]**
```python
#Pares de imagenes de marcas aumentadas para entrenamiento
import math

cant_img_entrenamiento = 25000

# First let's separate the dataset from 1 matrix to a list of matricies
image_list = marcasBDA
label_list = y_marcasBDA

left_input = []
right_input = []
targets = []

pares_similares = 0

#Number of pairs per image
pairs = 30
#Let's create the new dataset to train on
for _ in range(cant_img_entrenamiento):
    i = random.randint(0, 91349)
    for _ in range(pairs):
        compare_to = i
        #while compare_to == i: #Make sure it's not comparing to itself
        compare_to = random.randint(0, 91349)
        left_input.append(image_list[i])
        right_input.append(image_list[compare_to])
        if label_list[i] == label_list[compare_to]:# They are the same
            targets.append(0.)
            pares_similares += 1
        else:# Not the same
            targets.append(1.)

print('Cantidad de pares similares: ', pares_similares)
print('Cantidad total de pares: ', cant_img_entrenamiento * pairs)
left_input = np.squeeze(np.array(left_input))
right_input = np.squeeze(np.array(right_input))
targets = np.squeeze(np.array(targets))

iceimage = marcasBDA[101]
test_left = []
test_right = []
test_targets = []

for _ in range(math.trunc(cant_img_entrenamiento * 0.20)):
    r = random.randint(0, 91349)
    h = random.randint(0, 91349)
    test_left.append(marcasBDA[h])
    test_right.append(marcasBDA[r])
    if y_marcasBDA[r] == y_marcasBDA[h]:# They are the same
        test_targets.append(0.)
    else:# Not the same
        test_targets.append(1.)
    #test_targets.append(y_marcasBDA[r])

test_left = np.squeeze(np.array(test_left))
test_right = np.squeeze(np.array(test_right))
test_targets = np.squeeze(np.array(test_targets))
```


**[Celda 38 - Código]**
```python
'1'.find('_')
```


**[Celda 39 - Código]**
```python
import matplotlib.cm as cm


plt.imshow(left_input[30].reshape(28, 28), cmap=cm.Greys)
print(targets[:50])
```


**[Celda 40 - Código]**
```python
plt.imshow(right_input[30].reshape(28, 28), cmap=cm.Greys)
```

Funciones para crear Left_Input, Right_Input y Target.  

# Método para visualizar Listado de Imagenes


**[Celda 43 - Código]**
```python
# Toma los elementos de la lista y los busca en el directorio de drive


def mostrar_imagenes_complejo(lista_imagenes, ruta_carpeta):
    for nombre_imagen in lista_imagenes:
        ruta_imagen = buscar_imagen(ruta_carpeta, nombre_imagen)

        if ruta_imagen is None:
            print(f"No se encontró la imagen {nombre_imagen} en la carpeta {ruta_carpeta}.")
            continue

        image = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        image = 255 - image
        image = image / 255
        image = 1 - image
        image = image.reshape(28, 28, 1)

        plt.subplot(1, 2, 1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(nombre_imagen)
        plt.show()


```

Método para guardar un listado de imagenes


**[Celda 45 - Código]**
```python
def listado_imagenes_complejo(lista_imagenes, ruta_carpeta):
    imagenes = []
    for nombre_imagen in lista_imagenes:
        ruta_imagen = buscar_imagen(ruta_carpeta, nombre_imagen)

        if ruta_imagen is None:
            print(f"No se encontró la imagen {nombre_imagen} en la carpeta {ruta_carpeta}.")
            continue

        image = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        # image = 255 - image
        image = image / 255
        image = 1 - image
        image = image.reshape(28, 28, 1)

        imagenes.append(image)

    return imagenes

```

Caracteristicas


**[Celda 47 - Código]**
```python
# Lista de pares balanceada, es decir, la misma cantidad de pares similares que de pares no similares.


# La lista debe contener por lo menos una instancia de cada imagen, es decir, cada simbolo debe participar por lo menos en un par similar y en un par no similar.
# Los pares estaran organizados en 3 listas.
# La lista Left que contiene los primeros elementos de cada par. (x,y) elemento x
# La lista rigth que tiene el segundo elemento de cada par. (x,y) elemento y
# La lista Target que contiene un 0.0 si el par correspondiente es similar y un 1.0 si el par correspondiente es no similar.

# controlar la cantiadad de imagenes de pares, pares_similares = pares_distintos

# Preprocesar las imagenes. (para ver)

```

left_right_target_creator


**[Celda 49 - Código]**
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


**[Celda 51 - Código]**
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

left_names, right_names y target


**[Celda 53 - Código]**
```python
ruta_carpeta_principal = '/content/drive/MyDrive/Selección de Prueba'
# Toma los primeros elementos de los pares armados y los devuelve en una lista
pares_imagenes_distintos = obtener_pares_imagenes(ruta_carpeta_principal, 400)
pares_imagenes = obtener_pares_imagenes_similares(ruta_carpeta_principal, 400)

left_names, right_names, target = left_right_target_creator(pares_imagenes_distintos,pares_imagenes)
left_name_test, right_name_test, target_test = left_right_target_creator_test(pares_imagenes_distintos,pares_imagenes)
print(len(pares_imagenes_distintos))
#print(pares_imagenes_distintos)
print(len(pares_imagenes))
#print(pares_imagenes)
print(len(target))

```


**[Celda 54 - Código]**
```python

left = listado_imagenes_complejo(left_names, ruta_carpeta_principal)
right = listado_imagenes_complejo(right_names, ruta_carpeta_principal)

left_test = listado_imagenes_complejo(left_name_test, ruta_carpeta_principal)
right_test = listado_imagenes_complejo(right_name_test, ruta_carpeta_principal)
```


**[Celda 55 - Código]**
```python
left_input = np.squeeze(np.array(left))
right_input = np.squeeze(np.array(right))
targets = np.squeeze(np.array(target))

test_left = np.squeeze(np.array(left_test))
test_right = np.squeeze(np.array(right_test))
test_targets = np.squeeze(np.array(target_test))

print(right_input.shape)
```


**[Celda 56 - Código]**
```python
import random
def data_generator(x, y, batch_size=32):

    n_classes = 203

    class_ids = [None] * n_classes
    for n in range(n_classes):
        class_ids[n] = np.where(y == n)[0]

    id_class = 1

    while True:
        X_batch_0 = np.empty((batch_size, ANCHO, ALTO, 1), dtype=np.float32)
        X_batch_1 = np.empty((batch_size, ANCHO, ALTO, 1), dtype=np.float32)
        X_batch_2 = np.empty((batch_size, ANCHO, ALTO, 1), dtype=np.float32)

        for i in range(batch_size):
            idx_anchor = random.randint(0, len(y) - 1)
            while y[idx_anchor] != str(id_class):
                idx_anchor = random.randint(0, len(y) - 1)

            idx_positive = random.randint(0, len(y) - 1)
            while y[idx_positive] != str(id_class):
                idx_positive = random.randint(0, len(y) - 1)

            idx_negative = random.randint(0, len(y) - 1)
            while y[idx_negative] == str(id_class):
                idx_negative = random.randint(0, len(y) - 1)

            X_batch_0[i] = x[idx_anchor].reshape(28,28,1)
            X_batch_1[i] = x[idx_positive].reshape(28,28,1)
            X_batch_2[i] = x[idx_negative].reshape(28,28,1)

            id_class += 1
            if id_class > n_classes:
                id_class = 1

        yield [X_batch_0, X_batch_1, X_batch_2], np.empty((batch_size, VECTOR_SIZE * 3))
```


**[Celda 57 - Código]**
```python
print(len(left_input))
print(len(right_input))
print(len(targets))
```


**[Celda 58 - Código]**
```python
#Prueba

siamese_net.fit([left_input,right_input], targets,
          batch_size=32,
          epochs=10,
          verbose=1,
          validation_data=([test_left,test_right],test_targets))
```


**[Celda 59 - Código]**
```python
# SIAMESA

siamese_net.summary()
siamese_net.fit([left_input,right_input], targets,
          batch_size=16,
          epochs=5,
          verbose=1,
          validation_data=([test_left,test_right],test_targets))

```


**[Celda 60 - Código]**
```python
# Guardar el Modelo
#model.save(r'C:\Users\User\Documents\UTN\GIBD\Modelos\ModeloSiamesa(5)_ExtraccionCaracteristicas_65000.h5')
```


**[Celda 61 - Código]**
```python
# Recrea exactamente el mismo modelo solo desde el archivo
model = keras.models.load_model(r'C:\Users\User\Documents\UTN\GIBD\Modelos\ModeloSiamesa(5)_ExtraccionCaracteristicas_10000.h5')
```


**[Celda 62 - Código]**
```python
model2 = keras.models.load_model(r'C:\Users\User\Documents\UTN\GIBD\Modelos\ModeloMnistSiamesa(30)_SinBalance_45000_1S_0NS.h5')
```


**[Celda 63 - Código]**
```python
imagen = imagenesBD[177]

plt.imshow(imagen.reshape(28, 28), cmap=cm.Greys)
print(imagen.shape)

predictions1 = model.predict(imagen.reshape(1,28,28,1))
predictions2 = model2.predict(imagen.reshape(1,28,28,1))

for i in range(len(predictions1[0])):
    #print(format(predictions1[0][i]*100, ".2f")," ", format(predictions2[0][i]*100, ".2f"))
    print(format(predictions1[0][i]*100-predictions2[0][i]*100, ".2f"))

#print("{:.2f}".format(predictions1[0]*100)," ", "{:.2f}".format(predictions2[0]*100))

#print(predictions2[0]*100)
```


**[Celda 64 - Código]**
```python
p = 31
predictions = model.predict(x_test)
print(predictions[p])
print(np.argmax(predictions[p]))
plt.imshow(x_test[p].reshape(28, 28), cmap=cm.Greys)
print(x_test[p].shape)

```


**[Celda 65 - Código]**
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

image = r'C:\Users\User\Documents\UTN\GIBD\Prueba7.jpg'
img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_cvt)
plt.show()
print(img)
img5 = img / 255.0
img5 = 1 - img5
print(img5)
#img5 = img5.reshape(28,28,1)
print(img5.shape)
```


**[Celda 66 - Código]**
```python
model.summary()
```

##Carga de la base de datos


**[Celda 68 - Código]**
```python
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

directorio = '/content/drive/MyDrive/BD_CaracteresChinos'
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


**[Celda 69 - Código]**
```python

        plt.imshow(imagenesBD[0], cmap='gray')
        plt.show()

```

##Cargar imagenes de consulta


**[Celda 71 - Código]**
```python
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

directorio = '/content/drive/MyDrive/Queries_CaracteresChinos'
contenido = os.listdir(directorio)
print(contenido)
consultas = []
y_consultas = []
for fichero in contenido:
    if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.jpg'):
        image = directorio + '/' + fichero
        image_gris = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        image_gris = image_gris / 255
        image_gris = 1 - image_gris
        image_gris = image_gris.reshape(28,28,1)
        consultas.append(image_gris)
        n = fichero[:fichero.find('.')]
        #n = n[n.find('-')+1:]
        n = n[:n.find('_')]
        y_consultas.append(n)
print(y_consultas)
```

## Consulta por similitud


**[Celda 73 - Código]**
```python
def nnk(actuales,nuevo,k=1):
    p = 0
    while (p<len(actuales) and (nuevo[0]>=actuales[p][0])):
        p += 1
    actuales.insert(p, nuevo)
    while len(actuales)>k:
        actuales.pop(len(actuales)-1)
```


**[Celda 74 - Código]**
```python
def nnk2etiquetas(nnklist, y_BD):
    l = []
    for par in nnklist:
        nuevo = [par[0], y_BD[par[1]]]
        l.append(nuevo)
    return l
```


**[Celda 75 - Código]**
```python
# CONSULTAS POR SIMILITUD UTILIZANDO LA BASE DE CONSULTAS DIBUJADAS - OPTIMIZADO
# import numpy as np


contador_aciertos = 0
cantidad_consultas = len(consultas)
print(type(consultas[0][0][0]))
cantidad_BD = len(imagenesBD)

# tensor_consultas = tf.reshape(consultas, shape=(cantidad_consultas, 28, 28, 1))
tensor_consultas = tf.convert_to_tensor(np.array(consultas))
print(tensor_consultas.shape)
tensor_imagenesBD = tf.convert_to_tensor(np.array(imagenesBD))
print(tensor_imagenesBD.shape)

vectores_elemento = model.predict(tensor_imagenesBD)
vectores_consulta = model.predict(tensor_consultas)

#Inicio del for consulta
i = 0
for cons in vectores_consulta:
    a = np.array(cons)
    # print(np.sum(a))
    min = 2000000
    nnklist = []
    # Inicio del for DB
    j = 0
    for elem in vectores_elemento:
        b = np.array(elem)
        # print(np.sum(b))
        dist = np.sqrt(np.sum(np.square(a-b)))
        # dist = np.sum(np.abs(a-b))
        # print('distancia: ',dist)
        nnk(nnklist,[dist,j],5)
        j += 1

    print("Consulta número: ", i)
    print("   Imagen de consulta: ", y_consultas[i])
    print('   Mas cercanos: ', nnk2etiquetas(nnklist, y_imagenesBD))
    for par in nnklist:
        if y_imagenesBD[par[1]] == y_consultas[i]:
            contador_aciertos += 1
            print('--> ACIERTO')
    print("\n")
    i += 1

print("Porcentaje de aciertos: ", contador_aciertos/cantidad_consultas)
```


**[Celda 76 - Código]**
```python
import numpy as np
from keras.models import Model

#layer_name = 'flatten'
#extractor_consulta = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
#vector_consulta = extractor_consulta.predict(img5.reshape(1,28,28,1))
#a = np.array(vector_consulta[0])
contador_aciertos = 0
cantidad_consultas = len(consultas)
cantidad_BD = len(imagenesBD)

#Inicio del for consulta
for i in range(cantidad_consultas):
    vector_consulta = model.predict(consultas[i].reshape(1,28,28,1))
    a = np.array(vector_consulta[0])
    min = 20000

    # Inicio del for DB
    for j in range(cantidad_BD):
        vector_elemento = model.predict(imagenesBD[j].reshape(1,28,28,1))
        b = np.array(vector_elemento[0])
        #dist = np.sqrt(np.sum(np.square(a-b)))
        dist = np.sum(np.abs(a-b))
        if dist < min:
            min = dist
            vector_cercano = b
            posicion_cercano = j
    print("Distancia: ", min, " al vector en la posicion: ", posicion_cercano, " desde el vector que se encuentra en: ", i)
    print("Imagen de BD: ", y_imagenesBD[posicion_cercano], " Imagen de consulta: ", y_consultas[i], "\n")
    if y_imagenesBD[posicion_cercano] == y_consultas[i]:
        contador_aciertos += 1

print("Porcentaje de aciertos: ", contador_aciertos/cantidad_consultas)
```


**[Celda 77 - Código]**
```python
#Predicciones con la base de datos de numeros (sin transferencias)

import numpy as np
from keras.models import Model

#layer_name = 'flatten'
#extractor_consulta = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
#vector_consulta = extractor_consulta.predict(img5.reshape(1,28,28,1))
#a = np.array(vector_consulta[0])
consultas_digitos = x_test[:100]
imagenesBD_digitos = x_train[:1000]
y_consultas_digitos = y_test[:100]
y_imagenesBD_digitos = y_train[:1000]

contador_aciertos = 0
cantidad_consultas = len(consultas_digitos)
cantidad_BD = len(imagenesBD_digitos)

#Inicio del for consulta
for i in range(cantidad_consultas):
    vector_consulta = model.predict(consultas_digitos[i].reshape(1,28,28,1))
    a = np.array(vector_consulta[0])
    min = 20000

    # Inicio del for DB
    for j in range(cantidad_BD):
        vector_elemento = model.predict(imagenesBD_digitos[j].reshape(1,28,28,1))
        b = np.array(vector_elemento[0])
        dist = np.sqrt(np.sum(np.square(a-b)))
        #dist = np.sum(np.abs(a-b))
        if dist < min:
            min = dist
            vector_cercano = b
            posicion_cercano = j
    #print("Distancia: ", min, " al vector: ", vector_cercano, " en la posicion: ", posicion_cercano, " desde el vector: ", a, " que se encuentra en: ", i)
    #print("Imagen de BD: ", y_imagenesBD_digitos[posicion_cercano], " Imagen de consulta: ", y_consultas_digitos[i], "\n")
    print("Distancia: ", min, " al vector en la posicion: ", posicion_cercano, " desde el vector que se encuentra en: ", i)
    print("Imagen de BD: ", y_imagenesBD_digitos[posicion_cercano], " Imagen de consulta: ", y_consultas_digitos[i], "\n")
    if y_imagenesBD_digitos[posicion_cercano] == y_consultas_digitos[i]:
        contador_aciertos += 1

print("Porcentaje de aciertos: ", contador_aciertos/cantidad_consultas)
```

## Pruebas de aumentación con Marcas


**[Celda 79 - Código]**
```python
from skimage import transform
import tensorflow.compat.v1 as tf1

tf1.disable_v2_behavior()
```


**[Celda 80 - Código]**
```python
#FUNCIÓN "ROTATION" CON MARCAS

def rotar (imagen, angulo = 0):
    return transform.rotate(imagen, angle = angulo)
```


**[Celda 81 - Código]**
```python
#FUNCIÓN "SCALE" CON MARCAS

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
```


**[Celda 82 - Código]**
```python
#FUNCIÓN "CROP" CON MARCAS

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
```


**[Celda 83 - Código]**
```python
#FUNCIÓN "GAUSSIAN NOISE" CON MARCAS

def gauss_ruido (imagen, ruido):
    noise = tf.random.normal(shape = tf.shape(imagen), mean = 0.0, stddev = ruido, dtype = tf.float64)
    tensor_aum = tf.add(imagen, noise)
    return tensor_aum.eval(session = tf.compat.v1.Session())

```


**[Celda 84 - Código]**
```python
#FUNCIÓN "TRANSLATION" CON MARCAS

def trasladar (imagen, desp_fila, desp_colum):
    translation = tf.image.pad_to_bounding_box(imagen.reshape(1,28,28,1), desp_fila, desp_colum, 28 + desp_fila, 28 + desp_colum)
    tensor_aum = tf.image.crop_to_bounding_box(translation, 0, 0, 28, 28)
    t = tensor_aum.eval(session=tf.compat.v1.Session())
    return t.reshape(28,28,1)
```


**[Celda 85 - Código]**
```python
#Pruebas de funciones:

#aum = rotar(imagenesBD[0], -45)
#aum = escalar(imagenesBD[0], 0.7)
#aum = cortar(imagenesBD[0], [23, 23])
aum = gauss_ruido(imagenesBD[0], 0.05)
#aum = trasladar(imagenesBD[0], 2, 2)
```


**[Celda 86 - Código]**
```python
#Imagen de prueba original:

plt.imshow(imagenesBD[0].reshape(28, 28), cmap=cm.Greys)
```


**[Celda 87 - Código]**
```python
#Imagen aplicando alguna de las funciones:

plt.imshow(aum.reshape(28, 28), cmap=cm.Greys)
print(aum.shape)
```


**[Celda 88 - Código]**
```python
directorio = r'C:\Users\User\Documents\UTN\GIBD\Similitud de Marcas\BDNormalizada_R_Aumentada(28x28)'
contenido = os.listdir(directorio)

imagenesBD = []
y_imagenesBD = []
for fichero in contenido:
    if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.jpg'):
        image = directorio + '/' + fichero
        image_gris = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        image_gris = image_gris / 255
        image_gris = 1 - image_gris
        #imagenesBD.append(image_gris)
        n = fichero[:fichero.find('.')]

        for angulo in range(1,8):
            image_rot = rotar(image_gris, angulo * 5)
            nombre = n + '_rot' + str(angulo * 5) + '.jpg'
            cv2.imwrite(directorio + '/' + nombre, ((1 - image_rot)*255).astype(int))

        for angulo in range(1,8):
            image_rot = rotar(image_gris, angulo * -5)
            nombre = n + '_rot' + str(angulo * -5) + '.jpg'
            cv2.imwrite(directorio + '/' + nombre, ((1 - image_rot)*255).astype(int))
        #y_imagenesBD.append(n)
```


**[Celda 89 - Código]**
```python
#Aumentación por escala

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import time

directorio = r'C:\Users\User\Documents\UTN\GIBD\Similitud de Marcas\BDNormalizada_R_Aumentada(28x28)'
contenido = os.listdir(directorio)

imagenesBD = []
y_imagenesBD = []
for fichero in contenido:
    if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.jpg'):
        image = directorio + '/' + fichero
        image_gris = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        image_gris = image_gris / 255
        image_gris = 1 - image_gris
        #imagenesBD.append(image_gris)
        n = fichero[:fichero.find('.')]

        escalas = [60,70,80,90]

        for escala in escalas:
            start = time.time()

            image_esc = escalar(image_gris, escala/100)
            nombre = n + '_esc' + str(escala) + '.jpg'
            cv2.imwrite(directorio + '/' + nombre, ((1 - image_esc)*255).astype(int))

            end = time.time()
            print(nombre, '----->', end-start, 'seg', "\n")
        #y_imagenesBD.append(n)

print('Tiempo total de procesamiento: ', total)
```


**[Celda 90 - Código]**
```python
#Aumentación por cortes

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import time

directorio = r'C:\Users\User\Documents\UTN\GIBD\Similitud de Marcas\BDNormalizada_R_Aumentada(28x28)'
contenido = os.listdir(directorio)

imagenesBD = []
y_imagenesBD = []
for fichero in contenido:
    if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.jpg'):
        image = directorio + '/' + fichero
        image_gris = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        image_gris = image_gris / 255
        image_gris = 1 - image_gris
        #imagenesBD.append(image_gris)
        n = fichero[:fichero.find('.')]

        cortes = [[21, 21], [23, 23]]

        for corte in cortes:
            start = time.time()

            cortes = [[21, 21], [23, 23]]
            image_cor = cortar(image_gris, corte)
            nombre = n + '_cor' + str(corte[0]) + '.jpg'
            cv2.imwrite(directorio + '/' + nombre, ((1 - image_cor)*255).astype(int))

            end = time.time()
            print(nombre, '----->', end-start, 'seg', "\n")
        #y_imagenesBD.append(n)

print('Tiempo total de procesamiento: ', total)
```


**[Celda 91 - Código]**
```python
import time
#Aumentación por ruido

directorio = r'C:\Users\User\Documents\UTN\GIBD\Similitud de Marcas\BDNormalizada_R_Aumentada(28x28)'
contenido = os.listdir(directorio)

imagenesBD = []
y_imagenesBD = []
for fichero in contenido:
    if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.jpg'):
        image = directorio + '/' + fichero
        image_gris = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        image_gris = image_gris / 255
        image_gris = 1 - image_gris
        #imagenesBD.append(image_gris)
        n = fichero[:fichero.find('.')]

        start = time.time()
        ruido = 0.03

        image_rui = gauss_ruido(image_gris, ruido)
        nombre = n + '_rui' + '.jpg'
        cv2.imwrite(directorio + '/' + nombre, ((1 - image_rui)*255).astype(int))

        end = time.time()
        print(nombre, '----->', end-start, 'seg', "\n")
        #y_imagenesBD.append(n)
```


**[Celda 92 - Código]**
```python
#Aumentación por traslación

directorio = r'C:\Users\User\Documents\UTN\GIBD\Similitud de Marcas\ImagenesPrueba'
contenido = os.listdir(directorio)

imagenesBD = []
y_imagenesBD = []
for fichero in contenido:
    if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.jpg'):
        image = directorio + '/' + fichero
        print(image)
        image_gris = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        #print(image_gris)
        image_gris = image_gris / 255
        image_gris = 1 - image_gris
        #imagenesBD.append(image_gris)
        n = fichero[:fichero.find('.')]

        traslados = [[3,3], [2,1], [4,2]]

        for traslado in traslados:

            image_tra = trasladar(image_gris, traslado[0], traslado[1])
            print(image_tra)
            nombre = n + '_tra' + str(traslado[0]) + '-' + str(traslado[1]) + '.jpg'
            cv2.imwrite(directorio + '/' + nombre, ((1 - image_tra)*255).astype(int))

        #y_imagenesBD.append(n)
```


**[Celda 93 - Código]**
```python

```


**[Celda 94 - Código]**
```python
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import time

directorio = r'C:\Users\User\Documents\UTN\GIBD\Similitud de Marcas\BDNormalizada_R_Aumentada(28x28)_COPIA-ROTACION'
contenido = os.listdir(directorio)

imagenesBD = []
y_imagenesBD = []
fichero = contenido[0]
if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.jpg'):
    image = directorio + '/' + fichero
    image_gris = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image_gris = image_gris / 255
    image_gris = 1 - image_gris
    #imagenesBD.append(image_gris)
    n = fichero[:fichero.find('.')]

    #image_rot = rotar(image_gris, angulo * 5)
    #nombre = n + '_rot' + str(angulo * 5) + '.jpg'
    #print((image_rot*255).astype(int))
    #print(image_gris)
    #cv2.imwrite(directorio + '/' + nombre, ((1 - image_rot)*255).astype(int))

    #start = time.time()
    #escala = 60
    #total = 0
    #image_esc = escalar(image_gris, escala/100)
    #nombre = n + '_esc' + str(escala) + '.jpg'
    #cv2.imwrite(directorio + '/' + nombre, ((1 - image_esc)*255).astype(int))
    #end = time.time()
    #total = total + (end-start)
    #print(nombre, '--->', end-start, 'seg', "\n")
    #print(total)

    cortes = [[21, 21], [23, 23]]
    image_cor = cortar(image_gris, cortes[0])
    nombre = n + '_cor' + str(cortes[0][0]) + '.jpg'
    cv2.imwrite(directorio + '/' + nombre, ((1 - image_cor)*255).astype(int))
    #y_imagenesBD.append(n)
```


**[Celda 95 - Código]**
```python

```
