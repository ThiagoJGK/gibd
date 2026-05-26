---
aliases: [EntrenamientoRedSiamesas_LogosClubes-Andres]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2023-08-28
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/CNN Marcas/Notebooks/EntrenamientoRedSiamesas_LogosClubes-Andres.ipynb"
tamanio_bytes: 2843777
---

# Notebook: EntrenamientoRedSiamesas_LogosClubes-Andres.ipynb

Ruta interna: `GIBD/CNN Marcas/Notebooks/EntrenamientoRedSiamesas_LogosClubes-Andres.ipynb`

---

## Imports


**[Celda 2 - Código]**
```python
# SIAMESA

from tensorflow.keras.layers import Input, Conv2D, Lambda, Reshape, Dense, Flatten, MaxPooling2D, Activation, Dropout, concatenate, BatchNormalization, GlobalAveragePooling2D 
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
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
from PIL import Image, ImageEnhance
import os
import cv2
import math
# from google.colab import drive
# drive.mount('/content/drive')

ANCHO = 100
ALTO = 100

```

## Creación del modelo


**[Celda 4 - Código]**
```python
# DEFINICIÓN DEL NUEVO MODELO: JULIO 2023 #

VECTOR_SIZE = 128

model = Sequential()

model.add(Conv2D(32,kernel_size=3,padding='same',activation='relu',input_shape=(ANCHO,ALTO,3)))
model.add(BatchNormalization())

model.add(Conv2D(64,kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Conv2D(32,kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))          

model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,kernel_size=3, padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(256,kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(GlobalAveragePooling2D())
#model.add(Dropout(0.4))

model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))

model.add(Dense(VECTOR_SIZE))

#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Dropout(0.4))
#model.add(tf.keras.layers.Dense(10, activation='softmax'))

#model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
```

## Red Siamesa



**[Celda 6 - Código]**
```python
# We have 2 inputs, 1 for each picture
left_inp = Input((ANCHO, ALTO, 3))
right_inp = Input((ANCHO, ALTO, 3))

# We will use 2 instances of 1 network for this task

# Connect each 'leg' of the network to each input
# Remember, they have the same weights
encoded_l = model(left_inp)
encoded_r = model(right_inp)

# Getting the L1 Distance between the 2 encodings
L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))

# Add the distance function to the network
L1_distance = L1_layer([encoded_l, encoded_r])

print(L1_distance)

prediction = Dense(1,activation='sigmoid')(L1_distance)

print(prediction)

siamese_net = Model(inputs=[left_inp,right_inp],outputs=prediction)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=2.5e-4)

# siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])
siamese_net.compile(loss='mean_squared_error', metrics=['mean_absolute_error'],optimizer=optimizer)

```

## Definición de funciones


**[Celda 8 - Código]**
```python
def Aumentar(img):
    # Aplicar las transformaciones y guardar las imágenes resultantes
    dim = img.size
    r = random.random()
    if r < 0.60:
      factor_brillo = random.random() + 0.2 +random.random()  # Varia el brillo
      enhancer = ImageEnhance.Brightness(img)
      img= enhancer.enhance(factor_brillo)
    r = random.random()
    if r < 0.40:
      factor_contraste = random.uniform(0.25,1.75)  # Varia el contraste
      enhancer = ImageEnhance.Contrast(img)
      img= enhancer.enhance(factor_contraste)
    # r = random.random()
    # if r < 0.40:
    #  dx = int(img.size[0]*random.uniform(0,0.25))  # Traslación en el eje x
    #  dy = int(img.size[1]*random.uniform(0,0.25))  # Traslación en el eje y
    #  img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))
    r = random.random()
    if r < 0.40:
      angulo = random.randint(-20,20)
      img = img.rotate(angulo, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255))   
    return img.resize(dim)
```

## Cargar la Base de Datos y la de Consultas


**[Celda 10 - Código]**
```python
def addAlphaChannel(image):
    # Crear una nueva imagen RGBA (32 bits) con el mismo tamaño que la original
    new_image = Image.new('RGBA', image.size)

    # Copiar los canales RGB de la imagen original a la nueva imagen
    new_image.paste(image, (0, 0))

    # Agregar un canal alfa completamente opaco (255) a la nueva imagen
    alpha_channel = Image.new('L', image.size, 255)
    new_image.putalpha(alpha_channel)
    
    return new_image
```


**[Celda 11 - Código]**
```python
## CARGAR Y NORMALIZAR LAS IMAGENES DE LA BD Y LAS CONSULTAS

def crop_and_normalize_images(image_paths, target_size):
    normalized_images = []
    etiquetas = []

    for path in image_paths:
        image = Image.open(path)
        
        # nombre = etiqueta
        filename = os.path.basename(path)
        nombre=filename[:filename.find(".")]
        etiquetas.append(nombre)
        print(nombre)

        # Verificar si la imagen tiene un canal alfa
        has_alpha = 'A' in image.mode
        
        # Convertir la imagen en un array numpy
        image_array = np.array(image)

        # print(image_array) 
        
        if has_alpha:
            print('TIENE ALPHA...')
            # Encontrar las filas y columnas donde todos los valores alfa son igual a 0
            # non_empty_rows_alpha = np.where(image_array[:, :, 3] < 20)[0]
            # print(non_empty_rows_alpha)
            # non_empty_columns_alpha = np.where(image_array[:, :, 3] <20)[1]
            # Remover Alpha Channel
            for col in range(image_array.shape[0]):
                for row in range(image_array.shape[1]):
                    for chan in range(3):
                        if image_array[col, row, 3]<100:  # si es transparente
                            image_array[col, row, chan] = 255   # lo convierto a blanco, porque luego elimino el canal alpha
            # elimino el canal alpha  
            image_array = image_array[:,:,:3]
            image = Image.fromarray(image_array)
            
        # print(image_array)    
        non_empty_columns = np.where(image_array.min(axis=0) < 250)[0]
        non_empty_rows = np.where(image_array.min(axis=1) < 250)[0]
                
        minRow = np.min(non_empty_rows)
        maxRow = np.max(non_empty_rows)
        minCol = np.min(non_empty_columns)
        maxCol = np.max(non_empty_columns)
        
        print(minCol, maxCol, minRow, maxRow)
        # maxColCrop = maxCol if maxCol<maxColAlpha else maxColAlpha
        # maxRowCrop = maxRow if maxRow<maxRowAlpha else maxRowAlpha
        # Determinar las coordenadas del cuadro de recorte
        crop_box = (
            minCol, 
            minRow, 
            maxCol, 
            maxRow
        )
        
        # Recortar y redimensionar la imagen
        cropped_resized_image = image.crop(crop_box).resize(target_size, Image.Resampling.LANCZOS)
        
        # Convertir la imagen recortada y redimensionada en un array numpy y normalizar
        # normalized_image = (np.array(cropped_resized_image) - np.mean(cropped_resized_image)) / np.std(cropped_resized_image)
        normalized_image = np.array(cropped_resized_image)
        
        normalized_images.append(normalized_image)
    
    print(normalized_images[0].shape)

    return np.array(normalized_images), np.array(etiquetas)


target_size = (100, 100)  # El tamaño objetivo después de recortar y redimensionar

## CARGAR LAS RUTAS DE CADA IMAGEN DE LA BASE DE DATOS
orig_dir = r'G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD'

image_paths = []

for filename in os.listdir(orig_dir):
    image_paths.append(os.path.join(orig_dir, filename))

## NORMALIZAR LAS IMAGENES
imagenesBD, y_imagenesBD = crop_and_normalize_images(image_paths, target_size)



## CARGAR LAS RUTAS DE CADA IMAGEN DE CONSULTA
orig_dir = r'G:\GIBD\Investigacion\Similitud Logos Clubes\consultas'

image_paths = []

for filename in os.listdir(orig_dir):
    image_paths.append(os.path.join(orig_dir, filename))


## NORMALIZAR LAS IMAGENES

consultas, y_consultas = crop_and_normalize_images(image_paths, target_size)

```


**[Celda 12 - Código]**
```python
## MOSTRAR LAS IMAGENES DE LA BASE DE DATOS Y DE LAS CONSULTAS
# base de datos
i = 0
for img in imagenesBD:
    print(y_imagenesBD[i])
    i += 1
    plt.imshow(img)
    plt.show()
    break
    
# consultas
i = 0
for img in consultas:
    print(y_consultas[i])
    i += 1
    plt.imshow(img)
    plt.show()
    break
```


*Salida:*
```text
ac-milan
<Figure size 432x288 with 1 Axes>ac-milan
<Figure size 432x288 with 1 Axes>
```

## Creación de los pares de imágenes para entrenamiento


**[Celda 14 - Código]**
```python
#Parte 13/06

left_input = []
right_input = []
targets = []

# Number of pairs per image
pairs = 320    #El numero de pairs siempre debe ser multiplo de 4 para no perder pares
pares_similares = int(pairs/2)
mitad_pares_similares = int(pares_similares/2)
# Iterar sobre las imágenes originales

listaImagenes=imagenesBD

#for filename in os.listdir(orig_dir):
#    # Cargar la imagen original
#    img_orig = Image.open(os.path.join(orig_dir, filename))
#    listaImagenes.append(img_orig)

for i in range(len(listaImagenes)):
      img = Image.fromarray(listaImagenes[i].astype(np.uint8))
      print(i)
      # Generación de los pares similares, la mitad utilizando la imagen original vs una aumentada, la otra mitad entre dos aumentadas
      for j in range(mitad_pares_similares):
        aum=Aumentar(img)
        left_input.append((np.array(img)/255).astype(np.float32))
        right_input.append((np.array(aum)/255).astype(np.float32))
        targets.append(0.)
      for j in range(mitad_pares_similares):
        aum=Aumentar(img)
        aum2=Aumentar(img)
        left_input.append((np.array(aum2)/255).astype(np.float32))
        right_input.append((np.array(aum)/255).astype(np.float32))
        targets.append(0.)
      # Generación de los pares distintos, la mitad utilizando la imagen original vs una aumentada, la otra mitad entre dos aumentadas
      for j in range(mitad_pares_similares):
        p = random.randint(0,len(listaImagenes)-1)
        while p==i:
          p = random.randint(0,len(listaImagenes)-1)
        img_dif=Image.fromarray(listaImagenes[p].astype(np.uint8))
        if random.random() > 0.5:
            aum = Aumentar(img_dif)
        else:
            aum = img_dif
        left_input.append((np.array(img)/255).astype(np.float32))
        right_input.append((np.array(aum)/255).astype(np.float32))
        targets.append(1.)
      for j in range(mitad_pares_similares):
        p = random.randint(0,len(listaImagenes)-1)
        while p==i:
          p = random.randint(0,len(listaImagenes)-1)
        img_dif=Image.fromarray(listaImagenes[p].astype(np.uint8))
        aum=Aumentar(img)
        aum2=Aumentar(img_dif)
        left_input.append((np.array(aum2)/255).astype(np.float32))
        right_input.append((np.array(aum)/255).astype(np.float32))
        targets.append(1.)

print('Cantidad de pares similares:', pares_similares*len(listaImagenes))
print('Cantidad total de pares:', len(left_input))

#left_input = np.squeeze(np.array(left_input))
#right_input = np.squeeze(np.array(right_input))
#targets = np.squeeze(np.array(targets))
left_input = np.array(left_input)
right_input = np.array(right_input)
targets = np.array(targets)

```


**[Celda 15 - Código]**
```python
## MOSTRAR LAS IMAGENES DE LOS PARES
# base de datos
desde_par = 310
cantidad = 20
for i in range(desde_par,desde_par+cantidad):
    print(targets[i])
    plt.imshow(left_input[i])
    plt.show()
    plt.imshow(right_input[i])
    plt.show()
```


*Salida:*
```text
1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>
```

## Creación de los pares de imágenes para validación


**[Celda 17 - Código]**
```python
#Parte 13/06
test_left = []
test_right = []
test_targets = []

# Number of pairs per image
test_pairs = int(pairs * 0.20)    #El numero de pairs siempre debe ser multiplo de 4 para no perder pares
pares_similares = int(test_pairs/2)
mitad_pares_similares = int(pares_similares/2)

for i in range(len(listaImagenes)):
      img = Image.fromarray(listaImagenes[i].astype(np.uint8))
      print(i)
      # Generación de los pares similares, la mitad utilizando la imagen original vs una aumentada, la otra mitad entre dos aumentadas
      for j in range(mitad_pares_similares):
        aum=Aumentar(img)
        test_left.append((np.array(img)/255).astype(np.float32))
        test_right.append((np.array(aum)/255).astype(np.float32))
        test_targets.append(0.)
      for j in range(mitad_pares_similares):
        aum=Aumentar(img)
        aum2=Aumentar(img)
        test_left.append((np.array(aum2)/255).astype(np.float32))
        test_right.append((np.array(aum)/255).astype(np.float32))
        test_targets.append(0.)
      # Generación de los pares distintos, la mitad utilizando la imagen original vs una aumentada, la otra mitad entre dos aumentadas
      for j in range(mitad_pares_similares):
        p = random.randint(0,len(listaImagenes)-1)
        while p==i:
          p = random.randint(0,len(listaImagenes)-1)
        img_dif=Image.fromarray(listaImagenes[p].astype(np.uint8))
        if random.random() > 0.5:
            aum = Aumentar(img_dif)
        else:
            aum = img_dif
        test_left.append((np.array(img)/255).astype(np.float32))
        test_right.append((np.array(aum)/255).astype(np.float32))
        test_targets.append(1.)
      for j in range(mitad_pares_similares):
        p = random.randint(0,len(listaImagenes)-1)
        while p==i:
          p = random.randint(0,len(listaImagenes)-1)
        img_dif=Image.fromarray(listaImagenes[p].astype(np.uint8))
        aum=Aumentar(img)
        aum2=Aumentar(img_dif)
        test_left.append((np.array(aum2)/255).astype(np.float32))
        test_right.append((np.array(aum)/255).astype(np.float32))
        test_targets.append(1.)

print('Cantidad de pares similares:', pares_similares*len(listaImagenes))
print('Cantidad total de pares:', len(test_left))

test_left = np.array(test_left)
test_right = np.array(test_right)
test_targets = np.array(test_targets)

```


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


**[Celda 20 - Código]**
```python
#### CONSULTAS POR SIMILITUD UTILIZANDO LA BASE DE CONSULTAS DE ESCUDOS - OPTIMIZADO
# Base de consultas y de imagenes
contador_aciertos = 0
contador_aciertosNN1 = 0
contador_aciertosNN3 = 0
cantidad_consultas = len(consultas)  #Consultas es la lista de imagenes de consultas
print(type(consultas[0][0][0]))
cantidad_BD = len(imagenesBD)  #imagenesBD es la lista de imagenes de entrenamiento  y val(descargar el .zip)

#Utilizamos la funcion convertiratensor para transformar en tensor la carpeta y luego utilizar el modelo



#Obtenemos las características
vectores_elemento = model.predict(imagenesBD)
vectores_consulta = model.predict(consultas)

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
    ps = 1
    for par in nnklist:
        if y_imagenesBD[par[1]] == y_consultas[i]:
            contador_aciertos += 1
            print('--> ACIERTO')
            if ps == 1:
                contador_aciertosNN1 += 1
                print(f'    Primer Lugar: {contador_aciertosNN1}')
            if ps <= 3:
                contador_aciertosNN3 += 1
                print(f'    Entre los tres primeros: {contador_aciertosNN3}')
        ps += 1
    print("\n")
    i += 1

print("Porcentaje de aciertos NN1: ", contador_aciertosNN1/cantidad_consultas)
print("Porcentaje de aciertos NN3: ", contador_aciertosNN3/cantidad_consultas)
print("Porcentaje de aciertos NN5: ", contador_aciertos/cantidad_consultas)
print('  Cantidad consultas: ', cantidad_consultas)
```


**[Celda 21 - Código]**
```python
# SIAMESA
print(left_input.shape)
print(test_left.shape)
# print(targets[0])
# print(test_targets[0])
print(len(targets))
print(len(test_targets))


siamese_net.fit([left_input,right_input], targets,
          batch_size=16,
          epochs=5,
          verbose=1,
          validation_data=([test_left,test_right],test_targets))

```


*Salida:*
```text
(32000, 100, 100, 3)
(6400, 100, 100, 3)
32000
6400
Epoch 1/5
   2/2000 [..............................] - ETA: 5:04 - loss: 0.1186 - mean_absolute_error: 0.2682WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0110s vs `on_train_batch_end` time: 0.1510s). Check your callbacks.
2000/2000 [==============================] - ETA: 0s - loss: 0.1581 - mean_absolute_error: 0.3264WARNING:tensorflow:Callbacks method `on_test_batch_end` is slow compared to the batch time (batch time: 0.0050s vs `on_test_batch_end` time: 0.0370s). Check your callbacks.
2000/2000 [==============================] - 346s 173ms/step - loss: 0.1581 - mean_absolute_error: 0.3264 - val_loss: 0.1509 - val_mean_absolute_error: 0.3181
Epoch 2/5
2000/2000 [==============================] - 346s 173ms/step - loss: 0.1571 - mean_absolute_error: 0.3246 - val_loss: 0.1503 - val_mean_absolute_error: 0.3146
Epoch 3/5
2000/2000 [==============================] - 345s 173ms/step - loss: 0.1564 - mean_absolute_error: 0.3231 - val_loss: 0.1500 - val_mean_absolute_error: 0.3157
Epoch 4/5
2000/2000 [==============================] - 339s 169ms/step - loss: 0.1557 - mean_absolute_error: 0.3216 - val_loss: 0.1499 - val_mean_absolute_error: 0.3169
Epoch 5/5
2000/2000 [==============================] - 347s 173ms/step - loss: 0.1553 - mean_absolute_error: 0.3212 - val_loss: 0.1496 - val_mean_absolute_error: 0.3147
<tensorflow.python.keras.callbacks.History at 0x6beec1c8>
```


**[Celda 22 - Código]**
```python
print(left_input[0].shape)
print(right_input[0].shape)
# print(right_input[0])
siamese_net.predict([left_input[0].reshape(1,100,100,3), right_input[0].reshape(1,100,100,3)])
```


*Salida:*
```text
(100, 100, 3)
(100, 100, 3)
array([[0.5124654]], dtype=float32)
```


**[Celda 23 - Código]**
```python
# Guardar el Modelo
model.save(r'G:\GIBD\Investigacion\Similitud Logos Clubes\ModeloSiamesaClubes_ExtraccionCaracteristicas_OLDwithNEWPREPROCESING_60ep81hr.h5')
```


**[Celda 24 - Código]**
```python
# Recrea exactamente el mismo modelo solo desde el archivo
model = keras.models.load_model(r'G:\GIBD\Investigacion\Similitud Logos Clubes\ModeloSiamesaClubes_ExtraccionCaracteristicas_NEW_SLOW_15ep_50p.h5')
```


*Salida:*
```text
WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.
```


**[Celda 25 - Código]**
```python

```
