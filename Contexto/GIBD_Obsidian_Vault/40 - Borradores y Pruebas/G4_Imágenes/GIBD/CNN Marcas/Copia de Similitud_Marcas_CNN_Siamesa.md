---
aliases: [Copia de Similitud_Marcas_CNN_Siamesa]
tags:
  - grupo/g4_imágenes
  - estado/borrador
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2022-07-05
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/CNN Marcas/Función de Distancia/Copia de Similitud_Marcas_CNN_Siamesa.ipynb"
tamanio_bytes: 430038
---

# Notebook: Copia de Similitud_Marcas_CNN_Siamesa.ipynb

Ruta interna: `GIBD/CNN Marcas/Función de Distancia/Copia de Similitud_Marcas_CNN_Siamesa.ipynb`

---

# Busqueda por Similitud de Marcas

## 1 Cargar todas las imagenes de la carpeta de Marcas


**[Celda 2 - Código]**
```python
# SIAMESA

from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D,Activation, Dropout
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


**[Celda 3 - Código]**
```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```


**[Celda 4 - Código]**
```python
print(x_train.shape)
print(x_test.shape)
print(y_train)
```


*Salida:*
```text
(60000, 28, 28)
(10000, 28, 28)
[5 0 4 ... 5 6 8]
```


**[Celda 5 - Código]**
```python
# SIAMESA

# We have 2 inputs, 1 for each picture
left_input = Input((28,28,1))
right_input = Input((28,28,1))

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

optimizer = Adam(0.001, decay=2.5e-4)

#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])
```


**[Celda 6 - Código]**
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


*Salida:*
```text
(60000, 28, 28)
```


**[Celda 7 - Código]**
```python
# SIAMESA

cant_img_entrenamiento = 15000

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


*Salida:*
```text
Cantidad de pares similares:  45533
Cantidad total de pares:  450000
```


**[Celda 8 - Código]**
```python

```


**[Celda 9 - Código]**
```python
import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.imshow(left_input[30].reshape(28, 28), cmap=cm.Greys)
print(targets[:50])
```


*Salida:*
```text
[1. 0. 1. 0. 1. 1. 1. 1. 0. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1.]
<Figure size 432x288 with 1 Axes>
```


**[Celda 10 - Código]**
```python
plt.imshow(right_input[30].reshape(28, 28), cmap=cm.Greys)
```


*Salida:*
```text
<matplotlib.image.AxesImage at 0x7f94dd792050><Figure size 432x288 with 1 Axes>
```


**[Celda 11 - Código]**
```python
# SIAMESA

siamese_net.summary()
siamese_net.fit([left_input,right_input], targets,
          batch_size=16,
          epochs=30,
          verbose=1,
          validation_data=([test_left,test_right],test_targets))

```


*Salida:*
```text
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 28, 28, 1)]  0           []                               
                                                                                                  
 input_2 (InputLayer)           [(None, 28, 28, 1)]  0           []                               
                                                                                                  
 sequential (Sequential)        (None, 128)          325440      ['input_1[0][0]',                
                                                                  'input_2[0][0]']                
                                                                                                  
 lambda (Lambda)                (None, 128)          0           ['sequential[0][0]',             
                                                                  'sequential[1][0]']             
                                                                                                  
 dense_1 (Dense)                (None, 1)            129         ['lambda[0][0]']                 
                                                                                                  
==================================================================================================
Total params: 325,569
Trainable params: 324,993
Non-trainable params: 576
__________________________________________________________________________________________________
Epoch 1/30
28125/28125 [==============================] - 4315s 153ms/step - loss: 0.0290 - accuracy: 0.9908 - val_loss: -32.6869 - val_accuracy: 0.1120
Epoch 2/30
 6960/28125 [======>.......................] - ETA: 54:55 - loss: 0.0063 - accuracy: 0.9982
```


**[Celda 12 - Código]**
```python
# Guardar el Modelo
# model.save(r'C:\Users\User\Documents\UTN\GIBD\Modelos\ModeloMnistSiamesa(30)_SinBalance_45000_1S_0NS.h5')
```


**[Celda 13 - Código]**
```python
# Recrea exactamente el mismo modelo solo desde el archivo
import keras
model = keras.models.load_model(r'/media/ModeloMnistSiamesa(30)_SinBalance_45000_0S_1NS.h5')
```


*Salida:*
```text
WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.
```


**[Celda 14 - Código]**
```python
# PRUEBA DE EJECUCION DE LA RED SIAMESA COMO FUNCION DE DISTANCIA 

# Getting the L1 Distance between the 2 encodings
L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))

# Add the distance function to the network
L1_distance = L1_layer([model, model])

prediction = Dense(1,activation='sigmoid')(L1_distance)
siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

optimizer = Adam(0.001, decay=2.5e-4)

#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])
```


**[Celda 15 - Código]**
```python
p = 31
predictions = model.predict(x_test)
print(predictions[p])
print(np.argmax(predictions[p]))
plt.imshow(x_test[p].reshape(28, 28), cmap=cm.Greys)
print(x_test[p].shape)

```


*Salida:*
```text
[2.0820499e-07 9.9818015e-01 1.2690144e-05 1.4974022e-05 6.6369842e-04
 3.0099538e-06 6.2876775e-09 3.0767536e-05 9.9129055e-04 1.0337210e-04]
1
(28, 28)
<Figure size 432x288 with 1 Axes>
```


**[Celda 16 - Código]**
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


*Salida:*
```text
<Figure size 432x288 with 1 Axes>[[253 255 255 253 255 255 251 255 255 253 255 255 255 255 255 252 254 255
  255 253 255 255 254 255 255 255 255 255]
 [255 253 253 255 253 255 255 253 255 255 255 255 254 255 253 255 255 254
  254 255 255 255 252 253 255 255 255 255]
 [253 253 255 255 247 247 230 229 211 210 209 211 209 210 210 229 249 253
  255 255 254 253 255 255 255 255 255 255]
 [254 255 253 217 187 189 155 148 126 122 114 114 113 118 123 150 198 226
  255 250 254 255 255 249 255 255 255 255]
 [254 254 245 122  36  35  29  28  21  23  21  26  24  22  20  29  50 139
  249 255 255 255 255 255 255 255 255 255]
 [254 255 245  99  32  32  37  32  35  36  32  34  33  31  31  17  17  98
  244 251 252 255 254 253 255 255 255 255]
 [255 255 241 142 112 111 115 121 116 118 117 118 119 117 116  73  11  53
  227 249 254 255 255 255 255 255 255 255]
 [255 253 255 250 251 255 254 254 254 255 255 254 255 253 253 193  48  40
  214 244 253 255 255 254 255 255 255 255]
 [255 255 252 254 255 253 255 255 254 255 255 253 255 254 255 181  28  52
  220 245 255 255 255 255 255 253 254 255]
 [255 255 253 255 255 254 255 254 254 254 255 255 255 253 246 161   5  57
  241 249 255 254 254 253 254 255 255 254]
 [254 255 254 255 255 255 255 254 255 254 253 253 255 255 246 158   0  61
  248 250 255 254 254 255 254 252 255 255]
 [253 255 255 255 255 254 255 254 252 253 255 255 253 251 213 111   1  65
  239 249 246 250 246 246 248 249 252 255]
 [255 255 255 255 254 254 254 255 255 255 255 253 254 255 201  99   9  90
  245 250 210 208 204 206 209 247 250 254]
 [255 254 255 254 254 255 254 255 252 255 250 245 246 240 167  73   4  65
  172 180 141 135 138 139 158 239 250 252]
 [255 253 255 254 254 253 245 242 227 224 201 178 167 151  79  12   5   6
   31  32  44  60  88  91 159 237 250 254]
 [255 253 255 255 254 248 232 223 175 167 133 100  90  85  39   9   1   9
   28  35  86 113 158 158 204 246 251 255]
 [255 254 254 255 215 112  34  26  14  12   8   0   7  24  12   0  55 189
  240 240 246 248 251 254 255 255 255 255]
 [255 255 254 255 209  95  14  16  95 103 148 150 154 150  71  15  75 224
  254 254 255 255 255 255 255 255 255 255]
 [253 255 255 255 213 105  27  30 164 176 239 248 249 235 112  13  80 244
  255 255 255 255 254 254 255 255 255 255]
 [255 255 253 255 244 199 168 178 244 243 255 253 251 248 136  30  78 244
  252 255 253 255 254 255 255 255 255 255]
 [255 255 255 255 252 242 234 234 253 255 251 254 254 251 140  31  81 239
  255 255 254 255 255 255 255 255 255 255]
 [255 253 255 255 252 253 255 254 255 255 255 255 255 251 137  31  82 235
  255 255 253 255 254 254 255 255 255 255]
 [255 255 255 255 254 254 255 253 254 253 255 255 254 253 157  56  82 238
  255 255 254 255 255 255 255 255 255 255]
 [254 255 255 253 255 255 255 255 255 254 255 255 254 250 176  70  79 243
  254 255 254 255 254 255 255 255 255 255]
 [255 255 255 255 255 255 255 255 255 255 254 255 254 253 205  99  81 242
  254 255 254 251 255 255 255 255 255 255]
 [255 255 255 255 255 255 255 255 252 255 255 254 255 255 207 110  90 238
  255 252 255 255 251 255 255 255 255 255]
 [255 255 255 255 255 255 255 255 255 253 254 255 252 250 250 236 231 254
  253 255 251 255 255 254 255 255 255 255]
 [255 255 255 255 255 255 255 255 255 252 252 255 255 255 253 254 253 251
  255 255 253 253 255 255 255 255 255 255]]
[[0.00784314 0.         0.         0.00784314 0.         0.
  0.01568627 0.         0.         0.00784314 0.         0.
  0.         0.         0.         0.01176471 0.00392157 0.
  0.         0.00784314 0.         0.         0.00392157 0.
  0.         0.         0.         0.        ]
 [0.         0.00784314 0.00784314 0.         0.00784314 0.
  0.         0.00784314 0.         0.         0.         0.
  0.00392157 0.         0.00784314 0.         0.         0.00392157
  0.00392157 0.         0.         0.         0.01176471 0.00784314
  0.         0.         0.         0.        ]
 [0.00784314 0.00784314 0.         0.         0.03137255 0.03137255
  0.09803922 0.10196078 0.17254902 0.17647059 0.18039216 0.17254902
  0.18039216 0.17647059 0.17647059 0.10196078 0.02352941 0.00784314
  0.         0.         0.00392157 0.00784314 0.         0.
  0.         0.         0.         0.        ]
 [0.00392157 0.         0.00784314 0.14901961 0.26666667 0.25882353
  0.39215686 0.41960784 0.50588235 0.52156863 0.55294118 0.55294118
  0.55686275 0.5372549  0.51764706 0.41176471 0.22352941 0.11372549
  0.         0.01960784 0.00392157 0.         0.         0.02352941
  0.         0.         0.         0.        ]
 [0.00392157 0.00392157 0.03921569 0.52156863 0.85882353 0.8627451
  0.88627451 0.89019608 0.91764706 0.90980392 0.91764706 0.89803922
  0.90588235 0.91372549 0.92156863 0.88627451 0.80392157 0.45490196
  0.02352941 0.         0.         0.         0.         0.
  0.         0.         0.         0.        ]
 [0.00392157 0.         0.03921569 0.61176471 0.8745098  0.8745098
  0.85490196 0.8745098  0.8627451  0.85882353 0.8745098  0.86666667
  0.87058824 0.87843137 0.87843137 0.93333333 0.93333333 0.61568627
  0.04313725 0.01568627 0.01176471 0.         0.00392157 0.00784314
  0.         0.         0.         0.        ]
 [0.         0.         0.05490196 0.44313725 0.56078431 0.56470588
  0.54901961 0.5254902  0.54509804 0.5372549  0.54117647 0.5372549
  0.53333333 0.54117647 0.54509804 0.71372549 0.95686275 0.79215686
  0.10980392 0.02352941 0.00392157 0.         0.         0.
  0.         0.         0.         0.        ]
 [0.         0.00784314 0.         0.01960784 0.01568627 0.
  0.00392157 0.00392157 0.00392157 0.         0.         0.00392157
  0.         0.00784314 0.00784314 0.24313725 0.81176471 0.84313725
  0.16078431 0.04313725 0.00784314 0.         0.         0.00392157
  0.         0.         0.         0.        ]
 [0.         0.         0.01176471 0.00392157 0.         0.00784314
  0.         0.         0.00392157 0.         0.         0.00784314
  0.         0.00392157 0.         0.29019608 0.89019608 0.79607843
  0.1372549  0.03921569 0.         0.         0.         0.
  0.         0.00784314 0.00392157 0.        ]
 [0.         0.         0.00784314 0.         0.         0.00392157
  0.         0.00392157 0.00392157 0.00392157 0.         0.
  0.         0.00784314 0.03529412 0.36862745 0.98039216 0.77647059
  0.05490196 0.02352941 0.         0.00392157 0.00392157 0.00784314
  0.00392157 0.         0.         0.00392157]
 [0.00392157 0.         0.00392157 0.         0.         0.
  0.         0.00392157 0.         0.00392157 0.00784314 0.00784314
  0.         0.         0.03529412 0.38039216 1.         0.76078431
  0.02745098 0.01960784 0.         0.00392157 0.00392157 0.
  0.00392157 0.01176471 0.         0.        ]
 [0.00784314 0.         0.         0.         0.         0.00392157
  0.         0.00392157 0.01176471 0.00784314 0.         0.
  0.00784314 0.01568627 0.16470588 0.56470588 0.99607843 0.74509804
  0.0627451  0.02352941 0.03529412 0.01960784 0.03529412 0.03529412
  0.02745098 0.02352941 0.01176471 0.        ]
 [0.         0.         0.         0.         0.00392157 0.00392157
  0.00392157 0.         0.         0.         0.         0.00784314
  0.00392157 0.         0.21176471 0.61176471 0.96470588 0.64705882
  0.03921569 0.01960784 0.17647059 0.18431373 0.2        0.19215686
  0.18039216 0.03137255 0.01960784 0.00392157]
 [0.         0.00392157 0.         0.00392157 0.00392157 0.
  0.00392157 0.         0.01176471 0.         0.01960784 0.03921569
  0.03529412 0.05882353 0.34509804 0.71372549 0.98431373 0.74509804
  0.3254902  0.29411765 0.44705882 0.47058824 0.45882353 0.45490196
  0.38039216 0.0627451  0.01960784 0.01176471]
 [0.         0.00784314 0.         0.00392157 0.00392157 0.00784314
  0.03921569 0.05098039 0.10980392 0.12156863 0.21176471 0.30196078
  0.34509804 0.40784314 0.69019608 0.95294118 0.98039216 0.97647059
  0.87843137 0.8745098  0.82745098 0.76470588 0.65490196 0.64313725
  0.37647059 0.07058824 0.01960784 0.00392157]
 [0.         0.00784314 0.         0.         0.00392157 0.02745098
  0.09019608 0.1254902  0.31372549 0.34509804 0.47843137 0.60784314
  0.64705882 0.66666667 0.84705882 0.96470588 0.99607843 0.96470588
  0.89019608 0.8627451  0.6627451  0.55686275 0.38039216 0.38039216
  0.2        0.03529412 0.01568627 0.        ]
 [0.         0.00392157 0.00392157 0.         0.15686275 0.56078431
  0.86666667 0.89803922 0.94509804 0.95294118 0.96862745 1.
  0.97254902 0.90588235 0.95294118 1.         0.78431373 0.25882353
  0.05882353 0.05882353 0.03529412 0.02745098 0.01568627 0.00392157
  0.         0.         0.         0.        ]
 [0.         0.         0.00392157 0.         0.18039216 0.62745098
  0.94509804 0.9372549  0.62745098 0.59607843 0.41960784 0.41176471
  0.39607843 0.41176471 0.72156863 0.94117647 0.70588235 0.12156863
  0.00392157 0.00392157 0.         0.         0.         0.
  0.         0.         0.         0.        ]
 [0.00784314 0.         0.         0.         0.16470588 0.58823529
  0.89411765 0.88235294 0.35686275 0.30980392 0.0627451  0.02745098
  0.02352941 0.07843137 0.56078431 0.94901961 0.68627451 0.04313725
  0.         0.         0.         0.         0.00392157 0.00392157
  0.         0.         0.         0.        ]
 [0.         0.         0.00784314 0.         0.04313725 0.21960784
  0.34117647 0.30196078 0.04313725 0.04705882 0.         0.00784314
  0.01568627 0.02745098 0.46666667 0.88235294 0.69411765 0.04313725
  0.01176471 0.         0.00784314 0.         0.00392157 0.
  0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.01176471 0.05098039
  0.08235294 0.08235294 0.00784314 0.         0.01568627 0.00392157
  0.00392157 0.01568627 0.45098039 0.87843137 0.68235294 0.0627451
  0.         0.         0.00392157 0.         0.         0.
  0.         0.         0.         0.        ]
 [0.         0.00784314 0.         0.         0.01176471 0.00784314
  0.         0.00392157 0.         0.         0.         0.
  0.         0.01568627 0.4627451  0.87843137 0.67843137 0.07843137
  0.         0.         0.00784314 0.         0.00392157 0.00392157
  0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.00392157 0.00392157
  0.         0.00784314 0.00392157 0.00784314 0.         0.
  0.00392157 0.00784314 0.38431373 0.78039216 0.67843137 0.06666667
  0.         0.         0.00392157 0.         0.         0.
  0.         0.         0.         0.        ]
 [0.00392157 0.         0.         0.00784314 0.         0.
  0.         0.         0.         0.00392157 0.         0.
  0.00392157 0.01960784 0.30980392 0.7254902  0.69019608 0.04705882
  0.00392157 0.         0.00392157 0.         0.00392157 0.
  0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.00392157 0.
  0.00392157 0.00784314 0.19607843 0.61176471 0.68235294 0.05098039
  0.00392157 0.         0.00392157 0.01568627 0.         0.
  0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.         0.01176471 0.         0.         0.00392157
  0.         0.         0.18823529 0.56862745 0.64705882 0.06666667
  0.         0.01176471 0.         0.         0.01568627 0.
  0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.00784314 0.00392157 0.
  0.01176471 0.01960784 0.01960784 0.0745098  0.09411765 0.00392157
  0.00784314 0.         0.01568627 0.         0.         0.00392157
  0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.01176471 0.01176471 0.
  0.         0.         0.00784314 0.00392157 0.00784314 0.01568627
  0.         0.         0.00784314 0.00784314 0.         0.
  0.         0.         0.         0.        ]]
(28, 28)
0
```


**[Celda 17 - Código]**
```python
model.summary()
```


*Salida:*
```text
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_6 (Conv2D)           (None, 26, 26, 32)        320       
                                                                 
 batch_normalization_7 (Batc  (None, 26, 26, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_7 (Conv2D)           (None, 24, 24, 32)        9248      
                                                                 
 batch_normalization_8 (Batc  (None, 24, 24, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_8 (Conv2D)           (None, 12, 12, 32)        25632     
                                                                 
 batch_normalization_9 (Batc  (None, 12, 12, 32)       128       
 hNormalization)                                                 
                                                                 
 dropout_3 (Dropout)         (None, 12, 12, 32)        0         
                                                                 
 conv2d_9 (Conv2D)           (None, 10, 10, 64)        18496     
                                                                 
 batch_normalization_10 (Bat  (None, 10, 10, 64)       256       
 chNormalization)                                                
                                                                 
 conv2d_10 (Conv2D)          (None, 8, 8, 64)          36928     
                                                                 
 batch_normalization_11 (Bat  (None, 8, 8, 64)         256       
 chNormalization)                                                
                                                                 
 conv2d_11 (Conv2D)          (None, 4, 4, 64)          102464    
                                                                 
 batch_normalization_12 (Bat  (None, 4, 4, 64)         256       
 chNormalization)                                                
                                                                 
 dropout_4 (Dropout)         (None, 4, 4, 64)          0         
                                                                 
 flatten_1 (Flatten)         (None, 1024)              0         
                                                                 
 dense_2 (Dense)             (None, 128)               131200    
                                                                 
 batch_normalization_13 (Bat  (None, 128)              512       
 chNormalization)                                                
                                                                 
 dropout_5 (Dropout)         (None, 128)               0         
                                                                 
 dense_3 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 327,242
Trainable params: 326,410
Non-trainable params: 832
_________________________________________________________________
```


**[Celda 18 - Código]**
```python
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

directorio = r'C:\Users\User\Documents\UTN\GIBD\Similitud de Marcas\BDNormalizada_R(28x28)'
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


*Salida:*
```text
['1', '10', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '11', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '12', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '13', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '14', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '15', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '16', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '17', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '18', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '19', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '2', '20', '200', '201', '202', '203', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']
```


**[Celda 19 - Código]**
```python
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

directorio = r'C:\Users\User\Documents\UTN\GIBD\Similitud de Marcas\Queries_R(28x28)'
contenido = os.listdir(directorio)

consultas = []
y_consultas = []
for fichero in contenido:
    if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.jpg'):
        image = directorio + '/' + fichero
        image_gris = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        image_gris = image_gris / 255
        image_gris = 1 - image_gris
        consultas.append(image_gris)
        n = fichero[:fichero.find('.')]
        n = n[n.find('-')+1:]
        y_consultas.append(n)
print(y_consultas)
```


*Salida:*
```text
['197', '11', '14', '16', '18', '19', '24', '24', '27', '33', '36', '5', '36', '44', '45', '46', '47', '47', '47', '49', '51', '53', '7', '54', '56', '63', '66', '66', '66', '76', '77', '78', '80', '11', '83', '87', '93', '95', '100', '103', '107', '112', '113', '115', '11', '0', '120', '122', '125', '139', '142', '142', '2', '144', '147', '11', '148', '150', '156', '157', '157', '158', '158', '159', '161', '162', '11', '162', '165', '166', '170', '171', '175', '0', '183', '185', '188', '11', '190', '191', '197', '200', '11']
```


**[Celda 20 - Código]**
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
        dist = np.sqrt(np.sum(np.square(a-b)))
        #dist = np.sum(np.abs(a-b))
        if dist < min:
            min = dist
            vector_cercano = b
            posicion_cercano = j
    print("Distancia: ", min, " al vector: ", vector_cercano, " en la posicion: ", posicion_cercano, " desde el vector: ", a, " que se encuentra en: ", i)
    print("Imagen de BD: ", y_imagenesBD[posicion_cercano], " Imagen de consulta: ", y_consultas[i], "\n")
    if y_imagenesBD[posicion_cercano] == y_consultas[i]:
        contador_aciertos += 1
        
print("Porcentaje de aciertos: ", contador_aciertos/cantidad_consultas)
```


*Salida:*
```text
Distancia:  3.8148224  al vector:  [9.08506989e-01 9.99995351e-01 9.34330523e-02 9.99952793e-01
 8.99295330e-01 9.99739289e-01 8.19703579e-01 5.79684911e-06
 9.99965906e-01 7.09707737e-02 8.66323709e-04 1.14870638e-01
 1.10158617e-04 7.88644731e-01 9.80223656e-01 9.99989271e-01
 1.48695699e-05 9.99995649e-01 9.99972761e-01 2.65738070e-02
 9.13751423e-02 6.41466113e-06 9.98092651e-01 7.95273736e-05
 8.13287377e-01 2.15727538e-01 7.47398444e-05 9.99623060e-01
 2.02459306e-01 1.93178654e-03 9.98353183e-01 9.28708076e-01
 9.81021166e-01 9.91200209e-01 2.20060468e-01 9.94032962e-06
 1.57614522e-05 8.18565488e-03 5.74569349e-05 9.99911308e-01
 9.99903500e-01 9.77917194e-01 9.30311382e-01 9.99879181e-01
 3.14200918e-07 9.99616742e-01 6.95703626e-01 1.00803375e-03
 2.26807594e-03 9.99987423e-01 2.44334579e-01 2.73687750e-01
 2.91850356e-05 7.22132981e-01 6.32736087e-03 2.11322308e-03
 3.40491533e-04 7.48726070e-01 9.99949217e-01 2.74279842e-06
 8.71968865e-01 1.55842751e-01 9.99294758e-01 6.73568547e-02
 1.07841700e-01 5.92482984e-01 1.69844031e-02 3.31541300e-02
 9.99996185e-01 3.03414792e-01 9.98888254e-01 7.23894536e-01
 2.44090676e-01 6.47532463e-01 8.01413476e-01 6.44523740e-01
 9.99022603e-01 1.06633246e-01 8.68065655e-01 4.18376625e-02
 9.98291492e-01 9.64674234e-01 6.52249515e-01 9.91557479e-01
 6.15675867e-01 3.00249457e-03 9.98596549e-01 9.99990165e-01
 2.57313251e-04 5.42637706e-03 9.95685458e-01 3.25789750e-02
 7.29143620e-04 4.20511067e-02 5.14699757e-01 3.53544950e-04
 9.84049797e-01 4.56933833e-07 1.54539347e-02 9.33141828e-01
 2.84098387e-01 6.51472807e-03 9.95265663e-01 9.99933362e-01
 1.21399909e-01 4.22179699e-04 9.58420277e-01 4.28349995e-05
 9.87969875e-01 2.76623964e-01 9.99986887e-01 3.92087698e-02
 3.26087117e-01 9.95046973e-01 1.80438161e-03 9.99955654e-01
 9.99999702e-01 1.72394663e-01 4.89610434e-03 1.27270818e-03
 1.94376707e-03 1.02337271e-01 5.14987111e-03 9.94589329e-01
 4.67777252e-04 9.99860048e-01 9.99993324e-01 5.30398190e-02]  en la posicion:  16  desde el vector:  [9.99999344e-01 1.00000000e+00 6.03259206e-02 9.99941707e-01
 6.97012424e-01 1.00000000e+00 1.74503446e-01 2.91672131e-06
 9.99450922e-01 5.57200337e-07 1.74830678e-10 2.59148031e-01
 2.56812190e-14 9.31236207e-01 4.34062958e-01 9.99489069e-01
 8.35511088e-03 1.00000000e+00 9.98500168e-01 1.32038504e-01
 2.63342912e-07 4.49099602e-09 9.99995470e-01 9.72221841e-08
 9.99999881e-01 5.74152509e-05 2.84807300e-07 9.99645472e-01
 9.98016119e-01 1.36165867e-09 9.98077273e-01 7.89894283e-01
 9.99994874e-01 7.32372940e-01 8.88353467e-01 3.98254991e-02
 1.41427645e-05 2.47642100e-02 2.31935183e-11 9.99943435e-01
 1.00000000e+00 9.99115705e-01 7.11906850e-02 1.00000000e+00
 1.27279396e-08 9.99867320e-01 7.33717084e-02 6.61048762e-06
 7.83022106e-01 1.00000000e+00 3.19150090e-03 7.41919875e-03
 2.66551971e-04 8.11531961e-01 6.96838439e-01 1.76455081e-02
 4.37936324e-05 4.34502155e-01 9.96947467e-01 4.50848070e-08
 9.35025811e-02 6.69597983e-02 1.27198696e-02 1.59538984e-02
 9.91466403e-01 5.44462919e-01 1.44985318e-03 4.62347865e-02
 1.00000000e+00 6.48763776e-03 1.00000000e+00 9.96419072e-01
 5.78790859e-05 6.65361702e-01 7.88352430e-01 7.39564896e-02
 1.00000000e+00 9.97298241e-01 9.47523892e-01 7.18921423e-04
 4.45100665e-03 9.99724507e-01 9.99873877e-01 9.43567157e-01
 1.20586395e-01 4.51621413e-03 9.99970496e-01 9.74949419e-01
 1.64606273e-02 7.67444144e-05 9.99998808e-01 3.51414084e-03
 9.03070807e-01 8.63805035e-05 3.80155116e-01 9.85928273e-05
 6.86303258e-01 1.05285280e-09 1.53076087e-06 5.36513329e-03
 6.82243705e-03 1.46591283e-05 9.99993205e-01 1.00000000e+00
 2.38742928e-06 1.70279443e-02 5.97661734e-03 7.44462013e-04
 9.99853134e-01 5.37598491e-01 1.00000000e+00 5.85231185e-03
 8.31396580e-01 9.40082669e-01 5.35870075e-01 9.99170899e-01
 9.99995530e-01 2.02563435e-01 5.77417016e-03 4.59779503e-09
 6.18225336e-03 9.43145156e-03 6.21146023e-01 1.83926225e-01
 2.03603932e-07 9.98210311e-01 2.94248581e-01 3.44723463e-04]  que se encuentra en:  0
Imagen de BD:  113  Imagen de consulta:  197 

Distancia:  4.181457  al vector:  [5.15968502e-02 9.06137466e-01 5.36172807e-01 9.99846339e-01
 1.81242704e-01 1.22658014e-02 2.04131812e-01 4.32521105e-04
 9.79200363e-01 3.17970544e-01 9.99862552e-01 5.37652791e-01
 4.88821387e-01 5.96279263e-01 9.84396517e-01 9.99947906e-01
 4.85578179e-03 9.97546911e-01 2.27295160e-01 8.14087510e-01
 9.99974728e-01 3.98265123e-02 9.69464421e-01 8.90675187e-03
 7.20888376e-03 1.70164704e-02 8.51035118e-04 6.27324879e-01
 6.32448971e-01 9.94222760e-01 9.99900401e-01 2.42010951e-02
 1.56802982e-01 2.90336609e-02 6.77570462e-01 1.95759385e-05
 2.30818987e-04 9.02505279e-01 8.13037157e-04 8.20899367e-01
 8.47158313e-01 2.08503485e-01 8.95891905e-01 9.83992338e-01
 1.29224360e-02 3.11013877e-01 1.20675564e-01 9.81408358e-03
 9.73791003e-01 5.98579645e-03 9.92495477e-01 9.60665643e-01
 7.60978460e-03 2.02894300e-01 4.31415439e-02 1.91062689e-03
 1.35835111e-02 2.21032679e-01 3.96014839e-01 1.32172108e-02
 2.07752317e-01 1.98052168e-01 9.99877810e-01 7.09184766e-01
 1.38190389e-03 9.40454066e-01 1.86693668e-02 4.40657198e-01
 9.68624532e-01 9.80394006e-01 6.80464506e-03 8.89979422e-01
 9.54253674e-01 1.42339945e-01 5.11686027e-01 6.59001291e-01
 3.91492844e-02 3.27104330e-01 3.56300235e-01 2.95619369e-02
 9.27471042e-01 9.80865479e-01 9.96350884e-01 9.41715956e-01
 8.90829563e-01 9.51681137e-02 9.67009425e-01 9.99997616e-01
 1.24830819e-06 9.92845476e-01 9.57603633e-01 7.88879395e-02
 3.02126855e-01 9.64108467e-01 1.36337876e-02 7.40200281e-03
 1.56563163e-01 5.24848700e-04 9.99887645e-01 9.50156331e-01
 9.50066090e-01 1.09171003e-01 9.37182128e-01 6.28556430e-01
 2.74271488e-01 2.97576189e-04 9.99784946e-01 1.39765739e-02
 9.93176699e-01 9.84674692e-03 3.86674106e-02 4.07384634e-02
 9.92000461e-01 8.44877064e-01 6.55004203e-01 4.33980882e-01
 9.97982860e-01 9.18908715e-02 2.34859109e-01 4.48516011e-03
 2.86430538e-01 2.90363789e-01 8.38786364e-01 9.99987602e-01
 2.85011530e-03 9.97668862e-01 9.56885338e-01 8.99804831e-01]  en la posicion:  74  desde el vector:  [6.13355696e-01 9.78893280e-01 2.17866451e-01 9.72416878e-01
 8.43114138e-01 7.99310803e-02 9.98225451e-01 3.77982855e-04
 9.81335759e-01 1.37289345e-01 9.97284770e-01 1.63467407e-01
 9.73848760e-01 2.36237466e-01 8.36032987e-01 9.94745255e-01
 5.58774948e-01 9.99578238e-01 8.93906057e-01 7.03943372e-02
 8.47912908e-01 3.83931845e-01 9.83514190e-01 8.53099227e-02
 4.44521815e-01 4.20990556e-01 2.25399703e-01 9.79833841e-01
 9.68330145e-01 8.77515197e-01 9.95755613e-01 1.61287159e-01
 3.29324037e-01 8.91558647e-01 7.08441973e-01 7.92801380e-04
 1.95108652e-02 1.78538203e-01 5.56967556e-02 9.97079313e-01
 8.92777920e-01 5.95478594e-01 8.85208368e-01 9.95001435e-01
 1.70475245e-03 1.30333990e-01 3.44148815e-01 2.77470648e-01
 5.95231175e-01 5.64001203e-01 8.03659678e-01 2.28890300e-01
 5.36240280e-01 1.21554732e-02 7.53432512e-04 1.63659543e-01
 1.21040255e-01 5.61746061e-02 9.70577240e-01 5.52865863e-03
 4.92158234e-02 6.42500103e-01 9.99711275e-01 3.00728977e-02
 2.64495015e-02 9.40023720e-01 8.99636507e-01 9.50742781e-01
 8.03544879e-01 5.73822439e-01 2.31367350e-03 9.26058292e-01
 4.92197096e-01 8.57478380e-03 9.19502020e-01 2.37852335e-02
 2.89697766e-01 7.68220067e-01 1.34301454e-01 5.55069327e-01
 6.77710891e-01 9.70272422e-01 9.84952211e-01 9.99659777e-01
 6.49807096e-01 2.88367271e-04 8.32170069e-01 9.99997735e-01
 1.33541226e-03 1.28433615e-01 9.97676253e-01 3.57151031e-03
 1.31120086e-02 7.20373034e-01 7.37584293e-01 4.91636932e-01
 8.35090041e-01 4.91958381e-05 9.55950737e-01 9.46706474e-01
 2.64932811e-01 9.41869020e-01 6.02100134e-01 8.62969518e-01
 4.05988842e-01 4.85062599e-04 9.86658335e-01 6.23372376e-01
 9.51788783e-01 3.47383618e-02 2.18120992e-01 1.00586265e-01
 9.91079569e-01 2.19612598e-01 1.88501954e-01 9.94002700e-01
 9.51507092e-01 3.35910350e-01 9.68606830e-01 1.85438991e-03
 8.91274214e-03 9.14582491e-01 6.93637431e-01 9.99851704e-01
 3.30957770e-03 2.38592982e-01 9.99329448e-01 7.94435263e-01]  que se encuentra en:  1
Imagen de BD:  166  Imagen de consulta:  11 

Distancia:  2.8758457  al vector:  [1.46114230e-02 9.99354303e-01 6.47236526e-01 9.94827986e-01
 9.78531718e-01 3.32912803e-03 6.43763542e-02 1.16163203e-04
 9.99995828e-01 5.58427811e-01 9.88509417e-01 9.64802146e-01
 8.29202414e-01 2.85004079e-02 7.56824136e-01 9.99636829e-01
 5.58960736e-02 9.99994993e-01 2.77146339e-01 2.28979617e-01
 9.99161959e-01 1.12234498e-04 9.96111929e-01 1.54855847e-01
 1.11859008e-04 3.50501329e-01 4.14162874e-03 3.81945372e-02
 3.04234833e-01 9.75382924e-01 9.99830902e-01 1.21959776e-01
 9.56717610e-01 4.61180508e-02 9.98159647e-01 1.77234411e-03
 2.83092260e-04 9.97213483e-01 8.73565674e-04 9.96485412e-01
 9.99672890e-01 9.70947266e-01 8.41392636e-01 9.99995708e-01
 1.41948462e-04 6.73337579e-02 3.99678171e-01 2.57819891e-04
 8.55569839e-01 4.86756265e-02 9.95719910e-01 5.81312239e-01
 2.93625593e-02 4.00409311e-01 8.92226458e-01 8.29866694e-05
 5.90735835e-05 8.82343471e-01 3.61113638e-01 5.49891120e-05
 1.42787606e-01 5.29159486e-01 9.99998689e-01 1.26876920e-01
 5.45153022e-03 6.92773998e-01 6.20420277e-01 3.50329220e-01
 9.99842644e-01 6.81825280e-01 7.23311305e-03 2.61937648e-01
 9.96042907e-01 8.38785291e-01 4.00664508e-02 6.41746640e-01
 3.51931453e-02 9.11970437e-02 7.50176787e-01 4.13188934e-02
 4.94774193e-01 9.98557627e-01 9.99862850e-01 1.98211074e-01
 2.77108103e-01 2.82171071e-02 9.99530017e-01 1.00000000e+00
 1.66386366e-04 7.23143816e-01 9.99439597e-01 1.42661035e-01
 9.69761908e-01 9.77484465e-01 1.98722929e-01 1.47311389e-02
 4.30977345e-03 5.09906295e-06 9.98367190e-01 9.65297043e-01
 8.13472271e-01 4.05613482e-02 9.99800444e-01 8.86662304e-02
 3.72920692e-01 7.34299421e-04 9.79757667e-01 6.19547367e-02
 1.89061940e-01 5.08662224e-01 7.17171729e-02 1.10078454e-01
 1.71652764e-01 9.01210427e-01 8.41259837e-01 2.30609775e-01
 9.99927461e-01 5.17060637e-01 4.55842018e-02 5.80689448e-05
 9.98804808e-01 9.93350267e-01 3.11889380e-01 9.99998808e-01
 4.94480133e-04 9.93857145e-01 9.65594530e-01 9.96722877e-02]  en la posicion:  45  desde el vector:  [1.70135318e-05 8.61194551e-01 9.72544491e-01 9.95456457e-01
 9.87010002e-01 1.34398542e-05 3.90282273e-03 1.02153122e-01
 9.99885917e-01 3.86530221e-01 9.99627233e-01 7.39997923e-01
 1.02584422e-01 4.93507385e-02 5.49755991e-02 9.99668241e-01
 4.85391319e-02 9.99208331e-01 1.38461590e-04 4.78580296e-01
 9.99960899e-01 7.76827335e-04 9.73062396e-01 2.70277858e-01
 4.46517624e-05 2.69593596e-02 1.48636103e-03 1.19389802e-01
 4.49957132e-01 9.99938965e-01 9.98752356e-01 7.45384037e-01
 9.32630956e-01 1.16174502e-04 9.51601982e-01 1.54823065e-04
 1.06891036e-01 9.99682844e-01 2.01488823e-01 9.97343540e-01
 9.91051435e-01 9.43437815e-01 5.48901260e-02 9.99925554e-01
 1.96523070e-02 2.69627273e-02 2.10914403e-01 2.77408957e-03
 9.98764157e-01 1.37865543e-04 9.91032898e-01 9.41950798e-01
 8.91032517e-02 2.69494653e-01 9.90211248e-01 1.84983015e-04
 1.34363770e-03 8.03998590e-01 4.87167895e-01 2.11775303e-03
 5.56816816e-01 9.78599906e-01 9.99974012e-01 2.83487916e-01
 6.11021896e-05 8.88005137e-01 7.27594554e-01 7.04611182e-01
 9.39519525e-01 5.05855680e-03 5.58674335e-04 8.49898160e-02
 9.99996901e-01 9.45969939e-01 3.14231694e-01 8.45421553e-01
 1.42499506e-02 2.25360185e-01 7.26314545e-01 9.19642508e-01
 1.20689601e-01 9.97078180e-01 9.99033451e-01 9.61452723e-04
 4.05376822e-01 7.09310472e-02 8.84896040e-01 9.99972820e-01
 5.94791309e-05 9.43701327e-01 7.08446920e-01 9.49202180e-01
 9.73027408e-01 9.99991298e-01 6.81922436e-01 9.42501426e-03
 6.48114292e-05 3.10689211e-04 9.99934196e-01 8.82061720e-01
 4.94231254e-01 1.19220257e-01 9.88657355e-01 3.35681689e-05
 6.11902177e-01 2.53826380e-04 9.79060292e-01 8.19210112e-02
 1.76247358e-02 6.49320781e-01 1.02176251e-04 5.73553860e-01
 5.58335185e-01 5.50486028e-01 5.19417167e-01 8.66115093e-04
 9.98573422e-01 8.62857938e-01 1.72049999e-02 7.82114267e-03
 9.99906182e-01 9.98352051e-01 9.05546665e-01 9.99974132e-01
 4.79871035e-03 9.93543506e-01 3.17207575e-02 1.04052812e-01]  que se encuentra en:  2
Imagen de BD:  14  Imagen de consulta:  14 

Distancia:  3.6001961  al vector:  [1.44376136e-05 8.50040014e-05 3.72910798e-01 9.99106050e-01
 1.01077348e-01 2.28341578e-05 1.89042747e-01 1.64003670e-02
 9.04050946e-01 9.99994278e-01 9.97624993e-01 1.56119168e-02
 9.75876331e-01 9.90020514e-01 4.12822366e-02 9.98890698e-01
 1.11719877e-04 2.89925933e-03 9.82940555e-01 2.45737940e-01
 9.99952435e-01 9.13312435e-01 9.92278278e-01 9.28867102e-01
 2.33663377e-05 2.65431404e-03 9.12960887e-01 7.65117049e-01
 8.10774565e-01 9.99806285e-01 6.24501765e-01 1.43598467e-01
 9.82394814e-03 9.93581057e-01 2.66195565e-01 3.17797065e-03
 4.57823277e-04 6.83431327e-02 9.95528460e-01 9.99353528e-01
 1.44025385e-02 2.91439354e-01 9.79745507e-01 7.13018775e-02
 6.60972297e-02 9.99017000e-01 4.49279636e-01 7.71018863e-03
 7.15872347e-02 9.85034585e-01 2.04950154e-01 6.34675741e-01
 1.24990940e-04 5.09251237e-01 4.36655015e-01 9.98083591e-01
 4.42430377e-03 7.48579502e-01 9.56186175e-01 3.51686180e-02
 9.92208600e-01 9.54032898e-01 9.94501531e-01 7.99277723e-02
 7.96154141e-03 5.02529740e-03 1.86959505e-02 5.12328804e-01
 2.95627117e-03 9.93750572e-01 4.80600684e-05 5.95952868e-01
 9.99992788e-01 1.38247609e-02 1.45032078e-01 9.70015109e-01
 1.48057938e-04 1.76742285e-01 8.69369507e-03 3.13380957e-02
 2.92247683e-01 8.92746175e-05 3.77684832e-04 1.15016580e-01
 2.72938132e-01 8.88262749e-01 5.97466396e-06 8.83035421e-01
 5.90076447e-02 7.61547983e-02 4.00125980e-04 9.99961019e-01
 1.13389432e-01 9.99931574e-01 8.69292498e-01 5.12570143e-04
 9.31420207e-01 9.99844015e-01 9.99970317e-01 2.41835952e-01
 5.15281379e-01 1.98253393e-02 9.79646564e-01 3.08692455e-04
 9.58411813e-01 2.09351480e-02 9.79461730e-01 7.12960958e-04
 2.92640924e-03 6.18101060e-01 4.48352039e-05 8.93164873e-02
 9.90919590e-01 9.79990482e-01 1.08870596e-01 9.73836541e-01
 9.99989748e-01 8.30443859e-01 5.20943344e-01 9.97583866e-01
 2.72123069e-01 9.51296747e-01 8.40061903e-03 7.94412732e-01
 9.99976933e-01 9.99802828e-01 9.31818962e-01 9.99900937e-01]  en la posicion:  152  desde el vector:  [1.6176355e-06 1.8758564e-06 4.0331656e-01 9.9902248e-01 3.6796230e-01
 6.8906706e-07 8.9417690e-01 1.7187801e-01 9.1147238e-01 9.9973428e-01
 9.9994767e-01 4.1890979e-02 9.2397487e-01 9.2575228e-01 5.1693112e-02
 9.9670494e-01 2.8467178e-03 5.9982631e-05 5.3910440e-01 2.5962472e-02
 1.0000000e+00 9.8003590e-01 9.8486960e-01 9.8968738e-01 2.4960354e-07
 3.7833029e-01 9.8549116e-01 6.3305408e-01 9.2052865e-01 9.9999797e-01
 2.7326962e-01 9.0637439e-01 3.0987859e-03 5.8663547e-02 2.2072783e-01
 7.3823333e-04 3.0947745e-02 9.7525513e-01 9.9845707e-01 3.1530970e-01
 7.3698461e-03 5.0090700e-02 9.8704982e-01 5.8531761e-04 2.2269800e-01
 9.5917320e-01 7.3846400e-01 9.2797935e-02 9.7426426e-01 4.7467750e-01
 1.0533488e-01 8.0332637e-01 1.8352270e-04 4.2176443e-01 8.4908175e-01
 9.9093252e-01 1.3028085e-02 4.4721520e-01 9.9887228e-01 5.7633686e-01
 6.8087029e-01 4.7626016e-01 8.0733263e-01 6.5860260e-01 8.5774064e-04
 9.4668448e-02 1.4293820e-02 4.2797938e-01 5.2321473e-05 9.9216664e-01
 5.7537452e-08 9.8185921e-01 1.0000000e+00 1.6444623e-02 1.3461488e-01
 6.9709194e-01 1.9766048e-05 8.6694682e-01 2.3297608e-02 4.4635206e-02
 7.5531012e-01 3.4874678e-04 1.1366952e-04 8.7826848e-03 4.5023376e-01
 1.3107657e-02 3.1122050e-05 8.3400643e-01 1.8021464e-04 9.2588389e-01
 1.0070052e-04 9.9999809e-01 6.1445087e-02 1.0000000e+00 9.6156877e-01
 1.4340878e-04 1.8254668e-02 9.9999768e-01 1.0000000e+00 2.9439485e-01
 6.7645174e-01 1.5100136e-01 7.6518369e-01 2.2260185e-06 9.3550152e-01
 7.2106719e-04 8.1693029e-01 4.0060282e-04 1.1738390e-02 1.5756476e-01
 1.6040055e-07 8.9123052e-01 8.5731232e-01 8.7429821e-01 8.6374879e-03
 2.7677044e-01 9.9985957e-01 9.8817819e-01 7.6407689e-01 9.9967098e-01
 9.6159124e-01 3.2061934e-03 1.7273724e-03 7.3336172e-01 9.9987161e-01
 9.9870729e-01 4.4960368e-01 9.9997437e-01]  que se encuentra en:  3
Imagen de BD:  53  Imagen de consulta:  16 

Distancia:  3.723345  al vector:  [1.6323060e-02 4.6174318e-02 9.9476874e-01 9.9999368e-01 1.8190560e-01
 6.6123760e-01 5.5944532e-02 2.6188940e-01 3.9888550e-05 3.8640192e-01
 9.9992508e-01 2.3162833e-01 8.8600516e-03 9.8566449e-01 1.3101923e-01
 9.9999607e-01 3.4719389e-05 4.4404796e-01 1.2759283e-01 5.1790357e-02
 9.9720860e-01 7.3470473e-03 1.4936924e-04 6.6713374e-06 5.2531362e-03
 2.6747257e-02 1.1185825e-06 9.9925029e-01 3.4584731e-02 4.1385269e-01
 5.0315261e-03 9.6086544e-01 4.8058629e-03 2.4801493e-04 2.1766424e-03
 3.3417166e-06 9.1711450e-01 9.9947131e-01 9.5290351e-01 5.6157482e-01
 5.3575701e-01 6.3687444e-01 2.4810791e-02 8.4514403e-01 9.6053255e-01
 9.9748486e-01 5.7451749e-01 9.9994218e-01 9.9977982e-01 6.0865164e-02
 3.4199160e-01 8.7024599e-01 7.4744058e-01 4.2519069e-01 9.9348950e-01
 1.4343858e-04 9.9388492e-01 1.1263967e-02 9.8022532e-01 9.9230146e-01
 1.3213724e-01 9.3867302e-01 1.8042326e-04 9.8568749e-01 3.8713217e-04
 4.4768423e-02 6.4228934e-01 9.1027021e-01 6.1364907e-01 1.1183980e-01
 6.8862426e-01 5.3131878e-03 9.8092341e-01 9.6410966e-01 6.0203284e-01
 3.5448998e-01 1.6992390e-03 9.0202427e-01 1.8316507e-03 9.9826026e-01
 4.3597037e-01 9.6159875e-01 2.2720933e-02 1.9968450e-03 8.2887971e-01
 5.7132304e-01 6.6092765e-01 2.9278159e-02 2.0303465e-05 9.9932957e-01
 1.5250862e-02 9.7150791e-01 2.6258796e-02 9.9842632e-01 9.1772026e-01
 3.6607493e-05 4.4396520e-04 7.4944401e-01 9.9275923e-01 8.8639063e-01
 7.5854754e-01 9.9955189e-01 1.7142296e-04 5.3660840e-02 5.7908893e-03
 5.2628866e-06 5.1050985e-01 1.8101589e-05 3.0221701e-02 4.8394442e-02
 9.8615885e-04 2.2992486e-01 7.0366508e-01 1.3932586e-04 9.4021451e-01
 1.5430957e-01 8.6705983e-01 1.2512416e-02 3.5243869e-01 9.9695867e-01
 9.9961674e-01 9.6638536e-01 9.6481109e-01 1.6007139e-05 4.1797078e-01
 9.9993193e-01 1.5011591e-01 8.8676059e-01]  en la posicion:  197  desde el vector:  [2.03865230e-01 1.08960271e-03 7.82226920e-01 9.94943142e-01
 8.49714875e-01 1.36575103e-03 1.65786743e-02 5.76930821e-01
 4.02991474e-02 9.25050616e-01 9.99678433e-01 7.22317994e-01
 9.84202087e-01 8.20919514e-01 4.31193143e-01 9.99995232e-01
 4.24000391e-05 4.26604450e-02 1.90041065e-02 8.55159461e-02
 9.96756375e-01 3.71336460e-01 2.24649906e-04 2.75799632e-03
 8.42121243e-03 1.52509063e-01 2.55510549e-05 9.97771740e-01
 2.13075578e-02 9.40848053e-01 5.19510984e-01 5.96689343e-01
 9.54473019e-03 7.00123787e-01 4.23691064e-01 7.84610620e-06
 9.96762991e-01 9.43581820e-01 9.99718964e-01 8.57228041e-01
 2.72027254e-02 5.40935576e-01 4.63588029e-01 8.92094135e-01
 9.97017860e-01 9.93295074e-01 4.37582761e-01 9.89993334e-01
 7.33341455e-01 9.79995728e-03 8.46717238e-01 9.82010365e-01
 2.08201438e-01 4.59830344e-01 6.13682330e-01 8.05805466e-05
 9.64036226e-01 2.33306289e-02 3.57967675e-01 9.94068980e-01
 8.62199008e-01 9.21361804e-01 3.68419588e-01 9.37854826e-01
 4.12844347e-05 6.37163401e-01 5.65609872e-01 9.76973653e-01
 2.89916992e-04 8.75954568e-01 3.53676081e-03 5.16995430e-01
 9.66058731e-01 6.14733100e-01 9.65696573e-03 9.16940928e-01
 3.43978405e-04 6.39152408e-01 3.16422880e-02 8.33867371e-01
 9.28564429e-01 9.86166298e-01 1.75268352e-02 4.41776276e-01
 9.19733286e-01 8.62108350e-01 8.36091757e-01 7.79640973e-02
 1.59442425e-04 9.97642159e-01 1.92552805e-04 4.77955759e-01
 8.14466596e-01 8.96579385e-01 6.97725296e-01 4.95940447e-04
 6.04924858e-02 8.65221739e-01 9.82288837e-01 6.93876147e-01
 5.25941670e-01 9.97408032e-01 2.65267491e-03 2.41367221e-02
 1.05037928e-01 9.03371620e-06 9.99200821e-01 1.16875768e-03
 4.51713800e-03 2.02619553e-01 1.11796726e-04 7.99775720e-01
 8.29827487e-02 1.54781729e-01 4.99310076e-01 4.38814163e-02
 9.93422091e-01 4.35436338e-01 6.57266378e-02 9.97454286e-01
 9.79846239e-01 5.83578408e-01 4.66762871e-01 3.50811481e-02
 9.96342301e-01 9.99889612e-01 7.48315334e-01 9.05104995e-01]  que se encuentra en:  4
Imagen de BD:  94  Imagen de consulta:  18 

Distancia:  4.1564627  al vector:  [9.7469169e-01 5.2057803e-03 8.1321365e-01 7.7500665e-01 7.4288034e-01
 9.9141431e-01 5.2368343e-02 1.1565387e-03 9.3548447e-01 7.1993470e-04
 9.9973375e-01 9.2620438e-01 9.9818099e-01 2.7757537e-01 8.0631214e-01
 9.9999928e-01 7.7855587e-04 1.2369782e-02 9.6791589e-01 7.8664434e-01
 4.0796697e-03 9.9990940e-01 1.1951953e-02 1.5577477e-01 9.9908304e-01
 2.8042161e-01 2.4882257e-03 9.9998707e-01 9.9170947e-01 1.8858016e-03
 9.9981517e-01 4.8081470e-01 2.8098725e-06 9.9883664e-01 5.9380829e-01
 8.4869527e-07 3.3808476e-01 6.6462159e-04 9.9021697e-01 5.2549285e-01
 1.0672137e-01 9.2757988e-01 8.7835294e-01 2.1638870e-03 9.8043573e-01
 9.9882579e-01 1.3473687e-01 3.3255729e-01 4.2074919e-04 9.9505603e-01
 9.9294162e-01 2.0371476e-01 9.9681485e-01 3.7609652e-01 9.0632802e-06
 5.5056810e-04 5.5727035e-02 9.3995816e-01 5.5368012e-01 9.5190012e-01
 3.1800330e-01 6.9195783e-01 7.8814781e-01 5.1474422e-02 3.1395950e-05
 9.1398025e-01 9.2873907e-01 9.8276877e-01 2.9930472e-04 1.5341532e-01
 9.9987519e-01 1.2131593e-01 2.8362539e-01 1.3610473e-01 7.2878748e-02
 4.1187406e-03 1.0376489e-01 8.6128259e-01 1.6717896e-01 9.7593594e-01
 9.5091909e-01 9.9999636e-01 9.4170809e-01 9.9999869e-01 2.8870162e-01
 1.4264315e-02 9.6112907e-01 8.4251863e-01 6.6676162e-06 8.3451402e-01
 2.3978949e-04 2.1642447e-04 9.9836779e-01 4.1399240e-02 9.9475503e-01
 1.2881637e-02 9.9969935e-01 9.4422823e-01 3.4707785e-03 3.2189846e-02
 3.8566867e-01 9.9947989e-01 5.1213205e-02 9.8115242e-01 2.4275532e-01
 2.8718426e-07 6.5069216e-01 8.3103776e-04 2.7115995e-01 2.2673565e-01
 5.4973066e-03 2.1623433e-01 2.2421479e-03 9.2426014e-01 1.4709604e-01
 9.5373058e-01 1.1990786e-02 6.1376798e-01 7.1364462e-01 9.8628658e-01
 2.3864806e-03 2.2753918e-01 9.1069597e-01 9.7631335e-01 3.6643291e-01
 9.9968946e-01 9.9969566e-01 1.3435394e-02]  en la posicion:  94  desde el vector:  [8.9102459e-01 6.7774653e-03 8.9160740e-01 3.2357371e-01 5.0070882e-02
 6.3190556e-01 9.0248585e-03 2.5711739e-01 9.8589212e-01 1.1290044e-02
 9.9452543e-01 9.8674703e-01 9.8823977e-01 1.7236581e-01 9.9814487e-01
 9.9998653e-01 2.6753545e-03 9.6132630e-01 1.6169748e-01 9.8559636e-01
 1.0578096e-02 4.4904637e-01 7.2458386e-03 2.2998303e-02 9.2447364e-01
 5.5073231e-01 2.4679303e-04 9.9939764e-01 3.2387930e-01 6.3561296e-01
 9.7457767e-01 2.6021063e-02 1.3849139e-03 9.2674458e-01 6.4823204e-01
 3.3872802e-05 9.6979821e-01 1.7139843e-01 9.9406159e-01 9.8441094e-01
 6.7374820e-01 9.4607389e-01 1.4345694e-01 9.7672307e-01 9.9783707e-01
 6.9020069e-01 7.2686881e-01 8.8114691e-01 8.8011026e-03 3.1048059e-04
 8.9639080e-01 7.6761144e-01 9.9649537e-01 2.0814270e-02 8.3625317e-04
 7.1942850e-07 9.6135348e-02 8.3763564e-01 8.9194697e-01 9.6788132e-01
 8.5462403e-01 7.4948585e-01 9.9934489e-01 9.5965576e-01 9.0435981e-07
 9.9402034e-01 9.6936798e-01 9.2911172e-01 1.1332929e-03 6.0242236e-02
 7.4141538e-01 6.1616361e-01 3.0051887e-02 2.3038188e-01 9.9686611e-01
 4.6571669e-01 8.7675983e-01 6.8384135e-01 2.5356907e-01 9.8054945e-01
 6.2426567e-01 9.9999416e-01 9.9303555e-01 9.9993706e-01 9.9937057e-01
 2.3461404e-01 9.9891132e-01 9.9966997e-01 9.6898824e-05 5.5725932e-01
 3.9612097e-01 5.4457784e-04 9.9945629e-01 7.2381467e-02 9.1420627e-01
 9.7344607e-02 6.2200844e-01 4.4611394e-03 1.8145442e-03 4.4100124e-01
 3.2385027e-01 9.9862897e-01 3.6721259e-02 8.0794227e-01 9.9013370e-01
 4.0861614e-06 9.9780393e-01 3.9969683e-02 9.4011426e-02 9.8170131e-02
 3.4242570e-03 9.8627675e-01 9.6210837e-04 8.4436393e-01 2.1831781e-02
 2.2886229e-01 3.6430925e-02 5.8827758e-02 3.4650660e-01 5.2151269e-01
 1.0200009e-01 8.7141520e-01 9.4905442e-01 9.9999154e-01 2.0780295e-02
 9.7885227e-01 9.9753773e-01 9.4309449e-04]  que se encuentra en:  5
Imagen de BD:  184  Imagen de consulta:  19 

Distancia:  3.9788141  al vector:  [6.3126385e-03 7.1004564e-08 6.1413717e-01 5.8305264e-04 8.4852964e-02
 5.4842026e-06 3.3042073e-02 9.9996799e-01 7.8737342e-01 9.9986321e-01
 9.9999756e-01 9.0354955e-01 9.9996793e-01 7.7404976e-03 3.5774159e-01
 9.9984670e-01 9.4943440e-01 1.8513143e-02 8.9684129e-04 1.1551559e-02
 9.9321091e-01 9.9364716e-01 5.2920503e-05 9.9865985e-01 5.1854551e-03
 9.8791534e-01 9.8499417e-01 9.9997365e-01 1.3789445e-01 9.9998271e-01
 1.3841391e-03 6.6822988e-01 1.8256145e-07 1.8681586e-02 9.5990896e-01
 5.5491924e-03 9.9995327e-01 9.9799711e-01 9.9999899e-01 5.8370137e-01
 7.1294308e-03 9.7555804e-01 9.8385787e-01 3.4437591e-01 9.9998397e-01
 4.1819751e-01 6.1191493e-01 9.9991965e-01 9.7572601e-01 1.6894708e-05
 1.6290513e-01 1.9380960e-01 9.9898994e-01 2.8555489e-01 3.9309263e-04
 1.8045604e-03 7.3019755e-01 8.5090947e-01 4.4447860e-01 9.9999630e-01
 9.7258580e-01 5.4484606e-04 9.8973435e-01 9.7413838e-01 1.0570886e-05
 9.8645806e-01 5.9918904e-01 8.1389391e-01 1.3204958e-08 8.5925746e-01
 9.9884137e-06 5.5922765e-01 9.7264171e-01 6.9813204e-01 7.5003505e-04
 1.1190891e-02 2.0929366e-01 9.9810934e-01 7.8557575e-01 8.3251905e-01
 1.3256907e-02 7.2108293e-01 8.8739192e-01 9.9999261e-01 9.9857581e-01
 9.4510132e-01 3.5265088e-04 9.5300364e-01 9.5157120e-05 1.9446728e-01
 6.9795897e-06 7.9023540e-03 9.5769703e-01 9.6980876e-01 9.6937221e-01
 9.9455982e-01 1.8049777e-02 9.8613465e-01 9.9655139e-01 9.4559789e-04
 6.6825593e-01 9.9998176e-01 5.3602457e-04 6.6164136e-04 5.5823821e-01
 2.9101968e-04 4.4275832e-01 9.9261230e-01 9.0557194e-01 8.2733548e-01
 5.0033573e-06 3.8322264e-01 6.8960363e-01 9.7574222e-01 8.7912649e-02
 2.3928285e-04 4.0800869e-03 3.3718106e-01 1.3116002e-04 9.3624914e-01
 9.8387057e-01 1.4961246e-01 9.3702745e-01 9.8241782e-01 8.3722436e-01
 2.0924091e-02 9.6932757e-01 9.9913526e-01]  en la posicion:  190  desde el vector:  [2.5343895e-04 8.3544552e-03 8.6622095e-01 2.9045373e-02 1.5866756e-02
 2.4427931e-07 2.1168676e-01 9.9969292e-01 9.8589325e-01 9.9995458e-01
 9.9991441e-01 6.9583881e-01 7.9096031e-01 2.7995551e-01 4.6536607e-01
 9.9210340e-01 9.9168491e-01 9.7648978e-01 3.4627097e-05 4.5764446e-02
 9.9998868e-01 2.9792488e-02 1.9957989e-02 9.9747932e-01 2.8486218e-05
 9.2909670e-01 9.8183906e-01 9.8827285e-01 1.0371268e-02 9.9999535e-01
 1.0852516e-03 7.7117658e-01 4.6347380e-03 2.6049997e-06 1.8946806e-01
 3.3439696e-03 9.9933159e-01 9.9997091e-01 9.9918282e-01 6.9243699e-02
 9.8512077e-01 9.6398318e-01 6.8543845e-01 9.9968672e-01 9.8931462e-01
 2.3084879e-04 6.4853096e-01 9.8453462e-01 9.9996817e-01 3.9431453e-04
 4.1704416e-01 7.9369301e-01 9.7017252e-01 1.9111097e-02 3.8099349e-02
 1.0910541e-02 1.6896427e-02 7.5731575e-03 9.5582116e-01 9.8081243e-01
 2.3572743e-01 4.3952012e-01 9.8508626e-01 3.3682102e-01 2.1781921e-02
 9.2852545e-01 2.8638333e-01 9.9346077e-01 1.1637211e-03 7.6213735e-01
 2.1903268e-06 8.9137506e-01 9.9998128e-01 2.4304390e-02 2.6305497e-02
 3.3234271e-01 9.6081144e-01 9.8060906e-01 8.6022252e-01 8.9243251e-01
 9.5751792e-02 9.0586114e-01 5.5644834e-01 9.5763177e-01 1.0199025e-01
 2.4266192e-01 8.7216496e-04 9.9243975e-01 1.1446089e-02 9.0480894e-02
 1.6828775e-03 2.1860304e-01 9.3542361e-01 9.9998528e-01 5.2471042e-02
 9.8784912e-01 4.8928156e-05 2.0491183e-03 9.9999523e-01 5.7211757e-01
 9.5419109e-01 9.9903512e-01 3.8082600e-03 1.8211305e-03 8.0073464e-01
 5.9690177e-03 1.1158103e-01 9.9687469e-01 8.6703855e-01 1.1516482e-02
 9.7969174e-03 9.0848559e-01 4.6286315e-01 4.6837211e-01 6.3385820e-01
 4.3274216e-05 2.4356899e-01 8.0755079e-01 1.1010081e-02 2.2950292e-02
 9.9993801e-01 7.7004462e-02 9.8121631e-01 8.5376412e-01 6.7994905e-01
 8.0754161e-03 8.5566568e-01 9.9998629e-01]  que se encuentra en:  6
Imagen de BD:  88  Imagen de consulta:  24 

Distancia:  4.1394873  al vector:  [1.43974066e-01 2.26207674e-02 6.92189217e-01 6.21809065e-02
 9.48277712e-02 1.49542275e-05 2.05239356e-02 9.99293685e-01
 9.59654152e-01 9.98679876e-01 5.53525746e-01 5.71302414e-01
 8.65137160e-01 9.32547569e-01 9.13523912e-01 9.19403791e-01
 8.30495059e-01 8.16198707e-01 1.28418207e-04 3.40615690e-01
 9.93584931e-01 2.99528539e-02 4.85707730e-01 9.78404462e-01
 2.38374472e-02 3.40515375e-01 8.50967765e-01 3.78341079e-02
 3.82155180e-04 9.99924302e-01 9.51397419e-03 1.23354793e-03
 7.33407319e-01 2.11882472e-01 6.39728308e-02 1.90520495e-01
 9.99579310e-01 9.55461860e-01 9.99651074e-01 6.15990341e-01
 7.91996241e-01 1.19848847e-01 8.73973489e-01 7.94479847e-01
 9.59369898e-01 1.50109828e-02 5.22896230e-01 2.30609030e-01
 7.64728248e-01 2.85276492e-05 5.37487507e-01 8.76889408e-01
 7.34917819e-01 1.18238002e-01 5.27320743e-01 2.27935821e-01
 4.25312072e-01 3.77923250e-02 5.24956822e-01 9.40122247e-01
 9.98124838e-01 7.48931289e-01 4.23940539e-01 2.37168640e-01
 3.33684504e-01 2.54464865e-01 3.64851177e-01 9.81048524e-01
 1.94959104e-01 6.45566344e-01 3.32499076e-05 8.62290263e-01
 9.42219257e-01 1.23966962e-01 1.04670525e-02 6.22635722e-01
 7.11283386e-01 2.75972396e-01 2.49036252e-02 8.94903839e-02
 3.70618701e-03 6.06223524e-01 7.30829239e-02 7.08132446e-01
 9.02193904e-01 9.88969445e-01 4.30592299e-02 2.84807980e-02
 9.40945625e-01 3.79102230e-02 1.25616789e-04 3.55755061e-01
 9.82963204e-01 7.11909473e-01 1.14550889e-02 9.84823644e-01
 1.89562410e-01 9.42047894e-01 9.91719604e-01 5.64148366e-01
 1.73634738e-01 9.44343507e-02 8.49181354e-01 1.20128989e-02
 9.83827055e-01 2.96946913e-01 9.39907193e-01 9.95696902e-01
 6.84734285e-02 4.06679809e-02 3.35627794e-03 7.41910636e-01
 9.77065325e-01 8.84062767e-01 5.42998493e-01 1.19704008e-03
 3.25788349e-01 6.45727217e-02 5.18279672e-02 7.19875097e-03
 9.53695536e-01 3.95149887e-01 5.98499298e-01 7.23580599e-01
 9.99182761e-01 1.11114383e-02 5.18369675e-03 8.91452312e-01]  en la posicion:  171  desde el vector:  [4.6992689e-02 1.3939303e-01 6.8571919e-01 1.8108565e-05 7.6119542e-02
 2.8454124e-07 6.8601370e-03 9.9993211e-01 9.8722553e-01 9.9893832e-01
 9.9145210e-01 9.8892879e-01 9.9955392e-01 9.7290277e-03 8.6224318e-01
 6.3277370e-01 9.9989653e-01 9.9976748e-01 7.2272655e-05 8.6038816e-01
 9.9995565e-01 2.1806657e-03 1.5980005e-03 9.9741304e-01 5.3735077e-03
 9.4428992e-01 9.4661117e-01 2.3352432e-01 4.7944695e-02 9.9999928e-01
 2.9362053e-02 7.8181744e-02 5.2037615e-01 7.0881844e-04 1.0413429e-01
 3.4687907e-02 9.9905819e-01 9.9949318e-01 9.7583157e-01 9.5710635e-01
 3.7118176e-01 9.9433935e-01 2.3426500e-01 9.9997914e-01 9.9984443e-01
 1.5352529e-05 6.6832638e-01 9.7086966e-01 9.9044591e-01 2.5865202e-06
 1.9997635e-01 8.8846827e-01 9.8389506e-01 5.3230524e-03 3.7803352e-03
 3.4525633e-02 9.1495693e-01 9.9456012e-03 6.9688559e-01 9.9944413e-01
 6.1913091e-01 9.9010819e-01 9.9828094e-01 4.7384366e-01 6.3026488e-02
 8.3991039e-01 9.6876931e-01 7.3958111e-01 6.9248348e-02 2.5946081e-02
 5.2668105e-07 9.0798646e-01 8.5298443e-01 3.1500024e-01 2.4319941e-01
 8.6967492e-01 9.5704472e-01 9.9155748e-01 8.3227456e-03 7.0655596e-01
 2.6866794e-03 9.8642790e-01 9.4235659e-01 9.9949801e-01 4.4463909e-01
 1.3257465e-01 2.1838751e-01 9.8782212e-01 2.9340717e-01 1.1486560e-02
 2.2572359e-01 2.3543835e-04 9.5426893e-01 8.0924070e-01 9.6714449e-01
 9.9990332e-01 1.0647327e-02 1.0767065e-04 9.9680638e-01 9.7281390e-01
 2.4183300e-01 9.9069649e-01 1.2224054e-01 6.4850241e-02 8.9593422e-01
 2.6813477e-02 5.2809024e-01 9.9997693e-01 1.2248850e-01 2.3952425e-03
 2.1109939e-02 1.8559816e-01 8.6851060e-01 7.2838783e-01 5.7552081e-01
 6.1816609e-05 3.5997123e-02 7.7216351e-01 2.1496117e-03 6.1425567e-04
 9.9661469e-01 9.9179035e-01 7.4327064e-01 9.9982178e-01 4.0802383e-01
 2.2904780e-05 4.3466449e-02 9.9859679e-01]  que se encuentra en:  7
Imagen de BD:  70  Imagen de consulta:  24 

Distancia:  3.8771958  al vector:  [4.0345788e-03 2.4087140e-06 8.3453500e-01 5.4091215e-04 2.1150708e-04
 2.2255083e-05 2.2693843e-02 9.9999833e-01 6.2816769e-02 9.9998403e-01
 9.9924427e-01 9.7545958e-01 9.9984515e-01 3.0764496e-01 6.9327217e-01
 9.8549986e-01 9.8293018e-01 1.8173158e-02 2.3213550e-05 9.5409185e-01
 9.9860203e-01 5.0390679e-01 3.1056629e-06 9.8764014e-01 9.0436041e-03
 9.9221772e-01 9.6800041e-01 9.9697733e-01 1.9834667e-02 9.9995947e-01
 8.7319286e-06 5.1947474e-01 2.4892868e-05 6.4917513e-05 4.8274261e-01
 2.0561060e-01 9.9999011e-01 9.9949479e-01 9.9999273e-01 6.9603825e-01
 5.6019425e-04 9.9927139e-01 9.3819273e-01 3.2373056e-01 9.9999666e-01
 3.6121041e-02 1.6737974e-01 9.9996996e-01 9.9996722e-01 1.0868590e-06
 8.3542717e-01 6.6501784e-01 9.9471879e-01 7.8655702e-01 2.3780021e-01
 2.7895570e-03 7.1416157e-01 9.6920717e-01 5.9467852e-03 9.9999893e-01
 9.9864590e-01 4.5423985e-02 6.6041619e-02 9.7433430e-01 1.1420792e-01
 9.6640575e-01 5.6976485e-01 9.5010525e-01 4.8103232e-08 7.4505800e-01
 3.2057618e-05 2.8974456e-01 9.8958445e-01 8.9471322e-01 1.7013282e-01
 1.3210306e-01 4.0298319e-01 9.9189663e-01 9.8269546e-01 9.3088746e-01
 1.8738687e-02 2.1556380e-01 7.1431428e-02 8.1446803e-01 9.9776256e-01
 9.9598753e-01 4.7605725e-05 1.4119118e-02 3.8822263e-02 9.7193164e-01
 6.4086914e-04 2.4557322e-02 8.6986929e-01 9.9828124e-01 9.9619031e-01
 9.9801642e-01 3.2725930e-04 9.9385524e-01 9.9976456e-01 6.8139732e-03
 7.1330601e-01 9.9995911e-01 4.1543404e-05 9.2846751e-03 9.9284291e-01
 7.8158379e-03 6.1786473e-02 9.8492515e-01 6.6838443e-01 4.0350652e-01
 1.8620491e-04 9.5323735e-01 5.3035319e-03 9.9855816e-01 9.8039299e-01
 1.4087519e-05 7.5712800e-04 8.3758032e-01 3.0492842e-03 8.8293648e-01
 9.9966180e-01 1.8944412e-02 9.0691316e-01 6.1386180e-01 8.9958817e-01
 1.4093816e-03 1.6586074e-01 9.9809748e-01]  en la posicion:  157  desde el vector:  [2.22895443e-02 1.18883214e-08 7.52197027e-01 3.12168717e-01
 8.87866139e-01 2.12943554e-03 2.95330584e-01 9.99985576e-01
 3.40347230e-01 9.98640060e-01 9.99963880e-01 6.21721804e-01
 9.99986470e-01 7.52506852e-01 9.93784666e-01 9.99912381e-01
 5.96966386e-01 4.11566198e-02 1.24871731e-04 9.94946003e-01
 4.66349185e-01 9.91942585e-01 1.08756446e-07 5.87402582e-01
 5.25007248e-02 3.64716738e-01 8.88218343e-01 9.97617960e-01
 8.10192227e-01 9.98040915e-01 3.83418798e-03 4.91571426e-03
 3.53704527e-05 1.00474060e-02 2.26649106e-01 3.98278236e-04
 9.99355018e-01 9.99654889e-01 9.99998927e-01 9.77588117e-01
 2.16677785e-02 1.56954259e-01 9.59393024e-01 4.14149135e-01
 9.99998450e-01 4.61763412e-01 3.52465808e-02 9.99999523e-01
 9.26412463e-01 9.15840346e-08 4.65002745e-01 1.79427356e-01
 9.96178210e-01 9.93661106e-01 4.95422781e-02 2.47131084e-06
 7.89832115e-01 6.41026855e-01 2.74121761e-04 9.99999642e-01
 9.98585224e-01 2.78547406e-03 9.72426116e-01 1.89819932e-03
 7.28864677e-07 8.27963710e-01 4.07744646e-02 1.62530661e-01
 2.92374068e-07 2.14339465e-01 2.92521715e-03 6.45396590e-01
 6.85948789e-01 9.99984264e-01 1.78804696e-02 8.42865467e-01
 4.19110060e-04 8.53273332e-01 2.15137661e-01 4.18021977e-01
 8.51753354e-03 9.92101192e-01 9.61669207e-01 9.97932792e-01
 9.44561124e-01 3.56313914e-01 5.93924224e-02 6.82791293e-01
 1.67560577e-03 9.08377171e-01 1.35362148e-03 1.61518157e-02
 9.83832657e-01 7.35111594e-01 9.98077035e-01 8.01159620e-01
 2.67273188e-03 9.77082133e-01 7.76033759e-01 2.31572390e-02
 1.31809592e-01 9.99957800e-01 7.52498527e-05 6.74933195e-04
 9.74448860e-01 2.17378139e-04 1.11765027e-01 4.44319814e-01
 2.88065135e-01 1.27801776e-01 1.35846619e-06 9.51337695e-01
 1.37324929e-02 9.92472172e-01 9.77414131e-01 1.99234873e-05
 2.21341848e-04 8.86391163e-01 1.46389008e-02 9.57967043e-01
 9.96949911e-01 9.68407094e-01 5.47396541e-01 9.92996097e-01
 5.64877629e-01 1.94202214e-01 4.22088683e-01 4.67457861e-01]  que se encuentra en:  8
Imagen de BD:  58  Imagen de consulta:  27 

Distancia:  3.3232903  al vector:  [5.5366778e-05 3.0416250e-03 9.1629136e-01 9.9939251e-01 4.2485896e-01
 2.4350966e-05 6.9179535e-02 9.5450580e-03 7.4151295e-01 9.9923527e-01
 9.9991012e-01 9.8473358e-01 9.7358072e-01 5.7381886e-01 2.2591069e-01
 9.9436677e-01 8.9979172e-03 1.0244727e-02 8.6158508e-01 3.6008126e-01
 9.9997473e-01 1.6941801e-01 5.4042923e-01 7.5745350e-01 2.2404127e-05
 1.6385883e-01 2.8161487e-01 7.0044667e-01 1.6963804e-01 9.9948370e-01
 7.9022843e-01 1.9616568e-01 1.1402011e-02 7.1914184e-01 7.2464281e-01
 1.2344718e-03 7.7148378e-03 6.1496854e-01 9.9970651e-01 3.1287193e-01
 9.5230639e-03 8.1531942e-01 5.9761679e-01 3.0703491e-01 2.5729537e-03
 9.9136794e-01 1.3912019e-01 5.2636325e-02 2.4742690e-01 5.1368493e-01
 1.4365312e-01 8.5486227e-01 1.2150637e-04 2.1553165e-01 2.7017009e-01
 8.6717916e-01 7.0034862e-01 2.0376295e-02 4.3780059e-01 7.2616339e-02
 6.1717814e-01 6.0731721e-01 9.9364376e-01 8.1830907e-01 7.7140927e-03
 9.4474727e-01 6.0110897e-01 5.9546381e-01 9.2358589e-03 6.3694352e-01
 1.0700792e-04 2.8266263e-01 9.9995077e-01 2.8162271e-01 4.3050164e-01
 3.6523318e-01 6.7301888e-05 4.3675900e-02 4.9361321e-01 4.9255490e-03
 6.2208432e-01 3.3248961e-03 6.8956614e-04 1.5391111e-01 9.2701882e-01
 9.3980038e-01 2.5501549e-03 9.9971372e-01 2.7781725e-04 9.7021127e-01
 8.7526292e-02 9.9971330e-01 8.7513590e-01 9.9941427e-01 9.6447462e-01
 1.2409687e-03 1.6836756e-01 9.9587274e-01 9.9998212e-01 6.8443024e-01
 8.4157038e-01 5.2108639e-01 8.9864284e-01 1.6021729e-04 9.2472148e-01
 4.8757792e-03 9.9723774e-01 1.4460087e-03 2.5565541e-01 6.7623883e-01
 2.0965934e-04 4.7277004e-01 3.2383475e-01 9.9703115e-01 5.9049094e-01
 7.4327707e-01 9.9995816e-01 9.1461742e-01 2.5417882e-01 9.9759066e-01
 6.1491835e-01 5.6275934e-02 8.5720134e-01 9.8187101e-01 9.9682581e-01
 9.9985850e-01 9.5244372e-01 9.9384403e-01]  en la posicion:  92  desde el vector:  [3.17454338e-04 1.49323046e-02 7.19969034e-01 9.99929428e-01
 1.64369047e-02 2.85801291e-03 3.15691888e-01 2.01093554e-02
 6.40106678e-01 9.99437213e-01 9.98416662e-01 6.04494333e-01
 9.77199793e-01 1.12430215e-01 8.57598603e-01 9.93864298e-01
 4.90844250e-04 2.26702809e-01 8.09282482e-01 2.96506107e-01
 9.98255849e-01 2.19449669e-01 5.54760158e-01 6.48548722e-01
 7.89800906e-05 3.34080577e-01 6.70908988e-02 2.36624628e-01
 4.27307785e-02 9.73704100e-01 1.14835322e-01 2.87061691e-01
 2.89361835e-01 3.88538480e-01 4.32229459e-01 4.25059497e-02
 5.54114580e-04 9.97980237e-01 9.97002959e-01 6.58337176e-01
 5.66759288e-01 6.03970766e-01 7.57546723e-01 9.43591654e-01
 6.65089488e-03 9.99100089e-01 6.40354753e-01 8.25713634e-01
 4.32082593e-01 3.07527423e-01 3.42992455e-01 5.30567825e-01
 1.29911304e-03 2.00350255e-01 9.88379955e-01 2.08512485e-01
 4.58240002e-01 1.69054270e-02 9.13776278e-01 1.90007687e-03
 9.74538445e-01 5.03074646e-01 9.81510282e-01 6.80171549e-01
 1.94827020e-01 9.70074654e-01 7.57292032e-01 4.42593127e-01
 3.72126698e-03 8.99312556e-01 8.76805186e-03 8.04598451e-01
 9.98722434e-01 8.88912320e-01 8.43347669e-01 8.42060864e-01
 2.55104899e-03 6.70327604e-01 9.78232026e-02 2.34474838e-02
 2.79158473e-01 9.19491053e-04 6.35492802e-03 5.82396984e-04
 6.98309004e-01 9.97644246e-01 2.27112472e-02 9.99903202e-01
 1.55980885e-02 9.94843006e-01 9.25066471e-01 9.99959648e-01
 2.51568079e-01 9.98233676e-01 9.51384902e-01 8.22901726e-04
 2.07364559e-03 8.82132769e-01 9.99213517e-01 9.93781388e-01
 9.58267808e-01 6.54057503e-01 3.63980711e-01 1.95448399e-02
 9.37728107e-01 6.55163527e-02 7.58862913e-01 1.95357203e-03
 6.29738569e-02 1.41573191e-01 1.07103288e-02 3.91033590e-02
 8.84757638e-01 9.44565594e-01 4.67143953e-01 7.54607022e-01
 9.99967217e-01 7.59817183e-01 6.56810105e-02 9.99415219e-01
 9.98805046e-01 2.23665714e-01 6.40851259e-03 9.67627406e-01
 8.41991901e-01 9.99838591e-01 8.74542594e-01 9.75578725e-01]  que se encuentra en:  9
Imagen de BD:  182  Imagen de consulta:  33 

Distancia:  3.7385645  al vector:  [9.98146772e-01 9.75743890e-01 9.77753758e-01 4.03081208e-01
 3.00827622e-02 9.96288657e-01 1.12237304e-01 2.15590000e-04
 9.92758155e-01 2.07084777e-05 9.98862147e-01 5.24285734e-02
 3.29354346e-01 7.98855662e-01 2.24814624e-01 9.99999762e-01
 4.80244666e-01 9.99824703e-01 9.80538607e-01 3.18536460e-02
 3.06530595e-02 9.58769798e-01 7.70836473e-01 7.12308288e-03
 9.99978304e-01 5.06865680e-02 1.03881955e-03 9.99964178e-01
 4.97812182e-01 4.69446182e-04 9.99739230e-01 1.37557745e-01
 1.83969736e-03 5.93643785e-01 7.71453142e-01 4.41373146e-07
 9.95677471e-01 4.00891900e-03 4.78067696e-02 9.51151013e-01
 9.98902857e-01 3.86165082e-01 1.54382586e-02 9.59178329e-01
 9.96323228e-01 3.54093730e-01 3.50927114e-02 6.28272891e-02
 4.05174941e-01 9.59055066e-01 2.86800951e-01 5.48691869e-01
 9.99976039e-01 3.20756853e-01 4.88500837e-06 2.87722960e-05
 1.22323487e-04 1.73676610e-02 9.48147297e-01 8.10744107e-01
 3.20825279e-01 9.96899664e-01 3.71065438e-01 8.30666661e-01
 2.85947372e-05 3.66521597e-01 9.13498759e-01 9.38611507e-01
 9.92687523e-01 2.13380933e-01 9.99783397e-01 2.87825704e-01
 3.39865685e-04 5.50682545e-01 9.86551642e-01 3.95899147e-01
 9.99989033e-01 9.92590189e-03 4.49891061e-01 6.37384415e-01
 9.23957407e-01 1.00000000e+00 9.99212146e-01 9.99978662e-01
 9.99198854e-01 7.87036419e-01 9.99997854e-01 3.38362813e-01
 1.66563916e-06 4.49377298e-03 7.56747127e-02 6.66765436e-06
 1.27154797e-01 2.93791294e-04 3.59198213e-01 1.32272393e-01
 9.99095559e-01 5.15694337e-05 4.43926454e-03 5.08712649e-01
 8.92347395e-01 1.96084738e-01 8.36243510e-01 9.97169495e-01
 1.58318430e-01 1.41470125e-07 6.27218008e-01 1.93016589e-01
 9.87894297e-01 5.79182208e-02 6.03317618e-02 6.21892214e-01
 9.78888929e-01 1.98945284e-01 5.28013349e-01 9.96811390e-01
 2.73902515e-05 2.01058179e-01 9.91029561e-01 2.05665827e-04
 4.93079424e-04 9.54260170e-01 3.74567151e-01 8.92488956e-02
 1.34199126e-05 9.62600470e-01 9.94112074e-01 1.52379274e-04]  en la posicion:  133  desde el vector:  [9.8505127e-01 2.4307966e-03 5.0885147e-01 7.0465088e-02 8.0811679e-03
 6.6779506e-01 5.3990185e-02 2.4205059e-02 9.8510540e-01 2.2733808e-03
 9.9988472e-01 5.2971661e-02 9.9978340e-01 8.9293861e-01 8.1975329e-01
 9.9999750e-01 3.5368031e-01 6.9338810e-01 9.2620635e-01 8.5730255e-02
 1.9709617e-02 9.9997735e-01 6.6443175e-02 4.4203970e-01 9.9927896e-01
 4.0299824e-01 2.9978245e-02 9.9975562e-01 4.1929641e-01 1.8143046e-01
 9.9990100e-01 2.0947844e-02 7.7971854e-05 9.9323297e-01 9.7000659e-01
 1.4868668e-06 9.9040097e-01 3.8400292e-04 9.8820865e-01 1.7588702e-01
 7.2451729e-01 3.8066530e-01 3.4864551e-01 6.3636124e-02 9.9907619e-01
 2.5892848e-01 1.6935366e-01 6.2189198e-01 2.1666288e-03 5.1155347e-01
 5.1043409e-01 9.2923230e-01 9.9978292e-01 2.4494871e-01 1.6481189e-06
 1.2493730e-03 1.0751009e-02 4.3716562e-01 9.4837236e-01 9.7428846e-01
 7.3523676e-01 9.6795201e-01 9.8317939e-01 8.7259048e-01 8.1026456e-06
 6.0570836e-01 7.0461160e-01 7.7118748e-01 6.1392784e-04 9.7868299e-01
 7.5519383e-01 4.0692985e-02 2.9743344e-02 2.5135607e-01 8.0351448e-01
 9.1607970e-01 8.7479389e-01 6.3827783e-02 5.1102138e-01 1.2578973e-01
 9.2527258e-01 9.9999976e-01 9.9693483e-01 9.9998504e-01 9.8854649e-01
 5.4565787e-01 9.9987948e-01 5.3805983e-01 5.9871414e-05 4.6064675e-02
 3.4990907e-04 2.0498037e-04 6.6675079e-01 6.6952407e-03 1.7881581e-01
 1.3966230e-01 9.9893796e-01 3.2734877e-01 4.6387613e-03 8.1103361e-01
 9.4916242e-01 8.7739611e-01 1.4505073e-01 7.8009748e-01 4.1112882e-01
 1.3516378e-06 9.9552262e-01 3.5980782e-01 6.1499637e-01 5.9745193e-02
 7.9436948e-05 1.0595709e-01 9.8854804e-01 3.8630185e-01 7.7557802e-02
 9.6158588e-01 1.9931793e-04 2.8175801e-01 4.7099292e-01 2.8159675e-01
 7.2821975e-04 2.9203057e-01 3.7310603e-01 9.0902501e-01 2.5286108e-02
 8.9635646e-01 9.9708831e-01 3.4204125e-03]  que se encuentra en:  10
Imagen de BD:  36  Imagen de consulta:  36 

Distancia:  4.3776817  al vector:  [9.99999642e-01 1.01094544e-02 9.94495988e-01 1.19863861e-08
 2.23857462e-02 9.99946117e-01 4.22838330e-03 9.96616542e-01
 4.10945386e-01 1.07703301e-04 3.80822718e-02 3.34029317e-01
 8.85500312e-02 7.22527623e-01 1.54394150e-01 8.32378507e-01
 9.98713970e-01 8.56518745e-04 7.56450653e-01 9.59803760e-01
 6.34860669e-07 9.99999046e-01 6.06895864e-01 8.98534000e-01
 1.00000000e+00 9.29737270e-01 9.53370452e-01 8.98373961e-01
 8.72107387e-01 3.02016735e-04 9.93461311e-01 9.81120050e-01
 1.25454366e-02 4.26601380e-01 3.67122412e-01 2.33908415e-01
 9.99999642e-01 3.67134771e-06 9.92204070e-01 3.04065347e-02
 2.53961176e-01 8.79866719e-01 9.86621797e-01 1.14978491e-06
 9.99322057e-01 2.47225642e-01 2.75294483e-02 7.39622116e-02
 1.41475707e-01 9.99176741e-01 9.98797059e-01 9.98108208e-01
 9.99999881e-01 9.06965494e-01 4.38088791e-05 2.68506706e-01
 1.42119527e-02 9.81829524e-01 3.08262408e-02 9.99897361e-01
 9.98080611e-01 3.74169111e-01 1.08964505e-05 8.89548659e-02
 2.23267257e-01 9.99662638e-01 9.82645035e-01 9.17854548e-01
 2.72971392e-03 1.44596398e-02 9.99993801e-01 5.34311730e-05
 1.28328800e-04 9.98786628e-01 1.28485590e-01 9.43619728e-01
 9.99877095e-01 9.31532383e-01 9.71154451e-01 9.76670861e-01
 9.95511770e-01 9.99998212e-01 2.77183950e-02 9.99938786e-01
 1.28599912e-01 5.35955071e-01 9.95201588e-01 7.05347239e-11
 9.98802662e-01 5.99005818e-03 4.23256381e-08 1.07398928e-05
 1.00000000e+00 7.89379555e-06 9.47114706e-01 9.98058438e-01
 9.99987006e-01 9.99525905e-01 9.41239591e-07 1.14705785e-04
 9.69138145e-01 8.55491817e-01 4.61027920e-01 9.97360468e-01
 1.25339627e-03 3.33402991e-01 2.77806103e-01 9.98463035e-01
 6.46791458e-02 9.45216179e-01 2.68101305e-01 8.90583396e-01
 8.30176214e-05 1.01473629e-02 1.42509282e-01 3.89602602e-01
 8.40896917e-08 9.21428204e-04 8.51096153e-01 1.18659139e-02
 2.73271635e-05 9.99030232e-01 6.59121573e-02 1.43823028e-03
 9.86175299e-01 2.85974145e-03 9.90478396e-02 2.54458189e-03]  en la posicion:  148  desde el vector:  [9.98887062e-01 3.97330523e-03 9.92626667e-01 8.52853060e-04
 6.99292719e-02 8.85511756e-01 3.37389112e-03 9.92644548e-01
 9.39880252e-01 7.54332602e-01 3.10599804e-04 9.69738483e-01
 9.69497561e-01 3.70073497e-01 8.41712952e-01 6.62186742e-03
 3.93615693e-01 2.50101089e-04 9.81975615e-01 9.95676994e-01
 2.89932723e-06 9.99841988e-01 9.86434221e-01 9.99704957e-01
 9.99859214e-01 1.38789505e-01 9.99890327e-01 8.76157880e-01
 2.45898962e-02 5.38138747e-02 4.19030219e-01 4.19015646e-01
 2.39289850e-01 9.99993205e-01 2.00411916e-01 9.74847555e-01
 9.63967085e-01 1.91103391e-05 9.98808980e-01 9.55052555e-01
 6.66039884e-02 9.93655086e-01 9.41735566e-01 1.04483588e-05
 6.59403920e-01 9.82432008e-01 3.47743034e-01 4.40946221e-03
 2.16069816e-06 9.95933294e-01 9.98597145e-01 9.91774738e-01
 9.94313419e-01 8.50970387e-01 5.58191750e-05 9.98945057e-01
 9.41410065e-02 9.64909792e-01 1.16819769e-01 5.87101698e-01
 9.79018986e-01 4.94428426e-01 2.68942535e-01 2.95836985e-01
 7.96433926e-01 5.51150441e-01 9.88335013e-01 9.75637197e-01
 6.49392605e-03 3.43495309e-02 9.89183784e-01 9.97912288e-02
 1.88781321e-02 9.85929489e-01 2.00797826e-01 9.81179953e-01
 9.97758746e-01 8.29103589e-01 9.04248655e-02 1.01930559e-01
 1.08573705e-01 9.22465205e-01 6.24054670e-03 9.99791324e-01
 4.41582412e-01 9.90022659e-01 6.41714036e-02 7.20947981e-04
 9.99952972e-01 6.46710396e-04 2.26521492e-03 1.30662739e-01
 9.99925137e-01 1.53660774e-04 8.59429777e-01 9.74331617e-01
 9.99993205e-01 9.99985635e-01 2.78355183e-05 4.90409136e-03
 9.62002277e-02 1.46416426e-02 9.99518514e-01 9.43848610e-01
 8.35276902e-01 9.39611495e-01 4.38961864e-01 9.49970961e-01
 2.28720903e-03 6.28466547e-01 6.19100392e-01 9.86745000e-01
 1.01298988e-02 3.18006843e-01 7.65836358e-01 9.97852623e-01
 5.35991788e-03 3.63470972e-01 7.92523503e-01 3.66333425e-01
 2.12525410e-05 9.84233975e-01 1.67947114e-02 9.11630154e-01
 9.98860180e-01 3.27505171e-02 9.03091550e-01 4.70966101e-04]  que se encuentra en:  11
Imagen de BD:  5  Imagen de consulta:  5 

Distancia:  4.317569  al vector:  [8.31883967e-01 1.31253719e-01 9.96282995e-02 9.99924898e-01
 9.97981548e-01 6.64156556e-01 9.12544847e-01 7.48753548e-04
 3.38566601e-02 9.02542591e-01 9.64911938e-01 5.77101707e-01
 9.99990940e-01 5.52174330e-01 6.13579035e-01 2.77744979e-01
 2.96026468e-04 4.44033980e-01 9.11690235e-01 8.27783108e-01
 3.29005718e-02 3.19127083e-01 9.38889384e-02 1.17504209e-01
 1.33177638e-03 1.79219604e-01 7.20059872e-03 8.35087895e-03
 9.84159350e-01 3.58924866e-02 4.27827030e-01 2.76563704e-01
 8.71813297e-01 9.93619084e-01 7.86521018e-01 6.97599173e-01
 6.54488802e-04 9.94722128e-01 9.99810338e-01 6.95199907e-01
 5.84229648e-01 9.26248789e-01 8.76863837e-01 9.21610177e-01
 1.05861127e-02 9.94380951e-01 3.43772292e-01 9.69505072e-01
 1.71720982e-03 8.84651065e-01 9.50620770e-01 5.71880102e-01
 1.87858701e-01 9.78163540e-01 9.90286887e-01 4.21079993e-03
 9.69512343e-01 8.91312122e-01 6.72017336e-01 2.84008086e-02
 9.31148410e-01 2.83435225e-01 3.10995102e-01 1.83212340e-01
 4.99611020e-01 7.50647724e-01 3.90843272e-01 9.06686783e-02
 1.60885096e-01 9.77447748e-01 7.77712107e-01 4.93917018e-01
 2.06990689e-01 9.61795211e-01 1.20783538e-01 9.06236589e-01
 7.78742674e-07 2.72487104e-02 1.19175434e-01 1.20675981e-01
 3.22036356e-01 1.99463695e-01 2.90614367e-03 2.35155225e-03
 2.17843503e-01 2.11600333e-01 9.98374224e-01 9.97816205e-01
 1.73214942e-01 9.99868393e-01 9.62696493e-01 9.98851418e-01
 5.97783267e-01 4.09227163e-01 5.80241144e-01 3.74197960e-04
 7.04467297e-04 7.86645412e-01 2.83657163e-01 1.34868026e-02
 6.14117920e-01 9.96402323e-01 4.92635071e-02 6.47303760e-01
 8.19357514e-01 3.82996678e-01 3.52799058e-01 6.84469938e-04
 3.44984233e-02 9.42001343e-01 6.86371326e-02 3.59500706e-01
 1.62490875e-01 3.54219913e-01 9.66942787e-01 9.80767012e-01
 9.77979183e-01 4.85506028e-01 7.21800148e-01 9.99589682e-01
 9.98500466e-01 9.31216836e-01 4.61623073e-03 9.22599673e-01
 9.58767891e-01 9.99935925e-01 9.96496379e-01 2.76416540e-04]  en la posicion:  79  desde el vector:  [9.76003170e-01 2.55238920e-01 6.51729405e-02 9.73235250e-01
 2.07021594e-01 9.97746110e-01 6.30547464e-01 2.38075852e-03
 5.15011251e-02 2.31017172e-01 9.49773192e-01 3.80904108e-01
 9.96419549e-01 9.61879373e-01 9.48436975e-01 8.52577686e-01
 1.18550092e-01 6.71813071e-01 7.55142570e-01 3.31449807e-01
 6.32908940e-03 9.78120804e-01 1.36603713e-02 2.61486471e-02
 7.43116736e-01 3.92246366e-01 3.27908993e-03 1.89585209e-01
 9.16814089e-01 2.17482448e-03 9.63130653e-01 7.57157803e-04
 5.90727925e-02 9.89827752e-01 6.59253001e-01 2.87574828e-02
 5.08295000e-02 4.98170167e-01 9.80003238e-01 1.40547454e-02
 9.61431980e-01 4.51226234e-02 8.45379293e-01 6.37072325e-01
 6.26704514e-01 7.53003955e-01 5.46838939e-02 9.96536970e-01
 1.83932483e-02 7.95021892e-01 9.91936207e-01 9.36831713e-01
 9.91171658e-01 2.49163657e-01 6.37345433e-01 3.02109122e-03
 2.47135162e-01 1.65015757e-01 5.05332708e-01 4.37444806e-01
 1.58106357e-01 2.65959084e-01 6.56590760e-02 2.75538564e-02
 2.97497511e-01 7.19189167e-01 8.41489136e-01 9.08588290e-01
 6.49605393e-01 8.42555761e-01 9.97506261e-01 1.08388454e-01
 4.59807813e-02 3.92892182e-01 1.99223429e-01 2.72885799e-01
 1.02439404e-01 3.62304151e-02 1.20432496e-01 1.73206031e-02
 7.63381422e-02 9.97479677e-01 3.52188557e-01 8.12017620e-02
 1.93361878e-01 2.63249189e-01 9.98931050e-01 9.36952293e-01
 1.35312706e-01 9.91634011e-01 8.68520379e-01 3.62870693e-01
 3.50237131e-01 3.15053761e-02 4.46739316e-01 1.58423781e-02
 4.63562280e-01 7.84707308e-01 2.43512392e-02 9.37967062e-01
 8.53052258e-01 9.95469928e-01 2.18342841e-02 9.65805590e-01
 7.97421753e-01 1.19244188e-01 7.69165456e-02 4.83191013e-02
 7.67868876e-01 6.15312755e-01 4.33894694e-02 3.67893994e-01
 5.49676597e-01 5.14030039e-01 8.57116342e-01 9.82976437e-01
 9.87566710e-02 9.18245554e-01 7.98355460e-01 9.91332650e-01
 8.01314235e-01 7.78293788e-01 2.88151741e-01 9.81140137e-02
 2.58271873e-01 9.86668348e-01 9.82378364e-01 3.63072753e-03]  que se encuentra en:  12
Imagen de BD:  170  Imagen de consulta:  36 

Distancia:  2.8275175  al vector:  [1.00000000e+00 9.99999404e-01 2.58409262e-01 9.87183213e-01
 9.02936459e-01 1.00000000e+00 3.11356187e-01 7.40735615e-08
 2.82640755e-02 1.53273344e-04 5.29827412e-06 3.41131926e-01
 1.25225079e-05 7.23154247e-02 1.33885443e-02 9.99636114e-01
 3.58554721e-03 9.99959111e-01 1.00000000e+00 1.98939890e-01
 5.01018498e-08 5.16125560e-03 8.30826521e-01 4.10065604e-06
 1.00000000e+00 1.00714862e-01 1.82485728e-05 9.99999702e-01
 2.65702605e-01 4.65948027e-08 9.96143222e-01 4.81912941e-01
 5.31558871e-01 9.99999523e-01 3.45603824e-02 4.27991152e-04
 4.01502073e-01 1.26493038e-09 2.11667375e-05 9.96820450e-01
 9.95403349e-01 7.79199600e-01 2.40299255e-01 9.95810151e-01
 1.12986565e-03 1.00000000e+00 1.44100994e-01 7.19974995e-01
 1.49475329e-06 1.00000000e+00 7.16924548e-01 7.85621941e-01
 1.32026494e-01 5.32136798e-01 3.08549079e-06 3.40297759e-01
 9.76029217e-01 3.11031640e-02 9.98924613e-01 2.42801607e-02
 5.25357246e-01 9.89634991e-01 5.61660528e-03 4.29236889e-03
 8.70386064e-01 1.39355659e-03 2.89050549e-01 7.27202415e-01
 9.99999046e-01 2.61119440e-05 1.00000000e+00 9.43111598e-01
 8.39371012e-07 8.42819571e-01 4.63380277e-01 1.41096652e-01
 9.99992371e-01 1.80976897e-01 1.69088840e-02 3.51952016e-02
 7.68145919e-02 9.96387362e-01 7.93009996e-02 1.00000000e+00
 4.05435562e-01 1.59087777e-02 9.99995947e-01 9.40522492e-01
 2.09583044e-02 3.38984610e-05 9.99709666e-01 3.87737948e-08
 3.86466414e-01 2.13816200e-08 9.83201385e-01 3.03685665e-04
 1.00000000e+00 1.39578015e-05 5.01693620e-08 1.58175826e-03
 1.71794981e-01 5.88450134e-02 9.77952957e-01 1.00000000e+00
 9.34997201e-03 2.72676349e-03 9.45904016e-01 2.93284655e-04
 6.82394266e-01 2.39241123e-03 1.00000000e+00 3.74116600e-02
 9.56762731e-01 6.46233559e-01 9.44230914e-01 1.00000000e+00
 9.60471869e-01 1.99481845e-03 9.76970792e-03 3.16384435e-03
 4.22208970e-08 5.48160434e-01 3.78954947e-01 2.30368674e-02
 8.41230154e-04 9.99848962e-01 1.00000000e+00 1.45465136e-04]  en la posicion:  142  desde el vector:  [9.9876893e-01 9.9676895e-01 1.8968788e-01 9.9994946e-01 9.0797341e-01
 9.9999213e-01 8.1560826e-01 3.5670400e-04 8.9057684e-02 5.8060884e-04
 2.0359576e-02 4.8171264e-01 4.5439542e-06 6.4969301e-02 1.1286318e-02
 1.0000000e+00 8.0249709e-05 9.8529017e-01 9.9996912e-01 2.5416344e-01
 1.0354519e-03 7.0499480e-03 4.8735753e-01 9.2297978e-07 9.9999106e-01
 3.1648517e-02 1.7766739e-06 1.0000000e+00 8.2939190e-01 1.0180392e-04
 9.7990966e-01 2.4820548e-01 9.5758438e-03 9.9704766e-01 7.2192973e-01
 3.2131535e-07 9.4333333e-01 8.9186660e-06 1.5348196e-04 9.9852288e-01
 9.7039247e-01 9.9695486e-01 1.3421699e-01 9.8626602e-01 3.4295017e-01
 9.9999976e-01 6.7349893e-01 6.0451710e-01 5.4437518e-03 9.9988103e-01
 9.1972315e-01 3.7473720e-01 1.2663385e-01 2.1831742e-01 6.7651408e-06
 5.4490864e-03 3.4612072e-01 6.6578686e-03 9.9641311e-01 7.3292351e-01
 7.9148114e-03 9.3906581e-01 4.8276156e-02 4.6324900e-01 6.0901046e-04
 1.3731122e-02 2.3545372e-01 9.8124790e-01 9.8698175e-01 8.6483359e-04
 9.9932361e-01 9.2259628e-01 4.2545527e-02 6.7547500e-01 1.9840586e-01
 2.3951828e-03 9.9990398e-01 4.4039518e-02 2.8401613e-04 8.0758893e-01
 4.0997356e-01 9.9425483e-01 7.1360475e-01 9.9999386e-01 3.0008081e-01
 1.0463715e-02 9.3080318e-01 9.0979624e-01 4.8107181e-06 1.2595057e-03
 5.6239390e-01 6.1821411e-06 1.0774329e-01 3.1210482e-03 8.6055887e-01
 3.6401259e-05 9.9998212e-01 7.3063374e-04 1.7571449e-04 2.9816568e-01
 3.7176591e-01 6.4180893e-01 6.8402004e-01 9.9994504e-01 1.0623264e-01
 1.4245486e-06 9.9987590e-01 1.8589975e-05 8.2809418e-01 4.2703807e-02
 9.9981356e-01 3.0396342e-02 9.6096432e-01 1.3341326e-01 8.6089385e-01
 9.9997717e-01 9.9590379e-01 4.5650303e-03 1.8509620e-01 1.2855053e-02
 1.8275447e-05 5.0775129e-01 9.9709105e-01 4.5387745e-03 1.1955798e-03
 9.9995357e-01 9.9993265e-01 5.0222480e-01]  que se encuentra en:  13
Imagen de BD:  44  Imagen de consulta:  44 

Distancia:  4.2678666  al vector:  [1.0078847e-03 9.9950397e-01 8.0483460e-01 3.8963407e-02 3.2403344e-01
 5.0640206e-06 3.6851466e-03 9.9462068e-01 9.9998182e-01 9.7209263e-01
 2.7425939e-01 5.3233385e-01 1.5264750e-04 5.1556277e-01 7.1041608e-01
 8.1535518e-02 9.3219554e-01 9.9821550e-01 1.1165320e-04 2.2092760e-02
 9.9984539e-01 2.7411789e-02 9.9995989e-01 9.9959052e-01 2.5178492e-03
 5.3659081e-04 9.7313619e-01 1.2645188e-01 3.4205616e-03 9.9999714e-01
 6.9386762e-01 9.9438488e-01 9.9379474e-01 8.1255857e-05 4.2716575e-01
 7.4658859e-01 2.8201804e-01 9.9247563e-01 9.0062445e-01 8.2252491e-01
 9.9945349e-01 9.7734678e-01 8.6192536e-01 9.9760562e-01 1.2169778e-03
 4.3960214e-03 9.1791093e-01 3.3401682e-06 9.9518973e-01 9.8375410e-02
 4.4277295e-01 9.7866297e-01 3.0323386e-02 7.8928590e-02 8.9776385e-01
 9.9376076e-01 2.0200908e-03 6.8175256e-02 9.8732531e-01 2.4984777e-03
 2.1739799e-01 9.8682183e-01 8.3673000e-01 1.7814320e-01 1.3111255e-01
 6.3860035e-01 1.3810977e-01 9.0443963e-01 9.9796474e-01 1.2918115e-02
 5.0375838e-05 4.9786785e-01 9.9998897e-01 2.1089691e-01 3.3098036e-01
 9.4635105e-01 9.8211920e-01 9.8514521e-01 5.7675451e-02 2.0838264e-01
 4.2987275e-01 8.6798048e-01 2.4921986e-01 1.0155618e-02 6.9442868e-01
 1.0688493e-01 1.0804176e-02 7.5755870e-01 5.0848180e-01 9.7259879e-04
 1.1177957e-03 9.9878371e-01 9.6700424e-01 9.9995685e-01 1.8014270e-01
 9.9382949e-01 2.8901935e-02 1.4858246e-03 9.9962604e-01 2.0031267e-01
 8.5823268e-02 7.8064203e-04 9.9996781e-01 1.6197562e-04 1.0891914e-01
 7.8035843e-01 3.2185912e-02 9.7652006e-01 1.8829265e-01 1.4092055e-01
 4.5946017e-01 1.5901718e-01 9.7104990e-01 5.5617511e-01 2.8052801e-01
 4.2822957e-04 9.9663782e-01 6.6240251e-01 8.3541870e-04 8.9100003e-04
 9.9853718e-01 4.9589881e-01 6.5078831e-01 7.4725211e-01 9.9518836e-01
 4.2337120e-02 7.1858764e-03 9.8024470e-01]  en la posicion:  151  desde el vector:  [2.20682621e-02 9.93939400e-01 2.19235152e-01 1.43094361e-02
 6.48135483e-01 1.78678347e-05 1.36976182e-01 9.92439032e-01
 9.99985456e-01 9.99617279e-01 9.36270356e-01 9.57688689e-01
 2.04165488e-01 8.61895800e-01 1.45862103e-01 7.73593307e-01
 9.99938667e-01 9.99997079e-01 5.65409660e-04 7.44939327e-01
 9.99984145e-01 4.69211042e-02 9.66270447e-01 9.99815643e-01
 2.41577625e-04 5.43672919e-01 9.98846769e-01 6.53058887e-02
 1.58972740e-02 9.99979317e-01 1.05759203e-02 8.86284232e-01
 5.93492985e-01 2.01821327e-04 2.74482518e-01 1.19008929e-01
 8.45923960e-01 9.93900836e-01 3.40496868e-01 9.47409391e-01
 9.99988317e-01 5.88050902e-01 9.92999911e-01 9.99956906e-01
 1.56805009e-01 6.01254651e-05 1.82522893e-01 2.63019502e-02
 9.99824166e-01 9.24590826e-02 7.62863994e-01 7.54712582e-01
 9.51521039e-01 3.18000615e-01 2.86157727e-02 7.41848528e-01
 1.18938515e-05 9.36264992e-02 6.54910088e-01 2.93536782e-02
 1.55991226e-01 3.99100155e-01 9.81528044e-01 2.81918943e-02
 4.71524775e-01 4.06109691e-02 2.84323096e-03 9.72723842e-01
 9.93833899e-01 9.26297307e-01 1.21958197e-04 8.36148262e-01
 9.99438643e-01 9.50642526e-01 1.08339310e-01 9.96137798e-01
 9.99804139e-01 1.85901135e-01 6.59904480e-02 1.37494683e-01
 1.21635348e-01 9.66413796e-01 9.62066412e-01 8.07541609e-01
 2.05731988e-02 7.02134669e-02 3.54653597e-03 9.95749712e-01
 2.64733434e-01 2.67803669e-04 1.20356351e-01 2.49059498e-02
 9.96985197e-01 9.98487711e-01 1.86639249e-01 9.99669671e-01
 1.80341601e-02 3.50830965e-06 9.99995947e-01 1.54129028e-01
 8.98277938e-01 1.73169225e-01 8.46683681e-01 2.56228447e-03
 4.05242205e-01 1.51094854e-01 1.83478892e-02 9.99649644e-01
 2.92594314e-01 7.04560339e-01 7.17931628e-01 6.63738430e-01
 8.85844707e-01 8.97982717e-02 9.77003694e-01 3.69924307e-03
 6.04618251e-01 9.01344478e-01 4.92402315e-01 1.59912179e-06
 9.96241093e-01 2.40350753e-01 4.42349821e-01 9.85807538e-01
 2.34534740e-02 1.36077404e-04 9.68349099e-01 9.99800861e-01]  que se encuentra en:  14
Imagen de BD:  52  Imagen de consulta:  45 

Distancia:  3.3335667  al vector:  [8.9867866e-01 5.7715178e-04 9.8890555e-01 1.1376917e-02 5.5133855e-01
 3.1536818e-04 2.1760046e-01 9.9930847e-01 1.6455361e-01 9.8523420e-01
 8.7162423e-01 8.5380316e-02 9.9725604e-01 2.7680665e-02 4.8358113e-01
 1.8919417e-01 9.4827402e-01 2.4420023e-04 7.4731082e-02 8.6411119e-01
 9.8004687e-01 4.5949394e-01 6.7329139e-02 9.9189365e-01 7.2491300e-01
 9.3919027e-01 9.8358262e-01 7.3674703e-01 7.4811375e-01 9.9982774e-01
 6.9818318e-01 3.7855268e-01 1.2089822e-01 3.6858839e-01 7.7280080e-01
 7.1583354e-01 9.9583232e-01 2.6800960e-02 9.9514270e-01 9.6319580e-01
 1.4669169e-05 9.5549327e-01 9.5638084e-01 2.5339007e-02 9.9978107e-01
 6.1989522e-01 2.0732462e-02 4.7898573e-01 3.8033646e-01 1.6402215e-02
 5.6319374e-01 9.3886894e-01 7.1917415e-01 4.5747891e-01 7.7439249e-03
 6.8607068e-01 9.9984378e-01 1.8448204e-02 1.6957930e-01 9.9984765e-01
 9.2471033e-02 5.2094615e-01 8.4770346e-01 7.5733066e-01 1.9132197e-02
 9.8805529e-01 7.7811015e-01 5.0840586e-02 1.0038477e-04 4.2322949e-01
 7.8528778e-05 7.1323925e-01 1.9727677e-02 3.5124481e-02 2.1669179e-02
 6.6564906e-01 3.1879246e-03 9.7533172e-01 3.8082689e-01 5.7264990e-01
 6.5812826e-01 2.0829847e-01 6.1876333e-01 9.5243824e-01 1.4730632e-02
 4.6086273e-01 4.2153624e-01 7.8815222e-03 7.5019097e-01 4.4829857e-01
 5.3985208e-02 3.0774474e-03 8.8995570e-01 1.0539183e-01 9.5160067e-01
 9.5582187e-01 9.2801964e-01 9.8564243e-01 9.0315723e-01 6.2200701e-01
 9.8870921e-01 9.9755907e-01 2.9505688e-01 6.1059600e-01 9.6785998e-01
 7.5160170e-01 9.8444521e-01 9.9248558e-01 9.3685490e-01 9.4432384e-02
 2.8708696e-02 6.0435796e-01 4.7479570e-03 7.2254288e-01 8.5411835e-01
 2.7914524e-02 2.4710864e-01 4.2867422e-02 2.0053446e-01 3.5254365e-01
 2.7967155e-02 9.8091400e-01 1.6959241e-01 6.3412988e-01 9.9431765e-01
 3.6579937e-02 7.1125299e-02 9.8389375e-01]  en la posicion:  144  desde el vector:  [7.68514872e-01 6.48269057e-03 9.47821856e-01 3.58021259e-03
 1.10805362e-01 2.95078754e-03 6.55424595e-01 9.99368191e-01
 9.30198729e-02 9.20016289e-01 9.74754095e-01 7.20391035e-01
 9.95588481e-01 2.77957320e-03 5.92968822e-01 4.20704484e-02
 9.94782507e-01 2.89621949e-03 1.62750483e-03 1.06407225e-01
 8.90657544e-01 9.64478135e-01 1.38562322e-02 9.78753686e-01
 8.04176390e-01 9.20285583e-01 7.96460271e-01 1.44267380e-01
 7.11075783e-01 9.99617755e-01 1.69589460e-01 6.85850084e-01
 3.69756818e-01 1.34759575e-01 6.18094921e-01 6.70591235e-01
 9.99965072e-01 5.13400316e-01 9.99449611e-01 7.59796202e-02
 2.17020512e-04 6.46117866e-01 9.85089779e-01 7.03469217e-02
 9.98154521e-01 9.06929374e-03 4.03097272e-02 9.54948366e-01
 9.45069790e-01 1.52400732e-02 4.63604629e-01 8.59559059e-01
 9.50690925e-01 2.72549987e-02 1.74483806e-01 9.22613621e-01
 9.97768939e-01 1.16109848e-02 1.98695928e-01 9.99540448e-01
 9.35586691e-02 1.18438959e-01 2.09684312e-01 1.74598098e-02
 4.68749940e-01 9.71361458e-01 6.61680102e-02 6.53863907e-01
 2.14725733e-04 3.48936945e-01 7.31527805e-04 7.89296567e-01
 2.27372169e-01 3.22425365e-03 1.50264204e-02 8.99441600e-01
 1.37376785e-03 9.33011413e-01 5.69505453e-01 9.38171268e-01
 8.63558233e-01 5.48750043e-01 5.85784554e-01 9.30260420e-01
 8.93205404e-02 3.79413366e-03 2.12991625e-01 4.06032801e-03
 7.38297224e-01 9.20432210e-01 2.83025205e-02 2.20271945e-02
 9.88192677e-01 7.91440368e-01 2.84382403e-02 9.92281914e-01
 7.58653998e-01 9.78638887e-01 8.68239820e-01 8.18293631e-01
 9.40172136e-01 9.98722672e-01 4.68888581e-02 4.57712710e-02
 4.40494210e-01 9.10548091e-01 8.97318840e-01 9.99565959e-01
 8.48889351e-01 3.48058105e-01 1.38126910e-02 5.45014739e-02
 2.85878897e-01 4.11484182e-01 7.45217562e-01 9.38081741e-03
 7.60194659e-03 1.01901889e-01 2.67333388e-01 6.21511221e-01
 5.31304777e-02 9.88490701e-01 1.96219534e-01 1.88631296e-01
 9.89157915e-01 1.42107606e-02 2.70124376e-02 9.68610942e-01]  que se encuentra en:  15
Imagen de BD:  46  Imagen de consulta:  46 

Distancia:  4.88634  al vector:  [9.8938781e-01 9.8711574e-01 5.4505229e-02 1.5125871e-02 2.0638242e-01
 4.3390393e-03 9.8700297e-01 6.7284906e-01 9.9999905e-01 1.6885817e-02
 6.8248199e-05 3.0013263e-02 8.1322193e-03 8.9782268e-01 3.0049616e-01
 2.1097064e-04 9.9606097e-01 3.8598147e-01 2.0665973e-02 3.4196436e-02
 1.0098875e-02 8.2332826e-01 1.0000000e+00 9.9991226e-01 9.9097800e-01
 5.8418393e-02 9.9955648e-01 3.3209028e-05 2.2139719e-01 9.9544299e-01
 9.9968886e-01 1.3396227e-01 9.9999827e-01 9.8259163e-01 8.6674976e-01
 9.9919510e-01 6.1475009e-02 9.1117620e-04 1.7657876e-02 2.5889307e-01
 9.7677499e-01 7.1955580e-01 9.9218500e-01 7.0972741e-03 5.5697560e-04
 7.8681111e-04 6.8592775e-01 3.4492884e-09 2.7770281e-02 9.3117827e-01
 7.0993304e-03 9.7060996e-01 2.1115094e-02 6.9510639e-03 2.4334177e-01
 9.9998045e-01 2.7919412e-03 2.7893692e-01 9.8536074e-01 6.7290664e-04
 3.1243986e-01 3.1196025e-01 9.9877787e-01 7.8073847e-01 4.9898213e-01
 9.7331583e-01 7.9750645e-01 6.1874104e-01 9.9496269e-01 9.9477476e-01
 1.2366176e-03 7.3534858e-01 5.6223363e-02 5.8410192e-01 1.6328332e-01
 9.1451979e-01 9.7923315e-01 8.7272739e-01 1.2626469e-02 4.2482615e-03
 7.6850593e-01 9.7397184e-01 9.9612570e-01 7.8748822e-01 8.4272790e-01
 7.9870254e-02 9.2109227e-01 4.4335842e-02 9.9370664e-01 9.4117233e-05
 7.7977878e-01 4.6403074e-01 2.7560347e-01 7.4990958e-02 5.8546096e-02
 9.9499846e-01 9.9695963e-01 7.3846447e-01 1.5724295e-01 4.3782255e-01
 4.8845112e-03 5.9778351e-09 1.0000000e+00 4.2232871e-04 6.4168870e-03
 9.9982953e-01 9.7760773e-01 9.9946564e-01 9.8377854e-01 9.9956238e-01
 9.3760151e-01 4.0549085e-01 7.1184278e-01 9.5666194e-01 4.6714962e-02
 2.1140307e-02 9.3149453e-01 6.1580008e-01 3.6843687e-01 4.6279209e-05
 2.8195441e-02 9.6458018e-02 3.2061830e-01 9.9899232e-01 8.8554549e-01
 3.3339858e-04 2.9906571e-02 7.6040924e-03]  en la posicion:  84  desde el vector:  [9.99696255e-01 1.12010598e-01 4.33547318e-01 2.50375365e-08
 1.39332712e-02 9.53626966e-07 9.60760117e-01 9.99972105e-01
 9.99912977e-01 1.35742992e-01 4.21696143e-07 8.96310091e-01
 3.40445638e-01 9.22841132e-01 1.54863894e-02 1.32098568e-10
 1.00000000e+00 1.15607083e-02 2.67773867e-04 4.42753434e-02
 4.72515821e-03 2.65852034e-01 9.99992013e-01 1.00000000e+00
 8.43588948e-01 9.56035376e-01 1.00000000e+00 1.27501909e-09
 9.23167586e-01 9.99998510e-01 6.60145223e-01 1.11990720e-01
 9.99999881e-01 9.62993503e-01 9.18405473e-01 1.00000000e+00
 9.99986053e-01 2.06775367e-02 9.88998294e-01 1.65047646e-02
 1.74773932e-02 7.95856714e-01 9.24042940e-01 4.59307432e-03
 9.31841135e-01 4.11368364e-06 9.70382988e-01 8.18424724e-07
 1.00157857e-02 1.38463110e-01 6.74982131e-01 7.38908887e-01
 9.96249855e-01 3.73724401e-02 9.34945703e-01 1.00000000e+00
 8.85877490e-01 8.90030265e-02 9.79021788e-01 9.23929691e-01
 1.45769119e-03 7.04467297e-01 4.96980846e-02 2.58857310e-02
 9.92594719e-01 7.66556859e-01 2.85063863e-01 7.49769807e-03
 8.96355152e-01 2.25938857e-02 3.81688579e-07 8.27129602e-01
 4.57221270e-03 1.49813443e-01 1.00288987e-02 9.86863136e-01
 7.74828792e-01 6.39686823e-01 1.98420584e-02 1.08455122e-02
 1.26812160e-02 8.92367959e-03 1.39217675e-02 1.06417328e-01
 4.83780205e-02 5.07116318e-04 9.86158371e-01 4.44189432e-08
 1.00000000e+00 1.32679939e-04 1.85689330e-03 2.87485719e-02
 4.40720558e-01 4.52956557e-03 1.20878309e-01 9.99999523e-01
 9.66302335e-01 9.99379039e-01 5.13736308e-02 3.50382209e-01
 6.11322463e-01 5.01795557e-06 9.99995232e-01 3.77333164e-03
 4.08142805e-04 1.00000000e+00 6.71183288e-01 1.00000000e+00
 9.34144437e-01 7.17872024e-01 8.20374131e-01 1.25022948e-01
 7.52818882e-01 9.91720796e-01 3.41947436e-01 6.31445509e-05
 1.12307429e-01 9.48946714e-01 6.26354754e-01 6.39442578e-06
 1.16834939e-02 9.85025406e-01 5.85100651e-02 6.17756546e-02
 1.00000000e+00 6.89906710e-09 1.92355685e-07 5.00116825e-01]  que se encuentra en:  16
Imagen de BD:  175  Imagen de consulta:  47 

Distancia:  4.0767984  al vector:  [9.8938781e-01 9.8711574e-01 5.4505229e-02 1.5125871e-02 2.0638242e-01
 4.3390393e-03 9.8700297e-01 6.7284906e-01 9.9999905e-01 1.6885817e-02
 6.8248199e-05 3.0013263e-02 8.1322193e-03 8.9782268e-01 3.0049616e-01
 2.1097064e-04 9.9606097e-01 3.8598147e-01 2.0665973e-02 3.4196436e-02
 1.0098875e-02 8.2332826e-01 1.0000000e+00 9.9991226e-01 9.9097800e-01
 5.8418393e-02 9.9955648e-01 3.3209028e-05 2.2139719e-01 9.9544299e-01
 9.9968886e-01 1.3396227e-01 9.9999827e-01 9.8259163e-01 8.6674976e-01
 9.9919510e-01 6.1475009e-02 9.1117620e-04 1.7657876e-02 2.5889307e-01
 9.7677499e-01 7.1955580e-01 9.9218500e-01 7.0972741e-03 5.5697560e-04
 7.8681111e-04 6.8592775e-01 3.4492884e-09 2.7770281e-02 9.3117827e-01
 7.0993304e-03 9.7060996e-01 2.1115094e-02 6.9510639e-03 2.4334177e-01
 9.9998045e-01 2.7919412e-03 2.7893692e-01 9.8536074e-01 6.7290664e-04
 3.1243986e-01 3.1196025e-01 9.9877787e-01 7.8073847e-01 4.9898213e-01
 9.7331583e-01 7.9750645e-01 6.1874104e-01 9.9496269e-01 9.9477476e-01
 1.2366176e-03 7.3534858e-01 5.6223363e-02 5.8410192e-01 1.6328332e-01
 9.1451979e-01 9.7923315e-01 8.7272739e-01 1.2626469e-02 4.2482615e-03
 7.6850593e-01 9.7397184e-01 9.9612570e-01 7.8748822e-01 8.4272790e-01
 7.9870254e-02 9.2109227e-01 4.4335842e-02 9.9370664e-01 9.4117233e-05
 7.7977878e-01 4.6403074e-01 2.7560347e-01 7.4990958e-02 5.8546096e-02
 9.9499846e-01 9.9695963e-01 7.3846447e-01 1.5724295e-01 4.3782255e-01
 4.8845112e-03 5.9778351e-09 1.0000000e+00 4.2232871e-04 6.4168870e-03
 9.9982953e-01 9.7760773e-01 9.9946564e-01 9.8377854e-01 9.9956238e-01
 9.3760151e-01 4.0549085e-01 7.1184278e-01 9.5666194e-01 4.6714962e-02
 2.1140307e-02 9.3149453e-01 6.1580008e-01 3.6843687e-01 4.6279209e-05
 2.8195441e-02 9.6458018e-02 3.2061830e-01 9.9899232e-01 8.8554549e-01
 3.3339858e-04 2.9906571e-02 7.6040924e-03]  en la posicion:  84  desde el vector:  [9.9976742e-01 5.6365150e-01 3.6710441e-02 3.8240216e-08 2.6013577e-01
 2.3831290e-05 5.2397251e-01 9.9998891e-01 9.9999976e-01 5.6533962e-02
 4.2136327e-07 5.5532664e-02 3.3851483e-01 4.9252307e-01 6.3748240e-01
 2.3239229e-10 1.0000000e+00 3.7774324e-02 8.1188693e-05 2.0205808e-01
 1.0054111e-03 8.9574623e-01 1.0000000e+00 1.0000000e+00 9.9776721e-01
 9.4972903e-01 1.0000000e+00 1.4927936e-09 8.6469299e-01 9.9998105e-01
 9.9204099e-01 3.2539207e-01 1.0000000e+00 9.4010818e-01 8.0130398e-01
 1.0000000e+00 9.3072581e-01 6.4010024e-03 1.3725373e-01 1.1593103e-03
 1.2502027e-01 8.2344413e-01 6.1495221e-01 2.2116303e-04 2.3327768e-02
 4.9008122e-08 7.9804635e-01 1.6561171e-09 9.8122835e-02 5.7891905e-03
 1.6691357e-02 6.7034513e-01 9.0760994e-01 6.7349732e-02 9.0253532e-01
 1.0000000e+00 1.0890865e-01 3.2361746e-03 5.8717364e-01 3.0308783e-02
 2.2794011e-01 4.4235688e-01 8.9586377e-01 6.1420661e-01 9.6060300e-01
 9.3919086e-01 9.8168421e-01 2.3228586e-01 9.2665923e-01 1.5098780e-02
 2.5512741e-06 1.2299341e-01 6.0109317e-02 5.2131653e-01 1.7906815e-02
 9.4357312e-01 8.9111161e-01 5.0929755e-01 3.8513619e-01 8.1414413e-01
 8.5933882e-01 8.5721141e-01 9.9762362e-01 2.8760317e-01 7.3467255e-02
 9.4373822e-03 9.1772008e-01 2.2450613e-06 1.0000000e+00 6.9103908e-06
 5.5991858e-02 9.3989372e-02 9.9819380e-01 6.5246820e-03 5.9080243e-01
 1.0000000e+00 9.6840119e-01 9.9926287e-01 5.9258938e-03 9.8224539e-01
 9.5979947e-01 4.8671556e-08 1.0000000e+00 4.5099854e-04 4.4981837e-03
 1.0000000e+00 9.7674513e-01 1.0000000e+00 5.6763858e-01 8.2420295e-01
 7.8274888e-01 9.9519169e-01 1.6457415e-01 4.2738894e-01 3.4507263e-01
 3.5768747e-04 1.3462514e-02 2.9086006e-01 3.1795019e-01 4.4503627e-06
 7.6847076e-03 9.9896955e-01 1.4783555e-01 9.8230410e-01 9.9989510e-01
 2.5199220e-09 3.4276397e-07 8.5262036e-01]  que se encuentra en:  17
Imagen de BD:  175  Imagen de consulta:  47 

Distancia:  3.4354815  al vector:  [9.8938781e-01 9.8711574e-01 5.4505229e-02 1.5125871e-02 2.0638242e-01
 4.3390393e-03 9.8700297e-01 6.7284906e-01 9.9999905e-01 1.6885817e-02
 6.8248199e-05 3.0013263e-02 8.1322193e-03 8.9782268e-01 3.0049616e-01
 2.1097064e-04 9.9606097e-01 3.8598147e-01 2.0665973e-02 3.4196436e-02
 1.0098875e-02 8.2332826e-01 1.0000000e+00 9.9991226e-01 9.9097800e-01
 5.8418393e-02 9.9955648e-01 3.3209028e-05 2.2139719e-01 9.9544299e-01
 9.9968886e-01 1.3396227e-01 9.9999827e-01 9.8259163e-01 8.6674976e-01
 9.9919510e-01 6.1475009e-02 9.1117620e-04 1.7657876e-02 2.5889307e-01
 9.7677499e-01 7.1955580e-01 9.9218500e-01 7.0972741e-03 5.5697560e-04
 7.8681111e-04 6.8592775e-01 3.4492884e-09 2.7770281e-02 9.3117827e-01
 7.0993304e-03 9.7060996e-01 2.1115094e-02 6.9510639e-03 2.4334177e-01
 9.9998045e-01 2.7919412e-03 2.7893692e-01 9.8536074e-01 6.7290664e-04
 3.1243986e-01 3.1196025e-01 9.9877787e-01 7.8073847e-01 4.9898213e-01
 9.7331583e-01 7.9750645e-01 6.1874104e-01 9.9496269e-01 9.9477476e-01
 1.2366176e-03 7.3534858e-01 5.6223363e-02 5.8410192e-01 1.6328332e-01
 9.1451979e-01 9.7923315e-01 8.7272739e-01 1.2626469e-02 4.2482615e-03
 7.6850593e-01 9.7397184e-01 9.9612570e-01 7.8748822e-01 8.4272790e-01
 7.9870254e-02 9.2109227e-01 4.4335842e-02 9.9370664e-01 9.4117233e-05
 7.7977878e-01 4.6403074e-01 2.7560347e-01 7.4990958e-02 5.8546096e-02
 9.9499846e-01 9.9695963e-01 7.3846447e-01 1.5724295e-01 4.3782255e-01
 4.8845112e-03 5.9778351e-09 1.0000000e+00 4.2232871e-04 6.4168870e-03
 9.9982953e-01 9.7760773e-01 9.9946564e-01 9.8377854e-01 9.9956238e-01
 9.3760151e-01 4.0549085e-01 7.1184278e-01 9.5666194e-01 4.6714962e-02
 2.1140307e-02 9.3149453e-01 6.1580008e-01 3.6843687e-01 4.6279209e-05
 2.8195441e-02 9.6458018e-02 3.2061830e-01 9.9899232e-01 8.8554549e-01
 3.3339858e-04 2.9906571e-02 7.6040924e-03]  en la posicion:  84  desde el vector:  [9.9818718e-01 9.7695255e-01 5.5749500e-01 6.8898201e-03 2.0305932e-02
 1.1800140e-02 4.7325993e-01 7.7426910e-01 9.9992430e-01 5.1903427e-03
 4.8321486e-04 2.5087774e-02 7.8487927e-01 4.0339077e-01 1.1856529e-01
 3.9255619e-04 9.6719253e-01 9.2974019e-01 9.7202510e-02 3.2692230e-01
 4.0003657e-04 8.7692046e-01 9.9985182e-01 9.9963117e-01 9.9303949e-01
 4.0624961e-01 9.9811816e-01 2.4771690e-04 1.3612863e-01 9.4090170e-01
 9.9374104e-01 8.6327165e-02 9.9994063e-01 9.9502611e-01 9.1623867e-01
 9.9868268e-01 1.8856391e-01 3.0010343e-03 5.2966177e-01 7.1875703e-01
 9.7138381e-01 9.0382975e-01 4.3782309e-01 1.1739072e-01 1.2949556e-02
 6.7146677e-05 7.8747618e-01 1.7328639e-05 2.4710894e-03 1.6956332e-01
 2.5110960e-02 8.1500185e-01 9.5606482e-01 1.1790007e-02 7.3063672e-03
 9.9579787e-01 2.5706917e-02 3.0631864e-01 8.9529073e-01 8.3577335e-03
 9.2175007e-02 5.1527762e-01 9.9942970e-01 1.0038313e-01 5.4122442e-01
 8.0740809e-01 1.1918619e-01 9.9444056e-01 9.6231973e-01 9.8187518e-01
 1.9931793e-03 9.7508949e-01 1.2367874e-02 2.7284265e-02 4.9477026e-01
 9.6675563e-01 9.4105697e-01 2.6689523e-01 4.2069232e-01 6.3243675e-01
 6.3625216e-02 9.9340940e-01 9.9544895e-01 9.7661632e-01 9.7630548e-01
 2.7673548e-01 9.9589390e-01 7.4742734e-01 9.9924856e-01 3.2673180e-03
 9.4594079e-01 1.1596829e-02 9.9480343e-01 2.0749539e-02 6.3542902e-02
 9.9541742e-01 9.9787593e-01 4.8329473e-02 2.5263131e-03 9.8602831e-01
 4.9742588e-01 7.3452247e-05 9.9998450e-01 3.0436450e-01 3.7955293e-01
 9.9713790e-01 9.9024552e-01 9.9940223e-01 8.9471126e-01 9.8288214e-01
 9.3625832e-01 7.0954370e-01 5.7768130e-01 9.1218412e-01 2.8511912e-02
 4.2393392e-01 2.9434922e-01 9.3431342e-01 6.6380668e-01 4.7390560e-05
 1.0346830e-02 4.2201459e-02 8.4934872e-01 9.9982142e-01 2.4723500e-01
 1.0634065e-03 4.2011470e-01 1.4493465e-03]  que se encuentra en:  18
Imagen de BD:  175  Imagen de consulta:  47 

Distancia:  3.694109  al vector:  [9.99251664e-01 3.75992209e-01 9.44667459e-01 1.23821199e-02
 3.45815659e-01 9.51188922e-01 6.06158197e-01 1.39564276e-04
 4.94230151e-01 1.11300244e-04 8.25778008e-01 9.81791973e-01
 9.98755574e-01 8.85533452e-01 2.69469619e-03 9.62087154e-01
 3.70756209e-01 3.60240877e-01 9.93762136e-01 7.30378628e-02
 1.71831191e-01 9.56110179e-01 1.27461433e-01 4.45933044e-02
 9.99151409e-01 8.72598350e-01 4.72368181e-01 3.68684530e-03
 3.14583182e-01 2.82705903e-01 9.99998987e-01 1.40269011e-01
 6.27283275e-01 9.99964297e-01 9.27684844e-01 1.80552602e-02
 4.41009998e-01 1.69688315e-06 3.75068188e-03 9.64027405e-01
 5.90807199e-03 2.26981580e-01 9.90303278e-01 5.74367642e-02
 8.97481441e-01 2.63040125e-01 2.82262027e-01 1.33809924e-01
 3.50773335e-04 8.60005796e-01 5.88235974e-01 9.41744685e-01
 9.98949289e-01 6.05743229e-01 1.30228682e-06 8.51252735e-01
 9.99933422e-01 8.56071711e-04 9.31330740e-01 9.58378732e-01
 9.38862205e-01 9.82851505e-01 9.97492433e-01 9.50206995e-01
 1.00278527e-01 1.98464006e-01 6.30088329e-01 2.08069980e-02
 9.12296772e-01 9.89736795e-01 3.66991788e-01 7.19008565e-01
 2.85055939e-05 2.97895610e-01 7.64357924e-01 9.93783593e-01
 3.73849273e-03 5.87823987e-03 2.32558876e-01 6.33251667e-03
 9.37525988e-01 9.99943137e-01 9.77633059e-01 9.99997616e-01
 9.98470664e-01 2.67644227e-01 1.00000000e+00 8.59245896e-01
 1.10637575e-01 7.53330588e-01 7.73717165e-01 4.61236681e-07
 2.99347758e-01 2.00497452e-05 5.72273135e-02 7.75224149e-01
 9.99999702e-01 1.15677476e-01 8.41811299e-03 4.24641341e-01
 9.16698873e-02 4.67629731e-02 9.76603746e-01 9.93476391e-01
 6.53038919e-02 5.38902283e-02 9.99957561e-01 7.37733245e-01
 1.22382820e-01 2.67122835e-01 1.06896013e-01 2.88555384e-01
 7.65103221e-01 9.81240034e-01 6.82841778e-01 9.96344447e-01
 5.00970781e-02 9.72333074e-01 5.98850131e-01 7.87705183e-04
 9.79763513e-07 8.26136470e-01 4.80213076e-01 9.98614788e-01
 1.87277228e-01 4.63984787e-01 9.97304797e-01 1.65343285e-04]  en la posicion:  147  desde el vector:  [7.3282504e-01 3.6892784e-01 9.7517967e-01 3.5884202e-02 1.0970235e-01
 1.0895640e-02 5.6754470e-02 3.7167668e-03 9.8163879e-01 1.2256056e-02
 7.4429023e-01 7.3388338e-01 9.9869013e-01 8.4951895e-01 2.5324917e-01
 5.9520060e-01 3.0533373e-01 4.8901612e-01 9.5301092e-01 7.0461369e-01
 4.7754896e-01 9.8895383e-01 9.9438894e-01 9.6394229e-01 8.9620471e-01
 1.8774524e-01 9.6014684e-01 4.4324994e-04 9.4085813e-02 9.6612227e-01
 9.9998248e-01 3.0116767e-02 8.7920141e-01 9.9939263e-01 4.6101010e-01
 2.7446532e-01 4.0733814e-03 1.1403880e-04 3.7473732e-01 9.5347351e-01
 1.6682902e-01 7.6856184e-01 9.7388268e-01 1.0674059e-02 7.2598457e-03
 4.7835052e-02 2.4572870e-01 8.6683035e-04 1.4074147e-03 2.1077627e-01
 7.6140553e-01 9.9583149e-01 7.1289164e-01 4.7637910e-02 1.1453471e-04
 9.4194835e-01 4.7294784e-01 1.4697438e-01 5.7372671e-01 1.0214448e-02
 9.7727805e-01 8.3533126e-01 9.9996579e-01 8.5990441e-01 4.1451961e-02
 4.9541259e-01 7.9179275e-01 1.9610250e-01 9.2716461e-01 9.9738646e-01
 9.8227262e-03 2.8236097e-01 2.1525204e-02 5.1226288e-02 9.8841578e-01
 9.9952543e-01 6.9471300e-03 1.6482231e-01 4.0141824e-01 6.4349115e-02
 9.3664205e-01 9.9489427e-01 9.7038412e-01 9.9955595e-01 9.9900925e-01
 4.7713348e-01 9.9952793e-01 9.9577200e-01 5.2756661e-01 3.8454714e-01
 2.8223783e-01 1.3838470e-02 6.9602108e-01 2.9273748e-02 3.3249736e-02
 7.6626408e-01 9.9991715e-01 2.2501627e-01 5.4190546e-01 5.7396495e-01
 1.2831777e-02 6.0799718e-04 9.9994165e-01 2.7210662e-01 9.6402359e-01
 6.1609250e-01 9.9949598e-01 7.9300445e-01 2.2937357e-02 4.2386144e-02
 7.3068142e-03 5.6941104e-01 3.9008147e-01 8.5874164e-01 3.8576758e-01
 9.5697713e-01 6.4727426e-01 6.5614825e-01 6.1588943e-01 2.5934845e-02
 5.3232908e-04 7.2527385e-01 1.7725766e-01 9.9994773e-01 4.0628028e-01
 3.4431556e-01 9.6514273e-01 1.0879313e-04]  que se encuentra en:  19
Imagen de BD:  49  Imagen de consulta:  49 

Distancia:  3.8906584  al vector:  [4.05041277e-02 9.15444798e-06 7.73451805e-01 1.77530855e-01
 2.09440500e-01 8.80807638e-04 6.44203424e-02 2.16537654e-01
 9.99665678e-01 1.50480866e-03 9.99906361e-01 1.41290545e-01
 9.99728024e-01 3.23712826e-01 9.52200174e-01 9.99969363e-01
 5.82209527e-02 3.72588634e-04 5.23158908e-03 2.27828532e-01
 2.40069091e-01 1.00000000e+00 9.99466896e-01 9.99579310e-01
 9.92864609e-01 9.15606141e-01 9.86921549e-01 7.74605751e-01
 7.83319116e-01 9.70715284e-01 9.99997139e-01 8.17669511e-01
 2.87702680e-03 9.35951173e-01 8.62130880e-01 8.98878352e-05
 9.90643620e-01 9.41221733e-05 9.99998927e-01 2.67705560e-01
 1.38188899e-01 8.43497276e-01 2.68361449e-01 7.68207519e-06
 2.26734936e-01 1.62129998e-02 3.17447424e-01 1.28435777e-05
 1.46526098e-02 4.97943163e-03 1.53889030e-01 9.94001508e-01
 9.98101950e-01 6.00273788e-01 5.83973269e-05 9.81019497e-01
 9.46488362e-05 7.40618289e-01 8.22571754e-01 3.37224305e-01
 9.80300128e-01 8.75417233e-01 9.94980454e-01 4.64192927e-01
 8.13532006e-06 9.96859968e-01 9.44260657e-01 9.40632701e-01
 3.83902534e-06 9.00501847e-01 1.30805075e-02 2.73661911e-01
 9.97655034e-01 9.78937149e-02 6.87554777e-02 9.94331479e-01
 9.38177109e-04 1.55038536e-01 6.38352334e-01 9.83223438e-01
 9.87213850e-01 9.99998510e-01 5.89034498e-01 9.89879251e-01
 8.64376664e-01 3.55268359e-01 4.26403880e-01 2.34783292e-02
 1.54119730e-03 1.55330449e-01 4.93335392e-07 8.27111661e-01
 9.94956315e-01 9.91645157e-01 5.06747484e-01 7.33938813e-02
 9.99627113e-01 9.99504924e-01 1.96580231e-01 7.48071074e-03
 5.74175239e-01 1.35821104e-03 9.99336362e-01 5.15878783e-05
 8.48532915e-01 5.30543657e-05 9.90222633e-01 1.60753429e-02
 1.06631815e-02 9.93541121e-01 2.05397384e-07 2.40415007e-01
 3.54692340e-03 2.14616686e-01 9.92310047e-03 1.36941671e-01
 1.69256330e-03 5.93884468e-01 7.13804483e-01 8.70946825e-01
 1.79672241e-03 9.50946391e-01 5.44227362e-02 9.95963037e-01
 9.81949031e-01 9.66843843e-01 6.83662295e-01 2.31614709e-03]  en la posicion:  70  desde el vector:  [4.6087921e-02 1.4487678e-09 3.2214832e-01 4.6771860e-01 1.0082424e-03
 4.1559906e-06 8.4536374e-03 9.9586999e-01 9.8070359e-01 1.5256253e-01
 9.9127495e-01 8.7511790e-01 9.9955213e-01 7.0422184e-01 7.2097218e-01
 9.2196226e-01 3.3130646e-01 1.8727113e-09 4.5306683e-03 5.7341045e-01
 3.0869395e-02 9.9999899e-01 9.7414619e-01 9.9999470e-01 9.8995960e-01
 9.5931578e-01 9.9999261e-01 9.8657393e-01 6.0344040e-03 9.9998009e-01
 9.9859643e-01 8.2635230e-01 3.6375821e-03 9.9918377e-01 9.9401224e-01
 4.0580064e-02 9.8445225e-01 5.2939482e-05 1.0000000e+00 1.6534656e-02
 2.7161150e-06 9.8529559e-01 8.4349084e-01 3.2340104e-08 9.9902475e-01
 4.3733332e-01 4.9012566e-01 7.2237849e-04 1.1312962e-03 1.0410677e-04
 1.3686180e-01 7.9257226e-01 9.7875798e-01 7.5090998e-01 2.4479517e-05
 9.9991828e-01 1.6008791e-01 4.2591631e-01 5.0534701e-01 9.9937874e-01
 3.8588244e-01 2.2747949e-01 9.9965215e-01 9.5537090e-01 4.9281081e-08
 2.4204403e-01 7.7765912e-01 9.9697876e-01 1.3054171e-09 9.6789080e-01
 1.0127456e-07 8.3587170e-03 9.9487352e-01 7.3761344e-03 9.5510781e-01
 2.7053773e-02 9.1446105e-05 9.9140048e-01 6.7820114e-01 6.9479555e-02
 7.7625930e-01 9.8973179e-01 8.4310722e-01 9.9942291e-01 8.9738846e-02
 3.6729401e-01 1.3518631e-03 1.9400716e-03 1.3311982e-02 2.8045237e-02
 2.6019595e-07 9.8449695e-01 9.7958314e-01 9.9151325e-01 6.8870491e-01
 6.3932300e-02 9.9960065e-01 1.0000000e+00 1.2298426e-01 1.4509499e-01
 9.7800195e-01 3.7355721e-03 9.9887419e-01 2.7096689e-07 7.1567595e-01
 5.0561607e-02 9.9990368e-01 7.8742743e-02 6.1245173e-02 4.2802304e-02
 2.7814100e-09 2.1383381e-01 7.3019171e-01 3.9208069e-01 5.3071994e-01
 3.3599734e-03 5.9659868e-02 7.1709633e-02 8.6544883e-01 9.9904048e-01
 7.4906246e-05 2.2135139e-02 4.5767263e-01 9.5094442e-01 9.9997032e-01
 4.1982892e-01 4.5874119e-03 7.5007033e-01]  que se encuentra en:  20
Imagen de BD:  162  Imagen de consulta:  51 

Distancia:  3.8144565  al vector:  [2.4098158e-04 1.4380201e-07 1.8751302e-01 9.9887812e-01 6.5063792e-01
 1.2689829e-04 9.6412510e-01 3.3512568e-01 9.5392036e-01 8.1120777e-01
 9.9999857e-01 2.9334360e-01 9.9997920e-01 2.8162596e-01 9.8695552e-01
 9.9999499e-01 1.1610299e-02 1.1959374e-03 2.9431295e-01 2.2594747e-01
 9.6180499e-01 9.9959004e-01 1.4479965e-01 9.0934116e-01 4.8069030e-02
 7.2487772e-01 8.3977151e-01 9.9977982e-01 1.5490827e-01 9.9983108e-01
 9.7782052e-01 1.1236846e-02 9.9017794e-05 5.0773931e-01 3.6410952e-01
 2.3422638e-06 1.5695187e-01 1.1061007e-01 9.9998724e-01 8.4902036e-01
 1.6377866e-03 3.8469300e-01 9.2985284e-01 1.0797083e-03 7.2292852e-01
 9.8220396e-01 3.7686050e-02 7.5725770e-01 5.0100148e-02 1.5874207e-03
 2.6642424e-01 7.2107577e-01 7.8483462e-02 7.7336162e-02 6.6313744e-03
 2.2496706e-01 5.8764189e-02 1.6115683e-01 9.1643536e-01 9.7625005e-01
 4.4735479e-01 5.8687067e-01 9.9979532e-01 8.1245303e-02 1.4179708e-06
 5.7184482e-01 4.3927011e-01 3.2079136e-01 2.1556112e-07 3.2695922e-01
 5.9291498e-05 8.5172343e-01 9.9991667e-01 2.4193615e-02 8.7338054e-01
 9.0625048e-01 2.3347049e-05 1.7850751e-01 2.9352099e-01 7.1616900e-01
 2.8811347e-01 9.9326766e-01 9.0298092e-01 9.9552059e-01 1.7393190e-01
 2.0776957e-02 1.9168556e-03 9.9899089e-01 1.7893198e-06 7.9846859e-01
 4.1908461e-05 9.5147854e-01 3.6282718e-02 9.9979806e-01 7.9749519e-01
 1.3472438e-03 8.3495790e-01 9.9985456e-01 9.9383724e-01 9.3519783e-01
 9.0088356e-01 9.1932702e-01 3.5207045e-01 1.6873406e-05 9.7746301e-01
 9.2012133e-06 9.9534559e-01 2.7968287e-03 9.5299542e-01 1.2790859e-03
 3.7443407e-08 2.1929711e-01 8.9139366e-01 9.9301875e-01 8.8874757e-02
 4.0314972e-01 7.1578300e-01 5.2906406e-01 6.1006176e-01 9.9609411e-01
 1.1928925e-01 7.1757340e-01 1.8180326e-01 9.9927974e-01 7.0896363e-01
 9.9937665e-01 9.1334355e-01 7.9797781e-01]  en la posicion:  99  desde el vector:  [7.9360843e-02 1.2964194e-06 6.4154923e-01 9.8336393e-01 1.9314143e-01
 3.8918111e-05 3.6111778e-01 5.9426308e-02 3.8184115e-01 9.3915099e-01
 9.9999034e-01 2.3113742e-01 9.9999809e-01 8.0795801e-01 1.6730231e-01
 9.9997473e-01 4.1348040e-03 4.5761986e-05 7.7483261e-01 9.4446850e-01
 9.2041999e-01 9.9989682e-01 6.2441856e-02 9.6083802e-01 3.2550544e-02
 3.6457604e-01 8.0977124e-01 9.9730515e-01 3.1466842e-02 9.9974942e-01
 9.9768662e-01 3.2284766e-02 5.4753320e-05 9.9934590e-01 7.7647185e-01
 3.6901434e-05 8.3216453e-01 2.5445521e-03 9.9998170e-01 1.2117463e-01
 3.2043457e-04 9.9033356e-01 9.5471084e-01 5.7342649e-04 8.9843822e-01
 9.5688891e-01 1.6808960e-01 2.7922660e-01 1.0402799e-03 2.4185014e-01
 1.5677291e-01 9.6132350e-01 5.8658332e-01 5.7539773e-01 1.1017323e-03
 4.7215444e-01 9.4690847e-01 1.9083542e-01 8.4669471e-01 9.8729670e-01
 9.7893721e-01 3.9484882e-01 9.8365313e-01 3.1243145e-01 7.6202282e-06
 4.9697101e-01 9.7387934e-01 6.1290801e-01 5.2291239e-07 9.4712508e-01
 4.0557981e-04 3.2043308e-01 9.7972572e-01 7.3357224e-03 2.6388371e-01
 8.5865641e-01 8.2459042e-07 8.0572760e-01 2.6488543e-02 9.8230064e-01
 8.9809895e-01 9.5396686e-01 7.9456329e-02 9.9848586e-01 3.9836767e-01
 3.0426252e-01 3.8582921e-02 9.6130890e-01 5.8752298e-04 6.8820149e-01
 4.3743290e-05 6.2388480e-01 8.3576107e-01 9.5111430e-01 2.3737803e-01
 2.9509366e-03 9.9524164e-01 9.9995399e-01 9.7033828e-01 1.6612360e-01
 5.5636001e-01 5.6892169e-01 9.2729062e-01 4.9754977e-04 9.8264641e-01
 1.0802250e-04 9.9958104e-01 5.6621432e-03 5.1290512e-01 5.0668871e-01
 3.0626282e-07 1.1355129e-01 3.3790240e-01 9.0702534e-01 9.3686068e-01
 9.8485756e-01 5.6954193e-01 8.8309431e-01 4.0582851e-01 9.9957252e-01
 3.2351911e-03 6.6869116e-01 3.7655765e-01 9.9832398e-01 9.9967420e-01
 9.9661934e-01 9.9144769e-01 7.9370070e-01]  que se encuentra en:  21
Imagen de BD:  189  Imagen de consulta:  53 

Distancia:  2.8027093  al vector:  [9.25970078e-03 8.39918852e-04 6.74719810e-02 9.99959767e-01
 2.19431221e-02 5.24037540e-01 7.23544717e-01 6.87598658e-05
 9.99977231e-01 4.42656875e-03 9.67984676e-01 7.47448802e-02
 9.25982594e-02 7.83400178e-01 3.00122797e-02 9.99987006e-01
 4.77766991e-03 2.84296274e-03 9.97937560e-01 3.13207507e-03
 8.28479826e-02 9.99298453e-01 9.99999762e-01 9.95706022e-01
 9.88735199e-01 9.45503473e-01 9.89978015e-01 9.96320009e-01
 1.87216341e-01 7.70719886e-01 9.99907017e-01 6.51318073e-01
 1.84544623e-02 9.99771595e-01 9.96403992e-01 2.89767981e-04
 2.54810220e-05 1.46597624e-04 1.12172365e-01 9.71652806e-01
 8.95094156e-01 8.96357298e-02 8.79643917e-01 1.94120421e-05
 4.94837761e-04 9.99123514e-01 8.61365318e-01 3.94853515e-07
 4.59969044e-04 9.85574126e-01 9.62681651e-01 9.30548310e-01
 1.38989985e-02 1.71535313e-02 1.36584044e-04 9.97888327e-01
 8.33562481e-08 3.92408073e-01 9.69097495e-01 1.35093927e-04
 6.30065799e-01 9.51489925e-01 9.99999166e-01 2.42665052e-01
 3.58645161e-06 2.10503101e-01 9.23453033e-01 9.36657548e-01
 3.01818788e-01 7.54052520e-01 1.15699470e-02 2.04491615e-02
 9.99766290e-01 2.26115286e-02 8.14472735e-01 9.84422445e-01
 8.13835979e-01 6.78278208e-01 3.25326025e-02 2.03332305e-03
 2.37340927e-02 9.80596840e-01 9.99721706e-01 9.96183753e-01
 8.75488639e-01 6.38412356e-01 3.83726656e-02 9.99590516e-01
 6.44415617e-04 1.09961629e-03 6.20564520e-01 9.98147726e-01
 2.78765440e-01 9.75933790e-01 2.85826325e-02 1.04744877e-05
 9.99853313e-01 9.97786701e-01 7.36417174e-02 9.95285392e-01
 9.38735485e-01 6.28830503e-06 9.99999285e-01 5.33044338e-04
 2.12547630e-01 2.66075134e-04 9.98958230e-01 5.07324934e-04
 9.20313597e-03 3.78690720e-01 3.93003225e-04 4.71045256e-01
 9.89894271e-01 9.97416973e-01 5.84362745e-01 9.97459292e-01
 9.99624133e-01 4.97156382e-02 1.71730518e-02 8.86834025e-01
 4.40478325e-04 6.30191863e-02 6.76147878e-01 9.99965906e-01
 9.53665376e-03 9.97967601e-01 9.79942679e-01 9.46238637e-03]  en la posicion:  72  desde el vector:  [4.97003794e-02 4.69893217e-04 4.96948361e-02 9.99949753e-01
 3.45267057e-02 8.66666198e-01 2.36749649e-04 2.09301710e-04
 9.99537528e-01 9.76404250e-02 9.75996912e-01 2.90599465e-03
 1.95453793e-01 5.18979490e-01 3.42900425e-01 9.99995828e-01
 5.57035208e-04 9.50857997e-03 9.99885798e-01 4.62602079e-02
 3.91767710e-01 9.99808133e-01 9.99999464e-01 9.73055661e-01
 9.81668591e-01 5.99408925e-01 9.93662119e-01 9.99998689e-01
 2.50256509e-01 2.40251422e-01 9.99621630e-01 8.81331801e-01
 2.24232674e-03 9.98121619e-01 9.68646765e-01 1.82175245e-06
 6.39729842e-06 3.33700973e-06 3.22205812e-01 9.94188726e-01
 4.74051297e-01 4.41317648e-01 9.27669764e-01 1.25348979e-05
 8.23165246e-05 9.99878287e-01 3.47403646e-01 3.09083021e-06
 2.01582909e-04 9.98844743e-01 9.92244124e-01 9.90194678e-01
 1.74465775e-03 9.78893042e-03 5.67530788e-06 9.73186612e-01
 4.66908645e-09 9.28548336e-01 9.96047258e-01 1.30176544e-04
 9.96300220e-01 9.98376429e-01 9.99981403e-01 3.72770727e-02
 1.98759466e-07 1.57895237e-01 4.86909240e-01 8.55192780e-01
 3.24085355e-03 9.99376714e-01 2.07753986e-01 6.64180517e-03
 9.99259174e-01 3.61692905e-03 9.78657007e-01 9.39333439e-01
 9.75997806e-01 9.87478375e-01 4.63431478e-02 2.58220434e-02
 9.94826674e-01 9.95523274e-01 9.97450829e-01 9.99960721e-01
 8.53497028e-01 8.06587219e-01 2.94566154e-04 9.95872140e-01
 5.06699143e-06 1.01971626e-03 3.97422910e-03 9.60957408e-01
 7.51667857e-01 9.98936236e-01 2.75319815e-03 8.63105015e-07
 9.99994993e-01 9.98068035e-01 5.14205396e-02 3.35842609e-01
 8.81124258e-01 1.54621866e-05 9.99926984e-01 6.95431232e-03
 6.21847808e-02 1.42221725e-05 9.99944687e-01 2.17815668e-05
 4.86170650e-02 9.88657832e-01 5.72413206e-04 1.98968649e-02
 7.08315969e-01 8.30203235e-01 2.47301489e-01 9.99988317e-01
 9.98812199e-01 5.28365374e-04 9.77222085e-01 9.15761828e-01
 1.22369402e-05 6.03654683e-02 1.56756580e-01 9.99916136e-01
 2.49373019e-02 9.99307156e-01 9.99877334e-01 9.06352282e-01]  que se encuentra en:  22
Imagen de BD:  164  Imagen de consulta:  7 

Distancia:  4.5319757  al vector:  [8.99816632e-01 9.99992132e-01 5.36407113e-01 1.03253782e-01
 5.79427004e-01 9.98490095e-01 9.75921750e-01 8.68930459e-01
 9.22012150e-01 4.49544907e-01 1.15820855e-01 3.87546033e-01
 3.15637249e-06 7.99122453e-01 7.28229284e-02 7.63832510e-01
 9.99626338e-01 9.99999523e-01 1.18303299e-03 8.42934012e-01
 9.68176305e-01 3.46213579e-04 6.61697984e-01 3.05106342e-02
 4.23807770e-01 3.28491330e-02 2.17127800e-03 2.03900039e-02
 6.66430116e-01 3.22944522e-01 2.27921307e-02 6.32048547e-01
 9.96753693e-01 7.05995717e-06 4.19017673e-02 2.75357991e-01
 8.90103221e-01 9.92387056e-01 8.18544213e-05 9.97362018e-01
 9.99947608e-01 9.60922122e-01 9.48593378e-01 9.99967754e-01
 6.47276640e-03 5.98788261e-04 4.15600955e-01 5.58517396e-01
 9.99989867e-01 7.08522141e-01 1.89020932e-02 9.18849587e-01
 3.46989334e-01 2.96639889e-01 8.82758915e-01 1.01861835e-01
 3.91249955e-02 1.40340567e-01 7.71146417e-02 2.87851691e-03
 1.24873996e-01 4.10675853e-01 8.70907307e-03 4.03118134e-03
 9.99359846e-01 9.02256072e-01 2.31166720e-01 7.12194145e-01
 9.99999702e-01 2.01587975e-02 9.91984963e-01 8.17850888e-01
 4.43278283e-01 3.45580995e-01 4.22863752e-01 6.87484324e-01
 9.99980450e-01 5.15673399e-01 7.62317061e-01 2.54030824e-02
 9.78609920e-03 9.69579995e-01 8.73679519e-01 1.21871531e-02
 8.44597816e-04 7.72774518e-02 4.89339292e-01 8.07595372e-01
 6.70752704e-01 2.31476486e-01 9.99129176e-01 3.52543294e-02
 8.88452888e-01 8.91570270e-01 5.45141935e-01 9.68025088e-01
 1.70001388e-03 1.72371688e-06 9.67986703e-01 5.69749177e-02
 6.53203726e-02 5.84282160e-01 1.37195766e-01 9.88923073e-01
 5.81444204e-02 6.16313219e-01 2.37405300e-04 9.98802423e-01
 6.64705575e-01 9.99157906e-01 9.99559999e-01 4.47157472e-01
 7.60622025e-02 4.45937157e-01 9.77582812e-01 3.21628451e-02
 5.77295184e-01 3.18021178e-02 3.23385656e-01 5.56031466e-07
 9.79548156e-01 9.52386975e-01 4.28642780e-01 3.23456824e-02
 1.31622937e-05 1.20421648e-02 1.19006127e-01 9.44476008e-01]  en la posicion:  163  desde el vector:  [4.90613580e-02 7.32319176e-01 7.97352672e-01 7.85724998e-01
 8.07865739e-01 1.96179718e-01 9.70234871e-01 9.58801031e-01
 6.24715388e-01 9.96598005e-01 6.83628023e-01 5.40829360e-01
 2.18159646e-01 9.49842930e-01 5.76328278e-01 3.52971345e-01
 9.96488392e-01 9.99915421e-01 1.68440402e-01 3.96262139e-01
 9.99900579e-01 1.06913298e-01 2.51862705e-01 4.81478244e-01
 6.26832189e-05 1.20361474e-04 3.17183256e-01 9.88707840e-02
 9.57164168e-03 3.55996728e-01 1.29035115e-03 9.95771646e-01
 5.62307298e-01 9.29816233e-05 6.28089905e-03 4.71011847e-01
 3.24645549e-01 9.99720573e-01 2.31977105e-01 5.36583960e-01
 9.99129534e-01 1.27233058e-01 2.53484488e-01 9.92962122e-01
 3.93255651e-02 7.53825605e-02 8.25987577e-01 8.37757766e-01
 9.99258697e-01 6.97423279e-01 9.47089314e-01 5.93658149e-01
 6.20099068e-01 5.73505878e-01 9.96104479e-01 4.51872647e-02
 1.20651275e-01 1.67761743e-02 9.95366454e-01 4.03170973e-01
 4.21810061e-01 5.82055748e-02 2.43805647e-02 2.76676476e-01
 9.38304424e-01 5.39256334e-02 6.18952990e-01 9.91431057e-01
 9.99435902e-01 8.16168487e-02 5.19463301e-01 6.09853268e-02
 9.67039287e-01 2.10470796e-01 4.53839600e-02 8.42112780e-01
 9.99858260e-01 8.86349320e-01 4.28432226e-03 2.66078115e-01
 9.62587953e-01 2.92127013e-01 3.12065303e-01 1.12617910e-02
 3.51570427e-01 1.87078118e-03 1.38117075e-02 9.70995426e-01
 1.17063493e-01 2.45561898e-02 6.52624130e-01 8.51407647e-01
 1.15746856e-02 9.52683628e-01 6.68052077e-01 9.87827897e-01
 1.10659003e-03 1.81859434e-02 9.99453545e-01 3.78049970e-01
 6.93495274e-02 8.78998041e-01 4.97640371e-02 7.98956335e-01
 9.68336582e-01 7.76845753e-01 4.79280643e-05 9.70917225e-01
 7.45736659e-02 9.91697550e-01 3.37706566e-01 2.64390290e-01
 8.50488544e-01 8.64833593e-04 9.80288804e-01 3.77107680e-01
 7.43921638e-01 1.83257967e-01 9.98847842e-01 8.26805830e-03
 9.98625636e-01 8.79119396e-01 7.29563832e-03 8.26215744e-02
 2.39633620e-02 4.45193052e-02 9.14185882e-01 9.98571575e-01]  que se encuentra en:  23
Imagen de BD:  63  Imagen de consulta:  54 

Distancia:  3.4415185  al vector:  [1.6537309e-04 9.5936882e-01 4.0960068e-01 9.9984717e-01 1.7225981e-02
 4.5043230e-04 8.7115932e-01 2.9834121e-02 9.9999762e-01 1.5958076e-05
 9.9065948e-01 1.4615360e-01 1.0925233e-03 6.9484794e-01 1.3213670e-01
 9.9860203e-01 6.0299844e-02 9.7812551e-01 1.3071299e-04 7.0143825e-01
 9.5465255e-01 5.0344825e-02 9.9999857e-01 9.8624498e-02 5.7097012e-01
 2.5056413e-01 3.1072497e-03 8.1302822e-03 4.1498524e-01 9.9484015e-01
 9.9998498e-01 1.3178584e-01 9.9816471e-01 3.8100988e-02 1.7957583e-01
 6.8086386e-04 3.8830042e-03 9.6434915e-01 3.3849180e-02 6.7580307e-01
 9.9985045e-01 8.2097900e-01 5.5032414e-01 8.3407187e-01 2.7174056e-03
 9.1338158e-04 4.1881427e-01 2.3762534e-06 9.8848641e-01 2.8205135e-05
 7.3483676e-02 9.8628378e-01 2.0202571e-01 1.7632893e-01 9.5267987e-01
 8.6939842e-02 6.6177604e-06 2.2332430e-02 3.7080312e-01 7.6669455e-04
 3.6017281e-01 7.6118529e-01 9.9995339e-01 9.6758485e-01 9.8390505e-05
 1.7199647e-01 2.0490840e-01 9.2685640e-01 9.9966353e-01 5.0872386e-01
 2.2657514e-03 1.2402117e-01 9.9982524e-01 3.7138891e-01 2.5594926e-01
 8.5910773e-01 8.8577837e-02 9.9109113e-03 2.7443451e-01 6.3633877e-01
 9.3291336e-01 9.9998492e-01 9.9997985e-01 2.6361257e-02 8.2402754e-01
 7.4871254e-01 9.9130177e-01 9.9907809e-01 2.3528934e-04 6.5080702e-01
 9.6532810e-01 9.8354208e-01 8.5960865e-01 9.9946582e-01 3.2211280e-01
 1.9855499e-03 7.0544481e-03 7.7590317e-02 9.8656893e-01 9.7094917e-01
 5.4848677e-01 8.6473631e-07 9.9999845e-01 9.2716373e-06 7.3316550e-01
 2.4387240e-04 9.9851525e-01 3.2155693e-02 7.1016032e-01 7.4618936e-01
 4.0895357e-05 3.2792389e-01 9.7937065e-01 8.5854119e-01 5.9691763e-01
 4.8238039e-04 9.9395454e-01 3.8226593e-01 3.6869103e-01 1.5020370e-04
 9.7525573e-01 4.3825090e-02 9.7105360e-01 9.9992156e-01 3.5068901e-05
 9.8112953e-01 1.9228458e-04 2.8676008e-05]  en la posicion:  30  desde el vector:  [9.32870507e-02 9.99693513e-01 2.07543373e-01 8.85354638e-01
 7.57116973e-02 2.10580230e-03 3.75494778e-01 3.78378928e-01
 9.96172071e-01 6.60316110e-01 7.38201261e-01 3.80015224e-01
 7.45907724e-02 1.89120352e-01 9.29638863e-01 6.81452870e-01
 2.09243238e-01 9.98751760e-01 1.29780173e-03 1.67077214e-01
 9.83419180e-01 1.39325857e-04 9.91924405e-01 2.04097927e-02
 2.20632643e-01 2.14922428e-03 7.85946846e-04 2.39204466e-02
 6.71785533e-01 9.91906762e-01 9.60881591e-01 5.06222248e-02
 9.96498168e-01 3.60338390e-02 8.29893351e-01 8.98826718e-02
 2.84028053e-02 9.87248719e-01 5.88292480e-02 9.92923379e-01
 9.76950765e-01 9.34180021e-01 2.61535645e-02 9.99959171e-01
 2.11411417e-02 2.01213062e-02 2.86580622e-01 2.47016549e-03
 9.42599535e-01 3.05208564e-03 8.80844712e-01 5.48616171e-01
 2.82999873e-03 1.46269232e-01 9.72509623e-01 1.62741542e-03
 2.80506313e-02 9.72679257e-03 8.04359794e-01 2.39509344e-03
 3.16604495e-01 8.23555350e-01 9.92603779e-01 8.18677425e-01
 8.87924433e-03 8.26342344e-01 9.11990821e-01 4.79760170e-01
 9.99236524e-01 1.09970838e-01 1.60247087e-03 7.46345103e-01
 8.70788455e-01 4.07198071e-03 9.52280879e-01 5.56478202e-01
 2.30907470e-01 7.71686435e-02 3.39532495e-02 9.48800921e-01
 7.35285163e-01 9.12262917e-01 9.76205647e-01 1.54658347e-01
 9.87704098e-01 3.69552016e-01 7.58627415e-01 9.99830663e-01
 1.17125541e-01 9.38281357e-01 9.84935284e-01 7.35830903e-01
 8.16898584e-01 9.79126096e-01 3.03852916e-01 1.62897170e-01
 6.16747141e-03 7.30412939e-05 9.98682141e-01 9.91952181e-01
 8.77758265e-02 8.72364640e-03 9.95456100e-01 1.02814406e-01
 6.08697534e-02 2.29714692e-01 9.89183426e-01 6.98489070e-01
 2.10123569e-01 7.58332014e-02 6.18199170e-01 2.44110823e-03
 5.20616055e-01 2.35042185e-01 6.80487871e-01 1.05431974e-02
 9.99200940e-01 7.22423494e-02 4.92474139e-02 1.98337436e-03
 9.70007420e-01 7.01514423e-01 8.11764717e-01 9.92222488e-01
 1.35792017e-01 5.34464121e-01 9.00289416e-03 6.31688058e-01]  que se encuentra en:  24
Imagen de BD:  126  Imagen de consulta:  56 

Distancia:  3.018561  al vector:  [8.99816632e-01 9.99992132e-01 5.36407113e-01 1.03253782e-01
 5.79427004e-01 9.98490095e-01 9.75921750e-01 8.68930459e-01
 9.22012150e-01 4.49544907e-01 1.15820855e-01 3.87546033e-01
 3.15637249e-06 7.99122453e-01 7.28229284e-02 7.63832510e-01
 9.99626338e-01 9.99999523e-01 1.18303299e-03 8.42934012e-01
 9.68176305e-01 3.46213579e-04 6.61697984e-01 3.05106342e-02
 4.23807770e-01 3.28491330e-02 2.17127800e-03 2.03900039e-02
 6.66430116e-01 3.22944522e-01 2.27921307e-02 6.32048547e-01
 9.96753693e-01 7.05995717e-06 4.19017673e-02 2.75357991e-01
 8.90103221e-01 9.92387056e-01 8.18544213e-05 9.97362018e-01
 9.99947608e-01 9.60922122e-01 9.48593378e-01 9.99967754e-01
 6.47276640e-03 5.98788261e-04 4.15600955e-01 5.58517396e-01
 9.99989867e-01 7.08522141e-01 1.89020932e-02 9.18849587e-01
 3.46989334e-01 2.96639889e-01 8.82758915e-01 1.01861835e-01
 3.91249955e-02 1.40340567e-01 7.71146417e-02 2.87851691e-03
 1.24873996e-01 4.10675853e-01 8.70907307e-03 4.03118134e-03
 9.99359846e-01 9.02256072e-01 2.31166720e-01 7.12194145e-01
 9.99999702e-01 2.01587975e-02 9.91984963e-01 8.17850888e-01
 4.43278283e-01 3.45580995e-01 4.22863752e-01 6.87484324e-01
 9.99980450e-01 5.15673399e-01 7.62317061e-01 2.54030824e-02
 9.78609920e-03 9.69579995e-01 8.73679519e-01 1.21871531e-02
 8.44597816e-04 7.72774518e-02 4.89339292e-01 8.07595372e-01
 6.70752704e-01 2.31476486e-01 9.99129176e-01 3.52543294e-02
 8.88452888e-01 8.91570270e-01 5.45141935e-01 9.68025088e-01
 1.70001388e-03 1.72371688e-06 9.67986703e-01 5.69749177e-02
 6.53203726e-02 5.84282160e-01 1.37195766e-01 9.88923073e-01
 5.81444204e-02 6.16313219e-01 2.37405300e-04 9.98802423e-01
 6.64705575e-01 9.99157906e-01 9.99559999e-01 4.47157472e-01
 7.60622025e-02 4.45937157e-01 9.77582812e-01 3.21628451e-02
 5.77295184e-01 3.18021178e-02 3.23385656e-01 5.56031466e-07
 9.79548156e-01 9.52386975e-01 4.28642780e-01 3.23456824e-02
 1.31622937e-05 1.20421648e-02 1.19006127e-01 9.44476008e-01]  en la posicion:  163  desde el vector:  [9.95569587e-01 9.99964893e-01 1.95908546e-03 6.78408146e-03
 3.50089669e-02 9.35006022e-01 9.74362969e-01 9.80542064e-01
 5.29380560e-01 9.82974529e-01 4.48525846e-02 1.48141414e-01
 1.09952688e-03 2.78037816e-01 1.80254966e-01 3.33692431e-02
 9.99977708e-01 9.99999404e-01 1.23852789e-02 8.23702812e-01
 9.91551578e-01 6.38276339e-04 7.54365921e-02 1.59172624e-01
 3.26924622e-02 4.97877598e-03 6.11979663e-02 8.72707367e-03
 9.71638203e-01 9.04446483e-01 2.51795352e-02 9.01984811e-01
 9.91616845e-01 1.98299022e-05 1.12113088e-01 8.99700046e-01
 8.83163333e-01 9.95771468e-01 6.63101673e-04 6.69026256e-01
 9.99622047e-01 7.68744469e-01 7.31299937e-01 9.99996722e-01
 5.42821288e-02 2.35170126e-04 7.10358858e-01 9.91868019e-01
 9.99821424e-01 8.40949595e-01 6.35633826e-01 8.99604857e-01
 1.07717395e-01 6.65457249e-02 9.50200438e-01 2.33232737e-01
 6.63769603e-01 1.49900913e-02 1.40316010e-01 1.93347335e-01
 7.46008754e-03 1.55301154e-01 1.54398084e-02 2.97868252e-03
 9.99553919e-01 9.56150830e-01 5.85612178e-01 9.38575268e-01
 9.99960899e-01 2.04320550e-02 3.71864319e-01 9.74506378e-01
 8.64019096e-02 2.10720003e-02 8.54102850e-01 6.93796277e-02
 9.99463081e-01 8.68211269e-01 2.63399124e-01 2.96082199e-02
 3.75706404e-01 5.08991003e-01 8.22447002e-01 2.57614881e-01
 3.67963314e-03 3.32471728e-03 2.61622369e-02 9.29407835e-01
 9.45218801e-01 3.84357810e-01 9.99375582e-01 3.03480029e-03
 9.95524824e-01 7.55659223e-01 6.45175278e-02 9.96947765e-01
 7.27176666e-04 5.87999907e-08 9.94688272e-01 1.90313607e-01
 7.20015168e-03 9.83855724e-01 6.87408447e-03 9.94203627e-01
 3.18437815e-04 9.98456240e-01 3.99821997e-03 9.99943137e-01
 5.97405076e-01 9.85551119e-01 9.99952912e-01 1.73802674e-02
 8.95989060e-01 1.24000281e-01 7.10634470e-01 1.18185878e-01
 5.90193152e-01 2.50732899e-02 4.86020207e-01 2.31915214e-06
 9.98342633e-01 9.67115402e-01 6.26100183e-01 9.64046121e-02
 4.66522574e-03 3.53068113e-04 1.56673878e-01 9.99915719e-01]  que se encuentra en:  25
Imagen de BD:  63  Imagen de consulta:  63 

Distancia:  4.112519  al vector:  [9.92745876e-01 9.67943192e-01 5.05791962e-01 1.00000000e+00
 9.99275386e-01 9.99998748e-01 9.96021748e-01 5.14932108e-05
 3.10182571e-03 9.21608806e-01 1.34846568e-01 6.45532906e-02
 1.98971301e-01 9.10510063e-01 1.76488191e-01 9.99308228e-01
 7.69837413e-07 9.84458208e-01 1.00000000e+00 8.66202414e-01
 1.90383196e-03 2.28652656e-02 6.61864877e-03 9.48117668e-06
 1.18030012e-01 3.09141755e-01 2.18570232e-04 9.99979198e-01
 8.35214555e-01 9.25023342e-05 1.06855035e-02 5.67159712e-01
 2.94822156e-02 9.99998689e-01 6.09994471e-01 2.73752213e-03
 2.86528468e-03 2.12853283e-01 6.29355848e-01 9.99963760e-01
 9.71345544e-01 9.86252189e-01 5.50565958e-01 9.80694711e-01
 6.64114952e-03 1.00000000e+00 3.40736508e-02 9.60222602e-01
 7.65516961e-06 9.99999821e-01 4.06831503e-01 3.57799858e-01
 8.58008862e-04 9.30537045e-01 4.05857801e-01 2.33167410e-03
 8.76379967e-01 1.65235966e-01 9.89513516e-01 2.19699740e-03
 8.83150995e-01 8.51763606e-01 4.46629524e-03 3.20780277e-03
 9.41511393e-01 3.09251785e-01 7.76886940e-02 9.64353859e-01
 9.93644238e-01 7.62481391e-01 9.99902010e-01 6.10069990e-01
 5.26840389e-02 9.99748826e-01 1.34420097e-02 4.95434642e-01
 3.89066279e-01 9.29017544e-01 3.25139761e-02 5.22980094e-02
 7.59631097e-01 4.02837992e-04 1.00237131e-03 3.71222585e-01
 3.71278822e-02 8.16235542e-02 9.81272638e-01 9.95501399e-01
 1.57368481e-02 8.11414301e-01 9.28485155e-01 9.79469895e-01
 5.67674637e-03 2.28285789e-04 9.85288382e-01 3.58210201e-07
 9.34461832e-01 5.03331125e-02 2.96998024e-03 1.62560642e-01
 3.32517326e-02 9.94093239e-01 3.18268538e-02 9.99988437e-01
 9.16486740e-01 2.32607126e-03 1.39087230e-01 1.21445126e-07
 2.41618216e-01 3.39181632e-01 9.99900281e-01 9.76731718e-01
 4.97757941e-01 1.75661266e-01 9.34964657e-01 9.99999762e-01
 9.99962509e-01 9.39342976e-01 9.60594714e-02 9.97461319e-01
 8.66211057e-02 6.57455325e-01 2.21251488e-01 6.53797388e-03
 6.52528107e-01 9.99999523e-01 9.99999046e-01 1.68721974e-02]  en la posicion:  34  desde el vector:  [9.9994481e-01 9.0413344e-01 8.9544606e-01 9.9999070e-01 5.8120489e-04
 9.9902833e-01 8.3727437e-01 2.4145544e-03 3.3762130e-06 9.9582720e-01
 1.0641187e-02 1.2684295e-01 3.4274822e-01 6.2652540e-01 8.0724061e-03
 5.9091598e-01 1.3532043e-03 4.1814893e-02 1.0000000e+00 7.4845910e-01
 5.1081371e-01 3.8037896e-03 5.1803678e-02 2.5469065e-04 2.0719272e-01
 4.9803719e-01 1.6164541e-02 8.8057160e-01 8.4913594e-01 2.3036510e-02
 2.9746026e-02 3.7714243e-03 9.2783028e-01 9.9999857e-01 4.8073512e-01
 4.0999117e-01 4.4435859e-03 3.3126771e-03 3.2140315e-02 9.9561054e-01
 7.1351409e-02 5.2333498e-01 9.7063917e-01 7.2471875e-01 9.8771751e-03
 9.9999690e-01 3.8027388e-01 9.9923348e-01 6.1136548e-06 1.0000000e+00
 1.3567528e-01 9.6170449e-01 3.6343932e-04 2.7269515e-01 3.4855998e-01
 9.5336801e-01 9.9999905e-01 1.4385283e-03 9.3132520e-01 4.8726201e-02
 7.4429870e-01 9.1550064e-01 3.2194257e-03 1.9549853e-01 9.9145246e-01
 6.2090003e-01 1.8077400e-01 9.9397242e-02 9.4819826e-01 9.8872602e-01
 9.6402502e-01 8.1135368e-01 4.3433905e-03 9.1494107e-01 9.1516137e-01
 9.8937964e-01 5.5205822e-04 4.4168335e-01 6.9439411e-04 3.8528532e-02
 9.5214814e-01 1.7582406e-06 6.0855587e-06 4.2857951e-01 1.0275313e-01
 4.8316121e-03 8.7389672e-01 8.7736523e-01 6.3298965e-01 9.1790819e-01
 9.9746227e-01 6.5004659e-01 1.9249973e-01 4.9880147e-04 8.5746360e-01
 1.9196281e-05 9.9436617e-01 3.8933483e-01 1.9161844e-01 7.2911203e-01
 8.0093044e-01 9.3784952e-01 8.9785546e-02 9.9991822e-01 1.8972385e-01
 3.8359404e-01 9.9726319e-01 9.0202689e-04 4.9780262e-01 9.9255288e-01
 9.9981904e-01 4.2730188e-01 3.5880572e-01 9.9289298e-01 9.8835015e-01
 9.9999982e-01 9.9998939e-01 9.9010360e-01 8.7553859e-01 9.9796337e-01
 6.8602860e-03 4.9043426e-01 9.1178459e-01 1.1682510e-03 9.9593228e-01
 9.9956167e-01 9.9999642e-01 9.9705118e-01]  que se encuentra en:  26
Imagen de BD:  13  Imagen de consulta:  66 

Distancia:  3.5658474  al vector:  [9.92745876e-01 9.67943192e-01 5.05791962e-01 1.00000000e+00
 9.99275386e-01 9.99998748e-01 9.96021748e-01 5.14932108e-05
 3.10182571e-03 9.21608806e-01 1.34846568e-01 6.45532906e-02
 1.98971301e-01 9.10510063e-01 1.76488191e-01 9.99308228e-01
 7.69837413e-07 9.84458208e-01 1.00000000e+00 8.66202414e-01
 1.90383196e-03 2.28652656e-02 6.61864877e-03 9.48117668e-06
 1.18030012e-01 3.09141755e-01 2.18570232e-04 9.99979198e-01
 8.35214555e-01 9.25023342e-05 1.06855035e-02 5.67159712e-01
 2.94822156e-02 9.99998689e-01 6.09994471e-01 2.73752213e-03
 2.86528468e-03 2.12853283e-01 6.29355848e-01 9.99963760e-01
 9.71345544e-01 9.86252189e-01 5.50565958e-01 9.80694711e-01
 6.64114952e-03 1.00000000e+00 3.40736508e-02 9.60222602e-01
 7.65516961e-06 9.99999821e-01 4.06831503e-01 3.57799858e-01
 8.58008862e-04 9.30537045e-01 4.05857801e-01 2.33167410e-03
 8.76379967e-01 1.65235966e-01 9.89513516e-01 2.19699740e-03
 8.83150995e-01 8.51763606e-01 4.46629524e-03 3.20780277e-03
 9.41511393e-01 3.09251785e-01 7.76886940e-02 9.64353859e-01
 9.93644238e-01 7.62481391e-01 9.99902010e-01 6.10069990e-01
 5.26840389e-02 9.99748826e-01 1.34420097e-02 4.95434642e-01
 3.89066279e-01 9.29017544e-01 3.25139761e-02 5.22980094e-02
 7.59631097e-01 4.02837992e-04 1.00237131e-03 3.71222585e-01
 3.71278822e-02 8.16235542e-02 9.81272638e-01 9.95501399e-01
 1.57368481e-02 8.11414301e-01 9.28485155e-01 9.79469895e-01
 5.67674637e-03 2.28285789e-04 9.85288382e-01 3.58210201e-07
 9.34461832e-01 5.03331125e-02 2.96998024e-03 1.62560642e-01
 3.32517326e-02 9.94093239e-01 3.18268538e-02 9.99988437e-01
 9.16486740e-01 2.32607126e-03 1.39087230e-01 1.21445126e-07
 2.41618216e-01 3.39181632e-01 9.99900281e-01 9.76731718e-01
 4.97757941e-01 1.75661266e-01 9.34964657e-01 9.99999762e-01
 9.99962509e-01 9.39342976e-01 9.60594714e-02 9.97461319e-01
 8.66211057e-02 6.57455325e-01 2.21251488e-01 6.53797388e-03
 6.52528107e-01 9.99999523e-01 9.99999046e-01 1.68721974e-02]  en la posicion:  34  desde el vector:  [9.7777677e-01 8.7146401e-01 7.0416123e-02 9.9998122e-01 6.7392957e-01
 9.9487400e-01 9.1775864e-01 1.3130903e-04 3.0352473e-03 9.9653518e-01
 2.7759767e-01 9.8931515e-01 7.4246013e-01 8.3295244e-01 7.3683500e-02
 9.6189135e-01 1.2093186e-03 8.6862129e-01 9.9997962e-01 9.7818899e-01
 9.1464150e-01 8.7410510e-03 1.5262657e-01 9.3987584e-04 5.7544112e-03
 3.9738655e-02 4.7044158e-03 5.8481175e-01 6.5392423e-01 7.0680082e-02
 4.5463154e-01 1.7898649e-02 5.7415271e-01 9.9854773e-01 3.3045977e-01
 2.1617183e-01 1.9416633e-05 2.1763870e-01 8.3673000e-04 9.9339342e-01
 4.5321038e-01 9.1887444e-01 9.7552055e-01 9.8089194e-01 3.9985180e-03
 9.9980336e-01 7.1698648e-01 9.9248266e-01 1.9858778e-03 9.9992657e-01
 7.9430568e-01 6.7531848e-01 1.1776686e-03 3.4553504e-01 6.0632205e-01
 1.6970873e-01 9.9584913e-01 3.1045175e-01 9.8826861e-01 1.1036247e-02
 7.6086092e-01 4.5100397e-01 2.0888847e-01 3.3308864e-03 9.3453783e-01
 9.6804905e-01 7.0409954e-02 3.6078170e-01 9.9285185e-01 6.3842750e-01
 9.3351543e-01 9.1803581e-01 6.2993622e-01 9.0770853e-01 4.4615835e-02
 6.5285695e-01 2.2409230e-02 5.5847543e-01 8.3136857e-03 9.8056793e-03
 3.9787197e-01 1.5054345e-03 1.1202395e-03 5.8084756e-01 7.6373816e-03
 4.3845475e-03 7.8176022e-01 9.9960297e-01 1.0958773e-01 9.3128818e-01
 9.9983758e-01 7.4205488e-01 7.3411256e-02 2.8098017e-02 3.2738119e-01
 9.9071860e-04 9.4608271e-01 1.7025772e-01 8.2563514e-01 4.4522476e-01
 1.4485598e-02 9.5845473e-01 6.2990218e-02 9.9976277e-01 1.3541770e-01
 2.9853392e-01 6.9054991e-01 2.1125078e-03 4.9419671e-02 8.9730960e-01
 9.9287641e-01 7.8249782e-02 9.0544438e-01 7.1829975e-01 9.6990204e-01
 9.9997914e-01 9.9995506e-01 9.1539919e-01 7.0567727e-02 8.6845410e-01
 3.6162296e-01 6.2763250e-01 3.9518112e-01 4.7947153e-01 7.1905190e-01
 9.9988115e-01 9.9999177e-01 9.3294519e-01]  que se encuentra en:  27
Imagen de BD:  13  Imagen de consulta:  66 

Distancia:  3.1553442  al vector:  [9.92745876e-01 9.67943192e-01 5.05791962e-01 1.00000000e+00
 9.99275386e-01 9.99998748e-01 9.96021748e-01 5.14932108e-05
 3.10182571e-03 9.21608806e-01 1.34846568e-01 6.45532906e-02
 1.98971301e-01 9.10510063e-01 1.76488191e-01 9.99308228e-01
 7.69837413e-07 9.84458208e-01 1.00000000e+00 8.66202414e-01
 1.90383196e-03 2.28652656e-02 6.61864877e-03 9.48117668e-06
 1.18030012e-01 3.09141755e-01 2.18570232e-04 9.99979198e-01
 8.35214555e-01 9.25023342e-05 1.06855035e-02 5.67159712e-01
 2.94822156e-02 9.99998689e-01 6.09994471e-01 2.73752213e-03
 2.86528468e-03 2.12853283e-01 6.29355848e-01 9.99963760e-01
 9.71345544e-01 9.86252189e-01 5.50565958e-01 9.80694711e-01
 6.64114952e-03 1.00000000e+00 3.40736508e-02 9.60222602e-01
 7.65516961e-06 9.99999821e-01 4.06831503e-01 3.57799858e-01
 8.58008862e-04 9.30537045e-01 4.05857801e-01 2.33167410e-03
 8.76379967e-01 1.65235966e-01 9.89513516e-01 2.19699740e-03
 8.83150995e-01 8.51763606e-01 4.46629524e-03 3.20780277e-03
 9.41511393e-01 3.09251785e-01 7.76886940e-02 9.64353859e-01
 9.93644238e-01 7.62481391e-01 9.99902010e-01 6.10069990e-01
 5.26840389e-02 9.99748826e-01 1.34420097e-02 4.95434642e-01
 3.89066279e-01 9.29017544e-01 3.25139761e-02 5.22980094e-02
 7.59631097e-01 4.02837992e-04 1.00237131e-03 3.71222585e-01
 3.71278822e-02 8.16235542e-02 9.81272638e-01 9.95501399e-01
 1.57368481e-02 8.11414301e-01 9.28485155e-01 9.79469895e-01
 5.67674637e-03 2.28285789e-04 9.85288382e-01 3.58210201e-07
 9.34461832e-01 5.03331125e-02 2.96998024e-03 1.62560642e-01
 3.32517326e-02 9.94093239e-01 3.18268538e-02 9.99988437e-01
 9.16486740e-01 2.32607126e-03 1.39087230e-01 1.21445126e-07
 2.41618216e-01 3.39181632e-01 9.99900281e-01 9.76731718e-01
 4.97757941e-01 1.75661266e-01 9.34964657e-01 9.99999762e-01
 9.99962509e-01 9.39342976e-01 9.60594714e-02 9.97461319e-01
 8.66211057e-02 6.57455325e-01 2.21251488e-01 6.53797388e-03
 6.52528107e-01 9.99999523e-01 9.99999046e-01 1.68721974e-02]  en la posicion:  34  desde el vector:  [9.9982750e-01 9.9756521e-01 2.4817631e-01 9.9999928e-01 9.3312424e-01
 9.9998647e-01 6.3253373e-01 2.4359233e-05 1.3933182e-03 1.4695492e-01
 1.0216922e-02 9.2070925e-01 1.1539280e-02 9.9624163e-01 4.6721905e-02
 9.9878395e-01 3.2165692e-05 9.9560452e-01 9.9999768e-01 2.9671037e-01
 6.3068271e-03 6.3005090e-04 5.5411160e-03 8.7416976e-07 6.6707391e-01
 2.7871102e-02 5.4007487e-06 9.9039483e-01 9.4078511e-02 3.0581628e-05
 3.5975268e-01 2.2449109e-01 8.8267988e-01 9.9991375e-01 9.1016269e-01
 8.5684061e-03 1.5133619e-04 5.4002166e-02 4.1134110e-05 9.9943304e-01
 9.5050824e-01 8.3227408e-01 9.7769481e-01 9.9706399e-01 6.5509975e-03
 9.9999839e-01 9.1019213e-01 9.8831713e-01 6.5973401e-04 9.9999279e-01
 9.7393227e-01 9.7521091e-01 2.2383332e-03 7.0441896e-01 3.2247451e-01
 4.5368075e-04 9.7153425e-01 5.1088536e-01 9.8780537e-01 6.3652396e-03
 6.1000472e-01 7.0486003e-01 2.8966486e-02 6.5348744e-03 6.4605641e-01
 5.0334847e-01 6.2077343e-03 9.5713830e-01 9.9995053e-01 2.6409948e-01
 9.9995852e-01 7.3473585e-01 1.8202573e-02 6.0854983e-01 9.5455050e-03
 5.9329158e-01 4.3205699e-01 9.7633100e-01 9.6445978e-03 3.9548904e-02
 3.5919154e-01 2.9537261e-02 1.3630927e-02 7.7761537e-01 4.6702653e-02
 7.9187751e-04 9.9716306e-01 9.9708575e-01 6.6628844e-02 8.1902486e-01
 9.9978280e-01 2.1161500e-01 6.1092645e-02 4.3243894e-05 3.2661229e-01
 1.7123602e-05 9.9618685e-01 1.4576018e-02 1.3099313e-03 2.8393093e-01
 2.0496398e-01 9.5796478e-01 3.3402115e-02 9.9999678e-01 2.5956720e-01
 2.6287615e-02 6.3266999e-01 4.7567760e-06 1.7948508e-02 8.4855390e-01
 9.9949670e-01 6.9103742e-01 8.6909860e-01 1.6761395e-01 9.9009126e-01
 9.9999958e-01 9.9993598e-01 8.3972573e-01 2.6870117e-01 7.2267711e-01
 5.3868890e-03 8.1899375e-01 2.7429524e-01 1.7743975e-02 1.0160437e-01
 9.9999893e-01 9.9999845e-01 9.6148729e-02]  que se encuentra en:  28
Imagen de BD:  13  Imagen de consulta:  66 

Distancia:  4.3034306  al vector:  [6.0189664e-03 9.9999976e-01 6.8695992e-02 9.2262965e-01 1.5153092e-01
 4.5183098e-01 4.3447316e-02 1.4243364e-02 9.9808598e-01 5.5443680e-01
 4.9543351e-02 9.0400946e-01 3.8582486e-05 2.3412198e-02 9.7661901e-01
 9.2189980e-01 3.7406206e-02 9.9984014e-01 9.3635082e-01 6.2178141e-01
 9.0392751e-01 2.1296740e-04 9.9969906e-01 1.2700766e-02 4.0263635e-01
 3.9823398e-01 2.0904660e-02 3.9519477e-01 3.1646079e-01 3.0064806e-01
 7.7998996e-01 9.9553478e-01 9.6416509e-01 4.3098658e-02 3.5732371e-01
 1.9297457e-01 3.5107136e-04 9.3983686e-01 1.7150062e-01 9.2207503e-01
 9.9990785e-01 9.9498409e-01 9.7993922e-01 9.9813265e-01 4.4113099e-06
 7.1032286e-01 6.1505562e-01 9.9599361e-04 8.8212574e-01 9.9977595e-01
 3.8511482e-01 4.4375366e-01 1.1514127e-03 7.8011632e-02 9.9611676e-01
 7.0287186e-01 1.8361211e-04 8.2784456e-01 2.7977341e-01 1.9878498e-06
 8.6795390e-03 4.6068877e-02 4.6253502e-03 3.1468087e-01 9.2561412e-01
 7.8251678e-01 6.0478222e-01 8.0147398e-01 9.9999976e-01 3.6654174e-03
 9.8003900e-01 1.1750758e-03 9.9569488e-01 8.8487947e-01 8.2239062e-01
 5.5581033e-02 9.8275292e-01 3.8358581e-01 9.1332579e-01 8.4414458e-01
 1.0174540e-01 1.4133203e-01 7.2197020e-03 3.8989484e-03 7.8794026e-01
 4.3593317e-02 6.1319077e-01 8.1098258e-01 8.7761599e-01 2.1598452e-01
 9.2254424e-01 9.9378920e-01 9.4239008e-01 9.9102581e-01 8.9298105e-01
 2.5368690e-02 2.1641552e-02 1.3005733e-04 9.5384371e-01 2.9875880e-01
 7.1387386e-01 6.3902140e-04 9.9935794e-01 5.3850418e-01 4.2008221e-02
 5.5790561e-01 4.0646887e-01 1.9412935e-03 8.9939672e-01 1.2211594e-01
 9.9850500e-01 1.4235181e-01 8.9290768e-02 8.4427124e-01 4.9294263e-02
 8.8530481e-01 9.9995744e-01 4.4060558e-02 2.1550745e-02 1.3506532e-02
 9.8354995e-01 9.9449062e-01 9.5808625e-01 8.2682014e-02 6.5846157e-01
 9.7657722e-01 4.4132438e-01 5.4889637e-01]  en la posicion:  173  desde el vector:  [9.2196614e-02 9.9997318e-01 8.7832004e-01 9.9990439e-01 3.1188875e-02
 8.9699984e-01 5.9613496e-01 8.2209706e-04 9.9718976e-01 9.9259281e-01
 1.4310181e-02 7.2414130e-01 2.1260977e-03 8.0432296e-01 6.7011440e-01
 6.5780276e-01 3.0795783e-02 9.9967396e-01 8.2913983e-01 1.1618477e-01
 9.0022904e-01 4.5571840e-05 9.9877352e-01 2.6773399e-01 6.1502457e-03
 1.6269267e-02 1.2967885e-03 2.6823282e-01 1.2657285e-01 1.8673792e-01
 2.6472557e-01 3.1933516e-02 9.9402547e-01 2.5645494e-03 3.5422742e-03
 6.5031737e-02 1.5544891e-04 9.9505472e-01 3.6780882e-01 6.7432892e-01
 9.9999857e-01 9.9676859e-01 3.7377203e-01 9.9977249e-01 1.6174333e-08
 8.3074689e-01 2.8640622e-01 5.2705407e-03 8.1083632e-01 9.9971652e-01
 2.3360595e-01 4.1141689e-02 9.5260446e-05 5.8016300e-02 9.8358577e-01
 5.6764960e-02 4.2152405e-04 1.3175607e-02 9.9760330e-01 1.5826981e-08
 9.5553577e-01 9.1867697e-01 1.7210898e-01 3.8034919e-01 3.0641043e-01
 9.7408307e-01 2.9292202e-01 9.0259790e-01 9.9999863e-01 2.0518988e-02
 7.1841788e-01 3.3426207e-01 9.9888289e-01 1.8478334e-03 5.7216829e-01
 1.7314970e-01 9.9305046e-01 9.0883744e-01 2.5260955e-01 4.8170537e-01
 9.5389432e-01 7.0374310e-03 5.7625484e-01 8.8260174e-03 4.6466240e-01
 2.6133066e-01 9.3836784e-03 9.9978614e-01 6.6731572e-03 1.6064653e-01
 8.3295763e-01 9.9679887e-01 2.4605691e-03 9.9810791e-01 4.6864882e-01
 2.6497543e-03 3.2374859e-03 3.3643228e-05 9.9054694e-01 9.3596244e-01
 7.0240653e-01 1.7397612e-02 9.9032867e-01 3.5851306e-01 6.3751584e-01
 5.4945487e-01 2.1653447e-01 1.6631067e-02 7.9250002e-01 5.4218113e-02
 9.9983513e-01 3.8429141e-02 4.8198587e-01 3.1347626e-01 1.4307344e-01
 9.5037574e-01 9.9999833e-01 1.4623472e-01 6.7137873e-01 2.1089613e-02
 9.8702800e-01 9.2548251e-02 1.0359475e-01 7.9011726e-01 2.9974192e-01
 9.9776924e-01 9.5618933e-01 9.7795987e-01]  que se encuentra en:  29
Imagen de BD:  72  Imagen de consulta:  76 

Distancia:  2.7476044  al vector:  [8.89669478e-01 9.99997497e-01 2.70470679e-01 9.99998212e-01
 6.22138977e-02 9.99986708e-01 9.77578640e-01 6.54975224e-07
 9.98693109e-01 1.80131197e-03 1.03621215e-01 3.01323742e-01
 7.06193714e-06 9.48661447e-01 8.90657842e-01 9.99998808e-01
 4.71892953e-03 9.99912977e-01 9.97191072e-01 2.15220898e-01
 1.15914077e-01 5.67346811e-04 9.99954402e-01 2.57521868e-04
 9.98635888e-01 7.02407956e-03 3.97706099e-05 9.99660432e-01
 1.18794411e-01 4.38302755e-04 9.99419451e-01 1.27033889e-02
 7.55025744e-01 1.32889777e-01 7.73127973e-02 3.78019149e-06
 1.41620636e-04 3.63055468e-02 5.55927818e-06 9.68229175e-01
 9.99999404e-01 9.74044085e-01 9.18008029e-01 9.91660893e-01
 7.46392956e-08 9.98736739e-01 1.97880447e-01 7.08282896e-05
 4.39405590e-01 9.99958873e-01 8.12578380e-01 4.21009064e-02
 6.19515777e-03 2.21591860e-01 3.15895975e-02 3.10384631e-02
 8.29136934e-06 6.91363215e-03 9.98998106e-01 6.69499229e-07
 7.19460309e-01 8.12331915e-01 9.46948528e-01 9.35320258e-02
 1.51114464e-02 9.87384379e-01 2.29652196e-01 9.50043440e-01
 9.99999642e-01 1.27619207e-02 9.99724746e-01 8.00286889e-01
 9.88556504e-01 1.64676309e-02 4.74992961e-01 7.89433718e-04
 9.99940515e-01 7.65297830e-01 2.34530658e-01 1.62083775e-01
 7.99478829e-01 9.95111942e-01 9.99177158e-01 9.87147033e-01
 2.89164782e-02 6.03140593e-02 6.10583186e-01 9.99967575e-01
 2.33161973e-06 8.81689787e-03 9.78042662e-01 1.79694772e-01
 2.95488644e-05 7.25949883e-01 3.15630674e-01 8.28981400e-04
 9.65502977e-01 7.67105030e-06 1.76268816e-02 4.08647537e-01
 6.17100596e-01 1.71196461e-03 9.99663949e-01 9.99318123e-01
 7.43903399e-01 4.52644854e-05 9.97175336e-01 4.66883183e-04
 9.99304771e-01 3.80634665e-02 9.99953389e-01 2.94869542e-02
 8.48831534e-01 4.01622206e-01 8.46336842e-01 9.98830795e-01
 9.99911129e-01 4.25630957e-01 6.90239489e-01 4.44501638e-04
 1.31155550e-02 1.13553673e-01 5.52240074e-01 9.96711254e-01
 1.98890502e-05 9.99927759e-01 9.99114931e-01 4.43909973e-01]  en la posicion:  177  desde el vector:  [8.07249665e-01 9.99935150e-01 6.87049091e-01 9.99877334e-01
 6.22474551e-02 9.96984243e-01 9.12406087e-01 4.52846289e-04
 9.99877393e-01 9.56565142e-04 3.18442285e-02 6.68901205e-03
 5.45228504e-05 8.47736001e-01 1.18872255e-01 9.99961257e-01
 2.05523372e-02 9.99992549e-01 9.96065021e-01 2.27207690e-01
 2.65251964e-01 1.82732940e-03 9.99901295e-01 4.45690751e-03
 9.96072948e-01 4.09180820e-02 2.86984444e-03 9.99778330e-01
 8.44613433e-01 1.06955469e-02 9.98000741e-01 2.21510291e-01
 9.80241776e-01 4.81244236e-01 5.12930989e-01 4.00529207e-05
 4.68224287e-04 1.54342651e-02 2.04280204e-05 9.94330645e-01
 9.99976516e-01 9.80159342e-01 7.11603045e-01 9.97873127e-01
 1.25912329e-05 9.44968462e-01 5.17691672e-02 1.69634819e-04
 1.88803375e-01 9.99710798e-01 6.49149895e-01 9.46730077e-02
 5.19815087e-03 5.81944346e-01 1.01808310e-02 1.88738704e-02
 9.82438087e-06 1.74879134e-02 9.71360445e-01 3.14979770e-05
 6.50397837e-02 4.45838749e-01 9.82947648e-01 3.31159145e-01
 4.82569337e-02 8.77443433e-01 3.53583395e-01 6.60369754e-01
 9.99974132e-01 5.20121753e-01 9.87505317e-01 6.32315993e-01
 5.35720706e-01 7.00843334e-03 9.20475185e-01 1.01816654e-03
 9.99984801e-01 1.67010725e-01 6.17996097e-01 5.52623868e-01
 9.51452017e-01 9.96577024e-01 9.99591589e-01 9.97105837e-01
 5.90242088e-01 3.99267673e-03 8.65933001e-01 9.99972343e-01
 2.24776613e-05 7.68274069e-04 9.64813113e-01 4.61016297e-02
 1.51224136e-02 5.61597228e-01 9.40251887e-01 2.43994594e-03
 9.44037676e-01 3.96808559e-07 2.77203918e-02 3.68206739e-01
 5.13843775e-01 5.04121184e-03 9.99328911e-01 9.49613035e-01
 1.70808017e-01 7.43145356e-05 8.33402991e-01 3.69486213e-03
 9.98183191e-01 6.18113935e-01 9.99891162e-01 6.16529644e-01
 3.77986312e-01 2.49490827e-01 4.61573303e-02 9.99666929e-01
 9.99702096e-01 9.46921110e-03 9.14024532e-01 2.66523875e-05
 5.87344170e-03 7.44128823e-02 9.77962613e-01 9.97962415e-01
 1.28090382e-04 9.53661680e-01 9.99452353e-01 5.38306534e-02]  que se encuentra en:  30
Imagen de BD:  76  Imagen de consulta:  77 

Distancia:  3.2028697  al vector:  [8.1796557e-01 1.7481178e-02 9.0719986e-01 2.2246337e-01 2.3481250e-03
 3.0393004e-03 8.7981999e-02 5.8039528e-01 7.7077758e-01 2.4141052e-01
 5.1084489e-02 4.7069305e-01 9.9193645e-01 2.3647153e-01 1.7937180e-01
 7.8279436e-02 4.3808177e-01 2.0286739e-03 2.1995234e-01 7.3315835e-01
 2.6800483e-02 8.7978029e-01 9.8714304e-01 9.3268728e-01 8.8879555e-01
 4.8377246e-02 9.6307099e-01 9.0271086e-02 1.9775033e-03 9.9232960e-01
 9.8251998e-01 6.4679086e-02 1.9666567e-01 9.9952960e-01 8.6871076e-01
 9.6775258e-01 1.4618275e-01 4.9792081e-02 7.4381816e-01 6.8879145e-01
 4.2120755e-02 9.1918278e-01 9.9643767e-01 4.3656230e-03 7.0789963e-01
 3.5780525e-01 7.3708725e-01 1.4048815e-03 1.8569231e-03 4.7871560e-02
 5.5948418e-01 9.6265328e-01 7.3995203e-01 5.6045055e-03 9.0980530e-03
 9.8382968e-01 8.1824183e-01 9.8830390e-01 4.1686523e-01 3.0836725e-01
 9.9039793e-01 3.9377373e-01 9.9999225e-01 9.2698455e-01 1.5344918e-03
 5.6595612e-01 7.9731500e-01 9.3814129e-01 3.8384497e-03 9.6595687e-01
 1.0445416e-03 4.4020396e-01 3.9714810e-01 1.5027693e-01 2.5548786e-01
 5.2486736e-01 1.1253744e-02 7.8209245e-01 5.7182175e-01 6.8148738e-01
 9.1365302e-01 1.0460043e-01 9.7850466e-01 9.9917156e-01 9.6404719e-01
 9.8956227e-01 6.0087001e-01 9.9161142e-01 5.6803846e-01 9.7580492e-02
 2.4220341e-01 2.2080693e-01 6.4541614e-01 2.8083175e-02 1.6392988e-01
 2.3665935e-01 9.8877072e-01 9.8940754e-01 2.9152244e-02 7.8091919e-03
 2.1223912e-01 2.4204373e-02 9.9834001e-01 6.2016577e-02 6.8254268e-01
 6.5945232e-01 9.9906242e-01 6.5912205e-01 3.7766916e-01 1.7904550e-02
 3.4318268e-03 3.1077206e-01 4.1635826e-01 9.2445201e-01 1.9048959e-02
 2.9185030e-01 6.9574863e-01 5.3823411e-02 6.7017251e-01 4.7631887e-01
 2.5635153e-02 9.1177225e-04 9.4728231e-01 9.9969399e-01 9.9825352e-01
 3.0927902e-01 2.8004074e-01 5.2845061e-02]  en la posicion:  175  desde el vector:  [4.93392676e-01 5.52769601e-02 9.09527183e-01 3.04226875e-02
 2.03666389e-02 2.49142195e-05 5.78641891e-04 9.26549256e-01
 9.96709526e-01 9.61999595e-02 4.45335925e-01 9.32422101e-01
 9.90435719e-01 6.91987574e-02 8.21416616e-01 6.39617443e-04
 8.85667682e-01 1.81710899e-01 4.46602702e-03 8.10616612e-02
 7.67449379e-01 9.33113813e-01 9.99549866e-01 9.98924375e-01
 8.70097101e-01 3.95271301e-01 9.93906736e-01 2.83926725e-04
 2.45967507e-03 9.99960423e-01 9.96107578e-01 2.60543048e-01
 9.32358563e-01 8.33132505e-01 7.90907145e-01 9.88006711e-01
 2.20391363e-01 5.02170026e-02 6.76349580e-01 8.81485939e-02
 2.79727280e-02 9.83109474e-01 9.97699916e-01 6.00948930e-03
 4.70764250e-01 3.66553868e-05 3.24130535e-01 6.26683235e-04
 6.11210465e-02 2.11805105e-04 3.82712662e-01 8.78117919e-01
 4.78931546e-01 3.31139565e-02 1.05652392e-01 9.66819644e-01
 1.98039502e-01 8.85205626e-01 1.06531680e-02 1.69269562e-01
 5.53762197e-01 8.86030793e-02 9.99992013e-01 9.95817780e-01
 1.87159181e-02 4.58539158e-01 4.11766618e-01 9.99726355e-01
 3.07862133e-01 8.26142430e-01 1.12262060e-05 8.15182328e-02
 7.97309041e-01 1.34894252e-03 5.02084196e-02 4.13520396e-01
 1.51702762e-03 7.26783514e-01 7.92368054e-01 8.28581452e-01
 9.30069327e-01 9.78212714e-01 9.89798188e-01 7.42445171e-01
 9.05312002e-01 4.88885462e-01 9.81628537e-01 9.87387538e-01
 8.12364578e-01 6.55951142e-01 4.43325251e-01 1.21078551e-01
 9.98930871e-01 1.10895365e-01 5.42765856e-03 9.50645387e-01
 8.26442122e-01 9.81411576e-01 7.75536120e-01 4.66586947e-02
 8.96137774e-01 5.70297241e-04 9.99689698e-01 5.42026758e-03
 6.83922470e-01 9.90850091e-01 9.98786330e-01 9.76703584e-01
 1.19278014e-01 8.40426385e-02 1.95708871e-03 2.86795497e-02
 9.08808827e-01 7.56699920e-01 2.06977129e-01 2.56931186e-02
 5.91948688e-01 1.29368901e-03 3.86349857e-01 1.29068762e-01
 4.89389300e-02 6.98809028e-02 5.43270051e-01 9.99872684e-01
 8.79311323e-01 7.51835108e-03 3.73478830e-02 2.41321027e-02]  que se encuentra en:  31
Imagen de BD:  74  Imagen de consulta:  78 

Distancia:  3.7366235  al vector:  [1.0078847e-03 9.9950397e-01 8.0483460e-01 3.8963407e-02 3.2403344e-01
 5.0640206e-06 3.6851466e-03 9.9462068e-01 9.9998182e-01 9.7209263e-01
 2.7425939e-01 5.3233385e-01 1.5264750e-04 5.1556277e-01 7.1041608e-01
 8.1535518e-02 9.3219554e-01 9.9821550e-01 1.1165320e-04 2.2092760e-02
 9.9984539e-01 2.7411789e-02 9.9995989e-01 9.9959052e-01 2.5178492e-03
 5.3659081e-04 9.7313619e-01 1.2645188e-01 3.4205616e-03 9.9999714e-01
 6.9386762e-01 9.9438488e-01 9.9379474e-01 8.1255857e-05 4.2716575e-01
 7.4658859e-01 2.8201804e-01 9.9247563e-01 9.0062445e-01 8.2252491e-01
 9.9945349e-01 9.7734678e-01 8.6192536e-01 9.9760562e-01 1.2169778e-03
 4.3960214e-03 9.1791093e-01 3.3401682e-06 9.9518973e-01 9.8375410e-02
 4.4277295e-01 9.7866297e-01 3.0323386e-02 7.8928590e-02 8.9776385e-01
 9.9376076e-01 2.0200908e-03 6.8175256e-02 9.8732531e-01 2.4984777e-03
 2.1739799e-01 9.8682183e-01 8.3673000e-01 1.7814320e-01 1.3111255e-01
 6.3860035e-01 1.3810977e-01 9.0443963e-01 9.9796474e-01 1.2918115e-02
 5.0375838e-05 4.9786785e-01 9.9998897e-01 2.1089691e-01 3.3098036e-01
 9.4635105e-01 9.8211920e-01 9.8514521e-01 5.7675451e-02 2.0838264e-01
 4.2987275e-01 8.6798048e-01 2.4921986e-01 1.0155618e-02 6.9442868e-01
 1.0688493e-01 1.0804176e-02 7.5755870e-01 5.0848180e-01 9.7259879e-04
 1.1177957e-03 9.9878371e-01 9.6700424e-01 9.9995685e-01 1.8014270e-01
 9.9382949e-01 2.8901935e-02 1.4858246e-03 9.9962604e-01 2.0031267e-01
 8.5823268e-02 7.8064203e-04 9.9996781e-01 1.6197562e-04 1.0891914e-01
 7.8035843e-01 3.2185912e-02 9.7652006e-01 1.8829265e-01 1.4092055e-01
 4.5946017e-01 1.5901718e-01 9.7104990e-01 5.5617511e-01 2.8052801e-01
 4.2822957e-04 9.9663782e-01 6.6240251e-01 8.3541870e-04 8.9100003e-04
 9.9853718e-01 4.9589881e-01 6.5078831e-01 7.4725211e-01 9.9518836e-01
 4.2337120e-02 7.1858764e-03 9.8024470e-01]  en la posicion:  151  desde el vector:  [9.51048136e-02 9.99960542e-01 9.36389565e-01 1.98031425e-01
 2.28223979e-01 3.84360850e-02 5.80309629e-02 4.08057570e-02
 9.99989390e-01 2.39539742e-02 5.41236401e-02 3.40128273e-01
 6.52558947e-06 3.46421421e-01 7.13823199e-01 9.14165974e-01
 6.39365613e-01 9.99515653e-01 1.29768252e-03 5.67035675e-02
 9.42363262e-01 3.07884812e-03 9.99997258e-01 9.25994158e-01
 3.12806189e-01 3.04250717e-02 2.44221807e-01 3.28168869e-02
 1.91725403e-01 9.96719241e-01 9.97332692e-01 6.57199264e-01
 9.99898791e-01 1.16019386e-04 4.42802012e-01 3.34633827e-01
 4.27523851e-02 9.18378472e-01 1.18932307e-01 9.57320273e-01
 9.99994516e-01 9.96030748e-01 2.44982749e-01 9.99701560e-01
 4.75468710e-07 1.27302408e-02 9.67575431e-01 5.07434436e-07
 9.96367574e-01 7.09259868e-01 6.97441101e-01 8.09799433e-01
 2.12994516e-02 3.28099132e-02 8.38482738e-01 9.63577390e-01
 3.83120750e-05 7.13697076e-03 9.88712907e-01 9.86682608e-07
 9.39917207e-01 5.42690516e-01 9.81532991e-01 4.09549475e-02
 4.12441730e-01 9.37846839e-01 7.03142107e-01 1.58352584e-01
 9.99978781e-01 1.22636557e-03 7.41315782e-02 9.79734778e-01
 9.99498069e-01 1.74826413e-01 3.04126740e-01 9.91191030e-01
 9.97880101e-01 9.84538019e-01 7.33777404e-01 1.34385109e-01
 2.28335559e-01 9.96353447e-01 9.88536239e-01 1.60567462e-02
 9.90515351e-02 3.03888023e-02 4.72414613e-01 9.81491327e-01
 8.47065449e-03 3.10393572e-02 2.99414784e-01 9.56856489e-01
 1.88006282e-01 9.99445438e-01 1.97423846e-01 6.23374283e-01
 1.21581554e-02 8.36521387e-04 9.85608935e-01 1.97083384e-01
 1.33762360e-01 1.52379274e-04 9.99969780e-01 1.54979825e-02
 2.65640020e-03 5.70605695e-01 6.50278926e-01 9.12168145e-01
 9.34926987e-01 1.55311197e-01 9.83481526e-01 2.70691812e-02
 8.94498944e-01 7.99910128e-02 1.54821277e-01 4.66570258e-03
 9.99144018e-01 1.01840973e-01 1.42335296e-02 1.88565773e-05
 8.30824137e-01 4.13874894e-01 2.45600909e-01 9.89669621e-01
 1.39853656e-02 3.81955922e-01 1.46146715e-02 2.25616366e-01]  que se encuentra en:  32
Imagen de BD:  52  Imagen de consulta:  80 

Distancia:  4.20611  al vector:  [6.90981746e-03 9.99912143e-01 6.15294099e-01 2.27721575e-05
 8.60433102e-01 4.58449125e-04 5.24178803e-01 9.99850869e-01
 9.99410272e-01 9.99490678e-01 1.11793743e-04 9.35773671e-01
 3.73909235e-01 6.73082769e-01 3.19968849e-01 5.84584814e-06
 9.99958038e-01 9.99869227e-01 1.37141347e-03 5.97965062e-01
 9.98192847e-01 1.04728341e-03 9.98251081e-01 9.99987841e-01
 5.05685806e-04 1.99107826e-02 9.99749541e-01 1.96120964e-05
 5.00103831e-03 9.99492526e-01 4.03781851e-05 5.92091084e-01
 9.99943972e-01 2.54154205e-04 5.55589795e-03 9.99967694e-01
 9.99476492e-01 9.97748494e-01 9.99659181e-01 6.87971354e-01
 9.99913454e-01 1.59388185e-02 7.13039219e-01 9.99109864e-01
 8.17827167e-05 1.40238744e-05 9.51017618e-01 8.36908817e-04
 9.92389083e-01 2.27330625e-01 6.63856864e-02 7.68152297e-01
 9.67461467e-01 6.11234307e-01 9.97918904e-01 9.99675453e-01
 4.23252583e-04 4.45849061e-01 3.55184078e-01 6.80357218e-04
 9.31349635e-01 5.77320814e-01 2.28258967e-03 2.43670344e-01
 9.99997139e-01 7.31699288e-01 8.70521903e-01 9.54516649e-01
 9.99918818e-01 5.69105744e-01 5.95808029e-04 5.11592388e-01
 9.95947659e-01 4.01188821e-01 1.08120978e-01 8.90768290e-01
 9.99916196e-01 5.72538137e-01 3.99231613e-02 8.14288497e-01
 4.82408404e-02 2.63164639e-02 6.24141097e-03 6.39319420e-04
 9.94575977e-01 6.30476654e-01 1.63471699e-03 3.62131000e-03
 9.99998510e-01 1.96477771e-03 1.19666278e-01 9.93513227e-01
 9.89392877e-01 9.99256790e-01 2.32129991e-02 9.99990523e-01
 1.64317191e-02 1.81463361e-03 9.97745872e-01 1.05180174e-01
 6.86608791e-01 3.19558382e-03 9.98240471e-01 5.82724512e-02
 9.77339029e-01 9.99991536e-01 3.13474848e-05 9.99990404e-01
 8.03195536e-02 5.38063645e-02 9.99419451e-01 9.82536316e-01
 9.58685040e-01 1.22554004e-02 6.24433160e-03 2.33435631e-03
 2.55222559e-01 6.84420168e-01 9.50838447e-01 1.79201365e-04
 9.99471843e-01 4.03766811e-01 4.80207801e-03 6.47467375e-03
 9.97870207e-01 8.16742177e-05 1.81971192e-02 9.90347385e-01]  en la posicion:  71  desde el vector:  [2.06290960e-01 9.53782976e-01 8.31484795e-04 7.42309570e-01
 4.46562797e-01 1.68555945e-01 5.34005761e-01 9.99127030e-01
 2.83588886e-01 9.56267476e-01 1.12840533e-02 1.64588392e-02
 4.55465913e-03 9.88083601e-01 2.01828778e-02 2.97203362e-02
 6.54329777e-01 9.63334322e-01 3.80158424e-03 2.51500368e-01
 4.45089757e-01 3.44035029e-03 1.79986358e-01 1.20554864e-01
 1.11935169e-01 5.58521092e-01 2.89365143e-01 4.23284888e-01
 1.52129680e-01 9.10858035e-01 6.04063272e-04 6.53511286e-01
 9.84988391e-01 2.35941112e-02 4.97084111e-01 7.96042681e-01
 9.89199400e-01 9.97604370e-01 9.81489420e-01 7.60314703e-01
 9.80292976e-01 6.07505202e-01 2.44737953e-01 9.97860253e-01
 5.91046154e-01 3.07725698e-01 9.68102574e-01 6.69224799e-01
 9.56089258e-01 1.26197666e-01 9.50426579e-01 9.08704519e-01
 3.58844101e-02 5.76201200e-01 9.99499679e-01 2.11085647e-01
 3.83088022e-01 1.84501171e-01 3.31052542e-01 4.14633095e-01
 3.88676226e-01 5.55680037e-01 8.29440355e-03 1.67074502e-01
 9.52182293e-01 9.16487038e-01 8.16335320e-01 9.84979391e-01
 3.52597892e-01 9.31412220e-01 5.41907847e-02 9.11847830e-01
 9.51591492e-01 3.03998590e-03 3.07698905e-01 4.16138172e-02
 9.47696328e-01 8.25697124e-01 2.53288507e-01 2.67599702e-01
 4.52715427e-01 1.83913410e-02 1.53330863e-02 1.37861073e-02
 1.76715404e-01 4.89152670e-01 5.25966287e-03 1.09902143e-01
 9.43062782e-01 5.31315088e-01 2.93503910e-01 8.88438642e-01
 6.59865975e-01 9.55338240e-01 1.05649531e-02 8.84472609e-01
 2.16199458e-02 1.97966605e-01 8.14517319e-01 2.34630018e-01
 6.31478846e-01 9.44050074e-01 3.62204075e-01 3.17894429e-01
 9.61760938e-01 8.21210444e-01 3.60461116e-01 9.57126379e-01
 1.43263906e-01 2.43284643e-01 9.88902569e-01 4.52252507e-01
 9.75193799e-01 8.45325291e-02 6.32126331e-02 7.90872872e-02
 9.81869817e-01 8.56022835e-02 6.56797290e-02 3.58084738e-02
 9.85773683e-01 3.40821058e-01 2.39343256e-01 4.94971871e-03
 9.57288325e-01 2.91166604e-02 1.28904283e-02 9.30630088e-01]  que se encuentra en:  33
Imagen de BD:  163  Imagen de consulta:  11 

Distancia:  3.6141796  al vector:  [8.4503806e-01 4.7357380e-03 6.8109035e-01 1.3294816e-04 1.9828820e-01
 5.4589109e-06 7.4654520e-03 9.9999297e-01 9.9973369e-01 1.6725659e-03
 6.4394921e-02 6.1991960e-02 9.9751925e-01 2.0949990e-02 4.4144928e-01
 4.6804547e-04 9.9986476e-01 1.1285439e-01 1.0046220e-05 8.4258169e-02
 6.4513326e-02 9.9884033e-01 9.9815011e-01 9.9896479e-01 9.9800313e-01
 3.6400765e-01 9.8801637e-01 2.7448119e-05 1.3556689e-02 9.9989617e-01
 9.9911994e-01 9.2181122e-01 9.9587035e-01 7.3064655e-01 9.3727970e-01
 9.9737662e-01 9.9896812e-01 4.2053938e-02 8.8688982e-01 9.1094244e-01
 2.2244722e-02 9.3092656e-01 8.7684989e-01 1.9973516e-03 9.9975216e-01
 2.4797993e-07 2.0617384e-01 1.1163950e-03 7.8304976e-01 3.3435862e-07
 3.3254704e-01 9.9445415e-01 9.8914897e-01 1.4618397e-02 3.8961470e-03
 9.2969286e-01 7.1590501e-01 9.6453834e-01 4.2551875e-02 9.9934572e-01
 8.1055534e-01 7.0375025e-01 9.9997389e-01 9.7006816e-01 2.4386644e-03
 8.8767785e-01 3.8212359e-02 9.4203985e-01 7.1131587e-03 3.7554902e-01
 7.6632445e-05 3.4953418e-01 1.9625279e-01 5.6934655e-01 2.5400460e-02
 9.5792538e-01 1.5693188e-02 4.0342668e-01 2.3769027e-01 9.8693001e-01
 9.8377109e-01 9.9993539e-01 9.9999392e-01 8.9989197e-01 9.4470584e-01
 2.5037807e-01 9.9871159e-01 2.7003527e-02 9.1951382e-01 4.1301018e-01
 1.0347068e-02 8.9210276e-05 9.9873435e-01 1.5060306e-03 7.3420584e-02
 9.9995935e-01 9.5926833e-01 8.2847226e-01 9.7208917e-03 8.0428994e-01
 9.7966647e-01 1.0644495e-03 9.9883878e-01 5.4450929e-03 1.9339237e-01
 9.9891710e-01 9.9965870e-01 9.9998271e-01 3.1904608e-02 8.7272871e-01
 1.2227928e-04 1.9752890e-02 9.9894530e-01 8.8286161e-01 2.8822702e-01
 1.9387158e-05 8.8763237e-04 8.4356964e-03 4.4195056e-01 1.9951420e-05
 2.9180646e-02 3.0406857e-01 8.9813989e-01 9.9984324e-01 7.9562002e-01
 3.2675467e-05 8.1094600e-05 1.0630190e-03]  en la posicion:  118  desde el vector:  [6.8342358e-01 5.2109190e-05 9.9983990e-01 4.7870576e-03 2.0181257e-01
 7.1656839e-05 4.1705370e-04 9.9996215e-01 9.4001120e-01 3.5549131e-01
 1.4792725e-01 2.8657365e-01 9.7786319e-01 7.6432264e-01 3.8642168e-02
 4.8399180e-02 9.9423647e-01 3.4588575e-04 2.7674437e-04 4.5461476e-02
 7.9099029e-02 9.9655271e-01 7.5320923e-01 9.9989581e-01 9.3364131e-01
 6.0891473e-01 9.9268371e-01 2.2961318e-02 2.6668373e-01 9.9965626e-01
 9.4639641e-01 9.9928534e-01 3.0556101e-01 3.7974548e-01 5.5343080e-01
 9.6763980e-01 9.9947596e-01 3.7521511e-02 9.9997485e-01 9.3506706e-01
 3.0623674e-03 7.7959329e-02 6.3938612e-01 6.2867999e-04 9.9777675e-01
 7.8638822e-02 7.2611606e-01 8.1620812e-03 1.0923579e-01 6.5660477e-04
 9.7855985e-01 9.3504107e-01 9.5139605e-01 8.6709189e-01 2.5231183e-02
 9.9917156e-01 9.3692303e-01 6.4632571e-01 1.8717355e-01 9.9490786e-01
 9.0128356e-01 9.9652290e-01 9.8570603e-01 9.1571009e-01 1.8951297e-04
 9.4366664e-01 1.2548435e-01 5.8479273e-01 1.5527010e-04 2.0186692e-02
 1.3632774e-03 2.9661804e-02 9.8660016e-01 3.8778940e-01 6.6950619e-03
 9.9546337e-01 1.3476312e-03 3.3955729e-01 9.1484088e-01 9.7161663e-01
 6.4075798e-01 8.8348854e-01 9.2379892e-01 4.5398921e-01 1.8885911e-02
 8.7054813e-01 1.2582755e-01 1.4817715e-04 9.8965627e-01 1.1752972e-01
 3.4860794e-05 8.6337459e-01 9.9778199e-01 4.8502722e-01 7.7412218e-02
 8.9568985e-01 7.3128039e-01 9.9998796e-01 5.1405042e-02 8.9931858e-01
 9.9819195e-01 1.0373461e-01 9.8033059e-01 8.4047890e-05 2.4301261e-02
 9.5705903e-01 5.5273080e-01 8.1944883e-01 1.7459935e-05 9.8045015e-01
 2.1433830e-04 1.1948168e-01 7.9037452e-01 2.5801808e-02 5.4296905e-01
 2.8637052e-04 3.1655139e-01 6.1903656e-01 3.2268795e-01 5.8862591e-01
 1.6782308e-01 9.2233360e-01 2.0515898e-01 5.3737533e-01 9.9998868e-01
 6.7636460e-02 8.3342195e-04 4.0222496e-02]  que se encuentra en:  34
Imagen de BD:  22  Imagen de consulta:  83 

Distancia:  4.200229  al vector:  [9.99999642e-01 1.01094544e-02 9.94495988e-01 1.19863861e-08
 2.23857462e-02 9.99946117e-01 4.22838330e-03 9.96616542e-01
 4.10945386e-01 1.07703301e-04 3.80822718e-02 3.34029317e-01
 8.85500312e-02 7.22527623e-01 1.54394150e-01 8.32378507e-01
 9.98713970e-01 8.56518745e-04 7.56450653e-01 9.59803760e-01
 6.34860669e-07 9.99999046e-01 6.06895864e-01 8.98534000e-01
 1.00000000e+00 9.29737270e-01 9.53370452e-01 8.98373961e-01
 8.72107387e-01 3.02016735e-04 9.93461311e-01 9.81120050e-01
 1.25454366e-02 4.26601380e-01 3.67122412e-01 2.33908415e-01
 9.99999642e-01 3.67134771e-06 9.92204070e-01 3.04065347e-02
 2.53961176e-01 8.79866719e-01 9.86621797e-01 1.14978491e-06
 9.99322057e-01 2.47225642e-01 2.75294483e-02 7.39622116e-02
 1.41475707e-01 9.99176741e-01 9.98797059e-01 9.98108208e-01
 9.99999881e-01 9.06965494e-01 4.38088791e-05 2.68506706e-01
 1.42119527e-02 9.81829524e-01 3.08262408e-02 9.99897361e-01
 9.98080611e-01 3.74169111e-01 1.08964505e-05 8.89548659e-02
 2.23267257e-01 9.99662638e-01 9.82645035e-01 9.17854548e-01
 2.72971392e-03 1.44596398e-02 9.99993801e-01 5.34311730e-05
 1.28328800e-04 9.98786628e-01 1.28485590e-01 9.43619728e-01
 9.99877095e-01 9.31532383e-01 9.71154451e-01 9.76670861e-01
 9.95511770e-01 9.99998212e-01 2.77183950e-02 9.99938786e-01
 1.28599912e-01 5.35955071e-01 9.95201588e-01 7.05347239e-11
 9.98802662e-01 5.99005818e-03 4.23256381e-08 1.07398928e-05
 1.00000000e+00 7.89379555e-06 9.47114706e-01 9.98058438e-01
 9.99987006e-01 9.99525905e-01 9.41239591e-07 1.14705785e-04
 9.69138145e-01 8.55491817e-01 4.61027920e-01 9.97360468e-01
 1.25339627e-03 3.33402991e-01 2.77806103e-01 9.98463035e-01
 6.46791458e-02 9.45216179e-01 2.68101305e-01 8.90583396e-01
 8.30176214e-05 1.01473629e-02 1.42509282e-01 3.89602602e-01
 8.40896917e-08 9.21428204e-04 8.51096153e-01 1.18659139e-02
 2.73271635e-05 9.99030232e-01 6.59121573e-02 1.43823028e-03
 9.86175299e-01 2.85974145e-03 9.90478396e-02 2.54458189e-03]  en la posicion:  148  desde el vector:  [9.65465188e-01 1.66833401e-04 9.95345712e-01 2.25821137e-03
 2.04384327e-04 7.97219276e-01 2.77400017e-03 7.09747791e-01
 1.76923841e-01 1.62716478e-01 3.90239567e-01 8.94137740e-01
 9.83305156e-01 1.97098464e-01 6.58113480e-01 3.07199955e-02
 9.44904447e-01 1.88380291e-05 2.37222672e-01 4.39073294e-01
 3.72618437e-04 9.99967635e-01 6.35020137e-01 9.88439977e-01
 9.89277542e-01 9.19363856e-01 9.90467072e-01 2.43240088e-01
 6.07630312e-02 1.02816135e-01 2.77689815e-01 9.64019895e-01
 3.84461880e-03 9.95716274e-01 3.97676945e-01 9.93294895e-01
 9.58034873e-01 7.53459334e-03 9.99973893e-01 1.45283341e-01
 2.84848511e-02 9.67295587e-01 7.47438133e-01 1.21847057e-04
 9.66858506e-01 4.38901484e-02 4.95928526e-02 5.98414719e-01
 2.67761946e-03 9.53875899e-01 9.96077061e-01 9.99764800e-01
 9.96507287e-01 1.97286904e-01 2.48163044e-02 5.54709852e-01
 6.55902028e-02 9.41645265e-01 2.11795390e-01 9.39810395e-01
 7.48667955e-01 8.93660069e-01 4.40598041e-01 9.52422857e-01
 1.65503323e-02 9.71757293e-01 8.80183697e-01 9.68396187e-01
 8.16315413e-04 9.91020441e-01 9.22454178e-01 4.74721193e-04
 2.71522403e-01 8.79952013e-02 9.71064270e-01 9.02246296e-01
 5.22536933e-01 9.56815243e-01 9.91666317e-01 6.57427907e-01
 7.92146325e-01 9.91036832e-01 1.73551947e-01 9.85757828e-01
 9.54779148e-01 9.99243379e-01 7.29927421e-01 7.21621513e-02
 9.89453375e-01 4.09748197e-01 1.54998899e-03 2.39560068e-01
 9.99691486e-01 2.67125428e-01 7.95817256e-01 4.30785984e-01
 9.63055134e-01 9.99971628e-01 1.50924921e-03 1.61921352e-01
 8.20399702e-01 8.00333858e-01 7.32087135e-01 7.48898983e-02
 9.75595832e-01 9.77062345e-01 5.77035785e-01 9.32228327e-01
 6.97409213e-02 9.30486441e-01 4.06470899e-05 9.05952454e-02
 5.15997410e-04 6.57849252e-01 1.41399950e-01 4.18085098e-01
 3.05503607e-03 2.85285711e-03 9.65068817e-01 5.60561776e-01
 1.08608633e-01 6.67299330e-01 3.62240672e-01 4.46628392e-01
 9.94250178e-01 4.59111035e-02 9.14365888e-01 2.69728899e-03]  que se encuentra en:  35
Imagen de BD:  5  Imagen de consulta:  87 

Distancia:  4.454193  al vector:  [9.77151036e-01 4.92429351e-08 8.53393912e-01 4.65033338e-07
 6.66077435e-01 1.41084100e-09 4.18577254e-01 1.00000000e+00
 9.99976516e-01 1.12905684e-04 2.61740752e-05 7.00940549e-01
 9.99926805e-01 2.53681898e-01 7.23802388e-01 4.26407837e-07
 9.99997139e-01 7.24411620e-07 9.58658077e-08 3.33791971e-03
 5.67945281e-05 9.99992728e-01 9.99780774e-01 1.00000000e+00
 9.99956727e-01 9.59213555e-01 1.00000000e+00 4.73826731e-06
 5.59018850e-02 1.00000000e+00 9.98625696e-01 8.87242913e-01
 9.99748468e-01 9.99989569e-01 7.74223804e-02 9.99990702e-01
 1.00000000e+00 4.28253588e-05 1.00000000e+00 9.13317204e-02
 2.37679751e-06 2.99393773e-01 3.74663770e-02 2.71157916e-07
 9.99998331e-01 4.25968501e-05 3.75065804e-01 2.05076117e-06
 1.49577856e-04 2.49444927e-08 5.90872765e-03 7.66205668e-01
 9.99965608e-01 7.89703608e-01 3.68326902e-04 1.00000000e+00
 9.95602369e-01 1.42025650e-01 2.37542450e-01 1.00000000e+00
 3.74387741e-01 5.79993129e-01 9.94801700e-01 2.56436259e-01
 3.73031398e-05 6.50210261e-01 7.21253395e-01 6.50579810e-01
 3.02011387e-07 1.21375531e-01 2.64311795e-09 3.32857668e-01
 1.94387436e-02 2.16835946e-01 3.73554260e-01 7.19431758e-01
 4.57000732e-03 5.48109412e-02 7.99497604e-01 2.65786409e-01
 8.09805691e-02 9.99991059e-01 9.99089479e-01 9.99579549e-01
 9.14540172e-01 1.24248207e-01 7.03519106e-01 1.31786644e-08
 9.99998808e-01 2.76335368e-06 7.29312788e-08 1.40330106e-01
 8.42453659e-01 3.63224745e-03 3.65996629e-01 9.99999523e-01
 9.99795198e-01 1.00000000e+00 1.09096094e-04 1.80503368e-01
 3.61728191e-01 1.04387925e-06 9.99998987e-01 1.02299822e-07
 9.03705716e-01 9.99994397e-01 9.95939732e-01 9.99988556e-01
 1.93857491e-01 1.96446925e-01 1.12177179e-09 8.36692333e-01
 9.08338904e-01 7.07360208e-01 3.93103927e-01 1.20412039e-07
 1.25195229e-05 1.85259610e-01 8.92083645e-01 2.78169700e-05
 1.62721062e-05 3.21923494e-02 6.12901807e-01 9.71439600e-01
 1.00000000e+00 7.95529331e-07 5.38138822e-09 2.67922878e-04]  en la posicion:  117  desde el vector:  [9.87956285e-01 3.66715640e-01 8.26847911e-01 8.04740191e-03
 4.24981117e-04 1.44991279e-03 8.31001639e-01 9.99992132e-01
 3.15018654e-01 9.93001044e-01 5.38128661e-05 9.85401869e-03
 9.99937773e-01 2.57843137e-02 9.98976529e-01 2.46787351e-07
 9.99853015e-01 7.75407553e-01 2.52982974e-03 7.87540674e-01
 3.92630696e-03 7.02407837e-01 3.12234938e-01 9.99839127e-01
 9.62853909e-01 7.31287718e-01 9.99869466e-01 4.00062709e-05
 2.19871163e-01 9.99245346e-01 4.64662015e-02 2.07168758e-02
 9.99870598e-01 9.47131038e-01 2.26309896e-03 9.99997675e-01
 9.91737485e-01 6.78698182e-01 9.98705626e-01 1.30438805e-03
 4.30895686e-02 1.15017712e-01 4.27823961e-02 4.03120100e-01
 8.44303489e-01 1.47254765e-02 8.96002650e-01 6.44001603e-01
 2.21642554e-02 1.14747299e-05 3.07955742e-02 5.28829873e-01
 6.60529613e-01 2.44909793e-01 8.54686677e-01 9.99606192e-01
 9.99831200e-01 9.85091925e-03 8.08307528e-03 9.24346209e-01
 2.82328367e-01 9.84594226e-03 2.38887757e-01 9.09806609e-01
 9.74414408e-01 9.82404649e-01 9.96939063e-01 1.26323879e-01
 3.64777446e-03 3.86502087e-01 1.60396099e-04 9.77791011e-01
 1.09189749e-03 7.73671031e-01 3.06071460e-01 4.97571945e-01
 1.10359699e-01 6.45182729e-02 8.14196467e-02 1.67928904e-01
 1.56782001e-01 1.29850805e-02 4.83278811e-01 6.02074325e-01
 4.40487832e-01 9.12208915e-01 4.21812236e-02 4.19760644e-02
 9.99984026e-01 8.81710649e-03 7.67268419e-01 1.49692744e-01
 5.65466762e-01 1.12745166e-03 8.02734494e-01 9.99935389e-01
 6.72151446e-01 4.98706311e-01 7.20105469e-02 8.11767578e-01
 3.81765276e-01 5.49458563e-02 9.91319299e-01 4.84662056e-02
 7.30258286e-01 9.99987960e-01 7.78849125e-02 9.99619663e-01
 4.97248054e-01 2.12897897e-01 2.33635604e-01 5.71259260e-01
 7.96406746e-01 4.90238428e-01 2.44807005e-02 2.19526887e-03
 8.06191266e-02 2.23975241e-01 6.98668420e-01 1.07007623e-02
 9.46113706e-01 4.61219728e-01 6.31073773e-01 9.87341285e-01
 9.98141825e-01 6.03657427e-05 6.52819872e-04 2.82854140e-02]  que se encuentra en:  36
Imagen de BD:  21  Imagen de consulta:  93 

Distancia:  4.418997  al vector:  [8.95960510e-01 1.00000000e+00 8.10444772e-01 9.16874647e-01
 9.75253344e-01 9.56821918e-01 2.08393037e-02 1.53611064e-01
 9.99999046e-01 6.48122966e-01 4.92304437e-08 9.76454973e-01
 5.82179034e-08 6.81825697e-01 9.30335164e-01 2.66851783e-02
 9.78955269e-01 1.00000000e+00 4.20975029e-01 7.63262331e-01
 9.31331396e-01 1.07973094e-07 1.00000000e+00 4.43092316e-01
 6.81138873e-01 4.17006016e-03 1.15204871e-01 1.49396062e-02
 7.18856633e-01 2.08436996e-01 8.80421996e-01 6.58511996e-01
 9.99997854e-01 7.92467594e-03 6.86929584e-01 9.61237192e-01
 7.71969557e-04 8.04801822e-01 5.07206569e-05 9.82151389e-01
 9.99999762e-01 8.88400197e-01 6.13552690e-01 9.99997139e-01
 6.23405660e-10 4.87275034e-01 2.80437291e-01 3.96919262e-08
 9.69939291e-01 9.99998629e-01 9.68998373e-02 1.04674220e-01
 5.41240952e-05 8.73451114e-01 9.95923042e-01 9.93196845e-01
 1.83685677e-06 1.81191355e-01 7.73789048e-01 3.46766615e-09
 7.85513997e-01 1.43275887e-01 2.82579362e-01 1.74590945e-02
 9.99820232e-01 8.88248920e-01 9.66414809e-03 2.52681971e-01
 1.00000000e+00 4.94569540e-03 9.90844607e-01 9.32080269e-01
 9.50281262e-01 8.35563242e-01 4.48102027e-01 6.30972505e-01
 9.99996662e-01 9.92076516e-01 4.95842576e-01 6.61705792e-01
 8.05718303e-02 6.30825758e-03 5.26352406e-01 8.22338462e-03
 5.26749074e-01 1.64359808e-04 8.17781448e-01 8.31316411e-02
 9.96332049e-01 1.03031998e-05 9.30003166e-01 9.79344249e-01
 9.82678115e-01 9.82942224e-01 3.75366509e-02 5.19315243e-01
 4.30489808e-01 3.70554716e-07 9.36457276e-01 1.23539567e-03
 1.60182387e-01 2.08459298e-07 1.00000000e+00 9.62831438e-01
 2.26587057e-03 9.85260308e-01 2.31364369e-03 8.75375152e-01
 8.05858374e-01 2.86854506e-01 1.00000000e+00 7.64581919e-01
 2.44786471e-01 9.82682407e-01 3.65043044e-01 6.46702170e-01
 1.00000000e+00 8.35883677e-01 1.36017799e-04 2.19382855e-06
 9.83530641e-01 9.67980385e-01 3.40034783e-01 2.11723208e-01
 3.66826952e-02 3.86989534e-01 1.52270794e-02 6.18764043e-01]  en la posicion:  108  desde el vector:  [9.8889947e-01 9.9999577e-01 2.7276352e-01 9.9871176e-01 4.5335871e-01
 9.9931312e-01 6.7450702e-01 6.4240479e-01 4.4774005e-01 3.5109872e-01
 8.6434186e-03 9.5712602e-01 2.2560232e-07 9.2803955e-01 3.1969947e-01
 9.9757230e-01 6.4509964e-01 9.9992359e-01 9.2693663e-01 9.1982108e-01
 4.2993933e-01 6.1748904e-07 9.4528639e-01 9.4032884e-03 9.5043850e-01
 8.7846100e-01 2.0041168e-03 9.9634469e-01 4.8098078e-01 3.3294857e-03
 5.0711036e-03 8.6269915e-02 6.1505759e-01 2.9270947e-03 4.6308100e-01
 2.2958368e-02 1.2782568e-01 9.1160822e-01 3.0841362e-05 9.9454999e-01
 9.9927843e-01 3.1675071e-01 8.5927141e-01 9.9976194e-01 3.8356185e-03
 9.9908394e-01 1.4925522e-01 1.2176484e-01 9.3035781e-01 9.9769330e-01
 6.6255367e-01 7.6297224e-03 6.2023073e-02 8.6916465e-01 7.6963282e-01
 6.7311591e-01 2.1510094e-02 1.1657864e-01 9.9901748e-01 3.7830025e-02
 3.5017693e-01 1.2724876e-02 2.4028835e-01 2.7620167e-02 9.5086372e-01
 9.8677075e-01 3.1377673e-03 9.8758638e-03 9.9999940e-01 9.1333866e-02
 9.9982774e-01 6.0347879e-01 9.6099007e-01 9.5316982e-01 1.1543357e-04
 9.4283527e-01 9.9995726e-01 5.6212479e-01 7.4369258e-01 3.1322604e-01
 5.9506226e-01 1.4752746e-02 3.6922872e-01 4.9130350e-01 4.4410080e-02
 1.0592580e-02 8.2302940e-01 2.5108668e-01 5.4626465e-03 2.0879507e-04
 9.9270666e-01 1.5727663e-01 7.3272574e-01 1.6565681e-02 1.5827173e-01
 4.3829173e-02 2.3139387e-01 5.4425001e-04 2.6825297e-01 8.4114701e-01
 4.2391312e-01 1.0282382e-01 9.5275992e-01 9.9984407e-01 6.1885417e-03
 2.3227632e-03 1.7232442e-01 1.7529726e-01 2.6181340e-01 9.6609831e-01
 1.0000000e+00 4.2569613e-01 6.1556995e-02 9.7924769e-01 9.8277307e-01
 5.0145125e-01 9.9766397e-01 9.7583473e-01 1.8356264e-02 2.0628870e-03
 4.8716456e-01 6.0246551e-01 3.6234391e-01 4.3068945e-02 2.9826164e-04
 6.7234975e-01 9.5963609e-01 9.5172423e-01]  que se encuentra en:  37
Imagen de BD:  197  Imagen de consulta:  95 

Distancia:  4.671972  al vector:  [6.90981746e-03 9.99912143e-01 6.15294099e-01 2.27721575e-05
 8.60433102e-01 4.58449125e-04 5.24178803e-01 9.99850869e-01
 9.99410272e-01 9.99490678e-01 1.11793743e-04 9.35773671e-01
 3.73909235e-01 6.73082769e-01 3.19968849e-01 5.84584814e-06
 9.99958038e-01 9.99869227e-01 1.37141347e-03 5.97965062e-01
 9.98192847e-01 1.04728341e-03 9.98251081e-01 9.99987841e-01
 5.05685806e-04 1.99107826e-02 9.99749541e-01 1.96120964e-05
 5.00103831e-03 9.99492526e-01 4.03781851e-05 5.92091084e-01
 9.99943972e-01 2.54154205e-04 5.55589795e-03 9.99967694e-01
 9.99476492e-01 9.97748494e-01 9.99659181e-01 6.87971354e-01
 9.99913454e-01 1.59388185e-02 7.13039219e-01 9.99109864e-01
 8.17827167e-05 1.40238744e-05 9.51017618e-01 8.36908817e-04
 9.92389083e-01 2.27330625e-01 6.63856864e-02 7.68152297e-01
 9.67461467e-01 6.11234307e-01 9.97918904e-01 9.99675453e-01
 4.23252583e-04 4.45849061e-01 3.55184078e-01 6.80357218e-04
 9.31349635e-01 5.77320814e-01 2.28258967e-03 2.43670344e-01
 9.99997139e-01 7.31699288e-01 8.70521903e-01 9.54516649e-01
 9.99918818e-01 5.69105744e-01 5.95808029e-04 5.11592388e-01
 9.95947659e-01 4.01188821e-01 1.08120978e-01 8.90768290e-01
 9.99916196e-01 5.72538137e-01 3.99231613e-02 8.14288497e-01
 4.82408404e-02 2.63164639e-02 6.24141097e-03 6.39319420e-04
 9.94575977e-01 6.30476654e-01 1.63471699e-03 3.62131000e-03
 9.99998510e-01 1.96477771e-03 1.19666278e-01 9.93513227e-01
 9.89392877e-01 9.99256790e-01 2.32129991e-02 9.99990523e-01
 1.64317191e-02 1.81463361e-03 9.97745872e-01 1.05180174e-01
 6.86608791e-01 3.19558382e-03 9.98240471e-01 5.82724512e-02
 9.77339029e-01 9.99991536e-01 3.13474848e-05 9.99990404e-01
 8.03195536e-02 5.38063645e-02 9.99419451e-01 9.82536316e-01
 9.58685040e-01 1.22554004e-02 6.24433160e-03 2.33435631e-03
 2.55222559e-01 6.84420168e-01 9.50838447e-01 1.79201365e-04
 9.99471843e-01 4.03766811e-01 4.80207801e-03 6.47467375e-03
 9.97870207e-01 8.16742177e-05 1.81971192e-02 9.90347385e-01]  en la posicion:  71  desde el vector:  [9.44819868e-01 8.45864773e-01 4.02586758e-02 5.12719154e-04
 4.87266481e-02 2.93354988e-02 8.04799676e-01 9.99350786e-01
 9.12512422e-01 5.25221825e-01 1.66508555e-03 5.19878268e-02
 1.43381149e-01 4.26416039e-01 1.09629601e-01 4.37468290e-04
 9.99036252e-01 6.13064110e-01 1.97169185e-03 3.31907511e-01
 2.09750324e-01 1.49796695e-01 7.41167128e-01 9.99400020e-01
 7.79826164e-01 1.11144990e-01 9.93568480e-01 3.36796045e-03
 4.56732094e-01 9.97707367e-01 2.29156017e-03 9.26961899e-01
 9.99723494e-01 4.81201410e-02 3.62716824e-01 9.97487247e-01
 9.83087242e-01 7.43723273e-01 8.46144557e-01 7.20441043e-01
 9.88285422e-01 1.74819589e-01 4.44387376e-01 9.63964939e-01
 8.26821327e-01 2.35587358e-04 9.33766127e-01 7.43831694e-02
 8.38390231e-01 7.72743523e-02 8.23089361e-01 9.75244939e-01
 8.10352445e-01 1.71308756e-01 2.15759844e-01 9.94379282e-01
 5.59142947e-01 2.58490443e-03 9.46062624e-01 7.85905898e-01
 4.47809696e-04 1.20328158e-01 1.02551520e-01 8.12034845e-01
 7.61924505e-01 9.51179266e-01 9.00306702e-01 8.77591968e-01
 8.92795682e-01 9.87884998e-01 6.89089298e-04 4.41624999e-01
 9.91499424e-03 3.30795944e-02 9.54916835e-01 7.79137373e-01
 9.97425199e-01 8.94650519e-02 1.01214051e-01 4.34702337e-02
 1.68965966e-01 3.22394669e-01 2.97207892e-01 8.83059204e-02
 9.73059893e-01 9.02557254e-01 1.25785410e-01 3.16592455e-02
 9.99985814e-01 1.16456747e-02 8.23925853e-01 4.99778479e-01
 7.36957550e-01 4.57288802e-01 8.39510441e-01 9.98444438e-01
 1.84929878e-01 7.77611136e-03 7.94974864e-02 9.97658014e-01
 1.53916866e-01 2.76665688e-02 6.76246107e-01 1.39344513e-01
 1.36198670e-01 9.99745965e-01 1.32268041e-01 9.99742389e-01
 7.15224266e-01 4.23980296e-01 9.79950190e-01 4.34439480e-02
 9.84254718e-01 6.99641526e-01 6.45456553e-01 1.75062716e-02
 4.96774733e-01 6.09486580e-01 3.60885680e-01 1.55319273e-02
 8.54408622e-01 1.60406202e-01 9.02717829e-01 1.57170594e-02
 9.91615713e-01 6.11752272e-04 1.20202005e-02 8.26077700e-01]  que se encuentra en:  38
Imagen de BD:  163  Imagen de consulta:  100 

Distancia:  3.234592  al vector:  [1.50095403e-01 1.77028687e-05 5.72783709e-01 9.02265310e-03
 8.46096873e-03 4.19765711e-04 3.20607424e-03 9.93189394e-01
 8.67711544e-01 9.91836190e-01 9.90692556e-01 9.26855206e-03
 9.98416185e-01 9.81615245e-01 1.00707114e-02 7.81078577e-01
 9.06917810e-01 5.09250164e-03 1.88518465e-02 6.65662289e-02
 9.32325542e-01 9.98883188e-01 8.15953314e-02 9.99720395e-01
 1.88737810e-02 8.53157282e-01 9.92660820e-01 7.77439952e-01
 1.87461078e-02 9.99588490e-01 4.01685834e-02 8.41206372e-01
 3.65144014e-03 5.86299062e-01 9.88106847e-01 1.49464220e-01
 9.76134896e-01 6.06387496e-01 9.99975562e-01 7.43804872e-02
 4.72427547e-01 7.93148518e-01 8.00974369e-01 1.56042278e-02
 9.75119352e-01 5.49616814e-02 3.36291492e-02 4.43535268e-01
 8.46402109e-01 4.55176830e-02 8.96897376e-01 9.77834821e-01
 9.79310155e-01 7.78050303e-01 6.70760870e-03 6.58971012e-01
 2.98608840e-02 3.73173267e-01 8.96124363e-01 9.67161477e-01
 5.71247578e-01 9.34824705e-01 9.09010172e-01 9.96802211e-01
 4.99635935e-04 9.01250124e-01 8.47894609e-01 8.86006594e-01
 1.18836024e-05 9.99883294e-01 1.24067068e-04 5.55738091e-01
 6.52114630e-01 1.61005884e-01 2.57839918e-01 6.29769683e-01
 3.79616618e-02 9.84475851e-01 4.16402698e-01 6.35460258e-01
 8.93992782e-01 9.42662120e-01 4.50419784e-01 9.74116087e-01
 9.53221023e-01 9.97205198e-01 9.17121768e-03 9.22758579e-01
 2.94209659e-01 9.74585712e-02 1.33925676e-03 4.10428345e-01
 8.91201973e-01 9.94212031e-01 2.00115174e-01 6.20693445e-01
 1.65166587e-01 9.89048243e-01 8.89881015e-01 8.61858487e-01
 7.10377395e-01 9.16508675e-01 7.31662214e-02 1.17582276e-04
 4.97556001e-01 7.55881071e-02 3.58191431e-01 9.25843060e-01
 7.18408823e-03 4.75818932e-01 2.46793032e-04 5.47365189e-01
 5.97391486e-01 1.92615181e-01 6.68424487e-01 2.44050920e-02
 2.00202167e-02 6.28621221e-01 1.96149617e-01 9.43073571e-01
 7.29663193e-01 1.33773386e-02 9.30081487e-01 7.84081936e-01
 9.72901940e-01 2.66169697e-01 9.75708127e-01 9.43661690e-01]  en la posicion:  111  desde el vector:  [7.45457411e-03 2.55979576e-05 9.38121319e-01 3.14825773e-03
 4.52591479e-02 1.46132888e-05 1.45854354e-02 9.96566772e-01
 3.10327113e-01 9.99836564e-01 9.99702811e-01 4.15823847e-01
 9.81745601e-01 1.33854806e-01 7.89004266e-02 9.99625683e-01
 1.75779313e-01 1.53979361e-02 1.52380168e-02 2.53784180e-01
 9.99983609e-01 9.29509044e-01 5.47936559e-03 9.93604660e-01
 1.10033154e-03 7.41181076e-02 6.25142992e-01 9.97242212e-01
 5.54703176e-02 9.99933004e-01 1.84460282e-02 9.39739466e-01
 2.06619501e-04 2.34395266e-04 3.64959240e-02 5.28275967e-04
 9.97156680e-01 8.84462416e-01 9.99818563e-01 3.53469253e-02
 9.72671509e-02 4.33504015e-01 7.47142375e-01 5.57363927e-01
 9.96866345e-01 3.97735655e-01 8.02018523e-01 9.49728727e-01
 9.98890638e-01 6.23065233e-03 9.22621429e-01 9.96509135e-01
 8.49628747e-01 6.34175718e-01 1.72171891e-02 4.74282801e-02
 2.43191749e-01 2.09106803e-01 9.09683466e-01 9.98451471e-01
 3.41093779e-01 9.75307941e-01 5.93997121e-01 8.81285071e-01
 2.71520019e-03 8.43315721e-01 9.42845583e-01 9.91160691e-01
 7.74630389e-05 9.94888783e-01 7.10935055e-05 4.55110490e-01
 9.97829556e-01 4.02835608e-01 3.15516293e-02 2.01332688e-01
 1.86352193e-01 1.11234814e-01 5.14948368e-02 9.66563582e-01
 8.69374275e-01 9.41915035e-01 1.48890495e-01 9.89372253e-01
 9.66045022e-01 9.93321896e-01 1.70916319e-04 5.84399343e-01
 6.95988536e-03 2.21432716e-01 4.88040569e-05 5.18630445e-02
 9.95521963e-01 9.99837279e-01 8.37254286e-01 7.49737442e-01
 8.53175819e-02 7.37023950e-01 9.99839425e-01 6.52170658e-01
 9.55474734e-01 9.94461536e-01 4.99290228e-03 7.94082880e-04
 6.17956519e-02 1.74868107e-03 4.90480244e-01 8.60506594e-01
 2.37320065e-02 1.66510642e-01 1.02850795e-03 2.83529580e-01
 1.87743634e-01 1.37791216e-01 7.46653140e-01 5.78194857e-03
 2.13918567e-01 8.72311711e-01 1.11465454e-02 8.80059242e-01
 8.52003932e-01 5.42045236e-02 9.87906754e-01 9.06870961e-02
 9.86024618e-01 8.60135674e-01 8.21541667e-01 9.99998927e-01]  que se encuentra en:  39
Imagen de BD:  2  Imagen de consulta:  103 

Distancia:  3.4741957  al vector:  [2.5229859e-01 4.7019839e-02 4.4662002e-01 9.9986279e-01 9.0882927e-02
 8.7487698e-04 9.0495795e-02 2.0149350e-04 9.9989867e-01 1.3398284e-01
 8.1737858e-01 5.0286114e-02 4.6923801e-01 9.3227983e-01 2.0257428e-01
 9.9950141e-01 8.6253881e-04 3.1959054e-01 5.6518477e-01 2.3323178e-02
 9.2463577e-01 7.8690970e-01 9.9999219e-01 7.4776137e-01 5.2305496e-01
 4.4194511e-01 2.5264013e-01 9.2176944e-02 4.2369497e-01 9.9374580e-01
 9.9999273e-01 1.7825606e-01 9.9257278e-01 9.5501596e-01 9.7895831e-01
 3.1480193e-04 3.6987662e-04 1.0985404e-02 4.3642521e-03 9.8428035e-01
 8.0686975e-01 5.0622630e-01 6.1065787e-01 9.8136902e-02 8.0612302e-04
 7.0542079e-01 2.8730270e-01 2.0660848e-06 2.4582714e-02 2.9993862e-01
 3.1499863e-02 2.9331201e-01 1.2065202e-02 3.4784353e-01 9.6900880e-02
 5.9770972e-01 1.1719763e-03 6.9193989e-02 9.9443769e-01 8.5055828e-04
 8.0517066e-01 9.9594837e-01 9.9999833e-01 1.6264027e-01 1.5130639e-04
 7.7734917e-01 4.4978338e-01 7.2052240e-01 6.1025333e-01 9.8445737e-01
 1.4475882e-03 5.3551978e-01 9.9610358e-01 5.2624106e-02 1.5654296e-02
 4.9414185e-01 5.5654943e-03 3.7106544e-02 8.1397980e-02 2.7798176e-02
 2.2281706e-01 9.9596512e-01 9.9654138e-01 9.0090555e-01 1.3254043e-01
 3.4519804e-01 8.1996483e-01 9.9880695e-01 7.2130561e-04 1.2983328e-01
 3.4547561e-01 7.9598659e-01 3.5829216e-01 9.4971842e-01 1.0316038e-01
 5.7020783e-04 9.8636097e-01 9.2514241e-01 7.7917236e-01 8.0030489e-01
 9.8827899e-01 9.1274720e-05 9.9996209e-01 4.4703037e-02 5.3289509e-01
 5.4442883e-04 9.9997580e-01 1.4308393e-03 4.2942911e-02 8.9742219e-01
 8.8605285e-04 4.1192445e-01 9.3344426e-01 9.8509938e-01 6.2915504e-02
 8.4097600e-01 9.9042737e-01 7.0187974e-01 6.7352909e-01 3.3124685e-03
 2.4991304e-02 3.2308865e-01 5.6805104e-02 9.9993366e-01 3.1062740e-01
 9.9908513e-01 6.2754697e-01 1.4063263e-01]  en la posicion:  38  desde el vector:  [1.64258480e-03 1.05662942e-01 1.53901577e-01 9.96657670e-01
 2.22573012e-01 3.01140845e-02 9.11121964e-02 3.49830880e-05
 9.99994397e-01 1.07551336e-01 9.69224930e-01 2.48456299e-02
 4.00155485e-02 8.28362107e-01 6.49212837e-01 9.97187257e-01
 3.62128019e-04 1.08027756e-02 9.63202119e-01 1.14917159e-02
 9.35599387e-01 9.68336046e-01 1.00000000e+00 9.88241315e-01
 7.42849827e-01 1.03988081e-01 9.94849443e-01 7.92748094e-01
 1.74061358e-01 9.95770812e-01 9.99997616e-01 7.10796952e-01
 8.52579772e-01 8.78099382e-01 9.34902549e-01 8.61823559e-04
 7.12770657e-07 8.17686319e-04 5.57025373e-02 7.63363242e-02
 9.26145196e-01 5.83030522e-01 9.47285295e-01 6.62180391e-05
 2.09902123e-06 6.63240194e-01 3.35896850e-01 9.82504034e-09
 5.03915548e-03 9.64854956e-01 3.30088139e-02 8.96836877e-01
 3.21835279e-04 5.30480564e-01 4.17479873e-03 9.99377370e-01
 2.44354560e-05 3.07673693e-01 9.83294606e-01 4.10995426e-06
 3.31137419e-01 9.88960505e-01 9.99997973e-01 9.93820548e-01
 1.94519758e-04 9.82505202e-01 2.28620172e-01 9.74659443e-01
 9.39200401e-01 7.44332492e-01 3.03834677e-04 7.71284103e-01
 9.98654246e-01 1.44404173e-03 9.05187011e-01 6.09288037e-01
 5.51475585e-02 7.20418811e-01 4.06921804e-01 1.03512585e-01
 9.94135499e-01 9.49503303e-01 9.64240372e-01 9.23830569e-01
 9.11742032e-01 2.23245203e-01 4.12574708e-01 9.99988079e-01
 3.79490921e-05 3.60950530e-02 1.62934631e-01 9.91677523e-01
 5.14048219e-01 9.98912930e-01 4.74915892e-01 7.46697187e-04
 9.98368263e-01 9.89739478e-01 9.87824917e-01 1.99995816e-01
 8.62159431e-01 6.66133985e-07 9.99999523e-01 5.24461269e-04
 5.15005887e-02 6.64871931e-03 9.99411702e-01 3.00154090e-03
 8.43239188e-01 9.99123693e-01 9.27597284e-04 2.89405286e-02
 3.75446379e-02 9.67363000e-01 3.92237604e-02 9.51256216e-01
 9.99960959e-01 5.31913102e-01 9.51809227e-01 2.96614468e-02
 1.11651421e-03 1.66177750e-03 7.31492400e-01 9.99982357e-01
 7.68326819e-02 9.92438376e-01 9.65775907e-01 4.53634441e-01]  que se encuentra en:  40
Imagen de BD:  133  Imagen de consulta:  107 

Distancia:  4.16963  al vector:  [9.93507802e-01 2.12091678e-07 7.07409501e-01 3.96196544e-02
 2.48630941e-02 2.83902079e-01 8.78758132e-02 9.78052378e-01
 1.95654899e-01 2.90865600e-02 1.65987164e-01 1.36559606e-01
 9.97896194e-01 8.16691637e-01 4.00504917e-01 9.33703065e-01
 8.18887413e-01 2.11622555e-05 2.36910582e-03 8.14264297e-01
 2.13636558e-05 9.99901891e-01 2.69404948e-02 9.58325326e-01
 9.99244809e-01 2.54644096e-01 8.13358188e-01 2.29964465e-01
 8.10572743e-01 6.02979779e-01 9.78329420e-01 7.64697790e-04
 2.88407505e-02 9.99626398e-01 2.35763013e-01 3.36488962e-01
 9.99920130e-01 1.86836720e-03 9.99979496e-01 9.57060754e-02
 3.67120624e-01 8.22068095e-01 3.69532526e-01 1.50432106e-05
 9.99866128e-01 2.10078359e-01 1.84822083e-02 9.41369414e-01
 3.82500887e-03 4.69362736e-03 9.09493923e-01 8.92464876e-01
 9.99850988e-01 2.46811718e-01 2.76795030e-03 9.04920399e-02
 9.86410856e-01 5.10944605e-01 5.41284800e-01 9.99898195e-01
 9.23828423e-01 3.42119813e-01 7.15188324e-01 8.04335237e-01
 2.45043448e-05 9.90258813e-01 9.95513558e-01 2.33867079e-01
 4.79094183e-07 6.61337018e-01 9.45554614e-01 5.71721435e-01
 1.09941065e-02 2.92865753e-01 8.70741367e-01 5.10317147e-01
 1.30538046e-02 2.06713974e-02 3.11640203e-02 1.64615661e-01
 2.92624533e-02 9.99959230e-01 8.54162931e-01 9.99418378e-01
 9.64576066e-01 9.56387401e-01 8.21906924e-01 1.42991543e-04
 3.66490960e-01 6.68123543e-01 1.72150897e-06 3.15359235e-03
 9.95680153e-01 1.80947781e-03 8.79022062e-01 8.42953682e-01
 9.99368191e-01 9.99997258e-01 3.55511904e-04 2.39478648e-01
 1.05261117e-01 8.39510083e-01 1.15350813e-01 1.43278837e-02
 9.72393453e-01 1.70193076e-01 9.98766720e-01 9.39326167e-01
 1.26547456e-01 8.01371455e-01 7.29212661e-06 4.47488546e-01
 4.29636240e-03 9.98932302e-01 1.98992282e-01 5.29577136e-02
 1.87993050e-04 1.14747137e-01 8.12328577e-01 4.35351074e-01
 3.09893787e-02 9.87825871e-01 1.81621939e-01 9.62090611e-01
 9.89325166e-01 8.31848979e-02 4.57850099e-02 2.42891908e-03]  en la posicion:  146  desde el vector:  [9.80677605e-01 6.81257516e-05 9.97994184e-01 9.03165340e-03
 2.66361237e-03 6.76679611e-03 1.34307146e-03 9.86113012e-01
 1.97409213e-01 9.90781903e-01 5.63051462e-01 9.93950605e-01
 9.99436021e-01 4.34383750e-03 9.86514211e-01 9.06132579e-01
 1.22896165e-01 2.48800218e-02 5.38197577e-01 9.70587730e-01
 7.40209222e-03 9.40112829e-01 5.90277016e-02 9.87806916e-01
 9.57029700e-01 7.66368747e-01 9.69282031e-01 9.89753604e-01
 1.03915334e-02 9.29826081e-01 4.36582923e-01 1.17555261e-03
 2.00921297e-03 9.98769522e-01 6.16169810e-01 1.80424035e-01
 9.97912467e-01 1.29502118e-02 9.99612331e-01 9.85208154e-01
 9.16777551e-02 6.85240507e-01 1.36416554e-01 3.08935344e-02
 9.93095994e-01 7.66477346e-01 8.38804245e-01 9.04055953e-01
 4.29570675e-04 6.77049160e-04 8.47190976e-01 9.43742216e-01
 9.76216435e-01 1.45392716e-02 1.62425637e-03 1.81762576e-02
 9.20345187e-01 2.86271989e-01 6.06977642e-02 9.94614244e-01
 7.51112044e-01 1.99146837e-01 9.87464666e-01 9.00282025e-01
 1.09335780e-03 8.85091662e-01 8.62828434e-01 5.38905144e-01
 7.34567642e-04 1.43261433e-01 1.13548040e-02 9.71305013e-01
 1.14542246e-02 5.64835429e-01 8.65652680e-01 9.25763011e-01
 6.81143880e-01 7.03918934e-01 6.38434887e-01 4.41679299e-01
 3.20626199e-01 7.20828652e-01 9.52963591e-01 9.99986947e-01
 9.99015450e-01 9.99060869e-01 5.53029120e-01 7.32435107e-01
 7.39240646e-03 9.93993878e-02 1.63965166e-01 8.55287910e-03
 7.56088972e-01 3.59180570e-03 7.06969500e-01 7.96253085e-01
 9.96646166e-01 9.43400979e-01 5.66580296e-02 6.05989814e-01
 8.46337438e-01 9.01604176e-01 2.24717259e-01 3.65820229e-01
 8.91105413e-01 3.56566608e-02 9.99344707e-01 8.46071601e-01
 8.99763107e-01 5.15925884e-03 1.49779618e-02 2.39926666e-01
 5.04373610e-01 9.99798119e-01 6.18347526e-03 5.93525589e-01
 8.46721530e-02 2.93245614e-02 5.50309837e-01 2.35508591e-01
 6.86681867e-02 1.30253524e-01 9.27253962e-01 9.97597098e-01
 9.83039498e-01 2.72254020e-01 7.90394902e-01 8.76170397e-03]  que se encuentra en:  41
Imagen de BD:  48  Imagen de consulta:  112 

Distancia:  2.9134977  al vector:  [9.08506989e-01 9.99995351e-01 9.34330523e-02 9.99952793e-01
 8.99295330e-01 9.99739289e-01 8.19703579e-01 5.79684911e-06
 9.99965906e-01 7.09707737e-02 8.66323709e-04 1.14870638e-01
 1.10158617e-04 7.88644731e-01 9.80223656e-01 9.99989271e-01
 1.48695699e-05 9.99995649e-01 9.99972761e-01 2.65738070e-02
 9.13751423e-02 6.41466113e-06 9.98092651e-01 7.95273736e-05
 8.13287377e-01 2.15727538e-01 7.47398444e-05 9.99623060e-01
 2.02459306e-01 1.93178654e-03 9.98353183e-01 9.28708076e-01
 9.81021166e-01 9.91200209e-01 2.20060468e-01 9.94032962e-06
 1.57614522e-05 8.18565488e-03 5.74569349e-05 9.99911308e-01
 9.99903500e-01 9.77917194e-01 9.30311382e-01 9.99879181e-01
 3.14200918e-07 9.99616742e-01 6.95703626e-01 1.00803375e-03
 2.26807594e-03 9.99987423e-01 2.44334579e-01 2.73687750e-01
 2.91850356e-05 7.22132981e-01 6.32736087e-03 2.11322308e-03
 3.40491533e-04 7.48726070e-01 9.99949217e-01 2.74279842e-06
 8.71968865e-01 1.55842751e-01 9.99294758e-01 6.73568547e-02
 1.07841700e-01 5.92482984e-01 1.69844031e-02 3.31541300e-02
 9.99996185e-01 3.03414792e-01 9.98888254e-01 7.23894536e-01
 2.44090676e-01 6.47532463e-01 8.01413476e-01 6.44523740e-01
 9.99022603e-01 1.06633246e-01 8.68065655e-01 4.18376625e-02
 9.98291492e-01 9.64674234e-01 6.52249515e-01 9.91557479e-01
 6.15675867e-01 3.00249457e-03 9.98596549e-01 9.99990165e-01
 2.57313251e-04 5.42637706e-03 9.95685458e-01 3.25789750e-02
 7.29143620e-04 4.20511067e-02 5.14699757e-01 3.53544950e-04
 9.84049797e-01 4.56933833e-07 1.54539347e-02 9.33141828e-01
 2.84098387e-01 6.51472807e-03 9.95265663e-01 9.99933362e-01
 1.21399909e-01 4.22179699e-04 9.58420277e-01 4.28349995e-05
 9.87969875e-01 2.76623964e-01 9.99986887e-01 3.92087698e-02
 3.26087117e-01 9.95046973e-01 1.80438161e-03 9.99955654e-01
 9.99999702e-01 1.72394663e-01 4.89610434e-03 1.27270818e-03
 1.94376707e-03 1.02337271e-01 5.14987111e-03 9.94589329e-01
 4.67777252e-04 9.99860048e-01 9.99993324e-01 5.30398190e-02]  en la posicion:  16  desde el vector:  [8.45632076e-01 9.94029999e-01 5.17605901e-01 9.97512460e-01
 3.85385752e-01 9.93413329e-01 2.42380619e-01 4.45990509e-06
 9.99834776e-01 8.35371315e-02 5.61441422e-01 4.45327163e-02
 1.98907554e-01 9.61862445e-01 6.28852963e-01 9.99979615e-01
 1.99466944e-04 9.99943495e-01 9.93503273e-01 3.45163941e-02
 7.78645277e-02 3.77601385e-03 9.82593656e-01 2.80788541e-03
 4.40115035e-01 2.40494341e-01 1.06266554e-04 9.95984316e-01
 7.06428289e-02 1.61427259e-03 9.99869227e-01 5.53047538e-01
 8.52769852e-01 9.85717773e-01 8.42370391e-01 5.12909901e-05
 2.24542000e-05 9.72031057e-02 1.02357566e-02 9.23982382e-01
 9.99986172e-01 5.15272617e-01 9.71448064e-01 9.96485114e-01
 8.52823086e-06 9.96154130e-01 1.45283252e-01 1.99642777e-03
 2.36701965e-03 9.99913096e-01 9.89004433e-01 9.05760288e-01
 6.54622912e-03 6.48209512e-01 1.98405087e-02 9.58405435e-05
 1.23739243e-04 6.53896630e-01 9.99837518e-01 1.46555212e-05
 9.74163711e-01 9.83774185e-01 9.99850273e-01 1.68118089e-01
 5.59270382e-04 8.08444023e-01 5.38733006e-02 7.50381649e-01
 9.99689460e-01 7.48629093e-01 9.91806746e-01 6.14975691e-02
 9.46533322e-01 2.82637000e-01 8.45363975e-01 1.25368625e-01
 9.09244120e-01 3.09883177e-01 8.97616744e-01 4.61587220e-01
 9.96023715e-01 9.99719024e-01 9.96162415e-01 9.96249855e-01
 5.82578838e-01 5.24285436e-02 9.99528170e-01 9.99997377e-01
 7.55907095e-05 1.12298548e-01 8.20849299e-01 1.13416165e-01
 1.12600625e-02 7.30752051e-01 9.79076326e-01 6.76035881e-04
 9.87184763e-01 1.34034772e-05 1.11154228e-01 7.25876808e-01
 8.27900231e-01 5.11893034e-02 9.69843984e-01 9.90129113e-01
 1.36046141e-01 1.91807747e-04 9.80312347e-01 5.96791506e-04
 8.32373142e-01 9.36374903e-01 7.77430892e-01 8.87460411e-02
 1.07707113e-01 9.37267363e-01 3.05324793e-01 9.98723745e-01
 9.99941826e-01 1.26194000e-01 1.27811134e-01 8.81528854e-02
 7.40515590e-02 3.24201286e-01 2.20237970e-02 9.99917209e-01
 4.49898839e-03 9.99959826e-01 9.99993682e-01 1.00443363e-01]  que se encuentra en:  42
Imagen de BD:  113  Imagen de consulta:  113 

Distancia:  4.4543347  al vector:  [5.4062718e-01 3.3439015e-11 5.6233549e-01 3.2246709e-03 9.1146249e-01
 4.5518368e-02 4.2511970e-02 2.8610075e-01 7.5398409e-01 4.9543381e-03
 9.9481434e-01 1.2597144e-03 9.9801207e-01 9.9905682e-01 1.9276044e-05
 9.9995446e-01 6.5887272e-03 6.6933289e-08 9.9988747e-01 2.2999942e-02
 7.2993636e-03 1.0000000e+00 2.6244915e-01 9.9886024e-01 9.9685091e-01
 5.4376954e-01 9.9965155e-01 9.9999917e-01 4.7612923e-01 3.9194888e-01
 9.9996650e-01 9.9807578e-01 1.9433767e-07 9.9999678e-01 9.6752191e-01
 2.3517013e-04 9.9909043e-01 1.2913123e-06 9.9999940e-01 7.9163432e-01
 1.4322996e-04 1.1102885e-02 9.8921728e-01 2.7071039e-09 9.9770522e-01
 9.9992460e-01 2.6470804e-01 1.2506580e-01 1.5067648e-05 9.9972343e-01
 9.3228209e-01 5.5194074e-01 9.9427688e-01 9.9699181e-01 3.9630820e-07
 9.8669893e-01 2.6284873e-02 7.7260828e-01 9.9978799e-01 9.9993855e-01
 9.8453820e-01 9.9795318e-01 2.1338546e-01 6.5150529e-01 1.9636902e-08
 9.2538297e-03 2.4827060e-01 9.3853921e-01 3.0550142e-09 9.2754018e-01
 5.5194670e-01 4.9435437e-02 2.2185025e-01 6.9000077e-01 9.7948611e-03
 7.0894969e-01 7.3152184e-03 8.2901073e-01 8.2122701e-01 5.0882286e-01
 9.8932588e-01 9.9967754e-01 1.5180796e-02 9.9999702e-01 9.9321342e-01
 9.7131354e-01 7.4746996e-02 1.2406707e-04 6.0311943e-02 1.0418296e-03
 1.2334007e-09 4.8712134e-02 1.6393602e-02 7.2255713e-01 9.8807156e-01
 2.8384328e-02 9.9999940e-01 1.0000000e+00 3.0587018e-03 4.5029819e-02
 8.9311802e-01 9.4144553e-01 2.3670721e-01 2.6553571e-03 5.4302531e-01
 5.6683248e-06 6.8744206e-01 1.5516281e-03 1.5738606e-04 1.9506353e-01
 7.8818035e-10 6.3250011e-01 1.9027853e-01 3.3650756e-02 4.7215876e-01
 9.9988425e-01 2.0659864e-03 1.2418911e-01 9.3881035e-01 9.9934387e-01
 4.0604741e-07 1.0166937e-01 1.3677478e-02 1.4340878e-03 9.9999762e-01
 9.9740589e-01 9.9970883e-01 6.4368486e-01]  en la posicion:  21  desde el vector:  [9.5943183e-01 3.5398698e-05 3.1203860e-01 8.3410752e-01 1.3709587e-01
 5.7904333e-01 6.3210696e-01 4.2832273e-01 2.7014881e-02 4.0907824e-01
 7.4942964e-01 7.0806950e-02 9.9952126e-01 3.7773931e-01 4.9154633e-01
 7.7251059e-01 7.5639838e-01 4.4220686e-04 9.9815202e-01 1.9712290e-01
 4.2952299e-03 9.9992085e-01 4.1095647e-01 9.9532354e-01 9.9400651e-01
 2.1576226e-02 9.9216819e-01 9.9656188e-01 2.7629286e-02 6.5474015e-01
 7.3149550e-01 1.6912994e-01 3.7233829e-03 9.9978518e-01 4.8775464e-02
 1.3125020e-01 9.3286604e-01 1.3777614e-04 9.9719948e-01 2.9233941e-01
 1.3543397e-02 2.1025869e-01 5.6660557e-01 6.1938167e-04 9.9103552e-01
 9.9616241e-01 9.4399571e-02 6.4782172e-01 1.0180473e-03 9.8222053e-01
 1.5254211e-01 9.6331263e-01 9.8600787e-01 4.1702262e-01 9.0390444e-04
 9.9526459e-01 9.8824221e-01 3.9497495e-02 9.6581209e-01 9.9758267e-01
 6.5104961e-01 3.7951449e-01 2.4541020e-02 4.6032411e-01 1.6410649e-03
 3.2021132e-01 9.7302032e-01 4.4365835e-01 1.6608834e-04 9.9287778e-01
 4.0836677e-01 8.9007956e-01 7.0917875e-02 3.6933547e-01 3.8882977e-01
 8.9482927e-01 1.8718329e-01 5.1429617e-01 2.7167588e-02 1.0865018e-01
 7.3939860e-01 7.7536273e-01 2.1671593e-02 9.9902231e-01 8.4346879e-01
 7.6739657e-01 2.5060028e-01 7.7276975e-02 2.8645390e-01 7.8048706e-03
 5.4067373e-04 6.9235861e-03 1.3760567e-02 1.7960757e-02 4.5584047e-01
 5.1042134e-01 9.9991155e-01 9.9979973e-01 1.6192704e-02 9.4973266e-01
 1.8270999e-01 1.7954165e-01 6.8437093e-01 3.4214589e-01 8.7038177e-01
 1.5852520e-01 9.5275700e-01 4.1468167e-01 7.5025398e-01 2.5838852e-02
 1.5537143e-03 1.9860178e-02 5.3688061e-01 6.8685240e-01 4.2310485e-01
 9.7860432e-01 1.4245510e-02 2.6164097e-01 9.2618287e-01 9.5765829e-01
 5.3331256e-04 2.6718080e-03 9.0372646e-01 2.6507187e-01 9.9458563e-01
 4.6891767e-01 9.9718153e-01 2.6432443e-01]  que se encuentra en:  43
Imagen de BD:  118  Imagen de consulta:  115 

Distancia:  4.243036  al vector:  [6.90981746e-03 9.99912143e-01 6.15294099e-01 2.27721575e-05
 8.60433102e-01 4.58449125e-04 5.24178803e-01 9.99850869e-01
 9.99410272e-01 9.99490678e-01 1.11793743e-04 9.35773671e-01
 3.73909235e-01 6.73082769e-01 3.19968849e-01 5.84584814e-06
 9.99958038e-01 9.99869227e-01 1.37141347e-03 5.97965062e-01
 9.98192847e-01 1.04728341e-03 9.98251081e-01 9.99987841e-01
 5.05685806e-04 1.99107826e-02 9.99749541e-01 1.96120964e-05
 5.00103831e-03 9.99492526e-01 4.03781851e-05 5.92091084e-01
 9.99943972e-01 2.54154205e-04 5.55589795e-03 9.99967694e-01
 9.99476492e-01 9.97748494e-01 9.99659181e-01 6.87971354e-01
 9.99913454e-01 1.59388185e-02 7.13039219e-01 9.99109864e-01
 8.17827167e-05 1.40238744e-05 9.51017618e-01 8.36908817e-04
 9.92389083e-01 2.27330625e-01 6.63856864e-02 7.68152297e-01
 9.67461467e-01 6.11234307e-01 9.97918904e-01 9.99675453e-01
 4.23252583e-04 4.45849061e-01 3.55184078e-01 6.80357218e-04
 9.31349635e-01 5.77320814e-01 2.28258967e-03 2.43670344e-01
 9.99997139e-01 7.31699288e-01 8.70521903e-01 9.54516649e-01
 9.99918818e-01 5.69105744e-01 5.95808029e-04 5.11592388e-01
 9.95947659e-01 4.01188821e-01 1.08120978e-01 8.90768290e-01
 9.99916196e-01 5.72538137e-01 3.99231613e-02 8.14288497e-01
 4.82408404e-02 2.63164639e-02 6.24141097e-03 6.39319420e-04
 9.94575977e-01 6.30476654e-01 1.63471699e-03 3.62131000e-03
 9.99998510e-01 1.96477771e-03 1.19666278e-01 9.93513227e-01
 9.89392877e-01 9.99256790e-01 2.32129991e-02 9.99990523e-01
 1.64317191e-02 1.81463361e-03 9.97745872e-01 1.05180174e-01
 6.86608791e-01 3.19558382e-03 9.98240471e-01 5.82724512e-02
 9.77339029e-01 9.99991536e-01 3.13474848e-05 9.99990404e-01
 8.03195536e-02 5.38063645e-02 9.99419451e-01 9.82536316e-01
 9.58685040e-01 1.22554004e-02 6.24433160e-03 2.33435631e-03
 2.55222559e-01 6.84420168e-01 9.50838447e-01 1.79201365e-04
 9.99471843e-01 4.03766811e-01 4.80207801e-03 6.47467375e-03
 9.97870207e-01 8.16742177e-05 1.81971192e-02 9.90347385e-01]  en la posicion:  71  desde el vector:  [3.31384242e-01 9.53605890e-01 8.98122787e-04 6.24118388e-01
 4.49789643e-01 2.76893198e-01 5.27800500e-01 9.99125600e-01
 2.56606013e-01 9.56251204e-01 4.63548303e-03 1.67939067e-02
 5.44089079e-03 9.82324064e-01 1.91695392e-02 1.25748515e-02
 7.43046284e-01 9.58255708e-01 5.74049354e-03 2.67259687e-01
 2.47733355e-01 4.32533026e-03 1.87368065e-01 1.35056287e-01
 1.59049660e-01 6.25614703e-01 4.21669304e-01 3.65744621e-01
 1.41896993e-01 8.43873322e-01 5.58942556e-04 7.05203891e-01
 9.84230578e-01 4.12637889e-02 6.00522757e-01 8.83202195e-01
 9.88947511e-01 9.96274233e-01 9.85358357e-01 7.32818782e-01
 9.82568264e-01 6.34929061e-01 2.47154921e-01 9.97494221e-01
 5.94170570e-01 2.80976713e-01 9.76080477e-01 6.85611725e-01
 9.19541597e-01 2.18744367e-01 9.45024550e-01 9.07973766e-01
 3.57764065e-02 5.23080945e-01 9.99261081e-01 2.48857319e-01
 4.71350074e-01 1.50799155e-01 3.09301198e-01 4.11614805e-01
 2.96880126e-01 5.87321997e-01 7.40540028e-03 1.39063448e-01
 9.54682112e-01 9.04030323e-01 8.23067784e-01 9.83539402e-01
 3.29597354e-01 9.20825481e-01 9.40464139e-02 8.57866645e-01
 8.93974483e-01 2.41273642e-03 3.07952821e-01 4.11058962e-02
 9.58872080e-01 7.51369953e-01 2.14258432e-01 2.89883792e-01
 4.54480350e-01 1.69728398e-02 1.53616071e-02 1.85479522e-02
 2.35742986e-01 4.98231679e-01 5.16021252e-03 9.97679234e-02
 9.71797585e-01 4.88216132e-01 2.90888250e-01 8.56761456e-01
 6.11493409e-01 9.06569481e-01 1.49926543e-02 9.37194109e-01
 3.07524502e-02 2.02316642e-01 6.37955546e-01 1.80108905e-01
 5.78099430e-01 9.49036479e-01 3.11437666e-01 4.07081634e-01
 9.72835183e-01 9.15714145e-01 2.92365760e-01 9.72340584e-01
 1.77189618e-01 3.09335470e-01 9.93143320e-01 4.71085370e-01
 9.70310450e-01 9.73877907e-02 5.48220575e-02 1.10829085e-01
 9.79614556e-01 6.60860538e-02 7.41375685e-02 3.35323811e-02
 9.81303036e-01 2.77445465e-01 2.45733023e-01 5.00208139e-03
 9.68706548e-01 1.84672773e-02 1.67840123e-02 9.27124023e-01]  que se encuentra en:  44
Imagen de BD:  163  Imagen de consulta:  11 

Distancia:  3.7332287  al vector:  [1.10775232e-02 9.72796421e-09 3.72914076e-01 9.97708201e-01
 9.90433455e-01 8.65380406e-01 8.60156178e-01 8.57584119e-01
 7.91363054e-05 8.08908343e-01 9.99999046e-01 7.11198211e-01
 9.76729572e-01 1.54539376e-01 5.45310795e-01 1.00000000e+00
 1.01608862e-06 1.49190426e-04 7.63925195e-01 4.21337098e-01
 7.01432645e-01 6.77175999e-01 2.62583892e-08 7.64922152e-06
 9.37978923e-02 9.86311257e-01 2.29184570e-05 1.00000000e+00
 1.06152266e-01 7.75602758e-02 1.02759004e-02 1.92219585e-01
 4.80682871e-08 4.90559638e-02 8.74768972e-01 6.22591187e-08
 9.99991655e-01 7.24819899e-01 9.99994159e-01 9.48580146e-01
 3.66094628e-06 7.65280128e-01 6.19434655e-01 9.64143872e-03
 1.00000000e+00 9.99994457e-01 3.34111154e-01 1.00000000e+00
 9.77059543e-01 3.26511264e-03 7.11141229e-02 3.33407581e-01
 9.88230348e-01 9.86545324e-01 9.64192450e-02 1.40133253e-07
 9.95503426e-01 9.99396443e-01 1.08541191e-01 9.99999404e-01
 9.96696949e-01 4.56749111e-01 1.32539868e-02 1.39427334e-01
 4.69256520e-06 2.77185172e-01 4.41670418e-04 5.27644455e-01
 1.69046110e-08 2.63381243e-01 9.78364229e-01 3.14705074e-02
 7.08691001e-01 9.75832880e-01 2.52726674e-01 8.17868173e-01
 1.87602639e-03 3.21291924e-01 9.99734521e-01 7.52551913e-01
 1.04332864e-02 9.53058124e-01 2.74330378e-03 9.82230902e-01
 9.99555588e-01 1.11696362e-01 3.71055007e-02 2.36528397e-01
 1.21257756e-07 9.99988794e-01 6.64581312e-05 4.72107530e-03
 9.68932390e-01 9.59351182e-01 4.84458268e-01 7.19257287e-06
 3.20174843e-01 9.99994397e-01 4.54021692e-01 5.04091382e-03
 1.68038517e-01 1.00000000e+00 3.36266552e-08 3.72179568e-01
 9.86080587e-01 1.35863969e-08 9.64161158e-01 3.35449181e-06
 6.79608405e-01 6.95150197e-02 1.63394009e-06 9.98036981e-01
 3.12670469e-02 9.99660254e-01 3.75830591e-01 5.72021663e-01
 6.08955622e-02 9.16615486e-01 7.62345493e-02 9.99939740e-01
 2.90369749e-01 5.15621006e-01 9.84446347e-01 3.77297401e-04
 9.84403729e-01 9.99991417e-01 9.88638639e-01 5.36108255e-01]  en la posicion:  19  desde el vector:  [3.1850945e-05 2.7797818e-03 5.3895307e-01 9.9750590e-01 9.9645138e-01
 1.4380231e-01 7.7454650e-01 9.5725751e-01 3.9216876e-04 9.8712534e-01
 9.9999899e-01 3.9977235e-01 4.7457075e-01 9.6863520e-01 5.0761664e-01
 9.9986851e-01 3.9759158e-05 1.6724056e-01 1.7563045e-02 5.7914913e-01
 9.9973041e-01 8.6489469e-02 4.4985063e-05 1.1669099e-03 1.6690841e-05
 5.1951408e-03 3.5172701e-04 9.9928391e-01 5.2854449e-02 7.6884675e-01
 6.5123439e-03 5.6725538e-01 3.8473801e-05 4.3541680e-05 2.0252970e-01
 2.7356942e-05 9.9625611e-01 9.9977684e-01 9.9949157e-01 9.4941485e-01
 3.2037795e-03 2.4598438e-01 4.9669859e-01 9.2898440e-01 9.9747753e-01
 9.8267055e-01 3.6123663e-01 9.9998343e-01 9.9996644e-01 1.2326717e-02
 8.2692593e-02 6.1894661e-01 7.8724432e-01 9.6645796e-01 9.9799901e-01
 4.1300058e-04 9.7279537e-01 9.1573328e-01 2.4950182e-01 9.7482371e-01
 8.9590150e-01 2.4429855e-01 2.9239029e-02 7.3465735e-02 5.5164695e-03
 2.5540906e-01 8.1906319e-03 8.8196290e-01 6.6960454e-03 1.1964351e-01
 6.7904389e-01 4.7253934e-01 9.9946249e-01 9.9895632e-01 2.9176176e-03
 5.5777323e-01 5.5154264e-03 5.3904343e-01 9.9790853e-01 8.6175525e-01
 9.3490303e-02 2.2101757e-01 3.9665103e-03 7.4734092e-03 9.9100763e-01
 1.6117716e-01 2.3375988e-02 1.6488597e-01 1.1515539e-04 9.9999297e-01
 3.8576722e-02 9.7760016e-01 6.5662730e-01 9.9995756e-01 4.4433337e-01
 4.0417910e-04 3.6194921e-04 9.9124199e-01 9.9983227e-01 2.4882466e-02
 1.8800893e-01 9.9997216e-01 5.7616609e-05 8.7640047e-02 8.7627721e-01
 1.3238192e-04 8.5982084e-03 1.5270710e-04 1.8356979e-02 3.3576488e-02
 3.2579899e-04 9.9841875e-01 2.0196253e-01 6.7192656e-01 8.8209093e-01
 7.0915222e-02 5.7647914e-01 9.6778822e-01 6.1565912e-01 9.9946284e-01
 9.9882054e-01 5.9703469e-01 8.6602354e-01 4.4512749e-04 8.3622503e-01
 9.9996161e-01 8.3781505e-01 8.9611900e-01]  que se encuentra en:  45
Imagen de BD:  116  Imagen de consulta:  0 

Distancia:  3.9573789  al vector:  [5.4708719e-03 1.2487063e-01 5.1872653e-01 9.9605095e-01 7.0615327e-01
 2.6011467e-04 3.8919151e-02 9.4593203e-01 1.6789788e-01 9.9971330e-01
 9.6268523e-01 8.4009957e-01 9.9562037e-01 9.7167075e-02 9.9673116e-01
 5.3551048e-02 7.6292425e-02 3.0970275e-01 6.2629580e-03 9.0032071e-02
 9.7968191e-01 2.6823997e-02 7.6447672e-01 7.5634038e-01 6.1681867e-04
 7.7086389e-03 7.3106736e-02 7.2024763e-03 6.2988997e-02 9.9959624e-01
 5.6344831e-01 5.9920847e-03 8.0052686e-01 1.9179377e-01 6.9116712e-01
 7.4015331e-01 2.4430037e-02 9.9890363e-01 9.9986982e-01 2.4036491e-01
 1.8821388e-02 1.6263360e-01 1.9580525e-01 8.9359087e-01 2.7687639e-02
 7.9702127e-01 6.5690851e-01 8.1070960e-01 1.3210717e-01 6.9963932e-04
 8.5788876e-02 4.9443811e-02 1.5287399e-03 2.2236079e-02 9.7368008e-01
 4.3588752e-01 7.0457512e-01 1.1975169e-03 4.0499216e-01 3.3514598e-01
 9.8481643e-01 3.1676441e-01 9.4847500e-01 4.2452234e-01 3.2451123e-02
 5.9299296e-01 1.9186735e-01 1.0878897e-01 1.2872517e-03 1.5978402e-01
 1.5670061e-04 5.9969866e-01 9.1967523e-01 8.9159298e-01 7.3006785e-01
 5.1562548e-02 2.2464991e-03 2.0440254e-01 9.0226597e-01 9.4093210e-01
 1.6000605e-01 2.5631875e-02 4.4771761e-02 6.1084449e-02 3.2124120e-01
 3.6504591e-01 2.4261713e-02 9.9417353e-01 4.5440316e-02 5.7604003e-01
 1.0613620e-01 9.9737310e-01 6.8318957e-01 9.5509148e-01 8.9819133e-02
 3.6878341e-01 4.1416287e-02 9.1019005e-01 9.7487420e-01 2.0218918e-01
 5.0530970e-01 5.4380459e-01 6.6068542e-01 4.2909384e-04 2.8269178e-01
 8.6150229e-01 9.6568990e-01 3.9250290e-01 9.3352163e-01 4.0780389e-01
 6.2114000e-04 5.4052103e-01 6.3214749e-02 2.3036614e-01 9.1599494e-02
 1.9460618e-03 9.9807334e-01 5.6568170e-01 8.0451161e-02 9.9884140e-01
 9.8493743e-01 9.6644336e-01 9.6502888e-01 9.1650105e-01 9.9929088e-01
 8.3572650e-01 1.9923151e-03 9.4938791e-01]  en la posicion:  24  desde el vector:  [2.52429247e-02 9.15463209e-01 9.13309932e-01 9.89188194e-01
 1.09052926e-01 1.10192974e-04 7.23546743e-03 4.83813614e-01
 8.28714192e-01 9.99978960e-01 2.35825032e-01 6.86074972e-01
 9.09793854e-01 8.05849016e-01 9.89746809e-01 3.84652615e-03
 6.59950078e-02 7.32129276e-01 7.34199882e-02 1.04701310e-01
 9.82956052e-01 2.22083926e-03 9.99662757e-01 9.74353075e-01
 1.13618582e-04 1.57774091e-02 5.85648477e-01 1.80393457e-04
 6.29877746e-02 9.98994529e-01 4.34130490e-01 2.42115349e-01
 9.94657397e-01 7.20928311e-01 1.69640720e-01 9.96489167e-01
 1.41522288e-03 9.97582197e-01 9.99021888e-01 9.22649026e-01
 4.92549777e-01 6.67699575e-02 3.64148617e-03 9.87377822e-01
 1.64719731e-05 2.21307963e-01 4.05031621e-01 1.01659000e-02
 5.63065708e-02 6.23886585e-02 7.44065642e-03 7.41054714e-01
 4.53763596e-06 2.09652305e-01 9.98203158e-01 9.58283246e-01
 2.99445450e-01 2.45299935e-02 6.80456161e-01 7.20648022e-05
 9.87702250e-01 5.31665981e-01 9.29092288e-01 9.93340015e-01
 8.92024994e-01 8.55821490e-01 9.43350315e-01 1.70465022e-01
 6.02295816e-01 8.18001866e-01 7.70449638e-04 2.89495379e-01
 9.09072042e-01 6.59826934e-01 9.89857793e-01 4.94700402e-01
 5.13222218e-02 7.59351134e-01 9.77941155e-01 7.67246604e-01
 3.76064330e-01 6.07401198e-05 2.48111486e-02 2.85014510e-03
 9.86114383e-01 9.55833972e-01 1.96172595e-02 9.96550322e-01
 8.88885438e-01 9.42763090e-02 9.63365436e-01 9.99972105e-01
 3.03297639e-01 9.64844763e-01 1.85446441e-02 6.40799403e-01
 2.03680992e-02 3.11250269e-01 9.95445788e-01 6.43142283e-01
 1.00082457e-02 2.33227015e-03 9.98664677e-01 4.52068448e-03
 2.75081694e-01 9.98806596e-01 9.72436726e-01 2.09487408e-01
 7.93203712e-01 3.61581594e-01 5.02003431e-02 3.28715801e-01
 4.66684699e-02 8.89831424e-01 1.49738789e-02 4.78343666e-02
 9.99970078e-01 7.74234772e-01 4.89606142e-01 9.90011036e-01
 9.98081088e-01 3.03782821e-01 8.07196975e-01 9.67917562e-01
 9.99783516e-01 5.49784184e-01 1.67493522e-02 8.21638584e-01]  que se encuentra en:  46
Imagen de BD:  120  Imagen de consulta:  120 

Distancia:  3.380507  al vector:  [2.5229859e-01 4.7019839e-02 4.4662002e-01 9.9986279e-01 9.0882927e-02
 8.7487698e-04 9.0495795e-02 2.0149350e-04 9.9989867e-01 1.3398284e-01
 8.1737858e-01 5.0286114e-02 4.6923801e-01 9.3227983e-01 2.0257428e-01
 9.9950141e-01 8.6253881e-04 3.1959054e-01 5.6518477e-01 2.3323178e-02
 9.2463577e-01 7.8690970e-01 9.9999219e-01 7.4776137e-01 5.2305496e-01
 4.4194511e-01 2.5264013e-01 9.2176944e-02 4.2369497e-01 9.9374580e-01
 9.9999273e-01 1.7825606e-01 9.9257278e-01 9.5501596e-01 9.7895831e-01
 3.1480193e-04 3.6987662e-04 1.0985404e-02 4.3642521e-03 9.8428035e-01
 8.0686975e-01 5.0622630e-01 6.1065787e-01 9.8136902e-02 8.0612302e-04
 7.0542079e-01 2.8730270e-01 2.0660848e-06 2.4582714e-02 2.9993862e-01
 3.1499863e-02 2.9331201e-01 1.2065202e-02 3.4784353e-01 9.6900880e-02
 5.9770972e-01 1.1719763e-03 6.9193989e-02 9.9443769e-01 8.5055828e-04
 8.0517066e-01 9.9594837e-01 9.9999833e-01 1.6264027e-01 1.5130639e-04
 7.7734917e-01 4.4978338e-01 7.2052240e-01 6.1025333e-01 9.8445737e-01
 1.4475882e-03 5.3551978e-01 9.9610358e-01 5.2624106e-02 1.5654296e-02
 4.9414185e-01 5.5654943e-03 3.7106544e-02 8.1397980e-02 2.7798176e-02
 2.2281706e-01 9.9596512e-01 9.9654138e-01 9.0090555e-01 1.3254043e-01
 3.4519804e-01 8.1996483e-01 9.9880695e-01 7.2130561e-04 1.2983328e-01
 3.4547561e-01 7.9598659e-01 3.5829216e-01 9.4971842e-01 1.0316038e-01
 5.7020783e-04 9.8636097e-01 9.2514241e-01 7.7917236e-01 8.0030489e-01
 9.8827899e-01 9.1274720e-05 9.9996209e-01 4.4703037e-02 5.3289509e-01
 5.4442883e-04 9.9997580e-01 1.4308393e-03 4.2942911e-02 8.9742219e-01
 8.8605285e-04 4.1192445e-01 9.3344426e-01 9.8509938e-01 6.2915504e-02
 8.4097600e-01 9.9042737e-01 7.0187974e-01 6.7352909e-01 3.3124685e-03
 2.4991304e-02 3.2308865e-01 5.6805104e-02 9.9993366e-01 3.1062740e-01
 9.9908513e-01 6.2754697e-01 1.4063263e-01]  en la posicion:  38  desde el vector:  [6.1350763e-03 9.3006641e-02 5.2407682e-01 9.8164153e-01 2.3077220e-02
 1.4427304e-03 5.4565364e-01 3.0860364e-02 9.9454606e-01 7.7614838e-01
 6.0238099e-01 1.3198620e-01 7.9091829e-01 9.9545884e-01 5.3075433e-02
 8.1727803e-01 2.7264416e-02 1.9332859e-01 1.8049061e-02 1.7734200e-02
 8.4601700e-01 7.5209540e-01 9.9947560e-01 9.2065465e-01 8.7530673e-02
 5.7022721e-02 8.8687599e-01 2.6106834e-03 5.0297678e-03 9.9739933e-01
 9.8723757e-01 7.4973851e-02 8.8800746e-01 7.4389440e-01 3.4780979e-01
 2.6469529e-01 7.2097778e-04 1.3534987e-01 5.9979910e-01 9.8386025e-01
 9.8405015e-01 2.1532580e-01 5.3715897e-01 2.1709359e-01 2.1268576e-02
 2.6614374e-01 5.4409581e-01 1.5550852e-04 5.0432372e-01 6.2521011e-02
 1.5565321e-01 2.3997158e-01 4.0957361e-02 1.4577243e-01 1.3841000e-01
 9.2099380e-01 1.0273755e-03 8.9471966e-02 9.8800701e-01 2.9269516e-02
 9.4624913e-01 6.1205661e-01 9.9981809e-01 9.8872191e-01 5.7804286e-03
 8.8831484e-02 8.4857416e-01 8.2885730e-01 2.2298703e-01 9.9827218e-01
 4.0018559e-04 9.8082697e-01 9.9799055e-01 1.6053766e-02 3.1657737e-01
 7.6277864e-01 2.3050633e-01 1.4429384e-01 3.0503333e-01 1.0457754e-02
 1.4260909e-01 8.7291533e-01 9.5702934e-01 5.6046778e-01 9.7763562e-01
 9.4638121e-01 1.0678950e-01 9.8518622e-01 8.3324641e-02 3.6255842e-01
 2.2179377e-01 9.7435927e-01 6.8477392e-03 9.9912882e-01 2.2614700e-01
 2.3259789e-02 8.6211097e-01 8.9783698e-01 6.8205059e-01 6.7798102e-01
 3.6919916e-01 5.6424737e-04 9.9854094e-01 1.0285974e-03 4.4056228e-01
 9.0871572e-02 9.9193907e-01 1.5709603e-01 3.1562448e-02 1.6675889e-03
 1.2841821e-04 1.3684332e-02 9.9151003e-01 9.2978889e-01 3.4121364e-02
 7.9215527e-02 9.9065626e-01 4.2708868e-01 5.0391221e-01 7.2597206e-02
 6.9472086e-01 1.2575209e-02 1.1063665e-02 9.8095810e-01 9.0334404e-01
 8.6192584e-01 1.5091488e-01 4.9533719e-01]  que se encuentra en:  47
Imagen de BD:  133  Imagen de consulta:  122 

Distancia:  2.6218827  al vector:  [9.99998987e-01 9.99264121e-01 9.05551076e-01 1.83519423e-02
 4.77097332e-01 7.89351583e-01 2.40657628e-02 4.46647406e-04
 9.93994653e-01 1.38900161e-01 4.88337874e-03 9.87992048e-01
 9.76396918e-01 2.99788624e-01 3.81733298e-01 2.34605372e-02
 9.25794065e-01 9.80769277e-01 9.99031961e-01 7.39441037e-01
 1.06943935e-01 2.30637193e-03 9.32960868e-01 2.37134397e-01
 9.92040396e-01 7.06086993e-01 2.90780723e-01 4.36767489e-01
 1.87142819e-01 2.74111092e-01 9.99658227e-01 7.82693028e-01
 9.78103876e-01 9.99870539e-01 1.12731576e-01 8.84403110e-01
 8.35758448e-03 8.31548605e-05 1.09130124e-04 9.96918440e-01
 1.59908235e-02 9.74392891e-01 8.92122388e-01 9.94691253e-01
 5.27962148e-01 4.12519872e-01 3.46961021e-02 1.02519989e-02
 1.79997660e-05 9.99053419e-01 9.91734147e-01 9.31885302e-01
 8.52143168e-01 7.11214542e-03 1.25016622e-05 7.64034867e-01
 9.99900639e-01 1.28165096e-01 6.53541088e-01 2.89189339e-01
 7.54803419e-04 7.27998614e-02 8.63396168e-01 9.69056845e-01
 8.64080191e-01 7.49920547e-01 2.58744359e-02 4.38956738e-01
 9.98113871e-01 9.74961817e-02 7.35541046e-01 8.99912834e-01
 1.78273749e-05 4.12815481e-01 2.19752491e-02 7.09460795e-01
 6.64711654e-01 9.41953421e-01 1.81928277e-03 3.46083343e-02
 2.43633687e-02 8.17747235e-01 1.17232502e-02 9.99982476e-01
 2.70564258e-02 2.56971329e-01 9.99999344e-01 9.91178215e-01
 8.22746515e-01 2.73439288e-03 9.96254206e-01 4.62562788e-08
 1.93354726e-01 6.56154953e-07 8.22562814e-01 8.88517737e-01
 9.99990940e-01 3.84956598e-04 1.36133581e-01 1.49759650e-02
 9.61397350e-01 1.18543983e-01 9.40344810e-01 9.99987841e-01
 3.79468203e-02 4.68451172e-01 9.98602688e-01 9.71144378e-01
 6.50705695e-01 3.64830196e-02 9.99667645e-01 4.93906736e-02
 8.46230328e-01 3.36590022e-01 9.61354733e-01 9.94865179e-01
 7.77841687e-01 1.72745913e-01 1.84451640e-02 8.88973475e-04
 1.79946423e-04 7.34182358e-01 8.47658277e-01 9.79129732e-01
 9.55406666e-01 5.53939641e-02 9.99364138e-01 2.64183521e-01]  en la posicion:  29  desde el vector:  [9.9980295e-01 9.9983895e-01 9.3170512e-01 9.5412821e-02 2.3910344e-02
 8.4175289e-02 4.7783196e-02 1.2578070e-03 9.9940133e-01 7.8777969e-02
 7.7927947e-02 9.1761291e-01 6.2838900e-01 2.5032926e-01 6.0790908e-01
 6.8443197e-01 5.6920111e-01 9.9542046e-01 9.5674390e-01 6.2133181e-01
 7.8979272e-01 1.0596365e-02 9.9329174e-01 4.3034303e-01 9.8722792e-01
 6.2330610e-01 1.7318511e-01 5.1511323e-01 4.3922037e-02 9.1956145e-01
 9.9972343e-01 6.2960327e-01 9.1339028e-01 9.8921561e-01 1.2596527e-01
 3.3699203e-01 5.0072402e-02 1.1607074e-04 2.9799342e-04 9.7576463e-01
 3.8224790e-01 9.8209727e-01 9.7937930e-01 9.5468056e-01 7.8646541e-03
 5.6011200e-02 3.7914544e-02 7.0604682e-04 3.1005740e-03 9.9495173e-01
 9.8318112e-01 9.6439761e-01 6.9459033e-01 1.9632518e-02 7.0595772e-05
 9.5083582e-01 8.8285887e-01 2.4628821e-01 5.4580480e-01 3.2898247e-02
 2.1308631e-02 2.7109557e-01 9.9030244e-01 5.4443151e-01 6.9660354e-01
 4.6723211e-01 1.3121545e-02 9.6283901e-01 9.9872220e-01 6.8852812e-02
 3.9873418e-01 6.9793183e-01 6.5978467e-03 2.5643742e-01 5.8700383e-02
 7.2339135e-01 8.9671218e-01 8.6255181e-01 4.4066608e-03 8.6878943e-01
 1.5774098e-01 9.7946596e-01 4.7834826e-01 9.9973917e-01 1.8365565e-01
 1.6167054e-01 9.9992573e-01 9.8545253e-01 2.1698022e-01 2.1457076e-03
 9.2651349e-01 7.5183102e-06 5.9381473e-01 5.7485700e-04 9.0687609e-01
 7.6186323e-01 9.9991226e-01 9.1105663e-05 6.5590668e-01 3.3670396e-02
 9.6769416e-01 4.3733716e-03 9.9713004e-01 9.9704379e-01 2.3448485e-01
 3.0363947e-02 9.9687755e-01 9.7740930e-01 4.8829719e-01 2.8236717e-02
 9.9297261e-01 3.2018995e-01 6.0118431e-01 3.3536065e-01 9.4873810e-01
 9.5343107e-01 7.1206146e-01 3.1609082e-01 6.2840402e-02 4.1699409e-04
 1.1259317e-03 6.7592442e-01 5.1270562e-01 9.9649054e-01 5.8690864e-01
 1.3591743e-01 9.9619675e-01 1.3379642e-01]  que se encuentra en:  48
Imagen de BD:  125  Imagen de consulta:  125 

Distancia:  4.4387574  al vector:  [8.95960510e-01 1.00000000e+00 8.10444772e-01 9.16874647e-01
 9.75253344e-01 9.56821918e-01 2.08393037e-02 1.53611064e-01
 9.99999046e-01 6.48122966e-01 4.92304437e-08 9.76454973e-01
 5.82179034e-08 6.81825697e-01 9.30335164e-01 2.66851783e-02
 9.78955269e-01 1.00000000e+00 4.20975029e-01 7.63262331e-01
 9.31331396e-01 1.07973094e-07 1.00000000e+00 4.43092316e-01
 6.81138873e-01 4.17006016e-03 1.15204871e-01 1.49396062e-02
 7.18856633e-01 2.08436996e-01 8.80421996e-01 6.58511996e-01
 9.99997854e-01 7.92467594e-03 6.86929584e-01 9.61237192e-01
 7.71969557e-04 8.04801822e-01 5.07206569e-05 9.82151389e-01
 9.99999762e-01 8.88400197e-01 6.13552690e-01 9.99997139e-01
 6.23405660e-10 4.87275034e-01 2.80437291e-01 3.96919262e-08
 9.69939291e-01 9.99998629e-01 9.68998373e-02 1.04674220e-01
 5.41240952e-05 8.73451114e-01 9.95923042e-01 9.93196845e-01
 1.83685677e-06 1.81191355e-01 7.73789048e-01 3.46766615e-09
 7.85513997e-01 1.43275887e-01 2.82579362e-01 1.74590945e-02
 9.99820232e-01 8.88248920e-01 9.66414809e-03 2.52681971e-01
 1.00000000e+00 4.94569540e-03 9.90844607e-01 9.32080269e-01
 9.50281262e-01 8.35563242e-01 4.48102027e-01 6.30972505e-01
 9.99996662e-01 9.92076516e-01 4.95842576e-01 6.61705792e-01
 8.05718303e-02 6.30825758e-03 5.26352406e-01 8.22338462e-03
 5.26749074e-01 1.64359808e-04 8.17781448e-01 8.31316411e-02
 9.96332049e-01 1.03031998e-05 9.30003166e-01 9.79344249e-01
 9.82678115e-01 9.82942224e-01 3.75366509e-02 5.19315243e-01
 4.30489808e-01 3.70554716e-07 9.36457276e-01 1.23539567e-03
 1.60182387e-01 2.08459298e-07 1.00000000e+00 9.62831438e-01
 2.26587057e-03 9.85260308e-01 2.31364369e-03 8.75375152e-01
 8.05858374e-01 2.86854506e-01 1.00000000e+00 7.64581919e-01
 2.44786471e-01 9.82682407e-01 3.65043044e-01 6.46702170e-01
 1.00000000e+00 8.35883677e-01 1.36017799e-04 2.19382855e-06
 9.83530641e-01 9.67980385e-01 3.40034783e-01 2.11723208e-01
 3.66826952e-02 3.86989534e-01 1.52270794e-02 6.18764043e-01]  en la posicion:  108  desde el vector:  [9.97041941e-01 9.99999166e-01 2.21925974e-02 6.76992059e-01
 2.73182690e-02 9.99810457e-01 7.70467758e-01 1.04608744e-01
 9.94398475e-01 8.89747739e-02 2.70366669e-04 8.04260969e-02
 3.42851877e-03 3.90919030e-01 9.57838774e-01 5.26821613e-02
 9.83760536e-01 9.99987483e-01 3.57200414e-01 4.93453443e-02
 5.71417809e-03 1.75076723e-03 9.96163964e-01 4.75593209e-01
 9.92845774e-01 1.91453397e-02 3.23798954e-02 1.92737877e-02
 8.36401999e-01 5.58820367e-03 8.96678388e-01 7.80738831e-01
 9.99970734e-01 1.73076957e-01 5.18058121e-01 8.78012300e-01
 1.90091133e-03 8.92256677e-01 8.14676285e-04 6.17560267e-01
 9.99978244e-01 9.61871445e-01 9.01778758e-01 9.98976648e-01
 4.44635771e-06 4.07427251e-02 5.20163417e-01 1.84893608e-03
 6.41157269e-01 9.99502003e-01 9.16082442e-01 8.92616034e-01
 3.91737819e-02 5.79468310e-02 9.91988957e-01 7.86478460e-01
 2.11542845e-03 2.76771605e-01 6.88947976e-01 7.52533288e-06
 1.44825429e-01 2.23038197e-02 9.00804996e-03 1.08252466e-02
 9.99498367e-01 9.55396771e-01 9.44292963e-01 9.67482507e-01
 9.99999166e-01 4.11611795e-03 9.99030232e-01 1.66803628e-01
 4.80980068e-01 1.10104680e-03 4.21773970e-01 9.11853909e-02
 9.99210954e-01 9.00934517e-01 8.60929489e-02 8.15193057e-02
 1.02669001e-01 8.63497496e-01 9.00407434e-01 5.75554371e-02
 2.13827550e-01 5.38676977e-04 4.47055817e-01 9.44379032e-01
 9.67612922e-01 9.32626128e-02 9.96609390e-01 2.56746233e-01
 7.37819374e-01 2.04627037e-01 2.16551125e-02 7.94596672e-01
 2.24495411e-01 6.92850808e-05 5.61316311e-02 1.94982708e-01
 5.50781429e-01 1.16422474e-02 9.94906068e-01 9.95692015e-01
 2.48381495e-03 9.96439040e-01 3.73590291e-02 9.76947546e-01
 7.06555903e-01 3.87660503e-01 9.99986053e-01 1.11577272e-01
 9.24402118e-01 8.79117846e-03 3.22499752e-01 9.79514241e-01
 9.87366319e-01 3.48326564e-03 2.75240541e-02 1.21969133e-04
 3.15296233e-01 9.57168937e-01 2.18772888e-01 2.33276486e-01
 1.02245808e-02 3.58986914e-01 8.18846166e-01 5.45922816e-01]  que se encuentra en:  49
Imagen de BD:  197  Imagen de consulta:  139 

Distancia:  3.7675998  al vector:  [9.78194535e-01 9.99880910e-01 6.26135409e-01 9.98112559e-01
 1.60790622e-01 2.72260606e-02 6.31319165e-01 1.70363601e-05
 9.89551127e-01 9.97728288e-01 6.02036715e-04 5.40853739e-02
 9.69252408e-01 7.98580408e-01 9.98793244e-01 9.29653645e-04
 2.49720097e-01 9.99245167e-01 9.94737148e-01 1.18601441e-01
 6.53415620e-01 8.78900290e-03 9.99995947e-01 9.80144858e-01
 5.76276779e-02 2.93099821e-01 9.67898607e-01 9.43958759e-04
 4.16682333e-01 6.60214543e-01 9.94061470e-01 6.49550557e-02
 9.99859095e-01 9.99692082e-01 2.63428390e-01 9.99037504e-01
 3.35624506e-08 8.08187604e-01 2.27731168e-02 9.13276553e-01
 9.98870015e-01 6.04139566e-01 7.32605457e-01 9.92387235e-01
 1.16371147e-07 4.01876241e-01 8.61010551e-01 5.52976526e-05
 2.20984221e-04 9.99441028e-01 5.68518043e-03 3.11097205e-02
 5.49942015e-05 1.51874423e-02 9.87134457e-01 9.68367934e-01
 1.59001589e-01 9.35434341e-01 5.14410913e-01 7.46647459e-07
 1.43772483e-01 4.03157920e-01 9.66445029e-01 9.95013118e-01
 9.99164462e-01 7.02465892e-01 8.62574100e-01 8.79140794e-01
 9.99889016e-01 9.74451363e-01 4.72585618e-01 1.33036822e-01
 7.41402805e-02 2.46415585e-01 6.71091080e-02 8.66979361e-01
 7.17083991e-01 1.64098173e-01 9.14818585e-01 1.29950225e-01
 8.65641415e-01 6.58363104e-04 1.36117011e-01 1.71215802e-01
 3.05083632e-01 9.91411209e-02 8.67634654e-01 9.99999642e-01
 9.89025474e-01 1.32024288e-04 9.99905109e-01 8.02864552e-01
 6.78944588e-03 3.80559266e-02 8.37318003e-02 8.51208091e-01
 5.86126149e-02 3.58283520e-04 9.86808419e-01 6.61153316e-01
 4.41660911e-01 1.93685293e-04 9.99498725e-01 9.13104475e-01
 7.19507933e-02 9.97802317e-01 5.75258315e-01 6.52396619e-01
 9.61593330e-01 9.74104822e-01 9.97909606e-01 1.16116345e-01
 6.34902954e-01 1.72018379e-01 8.63510370e-03 9.98021126e-01
 9.99983251e-01 4.31012392e-01 9.49544191e-01 8.56423080e-02
 6.98573351e-01 1.37014985e-02 3.05151343e-01 9.99067068e-01
 3.81708324e-01 7.37770379e-01 9.96066511e-01 6.84983671e-01]  en la posicion:  172  desde el vector:  [9.28021550e-01 9.99732852e-01 1.53441578e-01 5.82754374e-01
 1.99947536e-01 1.78157181e-01 1.08599246e-01 4.84241247e-02
 9.40766811e-01 5.84519506e-01 6.71299398e-02 4.09283012e-01
 7.70861983e-01 8.16441655e-01 5.90842783e-01 2.68110037e-02
 4.93964881e-01 9.94768143e-01 9.41448510e-01 4.18769956e-01
 1.92405313e-01 5.52411854e-01 9.89344060e-01 9.66680884e-01
 1.83306009e-01 2.81351089e-01 7.56331325e-01 1.35454535e-03
 1.37890279e-02 6.92528903e-01 9.36533570e-01 4.95409250e-01
 9.99152601e-01 9.28798676e-01 1.02211893e-01 9.56978381e-01
 1.83743238e-03 5.98849237e-01 1.83905363e-02 9.39582169e-01
 9.94862199e-01 5.67947626e-01 6.78775668e-01 8.41627240e-01
 1.07177460e-04 2.03041136e-02 9.79303122e-01 5.64661622e-03
 3.47458720e-02 9.91765380e-01 4.48594391e-02 7.36152291e-01
 6.13274574e-02 1.33322507e-01 6.83498561e-01 7.94943511e-01
 4.84360754e-02 8.58122170e-01 2.03295648e-02 5.53438658e-05
 4.08991754e-01 3.47164094e-01 9.43863750e-01 3.27282488e-01
 9.96896267e-01 7.59727240e-01 2.63615131e-01 9.70405221e-01
 9.98647690e-01 9.41470087e-01 8.69443178e-01 9.56929326e-01
 7.43426204e-01 1.90752387e-01 1.61838531e-03 9.79693472e-01
 7.57370710e-01 8.26058865e-01 8.07772756e-01 2.34643698e-01
 6.18893504e-02 6.39249325e-01 4.09328371e-01 4.29049045e-01
 7.20333576e-01 5.28636754e-01 8.68681312e-01 9.92629170e-01
 9.13280129e-01 4.98759210e-01 9.97108579e-01 7.08708286e-01
 9.95990038e-01 4.61295307e-01 7.09146261e-04 4.04458702e-01
 9.33109403e-01 2.04171598e-01 7.85546541e-01 4.83406305e-01
 8.39155316e-01 1.32849813e-03 9.97611165e-01 9.14049506e-01
 2.63952494e-01 9.82420087e-01 8.72961164e-01 7.49206245e-01
 1.03900135e-02 2.21008033e-01 9.97981429e-01 9.58230257e-01
 9.27733362e-01 3.58007848e-02 7.23057389e-02 9.88497853e-01
 8.52766454e-01 5.49932301e-01 3.69482338e-02 8.14154744e-03
 3.90029907e-01 5.07745743e-01 1.00565225e-01 9.73903775e-01
 2.85515100e-01 6.33038878e-01 9.88633037e-01 2.70991445e-01]  que se encuentra en:  50
Imagen de BD:  71  Imagen de consulta:  142 

Distancia:  3.8833575  al vector:  [9.5168579e-01 9.9846447e-01 3.0871254e-01 8.8060474e-01 2.7718580e-01
 4.0668249e-03 6.7020690e-01 7.7212128e-05 9.9910456e-01 9.9692303e-01
 2.3448467e-04 1.0389954e-02 9.9180758e-01 5.3122491e-01 9.9859953e-01
 8.8435411e-04 9.3193591e-01 9.9743068e-01 9.8944128e-01 1.5626252e-02
 7.0710301e-01 9.8250359e-02 9.9999726e-01 9.9736953e-01 1.0170162e-01
 2.7949262e-01 9.9817348e-01 4.5707822e-04 4.2337409e-01 8.7543380e-01
 9.9857593e-01 3.3669591e-02 9.9993217e-01 9.9984467e-01 7.0884514e-01
 9.9931133e-01 5.1492913e-07 5.6287473e-01 2.8156757e-02 7.2115725e-01
 9.9452364e-01 1.6677088e-01 6.9620281e-01 8.6626768e-01 1.1569973e-06
 2.9905081e-02 6.2916648e-01 8.7486251e-06 3.2767653e-04 9.9930108e-01
 1.1104196e-02 3.5946226e-01 2.1839142e-04 6.0270309e-02 8.8323748e-01
 9.9025786e-01 8.2275033e-02 5.3743839e-01 2.7016434e-01 2.6031098e-06
 1.6817668e-01 6.7707622e-01 9.9957001e-01 9.6753860e-01 9.5349830e-01
 7.5140750e-01 5.3332508e-01 5.6298852e-01 9.9949682e-01 9.8434794e-01
 7.5415909e-02 1.7840290e-01 5.8715016e-02 7.7220798e-02 2.9966444e-02
 6.5820462e-01 5.3497761e-01 1.7140156e-01 8.4885883e-01 3.0246115e-01
 9.7444636e-01 1.7232299e-02 7.9665738e-01 8.3162510e-01 6.5155417e-02
 1.7380205e-01 8.4914184e-01 9.9999845e-01 9.8993969e-01 3.1005242e-05
 9.9879575e-01 4.4395515e-01 1.4556289e-02 5.5218726e-02 8.5386842e-02
 9.7952628e-01 5.8254993e-01 9.4297528e-04 9.3949187e-01 7.3785770e-01
 8.5438168e-01 3.3447217e-05 9.9996829e-01 5.9949023e-01 7.2971195e-02
 9.9913192e-01 8.5465926e-01 9.9142849e-01 8.1023914e-01 9.5112807e-01
 9.6870947e-01 1.6860703e-01 7.0528448e-01 5.7621396e-01 4.0389121e-02
 9.8124754e-01 9.9974525e-01 2.0543543e-01 9.6774244e-01 4.1293800e-03
 3.3727491e-01 6.8063498e-02 2.4624759e-01 9.9995196e-01 3.4786582e-01
 1.0100821e-01 9.9866354e-01 5.5776066e-01]  en la posicion:  40  desde el vector:  [9.98070776e-01 9.99999404e-01 3.18917632e-03 4.76124942e-01
 5.94756186e-01 6.30989671e-01 2.13127434e-02 6.72370195e-04
 9.93865967e-01 9.46212292e-01 3.72463465e-03 2.77088583e-01
 9.01176810e-01 6.68985605e-01 9.01541770e-01 1.26293898e-02
 4.64790136e-01 9.99999404e-01 9.90544915e-01 9.20743048e-01
 3.27064455e-01 1.03768706e-03 9.91532743e-01 3.15102994e-01
 3.14772129e-02 2.76064575e-01 2.82309771e-01 3.06695700e-04
 1.81369007e-01 1.85451716e-01 9.86987472e-01 4.94442791e-01
 9.99963760e-01 9.68568265e-01 9.33830082e-01 9.64414656e-01
 5.33401635e-06 9.52448010e-01 9.29115504e-06 9.50145602e-01
 9.95259166e-01 7.70307302e-01 9.65443373e-01 9.99689162e-01
 6.43394787e-06 8.54793191e-03 9.61302638e-01 3.33806872e-03
 2.22508609e-02 9.93205786e-01 5.49369991e-01 5.67389667e-01
 3.21647525e-03 5.59494495e-01 9.14399147e-01 2.42896408e-01
 7.66350627e-02 9.10233796e-01 1.49427056e-01 2.45612318e-06
 6.58519089e-01 8.00288320e-02 9.98375237e-01 6.47125006e-01
 9.99816000e-01 6.27701998e-01 1.72043830e-01 9.92644131e-01
 9.99995589e-01 6.48312569e-01 8.85033727e-01 8.39180350e-01
 5.79705536e-02 4.63416278e-01 7.81676173e-03 9.79722917e-01
 5.81086576e-01 3.51208329e-01 2.14883268e-01 1.55285031e-01
 2.74963945e-01 4.94053364e-02 5.55505574e-01 3.94659579e-01
 2.15871274e-01 1.64318681e-02 9.97700453e-01 9.99982178e-01
 9.69469190e-01 4.66460198e-01 9.99994516e-01 1.43875301e-01
 9.99660254e-01 1.76852047e-02 1.66746974e-03 6.78049922e-01
 4.91558373e-01 5.73910256e-06 6.98608756e-01 3.39600265e-01
 9.28167343e-01 9.34851170e-03 9.92902040e-01 9.99543607e-01
 1.39087737e-02 9.94100094e-01 9.64420319e-01 9.56047297e-01
 3.17075849e-03 8.17999601e-01 9.99991477e-01 9.64139044e-01
 9.58663523e-01 1.22826576e-01 1.57130957e-01 9.97777760e-01
 9.96242404e-01 1.01745933e-01 1.82180107e-01 1.61921978e-03
 7.33186424e-01 9.10151601e-01 8.17317963e-02 9.98952746e-01
 6.00298941e-02 1.42005891e-01 9.98469949e-01 4.11656469e-01]  que se encuentra en:  51
Imagen de BD:  135  Imagen de consulta:  142 

Distancia:  3.6401231  al vector:  [3.8587302e-02 2.3830861e-02 1.6722253e-01 9.9999702e-01 4.3301135e-02
 6.1630613e-01 4.5166194e-02 9.5352530e-04 9.9575204e-01 2.2027194e-03
 9.9966991e-01 5.0671160e-02 5.4223746e-01 9.0058720e-01 3.4050590e-01
 9.9998975e-01 1.1106133e-03 8.5703290e-01 7.3177576e-01 3.0044252e-01
 1.1288777e-01 9.9850595e-01 9.3361044e-01 6.4681709e-02 9.3898225e-01
 2.9789239e-01 1.8208534e-02 9.3556339e-01 6.3646519e-01 1.7171171e-01
 9.9964631e-01 4.6683520e-02 1.1838400e-01 9.4621295e-01 8.9976406e-01
 2.6394560e-05 1.0236263e-04 3.5590827e-01 5.8103055e-02 9.6898973e-01
 9.9901891e-01 9.5534265e-01 7.9992712e-03 1.4409572e-02 4.1490376e-02
 9.1963923e-01 3.7727463e-01 4.1964948e-03 9.0989560e-02 8.2121193e-03
 9.2798126e-01 7.0160651e-01 1.9865558e-01 8.4469229e-02 9.0810657e-04
 2.7299523e-03 5.2212235e-06 1.9841066e-01 8.6124712e-01 7.1176887e-04
 8.9220297e-01 9.6598035e-01 9.9999976e-01 4.1724038e-01 3.8889857e-06
 3.9461255e-04 2.8852433e-01 8.0761516e-01 3.3075097e-01 9.9936199e-01
 7.4290264e-01 2.7102202e-02 9.8729253e-01 1.2068808e-02 9.9851757e-01
 9.9720263e-01 1.3427258e-01 2.9373765e-03 1.0013312e-02 4.6775043e-03
 5.9274119e-01 9.9997199e-01 9.9999964e-01 8.9957178e-01 9.2718208e-01
 9.1121662e-01 9.6527749e-01 9.9999821e-01 4.9182022e-06 9.6595323e-01
 9.8516387e-01 6.2649232e-01 9.9202895e-01 3.3748144e-01 1.7164528e-02
 5.2402393e-06 6.4346898e-01 1.4799848e-01 7.9688609e-02 8.6828434e-01
 4.4750416e-01 1.2787849e-02 9.9081039e-01 3.6594719e-02 3.0562252e-01
 9.5192518e-05 9.9323767e-01 2.3103541e-05 2.6789665e-02 3.7934929e-02
 1.4477968e-04 1.6607702e-02 9.9933314e-01 9.9477434e-01 1.2288639e-01
 8.9851105e-01 9.7414303e-01 1.7731920e-01 8.5889149e-01 1.9759649e-01
 2.6221609e-01 7.5157553e-02 9.4347370e-01 9.9998653e-01 5.6246939e-05
 9.9939179e-01 7.9743314e-01 2.9381514e-03]  en la posicion:  134  desde el vector:  [8.84510577e-02 5.57541847e-04 1.25789255e-01 9.95788217e-01
 1.67969346e-01 1.57414436e-01 6.27096295e-02 9.14736092e-02
 9.99889493e-01 1.64010763e-01 9.99275327e-01 1.51747674e-01
 9.42059398e-01 9.83868778e-01 1.77428693e-01 9.99958754e-01
 1.21566057e-02 4.44706410e-01 3.57911676e-01 8.83349299e-01
 2.40263611e-01 9.97800052e-01 6.81914628e-01 8.53905201e-01
 8.15688491e-01 3.91682386e-02 4.05526817e-01 9.99447346e-01
 9.03350353e-01 7.75808215e-01 9.82036889e-01 2.85592854e-01
 1.54349208e-03 5.54475665e-01 9.71821070e-01 2.66925272e-05
 9.21530724e-02 5.29957056e-01 9.90717292e-01 9.83240366e-01
 9.92506802e-01 2.91517466e-01 2.70621777e-02 1.81662261e-01
 6.28935516e-01 9.69027877e-01 1.68243647e-02 7.15625882e-02
 3.00367773e-01 1.40920281e-03 9.87703204e-01 9.69706893e-01
 9.67208683e-01 6.35886192e-03 1.23835116e-05 8.09836388e-03
 1.24517408e-06 2.51114607e-01 9.89309311e-01 4.48265791e-01
 5.55354476e-01 9.91747379e-01 9.99997199e-01 7.93205857e-01
 4.47021371e-08 3.18056107e-01 6.13069117e-01 6.08609855e-01
 3.66568565e-04 9.98501897e-01 1.08048797e-01 9.31973755e-02
 8.32514584e-01 7.18931973e-01 8.97587776e-01 9.12540317e-01
 8.29146504e-01 7.30140507e-01 3.03049088e-02 4.85039055e-02
 9.93865013e-01 9.99972641e-01 9.99993205e-01 9.99725819e-01
 9.95606601e-01 9.80503440e-01 5.27818501e-02 9.99978900e-01
 1.07642409e-04 1.00404054e-01 2.34645277e-01 3.47093046e-02
 9.89560246e-01 7.28704214e-01 6.41564608e-01 1.52349472e-03
 5.06673276e-01 3.93449903e-01 8.06279480e-02 9.86789107e-01
 5.53140044e-01 1.39794558e-01 8.15042198e-01 3.39856744e-03
 4.83821481e-01 2.19672918e-04 5.89188874e-01 4.72431779e-02
 3.15294862e-02 9.63968039e-03 1.34348869e-04 2.90465355e-01
 9.45247412e-01 9.56079602e-01 2.64188081e-01 1.70069069e-01
 1.46695137e-01 1.45124942e-01 4.34166759e-01 1.22090399e-01
 4.82943982e-01 1.77339405e-01 9.45536196e-01 9.99996543e-01
 2.66730785e-04 9.95194793e-01 9.66407061e-01 1.92498565e-02]  que se encuentra en:  52
Imagen de BD:  37  Imagen de consulta:  2 

Distancia:  3.5269208  al vector:  [1.34115219e-02 1.40874237e-01 6.38979673e-01 9.99994397e-01
 9.20878947e-02 1.13564134e-02 2.67043144e-01 4.13488060e-01
 1.59984827e-03 8.74638677e-01 9.89533544e-01 2.75534928e-01
 6.54049635e-01 3.68033111e-01 8.89071226e-01 9.89609361e-01
 6.65363259e-05 2.35275030e-01 4.00310755e-03 4.98162240e-01
 9.14141178e-01 6.77585602e-04 3.47666355e-05 2.31602788e-03
 3.23921442e-04 2.33149290e-01 9.56909498e-05 1.82294458e-01
 1.58826739e-01 9.75441039e-01 2.47275829e-03 4.18328851e-01
 2.80471623e-01 3.69092822e-03 6.71504438e-02 1.11347735e-02
 1.23182386e-01 9.99992788e-01 9.97914076e-01 8.43065977e-01
 7.54994154e-01 8.41098249e-01 6.57046318e-01 9.90276694e-01
 5.58997154e-01 8.63440752e-01 4.23779070e-01 9.99904275e-01
 9.79815125e-01 3.07175517e-03 8.40147138e-01 9.19496536e-01
 1.48534477e-02 8.90077353e-02 9.99557555e-01 1.03048267e-04
 9.92013097e-01 3.16622287e-01 6.48971856e-01 5.49057841e-01
 7.37824380e-01 1.16700321e-01 4.07116711e-02 7.15275109e-01
 1.71485841e-02 8.67605150e-01 2.58732021e-01 5.65424919e-01
 4.07675594e-01 5.13922632e-01 2.74826586e-02 2.16831803e-01
 9.83025193e-01 9.49475050e-01 1.97373629e-02 7.17670023e-01
 3.99291515e-04 9.27052975e-01 4.95695174e-02 6.97999418e-01
 4.02770042e-02 5.80766201e-02 9.55745578e-03 4.55260277e-04
 7.95728922e-01 3.16340148e-01 2.90286839e-01 9.34423029e-01
 7.79110193e-03 9.99897122e-01 5.68734407e-01 9.99243855e-01
 6.71122313e-01 9.89779949e-01 2.20252872e-02 7.38149720e-06
 1.24321805e-05 7.04269469e-01 9.90995884e-01 9.86301064e-01
 8.91705394e-01 9.92381036e-01 4.24259901e-03 4.78527844e-02
 5.12118340e-01 2.40488350e-02 3.42343509e-01 1.88142061e-04
 6.99382126e-02 8.79788399e-02 5.88024557e-02 3.00526947e-01
 8.93231452e-01 4.18453932e-01 7.87131011e-01 5.53894043e-03
 9.99405921e-01 4.01334375e-01 6.17948174e-02 9.99601126e-01
 9.99973297e-01 2.82519758e-02 9.84457135e-01 4.69343364e-02
 8.53894591e-01 9.99968171e-01 2.70434737e-01 2.57936001e-01]  en la posicion:  50  desde el vector:  [1.4541328e-02 8.8280267e-01 6.6556871e-01 9.9998468e-01 3.1042141e-01
 2.2623956e-01 2.4242592e-01 6.2415898e-03 9.0220928e-02 9.8028922e-01
 6.3653797e-01 5.2888125e-01 7.6836556e-01 5.1477444e-01 9.6590638e-01
 7.1064562e-01 3.3196807e-04 9.7332692e-01 2.8465736e-01 4.1996837e-02
 9.0029192e-01 8.5493922e-04 3.0027366e-01 2.2378892e-02 2.3236871e-04
 5.8199942e-02 1.5957654e-03 4.2375714e-02 5.2769321e-01 3.8321382e-01
 3.3033282e-02 3.8157314e-01 8.3433604e-01 1.7110851e-01 8.8747013e-01
 1.6172746e-01 1.7151237e-04 9.9995601e-01 9.5857912e-01 8.0720764e-01
 9.9427736e-01 6.3273925e-01 9.0908694e-01 9.9638122e-01 3.2392144e-04
 9.2713284e-01 6.0378754e-01 9.8320866e-01 1.7836034e-01 1.3425112e-01
 9.6348691e-01 5.4433274e-01 1.3221204e-03 3.0926475e-01 9.9919724e-01
 2.3069382e-03 5.8957511e-01 9.1851425e-01 9.5260012e-01 1.5634596e-03
 6.3452435e-01 6.7906111e-01 3.8121495e-01 9.4965279e-02 3.4344101e-01
 8.2479227e-01 8.3751112e-02 9.6425581e-01 9.9420226e-01 3.2910413e-01
 2.8106678e-01 9.4301701e-03 9.8753130e-01 9.7084296e-01 9.5261037e-03
 6.4424157e-01 5.6608915e-03 7.5443083e-01 4.1710594e-01 6.9045365e-02
 3.9147943e-02 8.7105632e-03 8.3125114e-02 2.4074614e-03 5.6992382e-01
 1.2614313e-01 7.6020193e-01 9.9986768e-01 8.3410561e-02 9.9335760e-01
 9.9104643e-01 9.9962366e-01 2.2796717e-01 8.7332553e-01 3.2196403e-02
 3.1375885e-04 2.5752187e-04 2.8900564e-02 9.6167499e-01 9.2021608e-01
 6.5076566e-01 9.6756130e-01 8.3735198e-02 5.7968318e-01 5.0612992e-01
 5.2012700e-01 1.3949716e-01 2.7677417e-04 1.3567623e-01 3.8458824e-02
 6.9351947e-01 1.6016188e-01 9.9214208e-01 3.5238361e-01 7.2577113e-01
 8.7458014e-01 9.9996924e-01 4.3118262e-01 6.3179255e-02 9.8712516e-01
 9.9965525e-01 4.5756340e-02 5.1761246e-01 8.1248331e-01 8.1016922e-01
 9.9974853e-01 9.5132393e-01 6.6949546e-02]  que se encuentra en:  53
Imagen de BD:  144  Imagen de consulta:  144 

Distancia:  2.2320416  al vector:  [1.17046356e-01 1.09328255e-04 8.18962038e-01 9.22090232e-01
 3.14858317e-01 2.51927972e-03 2.81305313e-02 5.35849452e-01
 2.34350562e-02 3.69960964e-02 9.99907613e-01 8.51947188e-01
 9.92695451e-01 9.15878952e-01 9.10548806e-01 9.99937534e-01
 7.95304775e-04 1.40377879e-03 3.60733569e-02 7.15745986e-02
 8.18600178e-01 9.70938087e-01 5.66869974e-03 5.73490560e-02
 9.27245200e-01 9.13735509e-01 4.16702032e-03 9.83345866e-01
 4.09539342e-02 9.87423182e-01 9.91127133e-01 4.99453843e-02
 1.96425319e-02 8.20323288e-01 7.85166919e-02 3.40256192e-05
 9.91926551e-01 3.43689024e-02 9.94000673e-01 7.95369148e-02
 6.69449568e-04 3.45064998e-02 8.65200877e-01 2.68983841e-03
 9.93991494e-01 9.12327051e-01 7.57192612e-01 9.71040905e-01
 2.79336751e-01 4.97409701e-03 2.85676658e-01 8.10755849e-01
 9.46421146e-01 6.23579025e-01 2.76452899e-02 9.84966755e-03
 9.96404469e-01 1.88948095e-01 5.28250694e-01 9.95940566e-01
 2.98994482e-01 7.61168480e-01 6.08409405e-01 7.83438027e-01
 3.56272358e-05 9.66913104e-01 5.51215708e-01 9.35108423e-01
 6.90788031e-04 7.32306957e-01 1.76495016e-02 8.67688239e-01
 9.50703025e-01 1.19350553e-02 5.88347018e-02 6.26003027e-01
 5.60859226e-05 3.95562410e-01 8.06416571e-02 9.43996787e-01
 7.94588208e-01 9.99625146e-01 8.78433883e-02 9.98081505e-01
 9.63476896e-01 5.75386286e-02 7.79094577e-01 1.06158644e-01
 1.47104263e-04 9.93802905e-01 1.11228228e-03 6.92315698e-02
 9.13542867e-01 8.38797092e-01 1.76132888e-01 1.04218721e-02
 9.91094410e-01 9.99493718e-01 8.49141657e-01 2.31966853e-01
 6.92064822e-01 9.55210686e-01 7.13400245e-02 2.43583322e-02
 7.35866010e-01 3.93997434e-05 9.99938786e-01 8.90898705e-03
 1.04087621e-01 2.25122154e-01 6.22654716e-06 4.31403518e-02
 9.50822473e-01 8.90648067e-02 6.12688065e-03 3.07450593e-01
 3.97447348e-02 8.01172495e-01 1.41547382e-01 9.52098608e-01
 9.56496596e-03 4.76744771e-01 8.21314096e-01 6.64125919e-01
 9.67561483e-01 9.99375820e-01 9.47975755e-01 4.68549997e-01]  en la posicion:  53  desde el vector:  [1.19971037e-02 8.85191548e-05 8.89555454e-01 1.05509013e-01
 6.87437832e-01 2.12353468e-03 8.06308687e-02 9.51935709e-01
 1.48446858e-02 2.94791043e-01 9.99948025e-01 9.73483086e-01
 9.98281658e-01 8.89144421e-01 9.33474302e-01 9.99935865e-01
 3.70193124e-02 8.31037760e-03 4.01660800e-03 2.28887707e-01
 9.87377882e-01 9.61292028e-01 7.20858574e-04 4.32210267e-02
 7.02521443e-01 9.21342015e-01 2.11450458e-03 9.95123148e-01
 8.81004035e-02 9.94615912e-01 6.80028677e-01 4.98964787e-02
 9.72062349e-04 3.60418856e-02 3.10352504e-01 4.64451005e-05
 9.98857737e-01 4.21232373e-01 9.97135222e-01 1.23157203e-01
 2.43672729e-03 2.21379101e-02 8.37482274e-01 1.14447713e-01
 9.98968005e-01 7.29056954e-01 7.35709667e-01 9.92680907e-01
 9.62722182e-01 4.14907932e-04 6.33923054e-01 7.87356615e-01
 9.90547776e-01 4.76955742e-01 1.65877074e-01 3.42813134e-03
 9.84915674e-01 4.68942225e-02 1.64662063e-01 9.97647882e-01
 7.54073381e-01 8.98875773e-01 6.13224268e-01 8.23780656e-01
 4.55508962e-05 9.71494436e-01 7.85300851e-01 6.23948693e-01
 9.71253976e-05 8.64468157e-01 1.46159530e-02 5.93725324e-01
 9.86812472e-01 1.79753423e-01 3.57379913e-02 6.76067829e-01
 1.19906664e-03 2.01714873e-01 3.83914471e-01 9.60348129e-01
 8.38687301e-01 9.99228358e-01 1.21332854e-01 9.95676339e-01
 9.85424638e-01 2.30995804e-01 3.26661766e-01 1.38507962e-01
 1.16698066e-04 9.77084875e-01 3.70323658e-04 1.18882209e-01
 8.92131925e-01 9.74766016e-01 4.67522860e-01 2.18712837e-01
 8.29360247e-01 9.96388078e-01 9.62137818e-01 3.62503588e-01
 7.54919767e-01 9.96768832e-01 7.96949863e-03 5.71551919e-03
 9.42993283e-01 7.30995525e-05 9.96127486e-01 1.13006264e-01
 4.57396805e-02 1.42028451e-01 1.60213585e-05 8.18655491e-02
 3.43279958e-01 9.26154256e-02 8.42516720e-02 1.72314048e-02
 1.46722794e-02 4.08844084e-01 3.97492349e-02 9.79441285e-01
 5.65739572e-02 6.69724047e-01 8.57307673e-01 5.44761062e-01
 9.72898722e-01 9.91915882e-01 9.36841488e-01 9.24630463e-01]  que se encuentra en:  54
Imagen de BD:  147  Imagen de consulta:  147 

Distancia:  4.45089  al vector:  [7.19971120e-01 1.00000000e+00 7.02857971e-03 3.04979503e-01
 9.97884929e-01 2.78515220e-02 9.47516203e-01 9.63904738e-01
 8.62961769e-01 9.50101852e-01 4.90350112e-06 1.83030963e-03
 1.00277226e-04 9.99832511e-01 7.58221745e-03 8.78467563e-06
 9.88263845e-01 9.99999046e-01 8.14513862e-02 2.90348530e-02
 9.25784290e-01 4.82404801e-07 9.92848516e-01 2.86947548e-01
 2.97397375e-04 6.75002635e-02 8.02852213e-01 6.75797462e-04
 9.03068125e-01 9.84911323e-01 4.59933281e-03 8.03973198e-01
 9.99998569e-01 1.59208745e-01 8.57374191e-01 9.99825478e-01
 9.51878726e-02 9.99511838e-01 1.84448451e-01 7.88814068e-01
 9.96130884e-01 6.53883994e-01 6.60365522e-02 9.99997973e-01
 1.26196037e-05 7.90416896e-02 3.33259970e-01 6.07189536e-03
 8.81028175e-01 9.99786019e-01 3.72165978e-01 9.68350708e-01
 3.27313410e-06 9.81025219e-01 9.99999046e-01 9.64299560e-01
 7.27809548e-01 4.31244701e-01 2.26039886e-02 1.72974160e-05
 1.73398852e-03 4.87296373e-01 1.33246183e-04 1.29518926e-01
 9.99985099e-01 4.55766052e-01 7.34366953e-01 8.74600530e-01
 1.00000000e+00 3.03352177e-01 6.40131354e-01 7.92222023e-01
 2.68731177e-01 4.81287241e-02 5.93182147e-01 2.00314522e-02
 9.63545024e-01 1.93541139e-01 9.60289240e-01 7.28635788e-02
 9.65710163e-01 1.98036432e-04 2.33355840e-05 6.87360764e-04
 4.32300925e-01 3.35518986e-01 2.21075743e-01 2.28050947e-01
 9.99491692e-01 2.19153225e-01 9.97024536e-01 9.98838425e-01
 2.56088316e-01 8.83730888e-01 1.82318687e-03 9.82524991e-01
 1.04390681e-02 1.58871387e-06 9.89679575e-01 1.10875785e-01
 6.35328889e-03 8.42362642e-04 9.99693394e-01 8.80260229e-01
 8.62538815e-04 9.99981880e-01 7.97915757e-02 9.75188971e-01
 9.83246565e-01 5.53196967e-02 9.99999881e-01 9.91007149e-01
 8.41089249e-01 2.35382944e-01 3.18982005e-01 2.76338577e-01
 9.99960661e-01 2.24334002e-03 9.67321992e-01 5.34594059e-04
 9.98101592e-01 9.60953712e-01 9.51965630e-01 3.32981348e-04
 9.98947501e-01 1.65485740e-02 5.60390949e-03 9.62087512e-01]  en la posicion:  12  desde el vector:  [3.79708648e-01 9.82727766e-01 1.06027722e-03 9.05842721e-01
 4.63696718e-01 7.83463478e-01 5.77334344e-01 9.97276664e-01
 4.16087419e-01 7.93477178e-01 3.46541405e-03 1.43279135e-02
 1.49172544e-03 9.90310967e-01 4.03678417e-02 7.58462548e-02
 5.27485728e-01 9.94509459e-01 5.03230095e-03 2.56847650e-01
 8.33838284e-02 2.02265382e-03 2.24553168e-01 2.41120756e-02
 3.38448435e-01 7.81357348e-01 1.11906052e-01 5.10815978e-01
 1.39502168e-01 4.12012339e-01 9.21279192e-04 6.36311710e-01
 9.93942738e-01 3.37119997e-02 5.31113863e-01 6.43650711e-01
 9.59479451e-01 9.97574925e-01 7.78637230e-01 7.40992546e-01
 9.97607827e-01 6.89103603e-01 3.25835049e-01 9.99337077e-01
 2.08560765e-01 2.97291994e-01 9.67477679e-01 7.20080733e-01
 9.66842890e-01 2.56136894e-01 9.65009391e-01 8.21208119e-01
 7.10164011e-02 5.54952145e-01 9.99631047e-01 4.79080975e-02
 1.47645891e-01 3.59601885e-01 4.67141658e-01 2.18385249e-01
 3.18651557e-01 5.99159181e-01 1.67963207e-02 1.33966655e-01
 9.35098946e-01 8.60343814e-01 7.76674867e-01 9.76329327e-01
 7.45440483e-01 9.32633758e-01 4.38636988e-01 8.39220941e-01
 8.88358414e-01 7.67180324e-03 4.56646800e-01 4.63734865e-02
 9.87358809e-01 8.66231799e-01 3.25681776e-01 3.71070802e-01
 4.40204889e-01 8.81750286e-02 1.47257745e-01 7.34245777e-03
 1.89726740e-01 3.84025127e-01 4.24005091e-02 2.28852689e-01
 8.42858553e-01 5.81359148e-01 8.45584869e-01 7.88968921e-01
 6.72424674e-01 8.76903236e-01 1.20804310e-02 6.37605429e-01
 1.02300644e-02 2.82017589e-02 3.67503136e-01 2.18998075e-01
 7.60731101e-01 9.59844887e-01 3.34167838e-01 7.06346393e-01
 9.63659167e-01 6.29323959e-01 2.63954759e-01 8.94136310e-01
 1.61943316e-01 2.18716204e-01 9.98258710e-01 7.13427901e-01
 9.75069880e-01 1.67161912e-01 5.05480468e-02 1.41244888e-01
 9.76136684e-01 1.25897914e-01 6.80346489e-02 1.92006528e-02
 9.86084580e-01 3.44609082e-01 3.44153881e-01 8.49196315e-03
 6.34313047e-01 5.09661734e-02 2.97522545e-02 6.97417259e-01]  que se encuentra en:  55
Imagen de BD:  11  Imagen de consulta:  11 

Distancia:  3.856977  al vector:  [8.89669478e-01 9.99997497e-01 2.70470679e-01 9.99998212e-01
 6.22138977e-02 9.99986708e-01 9.77578640e-01 6.54975224e-07
 9.98693109e-01 1.80131197e-03 1.03621215e-01 3.01323742e-01
 7.06193714e-06 9.48661447e-01 8.90657842e-01 9.99998808e-01
 4.71892953e-03 9.99912977e-01 9.97191072e-01 2.15220898e-01
 1.15914077e-01 5.67346811e-04 9.99954402e-01 2.57521868e-04
 9.98635888e-01 7.02407956e-03 3.97706099e-05 9.99660432e-01
 1.18794411e-01 4.38302755e-04 9.99419451e-01 1.27033889e-02
 7.55025744e-01 1.32889777e-01 7.73127973e-02 3.78019149e-06
 1.41620636e-04 3.63055468e-02 5.55927818e-06 9.68229175e-01
 9.99999404e-01 9.74044085e-01 9.18008029e-01 9.91660893e-01
 7.46392956e-08 9.98736739e-01 1.97880447e-01 7.08282896e-05
 4.39405590e-01 9.99958873e-01 8.12578380e-01 4.21009064e-02
 6.19515777e-03 2.21591860e-01 3.15895975e-02 3.10384631e-02
 8.29136934e-06 6.91363215e-03 9.98998106e-01 6.69499229e-07
 7.19460309e-01 8.12331915e-01 9.46948528e-01 9.35320258e-02
 1.51114464e-02 9.87384379e-01 2.29652196e-01 9.50043440e-01
 9.99999642e-01 1.27619207e-02 9.99724746e-01 8.00286889e-01
 9.88556504e-01 1.64676309e-02 4.74992961e-01 7.89433718e-04
 9.99940515e-01 7.65297830e-01 2.34530658e-01 1.62083775e-01
 7.99478829e-01 9.95111942e-01 9.99177158e-01 9.87147033e-01
 2.89164782e-02 6.03140593e-02 6.10583186e-01 9.99967575e-01
 2.33161973e-06 8.81689787e-03 9.78042662e-01 1.79694772e-01
 2.95488644e-05 7.25949883e-01 3.15630674e-01 8.28981400e-04
 9.65502977e-01 7.67105030e-06 1.76268816e-02 4.08647537e-01
 6.17100596e-01 1.71196461e-03 9.99663949e-01 9.99318123e-01
 7.43903399e-01 4.52644854e-05 9.97175336e-01 4.66883183e-04
 9.99304771e-01 3.80634665e-02 9.99953389e-01 2.94869542e-02
 8.48831534e-01 4.01622206e-01 8.46336842e-01 9.98830795e-01
 9.99911129e-01 4.25630957e-01 6.90239489e-01 4.44501638e-04
 1.31155550e-02 1.13553673e-01 5.52240074e-01 9.96711254e-01
 1.98890502e-05 9.99927759e-01 9.99114931e-01 4.43909973e-01]  en la posicion:  177  desde el vector:  [1.99721128e-01 8.63073528e-01 1.37046486e-01 9.95273054e-01
 6.90520048e-01 5.21754146e-01 9.24317837e-01 7.33470917e-03
 9.69767094e-01 1.23081207e-02 9.65666175e-01 8.78199935e-03
 5.98338544e-02 6.21502340e-01 3.23252380e-01 9.79437947e-01
 6.12961650e-02 8.97357106e-01 6.09486639e-01 6.00774944e-01
 7.89405584e-01 3.40099931e-01 9.89461958e-01 5.30239046e-02
 5.73909640e-01 1.88401341e-02 5.44364154e-02 8.31878304e-01
 5.89582562e-01 8.75309467e-01 9.93604600e-01 4.75525856e-02
 8.27312648e-01 6.37934804e-01 6.26321495e-01 6.02141023e-03
 1.97910964e-02 2.42386520e-01 2.58583724e-02 9.37431097e-01
 9.81879354e-01 1.27638727e-01 9.99565363e-01 7.26347744e-01
 1.67477131e-03 8.30713272e-01 1.74971312e-01 6.10572100e-03
 4.76205111e-01 9.84843850e-01 7.14565873e-01 7.10711241e-01
 4.63102758e-02 3.93485308e-01 5.49273312e-01 4.95772034e-01
 4.97980714e-02 1.53036952e-01 9.96726513e-01 6.07660413e-03
 6.42540693e-01 8.33934009e-01 9.27967429e-01 9.35488939e-03
 2.75277853e-01 2.20483869e-01 9.50937748e-01 7.47427642e-01
 8.49252582e-01 4.54979181e-01 1.55948192e-01 9.87366557e-01
 9.72331405e-01 8.20493698e-03 7.45458484e-01 1.02739453e-01
 1.08024478e-02 1.73156768e-01 6.29925728e-02 1.94145530e-01
 9.61207747e-01 9.78902102e-01 9.62057114e-01 7.78889298e-01
 2.32315063e-03 1.09721124e-02 5.13124704e-01 9.98798847e-01
 9.45061445e-04 2.90425897e-01 6.46197677e-01 8.18017364e-01
 3.44430596e-01 9.87793803e-01 4.15519655e-01 3.45338583e-02
 3.18464339e-01 1.04965836e-01 6.21179640e-01 2.93253183e-01
 1.07291937e-01 3.86244863e-01 9.81042504e-01 1.65276289e-01
 9.88809824e-01 1.68226063e-02 9.65669513e-01 4.22259271e-02
 9.85141397e-01 6.42571211e-01 7.38310337e-01 2.18228728e-01
 6.64138138e-01 7.68101811e-02 8.79846871e-01 9.77597117e-01
 9.97840166e-01 3.58967811e-01 5.11509240e-01 1.05199516e-02
 1.23290718e-01 8.72209907e-01 2.55900800e-01 9.66557860e-01
 4.69834208e-02 9.79083776e-01 9.76015687e-01 6.34597838e-02]  que se encuentra en:  56
Imagen de BD:  76  Imagen de consulta:  148 

Distancia:  4.2334194  al vector:  [7.96202658e-05 7.69734383e-04 9.33544874e-01 4.64615226e-03
 3.33310783e-01 5.65685696e-07 3.51357460e-03 8.93946767e-01
 9.99985158e-01 9.92480040e-01 5.91671646e-01 7.55304337e-01
 9.88877177e-01 7.06218541e-01 7.12658346e-01 1.87961996e-01
 5.09100795e-01 7.67439604e-04 4.65459824e-02 8.30877179e-05
 9.88623142e-01 9.99404252e-01 9.99985516e-01 9.99999046e-01
 8.60939026e-02 9.27440047e-01 9.99999642e-01 1.52873397e-02
 1.66967511e-02 9.99999881e-01 9.93028104e-01 9.99734700e-01
 1.85972929e-01 9.82760489e-01 9.23324585e-01 9.67980027e-01
 1.15279257e-02 1.75278485e-02 9.99820590e-01 3.48210573e-01
 3.74644995e-04 1.18884295e-01 9.54397202e-01 2.78533626e-05
 3.94582748e-04 6.32429421e-02 3.92046064e-01 5.85929058e-07
 9.64581966e-04 6.26510382e-03 8.98850560e-02 5.42545497e-01
 3.62922549e-02 9.27059889e-01 9.12767053e-02 9.99999404e-01
 1.32569671e-03 4.24689531e-01 3.13264251e-01 3.29956412e-03
 9.92118597e-01 9.89722729e-01 9.99993145e-01 8.90419841e-01
 1.92072988e-03 2.19781131e-01 2.83968449e-02 8.96151185e-01
 3.84718180e-04 1.89983010e-01 4.59713050e-08 1.28126740e-02
 9.99910355e-01 9.06654119e-01 5.70165217e-01 8.87707233e-01
 9.48429108e-04 8.78716052e-01 9.88383234e-01 8.87503803e-01
 9.42439198e-01 1.05314225e-01 1.54460132e-01 9.56868231e-01
 9.79670525e-01 8.08125436e-02 1.39653683e-04 8.89180183e-01
 9.28626359e-01 1.67340040e-04 6.48051500e-04 9.99978364e-01
 9.93122697e-01 9.99380469e-01 8.85111928e-01 8.93533945e-01
 9.76301312e-01 9.99993682e-01 9.90998864e-01 7.81770110e-01
 8.69583845e-01 1.49003949e-06 9.99999642e-01 9.66271000e-06
 4.46109831e-01 5.99024296e-01 8.16833138e-01 7.32110798e-01
 1.52826309e-04 6.75205231e-01 7.80154733e-06 1.76504970e-01
 7.89162755e-01 2.55592406e-01 7.89684653e-02 2.68724561e-03
 9.64963555e-01 6.25192225e-02 2.59452760e-02 7.99423277e-01
 8.88916850e-03 5.29973805e-02 2.69356668e-02 9.97311652e-01
 9.99992430e-01 1.51590258e-01 5.22004068e-02 9.38660145e-01]  en la posicion:  25  desde el vector:  [9.11214948e-03 1.44592227e-12 3.83561671e-01 2.70536542e-03
 3.14201415e-02 9.75297496e-08 1.20112002e-02 9.99990225e-01
 6.55329823e-01 9.81448531e-01 9.76178169e-01 1.07176751e-01
 9.99911666e-01 1.90600753e-01 2.38809764e-01 5.30435383e-01
 9.90556717e-01 9.71689656e-12 5.09268045e-03 1.28726065e-02
 7.84753859e-01 9.99999404e-01 5.99288106e-01 9.99998808e-01
 4.75625902e-01 9.86300111e-01 9.99999762e-01 9.58219588e-01
 8.00737441e-02 9.99999523e-01 4.15962696e-01 9.98095334e-01
 7.97331333e-04 9.82144594e-01 9.87516642e-01 9.10623312e-01
 9.99970496e-01 2.46527791e-02 1.00000000e+00 2.21586227e-01
 5.09004792e-07 7.71522164e-01 9.96085703e-01 1.58080560e-09
 9.99991834e-01 4.83602315e-01 4.77042645e-01 2.77498960e-02
 9.07921791e-03 1.38536096e-03 7.53965378e-01 9.36725736e-01
 9.40418601e-01 9.69148159e-01 4.70700860e-03 9.99975681e-01
 9.68988240e-01 9.89170611e-01 1.22430235e-01 9.99996781e-01
 9.26699579e-01 6.63141966e-01 8.02657366e-01 9.98177409e-01
 1.22938286e-06 8.16038907e-01 8.89057219e-02 9.66940761e-01
 2.65429033e-11 9.97711062e-01 4.10188594e-09 1.47325695e-02
 9.94745374e-01 9.94764209e-01 1.83838218e-01 8.92675638e-01
 4.29239208e-06 9.83455777e-01 9.66005921e-01 6.50312006e-02
 9.28778589e-01 1.32528245e-02 7.32278824e-03 9.55164194e-01
 5.77210486e-01 9.04421687e-01 1.37722236e-05 4.34595586e-06
 9.64883327e-01 3.25819850e-03 8.12272860e-10 9.92539048e-01
 4.46283937e-01 9.99372005e-01 5.03080606e-01 8.52584362e-01
 9.58217740e-01 1.00000000e+00 8.43362987e-01 6.18034303e-02
 8.15743923e-01 3.64358425e-01 8.85459423e-01 2.36099442e-08
 1.18334144e-01 7.74458051e-01 9.98600364e-01 6.43063903e-01
 1.39306784e-02 9.93460953e-01 4.82497237e-11 4.16755080e-01
 1.09559238e-01 1.92447931e-01 1.77760720e-01 9.46735745e-05
 1.34285033e-01 1.20433509e-01 1.40821308e-01 9.99910057e-01
 8.13813508e-02 5.99157810e-03 8.74997258e-01 3.83612216e-02
 1.00000000e+00 1.92415714e-02 1.25187635e-03 9.81907964e-01]  que se encuentra en:  57
Imagen de BD:  121  Imagen de consulta:  150 

Distancia:  3.6680732  al vector:  [5.8720016e-01 8.8244283e-01 8.4461057e-01 9.9999809e-01 5.0859153e-03
 9.5226860e-01 2.0739138e-03 2.4754405e-03 9.9447852e-01 3.4487844e-03
 8.5866618e-01 5.8684438e-02 5.8967054e-02 9.0174395e-01 2.7131552e-01
 9.9926543e-01 1.2049694e-05 7.8531808e-01 9.7888529e-01 3.3244771e-01
 3.2040477e-04 8.3291966e-01 9.3211293e-01 1.0882759e-01 9.7431731e-01
 4.6227515e-02 2.7134567e-02 9.9733222e-01 6.3572228e-03 7.3698044e-02
 9.8851603e-01 5.3481519e-02 5.7175356e-01 9.9147934e-01 1.5043944e-02
 1.2876093e-03 3.1927615e-02 7.4279159e-02 9.8368406e-01 8.8973767e-01
 9.9987864e-01 9.8921609e-01 3.0709982e-02 6.1956179e-01 1.1667311e-03
 9.9818522e-01 4.4692105e-01 5.8904588e-03 3.2103658e-03 9.7178322e-01
 3.5825223e-01 9.9813688e-01 5.7841235e-01 8.4675133e-02 2.0569831e-02
 2.2751093e-02 1.2391806e-04 2.9405028e-02 9.6854067e-01 8.0311298e-04
 9.7856653e-01 8.7753516e-01 9.9482012e-01 1.5229899e-01 1.4236569e-04
 6.9876522e-02 6.1277872e-01 7.5427568e-01 9.5264935e-01 9.4219595e-01
 9.8555040e-01 3.6196893e-01 3.6087382e-01 1.9201252e-01 7.3123062e-01
 9.1635466e-01 7.6894617e-01 9.7446638e-01 2.3146644e-01 9.3467045e-01
 7.2243613e-01 9.9988306e-01 9.9977803e-01 9.5550144e-01 7.1048480e-01
 9.5194960e-01 9.9035728e-01 9.9771172e-01 1.7152280e-02 6.8796104e-01
 5.0436747e-01 8.2950944e-01 9.2025983e-01 1.9174111e-01 1.7766625e-02
 5.5727287e-06 8.0285525e-01 4.3739665e-01 9.5832348e-04 1.1211148e-01
 3.2097867e-01 8.6158514e-04 9.9948132e-01 3.3496064e-01 7.4065328e-01
 4.0657222e-03 7.3168159e-01 2.9817224e-04 1.4683902e-01 9.9813348e-01
 3.4151244e-01 7.5998962e-01 4.0425003e-02 2.8260797e-01 2.1034664e-01
 9.9835443e-01 8.0442399e-01 4.5960441e-01 3.6990011e-01 3.1405023e-01
 7.9654515e-02 1.6697079e-02 6.2048769e-01 9.9606311e-01 8.2853436e-04
 9.9997199e-01 9.9645066e-01 2.8945049e-05]  en la posicion:  8  desde el vector:  [1.9390529e-01 9.6665215e-01 8.1428212e-01 9.9999928e-01 1.8776810e-01
 9.2267942e-01 1.4046013e-02 2.9453781e-06 8.7959450e-01 1.4731884e-03
 9.7361612e-01 1.5968055e-01 4.3980479e-03 9.0430534e-01 5.8702457e-01
 9.9999940e-01 2.0340248e-07 9.9856377e-01 9.5351160e-01 2.2889435e-02
 3.3237457e-01 1.4655888e-03 8.1990767e-01 1.5354132e-06 6.6691726e-01
 2.4563757e-01 3.1627576e-06 8.8726604e-01 2.0145622e-01 9.7820461e-03
 9.9897748e-01 4.3949422e-01 9.6496010e-01 9.2123824e-01 8.7262166e-01
 2.4173460e-06 3.5104156e-04 2.3341963e-01 2.1967292e-04 9.9406767e-01
 9.9882400e-01 9.1652948e-01 7.9705650e-01 9.5149696e-01 7.9020858e-04
 9.9917829e-01 9.4551820e-01 1.9372076e-02 7.2756320e-02 9.2346901e-01
 5.4604411e-03 3.2768178e-01 2.9820502e-03 3.3083725e-01 3.9850339e-01
 2.2368324e-05 1.5165687e-02 4.4129744e-01 9.9662542e-01 4.7107041e-03
 3.6788571e-01 9.7011352e-01 9.9901420e-01 1.3418078e-02 1.0809608e-04
 2.2935718e-02 6.5562606e-02 9.7906733e-01 9.9704325e-01 9.6673608e-01
 9.7775793e-01 9.2066538e-01 4.9442971e-01 1.4249158e-01 5.7544386e-01
 9.6876395e-01 1.8321455e-02 5.5692279e-01 5.4085255e-03 6.6474646e-01
 7.8566897e-01 9.9979764e-01 9.9448514e-01 4.7163785e-02 3.5515210e-01
 3.5422683e-02 9.9985069e-01 9.9994391e-01 6.1735653e-07 9.9762267e-01
 9.9711156e-01 3.7934983e-01 9.8710179e-01 8.7341058e-01 6.9824290e-01
 1.2092956e-07 5.5904353e-01 3.6857426e-03 2.1546641e-01 6.9135427e-03
 8.9810312e-01 1.0854563e-01 8.0538440e-01 8.7312144e-01 4.3212116e-01
 1.2096883e-06 9.8657811e-01 3.1268539e-07 5.5994284e-01 9.3012226e-01
 5.2696139e-02 6.5205187e-01 1.0152444e-01 9.4636023e-01 2.8386328e-01
 9.9680746e-01 9.9988222e-01 9.7538829e-01 6.3582623e-01 1.7357355e-01
 2.7589047e-01 1.4373392e-02 2.4904817e-01 9.8772013e-01 2.0891428e-04
 9.9999982e-01 9.8752785e-01 3.8647652e-04]  que se encuentra en:  58
Imagen de BD:  106  Imagen de consulta:  156 

Distancia:  3.2713308  al vector:  [9.8853731e-01 9.9945676e-01 7.7734065e-01 9.9998987e-01 9.8770756e-01
 9.9130845e-01 1.5412360e-02 8.6958840e-05 7.7247620e-01 9.2009974e-01
 2.9573798e-01 7.8707856e-01 7.6167345e-02 4.7595796e-01 3.8600183e-01
 9.9968475e-01 2.7719140e-04 9.9973774e-01 9.9991214e-01 8.2770288e-02
 4.0293366e-02 4.9855127e-05 8.9998257e-01 8.1711113e-03 4.3136397e-01
 4.1176468e-02 1.2617707e-03 9.9957299e-01 2.6237965e-04 7.3260069e-03
 9.9974668e-01 2.8479391e-01 6.3904524e-03 9.9889481e-01 5.4643035e-02
 3.8224459e-04 2.7480721e-04 3.8039088e-02 6.9639087e-04 9.9994314e-01
 6.3403416e-01 2.9305226e-01 4.0748546e-01 9.9965203e-01 7.8579068e-02
 9.9987608e-01 9.8764861e-01 1.0413736e-01 1.9189715e-04 9.9506247e-01
 7.9353553e-01 2.6821196e-01 2.0038486e-03 1.7980844e-02 3.1530857e-04
 8.7436140e-03 9.5813268e-01 2.9405206e-02 2.2886011e-01 8.9301914e-02
 7.2890818e-03 5.4866761e-02 9.9429578e-01 1.2285823e-01 2.5367439e-02
 4.7481805e-02 1.8839836e-03 9.4580764e-01 9.9856383e-01 7.4853301e-03
 9.9556601e-01 9.6739215e-01 1.0552704e-03 3.1055248e-01 2.6109517e-03
 9.9933600e-01 7.4017650e-01 8.0833250e-01 1.0791838e-01 3.6772519e-02
 1.8318114e-01 1.2028405e-01 1.3441637e-01 9.9921250e-01 9.8133588e-01
 1.6070217e-01 9.9804568e-01 9.9991345e-01 8.1843573e-05 5.0162196e-02
 9.9965954e-01 7.3563159e-03 8.9879155e-01 1.5047193e-04 8.9195557e-05
 1.3390183e-04 9.9915379e-01 1.7607212e-04 1.0064840e-01 3.5807866e-01
 4.5322716e-02 5.8464348e-02 8.1920910e-01 9.9999458e-01 2.3696661e-02
 1.0064205e-04 9.9750531e-01 5.6648254e-04 7.0338905e-01 4.7762662e-02
 9.9967176e-01 1.4375895e-02 9.9461204e-01 7.1067923e-01 2.5400048e-01
 9.9995732e-01 9.9996710e-01 9.3183410e-01 4.8885801e-01 1.9874334e-01
 1.1205077e-03 1.5309772e-01 4.3937474e-01 9.9477601e-01 8.0280781e-02
 9.9989808e-01 9.9993080e-01 1.0339302e-01]  en la posicion:  162  desde el vector:  [9.99960065e-01 9.99992728e-01 8.36159110e-01 9.89794731e-01
 1.22846663e-01 9.99926090e-01 4.47809696e-04 4.77468967e-03
 9.95392561e-01 2.01020539e-02 1.63078308e-04 8.80004168e-01
 5.65624237e-03 6.54923320e-02 8.30036640e-01 9.96656835e-01
 2.85730720e-01 9.99990344e-01 9.99999285e-01 1.58988535e-01
 1.28448009e-04 1.79591775e-03 9.98965502e-01 5.23436368e-02
 9.99966979e-01 7.35765696e-02 8.43037367e-02 9.99899745e-01
 1.38202906e-02 1.34617090e-04 9.99145091e-01 1.09603226e-01
 7.73809910e-01 9.99981701e-01 1.78544074e-01 1.14440024e-02
 1.38789415e-04 4.54581777e-06 3.15386444e-07 9.98906195e-01
 9.99273896e-01 2.17984259e-01 2.15258390e-01 9.92109358e-01
 1.23200371e-04 9.99853611e-01 7.83014059e-01 7.15797287e-05
 9.47839726e-05 9.99981761e-01 3.14954460e-01 5.26872516e-01
 1.10542178e-02 5.63929379e-01 4.22097901e-06 4.07516509e-01
 4.09626961e-03 2.87976921e-01 5.47962427e-01 1.52081251e-04
 9.61262286e-01 4.85583127e-01 9.99212205e-01 7.05848515e-01
 1.27755821e-01 2.93346047e-01 3.86011899e-02 4.66032088e-01
 9.99991417e-01 4.23586369e-01 9.99947309e-01 9.84223664e-01
 4.14814458e-05 4.43882853e-01 2.96305418e-02 9.98919129e-01
 9.99996543e-01 5.31532764e-02 6.43450022e-02 5.39552987e-01
 3.12664628e-01 9.85524774e-01 9.98242974e-01 9.99999523e-01
 8.97832811e-01 1.32624269e-01 9.90928471e-01 9.95645940e-01
 7.80442357e-03 2.45579076e-06 9.97066021e-01 4.73957334e-06
 8.79739165e-01 2.05061451e-06 6.98496997e-02 1.46234035e-01
 9.99997616e-01 5.58552529e-06 8.79347324e-04 1.31861448e-01
 3.96674275e-01 4.78476286e-04 9.99948740e-01 9.99999702e-01
 2.29908228e-02 1.78951323e-02 9.90114093e-01 2.61196792e-01
 4.57541168e-01 8.53410661e-02 1.00000000e+00 1.81794167e-02
 9.91387904e-01 8.57692599e-01 8.62411857e-02 9.99998748e-01
 9.97533798e-01 5.78911722e-01 7.42372870e-03 1.05151485e-04
 1.65512654e-06 1.85358226e-02 1.06383801e-01 9.93169308e-01
 1.39862299e-04 9.72621679e-01 9.99997258e-01 5.23162961e-01]  que se encuentra en:  59
Imagen de BD:  62  Imagen de consulta:  157 

Distancia:  3.83305  al vector:  [1.7284945e-01 9.8572123e-01 6.0568482e-02 9.9999851e-01 6.6381902e-01
 9.8364151e-01 8.9713359e-01 1.5944242e-04 9.2567658e-01 1.1620671e-02
 3.8334370e-02 6.4135462e-01 9.2098117e-04 1.0277909e-01 5.4000533e-01
 9.9710536e-01 2.6810239e-05 6.8612218e-01 9.9545223e-01 3.1147665e-01
 8.3780289e-04 8.2357824e-03 9.9840647e-01 7.0631504e-03 9.8964274e-01
 3.0706227e-03 7.4490905e-04 6.0798031e-01 9.9583286e-01 6.1118811e-02
 9.9423367e-01 1.7487288e-02 9.0484059e-01 9.4370461e-01 3.7366055e-05
 3.0499995e-03 4.3852226e-05 7.7236760e-01 6.5572810e-01 9.6870577e-01
 9.9992466e-01 3.0999798e-01 2.3379028e-03 9.8669291e-01 1.8411407e-05
 9.9971592e-01 2.0329058e-03 6.4151287e-02 3.6720872e-02 9.4681597e-01
 5.4054439e-02 5.0663972e-01 9.9485219e-03 4.1878343e-02 4.8270357e-01
 1.7818987e-01 2.1362343e-01 3.5721064e-04 9.9607241e-01 3.2210350e-04
 2.2383738e-01 6.5085667e-01 5.0163174e-01 1.2003574e-01 5.2141249e-03
 5.2023399e-01 9.9874991e-01 4.8553228e-02 9.9838817e-01 1.8267035e-03
 9.4551480e-01 9.9863851e-01 4.9572605e-01 1.3673520e-01 9.8682106e-01
 7.2413099e-01 1.8190357e-01 6.4301878e-02 1.1448270e-01 6.6032112e-03
 5.6987935e-01 9.3945515e-01 8.3547461e-01 4.7131062e-02 9.7213948e-01
 6.2655580e-01 8.0183810e-01 9.9995261e-01 2.5999546e-04 6.8369961e-01
 9.9222600e-01 9.9888355e-01 4.0624827e-02 9.8233330e-01 5.7377529e-01
 8.3182422e-06 2.8128773e-02 5.6731284e-02 7.9422683e-02 9.4342083e-01
 1.9642711e-04 6.5109730e-03 9.9468195e-01 6.2719059e-01 1.0333210e-02
 6.2693059e-03 7.7476573e-01 5.6872386e-06 9.8940837e-01 4.0892065e-03
 9.0048242e-01 8.5292595e-05 9.3656695e-01 9.9388683e-01 3.6637813e-01
 9.9218619e-01 9.9940234e-01 6.2300330e-01 4.0972412e-01 4.8821551e-01
 9.0883875e-01 4.1797033e-01 8.0214709e-02 9.8753512e-01 1.4584690e-02
 9.9999714e-01 8.0619943e-01 1.4460087e-04]  en la posicion:  193  desde el vector:  [4.69643295e-01 9.96303797e-01 4.41234350e-01 1.00000000e+00
 3.20925355e-01 9.76729631e-01 1.02575988e-01 3.37319739e-06
 8.85001481e-01 8.29385281e-01 5.19532204e-01 9.88945723e-01
 1.47689968e-01 5.35422564e-03 5.55236936e-01 9.99958754e-01
 4.03224840e-06 9.83586073e-01 9.88767028e-01 8.96631837e-01
 1.22920394e-01 3.78310680e-04 6.39083743e-01 1.65969133e-04
 7.33871162e-02 2.37205029e-02 2.67356634e-04 9.96704459e-01
 8.46492052e-01 1.29768252e-03 9.72827733e-01 3.62515450e-03
 5.73958158e-02 9.08456445e-01 2.17135251e-01 7.75398294e-05
 4.13216367e-06 9.45073366e-01 1.56843662e-03 9.99876261e-01
 9.97652709e-01 9.99223709e-01 4.28885221e-04 9.94751036e-01
 2.51132250e-03 9.99925971e-01 8.60015869e-01 4.80201542e-02
 1.62198335e-01 9.70649898e-01 7.86741853e-01 3.22324932e-01
 1.17094161e-04 4.56386805e-03 1.09035105e-01 2.81146169e-03
 7.15524256e-02 1.02964342e-02 9.83894348e-01 1.45971775e-04
 9.28568304e-01 8.83775949e-03 9.95911002e-01 6.66997552e-01
 4.37468290e-04 3.74602437e-01 5.96049488e-01 2.60889530e-04
 8.31527352e-01 9.39554214e-01 9.98251677e-01 9.90375817e-01
 3.19074810e-01 1.27907544e-01 1.99640363e-01 9.98188734e-01
 1.58618659e-01 2.55377769e-01 7.62443900e-01 3.24067473e-03
 9.93387341e-01 2.41983503e-01 7.91697264e-01 8.51561725e-02
 9.82724667e-01 6.60066128e-01 7.47319341e-01 9.99995053e-01
 1.30197770e-06 9.96547639e-01 9.99991775e-01 9.57188725e-01
 5.10567427e-03 1.44447863e-01 4.21134979e-01 4.52969090e-07
 5.17565310e-02 3.13907862e-04 9.37218964e-01 9.85975504e-01
 5.06998360e-01 2.11012423e-01 8.28172326e-01 9.82565105e-01
 1.99106336e-02 6.76705386e-05 9.96470153e-01 1.89579150e-05
 9.98970509e-01 1.32760406e-03 9.91538048e-01 4.02277708e-03
 1.79470420e-01 9.99855638e-01 2.88026869e-01 9.87551808e-01
 9.99996960e-01 4.38667834e-01 9.69828844e-01 9.79040980e-01
 8.88766289e-01 1.07216537e-02 1.26343966e-01 9.98272777e-01
 2.14064121e-03 1.00000000e+00 9.74100947e-01 6.61534071e-03]  que se encuentra en:  60
Imagen de BD:  90  Imagen de consulta:  157 

Distancia:  4.3684735  al vector:  [8.99816632e-01 9.99992132e-01 5.36407113e-01 1.03253782e-01
 5.79427004e-01 9.98490095e-01 9.75921750e-01 8.68930459e-01
 9.22012150e-01 4.49544907e-01 1.15820855e-01 3.87546033e-01
 3.15637249e-06 7.99122453e-01 7.28229284e-02 7.63832510e-01
 9.99626338e-01 9.99999523e-01 1.18303299e-03 8.42934012e-01
 9.68176305e-01 3.46213579e-04 6.61697984e-01 3.05106342e-02
 4.23807770e-01 3.28491330e-02 2.17127800e-03 2.03900039e-02
 6.66430116e-01 3.22944522e-01 2.27921307e-02 6.32048547e-01
 9.96753693e-01 7.05995717e-06 4.19017673e-02 2.75357991e-01
 8.90103221e-01 9.92387056e-01 8.18544213e-05 9.97362018e-01
 9.99947608e-01 9.60922122e-01 9.48593378e-01 9.99967754e-01
 6.47276640e-03 5.98788261e-04 4.15600955e-01 5.58517396e-01
 9.99989867e-01 7.08522141e-01 1.89020932e-02 9.18849587e-01
 3.46989334e-01 2.96639889e-01 8.82758915e-01 1.01861835e-01
 3.91249955e-02 1.40340567e-01 7.71146417e-02 2.87851691e-03
 1.24873996e-01 4.10675853e-01 8.70907307e-03 4.03118134e-03
 9.99359846e-01 9.02256072e-01 2.31166720e-01 7.12194145e-01
 9.99999702e-01 2.01587975e-02 9.91984963e-01 8.17850888e-01
 4.43278283e-01 3.45580995e-01 4.22863752e-01 6.87484324e-01
 9.99980450e-01 5.15673399e-01 7.62317061e-01 2.54030824e-02
 9.78609920e-03 9.69579995e-01 8.73679519e-01 1.21871531e-02
 8.44597816e-04 7.72774518e-02 4.89339292e-01 8.07595372e-01
 6.70752704e-01 2.31476486e-01 9.99129176e-01 3.52543294e-02
 8.88452888e-01 8.91570270e-01 5.45141935e-01 9.68025088e-01
 1.70001388e-03 1.72371688e-06 9.67986703e-01 5.69749177e-02
 6.53203726e-02 5.84282160e-01 1.37195766e-01 9.88923073e-01
 5.81444204e-02 6.16313219e-01 2.37405300e-04 9.98802423e-01
 6.64705575e-01 9.99157906e-01 9.99559999e-01 4.47157472e-01
 7.60622025e-02 4.45937157e-01 9.77582812e-01 3.21628451e-02
 5.77295184e-01 3.18021178e-02 3.23385656e-01 5.56031466e-07
 9.79548156e-01 9.52386975e-01 4.28642780e-01 3.23456824e-02
 1.31622937e-05 1.20421648e-02 1.19006127e-01 9.44476008e-01]  en la posicion:  163  desde el vector:  [4.1183621e-02 1.0000000e+00 1.4466763e-01 5.3639400e-01 8.7731993e-01
 6.4791352e-02 1.0023448e-01 4.8936921e-01 9.9926192e-01 1.2381792e-02
 4.2764843e-03 7.5421691e-02 9.6263735e-09 4.6919376e-01 5.7420713e-01
 9.9165505e-01 9.0150321e-01 1.0000000e+00 2.7316809e-04 6.7157298e-01
 9.9866271e-01 1.8362095e-09 9.9956369e-01 4.6780231e-05 5.8907658e-02
 2.7273425e-05 5.0091726e-06 1.7740935e-02 2.1817964e-01 8.7590331e-01
 9.3902552e-01 9.7113919e-01 9.9995375e-01 2.2365102e-05 9.9454278e-01
 3.5165817e-02 4.6583414e-03 9.9965495e-01 5.3616073e-08 9.9999392e-01
 9.9994236e-01 9.9628794e-01 8.4090281e-01 1.0000000e+00 2.8929114e-04
 1.2695432e-02 9.8255479e-01 1.0702610e-03 9.9999726e-01 3.0846202e-01
 2.2103283e-01 4.5527071e-02 9.3084574e-04 7.6359558e-01 9.9979246e-01
 2.4169981e-03 2.1607280e-03 8.7275547e-01 9.8669136e-01 8.0377002e-05
 2.4734944e-02 3.2713187e-01 9.2154175e-01 4.1513604e-01 5.8957338e-01
 9.4380653e-01 1.3603052e-01 5.9222698e-01 1.0000000e+00 1.4304519e-03
 5.3563809e-01 8.8851511e-01 9.9735492e-01 9.8806155e-01 7.5118244e-01
 4.5231283e-03 9.9976528e-01 9.6573484e-01 1.4846182e-01 4.0020469e-01
 3.5974699e-01 8.3678293e-01 9.4340324e-01 2.6264489e-03 1.2437475e-01
 1.2146741e-02 9.2479503e-01 9.3825132e-01 1.6866854e-01 5.2823555e-01
 9.9984163e-01 6.1077261e-01 7.7864701e-01 9.9740565e-01 9.7971976e-01
 3.9226785e-01 1.9344687e-04 6.0050475e-08 9.9266231e-01 2.8329885e-01
 9.3664587e-02 5.6183636e-03 9.9942291e-01 9.1580784e-01 1.4558733e-03
 3.3083469e-01 4.9042717e-01 4.9131516e-01 9.6326172e-02 6.1173117e-01
 9.9995553e-01 5.4256797e-01 7.0306748e-02 8.9100587e-01 9.6611142e-01
 5.5942535e-03 9.9994767e-01 4.9176991e-02 3.0692816e-03 2.0049934e-07
 9.9921161e-01 7.2259331e-01 8.4165889e-01 6.5071046e-01 4.3365359e-04
 5.1142704e-01 1.5767515e-03 8.9230120e-01]  que se encuentra en:  61
Imagen de BD:  63  Imagen de consulta:  158 

Distancia:  3.9882503  al vector:  [9.8072529e-01 9.9999803e-01 3.9723909e-01 9.3369496e-01 8.3871245e-02
 9.4920534e-01 2.3712936e-01 1.9886303e-01 9.7242916e-01 9.8680723e-01
 1.2373442e-05 2.2737512e-01 2.1066877e-01 2.6710960e-01 8.3580518e-01
 1.1874489e-04 5.9916824e-01 9.9951661e-01 9.6842813e-01 3.3481956e-01
 2.3386860e-01 4.6679378e-04 9.9953085e-01 7.6735210e-01 5.3026974e-03
 1.0098836e-01 1.2646127e-01 4.4481498e-05 1.3716435e-01 2.7980924e-02
 8.2749128e-04 8.7682700e-01 9.9980152e-01 9.0694296e-01 3.3222109e-01
 9.9998689e-01 6.0959160e-03 9.9961436e-01 8.7987459e-01 9.5006472e-01
 9.9993277e-01 5.1057273e-01 5.3964096e-01 9.9936044e-01 5.5948306e-07
 3.2541776e-01 8.5087657e-01 7.9571903e-03 1.2662843e-01 9.9998087e-01
 3.0491710e-02 4.8556051e-01 6.5891445e-03 1.8472171e-01 9.9993765e-01
 9.6719217e-01 1.2413746e-01 9.6933579e-01 7.5264645e-01 2.2185954e-07
 4.6169960e-01 2.3195231e-01 1.8948317e-04 3.5846585e-01 9.9995929e-01
 4.7025743e-01 2.4739406e-01 9.8808849e-01 9.9999917e-01 6.3256884e-01
 9.9846739e-01 8.8741452e-02 6.3145477e-01 1.9297993e-01 1.9744155e-01
 5.6178981e-01 7.7269208e-01 2.2585458e-01 9.6349299e-01 7.0886058e-01
 7.9093552e-01 4.6688318e-03 2.5024712e-03 8.3603256e-05 9.2276615e-01
 6.4100170e-01 3.5187528e-01 9.4589305e-01 9.9998140e-01 1.3314667e-01
 9.9813008e-01 9.9909443e-01 2.8349668e-02 7.2810483e-01 1.7819554e-02
 4.8798752e-01 3.8281083e-04 6.0742550e-05 8.5157967e-01 6.0810238e-01
 5.3743994e-01 8.6806327e-02 9.7722614e-01 9.8707819e-01 3.4308761e-02
 9.9990094e-01 1.2832880e-04 4.0740523e-01 7.2132242e-01 2.6661399e-01
 9.9999899e-01 9.0238059e-01 8.5842866e-01 5.8965057e-01 8.9293027e-01
 9.1981053e-01 9.9941361e-01 5.6859642e-02 4.7416514e-01 1.2819913e-01
 9.9984074e-01 5.9618092e-01 1.5240708e-01 4.8884451e-03 9.9754453e-01
 8.3729881e-01 8.8704503e-01 2.6764601e-02]  en la posicion:  33  desde el vector:  [9.93402183e-01 9.99988198e-01 2.86709517e-01 9.96647298e-01
 6.16555154e-01 9.98046756e-01 1.11660361e-02 1.30676240e-01
 9.80121553e-01 9.28447008e-01 7.43855126e-05 2.68885791e-01
 2.48306751e-01 6.53505325e-03 9.88391876e-01 7.42468238e-03
 5.85539579e-01 9.99944210e-01 9.97243524e-01 3.84963423e-01
 5.32460213e-03 8.32833539e-05 9.83800292e-01 1.01551712e-01
 2.44683594e-01 3.59061360e-03 2.09237039e-02 6.77998960e-02
 4.92514968e-02 4.43369150e-04 1.15091503e-02 5.14896691e-01
 9.98872161e-01 9.92361665e-01 8.25396538e-01 9.97365534e-01
 8.65638285e-05 9.76445913e-01 1.15922987e-02 9.67366517e-01
 9.99988496e-01 9.35494184e-01 9.74821627e-01 9.99589562e-01
 1.60468562e-06 8.14478159e-01 6.98166370e-01 1.16257519e-01
 1.10813677e-02 9.98849988e-01 6.16162002e-01 2.34588087e-02
 4.85244393e-03 2.23208100e-01 9.88090992e-01 3.39932144e-02
 1.57526612e-01 9.48976159e-01 8.07908773e-02 1.06894504e-06
 2.09608197e-01 2.13513792e-01 2.56931007e-01 3.23929489e-02
 9.90359306e-01 6.98101819e-01 6.40236914e-01 9.48047757e-01
 9.99999225e-01 1.03074163e-01 9.99198020e-01 7.77480304e-02
 1.16836995e-01 8.44975829e-01 2.02726424e-02 9.86328721e-02
 9.87261534e-01 4.22365904e-01 6.50162339e-01 4.24196184e-01
 2.64533818e-01 1.46649927e-01 5.69395602e-01 8.82911086e-02
 3.41424763e-01 7.12697208e-02 9.01323736e-01 9.99684811e-01
 9.94136631e-01 1.62950158e-02 9.99901533e-01 7.84805417e-01
 9.73028660e-01 1.16932392e-03 6.96657896e-01 2.53812045e-01
 1.23479068e-02 1.89602375e-04 8.43845904e-02 3.29892009e-01
 2.67980248e-01 2.62980223e-01 9.84557688e-01 9.99295354e-01
 4.73551065e-01 9.95787263e-01 4.86165285e-03 2.18519330e-01
 6.15331531e-02 8.35853517e-01 9.99999225e-01 9.16132331e-03
 2.62207806e-01 9.16637003e-01 9.92602348e-01 9.99397159e-01
 9.99423742e-01 6.11184835e-01 1.50353909e-02 5.86634874e-03
 9.56916809e-01 7.71619380e-02 3.45933139e-02 9.58635569e-01
 6.19544387e-02 8.31326306e-01 9.97471452e-01 1.68042839e-01]  que se encuentra en:  62
Imagen de BD:  129  Imagen de consulta:  158 

Distancia:  4.1570373  al vector:  [8.31883967e-01 1.31253719e-01 9.96282995e-02 9.99924898e-01
 9.97981548e-01 6.64156556e-01 9.12544847e-01 7.48753548e-04
 3.38566601e-02 9.02542591e-01 9.64911938e-01 5.77101707e-01
 9.99990940e-01 5.52174330e-01 6.13579035e-01 2.77744979e-01
 2.96026468e-04 4.44033980e-01 9.11690235e-01 8.27783108e-01
 3.29005718e-02 3.19127083e-01 9.38889384e-02 1.17504209e-01
 1.33177638e-03 1.79219604e-01 7.20059872e-03 8.35087895e-03
 9.84159350e-01 3.58924866e-02 4.27827030e-01 2.76563704e-01
 8.71813297e-01 9.93619084e-01 7.86521018e-01 6.97599173e-01
 6.54488802e-04 9.94722128e-01 9.99810338e-01 6.95199907e-01
 5.84229648e-01 9.26248789e-01 8.76863837e-01 9.21610177e-01
 1.05861127e-02 9.94380951e-01 3.43772292e-01 9.69505072e-01
 1.71720982e-03 8.84651065e-01 9.50620770e-01 5.71880102e-01
 1.87858701e-01 9.78163540e-01 9.90286887e-01 4.21079993e-03
 9.69512343e-01 8.91312122e-01 6.72017336e-01 2.84008086e-02
 9.31148410e-01 2.83435225e-01 3.10995102e-01 1.83212340e-01
 4.99611020e-01 7.50647724e-01 3.90843272e-01 9.06686783e-02
 1.60885096e-01 9.77447748e-01 7.77712107e-01 4.93917018e-01
 2.06990689e-01 9.61795211e-01 1.20783538e-01 9.06236589e-01
 7.78742674e-07 2.72487104e-02 1.19175434e-01 1.20675981e-01
 3.22036356e-01 1.99463695e-01 2.90614367e-03 2.35155225e-03
 2.17843503e-01 2.11600333e-01 9.98374224e-01 9.97816205e-01
 1.73214942e-01 9.99868393e-01 9.62696493e-01 9.98851418e-01
 5.97783267e-01 4.09227163e-01 5.80241144e-01 3.74197960e-04
 7.04467297e-04 7.86645412e-01 2.83657163e-01 1.34868026e-02
 6.14117920e-01 9.96402323e-01 4.92635071e-02 6.47303760e-01
 8.19357514e-01 3.82996678e-01 3.52799058e-01 6.84469938e-04
 3.44984233e-02 9.42001343e-01 6.86371326e-02 3.59500706e-01
 1.62490875e-01 3.54219913e-01 9.66942787e-01 9.80767012e-01
 9.77979183e-01 4.85506028e-01 7.21800148e-01 9.99589682e-01
 9.98500466e-01 9.31216836e-01 4.61623073e-03 9.22599673e-01
 9.58767891e-01 9.99935925e-01 9.96496379e-01 2.76416540e-04]  en la posicion:  79  desde el vector:  [9.3679172e-01 8.1910241e-01 4.8234230e-01 9.9991536e-01 1.7783281e-01
 9.9481404e-01 2.7582240e-01 2.7507544e-04 2.0259619e-04 6.1970562e-01
 9.4607908e-01 6.6017830e-01 9.9926543e-01 8.1724763e-01 2.6333502e-01
 4.5358360e-01 7.5769487e-05 3.7292072e-01 9.7829604e-01 9.6756464e-01
 4.9313605e-03 1.5868798e-01 4.3833256e-04 2.0073056e-03 4.4681132e-03
 6.8101960e-01 3.8042665e-04 1.1674166e-02 1.7838460e-01 2.6674867e-03
 6.5264076e-02 3.2282472e-02 5.7375646e-01 9.9854249e-01 9.9206603e-01
 8.3806276e-01 2.5451951e-05 9.9633944e-01 9.9359083e-01 5.7565087e-01
 9.2078805e-01 8.9958864e-01 5.8265567e-02 8.3001906e-01 9.8575056e-03
 9.9869061e-01 4.4538984e-01 9.9932015e-01 7.7426434e-04 9.2287844e-01
 9.8558474e-01 9.6831924e-01 4.5201093e-02 2.8038284e-01 9.8580283e-01
 3.7518144e-04 8.2271838e-01 1.1569229e-01 7.1830332e-01 5.1421821e-03
 4.0214062e-02 5.3195125e-01 6.9316983e-01 9.2646801e-01 3.1021574e-01
 2.2148502e-01 4.3051550e-01 8.4977841e-01 8.6487436e-01 9.9491823e-01
 9.9828088e-01 3.2184243e-01 2.5806993e-02 1.3103247e-02 9.5734811e-01
 8.9089000e-01 3.1042099e-04 7.0157915e-02 6.4348239e-01 1.9919813e-02
 9.6918404e-02 1.6213879e-01 2.2373137e-01 1.4954269e-02 8.9341724e-01
 9.8524618e-01 9.9755603e-01 9.9998915e-01 4.5462489e-01 9.9994433e-01
 9.9811721e-01 9.9501443e-01 1.4042729e-01 8.5389227e-02 4.3833923e-01
 6.6836151e-06 6.7434311e-03 3.9755344e-02 7.2655678e-03 7.9551327e-01
 8.2457000e-01 9.9184519e-01 8.0505013e-03 8.7939453e-01 9.8714799e-01
 8.7670285e-01 6.5843296e-01 3.0425191e-04 5.1154858e-01 3.3183879e-01
 6.3742179e-01 1.6193736e-01 3.8854098e-01 7.0698202e-01 1.5975618e-01
 9.9882686e-01 9.9396354e-01 5.1713973e-01 7.3999858e-01 9.9954200e-01
 9.9254811e-01 1.6772228e-01 9.7236609e-01 5.3653222e-01 7.7272046e-01
 9.9986714e-01 9.9798882e-01 8.7510074e-05]  que se encuentra en:  63
Imagen de BD:  170  Imagen de consulta:  159 

Distancia:  2.9347124  al vector:  [9.73886788e-01 9.74078536e-01 2.54854560e-03 9.99995589e-01
 9.31477904e-01 9.85341072e-01 3.90576839e-01 3.19658930e-08
 1.00000000e+00 1.40637159e-04 1.86967850e-03 4.38779593e-04
 8.98003578e-04 8.47680926e-01 8.32749307e-02 9.99747515e-01
 3.19984853e-02 9.97727871e-01 9.78649855e-01 1.20714307e-02
 5.89902401e-02 1.67301893e-02 1.00000000e+00 1.25882834e-01
 9.86159801e-01 1.81827247e-02 1.33531809e-01 9.72202659e-01
 8.26666951e-01 1.19342506e-02 1.00000000e+00 4.12956685e-01
 9.96383190e-01 9.98544455e-01 9.84609604e-01 6.45178225e-05
 5.54440183e-09 2.41537473e-05 1.18192226e-07 9.99460042e-01
 9.97323036e-01 9.02753830e-01 8.48000824e-01 2.81129658e-01
 8.15801329e-08 9.95151281e-01 2.02801496e-01 2.30545971e-10
 9.02980566e-04 9.98626113e-01 9.81888890e-01 1.36996269e-01
 9.00149680e-05 3.55494440e-01 3.39198709e-02 8.97783279e-01
 1.20459879e-06 7.89633572e-01 9.98958111e-01 9.42606403e-06
 4.10124123e-01 9.78618801e-01 9.99999285e-01 6.11732125e-01
 4.45768783e-05 2.80247748e-01 9.39049006e-01 4.90868509e-01
 9.99213815e-01 2.33296067e-01 9.04864848e-01 4.17536199e-02
 8.70331168e-01 2.14391947e-02 1.32199287e-01 3.56606245e-02
 9.65154350e-01 8.60922635e-02 1.45174861e-01 4.19050455e-04
 9.01131392e-01 9.99443054e-01 9.99951482e-01 9.96293008e-01
 1.82879299e-01 5.03061414e-02 9.48791027e-01 9.99997318e-01
 2.11880138e-06 2.46736407e-03 9.84493017e-01 1.90147609e-01
 6.02148684e-05 5.94034314e-01 1.22123301e-01 1.28507614e-04
 9.99921918e-01 2.76862681e-02 1.79097325e-01 1.98474258e-01
 4.52884734e-01 3.14916839e-08 1.00000000e+00 6.31083846e-01
 3.64900827e-02 1.01074023e-04 9.99998510e-01 4.67300415e-04
 9.90257502e-01 9.99377906e-01 6.02772057e-01 1.63938701e-02
 6.06336772e-01 6.50705278e-01 7.29108930e-01 9.98718441e-01
 9.99994695e-01 4.98544276e-02 9.99065220e-01 3.71353563e-05
 6.82308237e-05 6.56965613e-01 7.19353557e-03 1.00000000e+00
 8.68502259e-03 9.99952972e-01 9.77057338e-01 3.48409414e-02]  en la posicion:  54  desde el vector:  [9.34804082e-01 9.99276996e-01 2.40088195e-01 9.99988914e-01
 5.01825511e-02 9.99988198e-01 6.53236985e-01 1.17746310e-08
 9.99545336e-01 5.62619865e-02 1.76040530e-02 1.93923712e-04
 5.69029035e-05 8.57488394e-01 4.94903624e-02 9.99533772e-01
 2.22799182e-03 9.98712301e-01 9.99996364e-01 2.13101625e-01
 1.67858899e-02 5.20274043e-03 1.00000000e+00 5.80051541e-03
 8.16495061e-01 9.78198051e-02 5.27012348e-02 9.99273777e-01
 4.27241474e-01 6.34878874e-04 9.99967813e-01 6.80155337e-01
 8.78856301e-01 9.97963488e-01 9.84901249e-01 1.26430392e-03
 3.78455933e-10 3.16828489e-04 1.29754071e-07 8.71105194e-01
 9.97581482e-01 7.80749559e-01 9.99650121e-01 4.15280133e-01
 2.45385667e-08 9.99940395e-01 7.67168283e-01 1.31462161e-06
 1.71035528e-03 9.99999762e-01 9.98604774e-01 9.20241117e-01
 2.67082487e-06 7.92042136e-01 1.47249430e-01 8.31584692e-01
 3.53721521e-06 7.73293495e-01 9.54733729e-01 1.55872769e-07
 1.41860545e-01 8.92080903e-01 9.99462366e-01 1.54976249e-02
 1.32274836e-01 7.79749870e-01 2.58946717e-01 5.84250212e-01
 9.99952495e-01 2.56987393e-01 9.96606290e-01 1.37815028e-01
 7.47137070e-01 8.69503617e-03 9.65231895e-01 6.95710182e-02
 9.94725704e-01 4.61166918e-01 5.50581157e-01 2.33739614e-03
 5.37519097e-01 1.67084366e-01 6.04820907e-01 9.93815541e-01
 1.03856385e-01 3.21586728e-02 2.77511060e-01 9.99998093e-01
 2.32045727e-06 3.39872539e-02 9.99686241e-01 9.28386569e-01
 1.68689191e-02 8.72261226e-01 5.16097426e-01 8.91741138e-07
 9.88762200e-01 1.55165792e-03 4.08317745e-02 2.01149613e-01
 1.10708177e-01 3.98129232e-05 9.99989033e-01 9.82707977e-01
 2.85787225e-01 1.99443102e-03 9.92206693e-01 3.15421348e-05
 6.34769917e-01 9.93035913e-01 9.99179125e-01 7.34088898e-01
 2.38177270e-01 2.48156697e-01 8.64718437e-01 9.99999523e-01
 9.99999762e-01 8.71188343e-02 9.89219785e-01 9.64924693e-03
 2.43257819e-05 3.95919800e-01 9.22755003e-02 9.98880267e-01
 2.41795182e-03 9.99966323e-01 9.99987364e-01 4.85563159e-01]  que se encuentra en:  64
Imagen de BD:  148  Imagen de consulta:  161 

Distancia:  4.7164516  al vector:  [4.05041277e-02 9.15444798e-06 7.73451805e-01 1.77530855e-01
 2.09440500e-01 8.80807638e-04 6.44203424e-02 2.16537654e-01
 9.99665678e-01 1.50480866e-03 9.99906361e-01 1.41290545e-01
 9.99728024e-01 3.23712826e-01 9.52200174e-01 9.99969363e-01
 5.82209527e-02 3.72588634e-04 5.23158908e-03 2.27828532e-01
 2.40069091e-01 1.00000000e+00 9.99466896e-01 9.99579310e-01
 9.92864609e-01 9.15606141e-01 9.86921549e-01 7.74605751e-01
 7.83319116e-01 9.70715284e-01 9.99997139e-01 8.17669511e-01
 2.87702680e-03 9.35951173e-01 8.62130880e-01 8.98878352e-05
 9.90643620e-01 9.41221733e-05 9.99998927e-01 2.67705560e-01
 1.38188899e-01 8.43497276e-01 2.68361449e-01 7.68207519e-06
 2.26734936e-01 1.62129998e-02 3.17447424e-01 1.28435777e-05
 1.46526098e-02 4.97943163e-03 1.53889030e-01 9.94001508e-01
 9.98101950e-01 6.00273788e-01 5.83973269e-05 9.81019497e-01
 9.46488362e-05 7.40618289e-01 8.22571754e-01 3.37224305e-01
 9.80300128e-01 8.75417233e-01 9.94980454e-01 4.64192927e-01
 8.13532006e-06 9.96859968e-01 9.44260657e-01 9.40632701e-01
 3.83902534e-06 9.00501847e-01 1.30805075e-02 2.73661911e-01
 9.97655034e-01 9.78937149e-02 6.87554777e-02 9.94331479e-01
 9.38177109e-04 1.55038536e-01 6.38352334e-01 9.83223438e-01
 9.87213850e-01 9.99998510e-01 5.89034498e-01 9.89879251e-01
 8.64376664e-01 3.55268359e-01 4.26403880e-01 2.34783292e-02
 1.54119730e-03 1.55330449e-01 4.93335392e-07 8.27111661e-01
 9.94956315e-01 9.91645157e-01 5.06747484e-01 7.33938813e-02
 9.99627113e-01 9.99504924e-01 1.96580231e-01 7.48071074e-03
 5.74175239e-01 1.35821104e-03 9.99336362e-01 5.15878783e-05
 8.48532915e-01 5.30543657e-05 9.90222633e-01 1.60753429e-02
 1.06631815e-02 9.93541121e-01 2.05397384e-07 2.40415007e-01
 3.54692340e-03 2.14616686e-01 9.92310047e-03 1.36941671e-01
 1.69256330e-03 5.93884468e-01 7.13804483e-01 8.70946825e-01
 1.79672241e-03 9.50946391e-01 5.44227362e-02 9.95963037e-01
 9.81949031e-01 9.66843843e-01 6.83662295e-01 2.31614709e-03]  en la posicion:  70  desde el vector:  [3.8538185e-01 2.6034117e-03 9.2858052e-01 2.6225644e-07 5.3993052e-01
 6.6357851e-04 1.5059769e-02 9.9985385e-01 9.9977493e-01 3.0746698e-02
 6.6587120e-01 9.7821057e-01 9.9892467e-01 9.5127851e-02 4.8871273e-01
 8.5224891e-01 9.9993813e-01 4.8502591e-01 3.5451671e-05 9.9437201e-01
 3.4360299e-01 9.9989915e-01 8.1932282e-01 9.9998522e-01 9.9124980e-01
 8.7920278e-01 9.9991459e-01 9.1150135e-02 1.9164035e-01 9.9312747e-01
 4.2632675e-01 8.2490814e-01 2.6093644e-01 1.7593980e-02 5.3384840e-02
 1.2289193e-01 9.9998873e-01 1.9026160e-02 9.9987513e-01 2.0396960e-01
 9.8740762e-01 9.8291421e-01 9.9632990e-01 1.8350810e-02 9.8020041e-01
 9.4842426e-06 5.6194836e-01 7.5930357e-04 8.6386549e-01 8.5252523e-04
 9.8012823e-01 9.9660945e-01 9.9998242e-01 7.0514923e-01 3.6540627e-04
 8.6760610e-01 6.7034364e-04 8.6832476e-01 4.6817756e-01 9.6632326e-01
 9.9869281e-01 7.1009243e-01 7.6392108e-01 2.2699505e-02 5.9174120e-02
 9.8385596e-01 9.5402062e-01 9.8574018e-01 2.8614998e-03 4.7954649e-02
 5.7907403e-03 3.7013757e-01 8.9521736e-01 7.8069812e-01 5.2654862e-01
 9.8387605e-01 9.9544090e-01 2.9969749e-01 2.3899221e-01 8.9631510e-01
 6.9529665e-01 9.9998552e-01 9.6207011e-01 9.9181473e-01 3.2242897e-01
 2.0656058e-01 1.4156842e-01 7.1033835e-04 9.1385531e-01 2.7778149e-03
 2.8405768e-06 1.9301951e-02 9.9985248e-01 9.5331711e-01 9.9213415e-01
 9.9999458e-01 9.6716321e-01 7.7935308e-02 2.9840696e-01 9.5618787e-05
 2.5191188e-02 1.7812133e-01 9.8451698e-01 6.8988800e-03 8.1359631e-01
 3.3829331e-02 1.6221106e-03 9.9980497e-01 8.9684129e-04 9.6798396e-01
 2.0585060e-03 7.1327317e-01 1.0069609e-03 7.8851581e-02 4.9546346e-01
 1.7005205e-04 2.0303523e-05 9.4221848e-01 3.7984583e-01 6.9871545e-04
 6.8641335e-02 6.5513521e-01 2.2598428e-01 9.5345479e-01 7.7767503e-01
 5.9267135e-05 4.3937835e-01 2.6104748e-03]  que se encuentra en:  65
Imagen de BD:  162  Imagen de consulta:  162 

Distancia:  3.1065762  al vector:  [5.5854797e-02 9.9966520e-01 3.7859198e-01 8.4245217e-01 6.5587986e-01
 1.9907415e-02 8.2843608e-01 8.5403901e-01 9.3584311e-01 9.9809623e-01
 1.9587636e-01 2.3870766e-02 3.8705635e-01 8.6997724e-01 7.0301020e-01
 5.6030542e-02 3.0985326e-01 9.9117088e-01 8.0032021e-01 9.8688197e-01
 9.9455655e-01 1.4685777e-01 9.4412947e-01 9.8990124e-01 5.0715208e-03
 2.0993936e-01 9.7365934e-01 1.1444122e-02 9.6376073e-01 9.7854400e-01
 7.4987859e-02 5.6203902e-03 9.9915040e-01 3.2757151e-01 4.4945183e-01
 9.4810617e-01 3.7315786e-03 9.6314609e-01 7.4283892e-01 3.5917985e-01
 9.9753428e-01 3.1345952e-01 4.7976196e-02 9.9141240e-01 2.1131944e-05
 1.6262150e-01 3.9911461e-01 9.9788308e-03 4.9068484e-01 9.8339581e-01
 4.7583640e-02 5.0182712e-01 8.4394515e-03 2.6670259e-01 9.9361259e-01
 9.6313083e-01 1.1442658e-01 4.2151004e-02 1.8905205e-01 5.4427981e-04
 4.8074484e-01 7.3677284e-01 6.6359055e-01 3.2606924e-01 9.9894428e-01
 7.7558899e-01 9.7866750e-01 1.9443843e-01 9.9897993e-01 9.5227289e-01
 1.4160544e-02 9.9586046e-01 9.8536003e-01 2.9234588e-03 3.3420837e-01
 9.7708035e-01 7.5252128e-01 1.4435321e-02 7.1353269e-01 5.9590179e-01
 7.0545238e-01 3.6986172e-03 2.0495921e-02 7.9776585e-02 1.0935086e-01
 6.5464646e-02 2.2582412e-03 9.9273342e-01 9.7833574e-01 2.2047102e-02
 9.9812043e-01 9.8951638e-01 5.9104860e-01 9.9519879e-01 5.1563776e-01
 7.5920635e-01 7.3974133e-03 1.5308559e-03 9.9898124e-01 9.9965346e-01
 1.6713142e-03 7.5522810e-02 9.8330867e-01 1.1666602e-01 5.1166379e-01
 9.8416847e-01 2.5145519e-01 7.7177835e-01 7.2015774e-01 9.7745287e-01
 9.9707508e-01 6.9523585e-01 6.9976950e-01 5.1674080e-01 5.9383512e-03
 9.4160891e-01 9.9365437e-01 7.8980291e-01 9.5006132e-01 2.4946421e-01
 9.9485725e-01 5.6000882e-01 8.5122120e-01 9.3112874e-01 1.4440116e-01
 1.3515633e-01 9.3366408e-01 9.8371142e-01]  en la posicion:  160  desde el vector:  [5.67552149e-02 9.99826312e-01 7.21157789e-02 8.16251993e-01
 7.15645194e-01 2.46950984e-03 8.21285844e-01 9.63155389e-01
 9.87914085e-01 9.35257077e-01 5.21364808e-03 6.63295984e-02
 8.69324803e-03 9.95283842e-01 9.68610942e-02 1.81715488e-02
 8.46494436e-01 9.99943018e-01 1.72483921e-02 7.26691961e-01
 9.46133494e-01 1.44550204e-03 9.92403388e-01 5.82337379e-01
 4.95094061e-03 1.52764499e-01 7.64059007e-01 5.04511595e-03
 6.21524513e-01 9.66668010e-01 3.19988132e-02 2.55913377e-01
 9.99224246e-01 1.75434798e-01 4.40716892e-01 9.50883389e-01
 2.01203316e-01 9.92616892e-01 8.90027940e-01 9.57992554e-01
 9.99511421e-01 6.72486663e-01 9.64129269e-02 9.99361932e-01
 6.18994236e-04 8.07833672e-03 1.85225189e-01 2.56921053e-02
 8.68798137e-01 7.76751339e-01 9.96380150e-02 9.79740381e-01
 4.18933332e-02 4.44722176e-01 9.99504447e-01 8.13224316e-01
 4.73967195e-03 3.65472615e-01 3.29295397e-02 3.36647034e-04
 8.85317326e-02 6.05776429e-01 1.39294207e-01 2.32112199e-01
 9.82335687e-01 6.26816273e-01 8.61349821e-01 8.65658641e-01
 9.99246895e-01 9.94336426e-01 8.66436362e-02 9.59073186e-01
 9.30588961e-01 1.01992011e-01 4.95981276e-01 8.25735331e-01
 9.79238987e-01 8.40983391e-01 3.18516970e-01 6.60008788e-01
 6.18294179e-01 8.83230865e-02 3.63801718e-02 7.02074170e-03
 4.20990288e-01 9.85140502e-02 3.16595227e-01 9.20724630e-01
 9.89216566e-01 3.13481748e-01 9.80502725e-01 9.77137327e-01
 8.80970359e-01 9.92767096e-01 1.04103088e-02 6.98345065e-01
 1.55931115e-02 6.36894692e-05 9.87465799e-01 6.43978298e-01
 2.86677629e-01 1.31625533e-02 9.98876095e-01 3.44919860e-02
 7.47785449e-01 9.82457459e-01 2.14026570e-02 9.09560919e-01
 6.12374127e-01 4.50844407e-01 9.99431789e-01 9.75350320e-01
 8.64471912e-01 4.32353914e-01 2.76132226e-02 1.93709821e-01
 9.80827332e-01 9.83573198e-02 6.00707412e-01 1.28182769e-02
 9.97760534e-01 7.32372403e-01 9.17713642e-01 1.38779581e-01
 6.01084113e-01 6.06329143e-02 9.86775160e-02 5.08965254e-01]  que se encuentra en:  66
Imagen de BD:  60  Imagen de consulta:  11 

Distancia:  3.392328  al vector:  [9.9854439e-01 1.0975644e-08 9.8964655e-01 8.5461055e-05 3.6662519e-03
 1.6007666e-06 7.3274970e-04 9.9999976e-01 1.2437144e-01 1.3683251e-01
 1.2351161e-01 9.9622297e-01 9.9999940e-01 2.6423633e-03 8.9936078e-01
 1.1148155e-03 9.9990499e-01 5.5589009e-08 2.1293759e-04 9.9578643e-01
 8.6661875e-03 9.9997163e-01 8.6826742e-02 9.9997222e-01 9.9839461e-01
 5.7485813e-01 9.9993426e-01 8.3929300e-04 2.3848274e-01 9.9999714e-01
 9.9460167e-01 3.2967240e-02 3.7947658e-01 9.9921143e-01 8.6759299e-01
 9.9979579e-01 9.9999768e-01 7.1445107e-04 9.9993068e-01 5.6442130e-01
 9.6149231e-09 9.5381570e-01 8.9914382e-01 2.2573568e-06 1.0000000e+00
 2.3472309e-04 2.8578505e-01 2.0873690e-01 2.5308132e-04 5.1531327e-07
 1.7321467e-01 9.6943271e-01 9.9989390e-01 1.3986295e-01 1.3211370e-04
 9.9442041e-01 9.9999899e-01 1.0282403e-01 8.3041191e-04 1.0000000e+00
 9.9719584e-01 2.3655087e-02 9.9962413e-01 9.7119087e-01 1.5291572e-04
 9.3515754e-01 5.6483370e-01 9.9896193e-03 4.5373199e-08 3.2549495e-01
 1.9265431e-06 4.6023375e-01 2.4959445e-04 9.6142620e-02 7.0116365e-01
 9.6395457e-01 2.8397155e-06 4.3919751e-01 4.4705132e-01 9.1786522e-01
 7.2782385e-01 7.9114270e-01 9.5697570e-01 9.9969673e-01 9.3876612e-01
 3.4877604e-01 9.2706120e-01 1.0020557e-05 9.9829233e-01 4.4523433e-01
 9.2586975e-05 6.6229701e-04 9.9985254e-01 1.6114116e-04 1.9489002e-01
 9.9992847e-01 9.9990338e-01 1.0000000e+00 1.9751787e-03 3.3279350e-01
 1.8131652e-01 1.5870944e-01 9.8905408e-01 2.1284819e-04 2.3040798e-01
 9.9800909e-01 9.9996346e-01 9.9930674e-01 8.2484013e-01 6.1251414e-01
 7.2835428e-06 1.1381373e-01 3.2708287e-02 9.9861729e-01 7.8484011e-01
 2.3245811e-04 1.6161799e-04 7.4225128e-01 9.5258296e-02 2.4326316e-01
 4.7922134e-04 7.5859165e-01 9.7419989e-01 9.9955571e-01 1.0000000e+00
 1.3890862e-04 3.3669869e-05 2.5506347e-02]  en la posicion:  123  desde el vector:  [9.7485352e-01 1.0204586e-04 9.6390831e-01 4.6470046e-05 2.7526170e-02
 1.8700659e-03 1.8274784e-03 9.9991918e-01 9.2797005e-01 7.6066852e-03
 9.1244829e-01 9.8792845e-01 9.9999720e-01 2.2615731e-02 6.4711523e-01
 4.9558181e-01 9.7413623e-01 7.6342345e-05 7.3373318e-04 9.9986315e-01
 3.7017763e-03 9.9997944e-01 1.7609507e-02 9.9779242e-01 9.9771798e-01
 5.4373664e-01 9.8855942e-01 2.4353772e-02 4.9320072e-02 9.9330419e-01
 9.8774654e-01 3.5516560e-02 1.8119824e-01 9.4739974e-01 6.5698922e-03
 6.6477436e-01 9.9942446e-01 4.3696463e-03 9.9998409e-01 7.2866708e-02
 1.1716223e-01 9.8426497e-01 9.6345454e-01 2.5513768e-04 9.9922305e-01
 6.1836839e-03 3.5701501e-01 7.2660953e-02 3.2358050e-02 2.2274256e-04
 8.2255644e-01 9.9524510e-01 9.9992228e-01 5.9783417e-01 7.6760948e-03
 8.1215143e-01 9.1341496e-01 2.4915713e-01 3.7729442e-03 9.9965239e-01
 9.9931800e-01 7.7740222e-02 9.7379661e-01 7.7481955e-02 1.2324750e-03
 9.9606562e-01 9.9724323e-01 6.5270078e-01 1.3897301e-05 2.5668949e-02
 2.5031507e-02 1.7059696e-01 1.5969747e-01 2.1039739e-01 1.7352521e-02
 9.6570063e-01 9.2460811e-03 1.3076764e-01 4.7853619e-02 9.6881342e-01
 8.6428148e-01 9.9998820e-01 4.2691743e-01 9.9644709e-01 4.4125843e-01
 9.3327159e-01 8.9925373e-01 3.2612681e-04 9.4878829e-01 6.0308570e-01
 8.6652370e-05 8.2098067e-02 9.9948281e-01 1.2467718e-01 9.9829161e-01
 9.9811947e-01 9.9513757e-01 9.9596167e-01 9.0253353e-04 9.2453957e-03
 6.2621683e-02 1.5066245e-01 9.4697899e-01 6.5881908e-03 8.0254149e-01
 2.0013213e-01 4.2199272e-01 9.9630010e-01 2.9246271e-02 9.7440183e-01
 2.5403500e-04 1.8184152e-01 1.3327599e-04 1.8057722e-01 7.9324740e-01
 7.5334907e-03 2.1809690e-05 5.0903386e-01 1.7770261e-01 7.9026002e-01
 1.4152229e-02 9.6016216e-01 1.0216454e-01 9.9585038e-01 9.9947023e-01
 4.5054406e-02 2.9193223e-02 7.0313035e-05]  que se encuentra en:  67
Imagen de BD:  27  Imagen de consulta:  162 

Distancia:  4.3584  al vector:  [9.97777939e-01 9.98695016e-01 1.62716269e-01 9.27789211e-01
 8.51189733e-01 9.44508553e-01 4.96339172e-01 1.14677561e-04
 9.10701394e-01 9.88398552e-01 1.94008052e-02 1.81991696e-01
 9.58529592e-01 6.41375601e-01 8.48583221e-01 4.02933061e-02
 1.24830008e-03 9.41642821e-01 9.99642074e-01 1.27524137e-03
 5.33172488e-03 1.48030490e-01 9.61303711e-01 2.93733537e-01
 2.45168060e-01 8.21571052e-02 1.42095476e-01 1.57532871e-01
 1.02256119e-01 1.36906207e-02 9.35210943e-01 2.26910710e-01
 9.78953242e-01 9.99781907e-01 9.36516762e-01 4.87976491e-01
 1.28133297e-02 1.54682487e-01 9.98282194e-01 5.19610286e-01
 7.29866982e-01 6.80106878e-03 8.62728953e-01 9.88024354e-01
 3.26022564e-05 9.53296304e-01 5.79521954e-01 1.68219507e-02
 1.46881648e-05 9.99986351e-01 6.65015280e-01 5.38781941e-01
 3.54871154e-03 3.59783232e-01 1.69917047e-01 4.45910275e-01
 6.09221697e-01 1.47594899e-01 1.88989520e-01 9.74595605e-06
 7.62441993e-01 9.07616377e-01 3.09764743e-01 6.23046160e-02
 8.52737427e-01 1.85962200e-01 4.57595110e-01 9.98522103e-01
 9.84238029e-01 3.66578400e-01 9.44326341e-01 9.80940998e-01
 1.43713653e-01 1.13500357e-02 1.09857023e-02 9.32395160e-01
 7.24065602e-02 7.37251699e-01 4.20134634e-01 9.55254793e-01
 4.09093738e-01 4.28568721e-02 2.15739012e-04 8.97441030e-01
 8.60197544e-01 6.18467867e-01 9.82844651e-01 9.89704609e-01
 8.93631518e-01 2.07805306e-01 8.07974339e-01 5.60691357e-01
 9.44952726e-01 6.01884723e-03 2.68667936e-04 6.22621179e-03
 9.86993074e-01 3.35708797e-01 2.58672535e-02 2.69249976e-01
 6.94437981e-01 5.12802303e-01 9.44875598e-01 9.96536553e-01
 2.78379023e-02 8.65234613e-01 8.47194552e-01 1.75605416e-02
 2.08305717e-02 2.39133716e-01 9.92584467e-01 3.89520496e-01
 8.63073111e-01 9.74267423e-02 7.38227963e-02 9.99883771e-01
 9.99528706e-01 1.37320995e-01 5.68099916e-02 7.39209712e-01
 1.79590434e-01 1.07670516e-01 7.89380074e-03 2.63780892e-01
 9.99033928e-01 9.97704327e-01 9.99934435e-01 1.03561878e-01]  en la posicion:  48  desde el vector:  [9.18422759e-01 9.99995410e-01 1.16402358e-01 9.85544086e-01
 9.84270096e-01 9.46949244e-01 2.72387564e-01 1.50077790e-01
 8.95691991e-01 9.98731017e-01 3.34710530e-05 5.18510103e-01
 6.94832206e-03 6.30409837e-01 7.67723680e-01 1.54991776e-01
 6.01350367e-02 9.99883175e-01 9.99984026e-01 9.62248087e-01
 2.91546881e-01 4.87050638e-05 9.99174356e-01 1.92244530e-01
 3.43087912e-01 2.87252665e-03 9.39235568e-01 9.98847187e-01
 1.13949746e-01 7.11819232e-02 2.66666353e-01 5.98923802e-01
 7.92005062e-02 9.85283852e-01 2.01195478e-04 5.60308754e-01
 5.16682863e-04 1.77538395e-03 4.05701995e-03 9.40118670e-01
 9.55458820e-01 6.33999109e-01 7.13499784e-01 9.99673128e-01
 2.09599150e-07 9.99720812e-01 7.05936193e-01 5.06020188e-02
 8.69700313e-03 9.99994516e-01 1.69166923e-03 1.77574158e-02
 2.49117613e-04 7.58731484e-01 8.70909989e-02 9.97497797e-01
 5.84698379e-01 9.55061316e-01 7.72526681e-01 1.17760537e-05
 8.88264954e-01 2.98736095e-02 1.56879544e-01 6.44109845e-02
 9.99774456e-01 5.81271410e-01 2.52161562e-01 6.59256756e-01
 9.99863923e-01 1.48425072e-01 9.94396389e-01 5.99279165e-01
 1.33660138e-02 3.89856517e-01 1.01056993e-02 6.49252176e-01
 9.88025665e-01 5.21112740e-01 6.66971743e-01 4.32778627e-01
 3.60096216e-01 2.42720889e-05 8.45670700e-04 9.92631674e-01
 9.87551391e-01 3.12114954e-02 1.72628760e-02 9.47461128e-01
 8.31405044e-01 1.03576313e-05 9.96356726e-01 2.73381889e-01
 9.92454052e-01 3.80355120e-03 1.47737294e-01 6.70214653e-01
 9.99561906e-01 2.03192234e-04 7.04383314e-01 4.22617793e-02
 1.07291341e-03 1.26779079e-03 9.99631882e-01 9.99974728e-01
 1.19784474e-03 2.74473488e-01 3.81693244e-03 1.71958625e-01
 2.88496494e-01 4.48376924e-01 1.00000000e+00 2.18007535e-01
 7.13967681e-02 3.45192224e-01 6.65535748e-01 9.99954760e-01
 9.99431849e-01 1.48192346e-01 1.89855695e-02 7.48305321e-01
 6.22212887e-04 3.90736282e-01 1.83351189e-01 3.13521385e-01
 8.57583940e-01 9.32790041e-01 9.99686718e-01 9.81603622e-01]  que se encuentra en:  68
Imagen de BD:  142  Imagen de consulta:  165 

Distancia:  3.794063  al vector:  [5.15968502e-02 9.06137466e-01 5.36172807e-01 9.99846339e-01
 1.81242704e-01 1.22658014e-02 2.04131812e-01 4.32521105e-04
 9.79200363e-01 3.17970544e-01 9.99862552e-01 5.37652791e-01
 4.88821387e-01 5.96279263e-01 9.84396517e-01 9.99947906e-01
 4.85578179e-03 9.97546911e-01 2.27295160e-01 8.14087510e-01
 9.99974728e-01 3.98265123e-02 9.69464421e-01 8.90675187e-03
 7.20888376e-03 1.70164704e-02 8.51035118e-04 6.27324879e-01
 6.32448971e-01 9.94222760e-01 9.99900401e-01 2.42010951e-02
 1.56802982e-01 2.90336609e-02 6.77570462e-01 1.95759385e-05
 2.30818987e-04 9.02505279e-01 8.13037157e-04 8.20899367e-01
 8.47158313e-01 2.08503485e-01 8.95891905e-01 9.83992338e-01
 1.29224360e-02 3.11013877e-01 1.20675564e-01 9.81408358e-03
 9.73791003e-01 5.98579645e-03 9.92495477e-01 9.60665643e-01
 7.60978460e-03 2.02894300e-01 4.31415439e-02 1.91062689e-03
 1.35835111e-02 2.21032679e-01 3.96014839e-01 1.32172108e-02
 2.07752317e-01 1.98052168e-01 9.99877810e-01 7.09184766e-01
 1.38190389e-03 9.40454066e-01 1.86693668e-02 4.40657198e-01
 9.68624532e-01 9.80394006e-01 6.80464506e-03 8.89979422e-01
 9.54253674e-01 1.42339945e-01 5.11686027e-01 6.59001291e-01
 3.91492844e-02 3.27104330e-01 3.56300235e-01 2.95619369e-02
 9.27471042e-01 9.80865479e-01 9.96350884e-01 9.41715956e-01
 8.90829563e-01 9.51681137e-02 9.67009425e-01 9.99997616e-01
 1.24830819e-06 9.92845476e-01 9.57603633e-01 7.88879395e-02
 3.02126855e-01 9.64108467e-01 1.36337876e-02 7.40200281e-03
 1.56563163e-01 5.24848700e-04 9.99887645e-01 9.50156331e-01
 9.50066090e-01 1.09171003e-01 9.37182128e-01 6.28556430e-01
 2.74271488e-01 2.97576189e-04 9.99784946e-01 1.39765739e-02
 9.93176699e-01 9.84674692e-03 3.86674106e-02 4.07384634e-02
 9.92000461e-01 8.44877064e-01 6.55004203e-01 4.33980882e-01
 9.97982860e-01 9.18908715e-02 2.34859109e-01 4.48516011e-03
 2.86430538e-01 2.90363789e-01 8.38786364e-01 9.99987602e-01
 2.85011530e-03 9.97668862e-01 9.56885338e-01 8.99804831e-01]  en la posicion:  74  desde el vector:  [2.37077504e-01 9.35012996e-01 3.66524994e-01 6.64628923e-01
 2.72647440e-02 7.30097294e-04 2.60441840e-01 4.52638358e-01
 9.99932647e-01 9.75780249e-01 9.17246401e-01 7.76609719e-01
 2.37882704e-01 7.76447058e-01 9.86971259e-01 9.99593437e-01
 5.70954442e-01 9.99467909e-01 3.05963874e-01 9.05458808e-01
 9.99741077e-01 8.02052021e-03 9.96812522e-01 6.45614147e-01
 1.06530488e-02 2.71703303e-02 9.53202844e-02 7.79353559e-01
 4.00126874e-01 9.95329857e-01 8.29861879e-01 1.92689896e-03
 4.76402521e-01 1.72550470e-01 7.87337124e-02 4.19676304e-03
 6.33462071e-02 5.77422619e-01 3.68437171e-02 9.66224432e-01
 9.94300961e-01 3.04928899e-01 5.98046720e-01 9.96784449e-01
 4.03523445e-04 1.39129817e-01 1.72608614e-01 4.86832857e-03
 8.65145326e-01 8.36274028e-02 3.87260854e-01 7.90543914e-01
 1.68902576e-02 4.22132432e-01 5.32530844e-02 1.09516710e-01
 9.04388362e-05 3.71565789e-01 2.94196635e-01 7.36176968e-04
 9.65804458e-01 8.98737311e-02 9.98368025e-01 2.40635723e-01
 4.13689315e-02 9.89394307e-01 6.56225085e-01 1.18410796e-01
 9.61071670e-01 9.93010402e-01 5.56600094e-03 9.95624125e-01
 9.29203391e-01 1.23333633e-02 2.84340739e-01 2.25342512e-01
 9.98516798e-01 6.19970858e-02 3.35555315e-01 4.32640254e-01
 3.17419827e-01 9.54174042e-01 9.45129395e-01 9.96840000e-01
 7.02167988e-01 7.29432404e-02 1.77067190e-01 9.99381006e-01
 9.75170732e-03 5.10096550e-03 4.29793596e-01 1.30054653e-02
 6.95295632e-01 9.32045102e-01 2.57546246e-01 8.22838783e-01
 9.15029168e-01 7.09116459e-04 9.98296380e-01 9.97915149e-01
 2.39187866e-01 1.68208182e-02 9.98114228e-01 3.74074727e-01
 8.03531706e-02 9.67580080e-03 9.85197127e-01 8.99964929e-01
 9.82483745e-01 6.81392193e-01 9.47511554e-01 7.93033898e-01
 3.73889804e-02 9.86701012e-01 1.99027985e-01 5.39703667e-01
 9.46629584e-01 1.57744884e-02 3.95939946e-02 8.97629434e-05
 3.78561378e-01 1.61238909e-01 9.39297676e-01 9.99604583e-01
 3.78778577e-03 4.72038746e-01 9.88056183e-01 9.98188436e-01]  que se encuentra en:  69
Imagen de BD:  166  Imagen de consulta:  166 

Distancia:  4.2680864  al vector:  [9.22218084e-01 9.54019308e-01 4.81969416e-02 9.97072518e-01
 4.90874052e-04 8.72144103e-01 6.94190085e-01 1.75714493e-04
 8.70961368e-01 3.84790182e-01 8.81703258e-01 7.90509582e-03
 9.79022563e-01 2.97935247e-01 5.27003646e-01 4.19712305e-01
 3.88850570e-02 3.51219893e-01 9.99956369e-01 9.88944530e-01
 2.88876891e-01 9.92285967e-01 9.96318102e-01 9.20987606e-01
 6.29003823e-01 3.85323405e-01 5.31540751e-01 1.66545212e-01
 9.90729451e-01 1.58337742e-01 9.75501060e-01 6.16028905e-03
 9.13003445e-01 9.99801874e-01 1.47719592e-01 5.09399354e-01
 2.57964366e-05 4.81345326e-01 9.16913152e-02 8.57407153e-02
 9.90068555e-01 9.52737689e-01 9.53674555e-01 5.65275550e-03
 6.18706908e-05 9.80514944e-01 4.86805826e-01 3.13779712e-03
 2.68402696e-03 9.99695837e-01 2.65998602e-01 9.71282423e-01
 2.80381441e-02 2.58754611e-01 1.42581910e-01 9.51901734e-01
 1.44168735e-03 8.43011558e-01 2.24007845e-01 1.08170636e-04
 3.71911824e-02 3.44790518e-01 8.86546016e-01 2.52390862e-01
 7.63532519e-01 7.71140635e-01 1.67853802e-01 8.37933779e-01
 8.87112558e-01 9.96292651e-01 8.73038411e-01 2.86773622e-01
 5.38146436e-01 6.86471164e-01 6.98003531e-01 9.88681316e-01
 1.81492865e-02 3.40446055e-01 2.60392249e-01 7.24198520e-02
 3.65418613e-01 2.28665113e-01 5.77234209e-01 8.09386134e-01
 8.67820382e-02 3.98239553e-01 7.02337623e-01 9.99665141e-01
 3.99485081e-01 2.08808213e-01 9.98145163e-01 9.47462440e-01
 9.92703795e-01 3.76638651e-01 1.77854002e-01 2.38701701e-03
 5.69848597e-01 6.46655083e-01 9.44787383e-01 1.79446727e-01
 4.24079746e-01 8.26945901e-03 9.94262576e-01 3.05207789e-01
 1.71686143e-01 5.98727524e-01 2.34839886e-01 1.11037463e-01
 1.70125425e-01 6.14422381e-01 9.38827515e-01 9.64022517e-01
 9.45713341e-01 8.92348945e-01 4.98284996e-02 9.99766827e-01
 9.33413684e-01 2.44494408e-01 2.94478863e-01 7.71714270e-01
 2.12863594e-01 8.79790783e-01 8.79486322e-01 9.94573236e-01
 1.33813620e-02 9.69557524e-01 9.99852002e-01 1.21820271e-02]  en la posicion:  37  desde el vector:  [9.62774396e-01 8.08606505e-01 4.91587818e-01 9.92271543e-01
 2.50722140e-01 4.26058888e-01 8.39044333e-01 2.23994255e-04
 5.92456400e-01 9.46506739e-01 7.99047291e-01 2.39466697e-01
 9.99989986e-01 4.49541122e-01 7.06668437e-01 4.38660979e-02
 1.40378535e-01 7.57445514e-01 9.99517262e-01 3.42773974e-01
 2.23994017e-01 9.40770566e-01 3.73305917e-01 9.77199316e-01
 1.87821090e-02 3.98622453e-01 8.88447642e-01 1.63502991e-02
 8.63771796e-01 2.57185400e-01 9.70635533e-01 3.23206186e-03
 9.59198356e-01 9.99599159e-01 1.89340442e-01 9.40469265e-01
 1.23012884e-04 6.67219639e-01 9.79611158e-01 1.83458328e-02
 6.66903377e-01 2.53150642e-01 8.68392944e-01 1.03539288e-01
 2.28226185e-04 7.29211807e-01 4.12267566e-01 5.45038521e-01
 4.94509935e-04 9.99204040e-01 2.66289055e-01 7.72965968e-01
 2.83560455e-02 5.84963560e-01 5.09838462e-01 7.82845378e-01
 9.36787426e-01 1.39821649e-01 4.92722124e-01 1.20824575e-03
 9.53889251e-01 4.34831381e-02 6.42989159e-01 1.52620375e-01
 9.95184183e-01 9.56854939e-01 6.98688209e-01 7.30755866e-01
 3.11317742e-01 9.23490286e-01 5.07178485e-01 6.98818803e-01
 1.01280361e-01 1.56550109e-02 3.73398453e-01 1.09488845e-01
 6.27803165e-05 5.80758333e-01 5.05323887e-01 5.26727021e-01
 8.60659838e-01 1.97439551e-01 1.53314650e-01 5.28527975e-01
 2.53428817e-02 1.87140703e-03 9.82257605e-01 9.99897718e-01
 3.59171122e-01 8.97134900e-01 9.42914128e-01 9.80222106e-01
 6.52817965e-01 1.30383581e-01 1.33102238e-01 3.72160077e-02
 4.73913044e-01 4.39544320e-01 9.33602691e-01 9.25186992e-01
 8.95115495e-01 9.78251696e-01 2.21074581e-01 9.59768176e-01
 6.59979701e-01 9.02258158e-01 5.34193873e-01 4.44195688e-01
 6.90288663e-01 9.67142940e-01 9.47458863e-01 1.57311678e-01
 8.69566560e-01 6.37546718e-01 8.40421975e-01 9.99893486e-01
 8.23773503e-01 8.36540401e-01 6.79818928e-01 9.97588158e-01
 3.73827517e-01 9.77975488e-01 5.42571843e-02 9.95850563e-01
 4.06432092e-01 9.57296193e-01 9.99842286e-01 1.18442804e-01]  que se encuentra en:  70
Imagen de BD:  132  Imagen de consulta:  170 

Distancia:  3.7489016  al vector:  [6.90981746e-03 9.99912143e-01 6.15294099e-01 2.27721575e-05
 8.60433102e-01 4.58449125e-04 5.24178803e-01 9.99850869e-01
 9.99410272e-01 9.99490678e-01 1.11793743e-04 9.35773671e-01
 3.73909235e-01 6.73082769e-01 3.19968849e-01 5.84584814e-06
 9.99958038e-01 9.99869227e-01 1.37141347e-03 5.97965062e-01
 9.98192847e-01 1.04728341e-03 9.98251081e-01 9.99987841e-01
 5.05685806e-04 1.99107826e-02 9.99749541e-01 1.96120964e-05
 5.00103831e-03 9.99492526e-01 4.03781851e-05 5.92091084e-01
 9.99943972e-01 2.54154205e-04 5.55589795e-03 9.99967694e-01
 9.99476492e-01 9.97748494e-01 9.99659181e-01 6.87971354e-01
 9.99913454e-01 1.59388185e-02 7.13039219e-01 9.99109864e-01
 8.17827167e-05 1.40238744e-05 9.51017618e-01 8.36908817e-04
 9.92389083e-01 2.27330625e-01 6.63856864e-02 7.68152297e-01
 9.67461467e-01 6.11234307e-01 9.97918904e-01 9.99675453e-01
 4.23252583e-04 4.45849061e-01 3.55184078e-01 6.80357218e-04
 9.31349635e-01 5.77320814e-01 2.28258967e-03 2.43670344e-01
 9.99997139e-01 7.31699288e-01 8.70521903e-01 9.54516649e-01
 9.99918818e-01 5.69105744e-01 5.95808029e-04 5.11592388e-01
 9.95947659e-01 4.01188821e-01 1.08120978e-01 8.90768290e-01
 9.99916196e-01 5.72538137e-01 3.99231613e-02 8.14288497e-01
 4.82408404e-02 2.63164639e-02 6.24141097e-03 6.39319420e-04
 9.94575977e-01 6.30476654e-01 1.63471699e-03 3.62131000e-03
 9.99998510e-01 1.96477771e-03 1.19666278e-01 9.93513227e-01
 9.89392877e-01 9.99256790e-01 2.32129991e-02 9.99990523e-01
 1.64317191e-02 1.81463361e-03 9.97745872e-01 1.05180174e-01
 6.86608791e-01 3.19558382e-03 9.98240471e-01 5.82724512e-02
 9.77339029e-01 9.99991536e-01 3.13474848e-05 9.99990404e-01
 8.03195536e-02 5.38063645e-02 9.99419451e-01 9.82536316e-01
 9.58685040e-01 1.22554004e-02 6.24433160e-03 2.33435631e-03
 2.55222559e-01 6.84420168e-01 9.50838447e-01 1.79201365e-04
 9.99471843e-01 4.03766811e-01 4.80207801e-03 6.47467375e-03
 9.97870207e-01 8.16742177e-05 1.81971192e-02 9.90347385e-01]  en la posicion:  71  desde el vector:  [8.42364967e-01 9.99068975e-01 2.15206832e-01 2.20068395e-02
 9.05364275e-01 2.05322921e-01 4.82637107e-01 9.68251705e-01
 7.73796618e-01 9.81188536e-01 1.83971226e-02 6.76486731e-01
 6.37258291e-01 8.42378139e-02 5.87646365e-02 5.59419394e-03
 9.94192123e-01 9.98364568e-01 4.21631932e-02 5.75691521e-01
 4.25666541e-01 6.64578974e-02 1.88892663e-01 9.64482903e-01
 3.00652981e-02 7.14070261e-01 8.37984264e-01 1.46656036e-02
 2.03729570e-02 2.67485738e-01 4.95707989e-03 8.58394742e-01
 9.94060397e-01 2.60451138e-02 9.47874963e-01 9.92931724e-01
 9.96869504e-01 9.69526947e-01 9.92812335e-01 3.06809247e-01
 9.93331730e-01 5.79316199e-01 9.99746621e-01 9.88030910e-01
 5.04689813e-02 4.70757484e-03 9.95032907e-01 3.59686941e-01
 8.65444243e-01 9.15001869e-01 7.07668066e-01 8.99049103e-01
 9.86003280e-01 1.18781745e-01 8.31371665e-01 9.34774160e-01
 3.28576773e-01 9.04528797e-01 2.05432743e-01 3.80439758e-02
 4.72355694e-01 3.81154597e-01 1.64059103e-02 5.52414656e-02
 9.99475837e-01 4.12239969e-01 1.25562280e-01 9.97573674e-01
 9.96819973e-01 7.24264860e-01 7.49933243e-01 8.09688210e-01
 7.15149820e-01 5.27520835e-01 8.74323845e-02 9.11852717e-01
 9.88118291e-01 9.70861435e-01 2.96349585e-01 3.18884015e-01
 1.81505591e-01 5.63844383e-01 2.94167101e-02 6.17690682e-02
 9.23996329e-01 6.48236573e-02 1.37255877e-01 7.19772279e-02
 9.99518156e-01 3.45602632e-01 5.22015154e-01 7.29052365e-01
 9.94320154e-01 6.73545837e-01 1.21271312e-02 9.94394779e-01
 9.90943015e-02 5.70890307e-03 7.66852140e-01 9.22214985e-03
 8.69860470e-01 8.85159373e-01 7.83951402e-01 7.70842314e-01
 7.44552255e-01 9.98353958e-01 4.55826521e-04 9.96040344e-01
 6.19224608e-02 1.68332279e-01 9.98378396e-01 8.22787166e-01
 6.81750953e-01 6.11556470e-02 7.13665366e-01 2.51851618e-01
 6.43155754e-01 8.10526371e-01 3.70380342e-01 1.30876005e-02
 9.65505123e-01 4.71228957e-02 2.72090942e-01 2.25165486e-03
 9.93812680e-01 1.27000332e-01 8.58340204e-01 3.24417233e-01]  que se encuentra en:  71
Imagen de BD:  163  Imagen de consulta:  171 

Distancia:  2.8935838  al vector:  [9.8938781e-01 9.8711574e-01 5.4505229e-02 1.5125871e-02 2.0638242e-01
 4.3390393e-03 9.8700297e-01 6.7284906e-01 9.9999905e-01 1.6885817e-02
 6.8248199e-05 3.0013263e-02 8.1322193e-03 8.9782268e-01 3.0049616e-01
 2.1097064e-04 9.9606097e-01 3.8598147e-01 2.0665973e-02 3.4196436e-02
 1.0098875e-02 8.2332826e-01 1.0000000e+00 9.9991226e-01 9.9097800e-01
 5.8418393e-02 9.9955648e-01 3.3209028e-05 2.2139719e-01 9.9544299e-01
 9.9968886e-01 1.3396227e-01 9.9999827e-01 9.8259163e-01 8.6674976e-01
 9.9919510e-01 6.1475009e-02 9.1117620e-04 1.7657876e-02 2.5889307e-01
 9.7677499e-01 7.1955580e-01 9.9218500e-01 7.0972741e-03 5.5697560e-04
 7.8681111e-04 6.8592775e-01 3.4492884e-09 2.7770281e-02 9.3117827e-01
 7.0993304e-03 9.7060996e-01 2.1115094e-02 6.9510639e-03 2.4334177e-01
 9.9998045e-01 2.7919412e-03 2.7893692e-01 9.8536074e-01 6.7290664e-04
 3.1243986e-01 3.1196025e-01 9.9877787e-01 7.8073847e-01 4.9898213e-01
 9.7331583e-01 7.9750645e-01 6.1874104e-01 9.9496269e-01 9.9477476e-01
 1.2366176e-03 7.3534858e-01 5.6223363e-02 5.8410192e-01 1.6328332e-01
 9.1451979e-01 9.7923315e-01 8.7272739e-01 1.2626469e-02 4.2482615e-03
 7.6850593e-01 9.7397184e-01 9.9612570e-01 7.8748822e-01 8.4272790e-01
 7.9870254e-02 9.2109227e-01 4.4335842e-02 9.9370664e-01 9.4117233e-05
 7.7977878e-01 4.6403074e-01 2.7560347e-01 7.4990958e-02 5.8546096e-02
 9.9499846e-01 9.9695963e-01 7.3846447e-01 1.5724295e-01 4.3782255e-01
 4.8845112e-03 5.9778351e-09 1.0000000e+00 4.2232871e-04 6.4168870e-03
 9.9982953e-01 9.7760773e-01 9.9946564e-01 9.8377854e-01 9.9956238e-01
 9.3760151e-01 4.0549085e-01 7.1184278e-01 9.5666194e-01 4.6714962e-02
 2.1140307e-02 9.3149453e-01 6.1580008e-01 3.6843687e-01 4.6279209e-05
 2.8195441e-02 9.6458018e-02 3.2061830e-01 9.9899232e-01 8.8554549e-01
 3.3339858e-04 2.9906571e-02 7.6040924e-03]  en la posicion:  84  desde el vector:  [8.83853555e-01 7.82908618e-01 5.19764423e-03 1.82420015e-03
 2.37756968e-03 7.93641806e-03 9.98216391e-01 9.42737877e-01
 9.99981165e-01 4.99427319e-04 4.24614549e-03 8.56935978e-04
 4.02390957e-03 9.94748950e-01 2.30472088e-02 1.12425450e-05
 9.99173880e-01 3.26004028e-02 1.48725510e-03 9.54991579e-02
 8.44493508e-03 8.69344771e-01 1.00000000e+00 9.99938965e-01
 9.95537043e-01 5.68594635e-02 9.99956369e-01 3.78690800e-07
 8.87504697e-01 9.99743044e-01 9.99029040e-01 5.83058298e-02
 9.99994516e-01 9.20047164e-01 9.91324902e-01 9.99660730e-01
 2.99823284e-02 1.07792020e-03 2.73754001e-02 6.89810514e-03
 8.00107002e-01 2.68314660e-01 8.83714199e-01 3.62258506e-05
 8.12220573e-03 1.03302620e-04 8.19072723e-01 2.18967200e-09
 1.15994126e-01 8.25911760e-03 1.04762316e-02 8.88782561e-01
 2.62040555e-01 2.21850395e-01 6.38342202e-01 1.00000000e+00
 6.03884459e-04 3.76911998e-01 9.76652741e-01 1.08989179e-02
 2.45683789e-02 6.24250650e-01 9.99019265e-01 7.74397016e-01
 3.32262963e-01 8.21552455e-01 8.93301129e-01 8.72615457e-01
 9.20457006e-01 9.71308708e-01 1.82215826e-05 4.60006833e-01
 3.98807406e-01 1.74457699e-01 9.28483009e-01 8.25393200e-01
 6.35214865e-01 3.55950594e-02 4.25749421e-02 7.85750151e-03
 4.46961343e-01 9.89045024e-01 9.94516253e-01 9.06600356e-02
 9.08158779e-01 2.14936465e-01 9.78021085e-01 1.98586881e-02
 9.98940229e-01 7.66536593e-03 9.80867565e-01 9.75091696e-01
 8.83221984e-01 7.05478072e-01 1.58901572e-01 9.97426033e-01
 9.47369337e-01 9.68387604e-01 1.38323009e-02 9.67888534e-01
 1.35384768e-01 4.10581524e-09 1.00000000e+00 9.01814055e-05
 9.83997583e-02 9.99958515e-01 9.87525105e-01 9.99678075e-01
 9.74341154e-01 9.85839427e-01 8.81879330e-02 9.28513646e-01
 9.62444425e-01 7.44324267e-01 3.31072360e-01 1.45590305e-02
 7.65256405e-01 8.92035604e-01 9.89708364e-01 1.45018101e-04
 2.21656263e-02 2.87796080e-01 7.40407705e-01 9.94859695e-01
 8.76996398e-01 9.91338238e-05 6.22279840e-05 5.77163696e-03]  que se encuentra en:  72
Imagen de BD:  175  Imagen de consulta:  175 

Distancia:  4.0861516  al vector:  [9.97777939e-01 9.98695016e-01 1.62716269e-01 9.27789211e-01
 8.51189733e-01 9.44508553e-01 4.96339172e-01 1.14677561e-04
 9.10701394e-01 9.88398552e-01 1.94008052e-02 1.81991696e-01
 9.58529592e-01 6.41375601e-01 8.48583221e-01 4.02933061e-02
 1.24830008e-03 9.41642821e-01 9.99642074e-01 1.27524137e-03
 5.33172488e-03 1.48030490e-01 9.61303711e-01 2.93733537e-01
 2.45168060e-01 8.21571052e-02 1.42095476e-01 1.57532871e-01
 1.02256119e-01 1.36906207e-02 9.35210943e-01 2.26910710e-01
 9.78953242e-01 9.99781907e-01 9.36516762e-01 4.87976491e-01
 1.28133297e-02 1.54682487e-01 9.98282194e-01 5.19610286e-01
 7.29866982e-01 6.80106878e-03 8.62728953e-01 9.88024354e-01
 3.26022564e-05 9.53296304e-01 5.79521954e-01 1.68219507e-02
 1.46881648e-05 9.99986351e-01 6.65015280e-01 5.38781941e-01
 3.54871154e-03 3.59783232e-01 1.69917047e-01 4.45910275e-01
 6.09221697e-01 1.47594899e-01 1.88989520e-01 9.74595605e-06
 7.62441993e-01 9.07616377e-01 3.09764743e-01 6.23046160e-02
 8.52737427e-01 1.85962200e-01 4.57595110e-01 9.98522103e-01
 9.84238029e-01 3.66578400e-01 9.44326341e-01 9.80940998e-01
 1.43713653e-01 1.13500357e-02 1.09857023e-02 9.32395160e-01
 7.24065602e-02 7.37251699e-01 4.20134634e-01 9.55254793e-01
 4.09093738e-01 4.28568721e-02 2.15739012e-04 8.97441030e-01
 8.60197544e-01 6.18467867e-01 9.82844651e-01 9.89704609e-01
 8.93631518e-01 2.07805306e-01 8.07974339e-01 5.60691357e-01
 9.44952726e-01 6.01884723e-03 2.68667936e-04 6.22621179e-03
 9.86993074e-01 3.35708797e-01 2.58672535e-02 2.69249976e-01
 6.94437981e-01 5.12802303e-01 9.44875598e-01 9.96536553e-01
 2.78379023e-02 8.65234613e-01 8.47194552e-01 1.75605416e-02
 2.08305717e-02 2.39133716e-01 9.92584467e-01 3.89520496e-01
 8.63073111e-01 9.74267423e-02 7.38227963e-02 9.99883771e-01
 9.99528706e-01 1.37320995e-01 5.68099916e-02 7.39209712e-01
 1.79590434e-01 1.07670516e-01 7.89380074e-03 2.63780892e-01
 9.99033928e-01 9.97704327e-01 9.99934435e-01 1.03561878e-01]  en la posicion:  48  desde el vector:  [9.93795455e-01 9.87562776e-01 7.83564627e-01 9.94570374e-01
 1.32788718e-02 9.97501075e-01 1.25799865e-01 2.97678113e-02
 3.80222499e-02 9.84425426e-01 4.33936417e-02 4.95490611e-01
 8.33066940e-01 4.38683927e-01 2.59963274e-02 5.48341334e-01
 2.72729993e-03 9.75917459e-01 9.99977648e-01 3.61433148e-01
 1.15016580e-01 2.64356434e-02 3.94975245e-02 2.34142125e-01
 3.03281575e-01 1.55613571e-01 2.34588265e-01 9.98814464e-01
 3.33863199e-02 1.97303295e-03 3.86846066e-03 7.88515329e-01
 4.36407804e-01 9.94827271e-01 3.50441307e-01 7.25763917e-01
 2.58720517e-02 2.81465560e-01 8.73029888e-01 9.35516953e-01
 9.84816849e-01 6.08578146e-01 1.18404955e-01 9.83150065e-01
 5.40217757e-03 9.99438524e-01 2.92100489e-01 4.06646669e-01
 6.42746687e-04 9.99699533e-01 9.17385697e-01 7.70322204e-01
 4.97352481e-02 6.26624644e-01 5.31636477e-02 1.41359180e-01
 2.91979730e-01 1.19269580e-01 9.00284767e-01 3.71188521e-02
 2.97365934e-01 9.83622551e-01 1.88040167e-01 9.48285878e-01
 9.32460546e-01 6.96774483e-01 3.85983884e-01 9.85810399e-01
 9.91845012e-01 9.57392335e-01 9.97571409e-01 9.57797348e-01
 1.23283863e-02 2.86005735e-01 4.44628626e-01 8.77379537e-01
 9.98246551e-01 9.70801592e-01 1.76496506e-02 6.23141229e-01
 1.17568761e-01 1.71392858e-02 1.73030198e-02 9.10369754e-01
 8.68556142e-01 9.82031345e-01 9.29918408e-01 9.93277431e-01
 5.90879679e-01 3.48516405e-02 9.82194543e-01 1.24180019e-01
 9.89099741e-01 2.00162530e-02 6.97390020e-01 6.59528375e-03
 9.17026520e-01 2.17005610e-03 9.37604904e-03 4.86056566e-01
 8.17177892e-01 5.59908450e-01 3.91613901e-01 9.98655677e-01
 5.66312373e-01 6.55081570e-01 4.62590158e-02 2.28042305e-02
 1.91970170e-02 4.93321419e-01 9.99956548e-01 7.72138000e-01
 8.27396631e-01 5.81975162e-01 1.26185060e-01 9.99980211e-01
 9.97722447e-01 9.28770185e-01 4.85281050e-02 6.88481569e-01
 3.74811858e-01 2.37269342e-01 6.56976700e-01 8.73532891e-02
 7.85442293e-01 9.96277630e-01 9.99972463e-01 7.39115715e-01]  que se encuentra en:  73
Imagen de BD:  142  Imagen de consulta:  0 

Distancia:  3.897695  al vector:  [5.18520415e-01 2.02307105e-03 1.05970800e-02 9.90459800e-01
 2.92758971e-01 9.86201823e-01 8.03405344e-01 7.93862700e-01
 9.85148145e-05 9.29119587e-01 9.99994040e-01 9.01036620e-01
 9.97635126e-01 1.64497793e-02 8.56601000e-01 9.99993980e-01
 1.98614597e-03 2.67501950e-01 2.23030150e-02 7.73756981e-01
 7.17208982e-02 2.34932303e-02 1.58313668e-07 7.69793987e-04
 1.59335762e-01 4.49229628e-01 1.29312939e-05 9.99684274e-01
 1.92056805e-01 2.55192995e-01 2.65508890e-04 1.37158126e-01
 2.44726543e-05 7.80504942e-03 8.20347667e-02 1.41166765e-05
 9.94490862e-01 9.97455955e-01 9.99991298e-01 8.23158324e-02
 2.09491193e-01 4.38946486e-01 1.49067730e-01 9.61323202e-01
 9.99953747e-01 9.98033762e-01 7.78421760e-02 1.00000000e+00
 9.77444172e-01 7.80190821e-05 5.80647409e-01 3.66143495e-01
 9.97516155e-01 2.22615182e-01 6.91240251e-01 1.42541085e-05
 9.97525930e-01 3.51147652e-02 8.95613909e-01 9.99866903e-01
 4.60846066e-01 4.78802055e-01 1.87516212e-03 1.70638621e-01
 4.32133675e-04 9.82332170e-01 8.87000799e-01 1.79122150e-01
 2.57947140e-05 6.52366996e-01 8.01033616e-01 4.68774140e-02
 1.22363299e-01 6.63457215e-01 8.54419768e-02 1.30548060e-01
 3.52406502e-03 5.59250653e-01 8.51814449e-01 2.04785794e-01
 1.13662273e-01 9.24939513e-01 5.76102734e-03 7.72411227e-01
 7.43441582e-01 6.22603655e-01 8.55294824e-01 8.70132506e-01
 5.50790210e-05 9.99921441e-01 5.68836033e-02 4.59354520e-02
 4.49922085e-02 2.88789272e-01 4.46934879e-01 2.13310122e-03
 1.14983320e-03 6.43993020e-01 9.67468619e-02 1.67237431e-01
 7.76881933e-01 1.00000000e+00 2.79259211e-07 9.72698927e-01
 5.07341862e-01 6.85934428e-05 2.59886801e-01 1.37719512e-03
 9.10526514e-02 2.04474032e-02 4.84520197e-03 4.76650149e-01
 5.17585874e-02 3.50380003e-01 8.93620491e-01 1.28132761e-01
 1.00857019e-03 5.63712537e-01 2.77750492e-02 9.99313533e-01
 9.89061356e-01 6.33400679e-01 3.07053924e-01 1.80488825e-03
 3.90377730e-01 9.98960316e-01 9.89186883e-01 2.27357119e-01]  en la posicion:  59  desde el vector:  [2.99307793e-01 3.01429864e-05 4.82240438e-01 6.55632198e-01
 1.25601888e-03 3.29118967e-02 8.00488114e-01 9.95811820e-01
 2.21617818e-02 9.97603297e-01 9.97190595e-01 8.38222742e-01
 9.97489452e-01 6.30166948e-01 3.59169245e-01 9.98321652e-01
 1.59499437e-01 2.52345502e-02 1.89538002e-02 7.65947700e-01
 4.28834230e-01 1.69086695e-01 1.08025175e-04 2.17065513e-02
 2.51386166e-02 7.42166579e-01 2.23002136e-02 9.90877509e-01
 1.76475048e-02 8.69010448e-01 8.30498338e-03 1.78724527e-04
 1.96486712e-04 7.66894817e-02 7.32329488e-02 1.27872825e-02
 9.99243617e-01 9.40267444e-01 9.99984741e-01 1.83461756e-01
 3.35395515e-01 3.02202523e-01 5.43371916e-01 3.02014202e-01
 9.99907434e-01 9.72463369e-01 6.88244402e-02 9.99974191e-01
 8.67046237e-01 1.26987696e-04 6.62410676e-01 9.21842217e-01
 9.91681397e-01 1.06440097e-01 1.81662470e-01 4.65810299e-04
 9.27935839e-01 1.99278593e-02 4.71223593e-02 9.99857903e-01
 9.50340033e-01 1.37422353e-01 6.79261684e-02 9.31413114e-01
 9.36233997e-03 9.38183308e-01 9.94274020e-01 2.88519681e-01
 1.14997022e-06 9.43232179e-01 7.65690804e-02 9.78197336e-01
 2.78382927e-01 1.99430287e-02 7.56711721e-01 1.37768090e-02
 6.92931831e-01 4.94253159e-01 4.69591767e-01 2.64963508e-03
 2.67892480e-02 8.88590217e-01 2.31611729e-02 9.27619934e-01
 8.76008511e-01 9.97347713e-01 1.90115839e-01 6.30455852e-01
 1.92704797e-03 9.93299127e-01 2.56717205e-04 9.16987658e-03
 2.80922949e-01 3.54645401e-01 3.88942659e-01 3.12751979e-01
 4.53906566e-01 9.35500979e-01 7.50343919e-01 4.78325188e-01
 1.46519601e-01 9.99824882e-01 1.02105922e-04 4.45582956e-01
 9.75341916e-01 8.24958086e-04 9.32032645e-01 7.32834280e-01
 9.28538740e-01 2.83348560e-03 5.75900078e-04 7.48126388e-01
 5.26412308e-01 9.99775290e-01 5.56153536e-01 1.41317248e-02
 2.83096433e-02 3.15258950e-01 2.87663758e-01 9.81652081e-01
 9.74852562e-01 3.56242388e-01 8.32222939e-01 3.30745131e-01
 9.05433118e-01 6.98348045e-01 5.04699528e-01 5.68357706e-01]  que se encuentra en:  74
Imagen de BD:  152  Imagen de consulta:  183 

Distancia:  4.081529  al vector:  [3.28294635e-01 9.99994934e-01 3.22444290e-01 9.99998689e-01
 2.75410384e-01 9.99926805e-01 9.98314261e-01 2.96407938e-03
 9.19328886e-05 4.59779769e-01 1.65997744e-01 7.28754401e-01
 1.07120931e-01 6.36652410e-02 1.69531584e-01 1.99910998e-03
 6.53496385e-03 9.99694526e-01 9.87642050e-01 9.88671184e-01
 1.44022763e-01 1.09047207e-04 2.29299068e-04 4.40175143e-07
 7.03155994e-04 7.12600946e-01 4.60041847e-05 8.49304197e-05
 9.61983442e-01 4.69237566e-04 3.36649864e-05 3.65340710e-02
 9.97149706e-01 6.86496377e-01 6.23722792e-01 9.89761114e-01
 1.95911527e-03 1.00000000e+00 1.75706983e-01 9.69332457e-01
 9.99988914e-01 9.83942151e-01 2.46098012e-01 9.99988735e-01
 6.41095638e-03 9.79976773e-01 1.35120928e-01 9.99970734e-01
 9.35961485e-01 9.85239267e-01 2.37226486e-03 1.35433614e-01
 1.87087059e-03 1.30660832e-02 9.99999344e-01 5.57661057e-04
 9.98331368e-01 3.22675705e-03 7.93546319e-01 4.01148200e-03
 1.29622221e-02 7.12286949e-01 3.00298234e-05 1.52895153e-02
 9.99984622e-01 5.97074509e-01 4.80113655e-01 3.84438127e-01
 9.99999404e-01 6.58659577e-01 9.98810172e-01 8.82009625e-01
 7.06210732e-01 3.78309011e-01 9.56522703e-01 8.00931454e-03
 1.01598144e-01 3.00267935e-02 2.63314247e-01 1.99424028e-02
 2.93479621e-01 4.24766541e-03 1.21393293e-01 2.66342937e-07
 1.56186223e-02 3.78257036e-03 9.74183440e-01 9.99977052e-01
 9.94889021e-01 9.99916852e-01 9.99997497e-01 9.99977529e-01
 1.06083586e-04 7.11609602e-01 9.86270368e-01 1.14741451e-04
 2.10911111e-07 1.20263430e-05 5.77617884e-01 9.92804408e-01
 2.86464393e-02 9.99826610e-01 1.40279531e-04 9.88108397e-01
 9.07956481e-01 9.85231519e-01 3.12954187e-04 9.92238522e-04
 9.99731302e-01 3.72336179e-01 9.99956191e-01 4.90854383e-01
 9.53772604e-01 6.88376129e-01 5.60885727e-01 9.95292664e-01
 9.99648333e-01 8.48787844e-01 2.98617423e-01 8.35879207e-01
 1.00000000e+00 9.87427831e-01 9.37581003e-01 6.76274300e-04
 1.02610350e-01 9.97895002e-01 6.88517511e-01 8.68372663e-05]  en la posicion:  109  desde el vector:  [5.1512587e-01 9.9999481e-01 1.3921171e-02 9.9872017e-01 8.7638962e-01
 9.4966739e-01 2.8182700e-01 3.6551777e-01 4.3751091e-02 9.9945211e-01
 6.5260619e-02 7.4468285e-02 8.5711777e-03 9.7109842e-01 4.8816025e-02
 9.1210306e-03 2.9758459e-01 9.9994552e-01 7.0756614e-01 8.7661266e-02
 9.9785739e-01 2.8400012e-07 2.0001537e-01 1.3670325e-03 2.1085143e-04
 7.1403384e-03 1.2033075e-02 1.7019689e-02 5.0817358e-01 1.6492397e-01
 3.2927662e-02 6.2280893e-03 9.9907619e-01 4.9379081e-02 8.0744445e-01
 9.4728094e-01 3.6625564e-03 9.9842298e-01 2.4169683e-04 9.8918635e-01
 9.1869432e-01 2.1421003e-01 8.6733037e-01 9.9999982e-01 4.9176812e-04
 6.9998348e-01 4.4438928e-02 5.9947288e-01 8.8610786e-01 9.8810375e-01
 9.8161662e-01 4.1910231e-01 5.1358733e-05 5.2492583e-01 9.9755788e-01
 1.6026616e-02 9.9171567e-01 4.3941140e-03 7.3255640e-01 3.2691658e-03
 1.1811516e-01 1.3424051e-01 1.6411027e-01 7.5279564e-02 9.9778134e-01
 5.2732402e-01 2.5751835e-01 2.0259172e-01 9.9999970e-01 9.8085034e-01
 8.4551382e-01 9.9331003e-01 2.7262330e-01 6.3708425e-04 2.4401546e-01
 7.4453396e-01 6.0107809e-01 5.0517988e-01 3.0589452e-01 2.0836055e-02
 7.9164588e-01 2.0658970e-04 2.2208482e-02 1.6802251e-03 8.6131310e-01
 2.2726008e-01 9.2745787e-01 9.6907735e-01 9.4690138e-01 9.7032869e-01
 9.9998331e-01 6.0197389e-01 2.2405493e-01 3.8761181e-01 3.0601886e-01
 2.0481348e-02 6.2370300e-04 9.2096099e-07 9.8496151e-01 4.6011356e-01
 4.0688813e-02 8.4903848e-01 6.1311483e-02 9.9992025e-01 6.2540472e-03
 9.8125720e-01 5.5564463e-02 5.5765671e-01 5.4309094e-01 6.7205346e-01
 9.9994004e-01 6.0530603e-03 9.9214065e-01 9.7628748e-01 5.4952705e-01
 9.8373204e-01 9.9999255e-01 8.8272208e-01 8.8914084e-01 2.8818005e-01
 9.9317878e-01 7.6612806e-01 6.4362371e-01 5.3941160e-02 3.0893809e-01
 7.4666893e-01 4.0386516e-01 9.9470174e-01]  que se encuentra en:  75
Imagen de BD:  198  Imagen de consulta:  185 

Distancia:  3.9640257  al vector:  [1.4645457e-03 6.7196459e-02 7.9044503e-01 8.7058210e-01 5.4919720e-03
 3.0458250e-05 4.7305226e-04 3.2232195e-01 9.9855399e-01 8.7094831e-01
 9.9010599e-01 9.4580847e-01 9.4599098e-01 4.8685637e-01 9.9595857e-01
 9.2822707e-01 2.2433341e-02 9.0417343e-01 1.3054013e-02 1.3781327e-01
 9.8059571e-01 8.5305959e-01 9.8448163e-01 9.9322444e-01 3.9537311e-02
 9.6424758e-01 7.9108322e-01 7.1626365e-02 1.3874829e-02 9.9932504e-01
 9.3722153e-01 4.6740210e-01 2.9155952e-01 1.1292011e-02 9.6544349e-01
 3.9013952e-02 3.1475276e-02 9.6291625e-01 9.9962789e-01 1.0956538e-01
 8.0527306e-01 9.9646908e-01 2.8459099e-01 5.5920541e-01 2.0039856e-02
 1.3043761e-02 8.7586749e-01 5.6106448e-03 7.4227411e-01 8.7274313e-03
 3.1875998e-01 7.6101100e-01 5.7265979e-01 8.2272589e-02 3.7081766e-01
 5.2707553e-02 2.2628903e-04 7.8386492e-01 1.2176788e-01 1.5305579e-03
 7.7972341e-01 5.9631181e-01 9.9868584e-01 6.6943532e-01 5.7728291e-03
 9.9307048e-01 4.7803581e-01 9.6383923e-01 3.0957282e-02 7.7103114e-01
 6.9716573e-04 4.7672820e-01 9.9936610e-01 1.5806317e-02 6.8128765e-02
 3.4972584e-01 3.7580213e-01 9.9840498e-01 9.9568111e-01 9.6506965e-01
 8.0253476e-01 9.9817336e-01 8.1745613e-01 4.0363669e-01 7.4371314e-01
 1.2459481e-01 4.2389917e-01 9.9940920e-01 1.0524094e-02 6.0900879e-01
 5.3870499e-02 9.7016788e-01 9.8595119e-01 9.9905503e-01 3.8365334e-02
 3.5057864e-01 1.7109364e-02 8.6815286e-01 9.9787152e-01 9.3357849e-01
 9.9346739e-01 9.9148899e-02 9.8851490e-01 6.5649152e-03 9.6871424e-01
 1.0474652e-02 8.9658135e-01 4.5356500e-01 1.3077113e-01 1.1594984e-01
 5.5322051e-03 4.2722923e-01 3.9368957e-02 5.1388490e-01 1.2853146e-03
 4.2695671e-02 5.0998646e-01 1.0223180e-02 1.8304288e-03 7.9916656e-02
 9.2732608e-01 2.7253032e-02 5.8615786e-01 9.9747419e-01 5.4151869e-01
 6.7524749e-01 9.1378635e-01 2.1561572e-01]  en la posicion:  156  desde el vector:  [3.1694144e-02 1.0377675e-02 2.0140800e-01 4.0732712e-02 7.3828429e-02
 6.9766045e-03 2.7224910e-01 8.8975966e-01 9.9326122e-01 8.2152599e-01
 9.9692661e-01 3.2905799e-01 9.9899065e-01 5.8016735e-01 9.0849757e-01
 9.7169161e-01 6.8653303e-01 3.7450302e-01 2.5389880e-02 2.6516330e-01
 4.3998355e-01 9.9763888e-01 2.5541896e-01 9.9356639e-01 1.0969183e-01
 7.8389418e-01 9.1612804e-01 2.8358698e-03 7.2475642e-02 9.7793651e-01
 4.0032220e-01 2.1943179e-01 1.7965463e-01 2.2854021e-01 4.4665277e-02
 2.9020700e-01 9.2326903e-01 8.8850009e-01 9.9996340e-01 7.7955937e-01
 9.9176371e-01 3.9790684e-01 7.5399327e-01 1.5606374e-01 2.2911364e-01
 1.1467338e-03 4.5842075e-01 6.0041904e-01 7.6110983e-01 1.8198460e-02
 7.4328804e-01 9.7248107e-01 9.8785102e-01 1.3848183e-01 3.8526416e-01
 1.0589439e-01 2.3565888e-03 1.2887496e-01 5.4960442e-01 3.9224064e-01
 9.4330370e-01 3.2711661e-01 9.8323578e-01 2.9866105e-01 2.6396871e-02
 9.8849922e-01 6.3469267e-01 7.7247953e-01 9.6169114e-03 9.9625409e-01
 3.2469332e-03 5.9749222e-01 9.6478128e-01 9.3202323e-02 4.3464583e-01
 9.9020779e-01 9.3162179e-02 8.9753234e-01 8.0259264e-01 3.9277697e-01
 2.2345990e-02 9.9668258e-01 5.3190267e-01 2.7625930e-01 9.4709325e-01
 6.6311574e-01 5.6322038e-01 9.7565126e-01 3.9335483e-01 7.0842099e-01
 4.1622221e-03 7.7564085e-01 5.2513385e-01 9.8687947e-01 5.7970858e-01
 9.0094101e-01 1.9166774e-01 6.0428351e-01 6.4345372e-01 8.9410162e-01
 1.9261906e-01 6.2625921e-01 6.1902297e-01 1.0868013e-03 9.8262858e-01
 4.0833971e-01 4.0067616e-01 9.5300913e-01 2.6488745e-01 2.9877108e-01
 1.4531612e-04 2.4776950e-01 1.9991055e-01 7.6149356e-01 7.0841014e-02
 1.9333631e-02 3.5284966e-02 7.8105652e-01 1.5855134e-03 3.7484956e-01
 9.9260587e-01 5.3701532e-01 8.2605451e-02 9.4836724e-01 5.4471552e-01
 3.1229708e-01 9.3799841e-01 3.5696149e-02]  que se encuentra en:  76
Imagen de BD:  57  Imagen de consulta:  188 

Distancia:  3.2913127  al vector:  [7.19971120e-01 1.00000000e+00 7.02857971e-03 3.04979503e-01
 9.97884929e-01 2.78515220e-02 9.47516203e-01 9.63904738e-01
 8.62961769e-01 9.50101852e-01 4.90350112e-06 1.83030963e-03
 1.00277226e-04 9.99832511e-01 7.58221745e-03 8.78467563e-06
 9.88263845e-01 9.99999046e-01 8.14513862e-02 2.90348530e-02
 9.25784290e-01 4.82404801e-07 9.92848516e-01 2.86947548e-01
 2.97397375e-04 6.75002635e-02 8.02852213e-01 6.75797462e-04
 9.03068125e-01 9.84911323e-01 4.59933281e-03 8.03973198e-01
 9.99998569e-01 1.59208745e-01 8.57374191e-01 9.99825478e-01
 9.51878726e-02 9.99511838e-01 1.84448451e-01 7.88814068e-01
 9.96130884e-01 6.53883994e-01 6.60365522e-02 9.99997973e-01
 1.26196037e-05 7.90416896e-02 3.33259970e-01 6.07189536e-03
 8.81028175e-01 9.99786019e-01 3.72165978e-01 9.68350708e-01
 3.27313410e-06 9.81025219e-01 9.99999046e-01 9.64299560e-01
 7.27809548e-01 4.31244701e-01 2.26039886e-02 1.72974160e-05
 1.73398852e-03 4.87296373e-01 1.33246183e-04 1.29518926e-01
 9.99985099e-01 4.55766052e-01 7.34366953e-01 8.74600530e-01
 1.00000000e+00 3.03352177e-01 6.40131354e-01 7.92222023e-01
 2.68731177e-01 4.81287241e-02 5.93182147e-01 2.00314522e-02
 9.63545024e-01 1.93541139e-01 9.60289240e-01 7.28635788e-02
 9.65710163e-01 1.98036432e-04 2.33355840e-05 6.87360764e-04
 4.32300925e-01 3.35518986e-01 2.21075743e-01 2.28050947e-01
 9.99491692e-01 2.19153225e-01 9.97024536e-01 9.98838425e-01
 2.56088316e-01 8.83730888e-01 1.82318687e-03 9.82524991e-01
 1.04390681e-02 1.58871387e-06 9.89679575e-01 1.10875785e-01
 6.35328889e-03 8.42362642e-04 9.99693394e-01 8.80260229e-01
 8.62538815e-04 9.99981880e-01 7.97915757e-02 9.75188971e-01
 9.83246565e-01 5.53196967e-02 9.99999881e-01 9.91007149e-01
 8.41089249e-01 2.35382944e-01 3.18982005e-01 2.76338577e-01
 9.99960661e-01 2.24334002e-03 9.67321992e-01 5.34594059e-04
 9.98101592e-01 9.60953712e-01 9.51965630e-01 3.32981348e-04
 9.98947501e-01 1.65485740e-02 5.60390949e-03 9.62087512e-01]  en la posicion:  12  desde el vector:  [8.49969745e-01 9.99785125e-01 6.36204779e-02 2.53550828e-01
 6.97110653e-01 2.00221837e-02 4.11623210e-01 9.27077889e-01
 9.73468065e-01 9.35284853e-01 2.75999308e-04 1.01116508e-01
 3.50216985e-01 9.85204279e-01 3.34254354e-01 8.38270353e-05
 9.85880375e-01 9.99948680e-01 9.33313370e-02 8.53954196e-01
 2.63027668e-01 1.55732036e-03 9.89695191e-01 9.02245402e-01
 2.38687098e-02 3.48684967e-01 9.61541533e-01 2.26235390e-03
 2.69397587e-01 8.52033138e-01 1.09395713e-01 6.92756295e-01
 9.99791324e-01 8.61034870e-01 4.75301802e-01 9.98101354e-01
 6.74084127e-02 9.73113537e-01 8.62035334e-01 7.22723246e-01
 9.95885968e-01 4.80909258e-01 6.33134246e-02 9.99258876e-01
 7.38888979e-04 2.01320648e-03 4.12578970e-01 1.31495893e-02
 1.86802506e-01 8.69259596e-01 9.22154188e-02 9.84166622e-01
 1.93852484e-02 4.80687410e-01 9.97407556e-01 8.91503453e-01
 2.25228071e-01 2.70074248e-01 6.08726144e-02 4.38213348e-04
 2.38794506e-01 7.46847332e-01 2.39367783e-01 1.54936284e-01
 9.94056463e-01 6.78550243e-01 8.11714470e-01 8.23290825e-01
 9.99301434e-01 9.69371676e-01 3.67193699e-01 9.52245951e-01
 7.96688497e-02 2.34758943e-01 1.34181529e-01 7.55352259e-01
 9.26180720e-01 5.97892463e-01 7.25478232e-01 3.67881775e-01
 8.60648692e-01 3.81872654e-02 1.41959488e-02 3.47158313e-02
 8.03962588e-01 3.39945018e-01 7.29544461e-01 9.03065681e-01
 9.98963356e-01 1.67421222e-01 9.92878437e-01 9.00802672e-01
 8.56779099e-01 4.81179744e-01 4.12170887e-02 9.69466448e-01
 1.75766855e-01 1.03431856e-04 7.50148535e-01 5.46101570e-01
 4.30334151e-01 1.35852098e-02 9.98003364e-01 7.72758126e-01
 8.83695245e-01 9.99745131e-01 7.72178471e-02 9.85642314e-01
 5.96667290e-01 1.35071516e-01 9.99729514e-01 8.78231406e-01
 7.66671300e-01 5.02582550e-01 1.08529329e-02 4.05829638e-01
 9.60716188e-01 2.72148252e-02 4.25775468e-01 1.57087147e-02
 9.92108285e-01 7.52156734e-01 5.77531338e-01 4.16018277e-01
 9.49195385e-01 1.05102956e-02 2.47391641e-01 4.69285846e-01]  que se encuentra en:  77
Imagen de BD:  11  Imagen de consulta:  11 

Distancia:  3.6008687  al vector:  [2.55542099e-01 9.99903679e-01 9.45710659e-01 9.99998927e-01
 3.45983058e-01 9.93956685e-01 3.36312145e-01 6.37312114e-05
 8.15478921e-01 5.88518977e-02 7.86433935e-01 6.72768235e-01
 4.10698533e-01 5.92509151e-01 5.02741039e-02 8.66480470e-01
 1.04546980e-05 9.98760223e-01 9.69864488e-01 4.08675015e-01
 5.42149842e-02 6.12342358e-03 9.14139152e-01 1.32828951e-04
 1.37753487e-02 7.11357594e-03 1.04932900e-04 4.66180742e-02
 2.25025177e-01 3.03116441e-03 9.56793189e-01 5.73241591e-01
 9.93247509e-01 9.66120124e-01 1.88309252e-02 1.22142494e-01
 2.74064132e-06 9.97661591e-01 3.81427735e-01 9.96210873e-01
 9.99931574e-01 5.06447256e-01 3.53138626e-01 9.92661476e-01
 2.55509331e-05 9.78612065e-01 2.70634830e-01 8.45163763e-02
 9.46824253e-02 9.95775819e-01 9.31900263e-01 9.03385878e-01
 2.14168429e-03 4.07433808e-02 9.98787284e-01 4.90546227e-03
 2.66047716e-02 5.60941994e-02 4.00463462e-01 7.63804655e-06
 9.10377145e-01 8.58545005e-01 8.38188708e-01 1.17991954e-01
 8.63334358e-01 3.39131713e-01 8.59692693e-01 9.42055345e-01
 9.99994874e-01 2.66558230e-01 9.97258306e-01 2.75662780e-01
 8.88715029e-01 8.04148674e-01 7.00348616e-02 9.75825667e-01
 5.07712364e-03 4.41714257e-01 5.18465102e-01 7.58800745e-01
 3.36797863e-01 8.87817681e-01 6.38383627e-01 9.61422775e-05
 6.55241728e-01 6.13845468e-01 9.99135196e-01 9.99967754e-01
 3.77829969e-02 9.97181416e-01 9.99970198e-01 9.99985933e-01
 4.68462586e-01 4.11614299e-01 7.82786012e-01 9.24278993e-07
 2.91287899e-04 5.85252047e-02 3.91662508e-01 2.07976699e-02
 2.33129263e-02 2.35110223e-02 9.93343830e-01 8.77655268e-01
 7.31094241e-01 2.14776397e-02 1.09229386e-02 2.57543034e-05
 9.36866701e-02 9.90209877e-01 9.53361094e-01 3.15781951e-01
 8.49218011e-01 5.89750707e-02 8.05026770e-01 9.84724283e-01
 9.99474406e-01 9.92348194e-01 5.74369133e-02 7.46614754e-01
 9.99703526e-01 8.30438018e-01 1.89332664e-02 9.87375855e-01
 5.82656264e-03 9.99996543e-01 8.86261344e-01 3.00165220e-06]  en la posicion:  36  desde el vector:  [7.95791745e-02 9.96711493e-01 8.66586864e-01 9.99878287e-01
 3.97471309e-01 2.80818701e-01 1.07439458e-02 3.74324620e-02
 9.32191372e-01 2.55931377e-01 5.07804155e-01 8.57750773e-02
 6.49335623e-01 6.97701097e-01 8.28084767e-01 4.93044585e-01
 1.02117658e-03 9.62634742e-01 7.29244351e-01 3.28503847e-01
 5.55133760e-01 1.54594779e-02 8.83313000e-01 3.18100750e-02
 5.97842038e-02 6.41077757e-04 2.16835737e-03 1.87315941e-02
 5.45192003e-01 4.41903889e-01 9.51046944e-01 6.90456927e-02
 9.90172386e-01 9.59416509e-01 6.43100739e-02 5.77457845e-02
 1.21505436e-04 9.80948031e-01 8.07495117e-01 8.88827205e-01
 9.98016357e-01 2.22018600e-01 6.71973467e-01 9.77684796e-01
 3.87936831e-04 6.27633691e-01 8.78688157e-01 8.12769234e-02
 1.03752822e-01 3.95818204e-01 3.19624037e-01 8.57908726e-01
 4.34681773e-03 2.31718719e-02 9.34518099e-01 2.66754627e-02
 3.29178274e-02 6.35943711e-02 8.42463851e-01 3.78072262e-04
 9.30899560e-01 9.95775700e-01 9.67626333e-01 4.08629477e-02
 2.25629330e-01 1.60391450e-01 2.65180230e-01 9.62913394e-01
 9.97823119e-01 2.66819537e-01 3.61261159e-01 1.84523255e-01
 9.37487423e-01 1.04937077e-01 5.52796245e-01 9.87544537e-01
 2.46103406e-02 7.46595681e-01 1.63999200e-03 3.00339222e-01
 2.01523364e-01 8.45151424e-01 8.51967692e-01 8.49613547e-03
 9.68143225e-01 4.67166632e-01 7.25757241e-01 9.99386430e-01
 2.09002972e-01 9.57454085e-01 9.93377924e-01 9.99557376e-01
 7.66165614e-01 7.34470367e-01 5.24534285e-02 1.17695332e-03
 1.17961764e-02 3.09465230e-02 4.19357777e-01 9.77870822e-01
 6.59246445e-02 2.81297266e-02 9.50160623e-01 1.56832069e-01
 2.96501517e-01 3.90350521e-01 6.02540731e-01 1.34381950e-02
 2.16678679e-02 3.41921091e-01 5.82565248e-01 1.08467638e-01
 9.58017826e-01 1.96985871e-01 1.66310579e-01 9.82117712e-01
 9.99720275e-01 7.38333464e-01 1.98437810e-01 9.59234715e-01
 9.84830141e-01 4.77350891e-01 8.67530584e-01 9.91005182e-01
 3.13793123e-01 9.94201005e-01 1.44753784e-01 4.06524539e-03]  que se encuentra en:  78
Imagen de BD:  131  Imagen de consulta:  190 

Distancia:  3.855229  al vector:  [2.21636891e-03 5.22147658e-08 2.55498856e-01 9.98282790e-01
 6.64915979e-01 4.59811563e-05 1.67751491e-01 9.75465298e-01
 8.76686871e-01 8.29593301e-01 9.84942734e-01 3.84703875e-02
 9.96374547e-01 9.94468153e-01 1.86306804e-01 9.35061693e-01
 5.12292981e-03 9.81647077e-07 1.59642398e-02 1.71748102e-02
 9.22424197e-02 9.99990582e-01 9.83552396e-01 9.97956514e-01
 1.86944753e-01 7.11140335e-01 9.89661694e-01 5.25070906e-01
 1.34705216e-01 9.99357045e-01 9.86502945e-01 6.72320008e-01
 2.60820389e-02 9.98433650e-01 4.71611619e-02 5.62225580e-02
 5.29787958e-01 3.11883986e-02 1.00000000e+00 8.21241736e-01
 3.55422497e-04 8.34614038e-04 8.66346955e-01 1.20871505e-06
 3.26308787e-01 9.91401553e-01 8.64742100e-02 1.58890188e-02
 4.33817506e-03 7.46801496e-03 5.64253330e-03 7.50483453e-01
 4.25908864e-02 7.46707439e-01 1.17939681e-01 9.97386634e-01
 8.64570737e-02 2.37090677e-01 3.60474676e-01 9.87726212e-01
 9.99305010e-01 9.95241523e-01 9.93068099e-01 4.50728744e-01
 1.98194456e-07 7.91766942e-02 5.51948607e-01 4.99693453e-02
 1.19352222e-07 9.96520042e-01 2.92211771e-04 1.66192085e-01
 9.99151349e-01 9.32887435e-01 1.65250003e-01 9.02702391e-01
 9.22120194e-07 7.48923779e-01 9.94404078e-01 1.66294575e-01
 6.87689543e-01 6.23434305e-01 5.24627328e-01 4.70777601e-01
 8.19143772e-01 9.83037353e-01 3.71605158e-04 7.40909576e-03
 2.93944120e-01 4.87364531e-02 5.62220521e-07 9.99997377e-01
 1.36584878e-01 9.99151111e-01 4.21373159e-01 1.58822536e-03
 9.94511783e-01 1.00000000e+00 9.52966809e-02 3.83762270e-01
 4.19867069e-01 3.45724821e-03 9.98467684e-01 8.97121382e-08
 9.94649351e-01 1.47943765e-01 9.98810053e-01 8.83311033e-04
 4.29716706e-03 9.46193933e-02 2.06354533e-09 5.07507324e-01
 1.65466577e-01 6.28306031e-01 2.05372572e-02 4.52437997e-03
 9.39948559e-01 5.58247983e-01 9.80595291e-01 9.99908268e-01
 4.01709080e-02 5.26305735e-01 9.38979089e-02 7.06169128e-01
 9.99993920e-01 9.95700479e-01 2.25546956e-02 1.09042436e-01]  en la posicion:  102  desde el vector:  [5.4314733e-03 6.9429043e-06 2.3194402e-02 9.8880845e-01 3.7135169e-01
 5.1514506e-03 1.9648287e-01 3.1883678e-01 9.9988627e-01 1.1416313e-01
 9.4117880e-01 5.4116696e-02 9.4479764e-01 8.7992656e-01 4.7878736e-01
 9.9897563e-01 8.2117468e-02 2.4127364e-03 1.9022703e-01 3.5602093e-02
 9.0748668e-03 9.9997103e-01 9.9998951e-01 9.7853625e-01 9.7084481e-01
 7.3002279e-03 9.7269499e-01 9.8057503e-01 2.6415640e-01 9.4464576e-01
 9.9987388e-01 1.7703837e-01 6.1794072e-02 9.7880614e-01 4.4273695e-01
 5.5232644e-04 6.1421484e-02 1.3819635e-03 9.9836898e-01 9.9253476e-01
 7.5065529e-01 1.7787069e-01 6.0361332e-01 3.4710765e-04 2.9089659e-02
 8.6300147e-01 4.0336406e-01 3.2336884e-05 9.4729364e-03 4.0798843e-02
 9.5870912e-01 8.6280346e-01 5.4200971e-01 2.2914708e-03 1.4480948e-03
 9.7063768e-01 3.1565607e-05 4.8859197e-01 9.9649704e-01 2.5390553e-01
 8.2455397e-01 9.8975909e-01 9.9997008e-01 3.5065621e-02 1.4924862e-07
 1.8090227e-01 9.7000003e-01 5.4077882e-01 2.1144748e-04 9.9466014e-01
 4.2335093e-03 8.8728487e-02 9.9952054e-01 3.3871680e-02 3.9928329e-01
 5.6311983e-01 5.4522455e-02 4.2471939e-01 6.4308196e-01 2.8341740e-02
 9.9192399e-01 9.9983007e-01 9.9980354e-01 9.9830663e-01 8.1073064e-01
 1.2867835e-01 1.3253093e-03 9.1976106e-01 6.5836310e-04 1.0962784e-03
 1.1324883e-03 9.2064393e-01 2.3583558e-01 9.9464715e-01 2.5577798e-01
 3.6059737e-02 9.9495029e-01 9.9995458e-01 9.4648182e-02 9.9708533e-01
 6.7244887e-02 3.1483173e-04 9.9960363e-01 1.7072467e-05 9.5258242e-01
 5.5525303e-03 9.9883628e-01 5.3370565e-02 5.0005394e-01 1.2920356e-01
 2.1581320e-06 4.6012402e-03 6.5882218e-01 9.2688107e-01 7.6824248e-02
 6.1526000e-01 6.2758267e-01 4.3041110e-03 9.3052739e-01 1.1982137e-01
 2.2028387e-03 8.2907951e-01 2.5143802e-02 9.9988973e-01 5.6646436e-01
 9.6167684e-01 6.5914589e-01 1.6892701e-01]  que se encuentra en:  79
Imagen de BD:  191  Imagen de consulta:  191 

Distancia:  3.7916875  al vector:  [9.9949855e-01 9.9483508e-01 1.0208458e-02 9.9999928e-01 7.6254785e-01
 9.9999410e-01 7.9926842e-01 1.4713407e-04 9.7485447e-01 4.3717027e-04
 4.4611096e-04 4.5264691e-01 7.3675608e-09 7.7557904e-01 1.7007431e-01
 1.0000000e+00 4.4790863e-06 9.7976589e-01 9.9998498e-01 7.9538512e-01
 1.7377673e-05 8.4117055e-04 9.9875712e-01 3.4523343e-06 9.9999976e-01
 7.4883306e-01 1.7256687e-05 1.0000000e+00 6.8045527e-01 2.2938728e-05
 9.9878830e-01 2.8880769e-01 2.4928153e-03 9.9978590e-01 6.4969361e-02
 1.1639917e-08 1.3459843e-01 5.5348529e-08 5.7661533e-04 9.9900091e-01
 9.9455881e-01 7.4393213e-01 4.1084158e-01 8.5822099e-01 2.5007367e-02
 1.0000000e+00 2.4466217e-02 1.6495287e-03 7.4023008e-04 9.9999487e-01
 5.7283115e-01 9.6750653e-01 1.3560742e-02 4.3862149e-01 7.0786928e-06
 1.3214350e-03 2.2885203e-04 1.3241255e-01 9.9574310e-01 6.3238442e-03
 5.3105080e-01 4.0458882e-01 3.9647314e-01 3.3673441e-01 5.0457070e-06
 1.0459453e-02 1.6118270e-01 1.7933285e-01 9.5591366e-01 4.1785598e-02
 9.9996591e-01 5.2781439e-01 1.9022226e-03 2.8499389e-01 1.5173051e-01
 8.1285203e-01 9.9992740e-01 3.7132084e-02 2.3383224e-01 2.8954297e-02
 2.9283223e-01 9.9920386e-01 6.5249574e-01 9.9999231e-01 1.3535866e-01
 1.9701570e-02 9.3590009e-01 2.6314139e-02 4.6365368e-08 7.4175965e-05
 4.0900391e-01 4.5952267e-05 6.6717714e-01 1.3157725e-04 1.7086267e-03
 2.4241033e-06 9.9999845e-01 1.0974324e-01 1.5810094e-05 4.7896010e-01
 7.6611978e-01 5.1772296e-03 9.9964172e-01 9.9990666e-01 1.3131991e-01
 3.3595178e-08 9.9905306e-01 1.7637238e-06 9.9794775e-01 2.2406748e-01
 9.9997735e-01 1.6137901e-01 9.9118459e-01 3.9836389e-01 4.4253305e-01
 9.9992454e-01 9.9987286e-01 1.3084114e-03 2.7609700e-01 1.2632221e-02
 3.6875963e-07 2.4963841e-01 5.4962051e-01 2.3751616e-02 6.4906478e-04
 9.9999815e-01 9.9924457e-01 1.9390315e-02]  en la posicion:  4  desde el vector:  [1.0000000e+00 1.0000000e+00 3.2010376e-02 9.9999499e-01 3.0113843e-01
 1.0000000e+00 4.2278767e-01 8.7766733e-07 7.5029814e-01 6.3136220e-04
 8.0212816e-11 8.4106672e-01 5.8974865e-12 6.2370503e-01 1.1107108e-01
 9.7240919e-01 2.4424791e-03 1.0000000e+00 9.9999660e-01 7.7466917e-01
 2.0482525e-05 1.8386057e-10 9.9936914e-01 7.4111539e-09 9.9994206e-01
 5.9175491e-04 2.3039059e-08 9.9708837e-01 9.9258715e-01 1.0916347e-09
 9.4598055e-01 5.4444754e-01 9.9999595e-01 9.9079359e-01 9.1025114e-01
 1.6128013e-01 4.8581460e-06 4.8156142e-02 6.1438334e-11 9.9998695e-01
 1.0000000e+00 9.9840665e-01 1.1896482e-01 1.0000000e+00 2.4135402e-10
 9.9999785e-01 5.4925144e-02 1.3544559e-03 3.2701761e-02 1.0000000e+00
 4.4793785e-03 5.4484814e-02 3.6341371e-06 6.2149668e-01 8.5945034e-01
 7.1534514e-03 6.9756806e-03 7.2770500e-01 9.9767673e-01 1.4033459e-08
 3.3428758e-02 9.2868507e-02 4.2747259e-03 9.2715621e-03 9.9990302e-01
 6.9202030e-01 1.6888976e-04 9.4787598e-02 1.0000000e+00 3.1234622e-03
 1.0000000e+00 9.9037826e-01 1.7144714e-05 6.2286359e-01 2.7311045e-01
 2.3209956e-01 1.0000000e+00 9.9759501e-01 6.1745632e-01 3.9428473e-04
 1.7844021e-02 3.8763931e-01 9.7650480e-01 9.8055577e-01 2.4221516e-01
 1.9079447e-04 9.9997860e-01 9.9917078e-01 5.4989630e-01 9.8644457e-05
 1.0000000e+00 9.3385577e-04 9.9157757e-01 1.6111229e-06 2.2446409e-01
 6.1404317e-05 7.1542776e-01 3.2657441e-10 2.1167545e-05 2.4433762e-02
 3.0915737e-03 7.4237585e-04 9.9975389e-01 1.0000000e+00 3.0408255e-06
 1.7866883e-01 1.2916356e-02 3.7384033e-04 9.9817199e-01 4.3752640e-01
 1.0000000e+00 6.0169399e-03 8.9372259e-01 9.9788159e-01 7.9925644e-01
 9.9999511e-01 1.0000000e+00 8.2605302e-02 3.8877130e-04 1.7137194e-07
 3.9657682e-02 6.9572479e-02 8.4072387e-01 5.9383810e-02 3.1089996e-06
 9.9957681e-01 9.6109378e-01 1.5047938e-02]  que se encuentra en:  80
Imagen de BD:  102  Imagen de consulta:  197 

Distancia:  3.8792732  al vector:  [8.5207474e-01 9.5463967e-01 8.2880461e-01 9.9973941e-01 5.3241980e-01
 7.8151435e-01 1.7233503e-01 5.1462011e-06 9.9997807e-01 4.8316208e-01
 1.7710358e-02 2.3513436e-03 7.8531229e-01 9.7880948e-01 2.4498871e-01
 7.1226412e-01 6.8139136e-03 9.3891466e-01 9.9997008e-01 6.6237003e-02
 5.9938312e-02 7.8863716e-01 9.9998498e-01 6.1397463e-01 1.9274139e-01
 7.8832477e-02 9.5717907e-01 6.3575566e-01 2.3811281e-01 2.9567963e-01
 9.9992883e-01 2.5416404e-01 9.9166489e-01 9.9995786e-01 9.1696489e-01
 2.6354027e-01 1.6842577e-06 5.5266023e-03 1.7794204e-01 9.6572673e-01
 9.8934448e-01 7.7710271e-01 9.9844688e-01 8.3394140e-02 2.3453708e-06
 9.9484867e-01 8.1252789e-01 8.4277644e-06 1.8873811e-04 9.9999976e-01
 9.4109762e-01 9.6509147e-01 7.7347584e-05 4.1319638e-01 5.0026178e-03
 9.2695224e-01 5.8326125e-04 8.0394095e-01 9.9443203e-01 2.5035786e-05
 6.2718052e-01 2.6587170e-01 9.9994731e-01 4.8441470e-02 6.3033879e-02
 1.7540336e-02 1.8019676e-02 1.0655549e-01 9.9525088e-01 9.8585397e-01
 2.4066767e-01 6.9100893e-01 9.7587335e-01 8.9264214e-02 4.7134012e-01
 8.7495685e-01 2.9964834e-01 5.6128174e-01 1.7740405e-01 3.9549768e-03
 9.8610657e-01 4.0732318e-01 4.4143045e-01 9.7167665e-01 1.0948658e-02
 1.1265427e-02 9.2447066e-01 9.9998915e-01 1.9504488e-02 6.5318942e-03
 8.4019244e-01 9.7762084e-01 4.6361685e-03 8.5330403e-01 5.3145915e-02
 8.2448125e-04 9.9461246e-01 5.2373052e-02 2.4151334e-01 7.0208967e-01
 5.3230882e-01 4.2736530e-04 9.9976456e-01 2.7502319e-01 2.9710621e-02
 5.0243032e-01 9.3584490e-01 3.2227039e-03 1.9772232e-01 9.0972745e-01
 7.3657203e-01 2.6434180e-01 7.5201762e-01 8.7525845e-01 2.5089025e-02
 9.9995255e-01 9.9997604e-01 1.9942844e-01 8.4740710e-01 3.4852988e-01
 2.8592944e-03 8.7244415e-01 1.2902352e-01 9.9996740e-01 5.7187706e-01
 9.9944907e-01 9.9997377e-01 1.9266933e-02]  en la posicion:  113  desde el vector:  [1.42077237e-01 9.89147723e-01 8.73401523e-01 9.92347360e-01
 9.05576527e-01 5.78869581e-01 4.60137427e-02 6.62361344e-06
 9.94470477e-01 9.53451157e-01 5.84414005e-02 2.81849504e-03
 2.55026877e-01 9.94622409e-01 5.37546098e-01 8.42511415e-01
 1.30355358e-04 9.93432879e-01 9.99998569e-01 1.62109733e-02
 2.67226428e-01 2.65335977e-01 9.99970496e-01 8.88272643e-01
 1.59196287e-01 1.07634068e-02 9.31191325e-01 9.89647031e-01
 2.46070415e-01 8.99938464e-01 9.97132540e-01 8.16183567e-01
 8.27276707e-01 9.99667406e-01 4.47031856e-03 7.36722350e-03
 2.28475437e-06 2.80913711e-03 4.70737725e-01 3.68971109e-01
 9.13976908e-01 8.22101533e-02 5.66942453e-01 8.28769922e-01
 1.14172032e-07 9.98983741e-01 2.19859809e-01 1.99288130e-04
 8.39255881e-05 9.99999285e-01 6.87908947e-01 5.04949391e-01
 7.18329829e-06 3.06213260e-01 6.55496120e-03 9.53541338e-01
 6.71750307e-03 1.46288306e-01 9.56890464e-01 9.19015861e-07
 9.48160887e-01 8.09220135e-01 9.79843974e-01 6.30054772e-02
 7.13576555e-01 1.36935711e-02 1.64700568e-01 2.51185238e-01
 9.99713182e-01 1.71924800e-01 4.25464362e-01 9.62331891e-01
 9.95402455e-01 3.32233310e-03 6.22390449e-01 7.97418356e-02
 6.33514762e-01 6.53736055e-01 9.47732687e-01 5.43608129e-01
 9.69530225e-01 4.62088585e-02 1.33258104e-03 9.98591900e-01
 8.66653204e-01 9.20674801e-02 4.59787548e-01 9.99988556e-01
 6.79987967e-02 3.41531634e-03 6.90993667e-02 9.94613826e-01
 1.01890326e-01 9.41646338e-01 1.26491547e-01 7.93039799e-04
 9.99085963e-01 1.89974606e-02 7.53295660e-01 2.47636020e-01
 2.80977786e-02 1.29431486e-04 9.98233557e-01 9.02099371e-01
 1.50613785e-01 1.91936821e-01 8.33791673e-01 4.17828560e-04
 5.15960217e-01 2.16727883e-01 9.97915626e-01 3.18705320e-01
 3.18320632e-01 3.91284883e-01 6.29723072e-04 9.99996126e-01
 9.99989152e-01 7.14461505e-02 8.85778666e-01 9.00008798e-01
 9.18596983e-04 8.99762809e-02 1.25307828e-01 9.96785283e-01
 9.30945694e-01 9.99759257e-01 9.99996901e-01 5.46385944e-01]  que se encuentra en:  81
Imagen de BD:  200  Imagen de consulta:  200 

Distancia:  3.702786  al vector:  [2.16108561e-02 1.01789832e-03 7.79095531e-01 9.99450982e-01
 8.29905272e-02 4.95016575e-03 8.37165117e-01 9.85485435e-01
 9.90181161e-06 9.97822642e-01 9.99516428e-01 6.92982078e-01
 8.04684222e-01 8.11190486e-01 8.51178169e-03 6.95234656e-01
 3.55601311e-04 9.71943140e-04 1.23718381e-02 3.87166977e-01
 9.94667709e-01 5.66926599e-03 1.77711248e-04 1.20106936e-02
 1.93724036e-03 7.36433268e-01 8.99374485e-03 8.10248673e-01
 6.74277544e-03 9.99456286e-01 1.02043152e-03 4.88453716e-01
 1.00985169e-02 1.64359808e-02 2.22653478e-01 6.48489594e-02
 9.97767568e-01 9.83191788e-01 9.99960363e-01 9.97995496e-01
 7.06047940e-05 3.43399465e-01 4.09322709e-01 2.32177973e-01
 9.99890924e-01 9.96146917e-01 1.93528324e-01 9.99868214e-01
 9.92880225e-01 5.08874655e-04 4.94131833e-01 4.26082075e-01
 5.33303618e-03 6.29714131e-01 8.50323439e-01 4.94684875e-02
 9.99992013e-01 1.29353613e-01 9.97479677e-01 9.99561071e-01
 9.95545626e-01 8.94864976e-01 2.95618176e-03 9.32829142e-01
 1.10629499e-01 1.47745013e-03 3.55954766e-02 4.38887477e-02
 1.24603510e-04 8.71067286e-01 4.35858965e-04 8.02383661e-01
 8.82720947e-01 9.54952121e-01 5.42386889e-01 6.97883904e-01
 2.03315431e-05 4.28999692e-01 9.14096117e-01 4.17382419e-02
 9.62653995e-01 5.59568405e-03 1.92412734e-03 7.87547231e-02
 9.98928547e-01 9.90561247e-01 1.82000220e-01 3.70069146e-02
 9.41598415e-03 9.99656796e-01 1.45837665e-03 9.61036205e-01
 4.70525086e-01 9.93282914e-01 1.21247828e-01 6.28167391e-03
 7.64040351e-02 9.99098241e-01 9.80239093e-01 4.69321162e-01
 3.00729126e-01 9.99899626e-01 8.28679622e-05 4.09547687e-02
 8.38523626e-01 1.22877896e-01 9.97495055e-01 1.53437257e-02
 4.28148866e-01 3.80575657e-04 2.51707435e-03 4.07695532e-01
 9.04581606e-01 9.05758262e-01 1.23573810e-01 2.60618329e-03
 9.70858216e-01 6.28361106e-01 5.46472967e-02 9.99074936e-01
 9.10134375e-01 5.11699319e-02 4.39468563e-01 1.08450651e-03
 9.96469378e-01 9.92866397e-01 3.83401215e-02 4.62089837e-01]  en la posicion:  41  desde el vector:  [7.0699323e-05 8.6540878e-02 5.3530157e-02 9.9473989e-01 9.7534382e-01
 8.1165183e-01 4.9719062e-01 9.8876739e-01 8.8434695e-05 9.8733753e-01
 9.9111277e-01 2.3647547e-03 6.5627079e-05 9.7709334e-01 3.3630431e-02
 9.9978459e-01 1.2136996e-03 4.9648821e-02 4.9528480e-04 2.4994254e-02
 9.9970049e-01 1.0967255e-03 8.3342195e-04 1.2008928e-04 8.2084537e-04
 3.5257366e-01 1.2228733e-04 9.9992073e-01 5.7162672e-02 9.6673769e-01
 3.5294299e-05 5.6048042e-01 8.3646178e-04 3.6568934e-05 9.9137151e-01
 1.3598800e-04 9.9645901e-01 9.9997967e-01 9.8440772e-01 9.8170388e-01
 3.0666739e-02 6.3242185e-01 8.3647716e-01 9.9719656e-01 9.9903893e-01
 9.9945438e-01 9.9562442e-01 9.9880195e-01 9.9982560e-01 2.2975519e-01
 9.9923992e-01 1.4310941e-01 8.1408024e-04 9.9877661e-01 9.9998748e-01
 4.0224195e-04 9.8898345e-01 2.2806028e-01 9.8312020e-01 9.9889123e-01
 2.9082865e-01 9.3728578e-01 9.6091628e-04 8.7516308e-01 1.6469929e-01
 2.5879765e-01 2.9161149e-01 9.9931800e-01 2.2816807e-02 5.3222060e-02
 2.4841654e-01 8.3275980e-01 9.9985790e-01 8.2613683e-01 7.3935032e-01
 3.8829446e-04 4.4175848e-01 5.4796517e-01 8.9205146e-01 1.2944341e-03
 4.4715419e-01 6.2176585e-04 3.5500526e-04 7.9247355e-04 6.2413812e-02
 9.7706532e-01 8.2692904e-06 4.8196018e-03 1.9603968e-04 9.9930131e-01
 7.1749091e-04 9.9982071e-01 1.0357165e-01 9.9999708e-01 9.6987545e-01
 1.6946882e-02 1.4615059e-04 8.5493708e-01 9.9867874e-01 2.8293824e-01
 1.1990231e-01 9.9998969e-01 8.5473484e-06 1.0990697e-01 9.8566949e-01
 1.1417270e-03 6.0437948e-01 1.6698629e-02 3.6407053e-02 3.6424091e-01
 8.5828602e-02 6.2123156e-01 9.3384707e-01 4.6776235e-03 2.5331837e-01
 5.6322813e-03 9.9940145e-01 2.3214477e-01 2.6361299e-01 9.0529478e-01
 9.9923551e-01 2.6955715e-01 9.7699821e-01 8.3694795e-06 9.6962136e-01
 9.8414010e-01 5.8383584e-02 9.9998975e-01]  que se encuentra en:  82
Imagen de BD:  136  Imagen de consulta:  11 

Porcentaje de aciertos:  0.2289156626506024
```


**[Celda 21 - Código]**
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
        dist = np.sqrt(np.sum(np.square(a-b))) # REEMPLAZAR POR LA SIAMESA FUNCIONANDO
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


*Salida:*
```text
Distancia:  1.3531255  al vector en la posicion:  522  desde el vector que se encuentra en:  0
Imagen de BD:  7  Imagen de consulta:  7 

Distancia:  1.8063743  al vector en la posicion:  187  desde el vector que se encuentra en:  1
Imagen de BD:  2  Imagen de consulta:  2 

Distancia:  0.9891061  al vector en la posicion:  276  desde el vector que se encuentra en:  2
Imagen de BD:  1  Imagen de consulta:  1 

Distancia:  1.8528072  al vector en la posicion:  612  desde el vector que se encuentra en:  3
Imagen de BD:  0  Imagen de consulta:  0 

Distancia:  1.4768372  al vector en la posicion:  616  desde el vector que se encuentra en:  4
Imagen de BD:  4  Imagen de consulta:  4 

Distancia:  1.2454683  al vector en la posicion:  466  desde el vector que se encuentra en:  5
Imagen de BD:  1  Imagen de consulta:  1 

Distancia:  2.3824866  al vector en la posicion:  272  desde el vector que se encuentra en:  6
Imagen de BD:  4  Imagen de consulta:  4 

Distancia:  1.929183  al vector en la posicion:  932  desde el vector que se encuentra en:  7
Imagen de BD:  9  Imagen de consulta:  9 

Distancia:  4.144176  al vector en la posicion:  846  desde el vector que se encuentra en:  8
Imagen de BD:  6  Imagen de consulta:  5 

Distancia:  1.9833982  al vector en la posicion:  525  desde el vector que se encuentra en:  9
Imagen de BD:  9  Imagen de consulta:  9 

Distancia:  1.4818515  al vector en la posicion:  293  desde el vector que se encuentra en:  10
Imagen de BD:  0  Imagen de consulta:  0 

Distancia:  2.2888472  al vector en la posicion:  488  desde el vector que se encuentra en:  11
Imagen de BD:  6  Imagen de consulta:  6 

Distancia:  1.4534576  al vector en la posicion:  87  desde el vector que se encuentra en:  12
Imagen de BD:  9  Imagen de consulta:  9 

Distancia:  1.6082133  al vector en la posicion:  210  desde el vector que se encuentra en:  13
Imagen de BD:  0  Imagen de consulta:  0 

Distancia:  1.6382506  al vector en la posicion:  366  desde el vector que se encuentra en:  14
Imagen de BD:  1  Imagen de consulta:  1 

Distancia:  1.6646671  al vector en la posicion:  544  desde el vector que se encuentra en:  15
Imagen de BD:  5  Imagen de consulta:  5 

Distancia:  1.7601887  al vector en la posicion:  350  desde el vector que se encuentra en:  16
Imagen de BD:  9  Imagen de consulta:  9 

Distancia:  1.3470286  al vector en la posicion:  522  desde el vector que se encuentra en:  17
Imagen de BD:  7  Imagen de consulta:  7 

Distancia:  3.6116762  al vector en la posicion:  966  desde el vector que se encuentra en:  18
Imagen de BD:  3  Imagen de consulta:  3 

Distancia:  1.8052909  al vector en la posicion:  92  desde el vector que se encuentra en:  19
Imagen de BD:  4  Imagen de consulta:  4 

Distancia:  1.7685779  al vector en la posicion:  195  desde el vector que se encuentra en:  20
Imagen de BD:  9  Imagen de consulta:  9 

Distancia:  1.7585044  al vector en la posicion:  106  desde el vector que se encuentra en:  21
Imagen de BD:  6  Imagen de consulta:  6 

Distancia:  2.1746206  al vector en la posicion:  488  desde el vector que se encuentra en:  22
Imagen de BD:  6  Imagen de consulta:  6 

Distancia:  1.7490054  al vector en la posicion:  396  desde el vector que se encuentra en:  23
Imagen de BD:  5  Imagen de consulta:  5 

Distancia:  1.9921435  al vector en la posicion:  338  desde el vector que se encuentra en:  24
Imagen de BD:  4  Imagen de consulta:  4 

Distancia:  2.5714393  al vector en la posicion:  114  desde el vector que se encuentra en:  25
Imagen de BD:  0  Imagen de consulta:  0 

Distancia:  2.104193  al vector en la posicion:  820  desde el vector que se encuentra en:  26
Imagen de BD:  7  Imagen de consulta:  7 

Distancia:  1.6903901  al vector en la posicion:  92  desde el vector que se encuentra en:  27
Imagen de BD:  4  Imagen de consulta:  4 

Distancia:  2.194256  al vector en la posicion:  216  desde el vector que se encuentra en:  28
Imagen de BD:  0  Imagen de consulta:  0 

Distancia:  1.576407  al vector en la posicion:  638  desde el vector que se encuentra en:  29
Imagen de BD:  1  Imagen de consulta:  1 

Distancia:  1.9662745  al vector en la posicion:  12  desde el vector que se encuentra en:  30
Imagen de BD:  3  Imagen de consulta:  3 

Distancia:  1.4924943  al vector en la posicion:  738  desde el vector que se encuentra en:  31
Imagen de BD:  1  Imagen de consulta:  1 

Distancia:  1.4472694  al vector en la posicion:  298  desde el vector que se encuentra en:  32
Imagen de BD:  3  Imagen de consulta:  3 

Distancia:  3.595277  al vector en la posicion:  850  desde el vector que se encuentra en:  33
Imagen de BD:  4  Imagen de consulta:  4 

Distancia:  1.7217901  al vector en la posicion:  728  desde el vector que se encuentra en:  34
Imagen de BD:  7  Imagen de consulta:  7 

Distancia:  2.1859536  al vector en la posicion:  365  desde el vector que se encuentra en:  35
Imagen de BD:  2  Imagen de consulta:  2 

Distancia:  2.4128609  al vector en la posicion:  349  desde el vector que se encuentra en:  36
Imagen de BD:  7  Imagen de consulta:  7 

Distancia:  1.5060928  al vector en la posicion:  14  desde el vector que se encuentra en:  37
Imagen de BD:  1  Imagen de consulta:  1 

Distancia:  2.555144  al vector en la posicion:  938  desde el vector que se encuentra en:  38
Imagen de BD:  2  Imagen de consulta:  2 

Distancia:  1.4442599  al vector en la posicion:  710  desde el vector que se encuentra en:  39
Imagen de BD:  1  Imagen de consulta:  1 

Distancia:  1.3394457  al vector en la posicion:  366  desde el vector que se encuentra en:  40
Imagen de BD:  1  Imagen de consulta:  1 

Distancia:  2.6743762  al vector en la posicion:  324  desde el vector que se encuentra en:  41
Imagen de BD:  7  Imagen de consulta:  7 

Distancia:  2.3607838  al vector en la posicion:  692  desde el vector que se encuentra en:  42
Imagen de BD:  4  Imagen de consulta:  4 

Distancia:  2.1580331  al vector en la posicion:  109  desde el vector que se encuentra en:  43
Imagen de BD:  2  Imagen de consulta:  2 

Distancia:  2.2101572  al vector en la posicion:  992  desde el vector que se encuentra en:  44
Imagen de BD:  3  Imagen de consulta:  3 

Distancia:  1.8341472  al vector en la posicion:  396  desde el vector que se encuentra en:  45
Imagen de BD:  5  Imagen de consulta:  5 

Distancia:  1.5543735  al vector en la posicion:  870  desde el vector que se encuentra en:  46
Imagen de BD:  1  Imagen de consulta:  1 

Distancia:  2.0568116  al vector en la posicion:  213  desde el vector que se encuentra en:  47
Imagen de BD:  2  Imagen de consulta:  2 

Distancia:  1.666597  al vector en la posicion:  336  desde el vector que se encuentra en:  48
Imagen de BD:  4  Imagen de consulta:  4 

Distancia:  1.5859164  al vector en la posicion:  804  desde el vector que se encuentra en:  49
Imagen de BD:  4  Imagen de consulta:  4 

Distancia:  1.8596282  al vector en la posicion:  794  desde el vector que se encuentra en:  50
Imagen de BD:  6  Imagen de consulta:  6 

Distancia:  1.8349102  al vector en la posicion:  895  desde el vector que se encuentra en:  51
Imagen de BD:  3  Imagen de consulta:  3 

Distancia:  1.9271755  al vector en la posicion:  396  desde el vector que se encuentra en:  52
Imagen de BD:  5  Imagen de consulta:  5 

Distancia:  0.8398129  al vector en la posicion:  316  desde el vector que se encuentra en:  53
Imagen de BD:  5  Imagen de consulta:  5 

Distancia:  2.8249562  al vector en la posicion:  93  desde el vector que se encuentra en:  54
Imagen de BD:  6  Imagen de consulta:  6 

Distancia:  1.4718294  al vector en la posicion:  952  desde el vector que se encuentra en:  55
Imagen de BD:  0  Imagen de consulta:  0 

Distancia:  1.361205  al vector en la posicion:  877  desde el vector que se encuentra en:  56
Imagen de BD:  4  Imagen de consulta:  4 

Distancia:  0.79395425  al vector en la posicion:  604  desde el vector que se encuentra en:  57
Imagen de BD:  1  Imagen de consulta:  1 

Distancia:  1.8446131  al vector en la posicion:  116  desde el vector que se encuentra en:  58
Imagen de BD:  9  Imagen de consulta:  9 

Distancia:  2.3629904  al vector en la posicion:  65  desde el vector que se encuentra en:  59
Imagen de BD:  5  Imagen de consulta:  5 

Distancia:  1.5679921  al vector en la posicion:  686  desde el vector que se encuentra en:  60
Imagen de BD:  7  Imagen de consulta:  7 

Distancia:  1.7921052  al vector en la posicion:  935  desde el vector que se encuentra en:  61
Imagen de BD:  8  Imagen de consulta:  8 

Distancia:  2.6534052  al vector en la posicion:  374  desde el vector que se encuentra en:  62
Imagen de BD:  9  Imagen de consulta:  9 

Distancia:  2.2519236  al vector en la posicion:  856  desde el vector que se encuentra en:  63
Imagen de BD:  3  Imagen de consulta:  3 

Distancia:  1.991672  al vector en la posicion:  730  desde el vector que se encuentra en:  64
Imagen de BD:  7  Imagen de consulta:  7 

Distancia:  1.8201824  al vector en la posicion:  26  desde el vector que se encuentra en:  65
Imagen de BD:  4  Imagen de consulta:  4 

Distancia:  2.4240603  al vector en la posicion:  770  desde el vector que se encuentra en:  66
Imagen de BD:  6  Imagen de consulta:  6 

Distancia:  2.021701  al vector en la posicion:  461  desde el vector que se encuentra en:  67
Imagen de BD:  4  Imagen de consulta:  4 

Distancia:  1.2871882  al vector en la posicion:  840  desde el vector que se encuentra en:  68
Imagen de BD:  3  Imagen de consulta:  3 

Distancia:  2.1097  al vector en la posicion:  249  desde el vector que se encuentra en:  69
Imagen de BD:  0  Imagen de consulta:  0 

Distancia:  1.3383442  al vector en la posicion:  562  desde el vector que se encuentra en:  70
Imagen de BD:  7  Imagen de consulta:  7 

Distancia:  1.4293733  al vector en la posicion:  429  desde el vector que se encuentra en:  71
Imagen de BD:  0  Imagen de consulta:  0 

Distancia:  2.699045  al vector en la posicion:  684  desde el vector que se encuentra en:  72
Imagen de BD:  2  Imagen de consulta:  2 

Distancia:  1.7563937  al vector en la posicion:  641  desde el vector que se encuentra en:  73
Imagen de BD:  9  Imagen de consulta:  9 

Distancia:  1.5989734  al vector en la posicion:  366  desde el vector que se encuentra en:  74
Imagen de BD:  1  Imagen de consulta:  1 

Distancia:  1.7540007  al vector en la posicion:  950  desde el vector que se encuentra en:  75
Imagen de BD:  7  Imagen de consulta:  7 

Distancia:  1.4199116  al vector en la posicion:  752  desde el vector que se encuentra en:  76
Imagen de BD:  3  Imagen de consulta:  3 

Distancia:  1.6817734  al vector en la posicion:  664  desde el vector que se encuentra en:  77
Imagen de BD:  2  Imagen de consulta:  2 

Distancia:  1.6181194  al vector en la posicion:  487  desde el vector que se encuentra en:  78
Imagen de BD:  9  Imagen de consulta:  9 

Distancia:  1.6161265  al vector en la posicion:  911  desde el vector que se encuentra en:  79
Imagen de BD:  7  Imagen de consulta:  7 

Distancia:  2.2231743  al vector en la posicion:  608  desde el vector que se encuentra en:  80
Imagen de BD:  7  Imagen de consulta:  7 

Distancia:  1.403631  al vector en la posicion:  32  desde el vector que se encuentra en:  81
Imagen de BD:  6  Imagen de consulta:  6 

Distancia:  1.4013425  al vector en la posicion:  642  desde el vector que se encuentra en:  82
Imagen de BD:  2  Imagen de consulta:  2 

Distancia:  1.8999169  al vector en la posicion:  608  desde el vector que se encuentra en:  83
Imagen de BD:  7  Imagen de consulta:  7 

Distancia:  1.8325435  al vector en la posicion:  532  desde el vector que se encuentra en:  84
Imagen de BD:  8  Imagen de consulta:  8 

Distancia:  2.067268  al vector en la posicion:  58  desde el vector que se encuentra en:  85
Imagen de BD:  4  Imagen de consulta:  4 

Distancia:  1.8630446  al vector en la posicion:  839  desde el vector que se encuentra en:  86
Imagen de BD:  7  Imagen de consulta:  7 

Distancia:  2.580766  al vector en la posicion:  992  desde el vector que se encuentra en:  87
Imagen de BD:  3  Imagen de consulta:  3 

Distancia:  1.636171  al vector en la posicion:  762  desde el vector que se encuentra en:  88
Imagen de BD:  6  Imagen de consulta:  6 

Distancia:  2.2128818  al vector en la posicion:  466  desde el vector que se encuentra en:  89
Imagen de BD:  1  Imagen de consulta:  1 

Distancia:  1.5565743  al vector en la posicion:  767  desde el vector que se encuentra en:  90
Imagen de BD:  3  Imagen de consulta:  3 

Distancia:  1.4444704  al vector en la posicion:  90  desde el vector que se encuentra en:  91
Imagen de BD:  6  Imagen de consulta:  6 

Distancia:  1.4995799  al vector en la posicion:  319  desde el vector que se encuentra en:  92
Imagen de BD:  9  Imagen de consulta:  9 

Distancia:  1.6607918  al vector en la posicion:  675  desde el vector que se encuentra en:  93
Imagen de BD:  3  Imagen de consulta:  3 

Distancia:  1.890162  al vector en la posicion:  248  desde el vector que se encuentra en:  94
Imagen de BD:  1  Imagen de consulta:  1 

Distancia:  2.8778827  al vector en la posicion:  906  desde el vector que se encuentra en:  95
Imagen de BD:  4  Imagen de consulta:  4 

Distancia:  1.6501905  al vector en la posicion:  738  desde el vector que se encuentra en:  96
Imagen de BD:  1  Imagen de consulta:  1 

Distancia:  2.5822005  al vector en la posicion:  38  desde el vector que se encuentra en:  97
Imagen de BD:  7  Imagen de consulta:  7 

Distancia:  2.4909582  al vector en la posicion:  846  desde el vector que se encuentra en:  98
Imagen de BD:  6  Imagen de consulta:  6 

Distancia:  1.4521662  al vector en la posicion:  434  desde el vector que se encuentra en:  99
Imagen de BD:  9  Imagen de consulta:  9 

Porcentaje de aciertos:  0.99
```


**[Celda 22 - Código]**
```python

```
