---
aliases: [Similitud-Marcas_CNN_TripletLossMining_2023_v3]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2023-06-13
origen_zip: GIBD-20260521T205218Z-3-003.zip
ruta_interna: "GIBD/CNN Marcas/Notebooks/Similitud-Marcas_CNN_TripletLossMining_2023_v3.ipynb"
tamanio_bytes: 3526073
---

# Notebook: Similitud-Marcas_CNN_TripletLossMining_2023_v3.ipynb

Ruta interna: `GIBD/CNN Marcas/Notebooks/Similitud-Marcas_CNN_TripletLossMining_2023_v3.ipynb`

---

# Busqueda por Similitud de Marcas

## TRIPLET LOSS con ONLINE HARD TRIPLET MINING


**[Celda 2 - Código]**
```python
# IMPORTACIÓN DE LIBRERÍAS #


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D,Activation, Dropout, BatchNormalization, Concatenate # merge,
from tensorflow.keras.models import Model, Sequential
#from keras.regularizers import l2
#from keras import backend as K
from tensorflow.keras.optimizers import Adam
from skimage.io import imshow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from tensorflow import keras as kr
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.cm as cm
import math
import cv2
import os
import glob
import random
from skimage.morphology import skeletonize
from matplotlib.pylab import *
import pickle


# Definición de constantes ANCHO y ALTO de las imágenes #

ANCHO = 28
ALTO = 28
```


**[Celda 3 - Código]**
```python
import tensorflow as tf

# Verifica que se esté usando la GPU
if tf.test.gpu_device_name():
    print('GPU encontrada:', tf.test.gpu_device_name())
else:
    print("No se encontró la GPU")
```


*Salida:*
```text
GPU encontrada: /device:GPU:0
```


**[Celda 4 - Código]**
```python
# DEFINICIÓN DE LA FUNCIÓN: DISTANCIA EUCLIDIANA #


def euclidiana (a, b):
    return tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=1))
```


**[Celda 5 - Código]**
```python
# CARGA DE IMÁGENES DE MARCAS #

import random

# Selección del directorio que contiene la base de datos de marcas (con aumentación) #

#directorio = r'C:\Users\User\Documents\UTN\GIBD\Similitud de Marcas\BDNormalizada_R_Aumentada(28x28)'


# Selección del directorio que contiene la base de datos de marcas (con aumentación y relieve) #

#directorio = r'C:\Users\gibd\Desktop\GIBD\CNN Marcas 2023\Bases de Datos\BDNormalizada_R_Aumentada(28x28)_Relieve'
directorio = r'C:\Users\User\Documents\UTN\GIBD\Similitud de Marcas\BDNormalizada_R_Aumentada(28x28)_Relieve'


# Definición de variables #

marcasBDA = []
y_marcasBDA = []
marcasBDA_test = []
y_marcasBDA_test = []
porcentaje_train = 0.7

# Se recorre y cargan las imagenes del directorio en las variables #

for i in range(203):
    clase = i + 1
    print(clase)
    # Si se utiliza el directorio con relieve: .txt / Si se utiliza el directorio sin relieve: .jpg
    ruta = directorio + '\\' + str(clase) + r'_*.txt'
    contenido = glob.glob(ruta)
    random.shuffle(contenido)
    n = 1
    n_train = len(contenido) * porcentaje_train
    for image in contenido:
        relieve = np.loadtxt(image, dtype=float)
        image_gris = relieve.reshape(28, 28, 1)
        if n < n_train:
            marcasBDA.append(image_gris)
            y_marcasBDA.append(str(clase))
        else:
            marcasBDA_test.append(image_gris)
            y_marcasBDA_test.append(str(clase))
        n += 1


# CÓDIGO EXTRA #


"""for fichero in contenido:
    if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.txt'):
        image = directorio + '/' + fichero
        print(image)
        relieve = np.loadtxt(directorio + '/' + fichero[:fichero.find('.')] + '.txt', dtype=float)
        marcasBDA.append(relieve)
        if (fichero.find('_') >= 0):
            n = fichero[:fichero.find('_')]
        else:
            n = fichero[:fichero.find('.')]
        y_marcasBDA.append(n)"""
```


*Salida:*
```text
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
182
183
184
185
186
187
188
189
190
191
192
193
194
195
196
197
198
199
200
201
202
203
"for fichero in contenido:\n    if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.txt'):\n        image = directorio + '/' + fichero\n        print(image)\n        relieve = np.loadtxt(directorio + '/' + fichero[:fichero.find('.')] + '.txt', dtype=float)\n        marcasBDA.append(relieve)\n        if (fichero.find('_') >= 0):\n            n = fichero[:fichero.find('_')]\n        else:\n            n = fichero[:fichero.find('.')]\n        y_marcasBDA.append(n)"
```


**[Celda 6 - Código]**
```python
# DEFINICIÓN DEL MODELO #


VECTOR_SIZE = 256

model = Sequential()

model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(ANCHO,ALTO,1)))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.4))

model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=3,strides=2,padding='same',activation='relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.4))

model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))

model.add(Dense(VECTOR_SIZE))

#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Dropout(0.4))
#model.add(tf.keras.layers.Dense(10, activation='softmax'))

#model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

```


**[Celda 7 - Código]**
```python
model.summary()
```


*Salida:*
```text
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                 
 batch_normalization (BatchN  (None, 26, 26, 32)       128       
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 24, 24, 32)        9248      
                                                                 
 batch_normalization_1 (Batc  (None, 24, 24, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_2 (Conv2D)           (None, 12, 12, 32)        25632     
                                                                 
 batch_normalization_2 (Batc  (None, 12, 12, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_3 (Conv2D)           (None, 10, 10, 64)        18496     
                                                                 
 batch_normalization_3 (Batc  (None, 10, 10, 64)       256       
 hNormalization)                                                 
                                                                 
 conv2d_4 (Conv2D)           (None, 8, 8, 64)          36928     
                                                                 
 batch_normalization_4 (Batc  (None, 8, 8, 64)         256       
 hNormalization)                                                 
                                                                 
 conv2d_5 (Conv2D)           (None, 4, 4, 64)          102464    
                                                                 
 max_pooling2d (MaxPooling2D  (None, 2, 2, 64)         0         
 )                                                               
                                                                 
 batch_normalization_5 (Batc  (None, 2, 2, 64)         256       
 hNormalization)                                                 
                                                                 
 conv2d_6 (Conv2D)           (None, 1, 1, 64)          36928     
                                                                 
 batch_normalization_6 (Batc  (None, 1, 1, 64)         256       
 hNormalization)                                                 
                                                                 
 flatten (Flatten)           (None, 64)                0         
                                                                 
 dense (Dense)               (None, 256)               16640     
                                                                 
=================================================================
Total params: 248,064
Trainable params: 247,360
Non-trainable params: 704
_________________________________________________________________
```


**[Celda 8 - Código]**
```python
# FUNCIÓN HARD TRIPLET LOSS #

def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.cast(tf.equal(distances, 0.0), dtype=tf.float32)
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = tf.cast(mask, dtype=tf.float32)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), dtype=tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


def hardest_positive(labels, embeddings):
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings)
    #print(pairwise_dist)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive =  tf.cast(mask_anchor_positive, dtype=tf.float32)
    #print(mask_anchor_positive)
    #print(np.sum(mask_anchor_positive))

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)
    #print(np.sum(anchor_positive_dist))

    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
    #print('HARDEST POSITIVE')
    #print(hardest_positive_dist)
    return hardest_positive_dist

def average_positive(labels, embeddings):
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings)
    #print(pairwise_dist)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive =  tf.cast(mask_anchor_positive, dtype=tf.float32)
    #print(mask_anchor_positive)
    #print(np.sum(mask_anchor_positive))

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)
    #print(np.sum(anchor_positive_dist))

    # shape (batch_size, 1)
    average_positive_dist = tf.reduce_mean(anchor_positive_dist, axis=1, keepdims=True)  # que pasa con los ceros?
    #print('HARDEST POSITIVE')
    #print(hardest_positive_dist)
    return average_positive_dist

def hardest_positive_metric(labels, embeddings):
    hp = hardest_positive(labels, embeddings)
    return tf.reduce_mean(hp)

def average_positive_metric(labels, embeddings):
    hp =average_positive(labels, embeddings)
    return tf.reduce_mean(hp)

def hardest_negative(labels, embeddings):
    pairwise_dist = _pairwise_distances(embeddings)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.cast(mask_anchor_negative, dtype=tf.float32)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)

    #print('HARDEST NEGATIVE')
    #print(hardest_negative_dist)
    return hardest_negative_dist

def average_negative(labels, embeddings):
    pairwise_dist = _pairwise_distances(embeddings)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.cast(mask_anchor_negative, dtype=tf.float32)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    average_negative_dist = tf.reduce_mean(anchor_negative_dist, axis=1, keepdims=True)

    #print('HARDEST NEGATIVE')
    #print(hardest_negative_dist)
    return average_negative_dist

def hardest_negative_metric(labels, embeddings):
    hp = hardest_negative(labels, embeddings)
    return tf.reduce_mean(hp)

def average_negative_metric(labels, embeddings):
    hp = average_negative(labels, embeddings)
    return tf.reduce_mean(hp)

def batch_hard_triplet_loss(labels, embeddings, margin=0.1, squared=False):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    hardest_positive_dist = hardest_positive(labels, embeddings)
    tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

    hardest_negative_dist = hardest_negative(labels, embeddings)
    tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    #triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)
    triplet_loss = tf.maximum(hardest_positive_dist+1/(hardest_negative_dist+0.0001),0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss
```


**[Celda 9 - Código]**
```python
a = tf.constant([[1.0, 0.0, 1.5], [1.0, 0.0, 2.0], [3.0, 5.0, 3.0]])
mateuc = _pairwise_distances(a)
print(mateuc)
print(euclidiana(tf.constant([[1.0, 0.0, 1.5]]), tf.constant([ [3.0, 5.0, 3.0]])))
```


*Salida:*
```text
tf.Tensor(
[[0.       0.5      5.59017 ]
 [0.5      0.       5.477226]
 [5.59017  5.477226 0.      ]], shape=(3, 3), dtype=float32)
tf.Tensor([5.59017], shape=(1,), dtype=float32)
```


**[Celda 10 - Código]**
```python
# DEFINICIÓN DE FUNCIONES: AUMENTACIÓN DE IMÁGENES #

from skimage import transform
import tensorflow.compat.v1 as tf1

tf1.disable_v2_behavior()

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

def gauss_ruido (imagen, ruido):
    noise = tf.random.normal(shape = tf.shape(imagen), mean = 0.0, stddev = ruido, dtype = tf.float64)
    tensor_aum = tf.add(imagen, noise)
    return tensor_aum.eval(session = tf.compat.v1.Session())

def rotar (imagen, angulo = 0):
    return transform.rotate(imagen, angle = angulo)

def trasladar (imagen, desp_fila, desp_colum):
    translation = tf.image.pad_to_bounding_box(imagen.reshape(1,28,28,1), desp_fila, desp_colum, 28 + desp_fila, 28 + desp_colum)
    tensor_aum = tf.image.crop_to_bounding_box(translation, 0, 0, 28, 28)
    t = tensor_aum.eval(session=tf.compat.v1.Session())
    return t.reshape(28,28,1)
```


*Salida:*
```text
WARNING:tensorflow:From C:\Users\User\AppData\Roaming\Python\Python39\site-packages\tensorflow\python\compat\v2_compat.py:107: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
```


**[Celda 11 - Código]**
```python
from scipy.ndimage import rotate

def escalar(imagen, escala):
    if escala > 1:
        escala = 1
    imagen_expandida = tf.expand_dims(imagen, axis=-1)  # Agregar una dimensión al final
    aum = tf.image.resize(imagen_expandida, size=(int(28 * escala), int(28 * escala)), method=tf.image.ResizeMethod.BILINEAR)
    pad_filas = (28 - aum.shape[0]) // 2
    pad_colum = (28 - aum.shape[1]) // 2
    tensor_aum = tf.image.pad_to_bounding_box(aum, pad_filas, pad_colum, 28, 28)
    return tensor_aum

def cortar(imagen, dim_corte):
    seed = tf.random.uniform(shape=[], minval=0, maxval=1234, dtype=tf.int32)
    tensor_crop = tf.image.random_crop(imagen, size=dim_corte, seed=seed)
    filas = tensor_crop.shape[0]
    columnas = tensor_crop.shape[1]
    aj_filas = (28 - filas) // 2
    aj_columnas = (28 - columnas) // 2

    if len(tensor_crop.shape) == 2:  # Si la imagen tiene solo 2 dimensiones
        tensor_crop = tf.expand_dims(tensor_crop, axis=-1)  # Añadir dimensión de canal

    tensor_aum = tf.image.pad_to_bounding_box(tensor_crop, aj_filas, aj_columnas, 28, 28)
    return tensor_aum

def gauss_ruido(imagen, ruido):
    imagen = tf.cast(imagen, dtype=tf.float64)  # Convertir imagen a float64
    noise = tf.random.normal(shape=tf.shape(imagen), mean=0.0, stddev=ruido, dtype=tf.float64)
    tensor_aum = tf.add(imagen, noise)
    return tensor_aum

def rotar(imagen, angulo=0):
    imagen_rotada = rotate(imagen, angle=-angulo, reshape=False)
    return imagen_rotada

def trasladar(imagen, desp_fila, desp_columna):
    imagen_traslada = np.roll(imagen, shift=(desp_fila, desp_columna), axis=(0, 1))
    imagen_traslada[:desp_fila, :] = 0
    imagen_traslada[:, :desp_columna] = 0
    return imagen_traslada[:28, :28]
```


**[Celda 12 - Código]**
```python
#Pruebas de funciones:

img = cv2.imread(r'C:\Users\User\Documents\UTN\GIBD\Similitud de Marcas\QueriesDibujadas\q_143.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap=cm.Greys)

img = preprocesarImagen(img, 28, 28, 5)
#aum = rotar(img, angulo=30)
#aum = escalar(img, 0.5)
#aum = cortar(img, [21, 21])
#aum = gauss_ruido(img, 0.01)
aum = trasladar(img, 3, 3)
#aum = aumentar(img)

plt.imshow(aum, cmap=cm.Greys)
```


*Salida:*
```text
MATRIZ DE LA IMAGEN:
<Figure size 432x288 with 1 Axes>ESCALADA CON DEFORMACIÓN:
<Figure size 432x288 with 1 Axes>SKELETONIZADA:
<Figure size 432x288 with 1 Axes>RELIEVE:
<Figure size 432x288 with 1 Axes><matplotlib.image.AxesImage at 0x1f71a453670><Figure size 432x288 with 1 Axes>
```


**[Celda 13 - Código]**
```python
# DEFINICIÓN DE LA FUNCIÓN: AUMENTAR para aumentar una imagen con función de aumentación aleatoria #
import random

def aumentar(img):

    ran = random.randint(0, 1)
    escala = random.choice([0.6, 0.7, 0.8])

    if (ran > 0.8):
        img = escalar(img, escala) #Definir valor para 'escala'. Ej: 0.6, 0.7, 0.8

    ran = random.randint(0, 1)
    corte = random.choice([[21,21], [23,23]])

    if (ran > 0.8):
        img = cortar(img, corte) #Definir valor para 'corte'. Ej: [21,21], [23,23]

    ran = random.randint(0, 1)
    ruido = random.choice([0.01, 0.02])

    if (ran > 0.8):
        img = gauss_ruido(img, ruido) #Definir valor para 'ruido'. Ej: 0.03

    ran = random.randint(0, 1)
    angulo = random.choice([5, 10, 15, 20, 25, 30, 35, 40])

    if (ran > 0.8):
        img = rotar(img, angulo) #Definir valor para 'angulo' entre 5 y 40 grados.

    ran = random.randint(0, 1)
    trasladox = random.choice([1, 2, 3])
    trasladoy = random.choice([1, 2, 3])

    if (ran > 0.8):
        img = trasladar(img, trasladox, trasladoy) #Definir valores para 'trasladox' y 'trasladoy'. Ej: 3 y 3, 2 y 1.

    return img
```


**[Celda 14 - Código]**
```python
# DEFINICIÓN DE LA FUNCIÓN: GENERADOR DE TRIPLES para TRIPLET HARD MINING CON AUMENTACIÓN ONLINE #

import random

def data_generator(x, y, batch_size=32):
    # tiene que devolver un tensor de batch_size imágenes y otro tensor de batch_size con las etiquetas correspondientes a los elementos del batch

    n_classes = 203

    id_class = random.randint(0, n_classes - 1)

    while True:
        etiquetas =  np.empty(shape=(batch_size,), dtype=np.int32)
        imagenes = np.empty((batch_size, ANCHO, ALTO, 1), dtype=np.float32)
        i = 0
        while i < batch_size:
            imagenes[i] = x[id_class].reshape(28,28,1)
            etiquetas[i] = y[id_class]
            i += 1
            if i < batch_size:
                imgAum = aumentar(x[id_class])
                imagenes[i] = imgAum.reshape(28,28,1)
                etiquetas[i] = y[id_class]
            i += 1
            if i < batch_size:
                imgAum = aumentar(x[id_class])
                imagenes[i] = imgAum.reshape(28,28,1)
                etiquetas[i] = y[id_class]

            id_class += 1
            if id_class > n_classes - 1:
                id_class = 0

        yield imagenes, etiquetas
```


**[Celda 15 - Código]**
```python
# EJECUCIÓN E IMPRESIÓN DE LA GENERACIÓN DE TRIPLES #
%matplotlib inline

# Definición del tamaño del lote #

tam_lote = 16


# Ejecución de la función data-generator #

gen = data_generator(marcasBDA, y_marcasBDA, batch_size=tam_lote)


# Se recorren e imprimen en pantallas los triples generados #

for i in range (4):
    X_list, Y_list = next(gen)
    for j in range (tam_lote):
        fig, axes = plt.subplots(1, 1)
        print(X_list[j].shape)
        print(np.sum(X_list[j]))
        plt.imshow(X_list[j], cmap=cm.Greys)
        print('Etiqueta:',Y_list[j])
        plt.show()
        print(i+1, ' - ', j+1)


```


*Salida:*
```text
(28, 28, 1)
21.533201
Etiqueta: 195
<Figure size 432x288 with 1 Axes>1  -  1
(28, 28, 1)
22.18191
Etiqueta: 196
<Figure size 432x288 with 1 Axes>1  -  2
(28, 28, 1)
21.08543
Etiqueta: 197
<Figure size 432x288 with 1 Axes>1  -  3
(28, 28, 1)
22.39373
Etiqueta: 198
<Figure size 432x288 with 1 Axes>1  -  4
(28, 28, 1)
23.97435
Etiqueta: 199
<Figure size 432x288 with 1 Axes>1  -  5
(28, 28, 1)
20.98257
Etiqueta: 200
<Figure size 432x288 with 1 Axes>1  -  6
(28, 28, 1)
22.628
Etiqueta: 201
<Figure size 432x288 with 1 Axes>1  -  7
(28, 28, 1)
20.81186
Etiqueta: 202
<Figure size 432x288 with 1 Axes>1  -  8
(28, 28, 1)
20.223282
Etiqueta: 203
<Figure size 432x288 with 1 Axes>1  -  9
(28, 28, 1)
19.73263
Etiqueta: 1
<Figure size 432x288 with 1 Axes>1  -  10
(28, 28, 1)
18.40012
Etiqueta: 2
<Figure size 432x288 with 1 Axes>1  -  11
(28, 28, 1)
21.857552
Etiqueta: 3
<Figure size 432x288 with 1 Axes>1  -  12
(28, 28, 1)
16.63577
Etiqueta: 4
<Figure size 432x288 with 1 Axes>1  -  13
(28, 28, 1)
21.78986
Etiqueta: 5
<Figure size 432x288 with 1 Axes>1  -  14
(28, 28, 1)
21.733679
Etiqueta: 6
<Figure size 432x288 with 1 Axes>1  -  15
(28, 28, 1)
18.53696
Etiqueta: 7
<Figure size 432x288 with 1 Axes>1  -  16
(28, 28, 1)
18.987598
Etiqueta: 8
<Figure size 432x288 with 1 Axes>2  -  1
(28, 28, 1)
23.43178
Etiqueta: 9
<Figure size 432x288 with 1 Axes>2  -  2
(28, 28, 1)
19.10195
Etiqueta: 10
<Figure size 432x288 with 1 Axes>2  -  3
(28, 28, 1)
19.84518
Etiqueta: 11
<Figure size 432x288 with 1 Axes>2  -  4
(28, 28, 1)
21.35833
Etiqueta: 12
<Figure size 432x288 with 1 Axes>2  -  5
(28, 28, 1)
20.463501
Etiqueta: 13
<Figure size 432x288 with 1 Axes>2  -  6
(28, 28, 1)
17.34142
Etiqueta: 14
<Figure size 432x288 with 1 Axes>2  -  7
(28, 28, 1)
23.61132
Etiqueta: 15
<Figure size 432x288 with 1 Axes>2  -  8
(28, 28, 1)
21.893421
Etiqueta: 16
<Figure size 432x288 with 1 Axes>2  -  9
(28, 28, 1)
22.56508
Etiqueta: 17
<Figure size 432x288 with 1 Axes>2  -  10
(28, 28, 1)
19.48327
Etiqueta: 18
<Figure size 432x288 with 1 Axes>2  -  11
(28, 28, 1)
23.32748
Etiqueta: 19
<Figure size 432x288 with 1 Axes>2  -  12
(28, 28, 1)
23.920738
Etiqueta: 20
<Figure size 432x288 with 1 Axes>2  -  13
(28, 28, 1)
20.84337
Etiqueta: 21
<Figure size 432x288 with 1 Axes>2  -  14
(28, 28, 1)
22.30014
Etiqueta: 22
<Figure size 432x288 with 1 Axes>2  -  15
(28, 28, 1)
23.317749
Etiqueta: 23
<Figure size 432x288 with 1 Axes>2  -  16
(28, 28, 1)
19.797739
Etiqueta: 24
<Figure size 432x288 with 1 Axes>3  -  1
(28, 28, 1)
22.66368
Etiqueta: 25
<Figure size 432x288 with 1 Axes>3  -  2
(28, 28, 1)
20.26112
Etiqueta: 26
<Figure size 432x288 with 1 Axes>3  -  3
(28, 28, 1)
19.24371
Etiqueta: 27
<Figure size 432x288 with 1 Axes>3  -  4
(28, 28, 1)
23.43482
Etiqueta: 28
<Figure size 432x288 with 1 Axes>3  -  5
(28, 28, 1)
21.86964
Etiqueta: 29
<Figure size 432x288 with 1 Axes>3  -  6
(28, 28, 1)
22.168411
Etiqueta: 30
<Figure size 432x288 with 1 Axes>3  -  7
(28, 28, 1)
21.6539
Etiqueta: 31
<Figure size 432x288 with 1 Axes>3  -  8
(28, 28, 1)
21.97042
Etiqueta: 32
<Figure size 432x288 with 1 Axes>3  -  9
(28, 28, 1)
23.85158
Etiqueta: 33
<Figure size 432x288 with 1 Axes>3  -  10
(28, 28, 1)
22.9659
Etiqueta: 34
<Figure size 432x288 with 1 Axes>3  -  11
(28, 28, 1)
22.29472
Etiqueta: 35
<Figure size 432x288 with 1 Axes>3  -  12
(28, 28, 1)
21.32716
Etiqueta: 36
<Figure size 432x288 with 1 Axes>3  -  13
(28, 28, 1)
22.74694
Etiqueta: 37
<Figure size 432x288 with 1 Axes>3  -  14
(28, 28, 1)
20.809193
Etiqueta: 38
<Figure size 432x288 with 1 Axes>3  -  15
(28, 28, 1)
20.77104
Etiqueta: 39
<Figure size 432x288 with 1 Axes>3  -  16
(28, 28, 1)
22.008839
Etiqueta: 40
<Figure size 432x288 with 1 Axes>4  -  1
(28, 28, 1)
22.18311
Etiqueta: 41
<Figure size 432x288 with 1 Axes>4  -  2
(28, 28, 1)
17.318539
Etiqueta: 42
<Figure size 432x288 with 1 Axes>4  -  3
(28, 28, 1)
19.57548
Etiqueta: 43
<Figure size 432x288 with 1 Axes>4  -  4
(28, 28, 1)
20.44178
Etiqueta: 44
<Figure size 432x288 with 1 Axes>4  -  5
(28, 28, 1)
24.233711
Etiqueta: 45
<Figure size 432x288 with 1 Axes>4  -  6
(28, 28, 1)
22.28615
Etiqueta: 46
<Figure size 432x288 with 1 Axes>4  -  7
(28, 28, 1)
22.737709
Etiqueta: 47
<Figure size 432x288 with 1 Axes>4  -  8
(28, 28, 1)
20.06144
Etiqueta: 48
<Figure size 432x288 with 1 Axes>4  -  9
(28, 28, 1)
18.299511
Etiqueta: 49
<Figure size 432x288 with 1 Axes>4  -  10
(28, 28, 1)
21.952461
Etiqueta: 50
<Figure size 432x288 with 1 Axes>4  -  11
(28, 28, 1)
20.367989
Etiqueta: 51
<Figure size 432x288 with 1 Axes>4  -  12
(28, 28, 1)
18.77378
Etiqueta: 52
<Figure size 432x288 with 1 Axes>4  -  13
(28, 28, 1)
20.615921
Etiqueta: 53
<Figure size 432x288 with 1 Axes>4  -  14
(28, 28, 1)
18.18739
Etiqueta: 54
<Figure size 432x288 with 1 Axes>4  -  15
(28, 28, 1)
19.95677
Etiqueta: 55
<Figure size 432x288 with 1 Axes>4  -  16
```


**[Celda 16 - Código]**
```python
tam_lote = 406


# Ejecución de la función data-generator #

gen = data_generator(marcasBDA, y_marcasBDA, batch_size=tam_lote)


# Se recorren e imprimen en pantallas los triples generados #

for i in range (4):
    X_list, Y_list = next(gen)
    embbebdings = model.predict(X_list)
    #print(X_list.shape)
    #print(Y_list.shape)
    maskp = _get_anchor_positive_triplet_mask(Y_list)
    print(np.sum(maskp))
    #amask = maskp.numpy()
    print(Y_list)
    #print('MASK  POSITIVES')
    #for h in range(len(amask)):
    #    print(amask[h])
    loss = batch_hard_triplet_loss(Y_list, embbebdings)
    print('Batch: ',len(X_list))
    print('triplet loss hard: ',loss)
```


*Salida:*
```text
406
[139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156
 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174
 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192
 193 194 195 196 197 198 199 200 201 202 203   1   2   3   4   5   6   7
   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25
  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41  42  43
  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61
  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79
  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97
  98  99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115
 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133
 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151
 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169
 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187
 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203   1   2
   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38
  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56
  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72  73  74
  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92
  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110
 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128
 129 130 131 132 133 134 135 136 137 138]
HARDEST POSTIVE
tf.Tensor(
[[0.00055277]
 [0.00087531]
 [0.0008532 ]
 [0.00061178]
 [0.0007798 ]
 [0.00090754]
 [0.0004319 ]
 [0.00061247]
 [0.00092492]
 [0.00076036]
 [0.00090253]
 [0.00130435]
 [0.00078442]
 [0.00080404]
 [0.00076078]
 [0.00093536]
 [0.00041887]
 [0.00081913]
 [0.00110467]
 [0.00086329]
 [0.00074264]
 [0.00114123]
 [0.00065659]
 [0.00067233]
 [0.00060294]
 [0.00064494]
 [0.00079962]
 [0.00133506]
 [0.00091911]
 [0.00046338]
 [0.00072571]
 [0.00063502]
 [0.00114455]
 [0.00121869]
 [0.00094341]
 [0.00099027]
 [0.0010425 ]
 [0.00078481]
 [0.00082993]
 [0.00076683]
 [0.00089956]
 [0.00044639]
 [0.00104076]
 [0.00051643]
 [0.00042358]
 [0.00062692]
 [0.0006596 ]
 [0.00064591]
 [0.00075072]
 [0.00095638]
 [0.00074449]
 [0.00080167]
 [0.00068478]
 [0.00075275]
 [0.0009092 ]
 [0.00057644]
 [0.00089448]
 [0.00079263]
 [0.00073922]
 [0.00100658]
 [0.00073026]
 [0.00131422]
 [0.00035173]
 [0.00106235]
 [0.00103343]
 [0.00098595]
 [0.00091431]
 [0.00046885]
 [0.000749  ]
 [0.00077136]
 [0.0004817 ]
 [0.00058913]
 [0.00077678]
 [0.00106632]
 [0.00040731]
 [0.00070142]
 [0.0008889 ]
 [0.00061591]
 [0.00095586]
 [0.00060345]
 [0.00039706]
 [0.00062642]
 [0.        ]
 [0.00035707]
 [0.00042519]
 [0.00097288]
 [0.00072333]
 [0.00046729]
 [0.00081301]
 [0.00080377]
 [0.00033773]
 [0.00089252]
 [0.00043428]
 [0.00072913]
 [0.00077646]
 [0.00020727]
 [0.00035968]
 [0.00048584]
 [0.00045387]
 [0.00069304]
 [0.00088984]
 [0.00077322]
 [0.00046671]
 [0.00058662]
 [0.00086912]
 [0.00056336]
 [0.00139877]
 [0.00083436]
 [0.00069601]
 [0.00054501]
 [0.00059154]
 [0.00042434]
 [0.00103563]
 [0.00132526]
 [0.00038448]
 [0.00062508]
 [0.00073804]
 [0.00071217]
 [0.0008822 ]
 [0.00061924]
 [0.0008398 ]
 [0.00059217]
 [0.00078265]
 [0.00069922]
 [0.00074876]
 [0.00086959]
 [0.00039224]
 [0.00103291]
 [0.00074585]
 [0.00057225]
 [0.00067441]
 [0.00070272]
 [0.00086957]
 [0.00113019]
 [0.00092097]
 [0.00084029]
 [0.00107264]
 [0.00075153]
 [0.00095304]
 [0.00075731]
 [0.00137789]
 [0.00097724]
 [0.00057986]
 [0.00092553]
 [0.00098112]
 [0.00044241]
 [0.00061275]
 [0.00113413]
 [0.00081204]
 [0.00095615]
 [0.00092588]
 [0.00042345]
 [0.00082789]
 [0.00108753]
 [0.00063142]
 [0.00045228]
 [0.00067711]
 [0.00083275]
 [0.0006998 ]
 [0.00072317]
 [0.00090353]
 [0.00066716]
 [0.00041684]
 [0.000324  ]
 [0.00075233]
 [0.00085128]
 [0.00081198]
 [0.00091095]
 [0.00074499]
 [0.00114198]
 [0.00064713]
 [0.00051114]
 [0.00072691]
 [0.00070108]
 [0.00045282]
 [0.00053352]
 [0.0010354 ]
 [0.00140748]
 [0.00051065]
 [0.00084196]
 [0.00051008]
 [0.00071921]
 [0.00075988]
 [0.00089827]
 [0.00064163]
 [0.00096788]
 [0.00083232]
 [0.00036616]
 [0.00090549]
 [0.00103754]
 [0.00097079]
 [0.00068808]
 [0.00075681]
 [0.000628  ]
 [0.00079312]
 [0.00090845]
 [0.00076557]
 [0.00062178]
 [0.00079218]
 [0.00048238]
 [0.00086945]
 [0.00065839]
 [0.00081037]
 [0.00055277]
 [0.00087531]
 [0.0008532 ]
 [0.00061178]
 [0.0007798 ]
 [0.00090754]
 [0.0004319 ]
 [0.00061247]
 [0.00092492]
 [0.00076036]
 [0.00090253]
 [0.00130435]
 [0.00078442]
 [0.00080404]
 [0.00076078]
 [0.00093536]
 [0.00041887]
 [0.00081913]
 [0.00110467]
 [0.00086329]
 [0.00074264]
 [0.00114123]
 [0.00065659]
 [0.00067233]
 [0.00060294]
 [0.00064494]
 [0.00079962]
 [0.00133506]
 [0.00091911]
 [0.00046338]
 [0.00072571]
 [0.00063502]
 [0.00114455]
 [0.00121869]
 [0.00094341]
 [0.00099027]
 [0.0010425 ]
 [0.00078481]
 [0.00082993]
 [0.00076683]
 [0.00089956]
 [0.00044639]
 [0.00104076]
 [0.00051643]
 [0.00042358]
 [0.00062692]
 [0.0006596 ]
 [0.00064591]
 [0.00075072]
 [0.00095638]
 [0.00074449]
 [0.00080167]
 [0.00068478]
 [0.00075275]
 [0.0009092 ]
 [0.00057644]
 [0.00089448]
 [0.00079263]
 [0.00073922]
 [0.00100658]
 [0.00073026]
 [0.00131422]
 [0.00035173]
 [0.00106235]
 [0.00103343]
 [0.00098595]
 [0.00091431]
 [0.00046885]
 [0.000749  ]
 [0.00077136]
 [0.0004817 ]
 [0.00058913]
 [0.00077678]
 [0.00106632]
 [0.00040731]
 [0.00070142]
 [0.0008889 ]
 [0.00061591]
 [0.00095586]
 [0.00060345]
 [0.00039706]
 [0.00062642]
 [0.        ]
 [0.00035707]
 [0.00042519]
 [0.00097288]
 [0.00072333]
 [0.00046729]
 [0.00081301]
 [0.00080377]
 [0.00033773]
 [0.00089252]
 [0.00043428]
 [0.00072913]
 [0.00077646]
 [0.00020727]
 [0.00035968]
 [0.00048584]
 [0.00045387]
 [0.00069304]
 [0.00088984]
 [0.00077322]
 [0.00046671]
 [0.00058662]
 [0.00086912]
 [0.00056336]
 [0.00139877]
 [0.00083436]
 [0.00069601]
 [0.00054501]
 [0.00059154]
 [0.00042434]
 [0.00103563]
 [0.00132526]
 [0.00038448]
 [0.00062508]
 [0.00073804]
 [0.00071217]
 [0.0008822 ]
 [0.00061924]
 [0.0008398 ]
 [0.00059217]
 [0.00078265]
 [0.00069922]
 [0.00074876]
 [0.00086959]
 [0.00039224]
 [0.00103291]
 [0.00074585]
 [0.00057225]
 [0.00067441]
 [0.00070272]
 [0.00086957]
 [0.00113019]
 [0.00092097]
 [0.00084029]
 [0.00107264]
 [0.00075153]
 [0.00095304]
 [0.00075732]
 [0.00137789]
 [0.00097724]
 [0.00057986]
 [0.00092553]
 [0.00098112]
 [0.00044241]
 [0.00061275]
 [0.00113413]
 [0.00081204]
 [0.00095615]
 [0.00092588]
 [0.00042345]
 [0.00082789]
 [0.00108753]
 [0.00063142]
 [0.00045228]
 [0.00067711]
 [0.00083275]
 [0.0006998 ]
 [0.00072317]
 [0.00090353]
 [0.00066716]
 [0.00041684]
 [0.000324  ]
 [0.00075233]
 [0.00085128]
 [0.00081198]
 [0.00091095]
 [0.00074499]
 [0.00114198]
 [0.00064713]
 [0.00051114]
 [0.00072691]
 [0.00070108]
 [0.00045282]
 [0.00053352]
 [0.0010354 ]
 [0.00140748]
 [0.00051065]
 [0.00084196]
 [0.00051008]
 [0.00071921]
 [0.00075988]
 [0.00089827]
 [0.00064163]
 [0.00096788]
 [0.00083232]
 [0.00036616]
 [0.00090549]
 [0.00103754]
 [0.00097079]
 [0.00068808]
 [0.00075681]
 [0.000628  ]
 [0.00079312]
 [0.00090845]
 [0.00076557]
 [0.00062178]
 [0.00079218]
 [0.00048238]
 [0.00086945]
 [0.00065839]
 [0.00081037]], shape=(406, 1), dtype=float32)
Batch:  406
triplet loss hard:  tf.Tensor(0.30023143, shape=(), dtype=float32)
406
[139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156
 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174
 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192
 193 194 195 196 197 198 199 200 201 202 203   1   2   3   4   5   6   7
   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25
  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41  42  43
  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61
  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79
  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97
  98  99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115
 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133
 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151
 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169
 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187
 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203   1   2
   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38
  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56
  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72  73  74
  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92
  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110
 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128
 129 130 131 132 133 134 135 136 137 138]
HARDEST POSTIVE
tf.Tensor(
[[0.00072718]
 [0.00095085]
 [0.00085855]
 [0.00072533]
 [0.00059716]
 [0.00146849]
 [0.00119763]
 [0.00089794]
 [0.00115105]
 [0.00118309]
 [0.000884  ]
 [0.00066032]
 [0.00076635]
 [0.00097381]
 [0.00109284]
 [0.00053936]
 [0.00050366]
 [0.00071994]
 [0.00103324]
 [0.00096725]
 [0.00086226]
 [0.00062508]
 [0.00089875]
 [0.00105327]
 [0.00039142]
 [0.00072028]
 [0.        ]
 [0.00056392]
 [0.00097158]
 [0.00063956]
 [0.0004597 ]
 [0.0007464 ]
 [0.00068619]
 [0.00119371]
 [0.00065803]
 [0.00046971]
 [0.00093352]
 [0.00061588]
 [0.0007647 ]
 [0.00062521]
 [0.00053141]
 [0.00073268]
 [0.00074455]
 [0.00062942]
 [0.00052711]
 [0.00088987]
 [0.00075764]
 [0.00081691]
 [0.00054784]
 [0.00056069]
 [0.00096613]
 [0.00045234]
 [0.00093606]
 [0.00095916]
 [0.00097609]
 [0.00090591]
 [0.00090339]
 [0.00040873]
 [0.00060546]
 [0.00084916]
 [0.00098759]
 [0.00074104]
 [0.00084073]
 [0.00091196]
 [0.00081163]
 [0.00097603]
 [0.00095578]
 [0.00050783]
 [0.00060053]
 [0.00086334]
 [0.00083718]
 [0.00119349]
 [0.00085316]
 [0.00056425]
 [0.00051769]
 [0.00067991]
 [0.00067109]
 [0.00055327]
 [0.00073777]
 [0.00070949]
 [0.00088116]
 [0.00077996]
 [0.00097956]
 [0.00062314]
 [0.00040719]
 [0.00104565]
 [0.00055838]
 [0.00037678]
 [0.00029501]
 [0.00073785]
 [0.00092417]
 [0.00045784]
 [0.00109027]
 [0.00055874]
 [0.00046065]
 [0.00038376]
 [0.00075817]
 [0.00059301]
 [0.00027485]
 [0.00051507]
 [0.00059585]
 [0.00063086]
 [0.00022565]
 [0.0007447 ]
 [0.00026187]
 [0.00066258]
 [0.00108312]
 [0.0007911 ]
 [0.00122601]
 [0.00054609]
 [0.00054096]
 [0.0003357 ]
 [0.00087145]
 [0.00077404]
 [0.00036204]
 [0.00042466]
 [0.00055152]
 [0.00052279]
 [0.00123469]
 [0.00083929]
 [0.00050968]
 [0.00115242]
 [0.00038038]
 [0.00043423]
 [0.0005544 ]
 [0.00033765]
 [0.00094279]
 [0.00091907]
 [0.00091823]
 [0.00065268]
 [0.00040109]
 [0.00070589]
 [0.00057871]
 [0.00086636]
 [0.0009877 ]
 [0.0007151 ]
 [0.00076082]
 [0.00092132]
 [0.00087676]
 [0.00061208]
 [0.0008004 ]
 [0.00070866]
 [0.00059429]
 [0.00095398]
 [0.00120726]
 [0.00078864]
 [0.00076473]
 [0.00084949]
 [0.00073778]
 [0.00045971]
 [0.00113048]
 [0.00047445]
 [0.00065507]
 [0.00090932]
 [0.00071967]
 [0.00083515]
 [0.00052142]
 [0.00064621]
 [0.00078412]
 [0.00047214]
 [0.00076519]
 [0.00106275]
 [0.0003814 ]
 [0.00062708]
 [0.0012151 ]
 [0.00092265]
 [0.00093524]
 [0.00082765]
 [0.00054289]
 [0.00085736]
 [0.00045028]
 [0.00115122]
 [0.00066498]
 [0.00065197]
 [0.00111142]
 [0.00073062]
 [0.00099795]
 [0.00097093]
 [0.00023047]
 [0.00067785]
 [0.00056932]
 [0.00040777]
 [0.00094524]
 [0.00054599]
 [0.00066685]
 [0.00094483]
 [0.00081313]
 [0.00045628]
 [0.00091484]
 [0.00080184]
 [0.00051325]
 [0.00040834]
 [0.00047729]
 [0.00083666]
 [0.00059038]
 [0.00092009]
 [0.00092561]
 [0.00119684]
 [0.00083157]
 [0.00105276]
 [0.00059631]
 [0.00078124]
 [0.00072497]
 [0.00072718]
 [0.00095085]
 [0.00085855]
 [0.00072533]
 [0.00059716]
 [0.00146849]
 [0.00119763]
 [0.00089794]
 [0.00115105]
 [0.00118309]
 [0.000884  ]
 [0.00066032]
 [0.00076635]
 [0.00097381]
 [0.00109284]
 [0.00053936]
 [0.00050366]
 [0.00071994]
 [0.00103324]
 [0.00096725]
 [0.00086226]
 [0.00062508]
 [0.00089875]
 [0.00105327]
 [0.00039142]
 [0.00072028]
 [0.        ]
 [0.00056392]
 [0.00097158]
 [0.00063956]
 [0.0004597 ]
 [0.0007464 ]
 [0.00068619]
 [0.00119371]
 [0.00065803]
 [0.00046971]
 [0.00093352]
 [0.00061588]
 [0.0007647 ]
 [0.00062521]
 [0.00053141]
 [0.00073268]
 [0.00074455]
 [0.00062942]
 [0.00052711]
 [0.00088987]
 [0.00075764]
 [0.00081691]
 [0.00054784]
 [0.00056069]
 [0.00096613]
 [0.00045234]
 [0.00093606]
 [0.00095916]
 [0.00097609]
 [0.00090591]
 [0.00090339]
 [0.00040873]
 [0.00060546]
 [0.00084916]
 [0.00098759]
 [0.00074104]
 [0.00084073]
 [0.00091196]
 [0.00081163]
 [0.00097603]
 [0.00095578]
 [0.00050783]
 [0.00060053]
 [0.00086334]
 [0.00083718]
 [0.00119349]
 [0.00085316]
 [0.00056425]
 [0.00051769]
 [0.00067991]
 [0.00067109]
 [0.00055327]
 [0.00073777]
 [0.00070949]
 [0.00088116]
 [0.00077996]
 [0.00097956]
 [0.00062314]
 [0.00040719]
 [0.00104565]
 [0.00055838]
 [0.00037678]
 [0.00029501]
 [0.00073785]
 [0.00092417]
 [0.00045784]
 [0.00109027]
 [0.00055874]
 [0.00046065]
 [0.00038376]
 [0.00075817]
 [0.00059301]
 [0.00027485]
 [0.00051507]
 [0.00059585]
 [0.00063086]
 [0.00022565]
 [0.0007447 ]
 [0.00026187]
 [0.00066258]
 [0.00108312]
 [0.0007911 ]
 [0.00122601]
 [0.00054609]
 [0.00054096]
 [0.0003357 ]
 [0.00087145]
 [0.00077404]
 [0.00036204]
 [0.00042466]
 [0.00055152]
 [0.00052279]
 [0.00123469]
 [0.00083929]
 [0.00050968]
 [0.00115242]
 [0.00038038]
 [0.00043423]
 [0.0005544 ]
 [0.00033765]
 [0.00094279]
 [0.00091907]
 [0.00091823]
 [0.00065268]
 [0.00040109]
 [0.00070589]
 [0.00057871]
 [0.00086636]
 [0.0009877 ]
 [0.0007151 ]
 [0.00076082]
 [0.00092132]
 [0.00087676]
 [0.00061208]
 [0.0008004 ]
 [0.00070866]
 [0.00059429]
 [0.00095398]
 [0.00120726]
 [0.00078864]
 [0.00076473]
 [0.00084949]
 [0.00073778]
 [0.00045971]
 [0.00113048]
 [0.00047445]
 [0.00065507]
 [0.00090932]
 [0.00071967]
 [0.00083515]
 [0.00052142]
 [0.00064621]
 [0.00078412]
 [0.00047214]
 [0.00076519]
 [0.00106275]
 [0.0003814 ]
 [0.00062708]
 [0.0012151 ]
 [0.00092265]
 [0.00093524]
 [0.00082765]
 [0.00054289]
 [0.00085736]
 [0.00045028]
 [0.00115122]
 [0.00066498]
 [0.00065197]
 [0.00111142]
 [0.00073062]
 [0.00099795]
 [0.00097093]
 [0.00023047]
 [0.00067785]
 [0.00056932]
 [0.00040777]
 [0.00094524]
 [0.00054599]
 [0.00066685]
 [0.00094483]
 [0.00081313]
 [0.00045628]
 [0.00091484]
 [0.00080184]
 [0.00051325]
 [0.00040834]
 [0.00047729]
 [0.00083666]
 [0.00059038]
 [0.00092009]
 [0.00092561]
 [0.00119684]
 [0.00083157]
 [0.00105276]
 [0.00059631]
 [0.00078124]
 [0.00072497]], shape=(406, 1), dtype=float32)
Batch:  406
triplet loss hard:  tf.Tensor(0.3002172, shape=(), dtype=float32)
406
[139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156
 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174
 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192
 193 194 195 196 197 198 199 200 201 202 203   1   2   3   4   5   6   7
   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25
  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41  42  43
  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61
  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79
  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97
  98  99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115
 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133
 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151
 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169
 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187
 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203   1   2
   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38
  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56
  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72  73  74
  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92
  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110
 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128
 129 130 131 132 133 134 135 136 137 138]
HARDEST POSTIVE
tf.Tensor(
[[0.00068228]
 [0.00103307]
 [0.00063174]
 [0.00086277]
 [0.00068233]
 [0.00080876]
 [0.00088196]
 [0.00069585]
 [0.00115531]
 [0.00057224]
 [0.00066475]
 [0.00119767]
 [0.00059793]
 [0.00054162]
 [0.00046592]
 [0.00074506]
 [0.00106192]
 [0.00045211]
 [0.000593  ]
 [0.00091884]
 [0.00043206]
 [0.00061561]
 [0.00076479]
 [0.00045676]
 [0.00079465]
 [0.00069338]
 [0.00070275]
 [0.00090196]
 [0.00060606]
 [0.00108961]
 [0.0007808 ]
 [0.00061522]
 [0.00084745]
 [0.00073768]
 [0.00103041]
 [0.00093389]
 [0.00035918]
 [0.00057108]
 [0.00076633]
 [0.00059058]
 [0.00057261]
 [0.00089962]
 [0.00080618]
 [0.00063878]
 [0.00051064]
 [0.00079708]
 [0.00079548]
 [0.00081414]
 [0.00063296]
 [0.00104727]
 [0.00085731]
 [0.00064837]
 [0.00068862]
 [0.00161679]
 [0.00045524]
 [0.00071425]
 [0.0010779 ]
 [0.00061989]
 [0.00059233]
 [0.00033807]
 [0.00091497]
 [0.00092132]
 [0.00053358]
 [0.00089542]
 [0.00113924]
 [0.00048505]
 [0.00083411]
 [0.00046628]
 [0.        ]
 [0.00091026]
 [0.00082644]
 [0.00121426]
 [0.00056756]
 [0.00046582]
 [0.00051052]
 [0.0010565 ]
 [0.00047357]
 [0.00081454]
 [0.00113024]
 [0.0006975 ]
 [0.00044056]
 [0.00059899]
 [0.00089136]
 [0.00045057]
 [0.0006954 ]
 [0.00062969]
 [0.00070231]
 [0.00057858]
 [0.00120452]
 [0.00101658]
 [0.00056073]
 [0.00080918]
 [0.00075794]
 [0.00085177]
 [0.00041873]
 [0.00093446]
 [0.00076227]
 [0.00061547]
 [0.00044762]
 [0.00069941]
 [0.00097429]
 [0.00086934]
 [0.00092804]
 [0.00075579]
 [0.00101545]
 [0.00068127]
 [0.00057921]
 [0.00046296]
 [0.00076296]
 [0.00087367]
 [0.00078562]
 [0.00073257]
 [0.00075235]
 [0.00064312]
 [0.00058739]
 [0.00057499]
 [0.0006388 ]
 [0.00086825]
 [0.00108947]
 [0.00113715]
 [0.00057661]
 [0.00159629]
 [0.00055948]
 [0.00065839]
 [0.00072431]
 [0.00101715]
 [0.0006637 ]
 [0.00128377]
 [0.0011262 ]
 [0.00052425]
 [0.00054279]
 [0.00044408]
 [0.00040179]
 [0.00101537]
 [0.00080065]
 [0.00102585]
 [0.00068524]
 [0.00068109]
 [0.00071143]
 [0.00083018]
 [0.00096748]
 [0.00105739]
 [0.00070987]
 [0.00092217]
 [0.00057669]
 [0.00069652]
 [0.00111518]
 [0.00035245]
 [0.00061226]
 [0.00056493]
 [0.00064295]
 [0.00063854]
 [0.00083274]
 [0.00046865]
 [0.00067606]
 [0.00033241]
 [0.00063744]
 [0.00068564]
 [0.00103352]
 [0.00057541]
 [0.00041401]
 [0.0006925 ]
 [0.00050554]
 [0.00082198]
 [0.00077809]
 [0.0006451 ]
 [0.00073769]
 [0.00102887]
 [0.00042695]
 [0.00017741]
 [0.00072732]
 [0.00050645]
 [0.00049747]
 [0.00075667]
 [0.00052824]
 [0.00072377]
 [0.00062724]
 [0.00104942]
 [0.000847  ]
 [0.00052987]
 [0.00083748]
 [0.00056718]
 [0.00089625]
 [0.0009141 ]
 [0.00057997]
 [0.00064064]
 [0.00109564]
 [0.00052742]
 [0.00091152]
 [0.00090884]
 [0.00058717]
 [0.0006962 ]
 [0.00076158]
 [0.0003253 ]
 [0.00113043]
 [0.00079193]
 [0.00086301]
 [0.00065093]
 [0.00080621]
 [0.00089604]
 [0.00065696]
 [0.00083   ]
 [0.00087265]
 [0.00068228]
 [0.00103307]
 [0.00063174]
 [0.00086277]
 [0.00068233]
 [0.00080876]
 [0.00088196]
 [0.00069585]
 [0.00115531]
 [0.00057224]
 [0.00066475]
 [0.00119767]
 [0.00059793]
 [0.00054162]
 [0.00046592]
 [0.00074506]
 [0.00106192]
 [0.00045211]
 [0.000593  ]
 [0.00091884]
 [0.00043206]
 [0.00061561]
 [0.00076479]
 [0.00045676]
 [0.00079465]
 [0.00069338]
 [0.00070275]
 [0.00090196]
 [0.00060606]
 [0.00108961]
 [0.0007808 ]
 [0.00061522]
 [0.00084745]
 [0.00073768]
 [0.00103041]
 [0.00093389]
 [0.00035918]
 [0.00057108]
 [0.00076633]
 [0.00059058]
 [0.00057261]
 [0.00089962]
 [0.00080618]
 [0.00063878]
 [0.00051064]
 [0.00079708]
 [0.00079548]
 [0.00081414]
 [0.00063296]
 [0.00104727]
 [0.00085731]
 [0.00064837]
 [0.00068862]
 [0.00161679]
 [0.00045524]
 [0.00071425]
 [0.0010779 ]
 [0.00061989]
 [0.00059233]
 [0.00033807]
 [0.00091497]
 [0.00092132]
 [0.00053358]
 [0.00089542]
 [0.00113924]
 [0.00048505]
 [0.00083411]
 [0.00046628]
 [0.        ]
 [0.00091026]
 [0.00082644]
 [0.00121426]
 [0.00056756]
 [0.00046582]
 [0.00051052]
 [0.0010565 ]
 [0.00047357]
 [0.00081454]
 [0.00113024]
 [0.0006975 ]
 [0.00044056]
 [0.00059899]
 [0.00089136]
 [0.00045057]
 [0.0006954 ]
 [0.00062969]
 [0.00070231]
 [0.00057858]
 [0.00120452]
 [0.00101658]
 [0.00056073]
 [0.00080918]
 [0.00075794]
 [0.00085177]
 [0.00041873]
 [0.00093446]
 [0.00076227]
 [0.00061547]
 [0.00044762]
 [0.00069941]
 [0.00097429]
 [0.00086934]
 [0.00092804]
 [0.00075579]
 [0.00101545]
 [0.00068127]
 [0.00057921]
 [0.00046296]
 [0.00076296]
 [0.00087367]
 [0.00078562]
 [0.00073257]
 [0.00075235]
 [0.00064312]
 [0.00058739]
 [0.00057499]
 [0.0006388 ]
 [0.00086825]
 [0.00108947]
 [0.00113715]
 [0.00057661]
 [0.00159629]
 [0.00055948]
 [0.00065839]
 [0.00072431]
 [0.00101715]
 [0.0006637 ]
 [0.00128377]
 [0.0011262 ]
 [0.00052425]
 [0.00054279]
 [0.00044408]
 [0.00040179]
 [0.00101537]
 [0.00080065]
 [0.00102585]
 [0.00068524]
 [0.00068109]
 [0.00071143]
 [0.00083018]
 [0.00096748]
 [0.00105739]
 [0.00070987]
 [0.00092217]
 [0.00057669]
 [0.00069652]
 [0.00111518]
 [0.00035245]
 [0.00061226]
 [0.00056493]
 [0.00064295]
 [0.00063854]
 [0.00083274]
 [0.00046865]
 [0.00067606]
 [0.00033241]
 [0.00063744]
 [0.00068564]
 [0.00103352]
 [0.00057541]
 [0.00041401]
 [0.0006925 ]
 [0.00050554]
 [0.00082198]
 [0.00077809]
 [0.0006451 ]
 [0.00073769]
 [0.00102887]
 [0.00042695]
 [0.00017741]
 [0.00072732]
 [0.00050645]
 [0.00049747]
 [0.00075667]
 [0.00052824]
 [0.00072377]
 [0.00062724]
 [0.00104942]
 [0.000847  ]
 [0.00052987]
 [0.00083748]
 [0.00056718]
 [0.00089625]
 [0.0009141 ]
 [0.00057997]
 [0.00064064]
 [0.00109564]
 [0.00052742]
 [0.00091152]
 [0.00090884]
 [0.00058717]
 [0.0006962 ]
 [0.00076158]
 [0.0003253 ]
 [0.00113043]
 [0.00079193]
 [0.00086301]
 [0.00065093]
 [0.00080621]
 [0.00089604]
 [0.00065696]
 [0.00083   ]
 [0.00087265]], shape=(406, 1), dtype=float32)
Batch:  406
triplet loss hard:  tf.Tensor(0.3002141, shape=(), dtype=float32)
406
[139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156
 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174
 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192
 193 194 195 196 197 198 199 200 201 202 203   1   2   3   4   5   6   7
   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25
  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41  42  43
  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61
  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79
  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97
  98  99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115
 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133
 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151
 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169
 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187
 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203   1   2
   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38
  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56
  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72  73  74
  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92
  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110
 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128
 129 130 131 132 133 134 135 136 137 138]
HARDEST POSTIVE
tf.Tensor(
[[4.7258340e-04]
 [1.0554038e-03]
 [6.9001684e-04]
 [7.6561415e-04]
 [6.4497226e-04]
 [1.2179979e-03]
 [1.2385404e-03]
 [3.6126073e-04]
 [1.0146959e-03]
 [5.7911477e-04]
 [6.9800264e-04]
 [6.5717363e-04]
 [5.1734131e-04]
 [9.0680539e-04]
 [7.6075003e-04]
 [1.0870115e-03]
 [1.0953388e-03]
 [2.6096284e-04]
 [6.2994752e-04]
 [4.0367807e-04]
 [6.5220345e-04]
 [9.9882530e-04]
 [9.7453804e-04]
 [1.1175909e-03]
 [8.1865344e-04]
 [7.9222675e-04]
 [8.4846199e-04]
 [1.3512647e-03]
 [1.0395525e-03]
 [6.5436552e-04]
 [7.9228211e-04]
 [7.1608211e-04]
 [5.7226635e-04]
 [7.7117590e-04]
 [6.2952115e-04]
 [9.0204761e-04]
 [1.0398069e-03]
 [9.1545831e-04]
 [7.6981459e-04]
 [6.8604259e-04]
 [8.1710855e-04]
 [9.3958096e-04]
 [1.0615430e-03]
 [8.0974656e-04]
 [4.9085193e-04]
 [1.2012119e-03]
 [6.9228909e-04]
 [6.7938736e-04]
 [5.4436928e-04]
 [1.1611419e-03]
 [7.7077595e-04]
 [6.2832341e-04]
 [1.0477550e-03]
 [6.0962955e-04]
 [8.2563912e-04]
 [9.4468793e-04]
 [8.6480303e-04]
 [8.3765865e-04]
 [7.3402078e-04]
 [2.0011260e-04]
 [6.6444860e-04]
 [8.3405891e-04]
 [1.0371946e-03]
 [7.2062924e-04]
 [1.1140147e-03]
 [4.7352450e-04]
 [9.2388428e-04]
 [6.3247234e-04]
 [1.4549640e-03]
 [4.5438454e-04]
 [8.6978677e-04]
 [3.0890512e-04]
 [6.5144507e-04]
 [6.8708608e-04]
 [5.7403470e-04]
 [5.7067466e-04]
 [8.2291593e-04]
 [7.4841321e-04]
 [9.9437323e-04]
 [6.6429219e-04]
 [7.8294630e-04]
 [1.0570692e-03]
 [8.0777792e-04]
 [6.5411144e-04]
 [3.4074078e-04]
 [8.2790933e-04]
 [3.7068376e-04]
 [6.7567034e-04]
 [7.7815994e-04]
 [4.7221663e-04]
 [7.7872310e-04]
 [9.9712890e-04]
 [6.8287342e-04]
 [8.0106710e-04]
 [4.6344599e-04]
 [7.0888671e-04]
 [4.2177425e-04]
 [3.2601785e-04]
 [6.0523127e-04]
 [6.5201690e-04]
 [1.1022468e-03]
 [4.7487809e-04]
 [9.3347445e-04]
 [7.1795116e-04]
 [5.9740903e-04]
 [7.5492857e-04]
 [1.1178332e-03]
 [8.6439151e-04]
 [9.3871739e-04]
 [4.3177756e-04]
 [4.3836198e-04]
 [8.0778665e-04]
 [4.8051748e-04]
 [8.8350760e-04]
 [5.0303090e-04]
 [2.2192147e-04]
 [1.0779895e-04]
 [8.4042765e-04]
 [8.3127391e-04]
 [8.5598975e-04]
 [7.2469772e-04]
 [9.0549653e-04]
 [7.2724529e-04]
 [8.9669175e-04]
 [6.8289641e-04]
 [8.2773989e-04]
 [5.0869293e-04]
 [7.0088671e-04]
 [7.5135013e-04]
 [8.8328135e-04]
 [6.4477889e-04]
 [4.9199705e-04]
 [5.9245666e-04]
 [9.0637448e-04]
 [5.2943616e-04]
 [7.5147027e-04]
 [1.0866545e-03]
 [5.2646711e-04]
 [9.1396016e-04]
 [8.5083250e-04]
 [1.1480612e-03]
 [1.0772402e-03]
 [5.5634207e-04]
 [7.3456479e-04]
 [6.0167216e-04]
 [2.9348774e-04]
 [7.8734278e-04]
 [9.9275750e-04]
 [6.0980447e-04]
 [1.0291971e-03]
 [6.0289417e-04]
 [9.1715710e-04]
 [4.0835305e-04]
 [1.2888757e-03]
 [5.8814033e-04]
 [6.5194018e-04]
 [6.3855492e-04]
 [6.5453711e-04]
 [7.5841427e-04]
 [7.7478745e-04]
 [6.8362086e-04]
 [1.0678640e-03]
 [6.0520426e-04]
 [3.5828911e-04]
 [4.2399217e-04]
 [7.3153101e-04]
 [7.6420652e-04]
 [7.1134645e-04]
 [5.3015456e-04]
 [8.0949871e-04]
 [4.2804203e-04]
 [9.0653828e-04]
 [8.9114753e-04]
 [6.5245439e-04]
 [7.1295752e-04]
 [4.4290215e-04]
 [6.6659210e-04]
 [7.6050666e-04]
 [7.4277056e-04]
 [7.0220709e-04]
 [5.0426921e-04]
 [5.9901451e-04]
 [7.5854314e-04]
 [7.9038343e-04]
 [8.3697838e-04]
 [6.8325625e-04]
 [1.1680078e-06]
 [5.5954396e-04]
 [8.7457860e-04]
 [9.9286041e-04]
 [8.8959897e-04]
 [7.6424965e-04]
 [5.2761810e-04]
 [2.8553826e-04]
 [7.4817770e-04]
 [6.8742491e-04]
 [4.3322696e-04]
 [7.5828203e-04]
 [8.7934540e-04]
 [3.0862013e-04]
 [8.3842454e-04]
 [8.3005149e-04]
 [7.7646982e-04]
 [4.7258340e-04]
 [1.0554038e-03]
 [6.9001684e-04]
 [7.6561415e-04]
 [6.4497226e-04]
 [1.2179979e-03]
 [1.2385404e-03]
 [3.6126073e-04]
 [1.0146959e-03]
 [5.7911477e-04]
 [6.9800264e-04]
 [6.5717363e-04]
 [5.1734131e-04]
 [9.0680539e-04]
 [7.6075003e-04]
 [1.0870115e-03]
 [1.0953388e-03]
 [2.6096284e-04]
 [6.2994752e-04]
 [4.0367807e-04]
 [6.5220328e-04]
 [9.9882530e-04]
 [9.7453804e-04]
 [1.1175909e-03]
 [8.1865344e-04]
 [7.9222705e-04]
 [8.4846199e-04]
 [1.3512647e-03]
 [1.0395525e-03]
 [6.5436552e-04]
 [7.9228211e-04]
 [7.1608211e-04]
 [5.7226635e-04]
 [7.7117590e-04]
 [6.2952115e-04]
 [9.0204761e-04]
 [1.0398069e-03]
 [9.1545831e-04]
 [7.6981459e-04]
 [6.8604259e-04]
 [8.1710855e-04]
 [9.3958096e-04]
 [1.0615430e-03]
 [8.0974656e-04]
 [4.9085193e-04]
 [1.2012119e-03]
 [6.9228909e-04]
 [6.7938736e-04]
 [5.4436928e-04]
 [1.1611419e-03]
 [7.7077595e-04]
 [6.2832341e-04]
 [1.0477550e-03]
 [6.0962955e-04]
 [8.2563912e-04]
 [9.4468793e-04]
 [8.6480303e-04]
 [8.3765865e-04]
 [7.3402078e-04]
 [2.0011260e-04]
 [6.6444860e-04]
 [8.3405891e-04]
 [1.0371946e-03]
 [7.2062924e-04]
 [1.1140147e-03]
 [4.7352450e-04]
 [9.2388451e-04]
 [6.3247234e-04]
 [1.4549640e-03]
 [4.5438454e-04]
 [8.6978677e-04]
 [3.0890512e-04]
 [6.5144507e-04]
 [6.8708608e-04]
 [5.7403470e-04]
 [5.7067466e-04]
 [8.2291593e-04]
 [7.4841321e-04]
 [9.9437323e-04]
 [6.6429219e-04]
 [7.8294630e-04]
 [1.0570692e-03]
 [8.0777792e-04]
 [6.5411144e-04]
 [3.4074078e-04]
 [8.2790933e-04]
 [3.7068376e-04]
 [6.7567034e-04]
 [7.7815994e-04]
 [4.7221663e-04]
 [7.7872310e-04]
 [9.9712866e-04]
 [6.8287342e-04]
 [8.0106710e-04]
 [4.6344599e-04]
 [7.0888671e-04]
 [4.2177425e-04]
 [3.2601785e-04]
 [6.0523127e-04]
 [6.5201690e-04]
 [1.1022468e-03]
 [4.7487809e-04]
 [9.3347445e-04]
 [7.1795116e-04]
 [5.9740903e-04]
 [7.5492857e-04]
 [1.1178332e-03]
 [8.6439151e-04]
 [9.3871739e-04]
 [4.3177756e-04]
 [4.3836198e-04]
 [8.0778665e-04]
 [4.8051748e-04]
 [8.8350760e-04]
 [5.0303090e-04]
 [2.2192147e-04]
 [1.0779895e-04]
 [8.4042765e-04]
 [8.3127391e-04]
 [8.5598975e-04]
 [7.2469772e-04]
 [9.0549653e-04]
 [7.2724500e-04]
 [8.9669175e-04]
 [6.8289641e-04]
 [8.2773989e-04]
 [5.0869293e-04]
 [7.0088671e-04]
 [7.5135013e-04]
 [8.8328135e-04]
 [6.4477889e-04]
 [4.9199705e-04]
 [5.9245666e-04]
 [9.0637448e-04]
 [5.2943616e-04]
 [7.5147027e-04]
 [1.0866545e-03]
 [5.2646711e-04]
 [9.1396016e-04]
 [8.5083250e-04]
 [1.1480612e-03]
 [1.0772402e-03]
 [5.5634207e-04]
 [7.3456479e-04]
 [6.0167216e-04]
 [2.9348774e-04]
 [7.8734261e-04]
 [9.9275750e-04]
 [6.0980447e-04]
 [1.0291971e-03]
 [6.0289417e-04]
 [9.1715710e-04]
 [4.0835305e-04]
 [1.2888757e-03]
 [5.8814010e-04]
 [6.5194018e-04]
 [6.3855492e-04]
 [6.5453711e-04]
 [7.5841427e-04]
 [7.7478745e-04]
 [6.8362086e-04]
 [1.0678640e-03]
 [6.0520426e-04]
 [3.5828911e-04]
 [4.2399217e-04]
 [7.3153101e-04]
 [7.6420652e-04]
 [7.1134657e-04]
 [5.3015456e-04]
 [8.0949871e-04]
 [4.2804203e-04]
 [9.0653828e-04]
 [8.9114753e-04]
 [6.5245439e-04]
 [7.1295752e-04]
 [4.4290215e-04]
 [6.6659210e-04]
 [7.6050666e-04]
 [7.4277056e-04]
 [7.0220709e-04]
 [5.0426921e-04]
 [5.9901451e-04]
 [7.5854314e-04]
 [7.9038343e-04]
 [8.3697838e-04]
 [6.8325625e-04]
 [1.1680078e-06]
 [5.5954396e-04]
 [8.7457860e-04]
 [9.9286041e-04]
 [8.8959897e-04]
 [7.6424965e-04]
 [5.2761810e-04]
 [2.8553826e-04]
 [7.4817770e-04]
 [6.8742491e-04]
 [4.3322696e-04]
 [7.5828203e-04]
 [8.7934540e-04]
 [3.0862013e-04]
 [8.3842454e-04]
 [8.3005149e-04]
 [7.7646982e-04]], shape=(406, 1), dtype=float32)
Batch:  406
triplet loss hard:  tf.Tensor(0.30021277, shape=(), dtype=float32)
```


**[Celda 17 - Código]**
```python
# TRIPLET LOSS

# base = Input(shape=(ANCHO,ALTO,1))
# positiva = Input(shape=(ANCHO,ALTO,1))
# negativa = Input(shape=(ANCHO,ALTO,1))

# modeloBase = model(base)
# modeloPositiva = model(positiva)
# modeloNegativa = model(negativa)

# vecs = Concatenate(axis=1, name='vectors')([modeloBase, modeloPositiva, modeloNegativa])

# modelTriplet = Model([base, positiva, negativa], vecs)

learning_rate = 0.0001  # Tasa de aprendizaje inicial
decay_rate = 0.00025  # Tasa de decaimiento
optimizer = Adam(learning_rate=learning_rate)

# optimizer = Adam(learning_rate=learning_rate, decay=decay_rate)
# modelTriplet.compile('adam', triplet_loss, metrics=[euc_pos, euc_neg])

# optimizer = Adam(0.001, lr_decay=2.5e-4)

modelTriplet = model

modelTriplet.compile(optimizer, batch_hard_triplet_loss, metrics=[average_positive_metric, average_negative_metric])


# siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
# siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])
```


**[Celda 18 - Código]**
```python
modelTriplet.summary()
```


*Salida:*
```text
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                 
 batch_normalization (BatchN  (None, 26, 26, 32)       128       
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 24, 24, 32)        9248      
                                                                 
 batch_normalization_1 (Batc  (None, 24, 24, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_2 (Conv2D)           (None, 12, 12, 32)        25632     
                                                                 
 batch_normalization_2 (Batc  (None, 12, 12, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_3 (Conv2D)           (None, 10, 10, 64)        18496     
                                                                 
 batch_normalization_3 (Batc  (None, 10, 10, 64)       256       
 hNormalization)                                                 
                                                                 
 conv2d_4 (Conv2D)           (None, 8, 8, 64)          36928     
                                                                 
 batch_normalization_4 (Batc  (None, 8, 8, 64)         256       
 hNormalization)                                                 
                                                                 
 conv2d_5 (Conv2D)           (None, 4, 4, 64)          102464    
                                                                 
 max_pooling2d (MaxPooling2D  (None, 2, 2, 64)         0         
 )                                                               
                                                                 
 batch_normalization_5 (Batc  (None, 2, 2, 64)         256       
 hNormalization)                                                 
                                                                 
 conv2d_6 (Conv2D)           (None, 1, 1, 64)          36928     
                                                                 
 batch_normalization_6 (Batc  (None, 1, 1, 64)         256       
 hNormalization)                                                 
                                                                 
 flatten (Flatten)           (None, 64)                0         
                                                                 
 dense (Dense)               (None, 256)               16640     
                                                                 
=================================================================
Total params: 248,064
Trainable params: 247,360
Non-trainable params: 704
_________________________________________________________________
```


**[Celda 19 - Código]**
```python
# CALL BACKS #


from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

log_dir = os.path.expanduser(r'C:\Users\User\Documents\UTN\GIBD\Modelos\ModeloTripletMining(BDA-R)_30E_v1')
tensorboard_callback = TensorBoard(log_dir=log_dir)

checkpoint_callback = ModelCheckpoint(
    filepath=r'C:\Users\User\Documents\UTN\GIBD\Modelos\ModeloTripletMining(BDA-R)_30E_v1\ModeloTripletMining(BDA-R)_30E_MP_v1.h5', # ruta para guardar el modelo
    monitor='val_loss', # métrica a monitorear
    save_best_only=True, # guardar solo los pesos del modelo que producen la mejor pérdida en el conjunto de validación
    mode='min' # elegir el modelo con la menor pérdida en el conjunto de validación
)
```


**[Celda 20 - Código]**
```python
# EJECUCIÓN DEL ENTRENAMIENTO #

# Deshabilitar la GPU
#tf.config.set_visible_devices([], 'GPU')

# with tf.device('/device:GPU:0'):
history = modelTriplet.fit(data_generator(imagenesBD, y_imagenesBD, batch_size=609), validation_data=data_generator(imagenesBD, y_imagenesBD, batch_size=609), steps_per_epoch=300, validation_steps=25, callbacks=[tensorboard_callback, checkpoint_callback], epochs=10)
```


*Salida:*
```text
Epoch 1/10
300/300 [==============================] - 1959s 6s/step - loss: 9950.9512 - average_positive_metric: 4.1700e-04 - average_negative_metric: 0.0855 - val_loss: 9950.9248 - val_average_positive_metric: 4.1991e-04 - val_average_negative_metric: 0.0863
Epoch 2/10
300/300 [==============================] - 1721s 6s/step - loss: 9950.9512 - average_positive_metric: 4.1540e-04 - average_negative_metric: 0.0852 - val_loss: 9950.9248 - val_average_positive_metric: 4.2161e-04 - val_average_negative_metric: 0.0868
Epoch 3/10
300/300 [==============================] - 1725s 6s/step - loss: 9950.9502 - average_positive_metric: 4.1418e-04 - average_negative_metric: 0.0849 - val_loss: 9950.9238 - val_average_positive_metric: 4.2187e-04 - val_average_negative_metric: 0.0868
Epoch 4/10
300/300 [==============================] - 1707s 6s/step - loss: 9950.9502 - average_positive_metric: 4.1306e-04 - average_negative_metric: 0.0847 - val_loss: 9950.9258 - val_average_positive_metric: 4.3693e-04 - val_average_negative_metric: 0.0902
Epoch 5/10
300/300 [==============================] - 1706s 6s/step - loss: 9950.9473 - average_positive_metric: 4.1087e-04 - average_negative_metric: 0.0842 - val_loss: 9950.9238 - val_average_positive_metric: 4.1536e-04 - val_average_negative_metric: 0.0853
Epoch 6/10
300/300 [==============================] - 1697s 6s/step - loss: 9950.9482 - average_positive_metric: 4.1096e-04 - average_negative_metric: 0.0842 - val_loss: 9950.9238 - val_average_positive_metric: 4.1483e-04 - val_average_negative_metric: 0.0854
Epoch 7/10
300/300 [==============================] - 1694s 6s/step - loss: 9950.9463 - average_positive_metric: 4.0883e-04 - average_negative_metric: 0.0837 - val_loss: 9950.9238 - val_average_positive_metric: 4.2787e-04 - val_average_negative_metric: 0.0880
Epoch 8/10
300/300 [==============================] - 1704s 6s/step - loss: 9950.9453 - average_positive_metric: 4.0892e-04 - average_negative_metric: 0.0838 - val_loss: 9950.9248 - val_average_positive_metric: 4.2697e-04 - val_average_negative_metric: 0.0880
Epoch 9/10
300/300 [==============================] - 1681s 6s/step - loss: 9950.9453 - average_positive_metric: 4.0798e-04 - average_negative_metric: 0.0836 - val_loss: 9950.9229 - val_average_positive_metric: 4.1137e-04 - val_average_negative_metric: 0.0846
Epoch 10/10
300/300 [==============================] - 1685s 6s/step - loss: 9950.9453 - average_positive_metric: 4.0743e-04 - average_negative_metric: 0.0835 - val_loss: 9950.9229 - val_average_positive_metric: 4.1731e-04 - val_average_negative_metric: 0.0858
```


**[Celda 21 - Código]**
```python
# MOSTRAR EL LEARNING RATE CON EL QUE SE ENTREÓ EL MODELO #

optimizer = modelTriplet.optimizer
print("Learning rate:", optimizer.learning_rate.numpy())
```


*Salida:*
```text
Learning rate: 1e-04
```


**[Celda 22 - Código]**
```python
# graficar la pérdida del modelo en cada época
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Pérdida del modelo')
plt.ylabel('Pérdida')
plt.xlabel('Época')
plt.legend(['entrenamiento', 'validación'], loc='upper right')
plt.show()

# graficar la distancia euclidiana en cada época
plt.plot(history['average_positive_metric'])
plt.plot(history['val_average_positive_metric'])
plt.title('Distancia euclidiana positiva')
plt.ylabel('Distancia euclidiana')
plt.xlabel('Época')
plt.legend(['entrenamiento', 'validación'], loc='lower right')
plt.show()

# graficar la distancia euclidiana en cada época
plt.plot(history['average_negative_metric'])
plt.plot(history['val_average_negative_metric'])
plt.title('Distancia euclidiana negativa')
plt.ylabel('Distancia euclidiana')
plt.xlabel('Época')
plt.legend(['entrenamiento', 'validación'], loc='lower right')
plt.show()

# graficar las distancias euclidianas en cada época
plt.plot(history['average_positive_metric'])
plt.plot(history['val_average_positive_metric'])
plt.plot(history['average_negative_metric'])
plt.plot(history['val_average_negative_metric'])
plt.title('Distancias euclidianas')
plt.ylabel('Distancia euclidiana')
plt.xlabel('Época')
plt.legend(['entrenamiento pos', 'validación pos', 'entrenamiento neg', 'validación neg'], loc='upper left')
plt.show()

# NOTA: agregar .history a cada history (cuando se ejecuta después de entrenar y no se carga desde el archivo)
```


*Salida:*
```text
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>
```


**[Celda 23 - Código]**
```python
# graficar la pérdida del modelo en cada época
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Pérdida del modelo')
plt.ylabel('Pérdida')
plt.xlabel('Época')
plt.legend(['entrenamiento', 'validación'], loc='upper right')
plt.show()
```


*Salida:*
```text
<Figure size 432x288 with 1 Axes>
```


**[Celda 24 - Código]**
```python
# GUARDADO DEL MODELO #


# model.save(r'C:\Users\User\Documents\UTN\GIBD\Modelos\ModeloSiamesa(5)_ExtraccionCaracteristicas_65000.h5')
# modelTriplet.save(r'C:\Users\gibd\Desktop\GIBD\CNN Marcas 2023\Modelos\ModeloTripletOnlineHardMining10.h5')
modelTriplet.save(r'C:\Users\User\Documents\UTN\GIBD\Modelos\ModeloTripletMining(BDA-R)_30E_v1\ModeloTripletMining(BDA-R)_30E_v1.h5')
# model.save(r'C:\Users\gibd\Desktop\GIBD\CNN Marcas 2023\Modelos\ModeloTriplet-I(BDA-R)_60E_v3.h5')
```


**[Celda 25 - Código]**
```python
# CARGA DEL MODELO #

#optimizer = Adam(0.001, decay=2.5e-4)
modelTriplet = tf.keras.models.load_model(r'C:\Users\User\Documents\UTN\GIBD\Modelos\ModeloTripletMining(BDA-R)_30E_v1\ModeloTripletMining(BDA-R)_30E_v1.h5',custom_objects={'Custom>Adam':optimizer, 'batch_hard_triplet_loss': batch_hard_triplet_loss, 'average_positive_metric': average_positive_metric, 'average_negative_metric': average_negative_metric})

# model = load_model('model/multi_task/try.h5', custom_objects={'loss_max': loss_max})
```


**[Celda 26 - Código]**
```python
# GUARDADO DEL HISTORY #


with open(r'C:\Users\User\Documents\UTN\GIBD\Modelos\ModeloTripletMining(BDA-R)_30E_v1\H_ModeloTripletMining(BDA-R)_30E_v1.pkl', 'wb') as file:
    pickle.dump(history.history, file)
```


**[Celda 27 - Código]**
```python
# CARGA DEL HISTORY #


with open(r'C:\Users\User\Documents\UTN\GIBD\Modelos\ModeloTripletMining(BDA-R)_30E_v1\H_ModeloTripletMining(BDA-R)_30E_v1.pkl', 'rb') as file:
    history = pickle.load(file)
```


**[Celda 28 - Código]**
```python
modelTriplet.summary()
model = modelTriplet.get_layer('sequential')
```


*Salida:*
```text
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                 
 batch_normalization (BatchN  (None, 26, 26, 32)       128       
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 24, 24, 32)        9248      
                                                                 
 batch_normalization_1 (Batc  (None, 24, 24, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_2 (Conv2D)           (None, 12, 12, 32)        25632     
                                                                 
 batch_normalization_2 (Batc  (None, 12, 12, 32)       128       
 hNormalization)                                                 
                                                                 
 conv2d_3 (Conv2D)           (None, 10, 10, 64)        18496     
                                                                 
 batch_normalization_3 (Batc  (None, 10, 10, 64)       256       
 hNormalization)                                                 
                                                                 
 conv2d_4 (Conv2D)           (None, 8, 8, 64)          36928     
                                                                 
 batch_normalization_4 (Batc  (None, 8, 8, 64)         256       
 hNormalization)                                                 
                                                                 
 conv2d_5 (Conv2D)           (None, 4, 4, 64)          102464    
                                                                 
 max_pooling2d (MaxPooling2D  (None, 2, 2, 64)         0         
 )                                                               
                                                                 
 batch_normalization_5 (Batc  (None, 2, 2, 64)         256       
 hNormalization)                                                 
                                                                 
 conv2d_6 (Conv2D)           (None, 1, 1, 64)          36928     
                                                                 
 batch_normalization_6 (Batc  (None, 1, 1, 64)         256       
 hNormalization)                                                 
                                                                 
 flatten (Flatten)           (None, 64)                0         
                                                                 
 dense (Dense)               (None, 256)               16640     
                                                                 
=================================================================
Total params: 248,064
Trainable params: 247,360
Non-trainable params: 704
_________________________________________________________________
```


**[Celda 29 - Código]**
```python
ANCHO = 28
ALTO = 28
RADIO = 5
```


**[Celda 30 - Código]**
```python
def crearMatrizDeImagen(img,xMax,yMax):
    mat = np.zeros((xMax, yMax), dtype=np.float32)
    for x in range(xMax):
        for y in range(yMax):
            if x<len(img) and y<len(img[0]):
                if img[x,y]>0.70:
                    mat[x,y] = float32(0.0) # np.float32(1.0)
                else:
                    mat[x,y] = float32(1.0) #  np.int32(0)
            else:
                mat[x,y] = float32(0.0) # np.float32(1.0)
    return mat
```


**[Celda 31 - Código]**
```python
def binarizar(img,xMax,yMax):
    mat = np.zeros((xMax, yMax), dtype=np.float32)
    for x in range(xMax):
        for y in range(yMax):
            if x<len(img) and y<len(img[0]):
                if img[x,y]>0.1:
                    mat[x,y] = float32(0.0) # np.float32(1.0)
                else:
                    mat[x,y] = float32(1.0) #  np.int32(0)
            else:
                mat[x,y] = float32(0.0) # np.float32(1.0)
    return mat
```


**[Celda 32 - Código]**
```python
def normalizarUnitario(img):
    norma = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            norma += img[i,j]**2
    norma = math.sqrt(norma)

    # normalizar
    for i in range(len(img)):
        for j in range(len(img[0])):
            img[i,j] = img[i,j]/norma

    return img
```


**[Celda 33 - Código]**
```python
def relieveSinCuda(mat, radio):
    # Precondicion: la matriz debe contener valores normalizados entre 0 y 1, donde 0 es el negro (el dibujo) y 1 es el blanco (el fondo)
    xmax = len(mat)
    ymax = len(mat[0])
    radio = int(radio)
    resultado = np.zeros_like(mat,dtype=np.float64)
    for row in range(xmax):
        for col in range(ymax):
            diam = radio * 2
            if mat[row,col]==1:
                for i in range(diam+1):
                    for j in range(diam+1):
                        xactual = row+radio-i           # 5 + 5 - 5 = 5
                        yactual = col+radio-j
                        if xactual>=0 and xactual<xmax and yactual>=0 and yactual<ymax:
                            dist = math.sqrt((xactual-row)**2 + (yactual-col)**2)
                            if dist<=radio:
                                # aux = yactual*xmax+xactual+1
                                #cuda.atomic.add(resultado,(xactual,yactual),abs(radio-dist)**2)
                                xa=int(xactual)
                                ya=int(yactual)
                                # if resultado[xa,ya]<abs(radio-dist)**2:
                                #    resultado[xa,ya]=abs(radio-dist)**2
                                resultado[xa,ya] += abs(radio-dist)  # *  (1 - mat[row,col])
    # resultado = normalizar(resultado)
    return resultado
```


**[Celda 34 - Código]**
```python
def relieveMaxSinCuda(mat, radio):
    # Precondicion: la matriz debe contener valores normalizados entre 0 y 1, donde 0 es el negro (el dibujo) y 1 es el blanco (el fondo)
    xmax = len(mat)
    ymax = len(mat[0])
    radio = int(radio)
    resultado = np.zeros_like(mat,dtype=np.float64)
    for row in range(xmax):
        for col in range(ymax):
            diam = radio * 2
            if mat[row,col]==1:
                for i in range(diam+1):
                    for j in range(diam+1):
                        xactual = row+radio-i           # 5 + 5 - 5 = 5
                        yactual = col+radio-j
                        if xactual>0 and xactual<xmax and yactual>0 and yactual<ymax:
                            dist = math.sqrt((xactual-row)**2 + (yactual-col)**2)
                            if dist<=radio:
                                # aux = yactual*xmax+xactual+1
                                #cuda.atomic.add(resultado,(xactual,yactual),abs(radio-dist)**2)
                                xa=int(xactual)
                                ya=int(yactual)
                                # if resultado[xa,ya]<abs(radio-dist)**2:
                                #    resultado[xa,ya]=abs(radio-dist)**2
                                resultado[xa,ya] = max(resultado[xa,ya], abs(radio-dist)/radio)  # *  (1 - mat[row,col])
    # resultado = normalizar(resultado)
    return resultado
```


**[Celda 35 - Código]**
```python
def centroide(img):
    xcent=0
    ycent=0
    cant=0
    for x in range(len(img)):
        for y in range(len(img[0])):
            if img[x,y]==1:
                xcent += x
                ycent += y
                cant += 1
    xcent=np.trunc(xcent/cant)
    ycent=np.trunc(ycent/cant)
    return int(xcent), int(ycent)
```


**[Celda 36 - Código]**
```python
 def centrarImagen(img,xMax,yMax):
        imgCentrada = np.zeros((xMax, yMax), dtype=np.float32)
        xcentro = xMax//2-1
        ycentro = yMax//2-1
        xcent, ycent = centroide(img)
        for x in range(len(img)):
            for y in range(len(img[0])):
                if img[x,y]==1 and x+xcentro-xcent>-1 and x+xcentro-xcent<len(imgCentrada) and y+ycentro-ycent>-1 and y+ycentro-ycent<len(imgCentrada[0]):
                    imgCentrada[x+xcentro-xcent, y+ycentro-ycent]=1
        return imgCentrada
```


**[Celda 37 - Código]**
```python
"""def escalar(src, escala):
    scale_percent = escala*100

    #calcular el 50 por ciento de las dimensiones originales
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    # cambiar el tamaño de la imagen
    # print(type(src[0][0]))
    # print(src)
    # print(dsize)
    output = cv2.resize(src, dsize)
    return output"""
```


*Salida:*
```text
'def escalar(src, escala):\n    scale_percent = escala*100\n \n    #calcular el 50 por ciento de las dimensiones originales\n    width = int(src.shape[1] * scale_percent / 100)\n    height = int(src.shape[0] * scale_percent / 100)\n \n    # dsize\n    dsize = (width, height)\n \n    # cambiar el tamaño de la imagen\n    # print(type(src[0][0]))\n    # print(src)\n    # print(dsize)\n    output = cv2.resize(src, dsize)\n    return output'
```


**[Celda 38 - Código]**
```python
"""def escalarImagen(img, xmax, ymax):
    # obtener los minimos y maximos x e y
    newimg = np.zeros((xmax, ymax), dtype=np.float32)
    minx=xmax
    miny=ymax
    maxx=0
    maxy=0
    for x in range(len(img)):
        for y in range(len(img[0])):
            if img[x,y]>0 or img[x,y]==True:
                newimg[x,y] = float32(1.0)
            else:
                newimg[x,y] = float32(0.0)
            if img[x,y]>0 or img[x,y]==True:
                if x<minx:
                    minx = x
                if x>maxx:
                    maxx = x
                if y<miny:
                    miny = y
                if y>maxy:
                    maxy = y
    rangox = maxx-minx+1
    rangoy = maxy-miny+1
    if rangox>rangoy:
        rango = rangox
        longitud = xmax
        desplazamiento = minx
    else:
        rango = rangoy
        longitud = ymax
        desplazamiento = miny
    # escalar

    print(longitud, rango)
    factor = longitud / rango
    print('Factor = ', factor)
    plt.imshow(newimg, cmap=cm.Greys)
    plt.show()
    escalada = escalar(newimg, factor)
    plt.imshow(escalada, cmap=cm.Greys)
    plt.show()
    desplazamiento = int(desplazamiento*factor)
    print(minx, miny, desplazamiento)
    #print(escalada[50])

    if factor>=1:
        despx = len(escalada)
        despy = len(escalada[0])

        #Obtenemos la coordenada en x e y donde inicia la imagen
        for x in range(len(escalada)):
            for y in range(len(escalada[0])):
                if escalada[x,y]>0.5 or escalada[x,y]==True:
                    if x < despx:
                        despx = x
                    if y < despy:
                        despy = y

        #Obtenemos la coordenada en x e y donde finaliza la imagen
        despxx = despx
        despyy = despy
        for x in range(despx, len(escalada)):
            for y in range(despy, len(escalada[0])):
                if escalada[x,y]>0.5 or escalada[x,y]==True:
                    if x > despxx:
                        despxx = x
                    if y > despyy:
                        despyy = y

        print(despx, despy, despxx, despyy)


        #Calculamos el recuadro de la imagen a tomar
        c_ancho = despyy - despy
        c_alto = despxx - despx
        if (c_alto) > (c_ancho):
            despxxx = despx
            despyyy = despy - math.trunc((c_alto - c_ancho) / 2)
        else:
            despxxx = despx - math.trunc((c_ancho - c_alto) / 2)
            despyyy = despy

        print(despxxx, despyyy)

        imgesc = np.zeros((xmax, ymax), dtype=np.float32)
        for x in range(len(imgesc)):
            for y in range(len(imgesc[0])):
                if (x + despxxx) < (len(escalada)) and (y + despyyy) < (len(escalada[0])):
                    imgesc[x,y] = escalada[x + despxxx, y + despyyy]
    else:
        imgesc = escalada

    return imgesc"""
```


*Salida:*
```text
"def escalarImagen(img, xmax, ymax):\n    # obtener los minimos y maximos x e y\n    newimg = np.zeros((xmax, ymax), dtype=np.float32) \n    minx=xmax\n    miny=ymax\n    maxx=0\n    maxy=0\n    for x in range(len(img)):\n        for y in range(len(img[0])):\n            if img[x,y]>0 or img[x,y]==True:\n                newimg[x,y] = float32(1.0)\n            else:\n                newimg[x,y] = float32(0.0)\n            if img[x,y]>0 or img[x,y]==True:\n                if x<minx:\n                    minx = x\n                if x>maxx:\n                    maxx = x\n                if y<miny:\n                    miny = y\n                if y>maxy:\n                    maxy = y\n    rangox = maxx-minx+1\n    rangoy = maxy-miny+1\n    if rangox>rangoy:\n        rango = rangox\n        longitud = xmax\n        desplazamiento = minx\n    else:\n        rango = rangoy\n        longitud = ymax\n        desplazamiento = miny\n    # escalar\n    \n    print(longitud, rango)\n    factor = longitud / rango\n    print('Factor = ', factor)\n    plt.imshow(newimg, cmap=cm.Greys)\n    plt.show()\n    escalada = escalar(newimg, factor)\n    plt.imshow(escalada, cmap=cm.Greys)\n    plt.show()\n    desplazamiento = int(desplazamiento*factor)\n    print(minx, miny, desplazamiento)\n    #print(escalada[50])\n    \n    if factor>=1:\n        despx = len(escalada)\n        despy = len(escalada[0])\n        \n        #Obtenemos la coordenada en x e y donde inicia la imagen\n        for x in range(len(escalada)):\n            for y in range(len(escalada[0])):\n                if escalada[x,y]>0.5 or escalada[x,y]==True:\n                    if x < despx:\n                        despx = x\n                    if y < despy:\n                        despy = y\n        \n        #Obtenemos la coordenada en x e y donde finaliza la imagen\n        despxx = despx\n        despyy = despy\n        for x in range(despx, len(escalada)):\n            for y in range(despy, len(escalada[0])):\n                if escalada[x,y]>0.5 or escalada[x,y]==True:\n                    if x > despxx:\n                        despxx = x\n                    if y > despyy:\n                        despyy = y\n        \n        print(despx, despy, despxx, despyy)\n        \n        \n        #Calculamos el recuadro de la imagen a tomar\n        c_ancho = despyy - despy\n        c_alto = despxx - despx\n        if (c_alto) > (c_ancho):\n            despxxx = despx\n            despyyy = despy - math.trunc((c_alto - c_ancho) / 2)\n        else:\n            despxxx = despx - math.trunc((c_ancho - c_alto) / 2)\n            despyyy = despy\n        \n        print(despxxx, despyyy)\n        \n        imgesc = np.zeros((xmax, ymax), dtype=np.float32)\n        for x in range(len(imgesc)):\n            for y in range(len(imgesc[0])):\n                if (x + despxxx) < (len(escalada)) and (y + despyyy) < (len(escalada[0])):\n                    imgesc[x,y] = escalada[x + despxxx, y + despyyy]\n    else:\n        imgesc = escalada\n          \n    return imgesc"
```


**[Celda 39 - Código]**
```python
def escalarImagenConDeformacion(img, xmax, ymax):

    escalada = img

    despx = len(escalada)
    despy = len(escalada[0])

    #Obtenemos la coordenada en x e y donde inicia la imagen
    for x in range(len(escalada)):
        for y in range(len(escalada[0])):
            if escalada[x,y]>0.5 or escalada[x,y]==True:
                if x < despx:
                    despx = x
                if y < despy:
                    despy = y

    #Obtenemos la coordenada en x e y donde finaliza la imagen
    despxx = despx
    despyy = despy
    for x in range(despx, len(escalada)):
        for y in range(despy, len(escalada[0])):
            if escalada[x,y]>0.5 or escalada[x,y]==True:
                if x > despxx:
                    despxx = x
                if y > despyy:
                    despyy = y

    #print(despx, despy, despxx, despyy)
    #if despx > 0:
    #    despx -= 1
    #if despy > 0:
    #    despy -= 1
    if despxx < len(img) - 1:
        despxx += 1
    if despyy < len(img[0]) - 1:
        despyy += 1
    cropped_image = img[despx:despxx, despy:despyy]
    resized_image = cv2.resize(cropped_image, (xmax, ymax), interpolation=cv2.INTER_LINEAR)

    return resized_image
```


**[Celda 40 - Código]**
```python
def preprocesarImagen(img, xMax, yMax, radio):
    image_gris = img / 255
    image_gris = np.array(image_gris)
    mat = crearMatrizDeImagen(image_gris, xMax, yMax)
    #escalada = escalarImagen(mat, xMax, yMax)
    escalada = escalarImagenConDeformacion(mat, xMax, yMax)
    matesc = binarizar(escalada, xMax, yMax)
    sk = skeletonize(1-matesc)
    #imgcent = centrarImagen(sk, xMax, yMax)
    relieve = relieveSinCuda(sk, radio)
    relieve = normalizarUnitario(relieve)
    #image_gris = escalar(relieve, 0.7)
    #image_gris = relieve.reshape(xMax, xMax, 1)

    # VISUALIZACIONES #
    #print('IMAGEN:')
    #plt.imshow(image_gris, cmap=cm.Greys)
    #plt.show()
    print('MATRIZ DE LA IMAGEN:')
    plt.imshow(mat, cmap=cm.Greys)
    plt.show()
    #print('ESCALADA:')
    #plt.imshow(escalada)
    #plt.show()
    print('ESCALADA CON DEFORMACIÓN:')
    plt.imshow(escalada)
    plt.show()
    #print('BINARIZADA:')
    #plt.imshow(matesc, cmap=cm.Greys)
    #plt.show()
    print('SKELETONIZADA:')
    plt.imshow(sk, cmap=cm.Greys)
    plt.show()
    #print('CENTRADA:')
    #plt.imshow(imgcent)
    #plt.show()
    print('RELIEVE:')
    plt.imshow(relieve, cmap=cm.Greys)
    plt.show()

    return relieve
```


**[Celda 41 - Código]**
```python
# PRUEBA DE PREPROCESAMIENTO
import cv2
#import numpy as np
#from matplotlib import pyplot as plt

#image = r'C:\Users\User\Documents\UTN\GIBD\Similitud de Marcas\BDNormalizada_R(28x28)\1.jpg'
# image = r'C:\Users\User\Documents\UTN\GIBD\Similitud de Marcas\QueriesDibujadas\q_100.jpg'
img = cv2.imread(r'C:\Users\User\Documents\UTN\GIBD\Similitud de Marcas\QueriesDibujadas\q_143.jpg', cv2.IMREAD_GRAYSCALE)
#image = r'C:\Users\User\Documents\UTN\GIBD\Alphabet\Consultas_Alphabet\q_Y.1.jpg'
# img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
# plt.imshow(img, cmap=cm.Greys)
# plt.show()
img5 = img / 255.0
img5 = 1 - img5
#plt.imshow(img5, cmap=cm.Greys)
#plt.show()
#img5 = img5.reshape(28,28,1)
print(img5.shape)

imgp = preprocesarImagen(img, 28, 28, 5)
#plt.imshow(imgp, cmap=cm.Greys)
#plt.show()
```


*Salida:*
```text
(28, 28)
MATRIZ DE LA IMAGEN:
<Figure size 432x288 with 1 Axes>ESCALADA CON DEFORMACIÓN:
<Figure size 432x288 with 1 Axes>SKELETONIZADA:
<Figure size 432x288 with 1 Axes>RELIEVE:
<Figure size 432x288 with 1 Axes>(28, 28)
```


**[Celda 42 - Código]**
```python
# CARGA DE LAS IMAGENES DE LA BASE DE DATOS
from matplotlib import pyplot as plt


directorio = r'C:\Users\User\Documents\UTN\GIBD\Similitud de Marcas\BDNormalizada_R(28x28)'
contenido = os.listdir(directorio)

imagenesBD = []
y_imagenesBD = []
for fichero in contenido:
    if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.jpg'):
    #if os.path.isfile(os.path.join(directorio, fichero)) and (fichero.endswith('.JPG') or fichero.endswith('.jpg')):
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


**[Celda 43 - Código]**
```python
# CARGAR LAS IMAGENES DE LA BASE DE DATOS ORIGINALY APLICARLES RELIEVE
import cv2
import os
#import numpy as np
#from matplotlib import pyplot as plt
from datetime import datetime

date = datetime.now()
print(date)

directorio = r'C:\Users\User\Documents\UTN\GIBD\Similitud de Marcas\BDNormalizada_R(28x28)'
#directorio = r'C:\GIBD\Investigacion\Similitud de Marcas\BDNormalizada(28x28)'
#directorio = r'C:\Users\User\Documents\UTN\GIBD\Alphabet\BD_Alphabet'
contenido = os.listdir(directorio)
# print(contenido)

imagenesBD = []
y_imagenesBD = []
for fichero in contenido:
    if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.jpg'):
    #if os.path.isfile(os.path.join(directorio, fichero)) and (fichero.endswith('.jpg') or fichero.endswith('.JPG')):
        image = directorio + '/' + fichero
        image_gris = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        relieve =preprocesarImagen(image_gris, ANCHO, ALTO, RADIO)
        imagenesBD.append(relieve)
        n = fichero[:fichero.find('.')]
        y_imagenesBD.append(n)
print(y_imagenesBD)

date = datetime.now()
print(date)
```


*Salida:*
```text
2023-05-24 17:56:05.723725
['1', '10', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '11', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '12', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '13', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '14', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '15', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '16', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '17', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '18', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '19', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '2', '20', '200', '201', '202', '203', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']
2023-05-24 17:56:11.084413
```


**[Celda 44 - Código]**
```python
# CARGAR CONSULTAS DIFICILES
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

directorio = r'C:\GIBD\Investigacion\Similitud de Marcas\Queries_R(28x28)'
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


**[Celda 45 - Código]**
```python
# CARGAR CONSULTAS DIBUJADAS A MANO
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

directorio = r'C:\Users\User\Documents\UTN\GIBD\Similitud de Marcas\QueriesDibujadas'

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
        n = n[n.find('_')+1:]
        y_consultas.append(n)
print(y_consultas)
```


*Salida:*
```text
['100', '107', '115', '118', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '134', '143', '144', '145', '146', '166', '170', '174', '175', '192', '195', '197', '2', '24', '31', '32', '37', '47', '55', '58', '59', '7', '79', '8', '88', '94']
```


**[Celda 46 - Código]**
```python
# CARGAR CONSULTAS DIBUJADAS A MANO Y APLICARLES RELIEVE
import cv2
import os
#import numpy as np
#from matplotlib import pyplot as plt
from datetime import datetime

#import tensorflow.compat.v1 as tf1
#tf1.disable_v2_behavior()

date = datetime.now()
print(date)

directorio = r'C:\Users\User\Documents\UTN\GIBD\Similitud de Marcas\QueriesDibujadas'
#directorio = r'C:\Users\User\Documents\UTN\GIBD\Alphabet\Consultas_Alphabet'
#directorio = r'C:\GIBD\Investigacion\Similitud de Marcas\QueriesDibujadas'
contenido = os.listdir(directorio)

consultas = []
y_consultas = []
for fichero in contenido:
    if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.jpg'):
        image = directorio + '/' + fichero
        image_gris = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        relieve =preprocesarImagen(image_gris, ANCHO, ALTO, RADIO)
        consultas.append(relieve)
        n = fichero[:fichero.find('.')]
        n = n[n.find('_')+1:]
        y_consultas.append(n)
print(y_consultas)

date = datetime.now()
print(date)
```


*Salida:*
```text
2023-05-23 18:05:47.567382
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 25
Factor =  1.12
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>2 5 2
2 6 29 22
2 1
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 4 0
0 4 27 22
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 24
Factor =  1.1666666666666667
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>2 6 2
2 7 29 23
2 2
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 4 0
0 4 27 22
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 23
Factor =  1.2173913043478262
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>4 3 3
5 4 31 30
5 4
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 0 0
0 0 26 27
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 21
Factor =  1.3333333333333333
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>4 4 5
6 5 32 32
6 5
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 0 0
0 0 27 27
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 25
Factor =  1.12
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>2 2 2
2 2 29 22
2 -1
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 3 0
0 3 27 24
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 24
Factor =  1.1666666666666667
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>3 1 3
3 1 30 23
3 -1
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 2 0
0 2 27 25
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 26
Factor =  1.0769230769230769
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>1 1 1
1 1 28 26
1 0
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 1 0
0 1 27 26
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 22
Factor =  1.2727272727272727
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>3 3 3
4 4 30 26
4 2
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 1 0
0 1 27 25
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 27
Factor =  1.037037037037037
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 1 0
0 1 27 24
0 -1
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 2 0
0 2 27 25
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 25
Factor =  1.12
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>4 2 2
4 2 27 29
2 2
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>2 0 0
2 0 25 27
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 25
Factor =  1.12
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>1 0 0
1 0 28 27
1 0
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 0 0
0 0 27 27
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 2 0
0 2 27 23
0 -1
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 3 0
0 3 27 24
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 25
Factor =  1.12
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>2 6 2
2 7 29 22
2 1
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 5 0
0 5 27 22
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 26
Factor =  1.0769230769230769
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>1 4 1
1 4 28 21
1 -1
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 5 0
0 5 27 23
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 24
Factor =  1.1666666666666667
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>2 3 2
2 3 29 26
2 1
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 2 0
0 2 27 26
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 23
Factor =  1.2173913043478262
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>2 7 2
3 9 29 26
3 5
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 3 0
0 3 27 21
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 26
Factor =  1.0769230769230769
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>3 1 1
3 1 26 28
1 1
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>2 0 0
2 0 25 27
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 23
Factor =  1.2173913043478262
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>9 2 2
11 2 23 29
4 2
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>6 0 0
6 0 20 27
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 21
Factor =  1.3333333333333333
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>3 7 4
4 9 31 29
4 6
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 3 0
0 3 27 24
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 24
Factor =  1.1666666666666667
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>3 2 2
3 2 25 29
1 2
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>2 0 0
2 0 25 27
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 22
Factor =  1.2727272727272727
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>3 8 3
4 10 30 25
4 5
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 4 0
0 4 27 21
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 18
Factor =  1.5555555555555556
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>5 8 7
8 12 34 30
8 8
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 4 0
0 4 27 22
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 24
Factor =  1.1666666666666667
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>2 7 2
2 8 29 25
2 3
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 4 0
0 4 27 23
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 27
Factor =  1.037037037037037
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 7 0
0 7 27 21
0 1
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 6 0
0 6 27 20
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 24
Factor =  1.1666666666666667
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>2 5 2
2 6 29 26
2 3
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 2 0
0 2 27 24
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 22
Factor =  1.2727272727272727
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>5 4 5
6 5 30 31
5 5
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>1 0 0
1 0 26 27
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 22
Factor =  1.2727272727272727
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>9 3 3
11 4 24 30
5 4
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>6 0 0
6 0 20 27
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 24
Factor =  1.1666666666666667
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>3 3 3
3 3 29 30
3 3
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 0 0
0 0 27 27
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 24
Factor =  1.1666666666666667
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>2 7 2
2 8 29 25
2 3
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 4 0
0 4 27 23
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 27
Factor =  1.037037037037037
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>1 0 0
1 0 25 27
0 0
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>1 0 0
1 0 25 27
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 27
Factor =  1.037037037037037
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>1 2 1
1 2 28 27
1 1
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 1 0
0 1 27 26
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 22
Factor =  1.2727272727272727
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>3 5 3
4 6 30 26
4 3
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 3 0
0 3 27 24
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 22
Factor =  1.2727272727272727
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>3 5 3
4 6 30 30
4 5
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 1 0
0 1 27 26
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>1 0 0
1 0 21 27
-2 0
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>3 0 0
3 0 23 27
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 27
Factor =  1.037037037037037
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>2 1 1
2 1 26 28
1 1
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>1 0 0
1 0 25 27
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 23
Factor =  1.2173913043478262
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>3 7 3
4 9 31 22
4 2
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 6 0
0 6 27 21
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 23
Factor =  1.2173913043478262
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>2 4 2
2 5 29 31
2 5
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 0 0
0 0 27 26
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 21
Factor =  1.3333333333333333
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>4 3 4
5 4 32 31
5 4
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>0 0 0
0 0 27 27
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>2 0 0
2 0 22 27
-1 0
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>3 0 0
3 0 23 27
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>4 0 0
4 0 21 27
-1 0
<Figure size 432x288 with 1 Axes>28 28
Factor =  1.0
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>5 0 0
5 0 22 27
DEFORMADA:
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
SKELETONIZADA:
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>['100', '107', '115', '118', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '134', '143', '144', '145', '146', '166', '170', '174', '175', '192', '195', '197', '2', '24', '31', '32', '37', '47', '55', '58', '59', '7', '79', '8', '88', '94']
2023-05-23 18:07:03.463384
```


**[Celda 47 - Código]**
```python
# CARGAR CONSULTAS -----NUEVAS----- DIBUJADAS A MANO Y APLICARLES RELIEVE
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

#import tensorflow.compat.v1 as tf1
#tf1.disable_v2_behavior()

#date = datetime.now()
#print(date)

directorio = r'C:\Users\User\Documents\UTN\GIBD\Similitud de Marcas\NuevasMarcasQueries'
contenido = os.listdir(directorio)

consultas = []
y_consultas = []
for fichero in contenido:
    if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.jpg'):
        image = directorio + '/' + fichero
        image_gris = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        relieve =preprocesarImagen(image_gris, ANCHO, ALTO, RADIO)
        consultas.append(relieve)
        n = fichero[:fichero.find('.')]
        n = n[n.find('-')+1:]
        y_consultas.append(n)
print(y_consultas)

#date = datetime.now()
#print(date)
```


*Salida:*
```text
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 26
Factor =  1.0769230769230769
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>5 1 1
5 1 15 28
-3 1
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 22
Factor =  1.2727272727272727
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>4 4 5
5 5 29 31
4 5
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 21
Factor =  1.3333333333333333
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>4 7 5
5 9 32 32
5 7
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 24
Factor =  1.1666666666666667
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>6 3 3
7 3 24 30
2 3
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 24
Factor =  1.1666666666666667
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>3 6 3
3 7 30 24
3 2
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 24
Factor =  1.1666666666666667
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>4 1 1
5 1 26 28
2 1
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 22
Factor =  1.2727272727272727
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>4 1 1
5 1 30 28
4 1
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 23
Factor =  1.2173913043478262
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>4 2 2
5 2 26 29
2 2
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 22
Factor =  1.2727272727272727
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>5 2 2
6 3 30 29
5 3
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>28 22
Factor =  1.2727272727272727
<Figure size 432x288 with 1 Axes><Figure size 432x288 with 1 Axes>4 3 3
5 4 30 30
5 4
<Figure size 432x288 with 1 Axes>BINARIZADA:
<Figure size 432x288 with 1 Axes><class 'numpy.bool_'>
<Figure size 432x288 with 1 Axes>['301', '310', '302', '303', '304', '305', '306', '307', '308', '309']
```


**[Celda 48 - Código]**
```python
def nnk(actuales,nuevo,k=1):
    p = 0
    while (p<len(actuales) and (nuevo[0]>=actuales[p][0])):
        p += 1
    actuales.insert(p, nuevo)
    while len(actuales)>k:
        actuales.pop(len(actuales)-1)
```


**[Celda 49 - Código]**
```python
def nnk2etiquetas(nnklist, y_BD):
    l = []
    for par in nnklist:
        nuevo = [par[0], y_BD[par[1]]]
        l.append(nuevo)
    return l
```


**[Celda 50 - Código]**
```python
# CONSULTAS POR SIMILITUD UTILIZANDO LA BASE DE CONSULTAS DIFICILES

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
    nnklist = []
    # Inicio del for DB
    for j in range(cantidad_BD):
        vector_elemento = model.predict(imagenesBD[j].reshape(1,28,28,1))
        b = np.array(vector_elemento[0])
        #dist = np.sqrt(np.sum(np.square(a-b)))
        dist = np.sum(np.abs(a-b))
        nnk(nnklist,[dist,j],10)
        if dist < min:
            min = dist
            vector_cercano = b
            posicion_cercano = j

    print("Distancia: ", min, " al vector en la posicion: ", posicion_cercano, " desde el vector que se encuentra en: ", i)
    print("Imagen de BD: ", y_imagenesBD[posicion_cercano], " Imagen de consulta: ", y_consultas[i])
    print('Mas cercanos: ',nnklist)
    for par in nnklist:
        if y_imagenesBD[par[1]] == y_consultas[i]:
            contador_aciertos += 1
            print('--> ACIERTO')
    print("\n")

print("Porcentaje de aciertos: ", contador_aciertos/cantidad_consultas)
```


*Salida:*
```text
Distancia:  25.58404  al vector en la posicion:  151  desde el vector que se encuentra en:  0
Imagen de BD:  52  Imagen de consulta:  197
Mas cercanos:  [[25.58404, 151], [26.801323, 184], [31.918922, 98], [32.56264, 50], [34.6362, 65], [36.31517, 153], [36.49797, 103], [37.063553, 46], [37.474197, 54], [38.08366, 48]]


Distancia:  26.416725  al vector en la posicion:  184  desde el vector que se encuentra en:  1
Imagen de BD:  82  Imagen de consulta:  11
Mas cercanos:  [[26.416725, 184], [26.475067, 50], [27.910236, 151], [30.591928, 103], [31.022715, 48], [31.343235, 76], [31.684406, 98], [32.580517, 54], [33.292553, 88], [34.171593, 153]]


Distancia:  25.425098  al vector en la posicion:  76  desde el vector que se encuentra en:  2
Imagen de BD:  168  Imagen de consulta:  14
Mas cercanos:  [[25.425098, 76], [26.379871, 151], [28.59275, 103], [30.245497, 153], [32.83477, 54], [34.458992, 48], [34.470284, 50], [34.745613, 156], [37.276043, 113], [37.318634, 184]]


Distancia:  16.465944  al vector en la posicion:  50  desde el vector que se encuentra en:  3
Imagen de BD:  144  Imagen de consulta:  16
Mas cercanos:  [[16.465944, 50], [22.65721, 184], [24.270504, 98], [25.219837, 160], [28.381178, 156], [29.939682, 48], [30.087036, 91], [30.607487, 88], [32.286564, 53], [32.422012, 55]]


Distancia:  22.093937  al vector en la posicion:  50  desde el vector que se encuentra en:  4
Imagen de BD:  144  Imagen de consulta:  18
Mas cercanos:  [[22.093937, 50], [27.421133, 151], [29.499672, 184], [32.292442, 76], [32.481224, 98], [33.80883, 103], [34.7016, 156], [35.61989, 53], [35.75319, 48], [35.946766, 160]]


Distancia:  14.643464  al vector en la posicion:  50  desde el vector que se encuentra en:  5
Imagen de BD:  144  Imagen de consulta:  19
Mas cercanos:  [[14.643464, 50], [22.359547, 184], [25.457504, 98], [28.23608, 160], [28.719757, 156], [28.795267, 46], [31.587177, 55], [31.941973, 48], [32.03826, 91], [32.160362, 42]]


Distancia:  22.084217  al vector en la posicion:  151  desde el vector que se encuentra en:  6
Imagen de BD:  52  Imagen de consulta:  24
Mas cercanos:  [[22.084217, 151], [28.528282, 50], [29.517048, 184], [31.179615, 98], [35.808002, 46], [37.83084, 54], [37.84556, 103], [37.856133, 76], [38.08272, 65], [38.38292, 48]]


Distancia:  22.599287  al vector en la posicion:  98  desde el vector que se encuentra en:  7
Imagen de BD:  188  Imagen de consulta:  24
Mas cercanos:  [[22.599287, 98], [24.960869, 184], [28.065603, 50], [28.980991, 46], [31.354908, 86], [32.033382, 48], [32.76628, 65], [33.967274, 55], [34.0319, 151], [34.389954, 88]]


Distancia:  20.248428  al vector en la posicion:  50  desde el vector que se encuentra en:  8
Imagen de BD:  144  Imagen de consulta:  27
Mas cercanos:  [[20.248428, 50], [21.68509, 184], [24.660732, 98], [29.061, 48], [29.222807, 151], [29.985863, 94], [31.931007, 46], [33.017807, 160], [33.457813, 55], [33.60845, 156]]


Distancia:  22.829386  al vector en la posicion:  160  desde el vector que se encuentra en:  9
Imagen de BD:  60  Imagen de consulta:  33
Mas cercanos:  [[22.829386, 160], [26.345432, 50], [28.464804, 184], [31.086765, 98], [34.80725, 47], [34.95186, 92], [35.224796, 79], [36.032898, 96], [36.1034, 91], [37.617584, 46]]


Distancia:  26.670242  al vector en la posicion:  50  desde el vector que se encuentra en:  10
Imagen de BD:  144  Imagen de consulta:  36
Mas cercanos:  [[26.670242, 50], [31.559364, 103], [32.176273, 156], [32.397987, 98], [32.54324, 42], [32.801125, 160], [33.916904, 53], [34.678833, 76], [35.09718, 184], [35.76729, 55]]


Distancia:  18.535782  al vector en la posicion:  50  desde el vector que se encuentra en:  11
Imagen de BD:  144  Imagen de consulta:  5
Mas cercanos:  [[18.535782, 50], [22.77426, 184], [23.187714, 98], [23.640486, 156], [24.985731, 88], [26.585491, 160], [26.882242, 91], [27.076115, 46], [27.551521, 48], [29.731806, 42]]


Distancia:  28.153484  al vector en la posicion:  50  desde el vector que se encuentra en:  12
Imagen de BD:  144  Imagen de consulta:  36
Mas cercanos:  [[28.153484, 50], [29.253246, 103], [29.878656, 76], [31.109901, 151], [32.195282, 98], [33.668133, 153], [33.711258, 184], [33.759636, 54], [34.12038, 156], [35.155952, 88]]


Distancia:  23.90936  al vector en la posicion:  184  desde el vector que se encuentra en:  13
Imagen de BD:  82  Imagen de consulta:  44
Mas cercanos:  [[23.90936, 184], [23.926243, 50], [27.186127, 98], [28.253786, 48], [29.075047, 156], [30.972591, 88], [31.026398, 151], [32.489033, 46], [32.673874, 103], [34.01512, 153]]


Distancia:  23.157953  al vector en la posicion:  151  desde el vector que se encuentra en:  14
Imagen de BD:  52  Imagen de consulta:  45
Mas cercanos:  [[23.157953, 151], [34.704525, 76], [35.477978, 184], [35.71987, 103], [36.851475, 50], [37.032658, 65], [37.558, 84], [37.84381, 77], [38.55405, 98], [38.664146, 46]]


Distancia:  24.615068  al vector en la posicion:  151  desde el vector que se encuentra en:  15
Imagen de BD:  52  Imagen de consulta:  46
Mas cercanos:  [[24.615068, 151], [29.367218, 65], [29.829435, 98], [30.460796, 184], [34.593914, 103], [34.99515, 46], [35.076183, 163], [35.081486, 86], [35.128387, 50], [35.320747, 153]]


Distancia:  25.268532  al vector en la posicion:  86  desde el vector que se encuentra en:  16
Imagen de BD:  177  Imagen de consulta:  47
Mas cercanos:  [[25.268532, 86], [27.265545, 153], [27.875576, 48], [28.13336, 103], [28.433962, 156], [29.030996, 157], [30.031466, 84], [30.779465, 116], [31.324825, 88], [31.41822, 77]]


Distancia:  24.237335  al vector en la posicion:  151  desde el vector que se encuentra en:  17
Imagen de BD:  52  Imagen de consulta:  47
Mas cercanos:  [[24.237335, 151], [31.68544, 98], [32.62396, 184], [35.464554, 76], [35.5702, 50], [36.079605, 103], [38.172478, 84], [38.689964, 156], [38.85278, 46], [39.22702, 48]]


Distancia:  25.579437  al vector en la posicion:  98  desde el vector que se encuentra en:  18
Imagen de BD:  188  Imagen de consulta:  47
Mas cercanos:  [[25.579437, 98], [26.966335, 184], [27.360428, 50], [29.340912, 156], [30.635853, 46], [30.640911, 103], [31.50124, 151], [31.744766, 160], [32.57777, 55], [33.589787, 48]]


Distancia:  25.65786  al vector en la posicion:  88  desde el vector que se encuentra en:  19
Imagen de BD:  179  Imagen de consulta:  49
Mas cercanos:  [[25.65786, 88], [27.442402, 103], [27.589687, 50], [28.655525, 156], [29.790724, 48], [30.273281, 42], [30.790867, 184], [31.783833, 98], [32.44963, 46], [32.450268, 53]]


Distancia:  24.653778  al vector en la posicion:  71  desde el vector que se encuentra en:  20
Imagen de BD:  163  Imagen de consulta:  51
Mas cercanos:  [[24.653778, 71], [24.88227, 46], [24.957573, 81], [25.23821, 106], [26.35567, 55], [27.394436, 156], [28.038235, 91], [28.732012, 98], [28.971333, 160], [30.801489, 48]]


Distancia:  27.236698  al vector en la posicion:  76  desde el vector que se encuentra en:  21
Imagen de BD:  168  Imagen de consulta:  53
Mas cercanos:  [[27.236698, 76], [27.35212, 103], [27.484482, 151], [29.707626, 50], [29.86685, 54], [32.33848, 153], [33.270363, 48], [34.183647, 184], [35.87136, 94], [36.42726, 156]]


Distancia:  27.406132  al vector en la posicion:  83  desde el vector que se encuentra en:  22
Imagen de BD:  174  Imagen de consulta:  7
Mas cercanos:  [[27.406132, 83], [27.880266, 62], [28.57722, 74], [29.78237, 113], [32.455765, 38], [34.999073, 52], [35.247948, 116], [35.506836, 84], [35.73603, 76], [35.957222, 200]]


Distancia:  25.697788  al vector en la posicion:  153  desde el vector que se encuentra en:  23
Imagen de BD:  54  Imagen de consulta:  54
Mas cercanos:  [[25.697788, 153], [29.501476, 151], [33.649517, 103], [33.87669, 76], [34.05972, 65], [35.471516, 84], [35.483612, 48], [36.8795, 184], [37.384354, 163], [37.832672, 54]]
--> ACIERTO


Distancia:  27.954391  al vector en la posicion:  184  desde el vector que se encuentra en:  24
Imagen de BD:  82  Imagen de consulta:  56
Mas cercanos:  [[27.954391, 184], [28.8224, 151], [29.541254, 98], [30.106232, 50], [35.20968, 65], [35.485092, 46], [36.03163, 76], [36.809605, 55], [37.328693, 160], [37.51331, 48]]


Distancia:  21.974716  al vector en la posicion:  163  desde el vector que se encuentra en:  25
Imagen de BD:  63  Imagen de consulta:  63
Mas cercanos:  [[21.974716, 163], [29.948265, 151], [33.577244, 98], [34.924103, 86], [35.083466, 65], [35.784164, 153], [36.456238, 46], [36.985703, 77], [38.13222, 157], [38.697876, 184]]
--> ACIERTO


Distancia:  27.157726  al vector en la posicion:  50  desde el vector que se encuentra en:  26
Imagen de BD:  144  Imagen de consulta:  66
Mas cercanos:  [[27.157726, 50], [27.443062, 98], [27.665857, 184], [31.44989, 55], [31.460125, 46], [31.774437, 151], [33.42531, 156], [33.766235, 105], [34.41383, 91], [34.9663, 42]]


Distancia:  37.623283  al vector en la posicion:  171  desde el vector que se encuentra en:  27
Imagen de BD:  70  Imagen de consulta:  66
Mas cercanos:  [[37.623283, 171], [38.246178, 70], [40.444504, 43], [40.937363, 105], [40.96951, 174], [42.08589, 54], [42.13133, 89], [42.26156, 73], [42.49086, 152], [43.085457, 63]]


Distancia:  31.16416  al vector en la posicion:  50  desde el vector que se encuentra en:  28
Imagen de BD:  144  Imagen de consulta:  66
Mas cercanos:  [[31.16416, 50], [35.270557, 88], [35.646736, 153], [36.230835, 54], [36.256355, 98], [36.399326, 53], [36.489536, 184], [36.850456, 151], [37.099182, 43], [37.52204, 76]]


Distancia:  27.81534  al vector en la posicion:  50  desde el vector que se encuentra en:  29
Imagen de BD:  144  Imagen de consulta:  76
Mas cercanos:  [[27.81534, 50], [28.764545, 98], [29.032885, 48], [29.47286, 151], [30.066792, 156], [30.309975, 76], [30.548422, 103], [31.214565, 184], [32.591087, 54], [33.485317, 88]]


Distancia:  24.413021  al vector en la posicion:  184  desde el vector que se encuentra en:  30
Imagen de BD:  82  Imagen de consulta:  77
Mas cercanos:  [[24.413021, 184], [27.973747, 98], [28.214218, 48], [29.28201, 50], [30.318916, 103], [30.422142, 156], [30.69285, 88], [30.814808, 76], [30.882242, 151], [31.74419, 153]]


Distancia:  23.922583  al vector en la posicion:  86  desde el vector que se encuentra en:  31
Imagen de BD:  177  Imagen de consulta:  78
Mas cercanos:  [[23.922583, 86], [26.343191, 55], [26.376295, 77], [26.57866, 103], [26.682629, 91], [26.927944, 156], [28.054865, 46], [29.896473, 48], [30.268476, 98], [31.17483, 88]]


Distancia:  16.85049  al vector en la posicion:  151  desde el vector que se encuentra en:  32
Imagen de BD:  52  Imagen de consulta:  80
Mas cercanos:  [[16.85049, 151], [24.39825, 184], [27.278128, 50], [29.671139, 98], [34.226196, 48], [34.688095, 46], [37.457424, 54], [38.4205, 94], [38.4876, 65], [38.74607, 76]]


Distancia:  27.720024  al vector en la posicion:  151  desde el vector que se encuentra en:  33
Imagen de BD:  52  Imagen de consulta:  11
Mas cercanos:  [[27.720024, 151], [31.855478, 76], [32.362297, 50], [32.6528, 103], [33.067787, 98], [33.305176, 184], [34.031467, 54], [35.334534, 156], [35.972736, 48], [36.810913, 65]]


Distancia:  21.41904  al vector en la posicion:  50  desde el vector que se encuentra en:  34
Imagen de BD:  144  Imagen de consulta:  83
Mas cercanos:  [[21.41904, 50], [25.512794, 184], [27.624523, 98], [28.352877, 156], [29.041363, 48], [29.139233, 103], [30.793602, 88], [31.614452, 151], [31.6479, 160], [32.384323, 53]]


Distancia:  29.7718  al vector en la posicion:  98  desde el vector que se encuentra en:  35
Imagen de BD:  188  Imagen de consulta:  87
Mas cercanos:  [[29.7718, 98], [30.029716, 151], [32.082794, 50], [32.86631, 184], [37.108696, 65], [38.02014, 46], [38.94769, 48], [39.626705, 54], [40.030422, 76], [41.016834, 103]]


Distancia:  20.24828  al vector en la posicion:  50  desde el vector que se encuentra en:  36
Imagen de BD:  144  Imagen de consulta:  93
Mas cercanos:  [[20.24828, 50], [20.65613, 98], [20.996294, 160], [23.964748, 184], [28.10586, 55], [28.876894, 88], [29.827774, 46], [30.372478, 156], [30.421549, 91], [30.538927, 48]]


Distancia:  25.99466  al vector en la posicion:  184  desde el vector que se encuentra en:  37
Imagen de BD:  82  Imagen de consulta:  95
Mas cercanos:  [[25.99466, 184], [26.344032, 98], [26.640114, 50], [27.945362, 151], [29.785028, 46], [31.771084, 160], [32.340736, 88], [32.606644, 156], [32.66313, 103], [34.023193, 48]]


Distancia:  27.444668  al vector en la posicion:  151  desde el vector que se encuentra en:  38
Imagen de BD:  52  Imagen de consulta:  100
Mas cercanos:  [[27.444668, 151], [28.417364, 65], [29.664156, 98], [30.166008, 184], [31.054018, 50], [35.07753, 103], [35.26715, 84], [35.435425, 153], [35.901123, 76], [37.42782, 53]]


Distancia:  26.495104  al vector en la posicion:  184  desde el vector que se encuentra en:  39
Imagen de BD:  82  Imagen de consulta:  103
Mas cercanos:  [[26.495104, 184], [27.032963, 98], [28.04533, 50], [28.583174, 151], [33.977936, 65], [34.350746, 46], [36.288963, 55], [36.65294, 48], [36.711155, 103], [36.811314, 160]]


Distancia:  20.067078  al vector en la posicion:  98  desde el vector que se encuentra en:  40
Imagen de BD:  188  Imagen de consulta:  107
Mas cercanos:  [[20.067078, 98], [22.374294, 184], [24.217323, 50], [27.737389, 46], [28.517632, 160], [29.410275, 48], [30.044178, 156], [30.84005, 88], [31.03764, 151], [31.911715, 55]]


Distancia:  20.94106  al vector en la posicion:  50  desde el vector que se encuentra en:  41
Imagen de BD:  144  Imagen de consulta:  112
Mas cercanos:  [[20.94106, 50], [23.581654, 184], [25.093477, 98], [25.67549, 160], [27.307135, 46], [30.459276, 91], [30.843262, 55], [31.147379, 156], [31.151323, 48], [31.16682, 94]]


Distancia:  24.98009  al vector en la posicion:  103  desde el vector que se encuentra en:  42
Imagen de BD:  192  Imagen de consulta:  113
Mas cercanos:  [[24.98009, 103], [26.738682, 153], [28.44614, 76], [28.783886, 54], [32.161087, 65], [32.890244, 84], [33.427788, 88], [33.535954, 151], [34.790764, 86], [34.856422, 48]]


Distancia:  28.37332  al vector en la posicion:  76  desde el vector que se encuentra en:  43
Imagen de BD:  168  Imagen de consulta:  115
Mas cercanos:  [[28.37332, 76], [28.593079, 54], [29.762856, 153], [30.684097, 103], [33.56338, 151], [33.665234, 84], [35.496094, 156], [36.78994, 88], [36.93042, 113], [38.199604, 48]]


Distancia:  24.320656  al vector en la posicion:  151  desde el vector que se encuentra en:  44
Imagen de BD:  52  Imagen de consulta:  11
Mas cercanos:  [[24.320656, 151], [27.872807, 184], [28.553436, 50], [29.675201, 98], [33.682236, 76], [34.536415, 65], [34.905746, 103], [34.99163, 46], [36.045723, 48], [36.2343, 153]]


Distancia:  28.711004  al vector en la posicion:  50  desde el vector que se encuentra en:  45
Imagen de BD:  144  Imagen de consulta:  0
Mas cercanos:  [[28.711004, 50], [29.85873, 88], [30.236935, 98], [31.175987, 184], [31.357643, 55], [31.70423, 42], [32.075657, 48], [32.66209, 46], [32.671562, 103], [33.598686, 86]]


Distancia:  22.712921  al vector en la posicion:  55  desde el vector que se encuentra en:  46
Imagen de BD:  149  Imagen de consulta:  120
Mas cercanos:  [[22.712921, 55], [25.933285, 91], [26.113506, 98], [26.385975, 160], [28.641632, 46], [28.935879, 81], [29.99139, 71], [30.304432, 50], [30.836565, 184], [30.88037, 48]]


Distancia:  21.125729  al vector en la posicion:  50  desde el vector que se encuentra en:  47
Imagen de BD:  144  Imagen de consulta:  122
Mas cercanos:  [[21.125729, 50], [25.918575, 160], [26.715855, 184], [26.725117, 98], [30.68831, 156], [31.147434, 88], [32.489, 48], [33.291428, 42], [34.36753, 91], [35.776875, 46]]


Distancia:  27.382069  al vector en la posicion:  103  desde el vector que se encuentra en:  48
Imagen de BD:  192  Imagen de consulta:  125
Mas cercanos:  [[27.382069, 103], [28.3479, 151], [29.198772, 76], [30.098305, 50], [30.203003, 156], [31.554192, 98], [31.593384, 84], [34.253704, 184], [34.412743, 48], [35.148808, 105]]


Distancia:  30.149136  al vector en la posicion:  151  desde el vector que se encuentra en:  49
Imagen de BD:  52  Imagen de consulta:  139
Mas cercanos:  [[30.149136, 151], [31.869394, 103], [32.670307, 156], [32.718243, 76], [32.844734, 48], [34.342957, 98], [35.256638, 84], [35.878315, 77], [36.3829, 44], [36.617905, 184]]
--> ACIERTO


Distancia:  19.483524  al vector en la posicion:  50  desde el vector que se encuentra en:  50
Imagen de BD:  144  Imagen de consulta:  142
Mas cercanos:  [[19.483524, 50], [26.336227, 184], [28.405716, 98], [30.004524, 156], [30.631365, 103], [31.804585, 94], [32.229256, 151], [32.623882, 48], [33.044846, 160], [33.095955, 46]]
--> ACIERTO


Distancia:  22.546305  al vector en la posicion:  50  desde el vector que se encuentra en:  51
Imagen de BD:  144  Imagen de consulta:  142
Mas cercanos:  [[22.546305, 50], [25.991974, 184], [29.01929, 98], [32.060028, 103], [32.175903, 151], [33.026253, 76], [33.31231, 48], [35.103813, 156], [35.38271, 94], [35.45264, 160]]
--> ACIERTO


Distancia:  19.850327  al vector en la posicion:  98  desde el vector que se encuentra en:  52
Imagen de BD:  188  Imagen de consulta:  2
Mas cercanos:  [[19.850327, 98], [21.102478, 50], [22.504383, 184], [23.118494, 91], [23.897682, 46], [25.114202, 160], [25.964083, 55], [30.400105, 48], [32.241028, 156], [32.65251, 86]]


Distancia:  12.672409  al vector en la posicion:  50  desde el vector que se encuentra en:  53
Imagen de BD:  144  Imagen de consulta:  144
Mas cercanos:  [[12.672409, 50], [17.986143, 184], [25.361874, 98], [27.063164, 160], [28.80877, 48], [30.13979, 94], [32.46894, 156], [32.9136, 46], [33.620766, 96], [34.189335, 79]]
--> ACIERTO


Distancia:  24.443174  al vector en la posicion:  50  desde el vector que se encuentra en:  54
Imagen de BD:  144  Imagen de consulta:  147
Mas cercanos:  [[24.443174, 50], [25.462132, 53], [29.399256, 98], [31.25235, 184], [33.436455, 156], [34.0813, 151], [34.547455, 55], [34.656097, 103], [34.91554, 160], [35.221867, 48]]
--> ACIERTO


Distancia:  28.741205  al vector en la posicion:  151  desde el vector que se encuentra en:  55
Imagen de BD:  52  Imagen de consulta:  11
Mas cercanos:  [[28.741205, 151], [30.32381, 50], [30.688362, 184], [30.97888, 76], [31.51182, 98], [32.922646, 103], [33.551994, 156], [34.21095, 54], [34.650543, 48], [37.365944, 153]]


Distancia:  25.646229  al vector en la posicion:  103  desde el vector que se encuentra en:  56
Imagen de BD:  192  Imagen de consulta:  148
Mas cercanos:  [[25.646229, 103], [26.651367, 76], [28.043278, 50], [29.367779, 54], [29.994835, 151], [31.506048, 48], [32.0242, 156], [33.39, 184], [33.891075, 153], [35.347366, 84]]
--> ACIERTO


Distancia:  25.69035  al vector en la posicion:  51  desde el vector que se encuentra en:  57
Imagen de BD:  145  Imagen de consulta:  150
Mas cercanos:  [[25.69035, 51], [30.837267, 175], [31.009338, 63], [32.229855, 46], [32.474384, 64], [32.756824, 57], [33.36058, 105], [33.439495, 55], [34.16721, 111], [34.37751, 104]]
--> ACIERTO


Distancia:  16.530159  al vector en la posicion:  50  desde el vector que se encuentra en:  58
Imagen de BD:  144  Imagen de consulta:  156
Mas cercanos:  [[16.530159, 50], [18.434666, 184], [23.040741, 98], [26.613693, 160], [30.430115, 46], [30.453335, 156], [31.84401, 48], [33.981834, 55], [34.34781, 81], [34.55928, 94]]


Distancia:  20.294682  al vector en la posicion:  98  desde el vector que se encuentra en:  59
Imagen de BD:  188  Imagen de consulta:  157
Mas cercanos:  [[20.294682, 98], [21.761543, 50], [23.300442, 184], [25.740982, 160], [27.185947, 46], [27.92892, 55], [28.669697, 156], [30.009573, 91], [30.08705, 48], [30.544245, 88]]


Distancia:  17.65754  al vector en la posicion:  50  desde el vector que se encuentra en:  60
Imagen de BD:  144  Imagen de consulta:  157
Mas cercanos:  [[17.65754, 50], [21.876211, 184], [25.06987, 98], [25.150879, 160], [31.461079, 88], [32.483818, 94], [33.150715, 91], [33.356834, 46], [33.736404, 156], [34.75782, 42]]


Distancia:  25.093313  al vector en la posicion:  65  desde el vector que se encuentra en:  61
Imagen de BD:  158  Imagen de consulta:  158
Mas cercanos:  [[25.093313, 65], [28.772171, 163], [31.938168, 137], [33.358818, 151], [37.10593, 153], [37.21128, 86], [38.161793, 84], [38.292625, 157], [38.55883, 77], [39.138878, 60]]
--> ACIERTO


Distancia:  26.614742  al vector en la posicion:  50  desde el vector que se encuentra en:  62
Imagen de BD:  144  Imagen de consulta:  158
Mas cercanos:  [[26.614742, 50], [26.834732, 98], [27.58398, 184], [27.65072, 65], [30.342285, 88], [31.51408, 160], [32.097683, 46], [32.33591, 91], [32.87791, 156], [33.33483, 48]]
--> ACIERTO


Distancia:  26.239038  al vector en la posicion:  98  desde el vector que se encuentra en:  63
Imagen de BD:  188  Imagen de consulta:  159
Mas cercanos:  [[26.239038, 98], [26.367466, 160], [26.61874, 50], [26.878475, 184], [28.187832, 48], [28.512531, 46], [28.915043, 156], [29.301197, 91], [31.358372, 88], [31.833282, 55]]


Distancia:  32.84424  al vector en la posicion:  77  desde el vector que se encuentra en:  64
Imagen de BD:  169  Imagen de consulta:  161
Mas cercanos:  [[32.84424, 77], [33.4591, 71], [35.589108, 157], [36.64872, 86], [37.00524, 175], [37.116905, 156], [37.195827, 46], [37.907505, 72], [38.926674, 48], [39.361942, 98]]


Distancia:  28.269354  al vector en la posicion:  70  desde el vector que se encuentra en:  65
Imagen de BD:  162  Imagen de consulta:  162
Mas cercanos:  [[28.269354, 70], [28.973074, 53], [29.363234, 50], [31.027328, 54], [31.84727, 94], [33.3301, 174], [33.819298, 184], [34.00655, 96], [34.027298, 151], [34.987144, 48]]
--> ACIERTO


Distancia:  26.496696  al vector en la posicion:  50  desde el vector que se encuentra en:  66
Imagen de BD:  144  Imagen de consulta:  11
Mas cercanos:  [[26.496696, 50], [28.670261, 184], [29.01591, 98], [32.301697, 156], [32.401516, 151], [33.23578, 103], [34.364265, 65], [34.48437, 46], [35.120857, 55], [35.308247, 88]]


Distancia:  20.232313  al vector en la posicion:  50  desde el vector que se encuentra en:  67
Imagen de BD:  144  Imagen de consulta:  162
Mas cercanos:  [[20.232313, 50], [22.466135, 184], [25.362411, 94], [26.063078, 48], [27.276188, 151], [28.82706, 98], [29.358635, 46], [30.166462, 156], [31.500252, 103], [33.093693, 55]]


Distancia:  18.052425  al vector en la posicion:  98  desde el vector que se encuentra en:  68
Imagen de BD:  188  Imagen de consulta:  165
Mas cercanos:  [[18.052425, 98], [20.699154, 46], [23.796776, 160], [23.798012, 184], [24.089226, 50], [24.243073, 55], [26.944443, 91], [30.448017, 156], [30.98132, 81], [31.591108, 48]]


Distancia:  26.267193  al vector en la posicion:  54  desde el vector que se encuentra en:  69
Imagen de BD:  148  Imagen de consulta:  166
Mas cercanos:  [[26.267193, 54], [29.084562, 103], [30.109102, 65], [30.201332, 76], [31.741955, 153], [31.986748, 151], [32.15931, 105], [32.62135, 60], [33.795742, 77], [34.026028, 84]]


Distancia:  20.6991  al vector en la posicion:  160  desde el vector que se encuentra en:  70
Imagen de BD:  60  Imagen de consulta:  170
Mas cercanos:  [[20.6991, 160], [20.928858, 50], [24.326714, 184], [24.92968, 98], [26.396542, 88], [27.281898, 48], [27.424667, 156], [30.304718, 91], [30.50903, 55], [31.62087, 79]]
--> ACIERTO


Distancia:  26.88028  al vector en la posicion:  98  desde el vector que se encuentra en:  71
Imagen de BD:  188  Imagen de consulta:  171
Mas cercanos:  [[26.88028, 98], [28.122885, 151], [29.726868, 184], [33.738663, 50], [33.84729, 46], [36.009438, 86], [36.21405, 65], [36.40571, 88], [36.880184, 48], [37.339485, 91]]


Distancia:  23.2146  al vector en la posicion:  151  desde el vector que se encuentra en:  72
Imagen de BD:  52  Imagen de consulta:  175
Mas cercanos:  [[23.2146, 151], [30.396767, 103], [30.403942, 98], [31.38033, 84], [32.440025, 184], [33.136406, 50], [33.17175, 76], [34.956375, 65], [35.466743, 46], [35.8043, 48]]
--> ACIERTO


Distancia:  19.439663  al vector en la posicion:  98  desde el vector que se encuentra en:  73
Imagen de BD:  188  Imagen de consulta:  0
Mas cercanos:  [[19.439663, 98], [27.132841, 184], [28.369537, 46], [28.890247, 50], [30.937065, 160], [31.222452, 151], [32.65013, 55], [33.16227, 48], [33.24781, 77], [33.542088, 86]]


Distancia:  24.567776  al vector en la posicion:  76  desde el vector que se encuentra en:  74
Imagen de BD:  168  Imagen de consulta:  183
Mas cercanos:  [[24.567776, 76], [26.589443, 103], [26.771994, 54], [27.407154, 151], [31.974937, 153], [32.62092, 65], [33.18161, 156], [33.325, 50], [33.682175, 48], [34.46599, 94]]


Distancia:  29.333294  al vector en la posicion:  103  desde el vector que se encuentra en:  75
Imagen de BD:  192  Imagen de consulta:  185
Mas cercanos:  [[29.333294, 103], [30.72181, 76], [33.16446, 151], [33.43844, 65], [33.969727, 153], [34.780647, 156], [35.038406, 98], [36.496956, 184], [36.902313, 50], [37.859818, 77]]


Distancia:  14.953297  al vector en la posicion:  98  desde el vector que se encuentra en:  76
Imagen de BD:  188  Imagen de consulta:  188
Mas cercanos:  [[14.953297, 98], [20.132927, 184], [22.625652, 50], [29.365376, 46], [29.88972, 160], [31.881306, 48], [32.54531, 151], [33.716232, 55], [34.003048, 88], [34.426933, 86]]
--> ACIERTO


Distancia:  19.207184  al vector en la posicion:  50  desde el vector que se encuentra en:  77
Imagen de BD:  144  Imagen de consulta:  11
Mas cercanos:  [[19.207184, 50], [24.748774, 184], [27.270733, 98], [30.050434, 160], [30.959993, 46], [32.290833, 156], [33.733364, 53], [34.294052, 103], [34.625153, 151], [34.649677, 94]]


Distancia:  18.492651  al vector en la posicion:  184  desde el vector que se encuentra en:  78
Imagen de BD:  82  Imagen de consulta:  190
Mas cercanos:  [[18.492651, 184], [20.575644, 50], [22.325605, 98], [26.215008, 48], [28.054132, 46], [29.336582, 160], [29.347239, 156], [30.429932, 55], [30.580732, 88], [30.961346, 91]]


Distancia:  26.890394  al vector en la posicion:  46  desde el vector que se encuentra en:  79
Imagen de BD:  140  Imagen de consulta:  191
Mas cercanos:  [[26.890394, 46], [29.48981, 184], [29.809237, 50], [30.078728, 55], [30.771275, 91], [30.841152, 160], [31.294952, 98], [32.476753, 42], [33.626373, 102], [34.083015, 48]]
--> ACIERTO


Distancia:  28.8289  al vector en la posicion:  151  desde el vector que se encuentra en:  80
Imagen de BD:  52  Imagen de consulta:  197
Mas cercanos:  [[28.8289, 151], [30.19967, 184], [31.8941, 50], [32.980423, 98], [33.626556, 65], [35.889984, 153], [36.741913, 54], [38.267532, 103], [38.62104, 88], [39.614674, 46]]


Distancia:  22.382391  al vector en la posicion:  91  desde el vector que se encuentra en:  81
Imagen de BD:  181  Imagen de consulta:  200
Mas cercanos:  [[22.382391, 91], [23.852636, 55], [24.536125, 160], [26.204208, 42], [27.395222, 50], [28.887888, 46], [30.16049, 156], [30.456287, 58], [31.037663, 98], [32.03779, 88]]


Distancia:  29.813965  al vector en la posicion:  151  desde el vector que se encuentra en:  82
Imagen de BD:  52  Imagen de consulta:  11
Mas cercanos:  [[29.813965, 151], [32.315105, 88], [32.675694, 50], [33.105198, 98], [34.187225, 184], [34.362, 76], [34.84797, 156], [35.126236, 77], [35.259644, 103], [35.324608, 65]]


Porcentaje de aciertos:  0.1927710843373494
```


**[Celda 51 - Código]**
```python
# CONSULTAS POR SIMILITUD UTILIZANDO LA BASE DE CONSULTAS DIBUJADAS - OPTIMIZADO
# import numpy as np

contador_aciertos = 0
cantidad_consultas = len(consultas)
cantidad_BD = len(imagenesBD)

tensor_consultas = tf.convert_to_tensor(consultas)
print(tensor_consultas.shape)
tensor_imagenesBD = tf.convert_to_tensor(imagenesBD)
print(tensor_imagenesBD.shape)

vectores_consulta = modelTriplet.predict(tensor_consultas)
vectores_elemento = modelTriplet.predict(tensor_imagenesBD)

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


*Salida:*
```text
(40, 28, 28, 1)
(203, 28, 28, 1)
2/2 [==============================] - 1s 13ms/step
7/7 [==============================] - 0s 20ms/step
Consulta número:  0
   Imagen de consulta:  100
   Mas cercanos:  [[0.069497034, '58'], [0.07389266, '172'], [0.07423751, '152'], [0.07527897, '39'], [0.07854392, '22']]


Consulta número:  1
   Imagen de consulta:  107
   Mas cercanos:  [[0.06006126, '173'], [0.060418066, '107'], [0.06311029, '17'], [0.06320736, '182'], [0.06406873, '184']]
--> ACIERTO


Consulta número:  2
   Imagen de consulta:  115
   Mas cercanos:  [[0.056708284, '107'], [0.05851414, '184'], [0.062443692, '61'], [0.062619224, '80'], [0.06372049, '148']]


Consulta número:  3
   Imagen de consulta:  118
   Mas cercanos:  [[0.061198622, '73'], [0.062026203, '89'], [0.062057786, '76'], [0.06210647, '152'], [0.062498793, '92']]


Consulta número:  4
   Imagen de consulta:  125
   Mas cercanos:  [[0.06453284, '171'], [0.069223024, '62'], [0.06932661, '181'], [0.06974785, '103'], [0.07021287, '136']]


Consulta número:  5
   Imagen de consulta:  126
   Mas cercanos:  [[0.054266155, '173'], [0.055408128, '133'], [0.055712763, '126'], [0.057513487, '54'], [0.061957695, '135']]
--> ACIERTO


Consulta número:  6
   Imagen de consulta:  127
   Mas cercanos:  [[0.0585086, '127'], [0.07644917, '173'], [0.07765546, '128'], [0.08018094, '71'], [0.08071399, '112']]
--> ACIERTO


Consulta número:  7
   Imagen de consulta:  128
   Mas cercanos:  [[0.05493794, '128'], [0.057449557, '106'], [0.060330097, '173'], [0.06352458, '129'], [0.06523039, '198']]
--> ACIERTO


Consulta número:  8
   Imagen de consulta:  129
   Mas cercanos:  [[0.066141725, '138'], [0.069005504, '47'], [0.06917239, '114'], [0.070802175, '66'], [0.07139715, '54']]


Consulta número:  9
   Imagen de consulta:  130
   Mas cercanos:  [[0.05339056, '82'], [0.05550972, '96'], [0.056563232, '56'], [0.05677132, '130'], [0.057604216, '85']]
--> ACIERTO


Consulta número:  10
   Imagen de consulta:  131
   Mas cercanos:  [[0.05084416, '131'], [0.06303198, '47'], [0.06505762, '93'], [0.06916796, '126'], [0.07050561, '50']]
--> ACIERTO


Consulta número:  11
   Imagen de consulta:  132
   Mas cercanos:  [[0.06988602, '7'], [0.07249202, '70'], [0.072632544, '102'], [0.074019045, '121'], [0.07415014, '42']]


Consulta número:  12
   Imagen de consulta:  133
   Mas cercanos:  [[0.05196293, '173'], [0.055740505, '135'], [0.056987304, '174'], [0.05904453, '25'], [0.060041193, '131']]


Consulta número:  13
   Imagen de consulta:  134
   Mas cercanos:  [[0.05556162, '70'], [0.0609157, '19'], [0.061346095, '147'], [0.06296528, '65'], [0.06508949, '114']]


Consulta número:  14
   Imagen de consulta:  134
   Mas cercanos:  [[0.06821551, '200'], [0.070618615, '123'], [0.07265125, '65'], [0.07610403, '55'], [0.076275736, '152']]


Consulta número:  15
   Imagen de consulta:  143
   Mas cercanos:  [[0.07423854, '145'], [0.07480411, '200'], [0.075290374, '54'], [0.07603186, '55'], [0.076345816, '133']]


Consulta número:  16
   Imagen de consulta:  144
   Mas cercanos:  [[0.056069296, '47'], [0.058353834, '77'], [0.058638785, '54'], [0.058745805, '16'], [0.059005648, '149']]


Consulta número:  17
   Imagen de consulta:  145
   Mas cercanos:  [[0.07161765, '30'], [0.071940854, '176'], [0.0731995, '159'], [0.07446118, '16'], [0.07460311, '24']]


Consulta número:  18
   Imagen de consulta:  146
   Mas cercanos:  [[0.02579196, '146'], [0.058034707, '123'], [0.065170206, '95'], [0.06839263, '50'], [0.0697982, '179']]
--> ACIERTO


Consulta número:  19
   Imagen de consulta:  166
   Mas cercanos:  [[0.058259778, '82'], [0.05853349, '173'], [0.05932527, '166'], [0.060631152, '41'], [0.061432533, '92']]
--> ACIERTO


Consulta número:  20
   Imagen de consulta:  170
   Mas cercanos:  [[0.069719926, '182'], [0.07019939, '15'], [0.0708235, '123'], [0.070990406, '106'], [0.07112531, '31']]


Consulta número:  21
   Imagen de consulta:  174
   Mas cercanos:  [[0.045473266, '181'], [0.05242075, '174'], [0.057078958, '88'], [0.06294308, '112'], [0.063014835, '92']]
--> ACIERTO


Consulta número:  22
   Imagen de consulta:  175
   Mas cercanos:  [[0.07261049, '70'], [0.073856615, '114'], [0.0746515, '80'], [0.07598016, '77'], [0.07643465, '125']]


Consulta número:  23
   Imagen de consulta:  192
   Mas cercanos:  [[0.056373958, '176'], [0.059425745, '173'], [0.06057073, '177'], [0.062204327, '137'], [0.062422305, '184']]


Consulta número:  24
   Imagen de consulta:  195
   Mas cercanos:  [[0.056318812, '126'], [0.057816725, '131'], [0.058271598, '54'], [0.058683205, '47'], [0.06035778, '93']]


Consulta número:  25
   Imagen de consulta:  197
   Mas cercanos:  [[0.058166575, '82'], [0.058895234, '197'], [0.059845917, '184'], [0.06170975, '130'], [0.06506292, '131']]
--> ACIERTO


Consulta número:  26
   Imagen de consulta:  2
   Mas cercanos:  [[0.06578099, '156'], [0.06747292, '89'], [0.0675199, '34'], [0.067741014, '185'], [0.06805324, '153']]


Consulta número:  27
   Imagen de consulta:  24
   Mas cercanos:  [[0.05664363, '24'], [0.06360555, '68'], [0.06498329, '159'], [0.06580918, '31'], [0.06617638, '53']]
--> ACIERTO


Consulta número:  28
   Imagen de consulta:  31
   Mas cercanos:  [[0.058775485, '123'], [0.06185132, '93'], [0.065358154, '31'], [0.065927014, '120'], [0.06675707, '111']]
--> ACIERTO


Consulta número:  29
   Imagen de consulta:  32
   Mas cercanos:  [[0.060377244, '86'], [0.06156999, '135'], [0.061755586, '32'], [0.064188205, '149'], [0.06561743, '199']]
--> ACIERTO


Consulta número:  30
   Imagen de consulta:  37
   Mas cercanos:  [[0.056206454, '54'], [0.05651499, '135'], [0.05659874, '29'], [0.056661554, '31'], [0.056835275, '70']]


Consulta número:  31
   Imagen de consulta:  47
   Mas cercanos:  [[0.053241678, '73'], [0.06061434, '129'], [0.06433464, '47'], [0.06446138, '71'], [0.06685095, '114']]
--> ACIERTO


Consulta número:  32
   Imagen de consulta:  55
   Mas cercanos:  [[0.055119243, '114'], [0.057243735, '113'], [0.057729058, '7'], [0.058996398, '110'], [0.059954993, '153']]


Consulta número:  33
   Imagen de consulta:  58
   Mas cercanos:  [[0.028325636, '58'], [0.052494925, '47'], [0.055685867, '172'], [0.056617297, '94'], [0.06081613, '77']]
--> ACIERTO


Consulta número:  34
   Imagen de consulta:  59
   Mas cercanos:  [[0.054813843, '31'], [0.063109994, '200'], [0.06733143, '203'], [0.06832823, '120'], [0.06838137, '28']]


Consulta número:  35
   Imagen de consulta:  7
   Mas cercanos:  [[0.05339641, '7'], [0.06297768, '31'], [0.06305272, '182'], [0.06426002, '70'], [0.0643684, '110']]
--> ACIERTO


Consulta número:  36
   Imagen de consulta:  79
   Mas cercanos:  [[0.060351256, '39'], [0.062260795, '34'], [0.068855025, '48'], [0.07162095, '124'], [0.07251559, '5']]


Consulta número:  37
   Imagen de consulta:  8
   Mas cercanos:  [[0.053477068, '8'], [0.06763133, '59'], [0.06813365, '147'], [0.07007129, '73'], [0.07114794, '121']]
--> ACIERTO


Consulta número:  38
   Imagen de consulta:  88
   Mas cercanos:  [[0.013152578, '88'], [0.05713968, '3'], [0.058095075, '181'], [0.062023062, '25'], [0.06308981, '114']]
--> ACIERTO


Consulta número:  39
   Imagen de consulta:  94
   Mas cercanos:  [[0.02964221, '94'], [0.05350951, '60'], [0.058553565, '149'], [0.05943731, '58'], [0.059917156, '71']]
--> ACIERTO


Porcentaje de aciertos:  0.475
```


**[Celda 52 - Código]**
```python
# CONSULTAS POR SIMILITUD UTILIZANDO LA BASE DE NUEVAS CONSULTAS
# import numpy as np

contador_aciertos = 0
cantidad_consultas = len(consultas)
cantidad_BD = len(imagenesBD)

tensor_consultas = tf.convert_to_tensor(consultas)
print(tensor_consultas.shape)
tensor_imagenesBD = tf.convert_to_tensor(imagenesBD)
print(tensor_imagenesBD.shape)

vectores_consulta = modelTriplet.predict(tensor_consultas)
vectores_elemento = modelTriplet.predict(tensor_imagenesBD)

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


**[Celda 53 - Código]**
```python
# CONSULTAS DIBUJADAS A MANO - CODIGO SIN OPTIMIZAR

import numpy as np
# from keras.models import Model

#layer_name = 'flatten'
#extractor_consulta = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
#vector_consulta = extractor_consulta.predict(img5.reshape(1,28,28,1))
#a = np.array(vector_consulta[0])
contador_aciertos = 0
cantidad_consultas = len(consultas)
cantidad_BD = len(imagenesBD)

#Inicio del for consulta
for i in range(cantidad_consultas):
    vector_consulta = modelTriplet.predict(consultas[i].reshape(1,28,28,1))
    a = np.array(vector_consulta[0])
    # print(np.sum(a))
    min = 2000000
    nnklist = []
    # Inicio del for DB
    for j in range(cantidad_BD):
        vector_elemento = modelTriplet.predict(imagenesBD[j].reshape(1,28,28,1))
        b = np.array(vector_elemento[0])
        # print(np.sum(b))
        #dist = np.sqrt(np.sum(np.square(a-b)))
        dist = np.sum(np.abs(a-b))
        # print('distancia: ',dist)
        nnk(nnklist,[dist,j],5)
        if dist < min:
            min = dist
            vector_cercano = b
            posicion_cercano = j

    print("Consulta número: ", i)
    print("Imagen de BD: ", y_imagenesBD[posicion_cercano], " Imagen de consulta: ", y_consultas[i])
    print('Mas cercanos: ', nnk2etiquetas(nnklist, y_imagenesBD))
    for par in nnklist:
        if y_imagenesBD[par[1]] == y_consultas[i]:
            contador_aciertos += 1
            print('--> ACIERTO')
    print("\n")

print("Porcentaje de aciertos: ", contador_aciertos/cantidad_consultas)
```


*Salida:*
```text
Consulta número:  0
Imagen de BD:  100  Imagen de consulta:  100
Mas cercanos:  [[0.74026746, '100'], [0.87536156, '51'], [0.91027325, '116'], [0.9235928, '150'], [0.9523059, '63']]
--> ACIERTO


Consulta número:  1
Imagen de BD:  155  Imagen de consulta:  107
Mas cercanos:  [[0.7189667, '155'], [0.9236342, '41'], [0.94274855, '154'], [0.96362746, '38'], [0.9709749, '25']]
```


**[Celda 54 - Código]**
```python
# CONSULTAS POR SIMILITUD UTILIZANDO COMO BASE CONSULTAS ALEATORIAS TOMADAS DESDE LAS IMAGENES AUMENTADAS

import numpy as np
from keras.models import Model

#layer_name = 'flatten'
#extractor_consulta = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
#vector_consulta = extractor_consulta.predict(img5.reshape(1,28,28,1))
#a = np.array(vector_consulta[0])
contador_aciertos = 0
cantidad_consultas = 100
cantidad_BD = len(imagenesBD)

#Inicio del for consulta
for i in range(cantidad_consultas):
    iquery = random.randint(0, len(marcasBDA)-1)
    vector_consulta = model.predict(marcasBDA[iquery].reshape(1,28,28,1))
    a = np.array(vector_consulta[0])
    min = 20000
    nnklist = []
    # Inicio del for DB
    for j in range(cantidad_BD):
        vector_elemento = model.predict(imagenesBD[j].reshape(1,28,28,1))
        b = np.array(vector_elemento[0])
        #dist = np.sqrt(np.sum(np.square(a-b)))
        dist = np.sum(np.abs(a-b))
        nnk(nnklist,[dist,j],1)
        if dist < min:
            min = dist
            vector_cercano = b
            posicion_cercano = j

    print("Distancia: ", min, " al vector en la posicion: ", posicion_cercano, " desde el vector que se encuentra en: ", iquery)
    print("Imagen de BD: ", y_imagenesBD[posicion_cercano], " Imagen de consulta: ", y_marcasBDA[iquery])
    print('Mas cercanos: ',nnklist)
    for par in nnklist:
        if y_imagenesBD[par[1]] == y_marcasBDA[iquery]:
            contador_aciertos += 1
            print('--> ACIERTO')
    print("\n")

print("Porcentaje de aciertos: ", contador_aciertos/cantidad_consultas)
```


*Salida:*
```text
Distancia:  23.322086  al vector en la posicion:  140  desde el vector que se encuentra en:  65621
Imagen de BD:  42  Imagen de consulta:  42
Mas cercanos:  [[23.322086, 140]]
--> ACIERTO


Distancia:  17.350151  al vector en la posicion:  106  desde el vector que se encuentra en:  46925
Imagen de BD:  195  Imagen de consulta:  195
Mas cercanos:  [[17.350151, 106]]
--> ACIERTO


Distancia:  27.359276  al vector en la posicion:  139  desde el vector que se encuentra en:  63457
Imagen de BD:  41  Imagen de consulta:  41
Mas cercanos:  [[27.359276, 139]]
--> ACIERTO


Distancia:  22.936062  al vector en la posicion:  19  desde el vector que se encuentra en:  7870
Imagen de BD:  116  Imagen de consulta:  116
Mas cercanos:  [[22.936062, 19]]
--> ACIERTO


Distancia:  29.841011  al vector en la posicion:  51  desde el vector que se encuentra en:  22330
Imagen de BD:  145  Imagen de consulta:  145
Mas cercanos:  [[29.841011, 51]]
--> ACIERTO


Distancia:  19.60564  al vector en la posicion:  177  desde el vector que se encuentra en:  127935
Imagen de BD:  76  Imagen de consulta:  76
Mas cercanos:  [[19.60564, 177]]
--> ACIERTO


Distancia:  13.821021  al vector en la posicion:  162  desde el vector que se encuentra en:  101141
Imagen de BD:  62  Imagen de consulta:  62
Mas cercanos:  [[13.821021, 162]]
--> ACIERTO


Distancia:  22.660162  al vector en la posicion:  157  desde el vector que se encuentra en:  93967
Imagen de BD:  58  Imagen de consulta:  58
Mas cercanos:  [[22.660162, 157]]
--> ACIERTO


Distancia:  21.059544  al vector en la posicion:  164  desde el vector que se encuentra en:  105177
Imagen de BD:  64  Imagen de consulta:  64
Mas cercanos:  [[21.059544, 164]]
--> ACIERTO


Distancia:  14.317154  al vector en la posicion:  67  desde el vector que se encuentra en:  34228
Imagen de BD:  16  Imagen de consulta:  16
Mas cercanos:  [[14.317154, 67]]
--> ACIERTO


Distancia:  23.474945  al vector en la posicion:  153  desde el vector que se encuentra en:  87688
Imagen de BD:  54  Imagen de consulta:  54
Mas cercanos:  [[23.474945, 153]]
--> ACIERTO


Distancia:  21.332346  al vector en la posicion:  127  desde el vector que se encuentra en:  57093
Imagen de BD:  30  Imagen de consulta:  30
Mas cercanos:  [[21.332346, 127]]
--> ACIERTO


Distancia:  23.070843  al vector en la posicion:  5  desde el vector que se encuentra en:  1716
Imagen de BD:  103  Imagen de consulta:  103
Mas cercanos:  [[23.070843, 5]]
--> ACIERTO


Distancia:  23.416779  al vector en la posicion:  151  desde el vector que se encuentra en:  82385
Imagen de BD:  52  Imagen de consulta:  52
Mas cercanos:  [[23.416779, 151]]
--> ACIERTO


Distancia:  28.175282  al vector en la posicion:  7  desde el vector que se encuentra en:  77237
Imagen de BD:  105  Imagen de consulta:  49
Mas cercanos:  [[28.175282, 7]]


Distancia:  21.658422  al vector en la posicion:  153  desde el vector que se encuentra en:  86582
Imagen de BD:  54  Imagen de consulta:  54
Mas cercanos:  [[21.658422, 153]]
--> ACIERTO


Distancia:  21.912516  al vector en la posicion:  182  desde el vector que se encuentra en:  135890
Imagen de BD:  80  Imagen de consulta:  80
Mas cercanos:  [[21.912516, 182]]
--> ACIERTO


Distancia:  20.986553  al vector en la posicion:  199  desde el vector que se encuentra en:  143266
Imagen de BD:  96  Imagen de consulta:  96
Mas cercanos:  [[20.986553, 199]]
--> ACIERTO


Distancia:  22.086323  al vector en la posicion:  71  desde el vector que se encuentra en:  31123
Imagen de BD:  163  Imagen de consulta:  163
Mas cercanos:  [[22.086323, 71]]
--> ACIERTO


Distancia:  6.7214384  al vector en la posicion:  154  desde el vector que se encuentra en:  89537
Imagen de BD:  55  Imagen de consulta:  55
Mas cercanos:  [[6.7214384, 154]]
--> ACIERTO


Distancia:  17.686754  al vector en la posicion:  146  desde el vector que se encuentra en:  74919
Imagen de BD:  48  Imagen de consulta:  48
Mas cercanos:  [[17.686754, 146]]
--> ACIERTO


Distancia:  19.382713  al vector en la posicion:  149  desde el vector que se encuentra en:  79737
Imagen de BD:  50  Imagen de consulta:  50
Mas cercanos:  [[19.382713, 149]]
--> ACIERTO


Distancia:  16.251543  al vector en la posicion:  165  desde el vector que se encuentra en:  107926
Imagen de BD:  65  Imagen de consulta:  65
Mas cercanos:  [[16.251543, 165]]
--> ACIERTO


Distancia:  12.382765  al vector en la posicion:  153  desde el vector que se encuentra en:  86873
Imagen de BD:  54  Imagen de consulta:  54
Mas cercanos:  [[12.382765, 153]]
--> ACIERTO


Distancia:  25.570354  al vector en la posicion:  144  desde el vector que se encuentra en:  53726
Imagen de BD:  46  Imagen de consulta:  24
Mas cercanos:  [[25.570354, 144]]


Distancia:  26.832989  al vector en la posicion:  30  desde el vector que se encuentra en:  12968
Imagen de BD:  126  Imagen de consulta:  126
Mas cercanos:  [[26.832989, 30]]
--> ACIERTO


Distancia:  15.090807  al vector en la posicion:  198  desde el vector que se encuentra en:  142676
Imagen de BD:  95  Imagen de consulta:  95
Mas cercanos:  [[15.090807, 198]]
--> ACIERTO


Distancia:  18.774288  al vector en la posicion:  149  desde el vector que se encuentra en:  79380
Imagen de BD:  50  Imagen de consulta:  50
Mas cercanos:  [[18.774288, 149]]
--> ACIERTO


Distancia:  17.84994  al vector en la posicion:  66  desde el vector que se encuentra en:  29098
Imagen de BD:  159  Imagen de consulta:  159
Mas cercanos:  [[17.84994, 66]]
--> ACIERTO


Distancia:  28.063232  al vector en la posicion:  82  desde el vector que se encuentra en:  36427
Imagen de BD:  173  Imagen de consulta:  173
Mas cercanos:  [[28.063232, 82]]
--> ACIERTO


Distancia:  28.980389  al vector en la posicion:  185  desde el vector que se encuentra en:  124897
Imagen de BD:  83  Imagen de consulta:  75
Mas cercanos:  [[28.980389, 185]]


Distancia:  23.559391  al vector en la posicion:  38  desde el vector que se encuentra en:  36862
Imagen de BD:  133  Imagen de consulta:  174
Mas cercanos:  [[23.559391, 38]]


Distancia:  27.606153  al vector en la posicion:  132  desde el vector que se encuentra en:  57883
Imagen de BD:  35  Imagen de consulta:  32
Mas cercanos:  [[27.606153, 132]]


Distancia:  27.684593  al vector en la posicion:  174  desde el vector que se encuentra en:  117154
Imagen de BD:  73  Imagen de consulta:  70
Mas cercanos:  [[27.684593, 174]]


Distancia:  30.289467  al vector en la posicion:  113  desde el vector que se encuentra en:  50077
Imagen de BD:  200  Imagen de consulta:  200
Mas cercanos:  [[30.289467, 113]]
--> ACIERTO


Distancia:  26.332218  al vector en la posicion:  37  desde el vector que se encuentra en:  15821
Imagen de BD:  132  Imagen de consulta:  132
Mas cercanos:  [[26.332218, 37]]
--> ACIERTO


Distancia:  26.400208  al vector en la posicion:  108  desde el vector que se encuentra en:  48077
Imagen de BD:  197  Imagen de consulta:  197
Mas cercanos:  [[26.400208, 108]]
--> ACIERTO


Distancia:  31.571692  al vector en la posicion:  135  desde el vector que se encuentra en:  60409
Imagen de BD:  38  Imagen de consulta:  38
Mas cercanos:  [[31.571692, 135]]
--> ACIERTO


Distancia:  24.194138  al vector en la posicion:  73  desde el vector que se encuentra en:  32338
Imagen de BD:  165  Imagen de consulta:  165
Mas cercanos:  [[24.194138, 73]]
--> ACIERTO


Distancia:  20.01665  al vector en la posicion:  141  desde el vector que se encuentra en:  66129
Imagen de BD:  43  Imagen de consulta:  43
Mas cercanos:  [[20.01665, 141]]
--> ACIERTO


Distancia:  17.032936  al vector en la posicion:  151  desde el vector que se encuentra en:  82460
Imagen de BD:  52  Imagen de consulta:  52
Mas cercanos:  [[17.032936, 151]]
--> ACIERTO


Distancia:  23.475147  al vector en la posicion:  79  desde el vector que se encuentra en:  35021
Imagen de BD:  170  Imagen de consulta:  170
Mas cercanos:  [[23.475147, 79]]
--> ACIERTO


Distancia:  12.711742  al vector en la posicion:  141  desde el vector que se encuentra en:  65944
Imagen de BD:  43  Imagen de consulta:  43
Mas cercanos:  [[12.711742, 141]]
--> ACIERTO


Distancia:  26.114511  al vector en la posicion:  143  desde el vector que se encuentra en:  69461
Imagen de BD:  45  Imagen de consulta:  45
Mas cercanos:  [[26.114511, 143]]
--> ACIERTO


Distancia:  22.820272  al vector en la posicion:  89  desde el vector que se encuentra en:  44223
Imagen de BD:  18  Imagen de consulta:  18
Mas cercanos:  [[22.820272, 89]]
--> ACIERTO


Distancia:  19.011866  al vector en la posicion:  167  desde el vector que se encuentra en:  110709
Imagen de BD:  67  Imagen de consulta:  67
Mas cercanos:  [[19.011866, 167]]
--> ACIERTO


Distancia:  16.781752  al vector en la posicion:  158  desde el vector que se encuentra en:  96734
Imagen de BD:  59  Imagen de consulta:  59
Mas cercanos:  [[16.781752, 158]]
--> ACIERTO


Distancia:  20.63781  al vector en la posicion:  158  desde el vector que se encuentra en:  95301
Imagen de BD:  59  Imagen de consulta:  59
Mas cercanos:  [[20.63781, 158]]
--> ACIERTO


Distancia:  10.391096  al vector en la posicion:  186  desde el vector que se encuentra en:  137489
Imagen de BD:  84  Imagen de consulta:  84
Mas cercanos:  [[10.391096, 186]]
--> ACIERTO


Distancia:  16.222748  al vector en la posicion:  99  desde el vector que se encuentra en:  43950
Imagen de BD:  189  Imagen de consulta:  189
Mas cercanos:  [[16.222748, 99]]
--> ACIERTO


Distancia:  24.18205  al vector en la posicion:  65  desde el vector que se encuentra en:  87483
Imagen de BD:  158  Imagen de consulta:  54
Mas cercanos:  [[24.18205, 65]]


Distancia:  27.146906  al vector en la posicion:  150  desde el vector que se encuentra en:  82194
Imagen de BD:  51  Imagen de consulta:  51
Mas cercanos:  [[27.146906, 150]]
--> ACIERTO


Distancia:  14.6291065  al vector en la posicion:  175  desde el vector que se encuentra en:  123783
Imagen de BD:  74  Imagen de consulta:  74
Mas cercanos:  [[14.6291065, 175]]
--> ACIERTO


Distancia:  31.642845  al vector en la posicion:  18  desde el vector que se encuentra en:  10550
Imagen de BD:  115  Imagen de consulta:  121
Mas cercanos:  [[31.642845, 18]]


Distancia:  25.543007  al vector en la posicion:  169  desde el vector que se encuentra en:  113863
Imagen de BD:  69  Imagen de consulta:  69
Mas cercanos:  [[25.543007, 169]]
--> ACIERTO


Distancia:  22.627811  al vector en la posicion:  182  desde el vector que se encuentra en:  134348
Imagen de BD:  80  Imagen de consulta:  80
Mas cercanos:  [[22.627811, 182]]
--> ACIERTO


Distancia:  18.524733  al vector en la posicion:  139  desde el vector que se encuentra en:  62675
Imagen de BD:  41  Imagen de consulta:  41
Mas cercanos:  [[18.524733, 139]]
--> ACIERTO


Distancia:  31.17553  al vector en la posicion:  105  desde el vector que se encuentra en:  105856
Imagen de BD:  194  Imagen de consulta:  64
Mas cercanos:  [[31.17553, 105]]


Distancia:  14.489027  al vector en la posicion:  100  desde el vector que se encuentra en:  49174
Imagen de BD:  19  Imagen de consulta:  19
Mas cercanos:  [[14.489027, 100]]
--> ACIERTO


Distancia:  16.034077  al vector en la posicion:  161  desde el vector que se encuentra en:  99316
Imagen de BD:  61  Imagen de consulta:  61
Mas cercanos:  [[16.034077, 161]]
--> ACIERTO


Distancia:  24.734106  al vector en la posicion:  35  desde el vector que se encuentra en:  15088
Imagen de BD:  130  Imagen de consulta:  130
Mas cercanos:  [[24.734106, 35]]
--> ACIERTO


Distancia:  19.952085  al vector en la posicion:  58  desde el vector que se encuentra en:  25250
Imagen de BD:  151  Imagen de consulta:  151
Mas cercanos:  [[19.952085, 58]]
--> ACIERTO


Distancia:  18.663296  al vector en la posicion:  188  desde el vector que se encuentra en:  138431
Imagen de BD:  86  Imagen de consulta:  86
Mas cercanos:  [[18.663296, 188]]
--> ACIERTO


Distancia:  24.790665  al vector en la posicion:  6  desde el vector que se encuentra en:  2220
Imagen de BD:  104  Imagen de consulta:  104
Mas cercanos:  [[24.790665, 6]]
--> ACIERTO


Distancia:  33.484882  al vector en la posicion:  158  desde el vector que se encuentra en:  96441
Imagen de BD:  59  Imagen de consulta:  59
Mas cercanos:  [[33.484882, 158]]
--> ACIERTO


Distancia:  22.838757  al vector en la posicion:  0  desde el vector que se encuentra en:  49669
Imagen de BD:  1  Imagen de consulta:  1
Mas cercanos:  [[22.838757, 0]]
--> ACIERTO


Distancia:  26.027664  al vector en la posicion:  65  desde el vector que se encuentra en:  28523
Imagen de BD:  158  Imagen de consulta:  158
Mas cercanos:  [[26.027664, 65]]
--> ACIERTO


Distancia:  21.444653  al vector en la posicion:  94  desde el vector que se encuentra en:  41727
Imagen de BD:  184  Imagen de consulta:  184
Mas cercanos:  [[21.444653, 94]]
--> ACIERTO


Distancia:  28.611345  al vector en la posicion:  169  desde el vector que se encuentra en:  113544
Imagen de BD:  69  Imagen de consulta:  69
Mas cercanos:  [[28.611345, 169]]
--> ACIERTO


Distancia:  21.471502  al vector en la posicion:  115  desde el vector que se encuentra en:  51291
Imagen de BD:  202  Imagen de consulta:  202
Mas cercanos:  [[21.471502, 115]]
--> ACIERTO


Distancia:  25.453407  al vector en la posicion:  138  desde el vector que se encuentra en:  62066
Imagen de BD:  40  Imagen de consulta:  40
Mas cercanos:  [[25.453407, 138]]
--> ACIERTO


Distancia:  19.967163  al vector en la posicion:  184  desde el vector que se encuentra en:  43242
Imagen de BD:  82  Imagen de consulta:  188
Mas cercanos:  [[19.967163, 184]]


Distancia:  9.201372  al vector en la posicion:  119  desde el vector que se encuentra en:  53369
Imagen de BD:  23  Imagen de consulta:  23
Mas cercanos:  [[9.201372, 119]]
--> ACIERTO


Distancia:  25.20678  al vector en la posicion:  178  desde el vector que se encuentra en:  135100
Imagen de BD:  77  Imagen de consulta:  80
Mas cercanos:  [[25.20678, 178]]


Distancia:  10.959885  al vector en la posicion:  143  desde el vector que se encuentra en:  70144
Imagen de BD:  45  Imagen de consulta:  45
Mas cercanos:  [[10.959885, 143]]
--> ACIERTO


Distancia:  27.536036  al vector en la posicion:  83  desde el vector que se encuentra en:  17741
Imagen de BD:  174  Imagen de consulta:  136
Mas cercanos:  [[27.536036, 83]]


Distancia:  31.485102  al vector en la posicion:  54  desde el vector que se encuentra en:  23558
Imagen de BD:  148  Imagen de consulta:  148
Mas cercanos:  [[31.485102, 54]]
--> ACIERTO


Distancia:  19.593647  al vector en la posicion:  145  desde el vector que se encuentra en:  73685
Imagen de BD:  47  Imagen de consulta:  47
Mas cercanos:  [[19.593647, 145]]
--> ACIERTO


Distancia:  25.34145  al vector en la posicion:  107  desde el vector que se encuentra en:  47418
Imagen de BD:  196  Imagen de consulta:  196
Mas cercanos:  [[25.34145, 107]]
--> ACIERTO


Distancia:  27.185051  al vector en la posicion:  3  desde el vector que se encuentra en:  807
Imagen de BD:  101  Imagen de consulta:  101
Mas cercanos:  [[27.185051, 3]]
--> ACIERTO


Distancia:  24.85622  al vector en la posicion:  41  desde el vector que se encuentra en:  17650
Imagen de BD:  136  Imagen de consulta:  136
Mas cercanos:  [[24.85622, 41]]
--> ACIERTO


Distancia:  19.351864  al vector en la posicion:  157  desde el vector que se encuentra en:  94502
Imagen de BD:  58  Imagen de consulta:  58
Mas cercanos:  [[19.351864, 157]]
--> ACIERTO


Distancia:  24.884993  al vector en la posicion:  111  desde el vector que se encuentra en:  135243
Imagen de BD:  2  Imagen de consulta:  80
Mas cercanos:  [[24.884993, 111]]


Distancia:  19.20304  al vector en la posicion:  135  desde el vector que se encuentra en:  60570
Imagen de BD:  38  Imagen de consulta:  38
Mas cercanos:  [[19.20304, 135]]
--> ACIERTO


Distancia:  20.603638  al vector en la posicion:  148  desde el vector que se encuentra en:  97082
Imagen de BD:  5  Imagen de consulta:  5
Mas cercanos:  [[20.603638, 148]]
--> ACIERTO


Distancia:  20.256397  al vector en la posicion:  106  desde el vector que se encuentra en:  46813
Imagen de BD:  195  Imagen de consulta:  195
Mas cercanos:  [[20.256397, 106]]
--> ACIERTO


Distancia:  20.33724  al vector en la posicion:  45  desde el vector que se encuentra en:  24620
Imagen de BD:  14  Imagen de consulta:  14
Mas cercanos:  [[20.33724, 45]]
--> ACIERTO


Distancia:  26.043211  al vector en la posicion:  59  desde el vector que se encuentra en:  25977
Imagen de BD:  152  Imagen de consulta:  152
Mas cercanos:  [[26.043211, 59]]
--> ACIERTO


Distancia:  19.872425  al vector en la posicion:  9  desde el vector que se encuentra en:  3541
Imagen de BD:  107  Imagen de consulta:  107
Mas cercanos:  [[19.872425, 9]]
--> ACIERTO


Distancia:  28.928087  al vector en la posicion:  151  desde el vector que se encuentra en:  28477
Imagen de BD:  52  Imagen de consulta:  158
Mas cercanos:  [[28.928087, 151]]


Distancia:  26.254856  al vector en la posicion:  147  desde el vector que se encuentra en:  77702
Imagen de BD:  49  Imagen de consulta:  49
Mas cercanos:  [[26.254856, 147]]
--> ACIERTO


Distancia:  27.785246  al vector en la posicion:  58  desde el vector que se encuentra en:  25466
Imagen de BD:  151  Imagen de consulta:  151
Mas cercanos:  [[27.785246, 58]]
--> ACIERTO


Distancia:  8.457191  al vector en la posicion:  160  desde el vector que se encuentra en:  97210
Imagen de BD:  60  Imagen de consulta:  60
Mas cercanos:  [[8.457191, 160]]
--> ACIERTO


Distancia:  17.464077  al vector en la posicion:  154  desde el vector que se encuentra en:  88719
Imagen de BD:  55  Imagen de consulta:  55
Mas cercanos:  [[17.464077, 154]]
--> ACIERTO


Distancia:  22.211693  al vector en la posicion:  191  desde el vector que se encuentra en:  139667
Imagen de BD:  89  Imagen de consulta:  89
Mas cercanos:  [[22.211693, 191]]
--> ACIERTO


Distancia:  27.20779  al vector en la posicion:  102  desde el vector que se encuentra en:  45069
Imagen de BD:  191  Imagen de consulta:  191
Mas cercanos:  [[27.20779, 102]]
--> ACIERTO


Distancia:  26.902885  al vector en la posicion:  163  desde el vector que se encuentra en:  103130
Imagen de BD:  63  Imagen de consulta:  63
Mas cercanos:  [[26.902885, 163]]
--> ACIERTO


Distancia:  24.030746  al vector en la posicion:  67  desde el vector que se encuentra en:  99535
Imagen de BD:  16  Imagen de consulta:  61
Mas cercanos:  [[24.030746, 67]]


Distancia:  21.977797  al vector en la posicion:  163  desde el vector que se encuentra en:  103867
Imagen de BD:  63  Imagen de consulta:  63
Mas cercanos:  [[21.977797, 163]]
--> ACIERTO


Distancia:  18.301857  al vector en la posicion:  143  desde el vector que se encuentra en:  70603
Imagen de BD:  45  Imagen de consulta:  45
Mas cercanos:  [[18.301857, 143]]
--> ACIERTO


Porcentaje de aciertos:  0.85
```


**[Celda 55 - Código]**
```python
learning_rate = optimizer.lr.numpy()
print(learning_rate)
```


*Salida:*
```text
1e-04
```


**[Celda 56 - Código]**
```python
# UTILIZANDO SIAMESA COMO FUNCION DE DISTANCIA: CONSULTAS POR SIMILITUD UTILIZANDO LA BASE DE CONSULTAS DIFICILES

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
    #vector_consulta = model.predict(consultas[i].reshape(1,28,28,1))
    #a = np.array(vector_consulta[0])
    min = 20000
    nnklist = []
    # Inicio del for DB
    for j in range(cantidad_BD):
        dist = siamese_net.predict([imagenesBD[j].reshape(1,28,28,1),consultas[i].reshape(1,28,28,1)])
        nnk(nnklist,[dist,j],5)
        if dist < min:
            min = dist
            vector_cercano = b
            posicion_cercano = j

    print("Distancia: ", min, " al vector en la posicion: ", posicion_cercano, " desde el vector que se encuentra en: ", i)
    print("Imagen de BD: ", y_imagenesBD[posicion_cercano], " Imagen de consulta: ", y_consultas[i])
    print('Mas cercanos: ',nnklist)
    for par in nnklist:
        if y_imagenesBD[par[1]] == y_consultas[i]:
            contador_aciertos += 1
            print('--> ACIERTO')
    print("\n")

print("Porcentaje de aciertos: ", contador_aciertos/cantidad_consultas)
```


*Salida:*
```text
Distancia:  [[0.10392378]]  al vector en la posicion:  151  desde el vector que se encuentra en:  0
Imagen de BD:  52  Imagen de consulta:  197
Mas cercanos:  [[array([[0.10392378]], dtype=float32), 151], [array([[0.57639736]], dtype=float32), 65], [array([[0.7565481]], dtype=float32), 163], [array([[0.7788029]], dtype=float32), 50], [array([[0.8833543]], dtype=float32), 184]]


Distancia:  [[0.4103446]]  al vector en la posicion:  151  desde el vector que se encuentra en:  1
Imagen de BD:  52  Imagen de consulta:  11
Mas cercanos:  [[array([[0.4103446]], dtype=float32), 151], [array([[0.46325916]], dtype=float32), 184], [array([[0.6784538]], dtype=float32), 50], [array([[0.79651654]], dtype=float32), 153], [array([[0.8219682]], dtype=float32), 98]]


Distancia:  [[0.01839073]]  al vector en la posicion:  76  desde el vector que se encuentra en:  2
Imagen de BD:  168  Imagen de consulta:  14
Mas cercanos:  [[array([[0.01839073]], dtype=float32), 76], [array([[0.34413707]], dtype=float32), 151], [array([[0.3567476]], dtype=float32), 103], [array([[0.74128217]], dtype=float32), 153], [array([[0.83867294]], dtype=float32), 54]]


Distancia:  [[0.01827533]]  al vector en la posicion:  50  desde el vector que se encuentra en:  3
Imagen de BD:  144  Imagen de consulta:  16
Mas cercanos:  [[array([[0.01827533]], dtype=float32), 50], [array([[0.06887724]], dtype=float32), 184], [array([[0.36283842]], dtype=float32), 98], [array([[0.8370172]], dtype=float32), 160], [array([[0.86420053]], dtype=float32), 94]]


Distancia:  [[0.2808653]]  al vector en la posicion:  50  desde el vector que se encuentra en:  4
Imagen de BD:  144  Imagen de consulta:  18
Mas cercanos:  [[array([[0.2808653]], dtype=float32), 50], [array([[0.30588844]], dtype=float32), 184], [array([[0.79833]], dtype=float32), 151], [array([[0.85611695]], dtype=float32), 76], [array([[0.8694854]], dtype=float32), 98]]


Distancia:  [[0.01493864]]  al vector en la posicion:  50  desde el vector que se encuentra en:  5
Imagen de BD:  144  Imagen de consulta:  19
Mas cercanos:  [[array([[0.01493864]], dtype=float32), 50], [array([[0.11397892]], dtype=float32), 184], [array([[0.1765252]], dtype=float32), 98], [array([[0.58711624]], dtype=float32), 160], [array([[0.7102891]], dtype=float32), 88]]


Distancia:  [[0.13791002]]  al vector en la posicion:  151  desde el vector que se encuentra en:  6
Imagen de BD:  52  Imagen de consulta:  24
Mas cercanos:  [[array([[0.13791002]], dtype=float32), 151], [array([[0.4681218]], dtype=float32), 184], [array([[0.8230737]], dtype=float32), 50], [array([[0.86818165]], dtype=float32), 76], [array([[0.8797046]], dtype=float32), 98]]


Distancia:  [[0.18886624]]  al vector en la posicion:  98  desde el vector que se encuentra en:  7
Imagen de BD:  188  Imagen de consulta:  24
Mas cercanos:  [[array([[0.18886624]], dtype=float32), 98], [array([[0.22569586]], dtype=float32), 184], [array([[0.5015437]], dtype=float32), 50], [array([[0.5791144]], dtype=float32), 46], [array([[0.6790806]], dtype=float32), 88]]


Distancia:  [[0.01066193]]  al vector en la posicion:  50  desde el vector que se encuentra en:  8
Imagen de BD:  144  Imagen de consulta:  27
Mas cercanos:  [[array([[0.01066193]], dtype=float32), 50], [array([[0.01594653]], dtype=float32), 184], [array([[0.06854472]], dtype=float32), 98], [array([[0.8691745]], dtype=float32), 46], [array([[0.8695753]], dtype=float32), 151]]


Distancia:  [[0.22362146]]  al vector en la posicion:  160  desde el vector que se encuentra en:  9
Imagen de BD:  60  Imagen de consulta:  33
Mas cercanos:  [[array([[0.22362146]], dtype=float32), 160], [array([[0.75484526]], dtype=float32), 184], [array([[0.78503937]], dtype=float32), 47], [array([[0.81235385]], dtype=float32), 50], [array([[0.8616675]], dtype=float32), 79]]


Distancia:  [[0.05096948]]  al vector en la posicion:  50  desde el vector que se encuentra en:  10
Imagen de BD:  144  Imagen de consulta:  36
Mas cercanos:  [[array([[0.05096948]], dtype=float32), 50], [array([[0.15750518]], dtype=float32), 184], [array([[0.35092184]], dtype=float32), 156], [array([[0.63244444]], dtype=float32), 98], [array([[0.73233473]], dtype=float32), 76]]


Distancia:  [[0.01162591]]  al vector en la posicion:  50  desde el vector que se encuentra en:  11
Imagen de BD:  144  Imagen de consulta:  5
Mas cercanos:  [[array([[0.01162591]], dtype=float32), 50], [array([[0.21478349]], dtype=float32), 184], [array([[0.49055225]], dtype=float32), 98], [array([[0.6178106]], dtype=float32), 42], [array([[0.65788877]], dtype=float32), 94]]


Distancia:  [[0.6934498]]  al vector en la posicion:  151  desde el vector que se encuentra en:  12
Imagen de BD:  52  Imagen de consulta:  36
Mas cercanos:  [[array([[0.6934498]], dtype=float32), 151], [array([[0.8076959]], dtype=float32), 76], [array([[0.91745055]], dtype=float32), 103], [array([[0.93270147]], dtype=float32), 50], [array([[0.97679365]], dtype=float32), 153]]


Distancia:  [[0.29466963]]  al vector en la posicion:  50  desde el vector que se encuentra en:  13
Imagen de BD:  144  Imagen de consulta:  44
Mas cercanos:  [[array([[0.29466963]], dtype=float32), 50], [array([[0.33953875]], dtype=float32), 98], [array([[0.42853463]], dtype=float32), 184], [array([[0.68129563]], dtype=float32), 156], [array([[0.68629104]], dtype=float32), 88]]


Distancia:  [[0.00279573]]  al vector en la posicion:  151  desde el vector que se encuentra en:  14
Imagen de BD:  52  Imagen de consulta:  45
Mas cercanos:  [[array([[0.00279573]], dtype=float32), 151], [array([[0.34559745]], dtype=float32), 76], [array([[0.77265805]], dtype=float32), 153], [array([[0.8557402]], dtype=float32), 65], [array([[0.88508034]], dtype=float32), 103]]


Distancia:  [[0.21463569]]  al vector en la posicion:  151  desde el vector que se encuentra en:  15
Imagen de BD:  52  Imagen de consulta:  46
Mas cercanos:  [[array([[0.21463569]], dtype=float32), 151], [array([[0.7197061]], dtype=float32), 163], [array([[0.7465228]], dtype=float32), 65], [array([[0.87873995]], dtype=float32), 98], [array([[0.92395586]], dtype=float32), 184]]


Distancia:  [[0.04137742]]  al vector en la posicion:  153  desde el vector que se encuentra en:  16
Imagen de BD:  54  Imagen de consulta:  47
Mas cercanos:  [[array([[0.04137742]], dtype=float32), 153], [array([[0.16525485]], dtype=float32), 103], [array([[0.46707472]], dtype=float32), 76], [array([[0.61640716]], dtype=float32), 48], [array([[0.75512457]], dtype=float32), 156]]


Distancia:  [[0.03008602]]  al vector en la posicion:  151  desde el vector que se encuentra en:  17
Imagen de BD:  52  Imagen de consulta:  47
Mas cercanos:  [[array([[0.03008602]], dtype=float32), 151], [array([[0.5959253]], dtype=float32), 76], [array([[0.9832382]], dtype=float32), 44], [array([[0.98395556]], dtype=float32), 98], [array([[0.9843035]], dtype=float32), 184]]


Distancia:  [[0.24726574]]  al vector en la posicion:  184  desde el vector que se encuentra en:  18
Imagen de BD:  82  Imagen de consulta:  47
Mas cercanos:  [[array([[0.24726574]], dtype=float32), 184], [array([[0.28433254]], dtype=float32), 98], [array([[0.5920062]], dtype=float32), 50], [array([[0.7040635]], dtype=float32), 151], [array([[0.8332762]], dtype=float32), 46]]


Distancia:  [[0.22500806]]  al vector en la posicion:  50  desde el vector que se encuentra en:  19
Imagen de BD:  144  Imagen de consulta:  49
Mas cercanos:  [[array([[0.22500806]], dtype=float32), 50], [array([[0.2889567]], dtype=float32), 42], [array([[0.33236802]], dtype=float32), 88], [array([[0.8023232]], dtype=float32), 76], [array([[0.8410595]], dtype=float32), 53]]


Distancia:  [[0.17520986]]  al vector en la posicion:  81  desde el vector que se encuentra en:  20
Imagen de BD:  172  Imagen de consulta:  51
Mas cercanos:  [[array([[0.17520986]], dtype=float32), 81], [array([[0.47966006]], dtype=float32), 106], [array([[0.7051979]], dtype=float32), 71], [array([[0.7475839]], dtype=float32), 55], [array([[0.8034778]], dtype=float32), 156]]


Distancia:  [[0.05882678]]  al vector en la posicion:  76  desde el vector que se encuentra en:  21
Imagen de BD:  168  Imagen de consulta:  53
Mas cercanos:  [[array([[0.05882678]], dtype=float32), 76], [array([[0.554836]], dtype=float32), 103], [array([[0.555415]], dtype=float32), 153], [array([[0.6460464]], dtype=float32), 151], [array([[0.7228546]], dtype=float32), 54]]


Distancia:  [[0.13438867]]  al vector en la posicion:  38  desde el vector que se encuentra en:  22
Imagen de BD:  133  Imagen de consulta:  7
Mas cercanos:  [[array([[0.13438867]], dtype=float32), 38], [array([[0.35791895]], dtype=float32), 62], [array([[0.36975062]], dtype=float32), 113], [array([[0.37739974]], dtype=float32), 74], [array([[0.6224158]], dtype=float32), 200]]


Distancia:  [[0.09801219]]  al vector en la posicion:  65  desde el vector que se encuentra en:  23
Imagen de BD:  158  Imagen de consulta:  54
Mas cercanos:  [[array([[0.09801219]], dtype=float32), 65], [array([[0.10892431]], dtype=float32), 151], [array([[0.11038061]], dtype=float32), 153], [array([[0.18756144]], dtype=float32), 163], [array([[0.82838845]], dtype=float32), 84]]
--> ACIERTO


Distancia:  [[0.00804425]]  al vector en la posicion:  151  desde el vector que se encuentra en:  24
Imagen de BD:  52  Imagen de consulta:  56
Mas cercanos:  [[array([[0.00804425]], dtype=float32), 151], [array([[0.4950731]], dtype=float32), 50], [array([[0.65516233]], dtype=float32), 184], [array([[0.73178375]], dtype=float32), 98], [array([[0.8136358]], dtype=float32), 76]]


Distancia:  [[0.15828104]]  al vector en la posicion:  151  desde el vector que se encuentra en:  25
Imagen de BD:  52  Imagen de consulta:  63
Mas cercanos:  [[array([[0.15828104]], dtype=float32), 151], [array([[0.17235103]], dtype=float32), 65], [array([[0.41288397]], dtype=float32), 163], [array([[0.5909981]], dtype=float32), 77], [array([[0.8786585]], dtype=float32), 137]]
--> ACIERTO


Distancia:  [[0.21536483]]  al vector en la posicion:  50  desde el vector que se encuentra en:  26
Imagen de BD:  144  Imagen de consulta:  66
Mas cercanos:  [[array([[0.21536483]], dtype=float32), 50], [array([[0.4678553]], dtype=float32), 98], [array([[0.5092309]], dtype=float32), 184], [array([[0.59718066]], dtype=float32), 156], [array([[0.6405409]], dtype=float32), 103]]


Distancia:  [[0.662993]]  al vector en la posicion:  171  desde el vector que se encuentra en:  27
Imagen de BD:  70  Imagen de consulta:  66
Mas cercanos:  [[array([[0.662993]], dtype=float32), 171], [array([[0.701888]], dtype=float32), 89], [array([[0.7640326]], dtype=float32), 174], [array([[0.83590555]], dtype=float32), 152], [array([[0.90033406]], dtype=float32), 73]]


Distancia:  [[0.35981843]]  al vector en la posicion:  50  desde el vector que se encuentra en:  28
Imagen de BD:  144  Imagen de consulta:  66
Mas cercanos:  [[array([[0.35981843]], dtype=float32), 50], [array([[0.81522346]], dtype=float32), 184], [array([[0.8363026]], dtype=float32), 54], [array([[0.9026335]], dtype=float32), 76], [array([[0.93161315]], dtype=float32), 70]]


Distancia:  [[0.24800093]]  al vector en la posicion:  151  desde el vector que se encuentra en:  29
Imagen de BD:  52  Imagen de consulta:  76
Mas cercanos:  [[array([[0.24800093]], dtype=float32), 151], [array([[0.4168868]], dtype=float32), 76], [array([[0.5552299]], dtype=float32), 103], [array([[0.8206707]], dtype=float32), 184], [array([[0.8249897]], dtype=float32), 153]]


Distancia:  [[0.31967023]]  al vector en la posicion:  156  desde el vector que se encuentra en:  30
Imagen de BD:  57  Imagen de consulta:  77
Mas cercanos:  [[array([[0.31967023]], dtype=float32), 156], [array([[0.40205866]], dtype=float32), 50], [array([[0.42716098]], dtype=float32), 48], [array([[0.42998737]], dtype=float32), 103], [array([[0.4879392]], dtype=float32), 153]]


Distancia:  [[0.12415869]]  al vector en la posicion:  77  desde el vector que se encuentra en:  31
Imagen de BD:  169  Imagen de consulta:  78
Mas cercanos:  [[array([[0.12415869]], dtype=float32), 77], [array([[0.17646588]], dtype=float32), 46], [array([[0.35705623]], dtype=float32), 55], [array([[0.45823032]], dtype=float32), 91], [array([[0.47399285]], dtype=float32), 88]]


Distancia:  [[0.2003317]]  al vector en la posicion:  151  desde el vector que se encuentra en:  32
Imagen de BD:  52  Imagen de consulta:  80
Mas cercanos:  [[array([[0.2003317]], dtype=float32), 151], [array([[0.747743]], dtype=float32), 76], [array([[0.85728854]], dtype=float32), 184], [array([[0.8648171]], dtype=float32), 98], [array([[0.8769541]], dtype=float32), 50]]


Distancia:  [[0.07938479]]  al vector en la posicion:  151  desde el vector que se encuentra en:  33
Imagen de BD:  52  Imagen de consulta:  11
Mas cercanos:  [[array([[0.07938479]], dtype=float32), 151], [array([[0.13402374]], dtype=float32), 76], [array([[0.24296185]], dtype=float32), 103], [array([[0.4439568]], dtype=float32), 153], [array([[0.7055054]], dtype=float32), 50]]


Distancia:  [[0.03467619]]  al vector en la posicion:  50  desde el vector que se encuentra en:  34
Imagen de BD:  144  Imagen de consulta:  83
Mas cercanos:  [[array([[0.03467619]], dtype=float32), 50], [array([[0.3399081]], dtype=float32), 184], [array([[0.5325345]], dtype=float32), 98], [array([[0.8675694]], dtype=float32), 88], [array([[0.9429462]], dtype=float32), 103]]


Distancia:  [[0.01444982]]  al vector en la posicion:  151  desde el vector que se encuentra en:  35
Imagen de BD:  52  Imagen de consulta:  87
Mas cercanos:  [[array([[0.01444982]], dtype=float32), 151], [array([[0.06115517]], dtype=float32), 184], [array([[0.1271043]], dtype=float32), 50], [array([[0.37107515]], dtype=float32), 98], [array([[0.86925626]], dtype=float32), 65]]


Distancia:  [[0.02580039]]  al vector en la posicion:  184  desde el vector que se encuentra en:  36
Imagen de BD:  82  Imagen de consulta:  93
Mas cercanos:  [[array([[0.02580039]], dtype=float32), 184], [array([[0.03122517]], dtype=float32), 50], [array([[0.15771559]], dtype=float32), 98], [array([[0.55266094]], dtype=float32), 160], [array([[0.8266656]], dtype=float32), 46]]


Distancia:  [[0.34128058]]  al vector en la posicion:  50  desde el vector que se encuentra en:  37
Imagen de BD:  144  Imagen de consulta:  95
Mas cercanos:  [[array([[0.34128058]], dtype=float32), 50], [array([[0.4704049]], dtype=float32), 98], [array([[0.5309666]], dtype=float32), 184], [array([[0.90194404]], dtype=float32), 103], [array([[0.9159022]], dtype=float32), 151]]


Distancia:  [[0.05986451]]  al vector en la posicion:  151  desde el vector que se encuentra en:  38
Imagen de BD:  52  Imagen de consulta:  100
Mas cercanos:  [[array([[0.05986451]], dtype=float32), 151], [array([[0.64523566]], dtype=float32), 163], [array([[0.7955687]], dtype=float32), 65], [array([[0.9017426]], dtype=float32), 184], [array([[0.9622646]], dtype=float32), 50]]


Distancia:  [[0.01543507]]  al vector en la posicion:  151  desde el vector que se encuentra en:  39
Imagen de BD:  52  Imagen de consulta:  103
Mas cercanos:  [[array([[0.01543507]], dtype=float32), 151], [array([[0.40764526]], dtype=float32), 50], [array([[0.45962477]], dtype=float32), 184], [array([[0.5534906]], dtype=float32), 98], [array([[0.97059333]], dtype=float32), 46]]


Distancia:  [[0.15089686]]  al vector en la posicion:  184  desde el vector que se encuentra en:  40
Imagen de BD:  82  Imagen de consulta:  107
Mas cercanos:  [[array([[0.15089686]], dtype=float32), 184], [array([[0.22725128]], dtype=float32), 50], [array([[0.23503666]], dtype=float32), 98], [array([[0.7611425]], dtype=float32), 48], [array([[0.81645346]], dtype=float32), 46]]


Distancia:  [[0.0067924]]  al vector en la posicion:  50  desde el vector que se encuentra en:  41
Imagen de BD:  144  Imagen de consulta:  112
Mas cercanos:  [[array([[0.0067924]], dtype=float32), 50], [array([[0.13619976]], dtype=float32), 98], [array([[0.20744906]], dtype=float32), 184], [array([[0.3345052]], dtype=float32), 47], [array([[0.5125481]], dtype=float32), 160]]


Distancia:  [[0.01495193]]  al vector en la posicion:  153  desde el vector que se encuentra en:  42
Imagen de BD:  54  Imagen de consulta:  113
Mas cercanos:  [[array([[0.01495193]], dtype=float32), 153], [array([[0.27136505]], dtype=float32), 76], [array([[0.39906865]], dtype=float32), 54], [array([[0.49385262]], dtype=float32), 103], [array([[0.53874075]], dtype=float32), 65]]


Distancia:  [[0.02871278]]  al vector en la posicion:  76  desde el vector que se encuentra en:  43
Imagen de BD:  168  Imagen de consulta:  115
Mas cercanos:  [[array([[0.02871278]], dtype=float32), 76], [array([[0.05338359]], dtype=float32), 153], [array([[0.14327493]], dtype=float32), 54], [array([[0.14656913]], dtype=float32), 103], [array([[0.7610804]], dtype=float32), 65]]


Distancia:  [[0.18670769]]  al vector en la posicion:  151  desde el vector que se encuentra en:  44
Imagen de BD:  52  Imagen de consulta:  11
Mas cercanos:  [[array([[0.18670769]], dtype=float32), 151], [array([[0.51661885]], dtype=float32), 50], [array([[0.7210639]], dtype=float32), 76], [array([[0.77162707]], dtype=float32), 103], [array([[0.92613065]], dtype=float32), 48]]


Distancia:  [[0.2662177]]  al vector en la posicion:  98  desde el vector que se encuentra en:  45
Imagen de BD:  188  Imagen de consulta:  0
Mas cercanos:  [[array([[0.2662177]], dtype=float32), 98], [array([[0.3112473]], dtype=float32), 88], [array([[0.3575494]], dtype=float32), 184], [array([[0.49933618]], dtype=float32), 50], [array([[0.6097401]], dtype=float32), 46]]


Distancia:  [[0.04024667]]  al vector en la posicion:  98  desde el vector que se encuentra en:  46
Imagen de BD:  188  Imagen de consulta:  120
Mas cercanos:  [[array([[0.04024667]], dtype=float32), 98], [array([[0.05066377]], dtype=float32), 46], [array([[0.06635595]], dtype=float32), 55], [array([[0.09618517]], dtype=float32), 160], [array([[0.15886211]], dtype=float32), 91]]


Distancia:  [[0.00896459]]  al vector en la posicion:  184  desde el vector que se encuentra en:  47
Imagen de BD:  82  Imagen de consulta:  122
Mas cercanos:  [[array([[0.00896459]], dtype=float32), 184], [array([[0.02211931]], dtype=float32), 50], [array([[0.19190142]], dtype=float32), 98], [array([[0.839027]], dtype=float32), 94], [array([[0.9243701]], dtype=float32), 46]]


Distancia:  [[0.11006845]]  al vector en la posicion:  76  desde el vector que se encuentra en:  48
Imagen de BD:  168  Imagen de consulta:  125
Mas cercanos:  [[array([[0.11006845]], dtype=float32), 76], [array([[0.32810822]], dtype=float32), 151], [array([[0.6201813]], dtype=float32), 103], [array([[0.75624484]], dtype=float32), 153], [array([[0.77139103]], dtype=float32), 156]]


Distancia:  [[0.33376032]]  al vector en la posicion:  151  desde el vector que se encuentra en:  49
Imagen de BD:  52  Imagen de consulta:  139
Mas cercanos:  [[array([[0.33376032]], dtype=float32), 151], [array([[0.5923271]], dtype=float32), 76], [array([[0.9266108]], dtype=float32), 163], [array([[0.9537875]], dtype=float32), 44], [array([[0.96642005]], dtype=float32), 103]]
--> ACIERTO


Distancia:  [[0.00301313]]  al vector en la posicion:  50  desde el vector que se encuentra en:  50
Imagen de BD:  144  Imagen de consulta:  142
Mas cercanos:  [[array([[0.00301313]], dtype=float32), 50], [array([[0.01664533]], dtype=float32), 184], [array([[0.11139978]], dtype=float32), 98], [array([[0.67552924]], dtype=float32), 94], [array([[0.67827046]], dtype=float32), 47]]


Distancia:  [[0.01142058]]  al vector en la posicion:  50  desde el vector que se encuentra en:  51
Imagen de BD:  144  Imagen de consulta:  142
Mas cercanos:  [[array([[0.01142058]], dtype=float32), 50], [array([[0.0902774]], dtype=float32), 184], [array([[0.3092619]], dtype=float32), 98], [array([[0.8254584]], dtype=float32), 160], [array([[0.83475083]], dtype=float32), 156]]


Distancia:  [[0.00871919]]  al vector en la posicion:  98  desde el vector que se encuentra en:  52
Imagen de BD:  188  Imagen de consulta:  2
Mas cercanos:  [[array([[0.00871919]], dtype=float32), 98], [array([[0.09884403]], dtype=float32), 184], [array([[0.10581133]], dtype=float32), 46], [array([[0.20071967]], dtype=float32), 160], [array([[0.27130213]], dtype=float32), 55]]


Distancia:  [[0.05687656]]  al vector en la posicion:  184  desde el vector que se encuentra en:  53
Imagen de BD:  82  Imagen de consulta:  144
Mas cercanos:  [[array([[0.05687656]], dtype=float32), 184], [array([[0.07073119]], dtype=float32), 50], [array([[0.57416636]], dtype=float32), 160], [array([[0.6263002]], dtype=float32), 98], [array([[0.93143976]], dtype=float32), 47]]
--> ACIERTO


Distancia:  [[0.15387648]]  al vector en la posicion:  50  desde el vector que se encuentra en:  54
Imagen de BD:  144  Imagen de consulta:  147
Mas cercanos:  [[array([[0.15387648]], dtype=float32), 50], [array([[0.32317522]], dtype=float32), 184], [array([[0.51346976]], dtype=float32), 76], [array([[0.8908877]], dtype=float32), 98], [array([[0.9139544]], dtype=float32), 103]]


Distancia:  [[0.20721716]]  al vector en la posicion:  76  desde el vector que se encuentra en:  55
Imagen de BD:  168  Imagen de consulta:  11
Mas cercanos:  [[array([[0.20721716]], dtype=float32), 76], [array([[0.286188]], dtype=float32), 151], [array([[0.37315857]], dtype=float32), 103], [array([[0.5508481]], dtype=float32), 50], [array([[0.85009754]], dtype=float32), 153]]


Distancia:  [[0.13495128]]  al vector en la posicion:  76  desde el vector que se encuentra en:  56
Imagen de BD:  168  Imagen de consulta:  148
Mas cercanos:  [[array([[0.13495128]], dtype=float32), 76], [array([[0.69037324]], dtype=float32), 153], [array([[0.7317644]], dtype=float32), 50], [array([[0.7910825]], dtype=float32), 103], [array([[0.802609]], dtype=float32), 151]]


Distancia:  [[0.10485482]]  al vector en la posicion:  51  desde el vector que se encuentra en:  57
Imagen de BD:  145  Imagen de consulta:  150
Mas cercanos:  [[array([[0.10485482]], dtype=float32), 51], [array([[0.31702602]], dtype=float32), 175], [array([[0.4462289]], dtype=float32), 102], [array([[0.63789666]], dtype=float32), 104], [array([[0.7485304]], dtype=float32), 64]]


Distancia:  [[0.02057691]]  al vector en la posicion:  50  desde el vector que se encuentra en:  58
Imagen de BD:  144  Imagen de consulta:  156
Mas cercanos:  [[array([[0.02057691]], dtype=float32), 50], [array([[0.07253375]], dtype=float32), 184], [array([[0.19335438]], dtype=float32), 98], [array([[0.5015736]], dtype=float32), 160], [array([[0.8092933]], dtype=float32), 94]]


Distancia:  [[0.07768444]]  al vector en la posicion:  184  desde el vector que se encuentra en:  59
Imagen de BD:  82  Imagen de consulta:  157
Mas cercanos:  [[array([[0.07768444]], dtype=float32), 184], [array([[0.26019174]], dtype=float32), 98], [array([[0.28634053]], dtype=float32), 50], [array([[0.8519327]], dtype=float32), 156], [array([[0.86910397]], dtype=float32), 48]]


Distancia:  [[0.00729529]]  al vector en la posicion:  50  desde el vector que se encuentra en:  60
Imagen de BD:  144  Imagen de consulta:  157
Mas cercanos:  [[array([[0.00729529]], dtype=float32), 50], [array([[0.1489899]], dtype=float32), 184], [array([[0.4609729]], dtype=float32), 98], [array([[0.7804445]], dtype=float32), 160], [array([[0.9203307]], dtype=float32), 156]]


Distancia:  [[0.00653355]]  al vector en la posicion:  163  desde el vector que se encuentra en:  61
Imagen de BD:  63  Imagen de consulta:  158
Mas cercanos:  [[array([[0.00653355]], dtype=float32), 163], [array([[0.09519291]], dtype=float32), 65], [array([[0.49326712]], dtype=float32), 137], [array([[0.53029275]], dtype=float32), 151], [array([[0.9183035]], dtype=float32), 77]]
--> ACIERTO


Distancia:  [[0.19675794]]  al vector en la posicion:  184  desde el vector que se encuentra en:  62
Imagen de BD:  82  Imagen de consulta:  158
Mas cercanos:  [[array([[0.19675794]], dtype=float32), 184], [array([[0.35543647]], dtype=float32), 98], [array([[0.57284266]], dtype=float32), 88], [array([[0.62276614]], dtype=float32), 50], [array([[0.9144623]], dtype=float32), 151]]


Distancia:  [[0.11412178]]  al vector en la posicion:  98  desde el vector que se encuentra en:  63
Imagen de BD:  188  Imagen de consulta:  159
Mas cercanos:  [[array([[0.11412178]], dtype=float32), 98], [array([[0.38495952]], dtype=float32), 160], [array([[0.38916007]], dtype=float32), 184], [array([[0.40017328]], dtype=float32), 50], [array([[0.4777603]], dtype=float32), 88]]


Distancia:  [[0.10029104]]  al vector en la posicion:  77  desde el vector que se encuentra en:  64
Imagen de BD:  169  Imagen de consulta:  161
Mas cercanos:  [[array([[0.10029104]], dtype=float32), 77], [array([[0.27009878]], dtype=float32), 71], [array([[0.31963792]], dtype=float32), 91], [array([[0.3558567]], dtype=float32), 55], [array([[0.3629674]], dtype=float32), 72]]


Distancia:  [[0.2415128]]  al vector en la posicion:  47  desde el vector que se encuentra en:  65
Imagen de BD:  141  Imagen de consulta:  162
Mas cercanos:  [[array([[0.2415128]], dtype=float32), 47], [array([[0.24815306]], dtype=float32), 94], [array([[0.45897058]], dtype=float32), 50], [array([[0.5827019]], dtype=float32), 53], [array([[0.75308084]], dtype=float32), 152]]


Distancia:  [[0.20428756]]  al vector en la posicion:  50  desde el vector que se encuentra en:  66
Imagen de BD:  144  Imagen de consulta:  11
Mas cercanos:  [[array([[0.20428756]], dtype=float32), 50], [array([[0.6932219]], dtype=float32), 151], [array([[0.75919914]], dtype=float32), 76], [array([[0.7737342]], dtype=float32), 103], [array([[0.81704694]], dtype=float32), 184]]


Distancia:  [[0.01329491]]  al vector en la posicion:  50  desde el vector que se encuentra en:  67
Imagen de BD:  144  Imagen de consulta:  162
Mas cercanos:  [[array([[0.01329491]], dtype=float32), 50], [array([[0.25763056]], dtype=float32), 184], [array([[0.32638162]], dtype=float32), 47], [array([[0.33630317]], dtype=float32), 98], [array([[0.51165694]], dtype=float32), 94]]


Distancia:  [[0.02516332]]  al vector en la posicion:  98  desde el vector que se encuentra en:  68
Imagen de BD:  188  Imagen de consulta:  165
Mas cercanos:  [[array([[0.02516332]], dtype=float32), 98], [array([[0.03789663]], dtype=float32), 184], [array([[0.06706049]], dtype=float32), 50], [array([[0.2616378]], dtype=float32), 160], [array([[0.7014292]], dtype=float32), 48]]


Distancia:  [[0.02167466]]  al vector en la posicion:  153  desde el vector que se encuentra en:  69
Imagen de BD:  54  Imagen de consulta:  166
Mas cercanos:  [[array([[0.02167466]], dtype=float32), 153], [array([[0.09476142]], dtype=float32), 76], [array([[0.24990523]], dtype=float32), 103], [array([[0.28547874]], dtype=float32), 54], [array([[0.6000133]], dtype=float32), 60]]


Distancia:  [[0.11065858]]  al vector en la posicion:  184  desde el vector que se encuentra en:  70
Imagen de BD:  82  Imagen de consulta:  170
Mas cercanos:  [[array([[0.11065858]], dtype=float32), 184], [array([[0.1339005]], dtype=float32), 98], [array([[0.18162383]], dtype=float32), 50], [array([[0.359914]], dtype=float32), 160], [array([[0.8318202]], dtype=float32), 46]]


Distancia:  [[0.19329321]]  al vector en la posicion:  151  desde el vector que se encuentra en:  71
Imagen de BD:  52  Imagen de consulta:  171
Mas cercanos:  [[array([[0.19329321]], dtype=float32), 151], [array([[0.41800267]], dtype=float32), 163], [array([[0.7474085]], dtype=float32), 65], [array([[0.85858613]], dtype=float32), 98], [array([[0.8737226]], dtype=float32), 184]]


Distancia:  [[0.00242495]]  al vector en la posicion:  151  desde el vector que se encuentra en:  72
Imagen de BD:  52  Imagen de consulta:  175
Mas cercanos:  [[array([[0.00242495]], dtype=float32), 151], [array([[0.7002957]], dtype=float32), 163], [array([[0.8820686]], dtype=float32), 65], [array([[0.90218544]], dtype=float32), 98], [array([[0.94962656]], dtype=float32), 184]]


Distancia:  [[0.13790686]]  al vector en la posicion:  98  desde el vector que se encuentra en:  73
Imagen de BD:  188  Imagen de consulta:  0
Mas cercanos:  [[array([[0.13790686]], dtype=float32), 98], [array([[0.29032573]], dtype=float32), 46], [array([[0.2940722]], dtype=float32), 184], [array([[0.3340875]], dtype=float32), 151], [array([[0.64728534]], dtype=float32), 91]]


Distancia:  [[0.0891769]]  al vector en la posicion:  76  desde el vector que se encuentra en:  74
Imagen de BD:  168  Imagen de consulta:  183
Mas cercanos:  [[array([[0.0891769]], dtype=float32), 76], [array([[0.12872595]], dtype=float32), 151], [array([[0.14574201]], dtype=float32), 153], [array([[0.56137383]], dtype=float32), 103], [array([[0.79072064]], dtype=float32), 54]]


Distancia:  [[0.38034537]]  al vector en la posicion:  153  desde el vector que se encuentra en:  75
Imagen de BD:  54  Imagen de consulta:  185
Mas cercanos:  [[array([[0.38034537]], dtype=float32), 153], [array([[0.3993258]], dtype=float32), 103], [array([[0.4565534]], dtype=float32), 151], [array([[0.61086625]], dtype=float32), 76], [array([[0.83569443]], dtype=float32), 156]]


Distancia:  [[0.01056044]]  al vector en la posicion:  184  desde el vector que se encuentra en:  76
Imagen de BD:  82  Imagen de consulta:  188
Mas cercanos:  [[array([[0.01056044]], dtype=float32), 184], [array([[0.13756101]], dtype=float32), 98], [array([[0.14061297]], dtype=float32), 50], [array([[0.5412446]], dtype=float32), 46], [array([[0.81958884]], dtype=float32), 48]]
--> ACIERTO


Distancia:  [[0.01746516]]  al vector en la posicion:  50  desde el vector que se encuentra en:  77
Imagen de BD:  144  Imagen de consulta:  11
Mas cercanos:  [[array([[0.01746516]], dtype=float32), 50], [array([[0.29332486]], dtype=float32), 184], [array([[0.43756476]], dtype=float32), 98], [array([[0.8138771]], dtype=float32), 160], [array([[0.83395207]], dtype=float32), 88]]


Distancia:  [[0.04642164]]  al vector en la posicion:  50  desde el vector que se encuentra en:  78
Imagen de BD:  144  Imagen de consulta:  190
Mas cercanos:  [[array([[0.04642164]], dtype=float32), 50], [array([[0.13492788]], dtype=float32), 98], [array([[0.22038144]], dtype=float32), 184], [array([[0.6186994]], dtype=float32), 88], [array([[0.8624711]], dtype=float32), 46]]


Distancia:  [[0.03743692]]  al vector en la posicion:  50  desde el vector que se encuentra en:  79
Imagen de BD:  144  Imagen de consulta:  191
Mas cercanos:  [[array([[0.03743692]], dtype=float32), 50], [array([[0.04749738]], dtype=float32), 98], [array([[0.06152542]], dtype=float32), 184], [array([[0.316602]], dtype=float32), 47], [array([[0.34487045]], dtype=float32), 94]]


Distancia:  [[0.26046848]]  al vector en la posicion:  151  desde el vector que se encuentra en:  80
Imagen de BD:  52  Imagen de consulta:  197
Mas cercanos:  [[array([[0.26046848]], dtype=float32), 151], [array([[0.30291206]], dtype=float32), 65], [array([[0.6803941]], dtype=float32), 163], [array([[0.9116012]], dtype=float32), 54], [array([[0.9448977]], dtype=float32), 76]]


Distancia:  [[0.14310889]]  al vector en la posicion:  160  desde el vector que se encuentra en:  81
Imagen de BD:  60  Imagen de consulta:  200
Mas cercanos:  [[array([[0.14310889]], dtype=float32), 160], [array([[0.17627534]], dtype=float32), 88], [array([[0.22286956]], dtype=float32), 42], [array([[0.24702986]], dtype=float32), 50], [array([[0.26525152]], dtype=float32), 47]]


Distancia:  [[0.38944983]]  al vector en la posicion:  151  desde el vector que se encuentra en:  82
Imagen de BD:  52  Imagen de consulta:  11
Mas cercanos:  [[array([[0.38944983]], dtype=float32), 151], [array([[0.64043224]], dtype=float32), 50], [array([[0.7488373]], dtype=float32), 76], [array([[0.7793426]], dtype=float32), 103], [array([[0.93845844]], dtype=float32), 88]]


Porcentaje de aciertos:  0.07228915662650602
```


**[Celda 57 - Código]**
```python
# UTILIZANDO SIAMESA COMO FUNCION DE DISTANCIA: CONSULTAS POR SIMILITUD UTILIZANDO COMO BASE CONSULTAS ALEATORIAS TOMADAS DESDE LAS IMAGENES AUMENTADAS

import numpy as np
from keras.models import Model

#layer_name = 'flatten'
#extractor_consulta = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
#vector_consulta = extractor_consulta.predict(img5.reshape(1,28,28,1))
#a = np.array(vector_consulta[0])
contador_aciertos = 0
cantidad_consultas = 100
cantidad_BD = len(imagenesBD)

#Inicio del for consulta
for i in range(cantidad_consultas):
    iquery = random.randint(0, len(marcasBDA)-1)
    min = 20000
    nnklist = []
    # Inicio del for DB
    for j in range(cantidad_BD):
        #dist = np.sqrt(np.sum(np.square(a-b)))
        dist = siamese_net.predict([marcasBDA[iquery].reshape(1,28,28,1),imagenesBD[j].reshape(1,28,28,1)])
        nnk(nnklist,[dist,j],5)
        if dist < min:
            min = dist
            vector_cercano = b
            posicion_cercano = j

    print("Distancia: ", min, " al vector en la posicion: ", posicion_cercano, " desde el vector que se encuentra en: ", iquery)
    print("Imagen de BD: ", y_imagenesBD[posicion_cercano], " Imagen de consulta: ", y_marcasBDA[iquery])
    print('Mas cercanos: ',nnklist)
    for par in nnklist:
        if y_imagenesBD[par[1]] == y_marcasBDA[iquery]:
            contador_aciertos += 1
            print('--> ACIERTO')
    print("\n")

print("Porcentaje de aciertos: ", contador_aciertos/cantidad_consultas)
```


*Salida:*
```text
Distancia:  [[0.22701581]]  al vector en la posicion:  164  desde el vector que se encuentra en:  105973
Imagen de BD:  64  Imagen de consulta:  64
Mas cercanos:  [[array([[0.22701581]], dtype=float32), 164], [array([[0.24626553]], dtype=float32), 104], [array([[0.3650243]], dtype=float32), 68], [array([[0.3824224]], dtype=float32), 105], [array([[0.42568904]], dtype=float32), 152]]
--> ACIERTO


Distancia:  [[0.01293622]]  al vector en la posicion:  165  desde el vector que se encuentra en:  107249
Imagen de BD:  65  Imagen de consulta:  65
Mas cercanos:  [[array([[0.01293622]], dtype=float32), 165], [array([[0.11423641]], dtype=float32), 13], [array([[0.11924805]], dtype=float32), 133], [array([[0.25570184]], dtype=float32), 185], [array([[0.32567725]], dtype=float32), 124]]
--> ACIERTO


Distancia:  [[0.00948951]]  al vector en la posicion:  4  desde el vector que se encuentra en:  955
Imagen de BD:  102  Imagen de consulta:  102
Mas cercanos:  [[array([[0.00948951]], dtype=float32), 4], [array([[0.1563228]], dtype=float32), 34], [array([[0.1601105]], dtype=float32), 1], [array([[0.3328024]], dtype=float32), 166], [array([[0.35794345]], dtype=float32), 145]]
--> ACIERTO


Distancia:  [[0.00196196]]  al vector en la posicion:  160  desde el vector que se encuentra en:  97528
Imagen de BD:  60  Imagen de consulta:  60
Mas cercanos:  [[array([[0.00196196]], dtype=float32), 160], [array([[0.09786437]], dtype=float32), 98], [array([[0.30425444]], dtype=float32), 55], [array([[0.36936295]], dtype=float32), 184], [array([[0.582332]], dtype=float32), 46]]
--> ACIERTO


Distancia:  [[0.00642296]]  al vector en la posicion:  83  desde el vector que se encuentra en:  22696
Imagen de BD:  174  Imagen de consulta:  146
Mas cercanos:  [[array([[0.00642296]], dtype=float32), 83], [array([[0.00675553]], dtype=float32), 52], [array([[0.09094519]], dtype=float32), 82], [array([[0.1128839]], dtype=float32), 128], [array([[0.12656532]], dtype=float32), 38]]
--> ACIERTO


Distancia:  [[0.00385042]]  al vector en la posicion:  73  desde el vector que se encuentra en:  31963
Imagen de BD:  165  Imagen de consulta:  165
Mas cercanos:  [[array([[0.00385042]], dtype=float32), 73], [array([[0.56074363]], dtype=float32), 171], [array([[0.66094637]], dtype=float32), 99], [array([[0.80621606]], dtype=float32), 174], [array([[0.87026924]], dtype=float32), 61]]
--> ACIERTO


Distancia:  [[0.09283563]]  al vector en la posicion:  148  desde el vector que se encuentra en:  96910
Imagen de BD:  5  Imagen de consulta:  5
Mas cercanos:  [[array([[0.09283563]], dtype=float32), 148], [array([[0.25030905]], dtype=float32), 0], [array([[0.3250297]], dtype=float32), 115], [array([[0.48935378]], dtype=float32), 29], [array([[0.5781413]], dtype=float32), 179]]
--> ACIERTO


Distancia:  [[0.00967047]]  al vector en la posicion:  71  desde el vector que se encuentra en:  31168
Imagen de BD:  163  Imagen de consulta:  163
Mas cercanos:  [[array([[0.00967047]], dtype=float32), 71], [array([[0.06735168]], dtype=float32), 72], [array([[0.61350244]], dtype=float32), 77], [array([[0.70257175]], dtype=float32), 81], [array([[0.7195252]], dtype=float32), 55]]
--> ACIERTO


Distancia:  [[0.02369034]]  al vector en la posicion:  150  desde el vector que se encuentra en:  80855
Imagen de BD:  51  Imagen de consulta:  51
Mas cercanos:  [[array([[0.02369034]], dtype=float32), 150], [array([[0.10687202]], dtype=float32), 69], [array([[0.2592021]], dtype=float32), 158], [array([[0.44124553]], dtype=float32), 179], [array([[0.7532282]], dtype=float32), 37]]
--> ACIERTO


Distancia:  [[0.00119993]]  al vector en la posicion:  155  desde el vector que se encuentra en:  90736
Imagen de BD:  56  Imagen de consulta:  56
Mas cercanos:  [[array([[0.00119993]], dtype=float32), 155], [array([[0.04061585]], dtype=float32), 25], [array([[0.10185087]], dtype=float32), 132], [array([[0.6349727]], dtype=float32), 8], [array([[0.68408036]], dtype=float32), 27]]
--> ACIERTO


Distancia:  [[0.02403055]]  al vector en la posicion:  80  desde el vector que se encuentra en:  35395
Imagen de BD:  171  Imagen de consulta:  171
Mas cercanos:  [[array([[0.02403055]], dtype=float32), 80], [array([[0.20996825]], dtype=float32), 195], [array([[0.2913417]], dtype=float32), 9], [array([[0.42392245]], dtype=float32), 164], [array([[0.8385486]], dtype=float32), 157]]
--> ACIERTO


Distancia:  [[0.00322368]]  al vector en la posicion:  103  desde el vector que se encuentra en:  45883
Imagen de BD:  192  Imagen de consulta:  192
Mas cercanos:  [[array([[0.00322368]], dtype=float32), 103], [array([[0.01665696]], dtype=float32), 156], [array([[0.0897562]], dtype=float32), 76], [array([[0.09413298]], dtype=float32), 48], [array([[0.5091706]], dtype=float32), 153]]
--> ACIERTO


Distancia:  [[0.00124276]]  al vector en la posicion:  179  desde el vector que se encuentra en:  131472
Imagen de BD:  78  Imagen de consulta:  78
Mas cercanos:  [[array([[0.00124276]], dtype=float32), 179], [array([[0.37416136]], dtype=float32), 15], [array([[0.57661396]], dtype=float32), 11], [array([[0.79473567]], dtype=float32), 189], [array([[0.8442189]], dtype=float32), 150]]
--> ACIERTO


Distancia:  [[0.02791084]]  al vector en la posicion:  147  desde el vector que se encuentra en:  78086
Imagen de BD:  49  Imagen de consulta:  49
Mas cercanos:  [[array([[0.02791084]], dtype=float32), 147], [array([[0.06913014]], dtype=float32), 7], [array([[0.3474341]], dtype=float32), 177], [array([[0.81601703]], dtype=float32), 21], [array([[0.8592974]], dtype=float32), 174]]
--> ACIERTO


Distancia:  [[0.01249316]]  al vector en la posicion:  19  desde el vector que se encuentra en:  8059
Imagen de BD:  116  Imagen de consulta:  116
Mas cercanos:  [[array([[0.01249316]], dtype=float32), 19], [array([[0.22898045]], dtype=float32), 135], [array([[0.48151326]], dtype=float32), 167], [array([[0.5417283]], dtype=float32), 176], [array([[0.72781813]], dtype=float32), 146]]
--> ACIERTO


Distancia:  [[0.00388919]]  al vector en la posicion:  117  desde el vector que se encuentra en:  22742
Imagen de BD:  21  Imagen de consulta:  146
Mas cercanos:  [[array([[0.00388919]], dtype=float32), 117], [array([[0.02521115]], dtype=float32), 52], [array([[0.08090327]], dtype=float32), 83], [array([[0.0976637]], dtype=float32), 82], [array([[0.2648063]], dtype=float32), 62]]
--> ACIERTO


Distancia:  [[0.00618916]]  al vector en la posicion:  55  desde el vector que se encuentra en:  40226
Imagen de BD:  149  Imagen de consulta:  181
Mas cercanos:  [[array([[0.00618916]], dtype=float32), 55], [array([[0.03117805]], dtype=float32), 46], [array([[0.05437973]], dtype=float32), 160], [array([[0.06552453]], dtype=float32), 91], [array([[0.06693409]], dtype=float32), 72]]
--> ACIERTO


Distancia:  [[0.0096516]]  al vector en la posicion:  135  desde el vector que se encuentra en:  60377
Imagen de BD:  38  Imagen de consulta:  38
Mas cercanos:  [[array([[0.0096516]], dtype=float32), 135], [array([[0.04751019]], dtype=float32), 136], [array([[0.13198772]], dtype=float32), 126], [array([[0.8393948]], dtype=float32), 149], [array([[0.9104845]], dtype=float32), 146]]
--> ACIERTO


Distancia:  [[0.02985689]]  al vector en la posicion:  117  desde el vector que se encuentra en:  52613
Imagen de BD:  21  Imagen de consulta:  21
Mas cercanos:  [[array([[0.02985689]], dtype=float32), 117], [array([[0.24060631]], dtype=float32), 63], [array([[0.5295122]], dtype=float32), 195], [array([[0.5859761]], dtype=float32), 95], [array([[0.6470133]], dtype=float32), 174]]
--> ACIERTO


Distancia:  [[0.00455218]]  al vector en la posicion:  23  desde el vector que se encuentra en:  14567
Imagen de BD:  12  Imagen de consulta:  12
Mas cercanos:  [[array([[0.00455218]], dtype=float32), 23], [array([[0.2050738]], dtype=float32), 146], [array([[0.29528388]], dtype=float32), 3], [array([[0.5905347]], dtype=float32), 139], [array([[0.67193484]], dtype=float32), 167]]
--> ACIERTO


Distancia:  [[0.00062706]]  al vector en la posicion:  171  desde el vector que se encuentra en:  116396
Imagen de BD:  70  Imagen de consulta:  70
Mas cercanos:  [[array([[0.00062706]], dtype=float32), 171], [array([[0.05404238]], dtype=float32), 174], [array([[0.06441991]], dtype=float32), 89], [array([[0.25267646]], dtype=float32), 73], [array([[0.56604505]], dtype=float32), 99]]
--> ACIERTO


Distancia:  [[0.00128408]]  al vector en la posicion:  182  desde el vector que se encuentra en:  134394
Imagen de BD:  80  Imagen de consulta:  80
Mas cercanos:  [[array([[0.00128408]], dtype=float32), 182], [array([[0.22466747]], dtype=float32), 178], [array([[0.77709776]], dtype=float32), 120], [array([[0.84073764]], dtype=float32), 157], [array([[0.9108669]], dtype=float32), 163]]
--> ACIERTO


Distancia:  [[0.01688357]]  al vector en la posicion:  142  desde el vector que se encuentra en:  68866
Imagen de BD:  44  Imagen de consulta:  44
Mas cercanos:  [[array([[0.01688357]], dtype=float32), 142], [array([[0.22082165]], dtype=float32), 108], [array([[0.5837607]], dtype=float32), 159], [array([[0.7008771]], dtype=float32), 59], [array([[0.7059206]], dtype=float32), 93]]
--> ACIERTO


Distancia:  [[0.01763482]]  al vector en la posicion:  158  desde el vector que se encuentra en:  95707
Imagen de BD:  59  Imagen de consulta:  59
Mas cercanos:  [[array([[0.01763482]], dtype=float32), 158], [array([[0.33351472]], dtype=float32), 150], [array([[0.37871838]], dtype=float32), 121], [array([[0.38758487]], dtype=float32), 6], [array([[0.43036693]], dtype=float32), 129]]
--> ACIERTO


Distancia:  [[0.00657659]]  al vector en la posicion:  196  desde el vector que se encuentra en:  142000
Imagen de BD:  93  Imagen de consulta:  93
Mas cercanos:  [[array([[0.00657659]], dtype=float32), 196], [array([[0.07042681]], dtype=float32), 127], [array([[0.14311968]], dtype=float32), 27], [array([[0.18264067]], dtype=float32), 107], [array([[0.19900812]], dtype=float32), 121]]
--> ACIERTO


Distancia:  [[0.20600934]]  al vector en la posicion:  106  desde el vector que se encuentra en:  35611
Imagen de BD:  195  Imagen de consulta:  172
Mas cercanos:  [[array([[0.20600934]], dtype=float32), 106], [array([[0.22338343]], dtype=float32), 81], [array([[0.24076442]], dtype=float32), 71], [array([[0.7920371]], dtype=float32), 49], [array([[0.82693183]], dtype=float32), 156]]
--> ACIERTO


Distancia:  [[0.06895881]]  al vector en la posicion:  147  desde el vector que se encuentra en:  77455
Imagen de BD:  49  Imagen de consulta:  49
Mas cercanos:  [[array([[0.06895881]], dtype=float32), 147], [array([[0.11766849]], dtype=float32), 7], [array([[0.37196723]], dtype=float32), 177], [array([[0.7191484]], dtype=float32), 89], [array([[0.9341779]], dtype=float32), 174]]
--> ACIERTO


Distancia:  [[0.00857498]]  al vector en la posicion:  81  desde el vector que se encuentra en:  35985
Imagen de BD:  172  Imagen de consulta:  172
Mas cercanos:  [[array([[0.00857498]], dtype=float32), 81], [array([[0.06286947]], dtype=float32), 106], [array([[0.31197986]], dtype=float32), 71], [array([[0.5785667]], dtype=float32), 55], [array([[0.64516187]], dtype=float32), 68]]
--> ACIERTO


Distancia:  [[0.00470335]]  al vector en la posicion:  20  desde el vector que se encuentra en:  8158
Imagen de BD:  117  Imagen de consulta:  117
Mas cercanos:  [[array([[0.00470335]], dtype=float32), 20], [array([[0.20917058]], dtype=float32), 69], [array([[0.8102801]], dtype=float32), 44], [array([[0.8656973]], dtype=float32), 179], [array([[0.87898856]], dtype=float32), 113]]
--> ACIERTO


Distancia:  [[0.00446073]]  al vector en la posicion:  164  desde el vector que se encuentra en:  104924
Imagen de BD:  64  Imagen de consulta:  64
Mas cercanos:  [[array([[0.00446073]], dtype=float32), 164], [array([[0.15567261]], dtype=float32), 105], [array([[0.2236844]], dtype=float32), 41], [array([[0.367788]], dtype=float32), 157], [array([[0.53085005]], dtype=float32), 61]]
--> ACIERTO


Distancia:  [[0.00337948]]  al vector en la posicion:  128  desde el vector que se encuentra en:  57564
Imagen de BD:  31  Imagen de consulta:  31
Mas cercanos:  [[array([[0.00337948]], dtype=float32), 128], [array([[0.01785087]], dtype=float32), 93], [array([[0.03757844]], dtype=float32), 38], [array([[0.03937144]], dtype=float32), 83], [array([[0.04175051]], dtype=float32), 52]]
--> ACIERTO


Distancia:  [[0.02720634]]  al vector en la posicion:  177  desde el vector que se encuentra en:  127499
Imagen de BD:  76  Imagen de consulta:  76
Mas cercanos:  [[array([[0.02720634]], dtype=float32), 177], [array([[0.38835195]], dtype=float32), 147], [array([[0.6712272]], dtype=float32), 164], [array([[0.7592431]], dtype=float32), 41], [array([[0.85710025]], dtype=float32), 59]]
--> ACIERTO


Distancia:  [[0.0370349]]  al vector en la posicion:  119  desde el vector que se encuentra en:  53369
Imagen de BD:  23  Imagen de consulta:  23
Mas cercanos:  [[array([[0.0370349]], dtype=float32), 119], [array([[0.10300957]], dtype=float32), 196], [array([[0.1442317]], dtype=float32), 127], [array([[0.56014687]], dtype=float32), 40], [array([[0.6408689]], dtype=float32), 194]]
--> ACIERTO


Distancia:  [[0.00122573]]  al vector en la posicion:  41  desde el vector que se encuentra en:  17932
Imagen de BD:  136  Imagen de consulta:  136
Mas cercanos:  [[array([[0.00122573]], dtype=float32), 41], [array([[0.02217085]], dtype=float32), 57], [array([[0.41750783]], dtype=float32), 5], [array([[0.4436225]], dtype=float32), 164], [array([[0.66209495]], dtype=float32), 61]]
--> ACIERTO


Distancia:  [[0.00441251]]  al vector en la posicion:  102  desde el vector que se encuentra en:  45155
Imagen de BD:  191  Imagen de consulta:  191
Mas cercanos:  [[array([[0.00441251]], dtype=float32), 102], [array([[0.19403909]], dtype=float32), 96], [array([[0.46526644]], dtype=float32), 106], [array([[0.5923103]], dtype=float32), 68], [array([[0.64888585]], dtype=float32), 79]]
--> ACIERTO


Distancia:  [[0.02331281]]  al vector en la posicion:  45  desde el vector que se encuentra en:  24681
Imagen de BD:  14  Imagen de consulta:  14
Mas cercanos:  [[array([[0.02331281]], dtype=float32), 45], [array([[0.27439234]], dtype=float32), 140], [array([[0.75644875]], dtype=float32), 170], [array([[0.87941754]], dtype=float32), 33], [array([[0.95549196]], dtype=float32), 153]]
--> ACIERTO


Distancia:  [[0.0112183]]  al vector en la posicion:  149  desde el vector que se encuentra en:  80069
Imagen de BD:  50  Imagen de consulta:  50
Mas cercanos:  [[array([[0.0112183]], dtype=float32), 149], [array([[0.26836687]], dtype=float32), 138], [array([[0.6550777]], dtype=float32), 136], [array([[0.81163466]], dtype=float32), 145], [array([[0.83228683]], dtype=float32), 39]]
--> ACIERTO


Distancia:  [[0.00151274]]  al vector en la posicion:  182  desde el vector que se encuentra en:  134347
Imagen de BD:  80  Imagen de consulta:  80
Mas cercanos:  [[array([[0.00151274]], dtype=float32), 182], [array([[0.19762099]], dtype=float32), 178], [array([[0.6720577]], dtype=float32), 157], [array([[0.8420718]], dtype=float32), 163], [array([[0.8443642]], dtype=float32), 134]]
--> ACIERTO


Distancia:  [[0.00933201]]  al vector en la posicion:  123  desde el vector que se encuentra en:  55055
Imagen de BD:  27  Imagen de consulta:  27
Mas cercanos:  [[array([[0.00933201]], dtype=float32), 123], [array([[0.0351222]], dtype=float32), 176], [array([[0.10072852]], dtype=float32), 60], [array([[0.86225003]], dtype=float32), 7], [array([[0.91919416]], dtype=float32), 53]]
--> ACIERTO


Distancia:  [[0.00098216]]  al vector en la posicion:  180  desde el vector que se encuentra en:  133418
Imagen de BD:  79  Imagen de consulta:  79
Mas cercanos:  [[array([[0.00098216]], dtype=float32), 180], [array([[0.06833412]], dtype=float32), 5], [array([[0.28304002]], dtype=float32), 95], [array([[0.37535104]], dtype=float32), 101], [array([[0.5584972]], dtype=float32), 2]]
--> ACIERTO


Distancia:  [[0.00379964]]  al vector en la posicion:  21  desde el vector que se encuentra en:  8880
Imagen de BD:  118  Imagen de consulta:  118
Mas cercanos:  [[array([[0.00379964]], dtype=float32), 21], [array([[0.03227345]], dtype=float32), 22], [array([[0.20644544]], dtype=float32), 35], [array([[0.30678722]], dtype=float32), 148], [array([[0.5735241]], dtype=float32), 29]]
--> ACIERTO


Distancia:  [[0.02299995]]  al vector en la posicion:  45  desde el vector que se encuentra en:  24746
Imagen de BD:  14  Imagen de consulta:  14
Mas cercanos:  [[array([[0.02299995]], dtype=float32), 45], [array([[0.18180297]], dtype=float32), 140], [array([[0.90380067]], dtype=float32), 170], [array([[0.9211762]], dtype=float32), 153], [array([[0.95878166]], dtype=float32), 84]]
--> ACIERTO


Distancia:  [[0.01122576]]  al vector en la posicion:  102  desde el vector que se encuentra en:  45299
Imagen de BD:  191  Imagen de consulta:  191
Mas cercanos:  [[array([[0.01122576]], dtype=float32), 102], [array([[0.3759326]], dtype=float32), 92], [array([[0.45067063]], dtype=float32), 47], [array([[0.50938034]], dtype=float32), 104], [array([[0.6388518]], dtype=float32), 106]]
--> ACIERTO


Distancia:  [[0.02909621]]  al vector en la posicion:  158  desde el vector que se encuentra en:  96504
Imagen de BD:  59  Imagen de consulta:  59
Mas cercanos:  [[array([[0.02909621]], dtype=float32), 158], [array([[0.56227726]], dtype=float32), 118], [array([[0.76842344]], dtype=float32), 128], [array([[0.8140257]], dtype=float32), 132], [array([[0.8400599]], dtype=float32), 6]]
--> ACIERTO


Distancia:  [[0.00748872]]  al vector en la posicion:  115  desde el vector que se encuentra en:  51150
Imagen de BD:  202  Imagen de consulta:  202
Mas cercanos:  [[array([[0.00748872]], dtype=float32), 115], [array([[0.18178423]], dtype=float32), 23], [array([[0.290663]], dtype=float32), 189], [array([[0.3910503]], dtype=float32), 201], [array([[0.4513318]], dtype=float32), 11]]
--> ACIERTO


Distancia:  [[0.00131377]]  al vector en la posicion:  167  desde el vector que se encuentra en:  110632
Imagen de BD:  67  Imagen de consulta:  67
Mas cercanos:  [[array([[0.00131377]], dtype=float32), 167], [array([[0.08174801]], dtype=float32), 146], [array([[0.4495335]], dtype=float32), 13], [array([[0.5353173]], dtype=float32), 126], [array([[0.62136644]], dtype=float32), 19]]
--> ACIERTO


Distancia:  [[0.02299598]]  al vector en la posicion:  47  desde el vector que se encuentra en:  20420
Imagen de BD:  141  Imagen de consulta:  141
Mas cercanos:  [[array([[0.02299598]], dtype=float32), 47], [array([[0.07715665]], dtype=float32), 42], [array([[0.41684985]], dtype=float32), 104], [array([[0.49665192]], dtype=float32), 75], [array([[0.65507823]], dtype=float32), 92]]
--> ACIERTO


Distancia:  [[0.05263701]]  al vector en la posicion:  31  desde el vector que se encuentra en:  56886
Imagen de BD:  127  Imagen de consulta:  30
Mas cercanos:  [[array([[0.05263701]], dtype=float32), 31], [array([[0.11069048]], dtype=float32), 3], [array([[0.11749859]], dtype=float32), 30], [array([[0.18973267]], dtype=float32), 126], [array([[0.19348545]], dtype=float32), 127]]
--> ACIERTO


Distancia:  [[0.04573318]]  al vector en la posicion:  68  desde el vector que se encuentra en:  30119
Imagen de BD:  160  Imagen de consulta:  160
Mas cercanos:  [[array([[0.04573318]], dtype=float32), 68], [array([[0.05365212]], dtype=float32), 106], [array([[0.16094628]], dtype=float32), 104], [array([[0.26667762]], dtype=float32), 102], [array([[0.35245016]], dtype=float32), 79]]
--> ACIERTO


Distancia:  [[0.00909653]]  al vector en la posicion:  156  desde el vector que se encuentra en:  92422
Imagen de BD:  57  Imagen de consulta:  57
Mas cercanos:  [[array([[0.00909653]], dtype=float32), 156], [array([[0.04467993]], dtype=float32), 81], [array([[0.2970276]], dtype=float32), 106], [array([[0.34564638]], dtype=float32), 103], [array([[0.6488061]], dtype=float32), 71]]
--> ACIERTO


Distancia:  [[0.00643297]]  al vector en la posicion:  45  desde el vector que se encuentra en:  24578
Imagen de BD:  14  Imagen de consulta:  14
Mas cercanos:  [[array([[0.00643297]], dtype=float32), 45], [array([[0.3888046]], dtype=float32), 140], [array([[0.885756]], dtype=float32), 33], [array([[0.91441566]], dtype=float32), 170], [array([[0.9205518]], dtype=float32), 169]]
--> ACIERTO


Distancia:  [[0.00398973]]  al vector en la posicion:  82  desde el vector que se encuentra en:  36346
Imagen de BD:  173  Imagen de consulta:  173
Mas cercanos:  [[array([[0.00398973]], dtype=float32), 82], [array([[0.02580415]], dtype=float32), 52], [array([[0.05119313]], dtype=float32), 38], [array([[0.07339742]], dtype=float32), 83], [array([[0.16080625]], dtype=float32), 117]]
--> ACIERTO


Distancia:  [[0.00292885]]  al vector en la posicion:  193  desde el vector que se encuentra en:  143078
Imagen de BD:  90  Imagen de consulta:  95
Mas cercanos:  [[array([[0.00292885]], dtype=float32), 193], [array([[0.00679538]], dtype=float32), 198], [array([[0.01211482]], dtype=float32), 199], [array([[0.0728729]], dtype=float32), 110], [array([[0.39788425]], dtype=float32), 68]]
--> ACIERTO


Distancia:  [[0.02903562]]  al vector en la posicion:  169  desde el vector que se encuentra en:  115195
Imagen de BD:  69  Imagen de consulta:  69
Mas cercanos:  [[array([[0.02903562]], dtype=float32), 169], [array([[0.5014271]], dtype=float32), 35], [array([[0.695992]], dtype=float32), 33], [array([[0.811163]], dtype=float32), 21], [array([[0.86915857]], dtype=float32), 148]]
--> ACIERTO


Distancia:  [[0.00764936]]  al vector en la posicion:  112  desde el vector que se encuentra en:  51972
Imagen de BD:  20  Imagen de consulta:  20
Mas cercanos:  [[array([[0.00764936]], dtype=float32), 112], [array([[0.03994794]], dtype=float32), 32], [array([[0.44456372]], dtype=float32), 131], [array([[0.71518993]], dtype=float32), 194], [array([[0.72612154]], dtype=float32), 125]]
--> ACIERTO


Distancia:  [[0.0041409]]  al vector en la posicion:  202  desde el vector que se encuentra en:  144897
Imagen de BD:  99  Imagen de consulta:  99
Mas cercanos:  [[array([[0.0041409]], dtype=float32), 202], [array([[0.14376962]], dtype=float32), 162], [array([[0.49432218]], dtype=float32), 28], [array([[0.737025]], dtype=float32), 143], [array([[0.8838865]], dtype=float32), 159]]
--> ACIERTO


Distancia:  [[0.01504719]]  al vector en la posicion:  24  desde el vector que se encuentra en:  38671
Imagen de BD:  120  Imagen de consulta:  178
Mas cercanos:  [[array([[0.01504719]], dtype=float32), 24], [array([[0.05637822]], dtype=float32), 26], [array([[0.0846546]], dtype=float32), 197], [array([[0.11833932]], dtype=float32), 87], [array([[0.3657518]], dtype=float32), 107]]
--> ACIERTO


Distancia:  [[0.19179738]]  al vector en la posicion:  115  desde el vector que se encuentra en:  112094
Imagen de BD:  202  Imagen de consulta:  68
Mas cercanos:  [[array([[0.19179738]], dtype=float32), 115], [array([[0.31528187]], dtype=float32), 187], [array([[0.35010618]], dtype=float32), 168], [array([[0.40608773]], dtype=float32), 122], [array([[0.41474754]], dtype=float32), 17]]
--> ACIERTO


Distancia:  [[0.00248031]]  al vector en la posicion:  178  desde el vector que se encuentra en:  129135
Imagen de BD:  77  Imagen de consulta:  77
Mas cercanos:  [[array([[0.00248031]], dtype=float32), 178], [array([[0.6531332]], dtype=float32), 182], [array([[0.7188618]], dtype=float32), 116], [array([[0.84986293]], dtype=float32), 157], [array([[0.8592598]], dtype=float32), 59]]
--> ACIERTO


Distancia:  [[0.04917414]]  al vector en la posicion:  102  desde el vector que se encuentra en:  45068
Imagen de BD:  191  Imagen de consulta:  191
Mas cercanos:  [[array([[0.04917414]], dtype=float32), 102], [array([[0.35481846]], dtype=float32), 152], [array([[0.57497144]], dtype=float32), 106], [array([[0.63146156]], dtype=float32), 63], [array([[0.666467]], dtype=float32), 104]]
--> ACIERTO


Distancia:  [[0.04448284]]  al vector en la posicion:  158  desde el vector que se encuentra en:  95142
Imagen de BD:  59  Imagen de consulta:  59
Mas cercanos:  [[array([[0.04448284]], dtype=float32), 158], [array([[0.68648064]], dtype=float32), 150], [array([[0.76319724]], dtype=float32), 118], [array([[0.7645041]], dtype=float32), 128], [array([[0.78781587]], dtype=float32), 69]]
--> ACIERTO


Distancia:  [[0.00650433]]  al vector en la posicion:  140  desde el vector que se encuentra en:  65568
Imagen de BD:  42  Imagen de consulta:  42
Mas cercanos:  [[array([[0.00650433]], dtype=float32), 140], [array([[0.10911746]], dtype=float32), 45], [array([[0.83668715]], dtype=float32), 16], [array([[0.8997356]], dtype=float32), 170], [array([[0.90534174]], dtype=float32), 169]]
--> ACIERTO


Distancia:  [[0.00085666]]  al vector en la posicion:  116  desde el vector que se encuentra en:  51520
Imagen de BD:  203  Imagen de consulta:  203
Mas cercanos:  [[array([[0.00085666]], dtype=float32), 116], [array([[0.00786487]], dtype=float32), 74], [array([[0.14003499]], dtype=float32), 41], [array([[0.19150423]], dtype=float32), 113], [array([[0.23847423]], dtype=float32), 38]]
--> ACIERTO


Distancia:  [[0.00177934]]  al vector en la posicion:  174  desde el vector que se encuentra en:  122726
Imagen de BD:  73  Imagen de consulta:  73
Mas cercanos:  [[array([[0.00177934]], dtype=float32), 174], [array([[0.01115856]], dtype=float32), 171], [array([[0.01313078]], dtype=float32), 89], [array([[0.67827666]], dtype=float32), 54], [array([[0.7166239]], dtype=float32), 73]]
--> ACIERTO


Distancia:  [[0.01343283]]  al vector en la posicion:  176  desde el vector que se encuentra en:  125769
Imagen de BD:  75  Imagen de consulta:  75
Mas cercanos:  [[array([[0.01343283]], dtype=float32), 176], [array([[0.20433895]], dtype=float32), 123], [array([[0.5221273]], dtype=float32), 60], [array([[0.5346067]], dtype=float32), 137], [array([[0.6357537]], dtype=float32), 70]]
--> ACIERTO


Distancia:  [[0.10244022]]  al vector en la posicion:  192  desde el vector que se encuentra en:  145169
Imagen de BD:  9  Imagen de consulta:  9
Mas cercanos:  [[array([[0.10244022]], dtype=float32), 192], [array([[0.22420134]], dtype=float32), 187], [array([[0.25425792]], dtype=float32), 78], [array([[0.60059935]], dtype=float32), 115], [array([[0.69925624]], dtype=float32), 188]]
--> ACIERTO


Distancia:  [[0.00117272]]  al vector en la posicion:  47  desde el vector que se encuentra en:  20689
Imagen de BD:  141  Imagen de consulta:  141
Mas cercanos:  [[array([[0.00117272]], dtype=float32), 47], [array([[0.06338052]], dtype=float32), 79], [array([[0.1008439]], dtype=float32), 96], [array([[0.11557971]], dtype=float32), 94], [array([[0.24842681]], dtype=float32), 104]]
--> ACIERTO


Distancia:  [[0.04692546]]  al vector en la posicion:  47  desde el vector que se encuentra en:  18183
Imagen de BD:  141  Imagen de consulta:  137
Mas cercanos:  [[array([[0.04692546]], dtype=float32), 47], [array([[0.05833451]], dtype=float32), 94], [array([[0.15137368]], dtype=float32), 42], [array([[0.22893505]], dtype=float32), 88], [array([[0.49473566]], dtype=float32), 75]]
--> ACIERTO


Distancia:  [[0.00197738]]  al vector en la posicion:  179  desde el vector que se encuentra en:  130438
Imagen de BD:  78  Imagen de consulta:  78
Mas cercanos:  [[array([[0.00197738]], dtype=float32), 179], [array([[0.08202448]], dtype=float32), 15], [array([[0.6144072]], dtype=float32), 0], [array([[0.73385316]], dtype=float32), 11], [array([[0.7455157]], dtype=float32), 29]]
--> ACIERTO


Distancia:  [[0.00774311]]  al vector en la posicion:  177  desde el vector que se encuentra en:  127828
Imagen de BD:  76  Imagen de consulta:  76
Mas cercanos:  [[array([[0.00774311]], dtype=float32), 177], [array([[0.84130025]], dtype=float32), 147], [array([[0.8454724]], dtype=float32), 164], [array([[0.8939382]], dtype=float32), 7], [array([[0.9292029]], dtype=float32), 89]]
--> ACIERTO


Distancia:  [[0.0064547]]  al vector en la posicion:  139  desde el vector que se encuentra en:  62159
Imagen de BD:  41  Imagen de consulta:  41
Mas cercanos:  [[array([[0.0064547]], dtype=float32), 139], [array([[0.74169165]], dtype=float32), 18], [array([[0.7431936]], dtype=float32), 32], [array([[0.7805624]], dtype=float32), 148], [array([[0.84992045]], dtype=float32), 187]]
--> ACIERTO


Distancia:  [[0.00897696]]  al vector en la posicion:  153  desde el vector que se encuentra en:  87308
Imagen de BD:  54  Imagen de consulta:  54
Mas cercanos:  [[array([[0.00897696]], dtype=float32), 153], [array([[0.09127241]], dtype=float32), 103], [array([[0.35907322]], dtype=float32), 65], [array([[0.4990194]], dtype=float32), 151], [array([[0.62004346]], dtype=float32), 48]]
--> ACIERTO


Distancia:  [[0.00133619]]  al vector en la posicion:  176  desde el vector que se encuentra en:  125079
Imagen de BD:  75  Imagen de consulta:  75
Mas cercanos:  [[array([[0.00133619]], dtype=float32), 176], [array([[0.3823593]], dtype=float32), 137], [array([[0.79437304]], dtype=float32), 70], [array([[0.8176695]], dtype=float32), 123], [array([[0.8729729]], dtype=float32), 80]]
--> ACIERTO


Distancia:  [[0.01249581]]  al vector en la posicion:  141  desde el vector que se encuentra en:  65980
Imagen de BD:  43  Imagen de consulta:  43
Mas cercanos:  [[array([[0.01249581]], dtype=float32), 141], [array([[0.01470887]], dtype=float32), 159], [array([[0.5120247]], dtype=float32), 59], [array([[0.78790236]], dtype=float32), 89], [array([[0.86119175]], dtype=float32), 142]]
--> ACIERTO


Distancia:  [[0.04424887]]  al vector en la posicion:  97  desde el vector que se encuentra en:  42850
Imagen de BD:  187  Imagen de consulta:  187
Mas cercanos:  [[array([[0.04424887]], dtype=float32), 97], [array([[0.43814233]], dtype=float32), 92], [array([[0.68231565]], dtype=float32), 102], [array([[0.8137423]], dtype=float32), 79], [array([[0.9261917]], dtype=float32), 90]]
--> ACIERTO


Distancia:  [[0.06423334]]  al vector en la posicion:  112  desde el vector que se encuentra en:  52021
Imagen de BD:  20  Imagen de consulta:  20
Mas cercanos:  [[array([[0.06423334]], dtype=float32), 112], [array([[0.20998394]], dtype=float32), 32], [array([[0.32596153]], dtype=float32), 125], [array([[0.36057648]], dtype=float32), 183], [array([[0.36888912]], dtype=float32), 14]]
--> ACIERTO


Distancia:  [[0.01175278]]  al vector en la posicion:  140  desde el vector que se encuentra en:  65021
Imagen de BD:  42  Imagen de consulta:  42
Mas cercanos:  [[array([[0.01175278]], dtype=float32), 140], [array([[0.13315935]], dtype=float32), 45], [array([[0.7850425]], dtype=float32), 16], [array([[0.913339]], dtype=float32), 12], [array([[0.92336524]], dtype=float32), 11]]
--> ACIERTO


Distancia:  [[0.02075563]]  al vector en la posicion:  167  desde el vector que se encuentra en:  110557
Imagen de BD:  67  Imagen de consulta:  67
Mas cercanos:  [[array([[0.02075563]], dtype=float32), 167], [array([[0.03751855]], dtype=float32), 183], [array([[0.04513408]], dtype=float32), 3], [array([[0.05017083]], dtype=float32), 17], [array([[0.09477453]], dtype=float32), 126]]
--> ACIERTO


Distancia:  [[0.11031038]]  al vector en la posicion:  87  desde el vector que se encuentra en:  38483
Imagen de BD:  178  Imagen de consulta:  178
Mas cercanos:  [[array([[0.11031038]], dtype=float32), 87], [array([[0.23231368]], dtype=float32), 24], [array([[0.42117432]], dtype=float32), 26], [array([[0.55539864]], dtype=float32), 130], [array([[0.6321399]], dtype=float32), 186]]
--> ACIERTO


Distancia:  [[0.01247011]]  al vector en la posicion:  166  desde el vector que se encuentra en:  108737
Imagen de BD:  66  Imagen de consulta:  66
Mas cercanos:  [[array([[0.01247011]], dtype=float32), 166], [array([[0.15992771]], dtype=float32), 67], [array([[0.19362795]], dtype=float32), 119], [array([[0.56083715]], dtype=float32), 32], [array([[0.5728412]], dtype=float32), 109]]
--> ACIERTO


Distancia:  [[0.0049476]]  al vector en la posicion:  154  desde el vector que se encuentra en:  88600
Imagen de BD:  55  Imagen de consulta:  55
Mas cercanos:  [[array([[0.0049476]], dtype=float32), 154], [array([[0.40755296]], dtype=float32), 93], [array([[0.65264416]], dtype=float32), 22], [array([[0.73662376]], dtype=float32), 142], [array([[0.79414535]], dtype=float32), 159]]
--> ACIERTO


Distancia:  [[0.00516247]]  al vector en la posicion:  138  desde el vector que se encuentra en:  62022
Imagen de BD:  40  Imagen de consulta:  40
Mas cercanos:  [[array([[0.00516247]], dtype=float32), 138], [array([[0.0342323]], dtype=float32), 145], [array([[0.20368311]], dtype=float32), 136], [array([[0.23467222]], dtype=float32), 149], [array([[0.5646877]], dtype=float32), 32]]
--> ACIERTO


Distancia:  [[0.01126524]]  al vector en la posicion:  172  desde el vector que se encuentra en:  117998
Imagen de BD:  71  Imagen de consulta:  71
Mas cercanos:  [[array([[0.01126524]], dtype=float32), 172], [array([[0.01566308]], dtype=float32), 40], [array([[0.03585409]], dtype=float32), 37], [array([[0.5428471]], dtype=float32), 119], [array([[0.7865128]], dtype=float32), 129]]
--> ACIERTO


Distancia:  [[0.00127441]]  al vector en la posicion:  166  desde el vector que se encuentra en:  108525
Imagen de BD:  66  Imagen de consulta:  66
Mas cercanos:  [[array([[0.00127441]], dtype=float32), 166], [array([[0.11070652]], dtype=float32), 32], [array([[0.6147209]], dtype=float32), 109], [array([[0.6201683]], dtype=float32), 119], [array([[0.65263474]], dtype=float32), 4]]
--> ACIERTO


Distancia:  [[0.00925811]]  al vector en la posicion:  144  desde el vector que se encuentra en:  71433
Imagen de BD:  46  Imagen de consulta:  46
Mas cercanos:  [[array([[0.00925811]], dtype=float32), 144], [array([[0.16646901]], dtype=float32), 197], [array([[0.41314766]], dtype=float32), 191], [array([[0.44207218]], dtype=float32), 120], [array([[0.50498056]], dtype=float32), 26]]
--> ACIERTO


Distancia:  [[0.0060973]]  al vector en la posicion:  116  desde el vector que se encuentra en:  51538
Imagen de BD:  203  Imagen de consulta:  203
Mas cercanos:  [[array([[0.0060973]], dtype=float32), 116], [array([[0.16121607]], dtype=float32), 74], [array([[0.42085585]], dtype=float32), 157], [array([[0.5372265]], dtype=float32), 93], [array([[0.5969781]], dtype=float32), 5]]
--> ACIERTO


Distancia:  [[0.00231682]]  al vector en la posicion:  171  desde el vector que se encuentra en:  116599
Imagen de BD:  70  Imagen de consulta:  70
Mas cercanos:  [[array([[0.00231682]], dtype=float32), 171], [array([[0.01969078]], dtype=float32), 174], [array([[0.06149078]], dtype=float32), 89], [array([[0.47878832]], dtype=float32), 70], [array([[0.7355412]], dtype=float32), 43]]
--> ACIERTO


Distancia:  [[0.00625675]]  al vector en la posicion:  126  desde el vector que se encuentra en:  111399
Imagen de BD:  3  Imagen de consulta:  67
Mas cercanos:  [[array([[0.00625675]], dtype=float32), 126], [array([[0.01585032]], dtype=float32), 135], [array([[0.05881222]], dtype=float32), 167], [array([[0.26049185]], dtype=float32), 146], [array([[0.65448403]], dtype=float32), 13]]
--> ACIERTO


Distancia:  [[0.00806335]]  al vector en la posicion:  120  desde el vector que se encuentra en:  72026
Imagen de BD:  24  Imagen de consulta:  46
Mas cercanos:  [[array([[0.00806335]], dtype=float32), 120], [array([[0.05663371]], dtype=float32), 144], [array([[0.27700943]], dtype=float32), 35], [array([[0.34671813]], dtype=float32), 182], [array([[0.47909135]], dtype=float32), 190]]
--> ACIERTO


Distancia:  [[0.00101284]]  al vector en la posicion:  164  desde el vector que se encuentra en:  105631
Imagen de BD:  64  Imagen de consulta:  64
Mas cercanos:  [[array([[0.00101284]], dtype=float32), 164], [array([[0.15067366]], dtype=float32), 41], [array([[0.30748972]], dtype=float32), 105], [array([[0.39597976]], dtype=float32), 177], [array([[0.62372553]], dtype=float32), 64]]
--> ACIERTO


Distancia:  [[0.00498342]]  al vector en la posicion:  142  desde el vector que se encuentra en:  68002
Imagen de BD:  44  Imagen de consulta:  44
Mas cercanos:  [[array([[0.00498342]], dtype=float32), 142], [array([[0.10238544]], dtype=float32), 108], [array([[0.4975132]], dtype=float32), 116], [array([[0.6557951]], dtype=float32), 159], [array([[0.7479592]], dtype=float32), 41]]
--> ACIERTO


Distancia:  [[0.01135288]]  al vector en la posicion:  82  desde el vector que se encuentra en:  16561
Imagen de BD:  173  Imagen de consulta:  133
Mas cercanos:  [[array([[0.01135288]], dtype=float32), 82], [array([[0.03688907]], dtype=float32), 38], [array([[0.14505994]], dtype=float32), 83], [array([[0.14849561]], dtype=float32), 49], [array([[0.18212236]], dtype=float32), 62]]
--> ACIERTO


Distancia:  [[0.00719634]]  al vector en la posicion:  176  desde el vector que se encuentra en:  126038
Imagen de BD:  75  Imagen de consulta:  75
Mas cercanos:  [[array([[0.00719634]], dtype=float32), 176], [array([[0.09395999]], dtype=float32), 123], [array([[0.43320906]], dtype=float32), 137], [array([[0.51078296]], dtype=float32), 60], [array([[0.6012989]], dtype=float32), 70]]
--> ACIERTO


Distancia:  [[0.00604254]]  al vector en la posicion:  85  desde el vector que se encuentra en:  37466
Imagen de BD:  176  Imagen de consulta:  176
Mas cercanos:  [[array([[0.00604254]], dtype=float32), 85], [array([[0.17269392]], dtype=float32), 104], [array([[0.17392798]], dtype=float32), 92], [array([[0.33769748]], dtype=float32), 152], [array([[0.47308272]], dtype=float32), 195]]
--> ACIERTO


Distancia:  [[0.00110455]]  al vector en la posicion:  162  desde el vector que se encuentra en:  101945
Imagen de BD:  62  Imagen de consulta:  62
Mas cercanos:  [[array([[0.00110455]], dtype=float32), 162], [array([[0.09235992]], dtype=float32), 192], [array([[0.11481733]], dtype=float32), 125], [array([[0.23926085]], dtype=float32), 202], [array([[0.27812138]], dtype=float32), 36]]
--> ACIERTO


Distancia:  [[0.01250223]]  al vector en la posicion:  54  desde el vector que se encuentra en:  23660
Imagen de BD:  148  Imagen de consulta:  148
Mas cercanos:  [[array([[0.01250223]], dtype=float32), 54], [array([[0.09393869]], dtype=float32), 153], [array([[0.7169969]], dtype=float32), 76], [array([[0.8653227]], dtype=float32), 171], [array([[0.891593]], dtype=float32), 152]]
--> ACIERTO


Distancia:  [[0.00213237]]  al vector en la posicion:  40  desde el vector que se encuentra en:  119242
Imagen de BD:  135  Imagen de consulta:  71
Mas cercanos:  [[array([[0.00213237]], dtype=float32), 40], [array([[0.00267507]], dtype=float32), 172], [array([[0.16529259]], dtype=float32), 119], [array([[0.17585747]], dtype=float32), 37], [array([[0.81900877]], dtype=float32), 39]]
--> ACIERTO


Distancia:  [[0.00160637]]  al vector en la posicion:  161  desde el vector que se encuentra en:  99170
Imagen de BD:  61  Imagen de consulta:  61
Mas cercanos:  [[array([[0.00160637]], dtype=float32), 161], [array([[0.02146506]], dtype=float32), 67], [array([[0.44214666]], dtype=float32), 166], [array([[0.6405527]], dtype=float32), 119], [array([[0.643882]], dtype=float32), 109]]
--> ACIERTO


Distancia:  [[0.00650819]]  al vector en la posicion:  135  desde el vector que se encuentra en:  60726
Imagen de BD:  38  Imagen de consulta:  38
Mas cercanos:  [[array([[0.00650819]], dtype=float32), 135], [array([[0.0072656]], dtype=float32), 136], [array([[0.03845652]], dtype=float32), 126], [array([[0.59824526]], dtype=float32), 149], [array([[0.73417586]], dtype=float32), 100]]
--> ACIERTO


Distancia:  [[0.00199963]]  al vector en la posicion:  144  desde el vector que se encuentra en:  71389
Imagen de BD:  46  Imagen de consulta:  46
Mas cercanos:  [[array([[0.00199963]], dtype=float32), 144], [array([[0.04162592]], dtype=float32), 120], [array([[0.433812]], dtype=float32), 15], [array([[0.56519914]], dtype=float32), 197], [array([[0.8222335]], dtype=float32), 187]]
--> ACIERTO


Porcentaje de aciertos:  1.0
```


**[Celda 58 - Código]**
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
