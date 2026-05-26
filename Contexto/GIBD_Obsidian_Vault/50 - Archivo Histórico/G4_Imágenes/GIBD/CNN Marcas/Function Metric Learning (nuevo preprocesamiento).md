---
aliases: [Function Metric Learning (nuevo preprocesamiento)]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2023-06-27
origen_zip: GIBD-20260521T205218Z-3-006.zip
ruta_interna: "GIBD/CNN Marcas/Notebooks/Metric Learning/Function Metric Learning (nuevo preprocesamiento).ipynb"
tamanio_bytes: 1948815
---

# Notebook: Function Metric Learning (nuevo preprocesamiento).ipynb

Ruta interna: `GIBD/CNN Marcas/Notebooks/Metric Learning/Function Metric Learning (nuevo preprocesamiento).ipynb`

---


**[Celda 1 - Código]**
```python
#Declaraciones 
from tensorflow.keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D,Activation, Dropout, BatchNormalization, Concatenate # merge,
from tensorflow.keras.models import Model, Sequential
#from keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from skimage.io import imshow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.cm as cm
import math
import os
import cv2
import keras.engine as KE
import keras.backend as K
import time
import heapq
BATCH_SIZE = 128
ANCHO = 28
ALTO = 28
RADIO = 5
N_CLASES = 203
DIRECTORIO = r'D:\Escritorio\GIBD'
```

# Carga de imágenes
###### Código necesario para la carga de las imágenes y armado de los pares


**[Celda 3 - Código]**
```python
#Cargar imágenes entrenamiento

inicio= time.time()
# Selección del directorio que contiene la base de datos de marcas (con aumentación y relieve) 
directorio = DIRECTORIO + '\Bases de datos\BDAumentada2023'

# Definición del diccionario
dic = {}
for i in range(1, 204):
    dic[i] = []
# Se recorre y cargan las imagenes del directorio en las variables    
contenido = os.listdir(directorio)
for fichero in contenido:  
    # Si se utiliza el directorio con relieve: .txt / Si se utiliza el directorio sin relieve: .jpg    
    if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.txt'):  
        image = directorio + '/' + fichero
        relieve = np.loadtxt(image, dtype=float, delimiter= ',')
        image_gris = relieve.reshape(28, 28, 1)
        nombre = fichero[:fichero.find('.')]
        n = nombre[:nombre.find('_')]
        #if len(nombre)>2:
        #    n = nombre[:nombre.find('_')]
        #else:
        #    n=nombre
        dic[int(n)].append(image_gris) 
        
fin = time.time()
sum = 0
for key in dic.keys():
    sum += len(dic[key])
print(str(sum)+ " imágenes cargadas, tiempo de ejecucion: " + str(fin-inicio))    
```


*Salida:*
```text
20909 imágenes cargadas, tiempo de ejecucion: 103.3790009021759
```


**[Celda 4 - Código]**
```python
print("cantidad de imágenes: " + str(sum))
for key in dic.keys():
    print("Clase: " + str(key) + " Cantidad de imágenes: " + str(len(dic[key]))) 
```


*Salida:*
```text
cantidad de imágenes: 20909
Clase: 1 Cantidad de imágenes: 103
Clase: 2 Cantidad de imágenes: 103
Clase: 3 Cantidad de imágenes: 103
Clase: 4 Cantidad de imágenes: 103
Clase: 5 Cantidad de imágenes: 103
Clase: 6 Cantidad de imágenes: 103
Clase: 7 Cantidad de imágenes: 103
Clase: 8 Cantidad de imágenes: 103
Clase: 9 Cantidad de imágenes: 103
Clase: 10 Cantidad de imágenes: 103
Clase: 11 Cantidad de imágenes: 103
Clase: 12 Cantidad de imágenes: 103
Clase: 13 Cantidad de imágenes: 103
Clase: 14 Cantidad de imágenes: 103
Clase: 15 Cantidad de imágenes: 103
Clase: 16 Cantidad de imágenes: 103
Clase: 17 Cantidad de imágenes: 103
Clase: 18 Cantidad de imágenes: 103
Clase: 19 Cantidad de imágenes: 103
Clase: 20 Cantidad de imágenes: 103
Clase: 21 Cantidad de imágenes: 103
Clase: 22 Cantidad de imágenes: 103
Clase: 23 Cantidad de imágenes: 103
Clase: 24 Cantidad de imágenes: 103
Clase: 25 Cantidad de imágenes: 103
Clase: 26 Cantidad de imágenes: 103
Clase: 27 Cantidad de imágenes: 103
Clase: 28 Cantidad de imágenes: 103
Clase: 29 Cantidad de imágenes: 103
Clase: 30 Cantidad de imágenes: 103
Clase: 31 Cantidad de imágenes: 103
Clase: 32 Cantidad de imágenes: 103
Clase: 33 Cantidad de imágenes: 103
Clase: 34 Cantidad de imágenes: 103
Clase: 35 Cantidad de imágenes: 103
Clase: 36 Cantidad de imágenes: 103
Clase: 37 Cantidad de imágenes: 103
Clase: 38 Cantidad de imágenes: 103
Clase: 39 Cantidad de imágenes: 103
Clase: 40 Cantidad de imágenes: 103
Clase: 41 Cantidad de imágenes: 103
Clase: 42 Cantidad de imágenes: 103
Clase: 43 Cantidad de imágenes: 103
Clase: 44 Cantidad de imágenes: 103
Clase: 45 Cantidad de imágenes: 103
Clase: 46 Cantidad de imágenes: 103
Clase: 47 Cantidad de imágenes: 103
Clase: 48 Cantidad de imágenes: 103
Clase: 49 Cantidad de imágenes: 103
Clase: 50 Cantidad de imágenes: 103
Clase: 51 Cantidad de imágenes: 103
Clase: 52 Cantidad de imágenes: 103
Clase: 53 Cantidad de imágenes: 103
Clase: 54 Cantidad de imágenes: 103
Clase: 55 Cantidad de imágenes: 103
Clase: 56 Cantidad de imágenes: 103
Clase: 57 Cantidad de imágenes: 103
Clase: 58 Cantidad de imágenes: 103
Clase: 59 Cantidad de imágenes: 103
Clase: 60 Cantidad de imágenes: 103
Clase: 61 Cantidad de imágenes: 103
Clase: 62 Cantidad de imágenes: 103
Clase: 63 Cantidad de imágenes: 103
Clase: 64 Cantidad de imágenes: 103
Clase: 65 Cantidad de imágenes: 103
Clase: 66 Cantidad de imágenes: 103
Clase: 67 Cantidad de imágenes: 103
Clase: 68 Cantidad de imágenes: 103
Clase: 69 Cantidad de imágenes: 103
Clase: 70 Cantidad de imágenes: 103
Clase: 71 Cantidad de imágenes: 103
Clase: 72 Cantidad de imágenes: 103
Clase: 73 Cantidad de imágenes: 103
Clase: 74 Cantidad de imágenes: 103
Clase: 75 Cantidad de imágenes: 103
Clase: 76 Cantidad de imágenes: 103
Clase: 77 Cantidad de imágenes: 103
Clase: 78 Cantidad de imágenes: 103
Clase: 79 Cantidad de imágenes: 103
Clase: 80 Cantidad de imágenes: 103
Clase: 81 Cantidad de imágenes: 103
Clase: 82 Cantidad de imágenes: 103
Clase: 83 Cantidad de imágenes: 103
Clase: 84 Cantidad de imágenes: 103
Clase: 85 Cantidad de imágenes: 103
Clase: 86 Cantidad de imágenes: 103
Clase: 87 Cantidad de imágenes: 103
Clase: 88 Cantidad de imágenes: 103
Clase: 89 Cantidad de imágenes: 103
Clase: 90 Cantidad de imágenes: 103
Clase: 91 Cantidad de imágenes: 103
Clase: 92 Cantidad de imágenes: 103
Clase: 93 Cantidad de imágenes: 103
Clase: 94 Cantidad de imágenes: 103
Clase: 95 Cantidad de imágenes: 103
Clase: 96 Cantidad de imágenes: 103
Clase: 97 Cantidad de imágenes: 103
Clase: 98 Cantidad de imágenes: 103
Clase: 99 Cantidad de imágenes: 103
Clase: 100 Cantidad de imágenes: 103
Clase: 101 Cantidad de imágenes: 103
Clase: 102 Cantidad de imágenes: 103
Clase: 103 Cantidad de imágenes: 103
Clase: 104 Cantidad de imágenes: 103
Clase: 105 Cantidad de imágenes: 103
Clase: 106 Cantidad de imágenes: 103
Clase: 107 Cantidad de imágenes: 103
Clase: 108 Cantidad de imágenes: 103
Clase: 109 Cantidad de imágenes: 103
Clase: 110 Cantidad de imágenes: 103
Clase: 111 Cantidad de imágenes: 103
Clase: 112 Cantidad de imágenes: 103
Clase: 113 Cantidad de imágenes: 103
Clase: 114 Cantidad de imágenes: 103
Clase: 115 Cantidad de imágenes: 103
Clase: 116 Cantidad de imágenes: 103
Clase: 117 Cantidad de imágenes: 103
Clase: 118 Cantidad de imágenes: 103
Clase: 119 Cantidad de imágenes: 103
Clase: 120 Cantidad de imágenes: 103
Clase: 121 Cantidad de imágenes: 103
Clase: 122 Cantidad de imágenes: 103
Clase: 123 Cantidad de imágenes: 103
Clase: 124 Cantidad de imágenes: 103
Clase: 125 Cantidad de imágenes: 103
Clase: 126 Cantidad de imágenes: 103
Clase: 127 Cantidad de imágenes: 103
Clase: 128 Cantidad de imágenes: 103
Clase: 129 Cantidad de imágenes: 103
Clase: 130 Cantidad de imágenes: 103
Clase: 131 Cantidad de imágenes: 103
Clase: 132 Cantidad de imágenes: 103
Clase: 133 Cantidad de imágenes: 103
Clase: 134 Cantidad de imágenes: 103
Clase: 135 Cantidad de imágenes: 103
Clase: 136 Cantidad de imágenes: 103
Clase: 137 Cantidad de imágenes: 103
Clase: 138 Cantidad de imágenes: 103
Clase: 139 Cantidad de imágenes: 103
Clase: 140 Cantidad de imágenes: 103
Clase: 141 Cantidad de imágenes: 103
Clase: 142 Cantidad de imágenes: 103
Clase: 143 Cantidad de imágenes: 103
Clase: 144 Cantidad de imágenes: 103
Clase: 145 Cantidad de imágenes: 103
Clase: 146 Cantidad de imágenes: 103
Clase: 147 Cantidad de imágenes: 103
Clase: 148 Cantidad de imágenes: 103
Clase: 149 Cantidad de imágenes: 103
Clase: 150 Cantidad de imágenes: 103
Clase: 151 Cantidad de imágenes: 103
Clase: 152 Cantidad de imágenes: 103
Clase: 153 Cantidad de imágenes: 103
Clase: 154 Cantidad de imágenes: 103
Clase: 155 Cantidad de imágenes: 103
Clase: 156 Cantidad de imágenes: 103
Clase: 157 Cantidad de imágenes: 103
Clase: 158 Cantidad de imágenes: 103
Clase: 159 Cantidad de imágenes: 103
Clase: 160 Cantidad de imágenes: 103
Clase: 161 Cantidad de imágenes: 103
Clase: 162 Cantidad de imágenes: 103
Clase: 163 Cantidad de imágenes: 103
Clase: 164 Cantidad de imágenes: 103
Clase: 165 Cantidad de imágenes: 103
Clase: 166 Cantidad de imágenes: 103
Clase: 167 Cantidad de imágenes: 103
Clase: 168 Cantidad de imágenes: 103
Clase: 169 Cantidad de imágenes: 103
Clase: 170 Cantidad de imágenes: 103
Clase: 171 Cantidad de imágenes: 103
Clase: 172 Cantidad de imágenes: 103
Clase: 173 Cantidad de imágenes: 103
Clase: 174 Cantidad de imágenes: 103
Clase: 175 Cantidad de imágenes: 103
Clase: 176 Cantidad de imágenes: 103
Clase: 177 Cantidad de imágenes: 103
Clase: 178 Cantidad de imágenes: 103
Clase: 179 Cantidad de imágenes: 103
Clase: 180 Cantidad de imágenes: 103
Clase: 181 Cantidad de imágenes: 103
Clase: 182 Cantidad de imágenes: 103
Clase: 183 Cantidad de imágenes: 103
Clase: 184 Cantidad de imágenes: 103
Clase: 185 Cantidad de imágenes: 103
Clase: 186 Cantidad de imágenes: 103
Clase: 187 Cantidad de imágenes: 103
Clase: 188 Cantidad de imágenes: 103
Clase: 189 Cantidad de imágenes: 103
Clase: 190 Cantidad de imágenes: 103
Clase: 191 Cantidad de imágenes: 103
Clase: 192 Cantidad de imágenes: 103
Clase: 193 Cantidad de imágenes: 103
Clase: 194 Cantidad de imágenes: 103
Clase: 195 Cantidad de imágenes: 103
Clase: 196 Cantidad de imágenes: 103
Clase: 197 Cantidad de imágenes: 103
Clase: 198 Cantidad de imágenes: 103
Clase: 199 Cantidad de imágenes: 103
Clase: 200 Cantidad de imágenes: 103
Clase: 201 Cantidad de imágenes: 103
Clase: 202 Cantidad de imágenes: 103
Clase: 203 Cantidad de imágenes: 103
```


**[Celda 5 - Código]**
```python
def obtener_pares(porc_pares_similares = 0.5) :
    left_input = []
    right_input = []
    targets = []    
    pares_similares = 0
    #Numero de pares por clase
    pairs = 2000
    #Recorremos el arreglo de keys para asegurarnos de que se van a utilizar todas las imágenes
    for i in range(1,len(dic.keys())):
        #for j in range(pairs):
            #para tener una cantidad ajustable de pares similares 
            #if j < pairs*(1-porc_pares_similares):
            j = 1
            compare_to = i
            while j <= pairs*(1-porc_pares_similares):                
                #Nos aseguramos que sean pares negativos
                if compare_to == i:
                    compare_to += 1              
                i_left = random.randint(0, len(dic[i])-1)
                i_right = random.randint(0, len(dic[compare_to])-1) 
                left_input.append(dic[i][i_left])
                right_input.append(dic[compare_to][i_right])
                #Repetimos para que los pares x,y tambien aparezcan como y,x
                right_input.append(dic[i][i_left])
                left_input.append(dic[compare_to][i_right])
                #Como agregamos 2 pares para repetir xy,yx tambien ponemos 2 targets
                targets.append(1.)    
                targets.append(1.) 
                if compare_to < 203:
                    compare_to += 1
                else:
                    compare_to = 1
                j +=1
               
            while j < pairs:
                i_left = random.randint(0, len(dic[i])-1)
                i_right = random.randint(0, len(dic[i])-1)            
                left_input.append(dic[i][i_left])
                right_input.append(dic[i][i_right])
                #Repetimos para que los pares x,y tambien aparezcan como y,x
                right_input.append(dic[i][i_left])
                left_input.append(dic[i][i_right])
                #Como agregamos 2 pares para repetir xy,yx tambien ponemos 2 targets
                targets.append(0.)
                targets.append(0.)
                pares_similares += 2 
                j += 1

    print('Cantidad de pares similares: ', pares_similares)
    print('Cantidad total de pares: ', len(targets))


    #Mezclamos los resultados para que todas las tuplas tengan la probabilidad de entrar en el batch
    lista_tuplas =list((zip(left_input,right_input,targets)))
    random.shuffle(lista_tuplas)
    left_input,right_input,targets = zip(*lista_tuplas)

    left_input = np.squeeze(np.array(left_input))
    right_input = np.squeeze(np.array(right_input))
    targets = np.squeeze(np.array(targets))
    print(targets)
    return left_input, right_input, targets
```


**[Celda 6 - Código]**
```python
#Cargar imágenes db

directorio = DIRECTORIO + '\Bases de datos\BDNormalizada(28x28)'
contenido = os.listdir(directorio)
imagenesBD = []
y_imagenesBD = []
for fichero in contenido:
    if os.path.isfile(os.path.join(directorio, fichero)) and (fichero.endswith('.jpg') or fichero.endswith('.JPG')):
        image = directorio + '/' + fichero
        image_gris = cv2.imread(image, cv2.IMREAD_GRAYSCALE)        
        imagenesBD.append(preprocesarImagen(image_gris,ANCHO,ALTO,RADIO))
        n = fichero[:fichero.find('.')]
        y_imagenesBD.append(n)
print(y_imagenesBD)
#return imagenesBD, y_imagenesBD
```


*Salida:*
```text
['1', '10', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '11', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '12', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '13', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '14', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '15', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '16', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '17', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '18', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '19', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '2', '20', '200', '201', '202', '203', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '301', '302', '303', '304', '305', '306', '307', '308', '309', '31', '310', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']
```


**[Celda 7 - Código]**
```python
#Carga Queries dibujadas
directorio = r'D:\Escritorio\GIBD\Bases de datos\QueriesDibujadas'
#directorio = r'D:\Escritorio\GIBD\Bases de datos\NuevasMarcasQueries'
contenido = os.listdir(directorio)
consultas = []
y_consultas = []
for fichero in contenido:
    if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.jpg'):
        image = directorio + '/' + fichero
        image_gris = cv2.imread(image, cv2.IMREAD_GRAYSCALE)       
        consultas.append(preprocesarImagen(image_gris,ANCHO,ALTO,RADIO))
        nombre = fichero[:fichero.find('.')]
        #Código útil si se quiere cargar las imágenes de la db aumentada
        #if len(nombre)>2:
        #    n = nombre[:nombre.find('_')]
        #else:
        #    n=nombre       
        n=nombre
        y_consultas.append(n)
print(y_consultas)
```


*Salida:*
```text
['q_100', 'q_107', 'q_115', 'q_118', 'q_125', 'q_126', 'q_127', 'q_128', 'q_129', 'q_130', 'q_131', 'q_132', 'q_133', 'q_134 (2)', 'q_134', 'q_134', 'q_143', 'q_144', 'q_145', 'q_146', 'q_166', 'q_170', 'q_174', 'q_175', 'q_192', 'q_195', 'q_197', 'q_2', 'q_24', 'q_31', 'q_32', 'q_37', 'q_47', 'q_55', 'q_58', 'q_59', 'q_7', 'q_79', 'q_8', 'q_88', 'q_94']
```

# Modelo
###### Código necesario para el entrenamiento, creación, carga, y guardado de modelos


**[Celda 9 - Código]**
```python
def crear_modelo_viejo():
    VECTOR_SIZE = 128

    model = Sequential()

    model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(ANCHO,ALTO,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Flatten())
    return model
    
def crear_modelo_nuevo():
    VECTOR_SIZE = 256

    model = Sequential()

    model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(ANCHO,ALTO,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32,kernel_size=3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Conv2D(32,kernel_size=3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))          
    model.add(Conv2D(32,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(64,kernel_size=3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())

    #model.add(Dense(VECTOR_SIZE, activation='softmax'))
    return model
```


**[Celda 10 - Código]**
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
                                                                 
 activation (Activation)     (None, 24, 24, 32)        0         
                                                                 
 dropout (Dropout)           (None, 24, 24, 32)        0         
                                                                 
 conv2d_2 (Conv2D)           (None, 22, 22, 32)        9248      
                                                                 
 batch_normalization_2 (Batc  (None, 22, 22, 32)       128       
 hNormalization)                                                 
                                                                 
 activation_1 (Activation)   (None, 22, 22, 32)        0         
                                                                 
 conv2d_3 (Conv2D)           (None, 20, 20, 32)        9248      
                                                                 
 batch_normalization_3 (Batc  (None, 20, 20, 32)       128       
 hNormalization)                                                 
                                                                 
 max_pooling2d (MaxPooling2D  (None, 10, 10, 32)       0         
 )                                                               
                                                                 
 conv2d_4 (Conv2D)           (None, 8, 8, 64)          18496     
                                                                 
 batch_normalization_4 (Batc  (None, 8, 8, 64)         256       
 hNormalization)                                                 
                                                                 
 dropout_1 (Dropout)         (None, 8, 8, 64)          0         
                                                                 
 conv2d_5 (Conv2D)           (None, 6, 6, 64)          36928     
                                                                 
 batch_normalization_5 (Batc  (None, 6, 6, 64)         256       
 hNormalization)                                                 
                                                                 
 activation_2 (Activation)   (None, 6, 6, 64)          0         
                                                                 
 flatten (Flatten)           (None, 2304)              0         
                                                                 
=================================================================
Total params: 84,512
Trainable params: 84,000
Non-trainable params: 512
_________________________________________________________________
```


**[Celda 11 - Código]**
```python
#DEFINIMOS EL MODELO BASE, LE AGREGAMOS LOS INPUTS Y COMPILAMOS
#Creamos que modelo usar
#model = crear_modelo_viejo()
model = crear_modelo_nuevo()

#Agregamos 2 entradas
input_a = tf.keras.Input(shape=(ANCHO,ALTO,1))
input_b = tf.keras.Input(shape=(ANCHO,ALTO,1))

# Obtenemos los resultados de las imágenes y los concatenamos 
representation_a = model(input_a)
representation_b = model(input_b)
merged_representation = tf.keras.layers.concatenate([representation_a, representation_b])

# Agregamos capas adicionales para obtener la representación final con una funcion de activacion sigmoid porque va entre 0 y 1
final_rep = Dense(256, activation='relu')(merged_representation)
final_rep = Dense(128, activation='relu')(final_rep)
output = Dense(1, activation='sigmoid')(final_rep)

# Crea el modelo siamese
model_siamese = tf.keras.Model(inputs=[input_a, input_b], outputs=output)

#Compilamos el modelo
#optimizer=keras.optimizers.Adam(1e-3) - Se puede probar con 1e-4 o 3e-4 o 5e-4
model_siamese.compile(loss='BinaryCrossentropy', optimizer=keras.optimizers.Adam(1e-3), metrics=['accuracy'])
```


**[Celda 12 - Código]**
```python
x = dic[75][0].reshape(1,28,28,1)
x2 = dic[75][0].reshape(1,28,28,1)
model_siamese.predict([x,x2])
```


*Salida:*
```text
1/1 [==============================] - 0s 15ms/step
array([[1.0341255e-11]], dtype=float32)
```


**[Celda 13 - Código]**
```python
model_siamese = tf.keras.models.load_model(DIRECTORIO+r'\Entrenamientos\10ep_adam3-4_modviejo\2doEntrenamiento78.h5')
```


**[Celda 14 - Código]**
```python
# SIAMESA
#for i in range(0,2):
left_input, right_input, targets = obtener_pares(porc_pares_similares = 0.5)   
#mean_squared_error    
model_siamese.fit([left_input,right_input], targets,
          batch_size=128,
          #steps_per_epoch=625,
          epochs=5,
          verbose=1,
          validation_split=0.2)
```


*Salida:*
```text
Cantidad de pares similares:  403596
Cantidad total de pares:  807596
[0. 1. 1. ... 0. 1. 1.]
Epoch 1/10
5048/5048 [==============================] - 101s 20ms/step - loss: 4.1070e-04 - accuracy: 0.9999 - val_loss: 3.3057e-04 - val_accuracy: 0.9999
Epoch 2/10
5048/5048 [==============================] - 101s 20ms/step - loss: 3.2885e-04 - accuracy: 0.9999 - val_loss: 1.4723e-04 - val_accuracy: 1.0000
Epoch 3/10
5048/5048 [==============================] - 101s 20ms/step - loss: 3.3285e-04 - accuracy: 0.9999 - val_loss: 1.1212e-04 - val_accuracy: 1.0000
Epoch 4/10
5048/5048 [==============================] - 98s 19ms/step - loss: 4.7816e-04 - accuracy: 0.9999 - val_loss: 1.7075e-04 - val_accuracy: 1.0000
Epoch 5/10
5048/5048 [==============================] - 96s 19ms/step - loss: 3.2741e-04 - accuracy: 0.9999 - val_loss: 2.6793e-04 - val_accuracy: 1.0000
Epoch 6/10
5048/5048 [==============================] - 96s 19ms/step - loss: 2.8805e-04 - accuracy: 0.9999 - val_loss: 5.0104e-04 - val_accuracy: 0.9999
Epoch 7/10
5048/5048 [==============================] - 97s 19ms/step - loss: 3.5482e-04 - accuracy: 0.9999 - val_loss: 2.0165e-04 - val_accuracy: 1.0000
Epoch 8/10
5048/5048 [==============================] - 96s 19ms/step - loss: 2.0663e-04 - accuracy: 0.9999 - val_loss: 3.3709e-05 - val_accuracy: 1.0000
Epoch 9/10
5048/5048 [==============================] - 96s 19ms/step - loss: 3.3190e-04 - accuracy: 0.9999 - val_loss: 3.4032e-04 - val_accuracy: 1.0000
Epoch 10/10
5048/5048 [==============================] - 96s 19ms/step - loss: 3.5515e-04 - accuracy: 0.9999 - val_loss: 9.4083e-05 - val_accuracy: 1.0000
<keras.callbacks.History at 0x20af52c5420>
```


**[Celda 15 - Código]**
```python
#Cuando se encuentre un modelo con un porcentaje mayor al anterior guardar indicando ese porcentaje
model_siamese.save(DIRECTORIO+r'\Entrenamientos\10ep_adam1-3_modnuevo\6toEntrenamiento.h5')
```

# Evaluar resultados
###### Código necesario para cargar evaluar resultados


**[Celda 17 - Código]**
```python
#Crear arrays para las consultas
def crear_arrays_consultas():
    array_consultas = []
    array_imagenes_db = []
    for consulta in consultas:      
        for imagen_db in imagenesBD:
            array_consultas.append(consulta.reshape(28,28,1)) 
            array_imagenes_db.append(imagen_db.reshape(28,28,1))
    return array_consultas, array_imagenes_db
```


**[Celda 18 - Código]**
```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def imprimir_imagenes(lista_imagenes,labels):
    # Crea una figura los subplots
    fig, axs = plt.subplots(1, len(lista_imagenes))
    for i in range(len(lista_imagenes)):
        imagen = lista_imagenes[i].reshape(28,28)
        axs[i].imshow(imagen)
        axs[i].axis('off')
        axs[i].set_title(labels[i], loc="center")    
    # Ajusta el espacio entre los subplots
    plt.subplots_adjust(wspace=0.1)
    # Muestra la figura
    plt.show()
```


**[Celda 19 - Código]**
```python
# CONSULTAS POR SIMILITUD UTILIZANDO LA BASE DE CONSULTAS DIBUJADAS - OPTIMIZADO

imprimir_imagenes_resultado = True
array_consultas,array_imagenes_db = crear_arrays_consultas()

contador_aciertos_1 = 0
contador_aciertos_3 = 0
contador_aciertos_5 = 0
cantidad_consultas = len(consultas)
print("Cantidad de consultas: " + str(cantidad_consultas))


np_consultas = np.array(array_consultas)
np_imagenes_db = np.array(array_imagenes_db)

#Obtenemos los resultados de las predicciones
resultados_a = model_siamese.predict([np_imagenes_db,np_consultas])
resultados_b = model_siamese.predict([np_consultas, np_imagenes_db])
resultados = resultados_a+resultados_b
#Iteramos sobre las consultas para obtener los 5 vecinos mas cercanos para cada una e imprimimos resultados
for i in range(0,cantidad_consultas):
    resultados_imagen = resultados[i*len(imagenesBD):(i*len(imagenesBD))+len(imagenesBD)]   
    distancias = heapq.nsmallest(5,resultados_imagen)
    indices = []
    y_imagenes_resultado = []          
    for j in range(5):    
        indices.append(np.where(resultados_imagen==distancias[j])[0][0]) 
        y_imagenes_resultado.append(y_imagenesBD[indices[j]])  
        imagenes = [imagenesBD[x] for x in indices]
    #imagenes =  [consultas[i]]+ imagenes
    if np.isin(y_consultas[i][y_consultas[i].find('_')+1:],y_imagenes_resultado):
        contador_aciertos_5 +=1
    if np.isin(y_consultas[i][y_consultas[i].find('_')+1:],y_imagenes_resultado[:3]):
        contador_aciertos_3 +=1
    if np.isin(y_consultas[i][y_consultas[i].find('_')+1:],y_imagenes_resultado[:1]):
        contador_aciertos_1 +=1 
    print("Imagen:" , y_consultas[i][y_consultas[i].find('_')+1:])
    print("Resultados:", y_imagenes_resultado)
    print("Distancias:", (['{:0.5e}'.format(x[0]) for x in distancias]))       
    if(imprimir_imagenes_resultado):
        imprimir_imagenes([consultas[i]]+imagenes,[y_consultas[i]]+y_imagenes_resultado)
    
print()
print("Porcentaje de aciertos primera posicion:", contador_aciertos_1/cantidad_consultas , "  Cantidad aciertos: " + str(contador_aciertos_1)+"/"+str(cantidad_consultas))
print("Porcentaje de aciertos tercera posicion:", contador_aciertos_3/cantidad_consultas , "  Cantidad aciertos: " + str(contador_aciertos_3)+"/"+str(cantidad_consultas) )
print("Porcentaje de aciertos quinta  posicion:", contador_aciertos_5/cantidad_consultas , "  Cantidad aciertos: " + str(contador_aciertos_5)+"/"+str(cantidad_consultas) )
```


*Salida:*
```text
Cantidad de consultas: 41
273/273 [==============================] - 1s 3ms/step
273/273 [==============================] - 1s 2ms/step
Imagen: 100
Resultados: ['92', '1', '1', '1', '1']
Distancias: ['1.99998e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 107
Resultados: ['85', '107', '1', '1', '1']
Distancias: ['1.44469e-03', '9.90322e-01', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 115
Resultados: ['1', '1', '1', '1', '1']
Distancias: ['2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 118
Resultados: ['118', '55', '172', '1', '1']
Distancias: ['2.98471e-01', '6.40410e-01', '1.99960e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 125
Resultados: ['44', '70', '1', '1', '1']
Distancias: ['1.00002e+00', '1.99604e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 126
Resultados: ['11', '1', '1', '1', '1']
Distancias: ['1.99982e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 127
Resultados: ['127', '143', '1', '1', '1']
Distancias: ['1.06506e-05', '1.99994e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 128
Resultados: ['128', '40', '1', '1', '1']
Distancias: ['1.01130e+00', '1.99998e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 129
Resultados: ['129', '1', '1', '1', '1']
Distancias: ['3.41854e-04', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 130
Resultados: ['1', '1', '1', '1', '1']
Distancias: ['2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 131
Resultados: ['131', '16', '21', '1', '1']
Distancias: ['8.46092e-07', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 132
Resultados: ['56', '1', '1', '1', '1']
Distancias: ['1.99999e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 133
Resultados: ['133', '173', '1', '1', '1']
Distancias: ['2.66292e-03', '1.99641e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 134 (2)
Resultados: ['134', '1', '1', '1', '1']
Distancias: ['1.29663e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 134
Resultados: ['134', '1', '1', '1', '1']
Distancias: ['1.29663e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 134
Resultados: ['172', '178', '1', '1', '1']
Distancias: ['1.05958e+00', '1.16144e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 143
Resultados: ['143', '1', '1', '1', '1']
Distancias: ['2.53492e-03', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 144
Resultados: ['144', '1', '1', '1', '1']
Distancias: ['1.29946e-07', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 145
Resultados: ['145', '30', '1', '1', '1']
Distancias: ['9.81819e-01', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 146
Resultados: ['146', '1', '1', '1', '1']
Distancias: ['4.75945e-06', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 166
Resultados: ['166', '1', '1', '1', '1']
Distancias: ['5.80500e-08', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 170
Resultados: ['181', '1', '1', '1', '1']
Distancias: ['1.99981e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 174
Resultados: ['174', '301', '1', '1', '1']
Distancias: ['4.03184e-07', '1.00132e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 175
Resultados: ['69', '1', '1', '1', '1']
Distancias: ['1.17308e-01', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 192
Resultados: ['172', '183', '1', '1', '1']
Distancias: ['5.06989e-03', '1.98257e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 195
Resultados: ['195', '1', '1', '1', '1']
Distancias: ['1.99999e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 197
Resultados: ['197', '1', '1', '1', '1']
Distancias: ['1.99923e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 2
Resultados: ['2', '45', '1', '1', '1']
Distancias: ['1.32812e-05', '1.99450e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 24
Resultados: ['24', '1', '1', '1', '1']
Distancias: ['1.99767e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 31
Resultados: ['31', '132', '139', '1', '1']
Distancias: ['8.11409e-01', '1.64230e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 32
Resultados: ['32', '1', '1', '1', '1']
Distancias: ['6.96393e-11', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 37
Resultados: ['64', '63', '1', '1', '1']
Distancias: ['1.99914e+00', '1.99984e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 47
Resultados: ['194', '302', '1', '1', '1']
Distancias: ['1.45537e+00', '1.49707e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 55
Resultados: ['55', '1', '1', '1', '1']
Distancias: ['6.00413e-08', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 58
Resultados: ['58', '1', '1', '1', '1']
Distancias: ['1.55782e-11', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 59
Resultados: ['72', '1', '1', '1', '1']
Distancias: ['2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 7
Resultados: ['50', '91', '137', '1', '1']
Distancias: ['1.53478e+00', '1.98538e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 79
Resultados: ['70', '89', '1', '1', '1']
Distancias: ['1.62308e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 8
Resultados: ['8', '302', '304', '194', '1']
Distancias: ['1.84427e-02', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 88
Resultados: ['88', '75', '1', '1', '1']
Distancias: ['2.40696e-15', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>Imagen: 94
Resultados: ['94', '1', '1', '1', '1']
Distancias: ['1.78841e-06', '2.00000e+00', '2.00000e+00', '2.00000e+00', '2.00000e+00']
<Figure size 640x480 with 6 Axes>
Porcentaje de aciertos primera posicion: 0.5853658536585366   Cantidad aciertos: 24/41
Porcentaje de aciertos tercera posicion: 0.6097560975609756   Cantidad aciertos: 25/41
Porcentaje de aciertos quinta  posicion: 0.6097560975609756   Cantidad aciertos: 25/41
```

# Preprocesar Imágenes
###### Código necesario para realizar el procesamiento de las imágenes


**[Celda 21 - Código]**
```python
def preprocesarImagen(img, xMax, yMax, radio):
    # image_gris = img / 255
    image_gris = np.array(img)
    #print(image_gris)
    mat = crearMatrizDeImagen(image_gris, xMax, yMax)
    #print(mat)
    #escalada = escalarImagen(mat, xMax, yMax)
    escalada = escalarImagenConDeformacion(mat, xMax, yMax)
    matesc = binarizar(escalada, xMax, yMax)
    sk = matesc #skeletonize(matesc) 
    #imgcent = centrarImagen(sk, xMax, yMax)
    relieve = relieveSinCuda(sk, radio)
    relieve = normalizarUnitario(relieve)
    #image_gris = escalar(relieve, 0.7)
    #image_gris = relieve.reshape(xMax, xMax, 1)
    
    # VISUALIZACIONES #
    #print('IMAGEN:')
    #plt.imshow(image_gris, cmap=cm.Greys)
    #plt.show()
    #print('MATRIZ DE LA IMAGEN:')
    #plt.imshow(mat, cmap=cm.Greys)
    #plt.show()
    #print('ESCALADA:')
    #plt.imshow(escalada)
    #plt.show()
    #print('ESCALADA CON DEFORMACIÓN:')
    #plt.imshow(escalada, cmap=cm.Greys)
    #plt.show()
    #print('BINARIZADA:')
    #plt.imshow(matesc, cmap=cm.Greys)
    #plt.show()
    #print('SKELETONIZADA:')
    #plt.imshow(sk, cmap=cm.Greys)
    #plt.show()  
    #print('CENTRADA:')
    #plt.imshow(imgcent)
    #plt.show()
    #print('RELIEVE:')
    #plt.imshow(relieve, cmap=cm.Greys)
    #plt.show()
    
    return relieve.reshape(xMax, xMax, 1)  # .astype(np.float32) 
```


**[Celda 22 - Código]**
```python
def crearMatrizDeImagen(img,xMax,yMax):              
    
    tensor = tf.constant(img, dtype=tf.float32)
    media = tf.reduce_mean(tensor)
    desviacion_tipica = tf.math.reduce_std(tensor)
    umbral = media #- 6 * desviacion_tipica
    #print('media: ', media)
    #print('desviacion: ', desviacion_tipica)
    #print('umbral: ', umbral)
    
    mat = np.zeros((xMax, yMax), dtype=np.float32)                
    for x in range(xMax):                                         
        for y in range(yMax):
            if x<len(img) and y<len(img[0]):
                if img[x,y]>umbral:                                    
                    mat[x,y] = np.float32(1.0) # np.float32(1.0)   
                else:                                                 
                    mat[x,y] = np.int32(0)  #  np.int32(0)  
            else:
                mat[x,y] = np.float32(1.0) # np.float32(1.0)
    return mat    

def binarizar(img,xMax,yMax):                           
    mat = np.zeros((xMax, yMax), dtype=np.float32)  
    # print(img)
    for x in range(xMax):                                         
        for y in range(yMax):
            if x<len(img) and y<len(img[0]):
                if img[x,y]<1.0:                                    
                    mat[x,y] = np.float32(1.0) # np.float32(1.0)   
                else:                                                 
                    mat[x,y] = np.int32(0)  #  np.int32(0)  
            else:
                mat[x,y] = np.float32(1.0) # np.float32(1.0)
    return mat  


def escalarImagenConDeformacion(img, xmax, ymax):
    
    escalada = img

    despx = len(escalada)
    despy = len(escalada[0])

    #Obtenemos la coordenada en x e y donde inicia la imagen
    for x in range(len(escalada)):
        for y in range(len(escalada[0])):
            if escalada[x,y]<0.5: # or escalada[x,y]==True:
                if x < despx:
                    despx = x
                if y < despy:
                    despy = y

    #Obtenemos la coordenada en x e y donde finaliza la imagen
    despxx = despx
    despyy = despy
    for x in range(despx, len(escalada)):
        for y in range(despy, len(escalada[0])):
            if escalada[x,y]<0.5: # or escalada[x,y]==True:
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
    #print('x:',despx,'..',despxx)
    #print('y:',despy,'..',despyy)
    cropped_image = img[despx:despxx, despy:despyy]
    resized_image = cv2.resize(cropped_image, (xmax, ymax), interpolation=cv2.INTER_LINEAR)
    
    return resized_image





def relieveSinCuda(mat, radio): 
    # Precondicion: la matriz debe contener valores normalizados entre 0 y 1, donde 0 es el negro (el dibujo) y 1 es el blanco (el fondo)
    xmax = len(mat)
    ymax = len(mat[0])
    radio = int(radio)
    diam = radio * 2 
    resultado = np.zeros_like(mat,dtype=np.float64)
    for row in range(xmax):
        for col in range(ymax):                                                                    
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


**[Celda 23 - Código]**
```python
#Testeo preprocesamiento (testeamos si el procesamiento sobre las imagenes de entrenamiento es el mismo con el que vamos a cargar las consultas)
imagenes_a_testear = [dic[100][0],imagenesBD[2]]
imprimir_imagenes(imagenes_a_testear,["a","b"])
print(dic[10][0]==imagenesBD[1])
```


*Salida:*
```text
<Figure size 640x480 with 2 Axes>[[[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]

 [[ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]
  [ True]]]
```


**[Celda 24 - Código]**
```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if len(left_input) < 1:
    left_input, right_input, targets = obtener_pares(porc_pares_similares = 0.5) 
k=random.randint(0,len(list(targets))-1)
print('Numero de par: '+ str(k))
print('target: '+ str(targets[k]))
# Carga las imágenes
imagen1 = left_input[k].reshape(28,28)
imagen2 = right_input[k].reshape(28,28)

# Crea una figura con dos subplots
fig, axs = plt.subplots(1, 2)

# Muestra la primera imagen en el primer subplot
axs[0].imshow(imagen1)
axs[0].axis('off')

# Muestra la segunda imagen en el segundo subplot
axs[1].imshow(imagen2)
axs[1].axis('off')

# Ajusta el espacio entre los subplots
plt.subplots_adjust(wspace=0.1)

# Muestra la figura
plt.show()
```


*Salida:*
```text
Numero de par: 546461
target: 0.0
<Figure size 640x480 with 2 Axes>
```


**[Celda 25 - Código]**
```python
print(left_input[k] == right_input[k] )
```


*Salida:*
```text
[[False False False False False False False False False False False False
  False False False False False False False False False  True False False
  False False  True  True]
 [False False False False False False False False False False False False
  False False False False False False False False False False False  True
  False  True False False]
 [False False False False False False False False False False False False
  False False False False False False False False False False False False
  False False False False]
 [False False False False False False False False False False False False
  False False False False False False False False False False False False
  False False False False]
 [False False False False False False False False False False False False
  False False False False False False False False False False False False
  False False False False]
 [False False False False False False False False False False False False
  False False False False False False False False False False False False
  False False False False]
 [False False False False False False False False False False False False
  False False False False False False False False False False False False
  False False False False]
 [False False False False False False False False False False False False
  False False False False False False False False False False False False
  False False False False]
 [False False False False False False False False False False False False
  False False False False False False False False False False False False
  False False False False]
 [False False False False False False False False False False False False
  False False False False False False False False False False False False
   True False False False]
 [False False False False False False False False False False False False
  False False False False False False False False False False False False
  False False False False]
 [False False False False False False False False False False False False
  False False False False False False False False False False False False
  False False False False]
 [False False False False False False False False False False False False
  False False False False False False False False False False False False
  False False False False]
 [False False False False False False False False False False False False
  False False False False False False False False False False False False
  False False False False]
 [False False False False False False False False False False False False
  False False False False False False False False False False False False
  False False False False]
 [False False False False False False False False False False False False
  False False False False False False False False False False False False
  False False False False]
 [False False False False False False False False False False False False
  False False False False False False False False False False False False
  False False False False]
 [False False False False False False False False False False False False
  False False False False False False False False False False False False
  False False False False]
 [False False False False False False False  True False False False False
  False False False False False False False False False False False False
  False False False False]
 [False False False False False False False False False False False False
   True False False False  True False False False False False False False
  False False False False]
 [False False False False False False False  True False False False False
  False False False False False False False False False False False False
  False False False False]
 [False False False False False False False False False False False False
  False False False False False False False False False False False False
  False False False False]
 [False False False False False False False False False False False False
   True False False False False False False False False False False False
  False False False False]
 [False False False False False False False False False False False False
  False False False False False False False False False False False False
  False False False False]
 [False  True False False False False  True False False False False False
  False False False False False False False False False False False False
  False False False False]
 [False False False False False False False False False False False False
  False False False False False False False False False False False False
  False False False False]
 [False False False False False False False False False False False False
  False False False False False False False False False False False False
  False False False False]
 [False False False False False False False False False False False False
  False False False False False False False False False False False False
  False False False False]]
```


**[Celda 26 - Código]**
```python

```
