---
aliases: [TransferLearning_FineTuning_LogosClubes]
tags:
  - grupo/general
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2024-06-06
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Transfer Learning y Fine Tuning/TransferLearning_FineTuning_LogosClubes.ipynb"
tamanio_bytes: 2935963
---

# Notebook: TransferLearning_FineTuning_LogosClubes.ipynb

Ruta interna: `GIBD/Transfer Learning y Fine Tuning/TransferLearning_FineTuning_LogosClubes.ipynb`

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
    i = 1
    for path in image_paths:
        image = Image.open(path)
        
        # nombre = etiqueta
        filename = os.path.basename(path)
        nombre=filename[:filename.find(".")]
        etiquetas.append(nombre)
        print(i,': ',nombre)
        i += 1

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


target_size = (299, 299)  # El tamaño objetivo después de recortar y redimensionar

## CARGAR LAS RUTAS DE CADA IMAGEN DE LA BASE DE DATOS
print('---- CARGANDO BASE DE DATOS -------')
orig_dir = r'G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD150'

image_paths = []

for filename in os.listdir(orig_dir):
    image_paths.append(os.path.join(orig_dir, filename))

## NORMALIZAR LAS IMAGENES
imagenesBD, y_imagenesBD = crop_and_normalize_images(image_paths, target_size)

print()
print('---- CARGANDO CONSULTAS -------')
## CARGAR LAS RUTAS DE CADA IMAGEN DE CONSULTA
orig_dir = r'G:\GIBD\Investigacion\Similitud Logos Clubes\consultas'

image_paths = []

for filename in os.listdir(orig_dir):
    image_paths.append(os.path.join(orig_dir, filename))


## NORMALIZAR LAS IMAGENES

consultas, y_consultas = crop_and_normalize_images(image_paths, target_size)

```


*Salida:*
```text
---- CARGANDO BASE DE DATOS -------
1 :  ac-milan
TIENE ALPHA...
23 81 2 97
2 :  ajax
TIENE ALPHA...
4 175 3 176
3 :  america
TIENE ALPHA...
16 162 3 173
4 :  angers-sco
TIENE ALPHA...
8 89 0 98
5 :  argentinos
TIENE ALPHA...
19 159 5 176
6 :  arsenal
TIENE ALPHA...
9 90 0 99
7 :  as-monaco
TIENE ALPHA...
22 78 0 99
8 :  as-saint-etienne
TIENE ALPHA...
14 85 2 98
9 :  aston-villa
TIENE ALPHA...
14 84 0 98
10 :  atalanta
TIENE ALPHA...
24 75 8 91
11 :  athletic
TIENE ALPHA...
21 78 17 82
12 :  atletico-madrid
TIENE ALPHA...
23 76 14 84
13 :  atletico-nacional
TIENE ALPHA...
31 148 4 173
14 :  augsburg
TIENE ALPHA...
13 86 2 97
15 :  barcelona
TIENE ALPHA...
17 82 17 82
16 :  bayern
TIENE ALPHA...
2 97 2 97
17 :  belgrano
TIENE ALPHA...
4 175 3 176
18 :  blackburn
TIENE ALPHA...
7 172 3 176
19 :  boca
TIENE ALPHA...
17 161 3 173
20 :  bologna
TIENE ALPHA...
18 81 2 97
21 :  botafogo
TIENE ALPHA...
15 164 3 175
22 :  bremen
TIENE ALPHA...
17 82 1 98
23 :  brentford
TIENE ALPHA...
0 99 0 99
24 :  brighton
TIENE ALPHA...
0 99 0 99
25 :  brujas
TIENE ALPHA...
26 153 3 173
26 :  burnley
TIENE ALPHA...
7 92 0 99
27 :  cadiz
TIENE ALPHA...
25 74 11 87
28 :  cagliari
TIENE ALPHA...
9 90 1 98
29 :  celta
TIENE ALPHA...
27 71 9 88
30 :  central
TIENE ALPHA...
18 163 5 172
31 :  chelsea
TIENE ALPHA...
0 99 0 99
32 :  clermont-foot-63
TIENE ALPHA...
8 91 1 98
33 :  colo-colo
TIENE ALPHA...
9 172 4 172
34 :  corinthians
TIENE ALPHA...
23 156 3 173
35 :  cruzeiro
TIENE ALPHA...
7 173 5 172
36 :  crystal-palace
TIENE ALPHA...
12 87 1 96
37 :  deportivo-alaves
TIENE ALPHA...
16 82 18 84
38 :  deportivo-cali
TIENE ALPHA...
6 175 4 171
39 :  dortmund
TIENE ALPHA...
0 99 0 99
40 :  dusseldorf
TIENE ALPHA...
0 99 0 99
41 :  elche
TIENE ALPHA...
22 77 15 83
42 :  empoli
TIENE ALPHA...
16 82 4 94
43 :  espanyol
TIENE ALPHA...
23 76 9 85
44 :  estac-troyes
TIENE ALPHA...
9 90 0 98
45 :  estrella-roja
TIENE ALPHA...
4 175 3 176
46 :  estudiantes
TIENE ALPHA...
6 175 14 162
47 :  everton
TIENE ALPHA...
1 97 1 99
48 :  fc-girondins-de-bordeaux
TIENE ALPHA...
9 89 0 99
49 :  fc-lorient
TIENE ALPHA...
16 85 0 99
50 :  fc-metz
TIENE ALPHA...
15 82 2 98
51 :  fc-nantes
TIENE ALPHA...
13 86 1 99
52 :  ferro
TIENE ALPHA...
15 164 3 175
53 :  fiorentina
TIENE ALPHA...
20 78 6 92
54 :  flamengo
TIENE ALPHA...
23 158 5 172
55 :  fluminense
TIENE ALPHA...
15 163 3 173
56 :  fortaleza
TIENE ALPHA...
7 172 3 174
57 :  frankfurt
TIENE ALPHA...
0 99 0 99
58 :  freiburg
TIENE ALPHA...
19 80 0 99
59 :  furth
TIENE ALPHA...
19 80 3 96
60 :  galatasaray
TIENE ALPHA...
35 143 3 172
61 :  galaxy
TIENE ALPHA...
22 157 4 175
62 :  genoa
TIENE ALPHA...
22 77 10 89
63 :  getafe
TIENE ALPHA...
16 81 15 83
64 :  gimnasia
TIENE ALPHA...
34 145 3 171
65 :  granada
TIENE ALPHA...
30 69 6 94
66 :  gremio
TIENE ALPHA...
18 161 5 176
67 :  hamburg
TIENE ALPHA...
0 99 15 84
68 :  hannover
TIENE ALPHA...
0 98 6 93
69 :  hellas-verona
TIENE ALPHA...
6 94 5 94
70 :  hertha-bsc-berlin
TIENE ALPHA...
16 88 17 84
71 :  hoffenheim
TIENE ALPHA...
5 94 0 99
72 :  huracan
TIENE ALPHA...
38 141 3 176
73 :  independiente
TIENE ALPHA...
7 172 4 175
74 :  inter
TIENE ALPHA...
9 90 9 90
75 :  internacional
TIENE ALPHA...
4 175 3 176
76 :  juventus
TIENE ALPHA...
22 77 6 93
77 :  lanus
TIENE ALPHA...
6 173 4 172
78 :  lazio
TIENE ALPHA...
1 98 16 82
79 :  leeds-united
TIENE ALPHA...
10 90 0 99
80 :  leicester-city
TIENE ALPHA...
0 99 0 99
81 :  levante
TIENE ALPHA...
20 79 14 86
82 :  leverkusen
TIENE ALPHA...
0 99 12 87
83 :  liverpool
TIENE ALPHA...
13 86 0 99
84 :  losc-lille
TIENE ALPHA...
1 98 4 96
85 :  luton
TIENE ALPHA...
7 174 4 171
86 :  maccabi-haifa
TIENE ALPHA...
25 156 3 172
87 :  mainz
TIENE ALPHA...
2 98 1 98
88 :  malaga
TIENE ALPHA...
14 165 3 173
89 :  mallorca
TIENE ALPHA...
21 77 14 89
90 :  manchester-city
TIENE ALPHA...
0 99 0 99
91 :  manchester-united
TIENE ALPHA...
1 98 0 99
92 :  millonarios
TIENE ALPHA...
8 171 4 174
93 :  mineiro
TIENE ALPHA...
33 146 5 174
94 :  moenchengladbach
TIENE ALPHA...
20 80 1 98
95 :  montpellier-herault
TIENE ALPHA...
1 99 1 99
96 :  nacional
TIENE ALPHA...
7 173 5 176
97 :  napoli
TIENE ALPHA...
12 87 12 87
98 :  newcastle-united
TIENE ALPHA...
4 96 4 96
99 :  newells
TIENE ALPHA...
20 159 3 175
100 :  norwich-city
TIENE ALPHA...
5 93 1 99
101 :  nuremberg
TIENE ALPHA...
2 98 2 98
102 :  ogc-nice
TIENE ALPHA...
10 89 0 98
103 :  olympique-de-marseille
TIENE ALPHA...
12 87 2 99
104 :  olympique-lyonnais
TIENE ALPHA...
8 91 1 98
105 :  osasuna
TIENE ALPHA...
22 77 14 90
106 :  palmeiras
TIENE ALPHA...
7 173 5 172
107 :  paris-saint-germain
TIENE ALPHA...
2 97 2 97
108 :  parma
TIENE ALPHA...
20 159 4 173
109 :  peñarol
TIENE ALPHA...
26 153 4 174
110 :  platense
TIENE ALPHA...
23 156 4 174
111 :  porto
TIENE ALPHA...
27 152 4 172
112 :  racing
TIENE ALPHA...
19 160 3 174
113 :  rayo-vallecano
TIENE ALPHA...
17 82 21 77
114 :  rc-lens
TIENE ALPHA...
14 86 0 98
115 :  rc-strasbourg-alsace
TIENE ALPHA...
1 98 1 98
116 :  real-betis
TIENE ALPHA...
8 91 14 83
117 :  real-madrid
TIENE ALPHA...
22 76 8 83
118 :  real-murcia
TIENE ALPHA...
43 135 3 172
119 :  real-sociedad
TIENE ALPHA...
18 82 17 89
120 :  redbull-leipzig
TIENE ALPHA...
3 96 22 77
121 :  roma
TIENE ALPHA...
18 81 9 91
122 :  salernitana
TIENE ALPHA...
8 91 8 91
123 :  sampdoria
TIENE ALPHA...
17 82 9 92
124 :  san-lorenzo
TIENE ALPHA...
14 165 5 175
125 :  santos
TIENE ALPHA...
16 163 4 174
126 :  sassuolo
TIENE ALPHA...
12 87 9 91
127 :  schalke
TIENE ALPHA...
3 96 3 96
128 :  seattle-sounders
TIENE ALPHA...
5 175 63 116
129 :  sevilla
TIENE ALPHA...
22 77 15 83
130 :  sheffield
TIENE ALPHA...
4 175 3 175
131 :  southampton
TIENE ALPHA...
6 92 0 99
132 :  spezia
TIENE ALPHA...
8 91 8 91
133 :  sporting-cristal
TIENE ALPHA...
39 139 3 172
134 :  stade-brestois-29
TIENE ALPHA...
9 90 0 99
135 :  stade-de-reims
TIENE ALPHA...
22 78 3 97
136 :  stade-rennais-fc
TIENE ALPHA...
10 89 1 98
137 :  stuttgart
TIENE ALPHA...
5 94 0 99
138 :  the-strongest
TIENE ALPHA...
14 166 3 175
139 :  torino
TIENE ALPHA...
15 85 8 92
140 :  tottenham-hotspur
TIENE ALPHA...
26 72 2 97
141 :  udinese
TIENE ALPHA...
5 94 6 93
142 :  universitarios
TIENE ALPHA...
4 175 3 176
143 :  valencia
TIENE ALPHA...
22 77 13 87
144 :  velez
TIENE ALPHA...
23 157 3 174
145 :  venezia
TIENE ALPHA...
11 88 11 88
146 :  villarreal
TIENE ALPHA...
19 79 11 86
147 :  watford
TIENE ALPHA...
7 90 3 97
148 :  west-ham-united
TIENE ALPHA...
5 94 0 99
149 :  wolfsburg
TIENE ALPHA...
0 97 0 98
(299, 299, 3)

---- CARGANDO CONSULTAS -------
1 :  ac-milan
TIENE ALPHA...
34 65 3 52
2 :  arsenal
TIENE ALPHA...
29 70 8 58
3 :  aston-villa
TIENE ALPHA...
14 85 0 99
4 :  augsburg
TIENE ALPHA...
13 86 2 97
5 :  barcelona
TIENE ALPHA...
9 90 9 90
6 :  bayern
TIENE ALPHA...
2 98 2 97
7 :  bologna
TIENE ALPHA...
19 80 2 97
8 :  bremen
TIENE ALPHA...
0 68 0 99
9 :  brentford
TIENE ALPHA...
5 94 6 94
10 :  burnley
TIENE ALPHA...
9 90 2 97
11 :  cadiz
TIENE ALPHA...
0 65 0 98
12 :  cagliari
TIENE ALPHA...
0 82 0 99
13 :  celta
TIENE ALPHA...
0 65 0 99
14 :  chelsea
TIENE ALPHA...
2 97 2 97
15 :  clermont-foot-63
TIENE ALPHA...
0 83 0 99
16 :  crystal-palace
TIENE ALPHA...
11 89 0 99
17 :  deportivo-alaves
TIENE ALPHA...
0 98 0 99
18 :  dortmund
TIENE ALPHA...
2 97 2 97
19 :  dusseldorf
TIENE ALPHA...
6 93 6 93
20 :  elche
TIENE ALPHA...
0 81 0 99
21 :  empoli
TIENE ALPHA...
15 84 2 97
22 :  espanyol
TIENE ALPHA...
32 67 2 53
23 :  fc-lorient
TIENE ALPHA...
1 68 1 98
24 :  fc-nantes
TIENE ALPHA...
0 75 0 99
25 :  fiorentina
TIENE ALPHA...
3 73 14 85
26 :  freiburg
TIENE ALPHA...
0 69 0 99
27 :  genoa
TIENE ALPHA...
16 83 2 97
28 :  getafe
TIENE ALPHA...
6 93 4 95
29 :  granada
TIENE ALPHA...
31 68 0 99
30 :  hoffenheim
TIENE ALPHA...
7 93 2 97
31 :  inter
TIENE ALPHA...
5 94 5 94
32 :  juventus
TIENE ALPHA...
18 78 3 98
33 :  levante
TIENE ALPHA...
0 79 0 98
34 :  liverpool
TIENE ALPHA...
13 86 1 99
35 :  mainz
TIENE ALPHA...
2 97 3 92
36 :  manchester-city
TIENE ALPHA...
11 88 12 88
37 :  manchester-united
TIENE ALPHA...
2 97 2 97
38 :  moenchengladbach
TIENE ALPHA...
19 80 1 98
39 :  napoli
TIENE ALPHA...
0 99 0 99
40 :  norwich-city
TIENE ALPHA...
0 87 0 99
41 :  ogc-nice
TIENE ALPHA...
10 89 0 99
42 :  olympique-lyonnais
TIENE ALPHA...
9 90 2 97
43 :  paris-saint-germain
TIENE ALPHA...
3 96 3 96
44 :  real-betis
TIENE ALPHA...
1 98 0 83
45 :  real-madrid
TIENE ALPHA...
30 68 1 54
46 :  real-sociedad
TIENE ALPHA...
29 70 4 51
47 :  redbull-leipzig
TIENE ALPHA...
2 97 2 64
48 :  roma
TIENE ALPHA...
16 84 5 94
49 :  sampdoria
TIENE ALPHA...
12 87 2 97
50 :  schalke
TIENE ALPHA...
23 76 2 53
51 :  southampton
TIENE ALPHA...
8 91 2 97
52 :  spezia
TIENE ALPHA...
0 99 0 99
53 :  stuttgart
TIENE ALPHA...
6 93 2 97
54 :  udinese
TIENE ALPHA...
0 99 1 97
55 :  valencia
TIENE ALPHA...
20 79 10 89
56 :  villarreal
TIENE ALPHA...
10 89 0 98
57 :  west-ham-united
TIENE ALPHA...
0 89 0 98
58 :  wolfsburg
TIENE ALPHA...
2 97 2 97
(299, 299, 3)
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

# Transfer Learning y Fine Tuning


**[Celda 26 - Código]**
```python
#### PARA PROBAR TRANSFER LEARNING Y FINE TUNING
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
### Siamesa
# vectores_elemento = model.predict(imagenesBD)
# vectores_consulta = model.predict(consultas)
#### DenseNet
# vectores_elemento = new_model_densenet.predict(imagenesBD)
# vectores_consulta = new_model_densenet.predict(consultas)
#### Inception
vectores_elemento = new_model_inception.predict(imagenesBD)
vectores_consulta = new_model_inception.predict(consultas)
### Resnet
# vectores_elemento = new_model_resnet.predict(imagenesBD)
# vectores_consulta = new_model_resnet.predict(consultas)

print(vectores_elemento[0].shape)


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


*Salida:*
```text
<class 'numpy.ndarray'>
(17, 17, 192)
Consulta número:  0
   Imagen de consulta:  ac-milan
   Mas cercanos:  [[21605.814, 'ac-milan'], [21658.105, 'elche'], [22622.82, 'villarreal'], [22643.975, 'augsburg'], [23098.06, 'bologna']]
--> ACIERTO
    Primer Lugar: 1
    Entre los tres primeros: 1


Consulta número:  1
   Imagen de consulta:  arsenal
   Mas cercanos:  [[14106.409, 'arsenal'], [14669.604, 'leeds-united'], [15483.497, 'fc-nantes'], [16656.975, 'aston-villa'], [17078.113, 'estac-troyes']]
--> ACIERTO
    Primer Lugar: 2
    Entre los tres primeros: 2


Consulta número:  2
   Imagen de consulta:  aston-villa
   Mas cercanos:  [[12841.078, 'aston-villa'], [12886.687, 'fc-nantes'], [18950.824, 'burnley'], [19706.963, 'boca'], [20305.707, 'central']]
--> ACIERTO
    Primer Lugar: 3
    Entre los tres primeros: 3


Consulta número:  3
   Imagen de consulta:  augsburg
   Mas cercanos:  [[3945.7314, 'augsburg'], [24283.516, 'aston-villa'], [24844.326, 'estrella-roja'], [24879.941, 'america'], [24959.553, 'villarreal']]
--> ACIERTO
    Primer Lugar: 4
    Entre los tres primeros: 4


Consulta número:  4
   Imagen de consulta:  barcelona
   Mas cercanos:  [[7823.9624, 'barcelona'], [20908.736, 'mallorca'], [21110.38, 'roma'], [21467.117, 'burnley'], [21839.064, 'villarreal']]
--> ACIERTO
    Primer Lugar: 5
    Entre los tres primeros: 5


Consulta número:  5
   Imagen de consulta:  bayern
   Mas cercanos:  [[2938.9358, 'bayern'], [18225.988, 'belgrano'], [18675.742, 'leicester-city'], [19230.39, 'huracan'], [19589.418, 'palmeiras']]
--> ACIERTO
    Primer Lugar: 6
    Entre los tres primeros: 6


Consulta número:  6
   Imagen de consulta:  bologna
   Mas cercanos:  [[4552.2563, 'bologna'], [25083.969, 'villarreal'], [25120.553, 'augsburg'], [25260.297, 'napoli'], [25820.752, 'galaxy']]
--> ACIERTO
    Primer Lugar: 7
    Entre los tres primeros: 7


Consulta número:  7
   Imagen de consulta:  bremen
   Mas cercanos:  [[21510.896, 'moenchengladbach'], [23318.217, 'bremen'], [23912.2, 'redbull-leipzig'], [24522.486, 'elche'], [24545.6, 'real-murcia']]
--> ACIERTO
    Entre los tres primeros: 8


Consulta número:  8
   Imagen de consulta:  brentford
   Mas cercanos:  [[13684.214, 'brentford'], [19089.922, 'manchester-united'], [20166.309, 'paris-saint-germain'], [20367.318, 'venezia'], [20835.365, 'internacional']]
--> ACIERTO
    Primer Lugar: 8
    Entre los tres primeros: 9


Consulta número:  9
   Imagen de consulta:  burnley
   Mas cercanos:  [[11278.815, 'burnley'], [19033.898, 'fc-nantes'], [20139.426, 'villarreal'], [20188.822, 'aston-villa'], [20288.22, 'leeds-united']]
--> ACIERTO
    Primer Lugar: 9
    Entre los tres primeros: 10


Consulta número:  10
   Imagen de consulta:  cadiz
   Mas cercanos:  [[16777.846, 'real-murcia'], [17155.967, 'cadiz'], [17870.674, 'elche'], [18232.111, 'rayo-vallecano'], [18507.963, 'lazio']]
--> ACIERTO
    Entre los tres primeros: 11


Consulta número:  11
   Imagen de consulta:  cagliari
   Mas cercanos:  [[2834.5505, 'cagliari'], [21421.518, 'sevilla'], [21657.086, 'rayo-vallecano'], [23317.73, 'velez'], [23699.822, 'angers-sco']]
--> ACIERTO
    Primer Lugar: 10
    Entre los tres primeros: 12


Consulta número:  12
   Imagen de consulta:  celta
   Mas cercanos:  [[16702.51, 'celta'], [22118.547, 'real-murcia'], [22320.48, 'angers-sco'], [22398.832, 'lazio'], [22591.568, 'fiorentina']]
--> ACIERTO
    Primer Lugar: 11
    Entre los tres primeros: 13


Consulta número:  13
   Imagen de consulta:  chelsea
   Mas cercanos:  [[3633.435, 'chelsea'], [17572.342, 'deportivo-alaves'], [18947.523, 'brentford'], [19853.574, 'bayern'], [20102.205, 'rc-strasbourg-alsace']]
--> ACIERTO
    Primer Lugar: 12
    Entre los tres primeros: 14


Consulta número:  14
   Imagen de consulta:  clermont-foot-63
   Mas cercanos:  [[15140.845, 'clermont-foot-63'], [22720.268, 'leeds-united'], [23049.441, 'boca'], [23183.545, 'estac-troyes'], [23377.535, 'malaga']]
--> ACIERTO
    Primer Lugar: 13
    Entre los tres primeros: 15


Consulta número:  15
   Imagen de consulta:  crystal-palace
   Mas cercanos:  [[5879.7935, 'crystal-palace'], [16517.729, 'lazio'], [19796.383, 'celta'], [19878.547, 'schalke'], [20251.05, 'granada']]
--> ACIERTO
    Primer Lugar: 14
    Entre los tres primeros: 16


Consulta número:  16
   Imagen de consulta:  deportivo-alaves
   Mas cercanos:  [[18146.55, 'deportivo-alaves'], [18624.404, 'bayern'], [19398.643, 'leicester-city'], [20364.033, 'palmeiras'], [20426.822, 'udinese']]
--> ACIERTO
    Primer Lugar: 15
    Entre los tres primeros: 17


Consulta número:  17
   Imagen de consulta:  dortmund
   Mas cercanos:  [[6238.6074, 'dortmund'], [15860.637, 'fc-nantes'], [17230.635, 'napoli'], [18243.953, 'burnley'], [18390.924, 'aston-villa']]
--> ACIERTO
    Primer Lugar: 16
    Entre los tres primeros: 18


Consulta número:  18
   Imagen de consulta:  dusseldorf
   Mas cercanos:  [[3887.054, 'dusseldorf'], [18389.828, 'frankfurt'], [19486.87, 'napoli'], [19711.45, 'rc-strasbourg-alsace'], [19961.156, 'nuremberg']]
--> ACIERTO
    Primer Lugar: 17
    Entre los tres primeros: 19


Consulta número:  19
   Imagen de consulta:  elche
   Mas cercanos:  [[13519.901, 'elche'], [19107.688, 'estac-troyes'], [19759.719, 'roma'], [20087.287, 'villarreal'], [20890.31, 'stuttgart']]
--> ACIERTO
    Primer Lugar: 18
    Entre los tres primeros: 20


Consulta número:  20
   Imagen de consulta:  empoli
   Mas cercanos:  [[4949.2866, 'empoli'], [19028.898, 'leeds-united'], [20614.02, 'torino'], [21024.547, 'nacional'], [21071.846, 'malaga']]
--> ACIERTO
    Primer Lugar: 19
    Entre los tres primeros: 21


Consulta número:  21
   Imagen de consulta:  espanyol
   Mas cercanos:  [[12697.064, 'espanyol'], [19703.416, 'real-madrid'], [19747.164, 'olympique-de-marseille'], [20029.951, 'as-saint-etienne'], [20750.518, 'rayo-vallecano']]
--> ACIERTO
    Primer Lugar: 20
    Entre los tres primeros: 22


Consulta número:  22
   Imagen de consulta:  fc-lorient
   Mas cercanos:  [[9754.043, 'fc-lorient'], [21980.057, 'estac-troyes'], [22398.91, 'torino'], [22699.03, 'as-monaco'], [22744.514, 'stuttgart']]
--> ACIERTO
    Primer Lugar: 21
    Entre los tres primeros: 23


Consulta número:  23
   Imagen de consulta:  fc-nantes
   Mas cercanos:  [[1662.6326, 'fc-nantes'], [12326.39, 'aston-villa'], [17394.38, 'burnley'], [17425.076, 'leeds-united'], [17506.455, 'boca']]
--> ACIERTO
    Primer Lugar: 22
    Entre los tres primeros: 24


Consulta número:  24
   Imagen de consulta:  fiorentina
   Mas cercanos:  [[16367.238, 'fiorentina'], [20058.865, 'real-betis'], [20099.54, 'elche'], [20526.088, 'redbull-leipzig'], [20608.186, 'celta']]
--> ACIERTO
    Primer Lugar: 23
    Entre los tres primeros: 25


Consulta número:  25
   Imagen de consulta:  freiburg
   Mas cercanos:  [[17928.51, 'freiburg'], [27069.06, 'frankfurt'], [27485.053, 'palmeiras'], [27758.695, 'rc-strasbourg-alsace'], [27854.514, 'manchester-city']]
--> ACIERTO
    Primer Lugar: 24
    Entre los tres primeros: 26


Consulta número:  26
   Imagen de consulta:  genoa
   Mas cercanos:  [[5083.979, 'genoa'], [21151.277, 'central'], [22661.658, 'peñarol'], [23073.115, 'fc-nantes'], [23609.875, 'porto']]
--> ACIERTO
    Primer Lugar: 25
    Entre los tres primeros: 27


Consulta número:  27
   Imagen de consulta:  getafe
   Mas cercanos:  [[17752.83, 'getafe'], [21847.992, 'rc-strasbourg-alsace'], [21885.014, 'nuremberg'], [22157.613, 'cruzeiro'], [22239.809, 'bayern']]
--> ACIERTO
    Primer Lugar: 26
    Entre los tres primeros: 28


Consulta número:  28
   Imagen de consulta:  granada
   Mas cercanos:  [[9325.449, 'granada'], [19990.488, 'rayo-vallecano'], [20224.047, 'lazio'], [20542.314, 'celta'], [21751.746, 'crystal-palace']]
--> ACIERTO
    Primer Lugar: 27
    Entre los tres primeros: 29


Consulta número:  29
   Imagen de consulta:  hoffenheim
   Mas cercanos:  [[1885.8999, 'hoffenheim'], [23464.053, 'granada'], [23724.355, 'rayo-vallecano'], [24812.297, 'leeds-united'], [25057.342, 'lazio']]
--> ACIERTO
    Primer Lugar: 28
    Entre los tres primeros: 30


Consulta número:  30
   Imagen de consulta:  inter
   Mas cercanos:  [[2733.5278, 'inter'], [21833.816, 'napoli'], [22204.256, 'chelsea'], [22219.887, 'estrella-roja'], [22709.203, 'universitarios']]
--> ACIERTO
    Primer Lugar: 29
    Entre los tres primeros: 31


Consulta número:  31
   Imagen de consulta:  juventus
   Mas cercanos:  [[9483.973, 'juventus'], [31886.34, 'olympique-de-marseille'], [31908.213, 'getafe'], [32084.186, 'stade-de-reims'], [32129.664, 'espanyol']]
--> ACIERTO
    Primer Lugar: 30
    Entre los tres primeros: 32


Consulta número:  32
   Imagen de consulta:  levante
   Mas cercanos:  [[20647.518, 'levante'], [22189.324, 'arsenal'], [23300.023, 'leeds-united'], [23789.018, 'malaga'], [23969.268, 'stade-brestois-29']]
--> ACIERTO
    Primer Lugar: 31
    Entre los tres primeros: 33


Consulta número:  33
   Imagen de consulta:  liverpool
   Mas cercanos:  [[12083.057, 'liverpool'], [18841.35, 'redbull-leipzig'], [18856.814, 'as-monaco'], [19009.877, 'olympique-de-marseille'], [19230.203, 'espanyol']]
--> ACIERTO
    Primer Lugar: 32
    Entre los tres primeros: 34


Consulta número:  34
   Imagen de consulta:  mainz
   Mas cercanos:  [[17038.314, 'mainz'], [19583.602, 'getafe'], [19914.867, 'rc-strasbourg-alsace'], [20107.754, 'olympique-de-marseille'], [20116.111, 'frankfurt']]
--> ACIERTO
    Primer Lugar: 33
    Entre los tres primeros: 35


Consulta número:  35
   Imagen de consulta:  manchester-city
   Mas cercanos:  [[11226.063, 'manchester-city'], [18306.264, 'leicester-city'], [19410.38, 'celta'], [19692.012, 'udinese'], [20249.922, 'elche']]
--> ACIERTO
    Primer Lugar: 34
    Entre los tres primeros: 36


Consulta número:  36
   Imagen de consulta:  manchester-united
   Mas cercanos:  [[9837.606, 'manchester-united'], [21469.13, 'roma'], [22041.525, 'burnley'], [22075.906, 'dortmund'], [22085.053, 'galaxy']]
--> ACIERTO
    Primer Lugar: 35
    Entre los tres primeros: 37


Consulta número:  37
   Imagen de consulta:  moenchengladbach
   Mas cercanos:  [[5826.9844, 'moenchengladbach'], [23401.03, 'elche'], [23423.93, 'redbull-leipzig'], [24253.152, 'real-murcia'], [24253.273, 'fiorentina']]
--> ACIERTO
    Primer Lugar: 36
    Entre los tres primeros: 38


Consulta número:  38
   Imagen de consulta:  napoli
   Mas cercanos:  [[8404.888, 'napoli'], [18914.6, 'lanus'], [20071.07, 'estrella-roja'], [20200.312, 'inter'], [20636.686, 'dortmund']]
--> ACIERTO
    Primer Lugar: 37
    Entre los tres primeros: 39


Consulta número:  39
   Imagen de consulta:  norwich-city
   Mas cercanos:  [[15468.951, 'fc-nantes'], [17533.043, 'boca'], [17571.242, 'norwich-city'], [18606.018, 'arsenal'], [18811.98, 'aston-villa']]
--> ACIERTO
    Entre los tres primeros: 40


Consulta número:  40
   Imagen de consulta:  ogc-nice
   Mas cercanos:  [[10382.089, 'ogc-nice'], [23343.545, 'porto'], [23671.676, 'cruzeiro'], [23696.396, 'roma'], [24172.9, 'newcastle-united']]
--> ACIERTO
    Primer Lugar: 38
    Entre los tres primeros: 41


Consulta número:  41
   Imagen de consulta:  olympique-lyonnais
   Mas cercanos:  [[5463.2285, 'olympique-lyonnais'], [21056.56, 'aston-villa'], [22439.53, 'fc-nantes'], [22469.21, 'racing'], [22871.855, 'leeds-united']]
--> ACIERTO
    Primer Lugar: 39
    Entre los tres primeros: 42


Consulta número:  42
   Imagen de consulta:  paris-saint-germain
   Mas cercanos:  [[7526.2715, 'paris-saint-germain'], [20770.277, 'internacional'], [20823.668, 'nuremberg'], [20873.637, 'rc-strasbourg-alsace'], [21513.281, 'bayern']]
--> ACIERTO
    Primer Lugar: 40
    Entre los tres primeros: 43


Consulta número:  43
   Imagen de consulta:  real-betis
   Mas cercanos:  [[5315.425, 'real-betis'], [19903.39, 'redbull-leipzig'], [20116.672, 'lazio'], [20268.553, 'fiorentina'], [21245.957, 'rayo-vallecano']]
--> ACIERTO
    Primer Lugar: 41
    Entre los tres primeros: 44


Consulta número:  44
   Imagen de consulta:  real-madrid
   Mas cercanos:  [[11058.512, 'real-madrid'], [17538.467, 'elche'], [17905.816, 'olympique-de-marseille'], [19123.234, 'espanyol'], [19493.559, 'celta']]
--> ACIERTO
    Primer Lugar: 42
    Entre los tres primeros: 45


Consulta número:  45
   Imagen de consulta:  real-sociedad
   Mas cercanos:  [[9416.783, 'real-sociedad'], [21462.299, 'espanyol'], [21817.203, 'getafe'], [21852.186, 'elche'], [22533.213, 'redbull-leipzig']]
--> ACIERTO
    Primer Lugar: 43
    Entre los tres primeros: 46


Consulta número:  46
   Imagen de consulta:  redbull-leipzig
   Mas cercanos:  [[15232.956, 'redbull-leipzig'], [18512.953, 'rayo-vallecano'], [20163.258, 'elche'], [20186.092, 'athletic'], [20450.145, 'real-betis']]
--> ACIERTO
    Primer Lugar: 44
    Entre los tres primeros: 47


Consulta número:  47
   Imagen de consulta:  roma
   Mas cercanos:  [[4421.7495, 'roma'], [18890.758, 'galaxy'], [19004.516, 'boca'], [20130.049, 'burnley'], [20753.705, 'villarreal']]
--> ACIERTO
    Primer Lugar: 45
    Entre los tres primeros: 48


Consulta número:  48
   Imagen de consulta:  sampdoria
   Mas cercanos:  [[13371.263, 'sampdoria'], [24871.107, 'real-sociedad'], [25511.133, 'getafe'], [25677.695, 'elche'], [26017.225, 'spezia']]
--> ACIERTO
    Primer Lugar: 46
    Entre los tres primeros: 49


Consulta número:  49
   Imagen de consulta:  schalke
   Mas cercanos:  [[19709.453, 'schalke'], [20163.332, 'dusseldorf'], [20486.684, 'elche'], [20764.67, 'rayo-vallecano'], [20952.863, 'olympique-de-marseille']]
--> ACIERTO
    Primer Lugar: 47
    Entre los tres primeros: 50


Consulta número:  50
   Imagen de consulta:  southampton
   Mas cercanos:  [[9609.632, 'southampton'], [21228.27, 'galatasaray'], [21606.777, 'huracan'], [22589.291, 'argentinos'], [22788.6, 'bayern']]
--> ACIERTO
    Primer Lugar: 48
    Entre los tres primeros: 51


Consulta número:  51
   Imagen de consulta:  spezia
   Mas cercanos:  [[16876.148, 'spezia'], [20681.254, 'manchester-city'], [20986.723, 'estac-troyes'], [21286.8, 'frankfurt'], [21638.268, 'villarreal']]
--> ACIERTO
    Primer Lugar: 49
    Entre los tres primeros: 52


Consulta número:  52
   Imagen de consulta:  stuttgart
   Mas cercanos:  [[15861.428, 'stuttgart'], [18490.031, 'redbull-leipzig'], [19036.088, 'rayo-vallecano'], [19540.97, 'real-murcia'], [19561.695, 'angers-sco']]
--> ACIERTO
    Primer Lugar: 50
    Entre los tres primeros: 53


Consulta número:  53
   Imagen de consulta:  udinese
   Mas cercanos:  [[15955.237, 'udinese'], [19849.604, 'leicester-city'], [20061.617, 'montpellier-herault'], [20813.275, 'bayern'], [22247.08, 'belgrano']]
--> ACIERTO
    Primer Lugar: 51
    Entre los tres primeros: 54


Consulta número:  54
   Imagen de consulta:  valencia
   Mas cercanos:  [[5288.921, 'valencia'], [20348.531, 'elche'], [20856.598, 'estac-troyes'], [21400.002, 'sevilla'], [21481.764, 'napoli']]
--> ACIERTO
    Primer Lugar: 52
    Entre los tres primeros: 55


Consulta número:  55
   Imagen de consulta:  villarreal
   Mas cercanos:  [[10633.236, 'villarreal'], [18431.953, 'mallorca'], [19681.752, 'porto'], [19877.572, 'barcelona'], [20171.727, 'paris-saint-germain']]
--> ACIERTO
    Primer Lugar: 53
    Entre los tres primeros: 56


Consulta número:  56
   Imagen de consulta:  west-ham-united
   Mas cercanos:  [[3462.199, 'west-ham-united'], [18921.783, 'boca'], [19677.756, 'fc-nantes'], [20939.605, 'flamengo'], [20968.38, 'roma']]
--> ACIERTO
    Primer Lugar: 54
    Entre los tres primeros: 57


Consulta número:  57
   Imagen de consulta:  wolfsburg
   Mas cercanos:  [[6487.3926, 'wolfsburg'], [20655.42, 'schalke'], [20779.537, 'fiorentina'], [21017.668, 'elche'], [21536.336, 'stade-de-reims']]
--> ACIERTO
    Primer Lugar: 55
    Entre los tres primeros: 58


Porcentaje de aciertos NN1:  0.9482758620689655
Porcentaje de aciertos NN3:  1.0
Porcentaje de aciertos NN5:  1.0
  Cantidad consultas:  58
```

## Fine Tuning


**[Celda 28 - Código]**
```python
imagenesBD[0].shape
print(len(imagenesBD))
```


*Salida:*
```text
150
```

## DenseNet


**[Celda 30 - Código]**
```python
from tensorflow.keras.applications.densenet import DenseNet121

def build_densenet(input_shape=(224, 224, 3), n_classes=10):
    input_layer = Input(shape=input_shape)
    densenet121 = DenseNet121(include_top=False, weights='imagenet', input_tensor=input_layer)
    x = GlobalAveragePooling2D()(densenet121.output)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(input_layer, x)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=3e-4), metrics=['accuracy'])
    return model

densenet = build_densenet(n_classes=100)
```


**[Celda 31 - Código]**
```python
densenet.summary()
```


*Salida:*
```text
Model: "functional_47"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_5 (InputLayer)            [(None, 224, 224, 3) 0                                            
__________________________________________________________________________________________________
zero_padding2d_4 (ZeroPadding2D (None, 230, 230, 3)  0           input_5[0][0]                    
__________________________________________________________________________________________________
conv1/conv (Conv2D)             (None, 112, 112, 64) 9408        zero_padding2d_4[0][0]           
__________________________________________________________________________________________________
conv1/bn (BatchNormalization)   (None, 112, 112, 64) 256         conv1/conv[0][0]                 
__________________________________________________________________________________________________
conv1/relu (Activation)         (None, 112, 112, 64) 0           conv1/bn[0][0]                   
__________________________________________________________________________________________________
zero_padding2d_5 (ZeroPadding2D (None, 114, 114, 64) 0           conv1/relu[0][0]                 
__________________________________________________________________________________________________
pool1 (MaxPooling2D)            (None, 56, 56, 64)   0           zero_padding2d_5[0][0]           
__________________________________________________________________________________________________
conv2_block1_0_bn (BatchNormali (None, 56, 56, 64)   256         pool1[0][0]                      
__________________________________________________________________________________________________
conv2_block1_0_relu (Activation (None, 56, 56, 64)   0           conv2_block1_0_bn[0][0]          
__________________________________________________________________________________________________
conv2_block1_1_conv (Conv2D)    (None, 56, 56, 128)  8192        conv2_block1_0_relu[0][0]        
__________________________________________________________________________________________________
conv2_block1_1_bn (BatchNormali (None, 56, 56, 128)  512         conv2_block1_1_conv[0][0]        
__________________________________________________________________________________________________
conv2_block1_1_relu (Activation (None, 56, 56, 128)  0           conv2_block1_1_bn[0][0]          
__________________________________________________________________________________________________
conv2_block1_2_conv (Conv2D)    (None, 56, 56, 32)   36864       conv2_block1_1_relu[0][0]        
__________________________________________________________________________________________________
conv2_block1_concat (Concatenat (None, 56, 56, 96)   0           pool1[0][0]                      
                                                                 conv2_block1_2_conv[0][0]        
__________________________________________________________________________________________________
conv2_block2_0_bn (BatchNormali (None, 56, 56, 96)   384         conv2_block1_concat[0][0]        
__________________________________________________________________________________________________
conv2_block2_0_relu (Activation (None, 56, 56, 96)   0           conv2_block2_0_bn[0][0]          
__________________________________________________________________________________________________
conv2_block2_1_conv (Conv2D)    (None, 56, 56, 128)  12288       conv2_block2_0_relu[0][0]        
__________________________________________________________________________________________________
conv2_block2_1_bn (BatchNormali (None, 56, 56, 128)  512         conv2_block2_1_conv[0][0]        
__________________________________________________________________________________________________
conv2_block2_1_relu (Activation (None, 56, 56, 128)  0           conv2_block2_1_bn[0][0]          
__________________________________________________________________________________________________
conv2_block2_2_conv (Conv2D)    (None, 56, 56, 32)   36864       conv2_block2_1_relu[0][0]        
__________________________________________________________________________________________________
conv2_block2_concat (Concatenat (None, 56, 56, 128)  0           conv2_block1_concat[0][0]        
                                                                 conv2_block2_2_conv[0][0]        
__________________________________________________________________________________________________
conv2_block3_0_bn (BatchNormali (None, 56, 56, 128)  512         conv2_block2_concat[0][0]        
__________________________________________________________________________________________________
conv2_block3_0_relu (Activation (None, 56, 56, 128)  0           conv2_block3_0_bn[0][0]          
__________________________________________________________________________________________________
conv2_block3_1_conv (Conv2D)    (None, 56, 56, 128)  16384       conv2_block3_0_relu[0][0]        
__________________________________________________________________________________________________
conv2_block3_1_bn (BatchNormali (None, 56, 56, 128)  512         conv2_block3_1_conv[0][0]        
__________________________________________________________________________________________________
conv2_block3_1_relu (Activation (None, 56, 56, 128)  0           conv2_block3_1_bn[0][0]          
__________________________________________________________________________________________________
conv2_block3_2_conv (Conv2D)    (None, 56, 56, 32)   36864       conv2_block3_1_relu[0][0]        
__________________________________________________________________________________________________
conv2_block3_concat (Concatenat (None, 56, 56, 160)  0           conv2_block2_concat[0][0]        
                                                                 conv2_block3_2_conv[0][0]        
__________________________________________________________________________________________________
conv2_block4_0_bn (BatchNormali (None, 56, 56, 160)  640         conv2_block3_concat[0][0]        
__________________________________________________________________________________________________
conv2_block4_0_relu (Activation (None, 56, 56, 160)  0           conv2_block4_0_bn[0][0]          
__________________________________________________________________________________________________
conv2_block4_1_conv (Conv2D)    (None, 56, 56, 128)  20480       conv2_block4_0_relu[0][0]        
__________________________________________________________________________________________________
conv2_block4_1_bn (BatchNormali (None, 56, 56, 128)  512         conv2_block4_1_conv[0][0]        
__________________________________________________________________________________________________
conv2_block4_1_relu (Activation (None, 56, 56, 128)  0           conv2_block4_1_bn[0][0]          
__________________________________________________________________________________________________
conv2_block4_2_conv (Conv2D)    (None, 56, 56, 32)   36864       conv2_block4_1_relu[0][0]        
__________________________________________________________________________________________________
conv2_block4_concat (Concatenat (None, 56, 56, 192)  0           conv2_block3_concat[0][0]        
                                                                 conv2_block4_2_conv[0][0]        
__________________________________________________________________________________________________
conv2_block5_0_bn (BatchNormali (None, 56, 56, 192)  768         conv2_block4_concat[0][0]        
__________________________________________________________________________________________________
conv2_block5_0_relu (Activation (None, 56, 56, 192)  0           conv2_block5_0_bn[0][0]          
__________________________________________________________________________________________________
conv2_block5_1_conv (Conv2D)    (None, 56, 56, 128)  24576       conv2_block5_0_relu[0][0]        
__________________________________________________________________________________________________
conv2_block5_1_bn (BatchNormali (None, 56, 56, 128)  512         conv2_block5_1_conv[0][0]        
__________________________________________________________________________________________________
conv2_block5_1_relu (Activation (None, 56, 56, 128)  0           conv2_block5_1_bn[0][0]          
__________________________________________________________________________________________________
conv2_block5_2_conv (Conv2D)    (None, 56, 56, 32)   36864       conv2_block5_1_relu[0][0]        
__________________________________________________________________________________________________
conv2_block5_concat (Concatenat (None, 56, 56, 224)  0           conv2_block4_concat[0][0]        
                                                                 conv2_block5_2_conv[0][0]        
__________________________________________________________________________________________________
conv2_block6_0_bn (BatchNormali (None, 56, 56, 224)  896         conv2_block5_concat[0][0]        
__________________________________________________________________________________________________
conv2_block6_0_relu (Activation (None, 56, 56, 224)  0           conv2_block6_0_bn[0][0]          
__________________________________________________________________________________________________
conv2_block6_1_conv (Conv2D)    (None, 56, 56, 128)  28672       conv2_block6_0_relu[0][0]        
__________________________________________________________________________________________________
conv2_block6_1_bn (BatchNormali (None, 56, 56, 128)  512         conv2_block6_1_conv[0][0]        
__________________________________________________________________________________________________
conv2_block6_1_relu (Activation (None, 56, 56, 128)  0           conv2_block6_1_bn[0][0]          
__________________________________________________________________________________________________
conv2_block6_2_conv (Conv2D)    (None, 56, 56, 32)   36864       conv2_block6_1_relu[0][0]        
__________________________________________________________________________________________________
conv2_block6_concat (Concatenat (None, 56, 56, 256)  0           conv2_block5_concat[0][0]        
                                                                 conv2_block6_2_conv[0][0]        
__________________________________________________________________________________________________
pool2_bn (BatchNormalization)   (None, 56, 56, 256)  1024        conv2_block6_concat[0][0]        
__________________________________________________________________________________________________
pool2_relu (Activation)         (None, 56, 56, 256)  0           pool2_bn[0][0]                   
__________________________________________________________________________________________________
pool2_conv (Conv2D)             (None, 56, 56, 128)  32768       pool2_relu[0][0]                 
__________________________________________________________________________________________________
pool2_pool (AveragePooling2D)   (None, 28, 28, 128)  0           pool2_conv[0][0]                 
__________________________________________________________________________________________________
conv3_block1_0_bn (BatchNormali (None, 28, 28, 128)  512         pool2_pool[0][0]                 
__________________________________________________________________________________________________
conv3_block1_0_relu (Activation (None, 28, 28, 128)  0           conv3_block1_0_bn[0][0]          
__________________________________________________________________________________________________
conv3_block1_1_conv (Conv2D)    (None, 28, 28, 128)  16384       conv3_block1_0_relu[0][0]        
__________________________________________________________________________________________________
conv3_block1_1_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block1_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block1_1_relu (Activation (None, 28, 28, 128)  0           conv3_block1_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block1_2_conv (Conv2D)    (None, 28, 28, 32)   36864       conv3_block1_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block1_concat (Concatenat (None, 28, 28, 160)  0           pool2_pool[0][0]                 
                                                                 conv3_block1_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block2_0_bn (BatchNormali (None, 28, 28, 160)  640         conv3_block1_concat[0][0]        
__________________________________________________________________________________________________
conv3_block2_0_relu (Activation (None, 28, 28, 160)  0           conv3_block2_0_bn[0][0]          
__________________________________________________________________________________________________
conv3_block2_1_conv (Conv2D)    (None, 28, 28, 128)  20480       conv3_block2_0_relu[0][0]        
__________________________________________________________________________________________________
conv3_block2_1_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block2_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block2_1_relu (Activation (None, 28, 28, 128)  0           conv3_block2_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block2_2_conv (Conv2D)    (None, 28, 28, 32)   36864       conv3_block2_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block2_concat (Concatenat (None, 28, 28, 192)  0           conv3_block1_concat[0][0]        
                                                                 conv3_block2_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block3_0_bn (BatchNormali (None, 28, 28, 192)  768         conv3_block2_concat[0][0]        
__________________________________________________________________________________________________
conv3_block3_0_relu (Activation (None, 28, 28, 192)  0           conv3_block3_0_bn[0][0]          
__________________________________________________________________________________________________
conv3_block3_1_conv (Conv2D)    (None, 28, 28, 128)  24576       conv3_block3_0_relu[0][0]        
__________________________________________________________________________________________________
conv3_block3_1_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block3_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block3_1_relu (Activation (None, 28, 28, 128)  0           conv3_block3_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block3_2_conv (Conv2D)    (None, 28, 28, 32)   36864       conv3_block3_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block3_concat (Concatenat (None, 28, 28, 224)  0           conv3_block2_concat[0][0]        
                                                                 conv3_block3_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block4_0_bn (BatchNormali (None, 28, 28, 224)  896         conv3_block3_concat[0][0]        
__________________________________________________________________________________________________
conv3_block4_0_relu (Activation (None, 28, 28, 224)  0           conv3_block4_0_bn[0][0]          
__________________________________________________________________________________________________
conv3_block4_1_conv (Conv2D)    (None, 28, 28, 128)  28672       conv3_block4_0_relu[0][0]        
__________________________________________________________________________________________________
conv3_block4_1_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block4_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block4_1_relu (Activation (None, 28, 28, 128)  0           conv3_block4_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block4_2_conv (Conv2D)    (None, 28, 28, 32)   36864       conv3_block4_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block4_concat (Concatenat (None, 28, 28, 256)  0           conv3_block3_concat[0][0]        
                                                                 conv3_block4_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block5_0_bn (BatchNormali (None, 28, 28, 256)  1024        conv3_block4_concat[0][0]        
__________________________________________________________________________________________________
conv3_block5_0_relu (Activation (None, 28, 28, 256)  0           conv3_block5_0_bn[0][0]          
__________________________________________________________________________________________________
conv3_block5_1_conv (Conv2D)    (None, 28, 28, 128)  32768       conv3_block5_0_relu[0][0]        
__________________________________________________________________________________________________
conv3_block5_1_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block5_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block5_1_relu (Activation (None, 28, 28, 128)  0           conv3_block5_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block5_2_conv (Conv2D)    (None, 28, 28, 32)   36864       conv3_block5_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block5_concat (Concatenat (None, 28, 28, 288)  0           conv3_block4_concat[0][0]        
                                                                 conv3_block5_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block6_0_bn (BatchNormali (None, 28, 28, 288)  1152        conv3_block5_concat[0][0]        
__________________________________________________________________________________________________
conv3_block6_0_relu (Activation (None, 28, 28, 288)  0           conv3_block6_0_bn[0][0]          
__________________________________________________________________________________________________
conv3_block6_1_conv (Conv2D)    (None, 28, 28, 128)  36864       conv3_block6_0_relu[0][0]        
__________________________________________________________________________________________________
conv3_block6_1_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block6_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block6_1_relu (Activation (None, 28, 28, 128)  0           conv3_block6_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block6_2_conv (Conv2D)    (None, 28, 28, 32)   36864       conv3_block6_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block6_concat (Concatenat (None, 28, 28, 320)  0           conv3_block5_concat[0][0]        
                                                                 conv3_block6_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block7_0_bn (BatchNormali (None, 28, 28, 320)  1280        conv3_block6_concat[0][0]        
__________________________________________________________________________________________________
conv3_block7_0_relu (Activation (None, 28, 28, 320)  0           conv3_block7_0_bn[0][0]          
__________________________________________________________________________________________________
conv3_block7_1_conv (Conv2D)    (None, 28, 28, 128)  40960       conv3_block7_0_relu[0][0]        
__________________________________________________________________________________________________
conv3_block7_1_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block7_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block7_1_relu (Activation (None, 28, 28, 128)  0           conv3_block7_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block7_2_conv (Conv2D)    (None, 28, 28, 32)   36864       conv3_block7_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block7_concat (Concatenat (None, 28, 28, 352)  0           conv3_block6_concat[0][0]        
                                                                 conv3_block7_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block8_0_bn (BatchNormali (None, 28, 28, 352)  1408        conv3_block7_concat[0][0]        
__________________________________________________________________________________________________
conv3_block8_0_relu (Activation (None, 28, 28, 352)  0           conv3_block8_0_bn[0][0]          
__________________________________________________________________________________________________
conv3_block8_1_conv (Conv2D)    (None, 28, 28, 128)  45056       conv3_block8_0_relu[0][0]        
__________________________________________________________________________________________________
conv3_block8_1_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block8_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block8_1_relu (Activation (None, 28, 28, 128)  0           conv3_block8_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block8_2_conv (Conv2D)    (None, 28, 28, 32)   36864       conv3_block8_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block8_concat (Concatenat (None, 28, 28, 384)  0           conv3_block7_concat[0][0]        
                                                                 conv3_block8_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block9_0_bn (BatchNormali (None, 28, 28, 384)  1536        conv3_block8_concat[0][0]        
__________________________________________________________________________________________________
conv3_block9_0_relu (Activation (None, 28, 28, 384)  0           conv3_block9_0_bn[0][0]          
__________________________________________________________________________________________________
conv3_block9_1_conv (Conv2D)    (None, 28, 28, 128)  49152       conv3_block9_0_relu[0][0]        
__________________________________________________________________________________________________
conv3_block9_1_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block9_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block9_1_relu (Activation (None, 28, 28, 128)  0           conv3_block9_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block9_2_conv (Conv2D)    (None, 28, 28, 32)   36864       conv3_block9_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block9_concat (Concatenat (None, 28, 28, 416)  0           conv3_block8_concat[0][0]        
                                                                 conv3_block9_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block10_0_bn (BatchNormal (None, 28, 28, 416)  1664        conv3_block9_concat[0][0]        
__________________________________________________________________________________________________
conv3_block10_0_relu (Activatio (None, 28, 28, 416)  0           conv3_block10_0_bn[0][0]         
__________________________________________________________________________________________________
conv3_block10_1_conv (Conv2D)   (None, 28, 28, 128)  53248       conv3_block10_0_relu[0][0]       
__________________________________________________________________________________________________
conv3_block10_1_bn (BatchNormal (None, 28, 28, 128)  512         conv3_block10_1_conv[0][0]       
__________________________________________________________________________________________________
conv3_block10_1_relu (Activatio (None, 28, 28, 128)  0           conv3_block10_1_bn[0][0]         
__________________________________________________________________________________________________
conv3_block10_2_conv (Conv2D)   (None, 28, 28, 32)   36864       conv3_block10_1_relu[0][0]       
__________________________________________________________________________________________________
conv3_block10_concat (Concatena (None, 28, 28, 448)  0           conv3_block9_concat[0][0]        
                                                                 conv3_block10_2_conv[0][0]       
__________________________________________________________________________________________________
conv3_block11_0_bn (BatchNormal (None, 28, 28, 448)  1792        conv3_block10_concat[0][0]       
__________________________________________________________________________________________________
conv3_block11_0_relu (Activatio (None, 28, 28, 448)  0           conv3_block11_0_bn[0][0]         
__________________________________________________________________________________________________
conv3_block11_1_conv (Conv2D)   (None, 28, 28, 128)  57344       conv3_block11_0_relu[0][0]       
__________________________________________________________________________________________________
conv3_block11_1_bn (BatchNormal (None, 28, 28, 128)  512         conv3_block11_1_conv[0][0]       
__________________________________________________________________________________________________
conv3_block11_1_relu (Activatio (None, 28, 28, 128)  0           conv3_block11_1_bn[0][0]         
__________________________________________________________________________________________________
conv3_block11_2_conv (Conv2D)   (None, 28, 28, 32)   36864       conv3_block11_1_relu[0][0]       
__________________________________________________________________________________________________
conv3_block11_concat (Concatena (None, 28, 28, 480)  0           conv3_block10_concat[0][0]       
                                                                 conv3_block11_2_conv[0][0]       
__________________________________________________________________________________________________
conv3_block12_0_bn (BatchNormal (None, 28, 28, 480)  1920        conv3_block11_concat[0][0]       
__________________________________________________________________________________________________
conv3_block12_0_relu (Activatio (None, 28, 28, 480)  0           conv3_block12_0_bn[0][0]         
__________________________________________________________________________________________________
conv3_block12_1_conv (Conv2D)   (None, 28, 28, 128)  61440       conv3_block12_0_relu[0][0]       
__________________________________________________________________________________________________
conv3_block12_1_bn (BatchNormal (None, 28, 28, 128)  512         conv3_block12_1_conv[0][0]       
__________________________________________________________________________________________________
conv3_block12_1_relu (Activatio (None, 28, 28, 128)  0           conv3_block12_1_bn[0][0]         
__________________________________________________________________________________________________
conv3_block12_2_conv (Conv2D)   (None, 28, 28, 32)   36864       conv3_block12_1_relu[0][0]       
__________________________________________________________________________________________________
conv3_block12_concat (Concatena (None, 28, 28, 512)  0           conv3_block11_concat[0][0]       
                                                                 conv3_block12_2_conv[0][0]       
__________________________________________________________________________________________________
pool3_bn (BatchNormalization)   (None, 28, 28, 512)  2048        conv3_block12_concat[0][0]       
__________________________________________________________________________________________________
pool3_relu (Activation)         (None, 28, 28, 512)  0           pool3_bn[0][0]                   
__________________________________________________________________________________________________
pool3_conv (Conv2D)             (None, 28, 28, 256)  131072      pool3_relu[0][0]                 
__________________________________________________________________________________________________
pool3_pool (AveragePooling2D)   (None, 14, 14, 256)  0           pool3_conv[0][0]                 
__________________________________________________________________________________________________
conv4_block1_0_bn (BatchNormali (None, 14, 14, 256)  1024        pool3_pool[0][0]                 
__________________________________________________________________________________________________
conv4_block1_0_relu (Activation (None, 14, 14, 256)  0           conv4_block1_0_bn[0][0]          
__________________________________________________________________________________________________
conv4_block1_1_conv (Conv2D)    (None, 14, 14, 128)  32768       conv4_block1_0_relu[0][0]        
__________________________________________________________________________________________________
conv4_block1_1_bn (BatchNormali (None, 14, 14, 128)  512         conv4_block1_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block1_1_relu (Activation (None, 14, 14, 128)  0           conv4_block1_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block1_2_conv (Conv2D)    (None, 14, 14, 32)   36864       conv4_block1_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block1_concat (Concatenat (None, 14, 14, 288)  0           pool3_pool[0][0]                 
                                                                 conv4_block1_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block2_0_bn (BatchNormali (None, 14, 14, 288)  1152        conv4_block1_concat[0][0]        
__________________________________________________________________________________________________
conv4_block2_0_relu (Activation (None, 14, 14, 288)  0           conv4_block2_0_bn[0][0]          
__________________________________________________________________________________________________
conv4_block2_1_conv (Conv2D)    (None, 14, 14, 128)  36864       conv4_block2_0_relu[0][0]        
__________________________________________________________________________________________________
conv4_block2_1_bn (BatchNormali (None, 14, 14, 128)  512         conv4_block2_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block2_1_relu (Activation (None, 14, 14, 128)  0           conv4_block2_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block2_2_conv (Conv2D)    (None, 14, 14, 32)   36864       conv4_block2_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block2_concat (Concatenat (None, 14, 14, 320)  0           conv4_block1_concat[0][0]        
                                                                 conv4_block2_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block3_0_bn (BatchNormali (None, 14, 14, 320)  1280        conv4_block2_concat[0][0]        
__________________________________________________________________________________________________
conv4_block3_0_relu (Activation (None, 14, 14, 320)  0           conv4_block3_0_bn[0][0]          
__________________________________________________________________________________________________
conv4_block3_1_conv (Conv2D)    (None, 14, 14, 128)  40960       conv4_block3_0_relu[0][0]        
__________________________________________________________________________________________________
conv4_block3_1_bn (BatchNormali (None, 14, 14, 128)  512         conv4_block3_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block3_1_relu (Activation (None, 14, 14, 128)  0           conv4_block3_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block3_2_conv (Conv2D)    (None, 14, 14, 32)   36864       conv4_block3_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block3_concat (Concatenat (None, 14, 14, 352)  0           conv4_block2_concat[0][0]        
                                                                 conv4_block3_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block4_0_bn (BatchNormali (None, 14, 14, 352)  1408        conv4_block3_concat[0][0]        
__________________________________________________________________________________________________
conv4_block4_0_relu (Activation (None, 14, 14, 352)  0           conv4_block4_0_bn[0][0]          
__________________________________________________________________________________________________
conv4_block4_1_conv (Conv2D)    (None, 14, 14, 128)  45056       conv4_block4_0_relu[0][0]        
__________________________________________________________________________________________________
conv4_block4_1_bn (BatchNormali (None, 14, 14, 128)  512         conv4_block4_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block4_1_relu (Activation (None, 14, 14, 128)  0           conv4_block4_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block4_2_conv (Conv2D)    (None, 14, 14, 32)   36864       conv4_block4_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block4_concat (Concatenat (None, 14, 14, 384)  0           conv4_block3_concat[0][0]        
                                                                 conv4_block4_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block5_0_bn (BatchNormali (None, 14, 14, 384)  1536        conv4_block4_concat[0][0]        
__________________________________________________________________________________________________
conv4_block5_0_relu (Activation (None, 14, 14, 384)  0           conv4_block5_0_bn[0][0]          
__________________________________________________________________________________________________
conv4_block5_1_conv (Conv2D)    (None, 14, 14, 128)  49152       conv4_block5_0_relu[0][0]        
__________________________________________________________________________________________________
conv4_block5_1_bn (BatchNormali (None, 14, 14, 128)  512         conv4_block5_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block5_1_relu (Activation (None, 14, 14, 128)  0           conv4_block5_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block5_2_conv (Conv2D)    (None, 14, 14, 32)   36864       conv4_block5_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block5_concat (Concatenat (None, 14, 14, 416)  0           conv4_block4_concat[0][0]        
                                                                 conv4_block5_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block6_0_bn (BatchNormali (None, 14, 14, 416)  1664        conv4_block5_concat[0][0]        
__________________________________________________________________________________________________
conv4_block6_0_relu (Activation (None, 14, 14, 416)  0           conv4_block6_0_bn[0][0]          
__________________________________________________________________________________________________
conv4_block6_1_conv (Conv2D)    (None, 14, 14, 128)  53248       conv4_block6_0_relu[0][0]        
__________________________________________________________________________________________________
conv4_block6_1_bn (BatchNormali (None, 14, 14, 128)  512         conv4_block6_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block6_1_relu (Activation (None, 14, 14, 128)  0           conv4_block6_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block6_2_conv (Conv2D)    (None, 14, 14, 32)   36864       conv4_block6_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block6_concat (Concatenat (None, 14, 14, 448)  0           conv4_block5_concat[0][0]        
                                                                 conv4_block6_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block7_0_bn (BatchNormali (None, 14, 14, 448)  1792        conv4_block6_concat[0][0]        
__________________________________________________________________________________________________
conv4_block7_0_relu (Activation (None, 14, 14, 448)  0           conv4_block7_0_bn[0][0]          
__________________________________________________________________________________________________
conv4_block7_1_conv (Conv2D)    (None, 14, 14, 128)  57344       conv4_block7_0_relu[0][0]        
__________________________________________________________________________________________________
conv4_block7_1_bn (BatchNormali (None, 14, 14, 128)  512         conv4_block7_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block7_1_relu (Activation (None, 14, 14, 128)  0           conv4_block7_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block7_2_conv (Conv2D)    (None, 14, 14, 32)   36864       conv4_block7_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block7_concat (Concatenat (None, 14, 14, 480)  0           conv4_block6_concat[0][0]        
                                                                 conv4_block7_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block8_0_bn (BatchNormali (None, 14, 14, 480)  1920        conv4_block7_concat[0][0]        
__________________________________________________________________________________________________
conv4_block8_0_relu (Activation (None, 14, 14, 480)  0           conv4_block8_0_bn[0][0]          
__________________________________________________________________________________________________
conv4_block8_1_conv (Conv2D)    (None, 14, 14, 128)  61440       conv4_block8_0_relu[0][0]        
__________________________________________________________________________________________________
conv4_block8_1_bn (BatchNormali (None, 14, 14, 128)  512         conv4_block8_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block8_1_relu (Activation (None, 14, 14, 128)  0           conv4_block8_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block8_2_conv (Conv2D)    (None, 14, 14, 32)   36864       conv4_block8_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block8_concat (Concatenat (None, 14, 14, 512)  0           conv4_block7_concat[0][0]        
                                                                 conv4_block8_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block9_0_bn (BatchNormali (None, 14, 14, 512)  2048        conv4_block8_concat[0][0]        
__________________________________________________________________________________________________
conv4_block9_0_relu (Activation (None, 14, 14, 512)  0           conv4_block9_0_bn[0][0]          
__________________________________________________________________________________________________
conv4_block9_1_conv (Conv2D)    (None, 14, 14, 128)  65536       conv4_block9_0_relu[0][0]        
__________________________________________________________________________________________________
conv4_block9_1_bn (BatchNormali (None, 14, 14, 128)  512         conv4_block9_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block9_1_relu (Activation (None, 14, 14, 128)  0           conv4_block9_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block9_2_conv (Conv2D)    (None, 14, 14, 32)   36864       conv4_block9_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block9_concat (Concatenat (None, 14, 14, 544)  0           conv4_block8_concat[0][0]        
                                                                 conv4_block9_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block10_0_bn (BatchNormal (None, 14, 14, 544)  2176        conv4_block9_concat[0][0]        
__________________________________________________________________________________________________
conv4_block10_0_relu (Activatio (None, 14, 14, 544)  0           conv4_block10_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block10_1_conv (Conv2D)   (None, 14, 14, 128)  69632       conv4_block10_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block10_1_bn (BatchNormal (None, 14, 14, 128)  512         conv4_block10_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block10_1_relu (Activatio (None, 14, 14, 128)  0           conv4_block10_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block10_2_conv (Conv2D)   (None, 14, 14, 32)   36864       conv4_block10_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block10_concat (Concatena (None, 14, 14, 576)  0           conv4_block9_concat[0][0]        
                                                                 conv4_block10_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block11_0_bn (BatchNormal (None, 14, 14, 576)  2304        conv4_block10_concat[0][0]       
__________________________________________________________________________________________________
conv4_block11_0_relu (Activatio (None, 14, 14, 576)  0           conv4_block11_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block11_1_conv (Conv2D)   (None, 14, 14, 128)  73728       conv4_block11_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block11_1_bn (BatchNormal (None, 14, 14, 128)  512         conv4_block11_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block11_1_relu (Activatio (None, 14, 14, 128)  0           conv4_block11_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block11_2_conv (Conv2D)   (None, 14, 14, 32)   36864       conv4_block11_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block11_concat (Concatena (None, 14, 14, 608)  0           conv4_block10_concat[0][0]       
                                                                 conv4_block11_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block12_0_bn (BatchNormal (None, 14, 14, 608)  2432        conv4_block11_concat[0][0]       
__________________________________________________________________________________________________
conv4_block12_0_relu (Activatio (None, 14, 14, 608)  0           conv4_block12_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block12_1_conv (Conv2D)   (None, 14, 14, 128)  77824       conv4_block12_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block12_1_bn (BatchNormal (None, 14, 14, 128)  512         conv4_block12_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block12_1_relu (Activatio (None, 14, 14, 128)  0           conv4_block12_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block12_2_conv (Conv2D)   (None, 14, 14, 32)   36864       conv4_block12_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block12_concat (Concatena (None, 14, 14, 640)  0           conv4_block11_concat[0][0]       
                                                                 conv4_block12_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block13_0_bn (BatchNormal (None, 14, 14, 640)  2560        conv4_block12_concat[0][0]       
__________________________________________________________________________________________________
conv4_block13_0_relu (Activatio (None, 14, 14, 640)  0           conv4_block13_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block13_1_conv (Conv2D)   (None, 14, 14, 128)  81920       conv4_block13_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block13_1_bn (BatchNormal (None, 14, 14, 128)  512         conv4_block13_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block13_1_relu (Activatio (None, 14, 14, 128)  0           conv4_block13_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block13_2_conv (Conv2D)   (None, 14, 14, 32)   36864       conv4_block13_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block13_concat (Concatena (None, 14, 14, 672)  0           conv4_block12_concat[0][0]       
                                                                 conv4_block13_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block14_0_bn (BatchNormal (None, 14, 14, 672)  2688        conv4_block13_concat[0][0]       
__________________________________________________________________________________________________
conv4_block14_0_relu (Activatio (None, 14, 14, 672)  0           conv4_block14_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block14_1_conv (Conv2D)   (None, 14, 14, 128)  86016       conv4_block14_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block14_1_bn (BatchNormal (None, 14, 14, 128)  512         conv4_block14_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block14_1_relu (Activatio (None, 14, 14, 128)  0           conv4_block14_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block14_2_conv (Conv2D)   (None, 14, 14, 32)   36864       conv4_block14_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block14_concat (Concatena (None, 14, 14, 704)  0           conv4_block13_concat[0][0]       
                                                                 conv4_block14_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block15_0_bn (BatchNormal (None, 14, 14, 704)  2816        conv4_block14_concat[0][0]       
__________________________________________________________________________________________________
conv4_block15_0_relu (Activatio (None, 14, 14, 704)  0           conv4_block15_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block15_1_conv (Conv2D)   (None, 14, 14, 128)  90112       conv4_block15_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block15_1_bn (BatchNormal (None, 14, 14, 128)  512         conv4_block15_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block15_1_relu (Activatio (None, 14, 14, 128)  0           conv4_block15_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block15_2_conv (Conv2D)   (None, 14, 14, 32)   36864       conv4_block15_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block15_concat (Concatena (None, 14, 14, 736)  0           conv4_block14_concat[0][0]       
                                                                 conv4_block15_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block16_0_bn (BatchNormal (None, 14, 14, 736)  2944        conv4_block15_concat[0][0]       
__________________________________________________________________________________________________
conv4_block16_0_relu (Activatio (None, 14, 14, 736)  0           conv4_block16_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block16_1_conv (Conv2D)   (None, 14, 14, 128)  94208       conv4_block16_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block16_1_bn (BatchNormal (None, 14, 14, 128)  512         conv4_block16_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block16_1_relu (Activatio (None, 14, 14, 128)  0           conv4_block16_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block16_2_conv (Conv2D)   (None, 14, 14, 32)   36864       conv4_block16_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block16_concat (Concatena (None, 14, 14, 768)  0           conv4_block15_concat[0][0]       
                                                                 conv4_block16_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block17_0_bn (BatchNormal (None, 14, 14, 768)  3072        conv4_block16_concat[0][0]       
__________________________________________________________________________________________________
conv4_block17_0_relu (Activatio (None, 14, 14, 768)  0           conv4_block17_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block17_1_conv (Conv2D)   (None, 14, 14, 128)  98304       conv4_block17_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block17_1_bn (BatchNormal (None, 14, 14, 128)  512         conv4_block17_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block17_1_relu (Activatio (None, 14, 14, 128)  0           conv4_block17_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block17_2_conv (Conv2D)   (None, 14, 14, 32)   36864       conv4_block17_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block17_concat (Concatena (None, 14, 14, 800)  0           conv4_block16_concat[0][0]       
                                                                 conv4_block17_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block18_0_bn (BatchNormal (None, 14, 14, 800)  3200        conv4_block17_concat[0][0]       
__________________________________________________________________________________________________
conv4_block18_0_relu (Activatio (None, 14, 14, 800)  0           conv4_block18_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block18_1_conv (Conv2D)   (None, 14, 14, 128)  102400      conv4_block18_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block18_1_bn (BatchNormal (None, 14, 14, 128)  512         conv4_block18_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block18_1_relu (Activatio (None, 14, 14, 128)  0           conv4_block18_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block18_2_conv (Conv2D)   (None, 14, 14, 32)   36864       conv4_block18_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block18_concat (Concatena (None, 14, 14, 832)  0           conv4_block17_concat[0][0]       
                                                                 conv4_block18_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block19_0_bn (BatchNormal (None, 14, 14, 832)  3328        conv4_block18_concat[0][0]       
__________________________________________________________________________________________________
conv4_block19_0_relu (Activatio (None, 14, 14, 832)  0           conv4_block19_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block19_1_conv (Conv2D)   (None, 14, 14, 128)  106496      conv4_block19_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block19_1_bn (BatchNormal (None, 14, 14, 128)  512         conv4_block19_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block19_1_relu (Activatio (None, 14, 14, 128)  0           conv4_block19_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block19_2_conv (Conv2D)   (None, 14, 14, 32)   36864       conv4_block19_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block19_concat (Concatena (None, 14, 14, 864)  0           conv4_block18_concat[0][0]       
                                                                 conv4_block19_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block20_0_bn (BatchNormal (None, 14, 14, 864)  3456        conv4_block19_concat[0][0]       
__________________________________________________________________________________________________
conv4_block20_0_relu (Activatio (None, 14, 14, 864)  0           conv4_block20_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block20_1_conv (Conv2D)   (None, 14, 14, 128)  110592      conv4_block20_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block20_1_bn (BatchNormal (None, 14, 14, 128)  512         conv4_block20_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block20_1_relu (Activatio (None, 14, 14, 128)  0           conv4_block20_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block20_2_conv (Conv2D)   (None, 14, 14, 32)   36864       conv4_block20_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block20_concat (Concatena (None, 14, 14, 896)  0           conv4_block19_concat[0][0]       
                                                                 conv4_block20_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block21_0_bn (BatchNormal (None, 14, 14, 896)  3584        conv4_block20_concat[0][0]       
__________________________________________________________________________________________________
conv4_block21_0_relu (Activatio (None, 14, 14, 896)  0           conv4_block21_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block21_1_conv (Conv2D)   (None, 14, 14, 128)  114688      conv4_block21_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block21_1_bn (BatchNormal (None, 14, 14, 128)  512         conv4_block21_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block21_1_relu (Activatio (None, 14, 14, 128)  0           conv4_block21_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block21_2_conv (Conv2D)   (None, 14, 14, 32)   36864       conv4_block21_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block21_concat (Concatena (None, 14, 14, 928)  0           conv4_block20_concat[0][0]       
                                                                 conv4_block21_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block22_0_bn (BatchNormal (None, 14, 14, 928)  3712        conv4_block21_concat[0][0]       
__________________________________________________________________________________________________
conv4_block22_0_relu (Activatio (None, 14, 14, 928)  0           conv4_block22_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block22_1_conv (Conv2D)   (None, 14, 14, 128)  118784      conv4_block22_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block22_1_bn (BatchNormal (None, 14, 14, 128)  512         conv4_block22_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block22_1_relu (Activatio (None, 14, 14, 128)  0           conv4_block22_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block22_2_conv (Conv2D)   (None, 14, 14, 32)   36864       conv4_block22_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block22_concat (Concatena (None, 14, 14, 960)  0           conv4_block21_concat[0][0]       
                                                                 conv4_block22_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block23_0_bn (BatchNormal (None, 14, 14, 960)  3840        conv4_block22_concat[0][0]       
__________________________________________________________________________________________________
conv4_block23_0_relu (Activatio (None, 14, 14, 960)  0           conv4_block23_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block23_1_conv (Conv2D)   (None, 14, 14, 128)  122880      conv4_block23_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block23_1_bn (BatchNormal (None, 14, 14, 128)  512         conv4_block23_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block23_1_relu (Activatio (None, 14, 14, 128)  0           conv4_block23_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block23_2_conv (Conv2D)   (None, 14, 14, 32)   36864       conv4_block23_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block23_concat (Concatena (None, 14, 14, 992)  0           conv4_block22_concat[0][0]       
                                                                 conv4_block23_2_conv[0][0]       
__________________________________________________________________________________________________
conv4_block24_0_bn (BatchNormal (None, 14, 14, 992)  3968        conv4_block23_concat[0][0]       
__________________________________________________________________________________________________
conv4_block24_0_relu (Activatio (None, 14, 14, 992)  0           conv4_block24_0_bn[0][0]         
__________________________________________________________________________________________________
conv4_block24_1_conv (Conv2D)   (None, 14, 14, 128)  126976      conv4_block24_0_relu[0][0]       
__________________________________________________________________________________________________
conv4_block24_1_bn (BatchNormal (None, 14, 14, 128)  512         conv4_block24_1_conv[0][0]       
__________________________________________________________________________________________________
conv4_block24_1_relu (Activatio (None, 14, 14, 128)  0           conv4_block24_1_bn[0][0]         
__________________________________________________________________________________________________
conv4_block24_2_conv (Conv2D)   (None, 14, 14, 32)   36864       conv4_block24_1_relu[0][0]       
__________________________________________________________________________________________________
conv4_block24_concat (Concatena (None, 14, 14, 1024) 0           conv4_block23_concat[0][0]       
                                                                 conv4_block24_2_conv[0][0]       
__________________________________________________________________________________________________
pool4_bn (BatchNormalization)   (None, 14, 14, 1024) 4096        conv4_block24_concat[0][0]       
__________________________________________________________________________________________________
pool4_relu (Activation)         (None, 14, 14, 1024) 0           pool4_bn[0][0]                   
__________________________________________________________________________________________________
pool4_conv (Conv2D)             (None, 14, 14, 512)  524288      pool4_relu[0][0]                 
__________________________________________________________________________________________________
pool4_pool (AveragePooling2D)   (None, 7, 7, 512)    0           pool4_conv[0][0]                 
__________________________________________________________________________________________________
conv5_block1_0_bn (BatchNormali (None, 7, 7, 512)    2048        pool4_pool[0][0]                 
__________________________________________________________________________________________________
conv5_block1_0_relu (Activation (None, 7, 7, 512)    0           conv5_block1_0_bn[0][0]          
__________________________________________________________________________________________________
conv5_block1_1_conv (Conv2D)    (None, 7, 7, 128)    65536       conv5_block1_0_relu[0][0]        
__________________________________________________________________________________________________
conv5_block1_1_bn (BatchNormali (None, 7, 7, 128)    512         conv5_block1_1_conv[0][0]        
__________________________________________________________________________________________________
conv5_block1_1_relu (Activation (None, 7, 7, 128)    0           conv5_block1_1_bn[0][0]          
__________________________________________________________________________________________________
conv5_block1_2_conv (Conv2D)    (None, 7, 7, 32)     36864       conv5_block1_1_relu[0][0]        
__________________________________________________________________________________________________
conv5_block1_concat (Concatenat (None, 7, 7, 544)    0           pool4_pool[0][0]                 
                                                                 conv5_block1_2_conv[0][0]        
__________________________________________________________________________________________________
conv5_block2_0_bn (BatchNormali (None, 7, 7, 544)    2176        conv5_block1_concat[0][0]        
__________________________________________________________________________________________________
conv5_block2_0_relu (Activation (None, 7, 7, 544)    0           conv5_block2_0_bn[0][0]          
__________________________________________________________________________________________________
conv5_block2_1_conv (Conv2D)    (None, 7, 7, 128)    69632       conv5_block2_0_relu[0][0]        
__________________________________________________________________________________________________
conv5_block2_1_bn (BatchNormali (None, 7, 7, 128)    512         conv5_block2_1_conv[0][0]        
__________________________________________________________________________________________________
conv5_block2_1_relu (Activation (None, 7, 7, 128)    0           conv5_block2_1_bn[0][0]          
__________________________________________________________________________________________________
conv5_block2_2_conv (Conv2D)    (None, 7, 7, 32)     36864       conv5_block2_1_relu[0][0]        
__________________________________________________________________________________________________
conv5_block2_concat (Concatenat (None, 7, 7, 576)    0           conv5_block1_concat[0][0]        
                                                                 conv5_block2_2_conv[0][0]        
__________________________________________________________________________________________________
conv5_block3_0_bn (BatchNormali (None, 7, 7, 576)    2304        conv5_block2_concat[0][0]        
__________________________________________________________________________________________________
conv5_block3_0_relu (Activation (None, 7, 7, 576)    0           conv5_block3_0_bn[0][0]          
__________________________________________________________________________________________________
conv5_block3_1_conv (Conv2D)    (None, 7, 7, 128)    73728       conv5_block3_0_relu[0][0]        
__________________________________________________________________________________________________
conv5_block3_1_bn (BatchNormali (None, 7, 7, 128)    512         conv5_block3_1_conv[0][0]        
__________________________________________________________________________________________________
conv5_block3_1_relu (Activation (None, 7, 7, 128)    0           conv5_block3_1_bn[0][0]          
__________________________________________________________________________________________________
conv5_block3_2_conv (Conv2D)    (None, 7, 7, 32)     36864       conv5_block3_1_relu[0][0]        
__________________________________________________________________________________________________
conv5_block3_concat (Concatenat (None, 7, 7, 608)    0           conv5_block2_concat[0][0]        
                                                                 conv5_block3_2_conv[0][0]        
__________________________________________________________________________________________________
conv5_block4_0_bn (BatchNormali (None, 7, 7, 608)    2432        conv5_block3_concat[0][0]        
__________________________________________________________________________________________________
conv5_block4_0_relu (Activation (None, 7, 7, 608)    0           conv5_block4_0_bn[0][0]          
__________________________________________________________________________________________________
conv5_block4_1_conv (Conv2D)    (None, 7, 7, 128)    77824       conv5_block4_0_relu[0][0]        
__________________________________________________________________________________________________
conv5_block4_1_bn (BatchNormali (None, 7, 7, 128)    512         conv5_block4_1_conv[0][0]        
__________________________________________________________________________________________________
conv5_block4_1_relu (Activation (None, 7, 7, 128)    0           conv5_block4_1_bn[0][0]          
__________________________________________________________________________________________________
conv5_block4_2_conv (Conv2D)    (None, 7, 7, 32)     36864       conv5_block4_1_relu[0][0]        
__________________________________________________________________________________________________
conv5_block4_concat (Concatenat (None, 7, 7, 640)    0           conv5_block3_concat[0][0]        
                                                                 conv5_block4_2_conv[0][0]        
__________________________________________________________________________________________________
conv5_block5_0_bn (BatchNormali (None, 7, 7, 640)    2560        conv5_block4_concat[0][0]        
__________________________________________________________________________________________________
conv5_block5_0_relu (Activation (None, 7, 7, 640)    0           conv5_block5_0_bn[0][0]          
__________________________________________________________________________________________________
conv5_block5_1_conv (Conv2D)    (None, 7, 7, 128)    81920       conv5_block5_0_relu[0][0]        
__________________________________________________________________________________________________
conv5_block5_1_bn (BatchNormali (None, 7, 7, 128)    512         conv5_block5_1_conv[0][0]        
__________________________________________________________________________________________________
conv5_block5_1_relu (Activation (None, 7, 7, 128)    0           conv5_block5_1_bn[0][0]          
__________________________________________________________________________________________________
conv5_block5_2_conv (Conv2D)    (None, 7, 7, 32)     36864       conv5_block5_1_relu[0][0]        
__________________________________________________________________________________________________
conv5_block5_concat (Concatenat (None, 7, 7, 672)    0           conv5_block4_concat[0][0]        
                                                                 conv5_block5_2_conv[0][0]        
__________________________________________________________________________________________________
conv5_block6_0_bn (BatchNormali (None, 7, 7, 672)    2688        conv5_block5_concat[0][0]        
__________________________________________________________________________________________________
conv5_block6_0_relu (Activation (None, 7, 7, 672)    0           conv5_block6_0_bn[0][0]          
__________________________________________________________________________________________________
conv5_block6_1_conv (Conv2D)    (None, 7, 7, 128)    86016       conv5_block6_0_relu[0][0]        
__________________________________________________________________________________________________
conv5_block6_1_bn (BatchNormali (None, 7, 7, 128)    512         conv5_block6_1_conv[0][0]        
__________________________________________________________________________________________________
conv5_block6_1_relu (Activation (None, 7, 7, 128)    0           conv5_block6_1_bn[0][0]          
__________________________________________________________________________________________________
conv5_block6_2_conv (Conv2D)    (None, 7, 7, 32)     36864       conv5_block6_1_relu[0][0]        
__________________________________________________________________________________________________
conv5_block6_concat (Concatenat (None, 7, 7, 704)    0           conv5_block5_concat[0][0]        
                                                                 conv5_block6_2_conv[0][0]        
__________________________________________________________________________________________________
conv5_block7_0_bn (BatchNormali (None, 7, 7, 704)    2816        conv5_block6_concat[0][0]        
__________________________________________________________________________________________________
conv5_block7_0_relu (Activation (None, 7, 7, 704)    0           conv5_block7_0_bn[0][0]          
__________________________________________________________________________________________________
conv5_block7_1_conv (Conv2D)    (None, 7, 7, 128)    90112       conv5_block7_0_relu[0][0]        
__________________________________________________________________________________________________
conv5_block7_1_bn (BatchNormali (None, 7, 7, 128)    512         conv5_block7_1_conv[0][0]        
__________________________________________________________________________________________________
conv5_block7_1_relu (Activation (None, 7, 7, 128)    0           conv5_block7_1_bn[0][0]          
__________________________________________________________________________________________________
conv5_block7_2_conv (Conv2D)    (None, 7, 7, 32)     36864       conv5_block7_1_relu[0][0]        
__________________________________________________________________________________________________
conv5_block7_concat (Concatenat (None, 7, 7, 736)    0           conv5_block6_concat[0][0]        
                                                                 conv5_block7_2_conv[0][0]        
__________________________________________________________________________________________________
conv5_block8_0_bn (BatchNormali (None, 7, 7, 736)    2944        conv5_block7_concat[0][0]        
__________________________________________________________________________________________________
conv5_block8_0_relu (Activation (None, 7, 7, 736)    0           conv5_block8_0_bn[0][0]          
__________________________________________________________________________________________________
conv5_block8_1_conv (Conv2D)    (None, 7, 7, 128)    94208       conv5_block8_0_relu[0][0]        
__________________________________________________________________________________________________
conv5_block8_1_bn (BatchNormali (None, 7, 7, 128)    512         conv5_block8_1_conv[0][0]        
__________________________________________________________________________________________________
conv5_block8_1_relu (Activation (None, 7, 7, 128)    0           conv5_block8_1_bn[0][0]          
__________________________________________________________________________________________________
conv5_block8_2_conv (Conv2D)    (None, 7, 7, 32)     36864       conv5_block8_1_relu[0][0]        
__________________________________________________________________________________________________
conv5_block8_concat (Concatenat (None, 7, 7, 768)    0           conv5_block7_concat[0][0]        
                                                                 conv5_block8_2_conv[0][0]        
__________________________________________________________________________________________________
conv5_block9_0_bn (BatchNormali (None, 7, 7, 768)    3072        conv5_block8_concat[0][0]        
__________________________________________________________________________________________________
conv5_block9_0_relu (Activation (None, 7, 7, 768)    0           conv5_block9_0_bn[0][0]          
__________________________________________________________________________________________________
conv5_block9_1_conv (Conv2D)    (None, 7, 7, 128)    98304       conv5_block9_0_relu[0][0]        
__________________________________________________________________________________________________
conv5_block9_1_bn (BatchNormali (None, 7, 7, 128)    512         conv5_block9_1_conv[0][0]        
__________________________________________________________________________________________________
conv5_block9_1_relu (Activation (None, 7, 7, 128)    0           conv5_block9_1_bn[0][0]          
__________________________________________________________________________________________________
conv5_block9_2_conv (Conv2D)    (None, 7, 7, 32)     36864       conv5_block9_1_relu[0][0]        
__________________________________________________________________________________________________
conv5_block9_concat (Concatenat (None, 7, 7, 800)    0           conv5_block8_concat[0][0]        
                                                                 conv5_block9_2_conv[0][0]        
__________________________________________________________________________________________________
conv5_block10_0_bn (BatchNormal (None, 7, 7, 800)    3200        conv5_block9_concat[0][0]        
__________________________________________________________________________________________________
conv5_block10_0_relu (Activatio (None, 7, 7, 800)    0           conv5_block10_0_bn[0][0]         
__________________________________________________________________________________________________
conv5_block10_1_conv (Conv2D)   (None, 7, 7, 128)    102400      conv5_block10_0_relu[0][0]       
__________________________________________________________________________________________________
conv5_block10_1_bn (BatchNormal (None, 7, 7, 128)    512         conv5_block10_1_conv[0][0]       
__________________________________________________________________________________________________
conv5_block10_1_relu (Activatio (None, 7, 7, 128)    0           conv5_block10_1_bn[0][0]         
__________________________________________________________________________________________________
conv5_block10_2_conv (Conv2D)   (None, 7, 7, 32)     36864       conv5_block10_1_relu[0][0]       
__________________________________________________________________________________________________
conv5_block10_concat (Concatena (None, 7, 7, 832)    0           conv5_block9_concat[0][0]        
                                                                 conv5_block10_2_conv[0][0]       
__________________________________________________________________________________________________
conv5_block11_0_bn (BatchNormal (None, 7, 7, 832)    3328        conv5_block10_concat[0][0]       
__________________________________________________________________________________________________
conv5_block11_0_relu (Activatio (None, 7, 7, 832)    0           conv5_block11_0_bn[0][0]         
__________________________________________________________________________________________________
conv5_block11_1_conv (Conv2D)   (None, 7, 7, 128)    106496      conv5_block11_0_relu[0][0]       
__________________________________________________________________________________________________
conv5_block11_1_bn (BatchNormal (None, 7, 7, 128)    512         conv5_block11_1_conv[0][0]       
__________________________________________________________________________________________________
conv5_block11_1_relu (Activatio (None, 7, 7, 128)    0           conv5_block11_1_bn[0][0]         
__________________________________________________________________________________________________
conv5_block11_2_conv (Conv2D)   (None, 7, 7, 32)     36864       conv5_block11_1_relu[0][0]       
__________________________________________________________________________________________________
conv5_block11_concat (Concatena (None, 7, 7, 864)    0           conv5_block10_concat[0][0]       
                                                                 conv5_block11_2_conv[0][0]       
__________________________________________________________________________________________________
conv5_block12_0_bn (BatchNormal (None, 7, 7, 864)    3456        conv5_block11_concat[0][0]       
__________________________________________________________________________________________________
conv5_block12_0_relu (Activatio (None, 7, 7, 864)    0           conv5_block12_0_bn[0][0]         
__________________________________________________________________________________________________
conv5_block12_1_conv (Conv2D)   (None, 7, 7, 128)    110592      conv5_block12_0_relu[0][0]       
__________________________________________________________________________________________________
conv5_block12_1_bn (BatchNormal (None, 7, 7, 128)    512         conv5_block12_1_conv[0][0]       
__________________________________________________________________________________________________
conv5_block12_1_relu (Activatio (None, 7, 7, 128)    0           conv5_block12_1_bn[0][0]         
__________________________________________________________________________________________________
conv5_block12_2_conv (Conv2D)   (None, 7, 7, 32)     36864       conv5_block12_1_relu[0][0]       
__________________________________________________________________________________________________
conv5_block12_concat (Concatena (None, 7, 7, 896)    0           conv5_block11_concat[0][0]       
                                                                 conv5_block12_2_conv[0][0]       
__________________________________________________________________________________________________
conv5_block13_0_bn (BatchNormal (None, 7, 7, 896)    3584        conv5_block12_concat[0][0]       
__________________________________________________________________________________________________
conv5_block13_0_relu (Activatio (None, 7, 7, 896)    0           conv5_block13_0_bn[0][0]         
__________________________________________________________________________________________________
conv5_block13_1_conv (Conv2D)   (None, 7, 7, 128)    114688      conv5_block13_0_relu[0][0]       
__________________________________________________________________________________________________
conv5_block13_1_bn (BatchNormal (None, 7, 7, 128)    512         conv5_block13_1_conv[0][0]       
__________________________________________________________________________________________________
conv5_block13_1_relu (Activatio (None, 7, 7, 128)    0           conv5_block13_1_bn[0][0]         
__________________________________________________________________________________________________
conv5_block13_2_conv (Conv2D)   (None, 7, 7, 32)     36864       conv5_block13_1_relu[0][0]       
__________________________________________________________________________________________________
conv5_block13_concat (Concatena (None, 7, 7, 928)    0           conv5_block12_concat[0][0]       
                                                                 conv5_block13_2_conv[0][0]       
__________________________________________________________________________________________________
conv5_block14_0_bn (BatchNormal (None, 7, 7, 928)    3712        conv5_block13_concat[0][0]       
__________________________________________________________________________________________________
conv5_block14_0_relu (Activatio (None, 7, 7, 928)    0           conv5_block14_0_bn[0][0]         
__________________________________________________________________________________________________
conv5_block14_1_conv (Conv2D)   (None, 7, 7, 128)    118784      conv5_block14_0_relu[0][0]       
__________________________________________________________________________________________________
conv5_block14_1_bn (BatchNormal (None, 7, 7, 128)    512         conv5_block14_1_conv[0][0]       
__________________________________________________________________________________________________
conv5_block14_1_relu (Activatio (None, 7, 7, 128)    0           conv5_block14_1_bn[0][0]         
__________________________________________________________________________________________________
conv5_block14_2_conv (Conv2D)   (None, 7, 7, 32)     36864       conv5_block14_1_relu[0][0]       
__________________________________________________________________________________________________
conv5_block14_concat (Concatena (None, 7, 7, 960)    0           conv5_block13_concat[0][0]       
                                                                 conv5_block14_2_conv[0][0]       
__________________________________________________________________________________________________
conv5_block15_0_bn (BatchNormal (None, 7, 7, 960)    3840        conv5_block14_concat[0][0]       
__________________________________________________________________________________________________
conv5_block15_0_relu (Activatio (None, 7, 7, 960)    0           conv5_block15_0_bn[0][0]         
__________________________________________________________________________________________________
conv5_block15_1_conv (Conv2D)   (None, 7, 7, 128)    122880      conv5_block15_0_relu[0][0]       
__________________________________________________________________________________________________
conv5_block15_1_bn (BatchNormal (None, 7, 7, 128)    512         conv5_block15_1_conv[0][0]       
__________________________________________________________________________________________________
conv5_block15_1_relu (Activatio (None, 7, 7, 128)    0           conv5_block15_1_bn[0][0]         
__________________________________________________________________________________________________
conv5_block15_2_conv (Conv2D)   (None, 7, 7, 32)     36864       conv5_block15_1_relu[0][0]       
__________________________________________________________________________________________________
conv5_block15_concat (Concatena (None, 7, 7, 992)    0           conv5_block14_concat[0][0]       
                                                                 conv5_block15_2_conv[0][0]       
__________________________________________________________________________________________________
conv5_block16_0_bn (BatchNormal (None, 7, 7, 992)    3968        conv5_block15_concat[0][0]       
__________________________________________________________________________________________________
conv5_block16_0_relu (Activatio (None, 7, 7, 992)    0           conv5_block16_0_bn[0][0]         
__________________________________________________________________________________________________
conv5_block16_1_conv (Conv2D)   (None, 7, 7, 128)    126976      conv5_block16_0_relu[0][0]       
__________________________________________________________________________________________________
conv5_block16_1_bn (BatchNormal (None, 7, 7, 128)    512         conv5_block16_1_conv[0][0]       
__________________________________________________________________________________________________
conv5_block16_1_relu (Activatio (None, 7, 7, 128)    0           conv5_block16_1_bn[0][0]         
__________________________________________________________________________________________________
conv5_block16_2_conv (Conv2D)   (None, 7, 7, 32)     36864       conv5_block16_1_relu[0][0]       
__________________________________________________________________________________________________
conv5_block16_concat (Concatena (None, 7, 7, 1024)   0           conv5_block15_concat[0][0]       
                                                                 conv5_block16_2_conv[0][0]       
__________________________________________________________________________________________________
bn (BatchNormalization)         (None, 7, 7, 1024)   4096        conv5_block16_concat[0][0]       
__________________________________________________________________________________________________
relu (Activation)               (None, 7, 7, 1024)   0           bn[0][0]                         
__________________________________________________________________________________________________
global_average_pooling2d_4 (Glo (None, 1024)         0           relu[0][0]                       
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 1024)         0           global_average_pooling2d_4[0][0] 
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 100)          102500      dropout_4[0][0]                  
==================================================================================================
Total params: 7,140,004
Trainable params: 7,056,356
Non-trainable params: 83,648
__________________________________________________________________________________________________
```


**[Celda 32 - Código]**
```python
#Asignar capa a una variable
global_pooling2d = densenet.get_layer('conv3_block9_0_bn')

# Crear un nuevo modelo que tenga la capa convolucional como salida
new_model_densenet = Model(inputs=densenet.input, outputs=global_pooling2d.output)
```


**[Celda 33 - Código]**
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
# Convertir la lista de imágenes a un array numpy
array_imagenes = np.array(imagenesBD)

# Crear el generador de imágenes aumentadas
data_generator = ImageDataGenerator(
    #rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)

# Obtener el generador de imágenes aumentadas
batch_size = 16
image_size = (224, 224)
train_generator = data_generator.flow(
    x=array_imagenes,
    y=to_categorical([i for i in range(len(y_imagenesBD))], num_classes=100),  # En caso de tener etiquetas, proporcionarlas aquí
    batch_size=batch_size,
    shuffle=True
)
```


**[Celda 34 - Código]**
```python
# Congelar todas las capas
for layer in densenet.layers:
    layer.trainable = False

# Descongelar las últimas capas
for layer in densenet.layers[371:]:
    layer.trainable = True
```


**[Celda 35 - Código]**
```python
history = densenet.fit(train_generator,
                    batch_size=16,
                    epochs=5,
                    validation_data=train_generator)
```


*Salida:*
```text
Epoch 1/5
7/7 [==============================] - 4s 527ms/step - loss: 1.6081 - accuracy: 0.7200 - val_loss: 2.2961 - val_accuracy: 0.5500
Epoch 2/5
7/7 [==============================] - 4s 518ms/step - loss: 0.9878 - accuracy: 0.9000 - val_loss: 1.8093 - val_accuracy: 0.7400
Epoch 3/5
7/7 [==============================] - 4s 503ms/step - loss: 0.7059 - accuracy: 0.9300 - val_loss: 1.2006 - val_accuracy: 0.8700
Epoch 4/5
7/7 [==============================] - 4s 523ms/step - loss: 0.6585 - accuracy: 0.9500 - val_loss: 1.0157 - val_accuracy: 0.8800
Epoch 5/5
7/7 [==============================] - 3s 495ms/step - loss: 0.4934 - accuracy: 0.9600 - val_loss: 0.6471 - val_accuracy: 0.9500
```

## Inception



**[Celda 37 - Código]**
```python
from tensorflow.keras.applications.inception_v3 import InceptionV3

def build_inception(input_shape=(299, 299, 3), n_classes=10):
    input_layer = Input(shape=input_shape)
    inception = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_layer)
    x = GlobalAveragePooling2D()(inception.output)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(input_layer, x)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=3e-4), metrics=['accuracy'])
    return model

inception = build_inception(n_classes = 100)

```


**[Celda 38 - Código]**
```python
inception.summary()
```


*Salida:*
```text
Model: "functional_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 299, 299, 3) 0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 149, 149, 32) 864         input_1[0][0]                    
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 149, 149, 32) 96          conv2d[0][0]                     
__________________________________________________________________________________________________
activation (Activation)         (None, 149, 149, 32) 0           batch_normalization[0][0]        
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 147, 147, 32) 9216        activation[0][0]                 
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 147, 147, 32) 96          conv2d_1[0][0]                   
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 147, 147, 32) 0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 147, 147, 64) 18432       activation_1[0][0]               
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 147, 147, 64) 192         conv2d_2[0][0]                   
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 147, 147, 64) 0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 73, 73, 64)   0           activation_2[0][0]               
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 73, 73, 80)   5120        max_pooling2d[0][0]              
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 73, 73, 80)   240         conv2d_3[0][0]                   
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 73, 73, 80)   0           batch_normalization_3[0][0]      
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 71, 71, 192)  138240      activation_3[0][0]               
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 71, 71, 192)  576         conv2d_4[0][0]                   
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 71, 71, 192)  0           batch_normalization_4[0][0]      
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 35, 35, 192)  0           activation_4[0][0]               
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 35, 35, 64)   12288       max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 35, 35, 64)   192         conv2d_8[0][0]                   
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 35, 35, 64)   0           batch_normalization_8[0][0]      
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 35, 35, 48)   9216        max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 35, 35, 96)   55296       activation_8[0][0]               
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 35, 35, 48)   144         conv2d_6[0][0]                   
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 35, 35, 96)   288         conv2d_9[0][0]                   
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 35, 35, 48)   0           batch_normalization_6[0][0]      
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 35, 35, 96)   0           batch_normalization_9[0][0]      
__________________________________________________________________________________________________
average_pooling2d (AveragePooli (None, 35, 35, 192)  0           max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 35, 35, 64)   12288       max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 35, 35, 64)   76800       activation_6[0][0]               
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 35, 35, 96)   82944       activation_9[0][0]               
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 35, 35, 32)   6144        average_pooling2d[0][0]          
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 35, 35, 64)   192         conv2d_5[0][0]                   
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 35, 35, 64)   192         conv2d_7[0][0]                   
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 35, 35, 96)   288         conv2d_10[0][0]                  
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 35, 35, 32)   96          conv2d_11[0][0]                  
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 35, 35, 64)   0           batch_normalization_5[0][0]      
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 35, 35, 64)   0           batch_normalization_7[0][0]      
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 35, 35, 96)   0           batch_normalization_10[0][0]     
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 35, 35, 32)   0           batch_normalization_11[0][0]     
__________________________________________________________________________________________________
mixed0 (Concatenate)            (None, 35, 35, 256)  0           activation_5[0][0]               
                                                                 activation_7[0][0]               
                                                                 activation_10[0][0]              
                                                                 activation_11[0][0]              
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 35, 35, 64)   16384       mixed0[0][0]                     
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 35, 35, 64)   192         conv2d_15[0][0]                  
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 35, 35, 64)   0           batch_normalization_15[0][0]     
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 35, 35, 48)   12288       mixed0[0][0]                     
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 35, 35, 96)   55296       activation_15[0][0]              
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 35, 35, 48)   144         conv2d_13[0][0]                  
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 35, 35, 96)   288         conv2d_16[0][0]                  
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 35, 35, 48)   0           batch_normalization_13[0][0]     
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 35, 35, 96)   0           batch_normalization_16[0][0]     
__________________________________________________________________________________________________
average_pooling2d_1 (AveragePoo (None, 35, 35, 256)  0           mixed0[0][0]                     
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 35, 35, 64)   16384       mixed0[0][0]                     
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 35, 35, 64)   76800       activation_13[0][0]              
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 35, 35, 96)   82944       activation_16[0][0]              
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 35, 35, 64)   16384       average_pooling2d_1[0][0]        
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 35, 35, 64)   192         conv2d_12[0][0]                  
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 35, 35, 64)   192         conv2d_14[0][0]                  
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 35, 35, 96)   288         conv2d_17[0][0]                  
__________________________________________________________________________________________________
batch_normalization_18 (BatchNo (None, 35, 35, 64)   192         conv2d_18[0][0]                  
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 35, 35, 64)   0           batch_normalization_12[0][0]     
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 35, 35, 64)   0           batch_normalization_14[0][0]     
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 35, 35, 96)   0           batch_normalization_17[0][0]     
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 35, 35, 64)   0           batch_normalization_18[0][0]     
__________________________________________________________________________________________________
mixed1 (Concatenate)            (None, 35, 35, 288)  0           activation_12[0][0]              
                                                                 activation_14[0][0]              
                                                                 activation_17[0][0]              
                                                                 activation_18[0][0]              
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 35, 35, 64)   18432       mixed1[0][0]                     
__________________________________________________________________________________________________
batch_normalization_22 (BatchNo (None, 35, 35, 64)   192         conv2d_22[0][0]                  
__________________________________________________________________________________________________
activation_22 (Activation)      (None, 35, 35, 64)   0           batch_normalization_22[0][0]     
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 35, 35, 48)   13824       mixed1[0][0]                     
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, 35, 35, 96)   55296       activation_22[0][0]              
__________________________________________________________________________________________________
batch_normalization_20 (BatchNo (None, 35, 35, 48)   144         conv2d_20[0][0]                  
__________________________________________________________________________________________________
batch_normalization_23 (BatchNo (None, 35, 35, 96)   288         conv2d_23[0][0]                  
__________________________________________________________________________________________________
activation_20 (Activation)      (None, 35, 35, 48)   0           batch_normalization_20[0][0]     
__________________________________________________________________________________________________
activation_23 (Activation)      (None, 35, 35, 96)   0           batch_normalization_23[0][0]     
__________________________________________________________________________________________________
average_pooling2d_2 (AveragePoo (None, 35, 35, 288)  0           mixed1[0][0]                     
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 35, 35, 64)   18432       mixed1[0][0]                     
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 35, 35, 64)   76800       activation_20[0][0]              
__________________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, 35, 35, 96)   82944       activation_23[0][0]              
__________________________________________________________________________________________________
conv2d_25 (Conv2D)              (None, 35, 35, 64)   18432       average_pooling2d_2[0][0]        
__________________________________________________________________________________________________
batch_normalization_19 (BatchNo (None, 35, 35, 64)   192         conv2d_19[0][0]                  
__________________________________________________________________________________________________
batch_normalization_21 (BatchNo (None, 35, 35, 64)   192         conv2d_21[0][0]                  
__________________________________________________________________________________________________
batch_normalization_24 (BatchNo (None, 35, 35, 96)   288         conv2d_24[0][0]                  
__________________________________________________________________________________________________
batch_normalization_25 (BatchNo (None, 35, 35, 64)   192         conv2d_25[0][0]                  
__________________________________________________________________________________________________
activation_19 (Activation)      (None, 35, 35, 64)   0           batch_normalization_19[0][0]     
__________________________________________________________________________________________________
activation_21 (Activation)      (None, 35, 35, 64)   0           batch_normalization_21[0][0]     
__________________________________________________________________________________________________
activation_24 (Activation)      (None, 35, 35, 96)   0           batch_normalization_24[0][0]     
__________________________________________________________________________________________________
activation_25 (Activation)      (None, 35, 35, 64)   0           batch_normalization_25[0][0]     
__________________________________________________________________________________________________
mixed2 (Concatenate)            (None, 35, 35, 288)  0           activation_19[0][0]              
                                                                 activation_21[0][0]              
                                                                 activation_24[0][0]              
                                                                 activation_25[0][0]              
__________________________________________________________________________________________________
conv2d_27 (Conv2D)              (None, 35, 35, 64)   18432       mixed2[0][0]                     
__________________________________________________________________________________________________
batch_normalization_27 (BatchNo (None, 35, 35, 64)   192         conv2d_27[0][0]                  
__________________________________________________________________________________________________
activation_27 (Activation)      (None, 35, 35, 64)   0           batch_normalization_27[0][0]     
__________________________________________________________________________________________________
conv2d_28 (Conv2D)              (None, 35, 35, 96)   55296       activation_27[0][0]              
__________________________________________________________________________________________________
batch_normalization_28 (BatchNo (None, 35, 35, 96)   288         conv2d_28[0][0]                  
__________________________________________________________________________________________________
activation_28 (Activation)      (None, 35, 35, 96)   0           batch_normalization_28[0][0]     
__________________________________________________________________________________________________
conv2d_26 (Conv2D)              (None, 17, 17, 384)  995328      mixed2[0][0]                     
__________________________________________________________________________________________________
conv2d_29 (Conv2D)              (None, 17, 17, 96)   82944       activation_28[0][0]              
__________________________________________________________________________________________________
batch_normalization_26 (BatchNo (None, 17, 17, 384)  1152        conv2d_26[0][0]                  
__________________________________________________________________________________________________
batch_normalization_29 (BatchNo (None, 17, 17, 96)   288         conv2d_29[0][0]                  
__________________________________________________________________________________________________
activation_26 (Activation)      (None, 17, 17, 384)  0           batch_normalization_26[0][0]     
__________________________________________________________________________________________________
activation_29 (Activation)      (None, 17, 17, 96)   0           batch_normalization_29[0][0]     
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 17, 17, 288)  0           mixed2[0][0]                     
__________________________________________________________________________________________________
mixed3 (Concatenate)            (None, 17, 17, 768)  0           activation_26[0][0]              
                                                                 activation_29[0][0]              
                                                                 max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
conv2d_34 (Conv2D)              (None, 17, 17, 128)  98304       mixed3[0][0]                     
__________________________________________________________________________________________________
batch_normalization_34 (BatchNo (None, 17, 17, 128)  384         conv2d_34[0][0]                  
__________________________________________________________________________________________________
activation_34 (Activation)      (None, 17, 17, 128)  0           batch_normalization_34[0][0]     
__________________________________________________________________________________________________
conv2d_35 (Conv2D)              (None, 17, 17, 128)  114688      activation_34[0][0]              
__________________________________________________________________________________________________
batch_normalization_35 (BatchNo (None, 17, 17, 128)  384         conv2d_35[0][0]                  
__________________________________________________________________________________________________
activation_35 (Activation)      (None, 17, 17, 128)  0           batch_normalization_35[0][0]     
__________________________________________________________________________________________________
conv2d_31 (Conv2D)              (None, 17, 17, 128)  98304       mixed3[0][0]                     
__________________________________________________________________________________________________
conv2d_36 (Conv2D)              (None, 17, 17, 128)  114688      activation_35[0][0]              
__________________________________________________________________________________________________
batch_normalization_31 (BatchNo (None, 17, 17, 128)  384         conv2d_31[0][0]                  
__________________________________________________________________________________________________
batch_normalization_36 (BatchNo (None, 17, 17, 128)  384         conv2d_36[0][0]                  
__________________________________________________________________________________________________
activation_31 (Activation)      (None, 17, 17, 128)  0           batch_normalization_31[0][0]     
__________________________________________________________________________________________________
activation_36 (Activation)      (None, 17, 17, 128)  0           batch_normalization_36[0][0]     
__________________________________________________________________________________________________
conv2d_32 (Conv2D)              (None, 17, 17, 128)  114688      activation_31[0][0]              
__________________________________________________________________________________________________
conv2d_37 (Conv2D)              (None, 17, 17, 128)  114688      activation_36[0][0]              
__________________________________________________________________________________________________
batch_normalization_32 (BatchNo (None, 17, 17, 128)  384         conv2d_32[0][0]                  
__________________________________________________________________________________________________
batch_normalization_37 (BatchNo (None, 17, 17, 128)  384         conv2d_37[0][0]                  
__________________________________________________________________________________________________
activation_32 (Activation)      (None, 17, 17, 128)  0           batch_normalization_32[0][0]     
__________________________________________________________________________________________________
activation_37 (Activation)      (None, 17, 17, 128)  0           batch_normalization_37[0][0]     
__________________________________________________________________________________________________
average_pooling2d_3 (AveragePoo (None, 17, 17, 768)  0           mixed3[0][0]                     
__________________________________________________________________________________________________
conv2d_30 (Conv2D)              (None, 17, 17, 192)  147456      mixed3[0][0]                     
__________________________________________________________________________________________________
conv2d_33 (Conv2D)              (None, 17, 17, 192)  172032      activation_32[0][0]              
__________________________________________________________________________________________________
conv2d_38 (Conv2D)              (None, 17, 17, 192)  172032      activation_37[0][0]              
__________________________________________________________________________________________________
conv2d_39 (Conv2D)              (None, 17, 17, 192)  147456      average_pooling2d_3[0][0]        
__________________________________________________________________________________________________
batch_normalization_30 (BatchNo (None, 17, 17, 192)  576         conv2d_30[0][0]                  
__________________________________________________________________________________________________
batch_normalization_33 (BatchNo (None, 17, 17, 192)  576         conv2d_33[0][0]                  
__________________________________________________________________________________________________
batch_normalization_38 (BatchNo (None, 17, 17, 192)  576         conv2d_38[0][0]                  
__________________________________________________________________________________________________
batch_normalization_39 (BatchNo (None, 17, 17, 192)  576         conv2d_39[0][0]                  
__________________________________________________________________________________________________
activation_30 (Activation)      (None, 17, 17, 192)  0           batch_normalization_30[0][0]     
__________________________________________________________________________________________________
activation_33 (Activation)      (None, 17, 17, 192)  0           batch_normalization_33[0][0]     
__________________________________________________________________________________________________
activation_38 (Activation)      (None, 17, 17, 192)  0           batch_normalization_38[0][0]     
__________________________________________________________________________________________________
activation_39 (Activation)      (None, 17, 17, 192)  0           batch_normalization_39[0][0]     
__________________________________________________________________________________________________
mixed4 (Concatenate)            (None, 17, 17, 768)  0           activation_30[0][0]              
                                                                 activation_33[0][0]              
                                                                 activation_38[0][0]              
                                                                 activation_39[0][0]              
__________________________________________________________________________________________________
conv2d_44 (Conv2D)              (None, 17, 17, 160)  122880      mixed4[0][0]                     
__________________________________________________________________________________________________
batch_normalization_44 (BatchNo (None, 17, 17, 160)  480         conv2d_44[0][0]                  
__________________________________________________________________________________________________
activation_44 (Activation)      (None, 17, 17, 160)  0           batch_normalization_44[0][0]     
__________________________________________________________________________________________________
conv2d_45 (Conv2D)              (None, 17, 17, 160)  179200      activation_44[0][0]              
__________________________________________________________________________________________________
batch_normalization_45 (BatchNo (None, 17, 17, 160)  480         conv2d_45[0][0]                  
__________________________________________________________________________________________________
activation_45 (Activation)      (None, 17, 17, 160)  0           batch_normalization_45[0][0]     
__________________________________________________________________________________________________
conv2d_41 (Conv2D)              (None, 17, 17, 160)  122880      mixed4[0][0]                     
__________________________________________________________________________________________________
conv2d_46 (Conv2D)              (None, 17, 17, 160)  179200      activation_45[0][0]              
__________________________________________________________________________________________________
batch_normalization_41 (BatchNo (None, 17, 17, 160)  480         conv2d_41[0][0]                  
__________________________________________________________________________________________________
batch_normalization_46 (BatchNo (None, 17, 17, 160)  480         conv2d_46[0][0]                  
__________________________________________________________________________________________________
activation_41 (Activation)      (None, 17, 17, 160)  0           batch_normalization_41[0][0]     
__________________________________________________________________________________________________
activation_46 (Activation)      (None, 17, 17, 160)  0           batch_normalization_46[0][0]     
__________________________________________________________________________________________________
conv2d_42 (Conv2D)              (None, 17, 17, 160)  179200      activation_41[0][0]              
__________________________________________________________________________________________________
conv2d_47 (Conv2D)              (None, 17, 17, 160)  179200      activation_46[0][0]              
__________________________________________________________________________________________________
batch_normalization_42 (BatchNo (None, 17, 17, 160)  480         conv2d_42[0][0]                  
__________________________________________________________________________________________________
batch_normalization_47 (BatchNo (None, 17, 17, 160)  480         conv2d_47[0][0]                  
__________________________________________________________________________________________________
activation_42 (Activation)      (None, 17, 17, 160)  0           batch_normalization_42[0][0]     
__________________________________________________________________________________________________
activation_47 (Activation)      (None, 17, 17, 160)  0           batch_normalization_47[0][0]     
__________________________________________________________________________________________________
average_pooling2d_4 (AveragePoo (None, 17, 17, 768)  0           mixed4[0][0]                     
__________________________________________________________________________________________________
conv2d_40 (Conv2D)              (None, 17, 17, 192)  147456      mixed4[0][0]                     
__________________________________________________________________________________________________
conv2d_43 (Conv2D)              (None, 17, 17, 192)  215040      activation_42[0][0]              
__________________________________________________________________________________________________
conv2d_48 (Conv2D)              (None, 17, 17, 192)  215040      activation_47[0][0]              
__________________________________________________________________________________________________
conv2d_49 (Conv2D)              (None, 17, 17, 192)  147456      average_pooling2d_4[0][0]        
__________________________________________________________________________________________________
batch_normalization_40 (BatchNo (None, 17, 17, 192)  576         conv2d_40[0][0]                  
__________________________________________________________________________________________________
batch_normalization_43 (BatchNo (None, 17, 17, 192)  576         conv2d_43[0][0]                  
__________________________________________________________________________________________________
batch_normalization_48 (BatchNo (None, 17, 17, 192)  576         conv2d_48[0][0]                  
__________________________________________________________________________________________________
batch_normalization_49 (BatchNo (None, 17, 17, 192)  576         conv2d_49[0][0]                  
__________________________________________________________________________________________________
activation_40 (Activation)      (None, 17, 17, 192)  0           batch_normalization_40[0][0]     
__________________________________________________________________________________________________
activation_43 (Activation)      (None, 17, 17, 192)  0           batch_normalization_43[0][0]     
__________________________________________________________________________________________________
activation_48 (Activation)      (None, 17, 17, 192)  0           batch_normalization_48[0][0]     
__________________________________________________________________________________________________
activation_49 (Activation)      (None, 17, 17, 192)  0           batch_normalization_49[0][0]     
__________________________________________________________________________________________________
mixed5 (Concatenate)            (None, 17, 17, 768)  0           activation_40[0][0]              
                                                                 activation_43[0][0]              
                                                                 activation_48[0][0]              
                                                                 activation_49[0][0]              
__________________________________________________________________________________________________
conv2d_54 (Conv2D)              (None, 17, 17, 160)  122880      mixed5[0][0]                     
__________________________________________________________________________________________________
batch_normalization_54 (BatchNo (None, 17, 17, 160)  480         conv2d_54[0][0]                  
__________________________________________________________________________________________________
activation_54 (Activation)      (None, 17, 17, 160)  0           batch_normalization_54[0][0]     
__________________________________________________________________________________________________
conv2d_55 (Conv2D)              (None, 17, 17, 160)  179200      activation_54[0][0]              
__________________________________________________________________________________________________
batch_normalization_55 (BatchNo (None, 17, 17, 160)  480         conv2d_55[0][0]                  
__________________________________________________________________________________________________
activation_55 (Activation)      (None, 17, 17, 160)  0           batch_normalization_55[0][0]     
__________________________________________________________________________________________________
conv2d_51 (Conv2D)              (None, 17, 17, 160)  122880      mixed5[0][0]                     
__________________________________________________________________________________________________
conv2d_56 (Conv2D)              (None, 17, 17, 160)  179200      activation_55[0][0]              
__________________________________________________________________________________________________
batch_normalization_51 (BatchNo (None, 17, 17, 160)  480         conv2d_51[0][0]                  
__________________________________________________________________________________________________
batch_normalization_56 (BatchNo (None, 17, 17, 160)  480         conv2d_56[0][0]                  
__________________________________________________________________________________________________
activation_51 (Activation)      (None, 17, 17, 160)  0           batch_normalization_51[0][0]     
__________________________________________________________________________________________________
activation_56 (Activation)      (None, 17, 17, 160)  0           batch_normalization_56[0][0]     
__________________________________________________________________________________________________
conv2d_52 (Conv2D)              (None, 17, 17, 160)  179200      activation_51[0][0]              
__________________________________________________________________________________________________
conv2d_57 (Conv2D)              (None, 17, 17, 160)  179200      activation_56[0][0]              
__________________________________________________________________________________________________
batch_normalization_52 (BatchNo (None, 17, 17, 160)  480         conv2d_52[0][0]                  
__________________________________________________________________________________________________
batch_normalization_57 (BatchNo (None, 17, 17, 160)  480         conv2d_57[0][0]                  
__________________________________________________________________________________________________
activation_52 (Activation)      (None, 17, 17, 160)  0           batch_normalization_52[0][0]     
__________________________________________________________________________________________________
activation_57 (Activation)      (None, 17, 17, 160)  0           batch_normalization_57[0][0]     
__________________________________________________________________________________________________
average_pooling2d_5 (AveragePoo (None, 17, 17, 768)  0           mixed5[0][0]                     
__________________________________________________________________________________________________
conv2d_50 (Conv2D)              (None, 17, 17, 192)  147456      mixed5[0][0]                     
__________________________________________________________________________________________________
conv2d_53 (Conv2D)              (None, 17, 17, 192)  215040      activation_52[0][0]              
__________________________________________________________________________________________________
conv2d_58 (Conv2D)              (None, 17, 17, 192)  215040      activation_57[0][0]              
__________________________________________________________________________________________________
conv2d_59 (Conv2D)              (None, 17, 17, 192)  147456      average_pooling2d_5[0][0]        
__________________________________________________________________________________________________
batch_normalization_50 (BatchNo (None, 17, 17, 192)  576         conv2d_50[0][0]                  
__________________________________________________________________________________________________
batch_normalization_53 (BatchNo (None, 17, 17, 192)  576         conv2d_53[0][0]                  
__________________________________________________________________________________________________
batch_normalization_58 (BatchNo (None, 17, 17, 192)  576         conv2d_58[0][0]                  
__________________________________________________________________________________________________
batch_normalization_59 (BatchNo (None, 17, 17, 192)  576         conv2d_59[0][0]                  
__________________________________________________________________________________________________
activation_50 (Activation)      (None, 17, 17, 192)  0           batch_normalization_50[0][0]     
__________________________________________________________________________________________________
activation_53 (Activation)      (None, 17, 17, 192)  0           batch_normalization_53[0][0]     
__________________________________________________________________________________________________
activation_58 (Activation)      (None, 17, 17, 192)  0           batch_normalization_58[0][0]     
__________________________________________________________________________________________________
activation_59 (Activation)      (None, 17, 17, 192)  0           batch_normalization_59[0][0]     
__________________________________________________________________________________________________
mixed6 (Concatenate)            (None, 17, 17, 768)  0           activation_50[0][0]              
                                                                 activation_53[0][0]              
                                                                 activation_58[0][0]              
                                                                 activation_59[0][0]              
__________________________________________________________________________________________________
conv2d_64 (Conv2D)              (None, 17, 17, 192)  147456      mixed6[0][0]                     
__________________________________________________________________________________________________
batch_normalization_64 (BatchNo (None, 17, 17, 192)  576         conv2d_64[0][0]                  
__________________________________________________________________________________________________
activation_64 (Activation)      (None, 17, 17, 192)  0           batch_normalization_64[0][0]     
__________________________________________________________________________________________________
conv2d_65 (Conv2D)              (None, 17, 17, 192)  258048      activation_64[0][0]              
__________________________________________________________________________________________________
batch_normalization_65 (BatchNo (None, 17, 17, 192)  576         conv2d_65[0][0]                  
__________________________________________________________________________________________________
activation_65 (Activation)      (None, 17, 17, 192)  0           batch_normalization_65[0][0]     
__________________________________________________________________________________________________
conv2d_61 (Conv2D)              (None, 17, 17, 192)  147456      mixed6[0][0]                     
__________________________________________________________________________________________________
conv2d_66 (Conv2D)              (None, 17, 17, 192)  258048      activation_65[0][0]              
__________________________________________________________________________________________________
batch_normalization_61 (BatchNo (None, 17, 17, 192)  576         conv2d_61[0][0]                  
__________________________________________________________________________________________________
batch_normalization_66 (BatchNo (None, 17, 17, 192)  576         conv2d_66[0][0]                  
__________________________________________________________________________________________________
activation_61 (Activation)      (None, 17, 17, 192)  0           batch_normalization_61[0][0]     
__________________________________________________________________________________________________
activation_66 (Activation)      (None, 17, 17, 192)  0           batch_normalization_66[0][0]     
__________________________________________________________________________________________________
conv2d_62 (Conv2D)              (None, 17, 17, 192)  258048      activation_61[0][0]              
__________________________________________________________________________________________________
conv2d_67 (Conv2D)              (None, 17, 17, 192)  258048      activation_66[0][0]              
__________________________________________________________________________________________________
batch_normalization_62 (BatchNo (None, 17, 17, 192)  576         conv2d_62[0][0]                  
__________________________________________________________________________________________________
batch_normalization_67 (BatchNo (None, 17, 17, 192)  576         conv2d_67[0][0]                  
__________________________________________________________________________________________________
activation_62 (Activation)      (None, 17, 17, 192)  0           batch_normalization_62[0][0]     
__________________________________________________________________________________________________
activation_67 (Activation)      (None, 17, 17, 192)  0           batch_normalization_67[0][0]     
__________________________________________________________________________________________________
average_pooling2d_6 (AveragePoo (None, 17, 17, 768)  0           mixed6[0][0]                     
__________________________________________________________________________________________________
conv2d_60 (Conv2D)              (None, 17, 17, 192)  147456      mixed6[0][0]                     
__________________________________________________________________________________________________
conv2d_63 (Conv2D)              (None, 17, 17, 192)  258048      activation_62[0][0]              
__________________________________________________________________________________________________
conv2d_68 (Conv2D)              (None, 17, 17, 192)  258048      activation_67[0][0]              
__________________________________________________________________________________________________
conv2d_69 (Conv2D)              (None, 17, 17, 192)  147456      average_pooling2d_6[0][0]        
__________________________________________________________________________________________________
batch_normalization_60 (BatchNo (None, 17, 17, 192)  576         conv2d_60[0][0]                  
__________________________________________________________________________________________________
batch_normalization_63 (BatchNo (None, 17, 17, 192)  576         conv2d_63[0][0]                  
__________________________________________________________________________________________________
batch_normalization_68 (BatchNo (None, 17, 17, 192)  576         conv2d_68[0][0]                  
__________________________________________________________________________________________________
batch_normalization_69 (BatchNo (None, 17, 17, 192)  576         conv2d_69[0][0]                  
__________________________________________________________________________________________________
activation_60 (Activation)      (None, 17, 17, 192)  0           batch_normalization_60[0][0]     
__________________________________________________________________________________________________
activation_63 (Activation)      (None, 17, 17, 192)  0           batch_normalization_63[0][0]     
__________________________________________________________________________________________________
activation_68 (Activation)      (None, 17, 17, 192)  0           batch_normalization_68[0][0]     
__________________________________________________________________________________________________
activation_69 (Activation)      (None, 17, 17, 192)  0           batch_normalization_69[0][0]     
__________________________________________________________________________________________________
mixed7 (Concatenate)            (None, 17, 17, 768)  0           activation_60[0][0]              
                                                                 activation_63[0][0]              
                                                                 activation_68[0][0]              
                                                                 activation_69[0][0]              
__________________________________________________________________________________________________
conv2d_72 (Conv2D)              (None, 17, 17, 192)  147456      mixed7[0][0]                     
__________________________________________________________________________________________________
batch_normalization_72 (BatchNo (None, 17, 17, 192)  576         conv2d_72[0][0]                  
__________________________________________________________________________________________________
activation_72 (Activation)      (None, 17, 17, 192)  0           batch_normalization_72[0][0]     
__________________________________________________________________________________________________
conv2d_73 (Conv2D)              (None, 17, 17, 192)  258048      activation_72[0][0]              
__________________________________________________________________________________________________
batch_normalization_73 (BatchNo (None, 17, 17, 192)  576         conv2d_73[0][0]                  
__________________________________________________________________________________________________
activation_73 (Activation)      (None, 17, 17, 192)  0           batch_normalization_73[0][0]     
__________________________________________________________________________________________________
conv2d_70 (Conv2D)              (None, 17, 17, 192)  147456      mixed7[0][0]                     
__________________________________________________________________________________________________
conv2d_74 (Conv2D)              (None, 17, 17, 192)  258048      activation_73[0][0]              
__________________________________________________________________________________________________
batch_normalization_70 (BatchNo (None, 17, 17, 192)  576         conv2d_70[0][0]                  
__________________________________________________________________________________________________
batch_normalization_74 (BatchNo (None, 17, 17, 192)  576         conv2d_74[0][0]                  
__________________________________________________________________________________________________
activation_70 (Activation)      (None, 17, 17, 192)  0           batch_normalization_70[0][0]     
__________________________________________________________________________________________________
activation_74 (Activation)      (None, 17, 17, 192)  0           batch_normalization_74[0][0]     
__________________________________________________________________________________________________
conv2d_71 (Conv2D)              (None, 8, 8, 320)    552960      activation_70[0][0]              
__________________________________________________________________________________________________
conv2d_75 (Conv2D)              (None, 8, 8, 192)    331776      activation_74[0][0]              
__________________________________________________________________________________________________
batch_normalization_71 (BatchNo (None, 8, 8, 320)    960         conv2d_71[0][0]                  
__________________________________________________________________________________________________
batch_normalization_75 (BatchNo (None, 8, 8, 192)    576         conv2d_75[0][0]                  
__________________________________________________________________________________________________
activation_71 (Activation)      (None, 8, 8, 320)    0           batch_normalization_71[0][0]     
__________________________________________________________________________________________________
activation_75 (Activation)      (None, 8, 8, 192)    0           batch_normalization_75[0][0]     
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 8, 8, 768)    0           mixed7[0][0]                     
__________________________________________________________________________________________________
mixed8 (Concatenate)            (None, 8, 8, 1280)   0           activation_71[0][0]              
                                                                 activation_75[0][0]              
                                                                 max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
conv2d_80 (Conv2D)              (None, 8, 8, 448)    573440      mixed8[0][0]                     
__________________________________________________________________________________________________
batch_normalization_80 (BatchNo (None, 8, 8, 448)    1344        conv2d_80[0][0]                  
__________________________________________________________________________________________________
activation_80 (Activation)      (None, 8, 8, 448)    0           batch_normalization_80[0][0]     
__________________________________________________________________________________________________
conv2d_77 (Conv2D)              (None, 8, 8, 384)    491520      mixed8[0][0]                     
__________________________________________________________________________________________________
conv2d_81 (Conv2D)              (None, 8, 8, 384)    1548288     activation_80[0][0]              
__________________________________________________________________________________________________
batch_normalization_77 (BatchNo (None, 8, 8, 384)    1152        conv2d_77[0][0]                  
__________________________________________________________________________________________________
batch_normalization_81 (BatchNo (None, 8, 8, 384)    1152        conv2d_81[0][0]                  
__________________________________________________________________________________________________
activation_77 (Activation)      (None, 8, 8, 384)    0           batch_normalization_77[0][0]     
__________________________________________________________________________________________________
activation_81 (Activation)      (None, 8, 8, 384)    0           batch_normalization_81[0][0]     
__________________________________________________________________________________________________
conv2d_78 (Conv2D)              (None, 8, 8, 384)    442368      activation_77[0][0]              
__________________________________________________________________________________________________
conv2d_79 (Conv2D)              (None, 8, 8, 384)    442368      activation_77[0][0]              
__________________________________________________________________________________________________
conv2d_82 (Conv2D)              (None, 8, 8, 384)    442368      activation_81[0][0]              
__________________________________________________________________________________________________
conv2d_83 (Conv2D)              (None, 8, 8, 384)    442368      activation_81[0][0]              
__________________________________________________________________________________________________
average_pooling2d_7 (AveragePoo (None, 8, 8, 1280)   0           mixed8[0][0]                     
__________________________________________________________________________________________________
conv2d_76 (Conv2D)              (None, 8, 8, 320)    409600      mixed8[0][0]                     
__________________________________________________________________________________________________
batch_normalization_78 (BatchNo (None, 8, 8, 384)    1152        conv2d_78[0][0]                  
__________________________________________________________________________________________________
batch_normalization_79 (BatchNo (None, 8, 8, 384)    1152        conv2d_79[0][0]                  
__________________________________________________________________________________________________
batch_normalization_82 (BatchNo (None, 8, 8, 384)    1152        conv2d_82[0][0]                  
__________________________________________________________________________________________________
batch_normalization_83 (BatchNo (None, 8, 8, 384)    1152        conv2d_83[0][0]                  
__________________________________________________________________________________________________
conv2d_84 (Conv2D)              (None, 8, 8, 192)    245760      average_pooling2d_7[0][0]        
__________________________________________________________________________________________________
batch_normalization_76 (BatchNo (None, 8, 8, 320)    960         conv2d_76[0][0]                  
__________________________________________________________________________________________________
activation_78 (Activation)      (None, 8, 8, 384)    0           batch_normalization_78[0][0]     
__________________________________________________________________________________________________
activation_79 (Activation)      (None, 8, 8, 384)    0           batch_normalization_79[0][0]     
__________________________________________________________________________________________________
activation_82 (Activation)      (None, 8, 8, 384)    0           batch_normalization_82[0][0]     
__________________________________________________________________________________________________
activation_83 (Activation)      (None, 8, 8, 384)    0           batch_normalization_83[0][0]     
__________________________________________________________________________________________________
batch_normalization_84 (BatchNo (None, 8, 8, 192)    576         conv2d_84[0][0]                  
__________________________________________________________________________________________________
activation_76 (Activation)      (None, 8, 8, 320)    0           batch_normalization_76[0][0]     
__________________________________________________________________________________________________
mixed9_0 (Concatenate)          (None, 8, 8, 768)    0           activation_78[0][0]              
                                                                 activation_79[0][0]              
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 8, 8, 768)    0           activation_82[0][0]              
                                                                 activation_83[0][0]              
__________________________________________________________________________________________________
activation_84 (Activation)      (None, 8, 8, 192)    0           batch_normalization_84[0][0]     
__________________________________________________________________________________________________
mixed9 (Concatenate)            (None, 8, 8, 2048)   0           activation_76[0][0]              
                                                                 mixed9_0[0][0]                   
                                                                 concatenate[0][0]                
                                                                 activation_84[0][0]              
__________________________________________________________________________________________________
conv2d_89 (Conv2D)              (None, 8, 8, 448)    917504      mixed9[0][0]                     
__________________________________________________________________________________________________
batch_normalization_89 (BatchNo (None, 8, 8, 448)    1344        conv2d_89[0][0]                  
__________________________________________________________________________________________________
activation_89 (Activation)      (None, 8, 8, 448)    0           batch_normalization_89[0][0]     
__________________________________________________________________________________________________
conv2d_86 (Conv2D)              (None, 8, 8, 384)    786432      mixed9[0][0]                     
__________________________________________________________________________________________________
conv2d_90 (Conv2D)              (None, 8, 8, 384)    1548288     activation_89[0][0]              
__________________________________________________________________________________________________
batch_normalization_86 (BatchNo (None, 8, 8, 384)    1152        conv2d_86[0][0]                  
__________________________________________________________________________________________________
batch_normalization_90 (BatchNo (None, 8, 8, 384)    1152        conv2d_90[0][0]                  
__________________________________________________________________________________________________
activation_86 (Activation)      (None, 8, 8, 384)    0           batch_normalization_86[0][0]     
__________________________________________________________________________________________________
activation_90 (Activation)      (None, 8, 8, 384)    0           batch_normalization_90[0][0]     
__________________________________________________________________________________________________
conv2d_87 (Conv2D)              (None, 8, 8, 384)    442368      activation_86[0][0]              
__________________________________________________________________________________________________
conv2d_88 (Conv2D)              (None, 8, 8, 384)    442368      activation_86[0][0]              
__________________________________________________________________________________________________
conv2d_91 (Conv2D)              (None, 8, 8, 384)    442368      activation_90[0][0]              
__________________________________________________________________________________________________
conv2d_92 (Conv2D)              (None, 8, 8, 384)    442368      activation_90[0][0]              
__________________________________________________________________________________________________
average_pooling2d_8 (AveragePoo (None, 8, 8, 2048)   0           mixed9[0][0]                     
__________________________________________________________________________________________________
conv2d_85 (Conv2D)              (None, 8, 8, 320)    655360      mixed9[0][0]                     
__________________________________________________________________________________________________
batch_normalization_87 (BatchNo (None, 8, 8, 384)    1152        conv2d_87[0][0]                  
__________________________________________________________________________________________________
batch_normalization_88 (BatchNo (None, 8, 8, 384)    1152        conv2d_88[0][0]                  
__________________________________________________________________________________________________
batch_normalization_91 (BatchNo (None, 8, 8, 384)    1152        conv2d_91[0][0]                  
__________________________________________________________________________________________________
batch_normalization_92 (BatchNo (None, 8, 8, 384)    1152        conv2d_92[0][0]                  
__________________________________________________________________________________________________
conv2d_93 (Conv2D)              (None, 8, 8, 192)    393216      average_pooling2d_8[0][0]        
__________________________________________________________________________________________________
batch_normalization_85 (BatchNo (None, 8, 8, 320)    960         conv2d_85[0][0]                  
__________________________________________________________________________________________________
activation_87 (Activation)      (None, 8, 8, 384)    0           batch_normalization_87[0][0]     
__________________________________________________________________________________________________
activation_88 (Activation)      (None, 8, 8, 384)    0           batch_normalization_88[0][0]     
__________________________________________________________________________________________________
activation_91 (Activation)      (None, 8, 8, 384)    0           batch_normalization_91[0][0]     
__________________________________________________________________________________________________
activation_92 (Activation)      (None, 8, 8, 384)    0           batch_normalization_92[0][0]     
__________________________________________________________________________________________________
batch_normalization_93 (BatchNo (None, 8, 8, 192)    576         conv2d_93[0][0]                  
__________________________________________________________________________________________________
activation_85 (Activation)      (None, 8, 8, 320)    0           batch_normalization_85[0][0]     
__________________________________________________________________________________________________
mixed9_1 (Concatenate)          (None, 8, 8, 768)    0           activation_87[0][0]              
                                                                 activation_88[0][0]              
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 8, 8, 768)    0           activation_91[0][0]              
                                                                 activation_92[0][0]              
__________________________________________________________________________________________________
activation_93 (Activation)      (None, 8, 8, 192)    0           batch_normalization_93[0][0]     
__________________________________________________________________________________________________
mixed10 (Concatenate)           (None, 8, 8, 2048)   0           activation_85[0][0]              
                                                                 mixed9_1[0][0]                   
                                                                 concatenate_1[0][0]              
                                                                 activation_93[0][0]              
__________________________________________________________________________________________________
global_average_pooling2d (Globa (None, 2048)         0           mixed10[0][0]                    
__________________________________________________________________________________________________
dropout (Dropout)               (None, 2048)         0           global_average_pooling2d[0][0]   
__________________________________________________________________________________________________
dense (Dense)                   (None, 100)          204900      dropout[0][0]                    
==================================================================================================
Total params: 22,007,684
Trainable params: 21,973,252
Non-trainable params: 34,432
__________________________________________________________________________________________________
```


**[Celda 39 - Código]**
```python
# Asignar la capa de pooling global a una variable
salida = inception.get_layer('batch_normalization_40')

# Crear un nuevo modelo que tenga la capa de pooling global como salida
new_model_inception = Model(inputs=inception.input, outputs=salida.output)
```

## Resnet



**[Celda 41 - Código]**
```python
from tensorflow.keras.applications.resnet import ResNet50

def build_resnet(input_shape=(224, 224, 3), n_classes=10):
    input_layer = Input(shape=input_shape)
    resnet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=input_layer)
    x = GlobalAveragePooling2D()(resnet50.output)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(input_layer, x)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=3e-4), metrics=['accuracy'])
    return model

resnet = build_resnet(n_classes=100)
```


*Salida:*
```text
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
94773248/94765736 [==============================] - 7s 0us/step
```


**[Celda 42 - Código]**
```python
resnet.summary()
```


*Salida:*
```text
Model: "functional_37"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_4 (InputLayer)            [(None, 224, 224, 3) 0                                            
__________________________________________________________________________________________________
conv1_pad (ZeroPadding2D)       (None, 230, 230, 3)  0           input_4[0][0]                    
__________________________________________________________________________________________________
conv1_conv (Conv2D)             (None, 112, 112, 64) 9472        conv1_pad[0][0]                  
__________________________________________________________________________________________________
conv1_bn (BatchNormalization)   (None, 112, 112, 64) 256         conv1_conv[0][0]                 
__________________________________________________________________________________________________
conv1_relu (Activation)         (None, 112, 112, 64) 0           conv1_bn[0][0]                   
__________________________________________________________________________________________________
pool1_pad (ZeroPadding2D)       (None, 114, 114, 64) 0           conv1_relu[0][0]                 
__________________________________________________________________________________________________
pool1_pool (MaxPooling2D)       (None, 56, 56, 64)   0           pool1_pad[0][0]                  
__________________________________________________________________________________________________
conv2_block1_1_conv (Conv2D)    (None, 56, 56, 64)   4160        pool1_pool[0][0]                 
__________________________________________________________________________________________________
conv2_block1_1_bn (BatchNormali (None, 56, 56, 64)   256         conv2_block1_1_conv[0][0]        
__________________________________________________________________________________________________
conv2_block1_1_relu (Activation (None, 56, 56, 64)   0           conv2_block1_1_bn[0][0]          
__________________________________________________________________________________________________
conv2_block1_2_conv (Conv2D)    (None, 56, 56, 64)   36928       conv2_block1_1_relu[0][0]        
__________________________________________________________________________________________________
conv2_block1_2_bn (BatchNormali (None, 56, 56, 64)   256         conv2_block1_2_conv[0][0]        
__________________________________________________________________________________________________
conv2_block1_2_relu (Activation (None, 56, 56, 64)   0           conv2_block1_2_bn[0][0]          
__________________________________________________________________________________________________
conv2_block1_0_conv (Conv2D)    (None, 56, 56, 256)  16640       pool1_pool[0][0]                 
__________________________________________________________________________________________________
conv2_block1_3_conv (Conv2D)    (None, 56, 56, 256)  16640       conv2_block1_2_relu[0][0]        
__________________________________________________________________________________________________
conv2_block1_0_bn (BatchNormali (None, 56, 56, 256)  1024        conv2_block1_0_conv[0][0]        
__________________________________________________________________________________________________
conv2_block1_3_bn (BatchNormali (None, 56, 56, 256)  1024        conv2_block1_3_conv[0][0]        
__________________________________________________________________________________________________
conv2_block1_add (Add)          (None, 56, 56, 256)  0           conv2_block1_0_bn[0][0]          
                                                                 conv2_block1_3_bn[0][0]          
__________________________________________________________________________________________________
conv2_block1_out (Activation)   (None, 56, 56, 256)  0           conv2_block1_add[0][0]           
__________________________________________________________________________________________________
conv2_block2_1_conv (Conv2D)    (None, 56, 56, 64)   16448       conv2_block1_out[0][0]           
__________________________________________________________________________________________________
conv2_block2_1_bn (BatchNormali (None, 56, 56, 64)   256         conv2_block2_1_conv[0][0]        
__________________________________________________________________________________________________
conv2_block2_1_relu (Activation (None, 56, 56, 64)   0           conv2_block2_1_bn[0][0]          
__________________________________________________________________________________________________
conv2_block2_2_conv (Conv2D)    (None, 56, 56, 64)   36928       conv2_block2_1_relu[0][0]        
__________________________________________________________________________________________________
conv2_block2_2_bn (BatchNormali (None, 56, 56, 64)   256         conv2_block2_2_conv[0][0]        
__________________________________________________________________________________________________
conv2_block2_2_relu (Activation (None, 56, 56, 64)   0           conv2_block2_2_bn[0][0]          
__________________________________________________________________________________________________
conv2_block2_3_conv (Conv2D)    (None, 56, 56, 256)  16640       conv2_block2_2_relu[0][0]        
__________________________________________________________________________________________________
conv2_block2_3_bn (BatchNormali (None, 56, 56, 256)  1024        conv2_block2_3_conv[0][0]        
__________________________________________________________________________________________________
conv2_block2_add (Add)          (None, 56, 56, 256)  0           conv2_block1_out[0][0]           
                                                                 conv2_block2_3_bn[0][0]          
__________________________________________________________________________________________________
conv2_block2_out (Activation)   (None, 56, 56, 256)  0           conv2_block2_add[0][0]           
__________________________________________________________________________________________________
conv2_block3_1_conv (Conv2D)    (None, 56, 56, 64)   16448       conv2_block2_out[0][0]           
__________________________________________________________________________________________________
conv2_block3_1_bn (BatchNormali (None, 56, 56, 64)   256         conv2_block3_1_conv[0][0]        
__________________________________________________________________________________________________
conv2_block3_1_relu (Activation (None, 56, 56, 64)   0           conv2_block3_1_bn[0][0]          
__________________________________________________________________________________________________
conv2_block3_2_conv (Conv2D)    (None, 56, 56, 64)   36928       conv2_block3_1_relu[0][0]        
__________________________________________________________________________________________________
conv2_block3_2_bn (BatchNormali (None, 56, 56, 64)   256         conv2_block3_2_conv[0][0]        
__________________________________________________________________________________________________
conv2_block3_2_relu (Activation (None, 56, 56, 64)   0           conv2_block3_2_bn[0][0]          
__________________________________________________________________________________________________
conv2_block3_3_conv (Conv2D)    (None, 56, 56, 256)  16640       conv2_block3_2_relu[0][0]        
__________________________________________________________________________________________________
conv2_block3_3_bn (BatchNormali (None, 56, 56, 256)  1024        conv2_block3_3_conv[0][0]        
__________________________________________________________________________________________________
conv2_block3_add (Add)          (None, 56, 56, 256)  0           conv2_block2_out[0][0]           
                                                                 conv2_block3_3_bn[0][0]          
__________________________________________________________________________________________________
conv2_block3_out (Activation)   (None, 56, 56, 256)  0           conv2_block3_add[0][0]           
__________________________________________________________________________________________________
conv3_block1_1_conv (Conv2D)    (None, 28, 28, 128)  32896       conv2_block3_out[0][0]           
__________________________________________________________________________________________________
conv3_block1_1_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block1_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block1_1_relu (Activation (None, 28, 28, 128)  0           conv3_block1_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block1_2_conv (Conv2D)    (None, 28, 28, 128)  147584      conv3_block1_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block1_2_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block1_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block1_2_relu (Activation (None, 28, 28, 128)  0           conv3_block1_2_bn[0][0]          
__________________________________________________________________________________________________
conv3_block1_0_conv (Conv2D)    (None, 28, 28, 512)  131584      conv2_block3_out[0][0]           
__________________________________________________________________________________________________
conv3_block1_3_conv (Conv2D)    (None, 28, 28, 512)  66048       conv3_block1_2_relu[0][0]        
__________________________________________________________________________________________________
conv3_block1_0_bn (BatchNormali (None, 28, 28, 512)  2048        conv3_block1_0_conv[0][0]        
__________________________________________________________________________________________________
conv3_block1_3_bn (BatchNormali (None, 28, 28, 512)  2048        conv3_block1_3_conv[0][0]        
__________________________________________________________________________________________________
conv3_block1_add (Add)          (None, 28, 28, 512)  0           conv3_block1_0_bn[0][0]          
                                                                 conv3_block1_3_bn[0][0]          
__________________________________________________________________________________________________
conv3_block1_out (Activation)   (None, 28, 28, 512)  0           conv3_block1_add[0][0]           
__________________________________________________________________________________________________
conv3_block2_1_conv (Conv2D)    (None, 28, 28, 128)  65664       conv3_block1_out[0][0]           
__________________________________________________________________________________________________
conv3_block2_1_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block2_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block2_1_relu (Activation (None, 28, 28, 128)  0           conv3_block2_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block2_2_conv (Conv2D)    (None, 28, 28, 128)  147584      conv3_block2_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block2_2_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block2_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block2_2_relu (Activation (None, 28, 28, 128)  0           conv3_block2_2_bn[0][0]          
__________________________________________________________________________________________________
conv3_block2_3_conv (Conv2D)    (None, 28, 28, 512)  66048       conv3_block2_2_relu[0][0]        
__________________________________________________________________________________________________
conv3_block2_3_bn (BatchNormali (None, 28, 28, 512)  2048        conv3_block2_3_conv[0][0]        
__________________________________________________________________________________________________
conv3_block2_add (Add)          (None, 28, 28, 512)  0           conv3_block1_out[0][0]           
                                                                 conv3_block2_3_bn[0][0]          
__________________________________________________________________________________________________
conv3_block2_out (Activation)   (None, 28, 28, 512)  0           conv3_block2_add[0][0]           
__________________________________________________________________________________________________
conv3_block3_1_conv (Conv2D)    (None, 28, 28, 128)  65664       conv3_block2_out[0][0]           
__________________________________________________________________________________________________
conv3_block3_1_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block3_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block3_1_relu (Activation (None, 28, 28, 128)  0           conv3_block3_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block3_2_conv (Conv2D)    (None, 28, 28, 128)  147584      conv3_block3_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block3_2_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block3_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block3_2_relu (Activation (None, 28, 28, 128)  0           conv3_block3_2_bn[0][0]          
__________________________________________________________________________________________________
conv3_block3_3_conv (Conv2D)    (None, 28, 28, 512)  66048       conv3_block3_2_relu[0][0]        
__________________________________________________________________________________________________
conv3_block3_3_bn (BatchNormali (None, 28, 28, 512)  2048        conv3_block3_3_conv[0][0]        
__________________________________________________________________________________________________
conv3_block3_add (Add)          (None, 28, 28, 512)  0           conv3_block2_out[0][0]           
                                                                 conv3_block3_3_bn[0][0]          
__________________________________________________________________________________________________
conv3_block3_out (Activation)   (None, 28, 28, 512)  0           conv3_block3_add[0][0]           
__________________________________________________________________________________________________
conv3_block4_1_conv (Conv2D)    (None, 28, 28, 128)  65664       conv3_block3_out[0][0]           
__________________________________________________________________________________________________
conv3_block4_1_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block4_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block4_1_relu (Activation (None, 28, 28, 128)  0           conv3_block4_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block4_2_conv (Conv2D)    (None, 28, 28, 128)  147584      conv3_block4_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block4_2_bn (BatchNormali (None, 28, 28, 128)  512         conv3_block4_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block4_2_relu (Activation (None, 28, 28, 128)  0           conv3_block4_2_bn[0][0]          
__________________________________________________________________________________________________
conv3_block4_3_conv (Conv2D)    (None, 28, 28, 512)  66048       conv3_block4_2_relu[0][0]        
__________________________________________________________________________________________________
conv3_block4_3_bn (BatchNormali (None, 28, 28, 512)  2048        conv3_block4_3_conv[0][0]        
__________________________________________________________________________________________________
conv3_block4_add (Add)          (None, 28, 28, 512)  0           conv3_block3_out[0][0]           
                                                                 conv3_block4_3_bn[0][0]          
__________________________________________________________________________________________________
conv3_block4_out (Activation)   (None, 28, 28, 512)  0           conv3_block4_add[0][0]           
__________________________________________________________________________________________________
conv4_block1_1_conv (Conv2D)    (None, 14, 14, 256)  131328      conv3_block4_out[0][0]           
__________________________________________________________________________________________________
conv4_block1_1_bn (BatchNormali (None, 14, 14, 256)  1024        conv4_block1_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block1_1_relu (Activation (None, 14, 14, 256)  0           conv4_block1_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block1_2_conv (Conv2D)    (None, 14, 14, 256)  590080      conv4_block1_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block1_2_bn (BatchNormali (None, 14, 14, 256)  1024        conv4_block1_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block1_2_relu (Activation (None, 14, 14, 256)  0           conv4_block1_2_bn[0][0]          
__________________________________________________________________________________________________
conv4_block1_0_conv (Conv2D)    (None, 14, 14, 1024) 525312      conv3_block4_out[0][0]           
__________________________________________________________________________________________________
conv4_block1_3_conv (Conv2D)    (None, 14, 14, 1024) 263168      conv4_block1_2_relu[0][0]        
__________________________________________________________________________________________________
conv4_block1_0_bn (BatchNormali (None, 14, 14, 1024) 4096        conv4_block1_0_conv[0][0]        
__________________________________________________________________________________________________
conv4_block1_3_bn (BatchNormali (None, 14, 14, 1024) 4096        conv4_block1_3_conv[0][0]        
__________________________________________________________________________________________________
conv4_block1_add (Add)          (None, 14, 14, 1024) 0           conv4_block1_0_bn[0][0]          
                                                                 conv4_block1_3_bn[0][0]          
__________________________________________________________________________________________________
conv4_block1_out (Activation)   (None, 14, 14, 1024) 0           conv4_block1_add[0][0]           
__________________________________________________________________________________________________
conv4_block2_1_conv (Conv2D)    (None, 14, 14, 256)  262400      conv4_block1_out[0][0]           
__________________________________________________________________________________________________
conv4_block2_1_bn (BatchNormali (None, 14, 14, 256)  1024        conv4_block2_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block2_1_relu (Activation (None, 14, 14, 256)  0           conv4_block2_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block2_2_conv (Conv2D)    (None, 14, 14, 256)  590080      conv4_block2_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block2_2_bn (BatchNormali (None, 14, 14, 256)  1024        conv4_block2_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block2_2_relu (Activation (None, 14, 14, 256)  0           conv4_block2_2_bn[0][0]          
__________________________________________________________________________________________________
conv4_block2_3_conv (Conv2D)    (None, 14, 14, 1024) 263168      conv4_block2_2_relu[0][0]        
__________________________________________________________________________________________________
conv4_block2_3_bn (BatchNormali (None, 14, 14, 1024) 4096        conv4_block2_3_conv[0][0]        
__________________________________________________________________________________________________
conv4_block2_add (Add)          (None, 14, 14, 1024) 0           conv4_block1_out[0][0]           
                                                                 conv4_block2_3_bn[0][0]          
__________________________________________________________________________________________________
conv4_block2_out (Activation)   (None, 14, 14, 1024) 0           conv4_block2_add[0][0]           
__________________________________________________________________________________________________
conv4_block3_1_conv (Conv2D)    (None, 14, 14, 256)  262400      conv4_block2_out[0][0]           
__________________________________________________________________________________________________
conv4_block3_1_bn (BatchNormali (None, 14, 14, 256)  1024        conv4_block3_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block3_1_relu (Activation (None, 14, 14, 256)  0           conv4_block3_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block3_2_conv (Conv2D)    (None, 14, 14, 256)  590080      conv4_block3_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block3_2_bn (BatchNormali (None, 14, 14, 256)  1024        conv4_block3_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block3_2_relu (Activation (None, 14, 14, 256)  0           conv4_block3_2_bn[0][0]          
__________________________________________________________________________________________________
conv4_block3_3_conv (Conv2D)    (None, 14, 14, 1024) 263168      conv4_block3_2_relu[0][0]        
__________________________________________________________________________________________________
conv4_block3_3_bn (BatchNormali (None, 14, 14, 1024) 4096        conv4_block3_3_conv[0][0]        
__________________________________________________________________________________________________
conv4_block3_add (Add)          (None, 14, 14, 1024) 0           conv4_block2_out[0][0]           
                                                                 conv4_block3_3_bn[0][0]          
__________________________________________________________________________________________________
conv4_block3_out (Activation)   (None, 14, 14, 1024) 0           conv4_block3_add[0][0]           
__________________________________________________________________________________________________
conv4_block4_1_conv (Conv2D)    (None, 14, 14, 256)  262400      conv4_block3_out[0][0]           
__________________________________________________________________________________________________
conv4_block4_1_bn (BatchNormali (None, 14, 14, 256)  1024        conv4_block4_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block4_1_relu (Activation (None, 14, 14, 256)  0           conv4_block4_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block4_2_conv (Conv2D)    (None, 14, 14, 256)  590080      conv4_block4_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block4_2_bn (BatchNormali (None, 14, 14, 256)  1024        conv4_block4_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block4_2_relu (Activation (None, 14, 14, 256)  0           conv4_block4_2_bn[0][0]          
__________________________________________________________________________________________________
conv4_block4_3_conv (Conv2D)    (None, 14, 14, 1024) 263168      conv4_block4_2_relu[0][0]        
__________________________________________________________________________________________________
conv4_block4_3_bn (BatchNormali (None, 14, 14, 1024) 4096        conv4_block4_3_conv[0][0]        
__________________________________________________________________________________________________
conv4_block4_add (Add)          (None, 14, 14, 1024) 0           conv4_block3_out[0][0]           
                                                                 conv4_block4_3_bn[0][0]          
__________________________________________________________________________________________________
conv4_block4_out (Activation)   (None, 14, 14, 1024) 0           conv4_block4_add[0][0]           
__________________________________________________________________________________________________
conv4_block5_1_conv (Conv2D)    (None, 14, 14, 256)  262400      conv4_block4_out[0][0]           
__________________________________________________________________________________________________
conv4_block5_1_bn (BatchNormali (None, 14, 14, 256)  1024        conv4_block5_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block5_1_relu (Activation (None, 14, 14, 256)  0           conv4_block5_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block5_2_conv (Conv2D)    (None, 14, 14, 256)  590080      conv4_block5_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block5_2_bn (BatchNormali (None, 14, 14, 256)  1024        conv4_block5_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block5_2_relu (Activation (None, 14, 14, 256)  0           conv4_block5_2_bn[0][0]          
__________________________________________________________________________________________________
conv4_block5_3_conv (Conv2D)    (None, 14, 14, 1024) 263168      conv4_block5_2_relu[0][0]        
__________________________________________________________________________________________________
conv4_block5_3_bn (BatchNormali (None, 14, 14, 1024) 4096        conv4_block5_3_conv[0][0]        
__________________________________________________________________________________________________
conv4_block5_add (Add)          (None, 14, 14, 1024) 0           conv4_block4_out[0][0]           
                                                                 conv4_block5_3_bn[0][0]          
__________________________________________________________________________________________________
conv4_block5_out (Activation)   (None, 14, 14, 1024) 0           conv4_block5_add[0][0]           
__________________________________________________________________________________________________
conv4_block6_1_conv (Conv2D)    (None, 14, 14, 256)  262400      conv4_block5_out[0][0]           
__________________________________________________________________________________________________
conv4_block6_1_bn (BatchNormali (None, 14, 14, 256)  1024        conv4_block6_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block6_1_relu (Activation (None, 14, 14, 256)  0           conv4_block6_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block6_2_conv (Conv2D)    (None, 14, 14, 256)  590080      conv4_block6_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block6_2_bn (BatchNormali (None, 14, 14, 256)  1024        conv4_block6_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block6_2_relu (Activation (None, 14, 14, 256)  0           conv4_block6_2_bn[0][0]          
__________________________________________________________________________________________________
conv4_block6_3_conv (Conv2D)    (None, 14, 14, 1024) 263168      conv4_block6_2_relu[0][0]        
__________________________________________________________________________________________________
conv4_block6_3_bn (BatchNormali (None, 14, 14, 1024) 4096        conv4_block6_3_conv[0][0]        
__________________________________________________________________________________________________
conv4_block6_add (Add)          (None, 14, 14, 1024) 0           conv4_block5_out[0][0]           
                                                                 conv4_block6_3_bn[0][0]          
__________________________________________________________________________________________________
conv4_block6_out (Activation)   (None, 14, 14, 1024) 0           conv4_block6_add[0][0]           
__________________________________________________________________________________________________
conv5_block1_1_conv (Conv2D)    (None, 7, 7, 512)    524800      conv4_block6_out[0][0]           
__________________________________________________________________________________________________
conv5_block1_1_bn (BatchNormali (None, 7, 7, 512)    2048        conv5_block1_1_conv[0][0]        
__________________________________________________________________________________________________
conv5_block1_1_relu (Activation (None, 7, 7, 512)    0           conv5_block1_1_bn[0][0]          
__________________________________________________________________________________________________
conv5_block1_2_conv (Conv2D)    (None, 7, 7, 512)    2359808     conv5_block1_1_relu[0][0]        
__________________________________________________________________________________________________
conv5_block1_2_bn (BatchNormali (None, 7, 7, 512)    2048        conv5_block1_2_conv[0][0]        
__________________________________________________________________________________________________
conv5_block1_2_relu (Activation (None, 7, 7, 512)    0           conv5_block1_2_bn[0][0]          
__________________________________________________________________________________________________
conv5_block1_0_conv (Conv2D)    (None, 7, 7, 2048)   2099200     conv4_block6_out[0][0]           
__________________________________________________________________________________________________
conv5_block1_3_conv (Conv2D)    (None, 7, 7, 2048)   1050624     conv5_block1_2_relu[0][0]        
__________________________________________________________________________________________________
conv5_block1_0_bn (BatchNormali (None, 7, 7, 2048)   8192        conv5_block1_0_conv[0][0]        
__________________________________________________________________________________________________
conv5_block1_3_bn (BatchNormali (None, 7, 7, 2048)   8192        conv5_block1_3_conv[0][0]        
__________________________________________________________________________________________________
conv5_block1_add (Add)          (None, 7, 7, 2048)   0           conv5_block1_0_bn[0][0]          
                                                                 conv5_block1_3_bn[0][0]          
__________________________________________________________________________________________________
conv5_block1_out (Activation)   (None, 7, 7, 2048)   0           conv5_block1_add[0][0]           
__________________________________________________________________________________________________
conv5_block2_1_conv (Conv2D)    (None, 7, 7, 512)    1049088     conv5_block1_out[0][0]           
__________________________________________________________________________________________________
conv5_block2_1_bn (BatchNormali (None, 7, 7, 512)    2048        conv5_block2_1_conv[0][0]        
__________________________________________________________________________________________________
conv5_block2_1_relu (Activation (None, 7, 7, 512)    0           conv5_block2_1_bn[0][0]          
__________________________________________________________________________________________________
conv5_block2_2_conv (Conv2D)    (None, 7, 7, 512)    2359808     conv5_block2_1_relu[0][0]        
__________________________________________________________________________________________________
conv5_block2_2_bn (BatchNormali (None, 7, 7, 512)    2048        conv5_block2_2_conv[0][0]        
__________________________________________________________________________________________________
conv5_block2_2_relu (Activation (None, 7, 7, 512)    0           conv5_block2_2_bn[0][0]          
__________________________________________________________________________________________________
conv5_block2_3_conv (Conv2D)    (None, 7, 7, 2048)   1050624     conv5_block2_2_relu[0][0]        
__________________________________________________________________________________________________
conv5_block2_3_bn (BatchNormali (None, 7, 7, 2048)   8192        conv5_block2_3_conv[0][0]        
__________________________________________________________________________________________________
conv5_block2_add (Add)          (None, 7, 7, 2048)   0           conv5_block1_out[0][0]           
                                                                 conv5_block2_3_bn[0][0]          
__________________________________________________________________________________________________
conv5_block2_out (Activation)   (None, 7, 7, 2048)   0           conv5_block2_add[0][0]           
__________________________________________________________________________________________________
conv5_block3_1_conv (Conv2D)    (None, 7, 7, 512)    1049088     conv5_block2_out[0][0]           
__________________________________________________________________________________________________
conv5_block3_1_bn (BatchNormali (None, 7, 7, 512)    2048        conv5_block3_1_conv[0][0]        
__________________________________________________________________________________________________
conv5_block3_1_relu (Activation (None, 7, 7, 512)    0           conv5_block3_1_bn[0][0]          
__________________________________________________________________________________________________
conv5_block3_2_conv (Conv2D)    (None, 7, 7, 512)    2359808     conv5_block3_1_relu[0][0]        
__________________________________________________________________________________________________
conv5_block3_2_bn (BatchNormali (None, 7, 7, 512)    2048        conv5_block3_2_conv[0][0]        
__________________________________________________________________________________________________
conv5_block3_2_relu (Activation (None, 7, 7, 512)    0           conv5_block3_2_bn[0][0]          
__________________________________________________________________________________________________
conv5_block3_3_conv (Conv2D)    (None, 7, 7, 2048)   1050624     conv5_block3_2_relu[0][0]        
__________________________________________________________________________________________________
conv5_block3_3_bn (BatchNormali (None, 7, 7, 2048)   8192        conv5_block3_3_conv[0][0]        
__________________________________________________________________________________________________
conv5_block3_add (Add)          (None, 7, 7, 2048)   0           conv5_block2_out[0][0]           
                                                                 conv5_block3_3_bn[0][0]          
__________________________________________________________________________________________________
conv5_block3_out (Activation)   (None, 7, 7, 2048)   0           conv5_block3_add[0][0]           
__________________________________________________________________________________________________
global_average_pooling2d_3 (Glo (None, 2048)         0           conv5_block3_out[0][0]           
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 2048)         0           global_average_pooling2d_3[0][0] 
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 100)          204900      dropout_3[0][0]                  
==================================================================================================
Total params: 23,792,612
Trainable params: 23,739,492
Non-trainable params: 53,120
__________________________________________________________________________________________________
```


**[Celda 43 - Código]**
```python
# Asignar la capa de pooling global a una variable
salida = resnet.get_layer('conv4_block5_add')

# Crear un nuevo modelo que tenga la capa de pooling global como salida
new_model_resnet = Model(inputs=resnet.input, outputs=salida.output)
```


**[Celda 44 - Código]**
```python

```
