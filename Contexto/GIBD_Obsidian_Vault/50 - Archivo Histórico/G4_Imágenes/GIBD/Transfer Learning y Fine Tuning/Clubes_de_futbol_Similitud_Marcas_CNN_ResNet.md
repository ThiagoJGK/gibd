---
aliases: [Clubes_de_futbol_Similitud_Marcas_CNN_ResNet]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2024-05-30
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Transfer Learning y Fine Tuning/Pruebas 2023/Clubes_de_futbol_Similitud_Marcas_CNN_ResNet.ipynb"
tamanio_bytes: 72765
---

# Notebook: Clubes_de_futbol_Similitud_Marcas_CNN_ResNet.ipynb

Ruta interna: `GIBD/Transfer Learning y Fine Tuning/Pruebas 2023/Clubes_de_futbol_Similitud_Marcas_CNN_ResNet.ipynb`

---


**[Celda 1 - Código]**
```python
from google.colab import drive
drive.mount('/content/drive')
```


*Salida:*
```text
Mounted at /content/drive
```


**[Celda 2 - Código]**
```python
%cd /content/drive/MyDrive/GIBD/E.C de imágenes a color/E.C archivos/imagenesBD
```


*Salida:*
```text
/content/drive/.shortcut-targets-by-id/12cVA9aUXiq20uRXWRTzC8gXJBHvVHliY/Extraccion de Caracteristicas archivos/imagenesBD
```


**[Celda 3 - Código]**
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


image = r'/content/drive/MyDrive/GIBD/E.C de imágenes a color/E.C archivos/imagenesBD/wolfsburg.png'
#directorio = os.listdir(directory)
img = cv2.imread(image)
plt.imshow(img)
plt.show()
print(img)
print(img.shape)
#print(type(img))
#img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#plt.imshow(img)


#img5 = img / 255.0
#img5 = 1 - img5
#print(img5)
#img5 = img5.reshape(28,28,1)
#print(img5.shape)

```


*Salida:*
```text
<Figure size 640x480 with 1 Axes>[[[128 128 128]
  [128 128 128]
  [128 128 128]
  ...
  [128 128 128]
  [128 128 128]
  [128 128 128]]

 [[128 128 128]
  [128 128 128]
  [128 128 128]
  ...
  [128 128 128]
  [128 128 128]
  [128 128 128]]

 [[128 128 128]
  [128 128 128]
  [128 128 128]
  ...
  [128 128 128]
  [128 128 128]
  [128 128 128]]

 ...

 [[128 128 128]
  [128 128 128]
  [128 128 128]
  ...
  [128 128 128]
  [128 128 128]
  [128 128 128]]

 [[128 128 128]
  [128 128 128]
  [128 128 128]
  ...
  [128 128 128]
  [128 128 128]
  [128 128 128]]

 [[128 128 128]
  [128 128 128]
  [128 128 128]
  ...
  [128 128 128]
  [128 128 128]
  [128 128 128]]]
(100, 100, 3)
```


**[Celda 4 - Código]**
```python
#No se ejecuta
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

directorio = r'/content/drive/MyDrive/GIBD/E.C de imágenes a color/E.C archivos/imagenesBD'
contenido = os.listdir(directorio)

imagenesBD = []
y_imagenesBD = []
for fichero in contenido:
    if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.jpg'):
        image = directorio + '/' + fichero
        image_gris = cv2.imread(image)
        image_gris = image_gris / 255
        image_gris = 1 - image_gris
        imagenesBD.append(image_gris)
        n = fichero[:fichero.find('.')]
        y_imagenesBD.append(n)
print(y_imagenesBD)
```


*Salida:*
```text
['190', '189', '309', '304', '302', '305', '191', '187', '306', '173', '307', '150', '70', '85', '186', '194', '130', '132', '203', '195', '152', '167', '87', '196', '116', '198', '308', '200', '129', '151', '310', '164', '149', '81', '192', '128', '193', '199', '201', '202', '96', '301', '103', '84', '303', '188', '98', '165', '75', '76', '120', '166', '155', '154', '162', '139', '197', '111', '88', '124', '77', '145', '117', '112', '147', '172', '69', '125', '97', '160', '94', '79', '127', '142', '170', '106', '176', '134', '171', '114', '174', '95', '74', '93', '109', '123', '113', '89', '115', '168', '82', '72', '175', '101', '180', '71', '161', '138', '184', '169', '100', '99', '148', '122', '119', '146', '153', '90', '73', '133', '104', '185', '102', '83', '177', '181', '110', '86', '136', '105', '58', '44', '163', '78', '143', '144', '80', '158', '178', '126', '137', '64', '135', '183', '30', '14', '20', '118', '182', '107', '157', '108', '141', '121', '46', '9', '156', '8', '179', '15', '131', '92', '19', '26', '34', '59', '7', '49', '4', '60', '1', '22', '43', '21', '10', '41', '27', '24', '50', '39', '62', '2', '13', '25', '17', '35', '40', '53', '140', '28', '6', '18', '159', '11', '38', '63', '42', '36', '23', '91', '66', '68', '45', '37', '16', '31', '3', '54', '32', '56', '5', '61', '29', '55', '57', '67', '52', '47', '65', '33', '12', '48', '51']
```


**[Celda 5 - Código]**
```python
from PIL import Image
import numpy as np
import os

def crop_and_normalize_images(image_paths, target_size):
    normalized_images = []
    etiquetas = []

    for path in image_paths:
        image = Image.open(path)

        if image.size != target_size:
            image = image.resize(target_size)

        # Obtener la etiqueta de la imagen
        filename = os.path.basename(path)
        nombre = filename[:filename.find(".")]
        etiquetas.append(nombre)
        print(nombre)

        # Verificar si la imagen tiene un canal alfa
        has_alpha = 'A' in image.mode

        # Convertir la imagen en un array numpy
        image_array = np.array(image)

        if has_alpha:
            print('TIENE ALPHA...')
            # Si tiene canal alfa, eliminarlo y convertir a RGB
            image_array = image_array[:, :, :3]

        # Encontrar las coordenadas del cuadro de recorte
        non_empty_columns = np.where(image_array.min(axis=0) < 250)[0]
        non_empty_rows = np.where(image_array.min(axis=1) < 250)[0]

        minRow, maxRow = np.min(non_empty_rows), np.max(non_empty_rows)
        minCol, maxCol = np.min(non_empty_columns), np.max(non_empty_columns)

        crop_box = (minCol, minRow, maxCol, maxRow)

        # Recortar y redimensionar la imagen
        cropped_resized_image = image.crop(crop_box).resize(target_size, Image.LANCZOS)

        # Convertir la imagen recortada y redimensionada en un array numpy y normalizar
        normalized_image = np.array(cropped_resized_image)

        normalized_images.append(normalized_image)

    print(normalized_images[0].shape)

    return np.array(normalized_images), np.array(etiquetas)

# Tamaño objetivo después de recortar y redimensionar
target_size = (224, 224)

# Cargar las rutas de cada imagen de la base de datos
orig_dir = r'/content/drive/MyDrive/GIBD/E.C de imágenes a color/E.C archivos/imagenesBD'
image_paths = [os.path.join(orig_dir, filename) for filename in os.listdir(orig_dir)]

# Normalizar las imágenes de la base de datos
imagenesBD, y_imagenesBD = crop_and_normalize_images(image_paths, target_size)

# Cargar las rutas de cada imagen de consulta
orig_dir = r'/content/drive/MyDrive/GIBD/E.C de imágenes a color/E.C archivos/consultas'
image_paths = [os.path.join(orig_dir, filename) for filename in os.listdir(orig_dir)]

# Normalizar las imágenes de consulta
consultas, y_consultas = crop_and_normalize_images(image_paths, target_size)

```


*Salida:*
```text
freiburg
TIENE ALPHA...
atletico-madrid
TIENE ALPHA...
athletic
TIENE ALPHA...
estac-troyes
TIENE ALPHA...
hamburg
TIENE ALPHA...
montpellier-herault
TIENE ALPHA...
stade-de-reims
TIENE ALPHA...
bremen
TIENE ALPHA...
fc-lorient
TIENE ALPHA...
clermont-foot-63
TIENE ALPHA...
barcelona
TIENE ALPHA...
redbull-leipzig
TIENE ALPHA...
schalke
TIENE ALPHA...
nuremberg
TIENE ALPHA...
fc-nantes
TIENE ALPHA...
stade-rennais-fc
TIENE ALPHA...
deportivo-alavez
TIENE ALPHA...
moenchengladbach
TIENE ALPHA...
elche
TIENE ALPHA...
mainz
TIENE ALPHA...
fc-girondins-de-bordeaux
TIENE ALPHA...
cadiz
TIENE ALPHA...
losc-lille
TIENE ALPHA...
olympique-de-marseille
TIENE ALPHA...
as-saint-etienne
TIENE ALPHA...
rc-strasbourg-alsace
TIENE ALPHA...
espanyol
TIENE ALPHA...
stuttgart
TIENE ALPHA...
stade-brestois-29
TIENE ALPHA...
rc-lens
TIENE ALPHA...
fc-metz
TIENE ALPHA...
hertha-bsc-berlin
TIENE ALPHA...
frankfurt
TIENE ALPHA...
wolfsburg
TIENE ALPHA...
dusseldorf
TIENE ALPHA...
dortmund
TIENE ALPHA...
augsburg
TIENE ALPHA...
as-monaco
TIENE ALPHA...
celta
TIENE ALPHA...
getafe
TIENE ALPHA...
leverkusen
TIENE ALPHA...
angers-sco
TIENE ALPHA...
paris-saint-germain
TIENE ALPHA...
hoffenheim
TIENE ALPHA...
furth
TIENE ALPHA...
hannover
TIENE ALPHA...
ogc-nice
TIENE ALPHA...
bayern
TIENE ALPHA...
olympique-lyonnais
TIENE ALPHA...
leeds-united
TIENE ALPHA...
osasuna
TIENE ALPHA...
chelsea
TIENE ALPHA...
norwich-city
TIENE ALPHA...
real-sociedad
TIENE ALPHA...
southampton
TIENE ALPHA...
mallorca
TIENE ALPHA...
atalanta
TIENE ALPHA...
hellas-verona
TIENE ALPHA...
newcastle-united
TIENE ALPHA...
real-madrid
TIENE ALPHA...
levante
TIENE ALPHA...
napoli
TIENE ALPHA...
rayo-vallecano
TIENE ALPHA...
lazio
TIENE ALPHA...
granada
TIENE ALPHA...
udinese
TIENE ALPHA...
crystal-palace
TIENE ALPHA...
torino
TIENE ALPHA...
watford
TIENE ALPHA...
salernitana
TIENE ALPHA...
liverpool
TIENE ALPHA...
west-ham-united
TIENE ALPHA...
arsenal
TIENE ALPHA...
sassuolo
TIENE ALPHA...
bologna
TIENE ALPHA...
venezia
TIENE ALPHA...
spezia
TIENE ALPHA...
genoa
TIENE ALPHA...
roma
TIENE ALPHA...
everton
TIENE ALPHA...
valencia
TIENE ALPHA...
real-betis
TIENE ALPHA...
manchester-united
TIENE ALPHA...
sevilla
TIENE ALPHA...
brighton
TIENE ALPHA...
aston-villa
TIENE ALPHA...
inter
TIENE ALPHA...
fiorentina
TIENE ALPHA...
tottenham-hotspur
TIENE ALPHA...
brentford
TIENE ALPHA...
burnley
TIENE ALPHA...
sampdoria
TIENE ALPHA...
cagliari
TIENE ALPHA...
ac-milan
TIENE ALPHA...
empoli
TIENE ALPHA...
leicester-city
TIENE ALPHA...
villarreal
TIENE ALPHA...
manchester-city
TIENE ALPHA...
juventus
TIENE ALPHA...
the-strongest
TIENE ALPHA...
mineiro
TIENE ALPHA...
botafogo
TIENE ALPHA...
corinthians
TIENE ALPHA...
cruzeiro
TIENE ALPHA...
gremio
TIENE ALPHA...
fluminense
TIENE ALPHA...
flamengo
TIENE ALPHA...
internacional
TIENE ALPHA...
palmeiras
TIENE ALPHA...
fortaleza
TIENE ALPHA...
santos
TIENE ALPHA...
millonarios
TIENE ALPHA...
deportivo-cali
TIENE ALPHA...
atletico-nacional
TIENE ALPHA...
luton
TIENE ALPHA...
velez
TIENE ALPHA...
san-lorenzo
TIENE ALPHA...
central
TIENE ALPHA...
racing
TIENE ALPHA...
independiente
TIENE ALPHA...
newells
TIENE ALPHA...
lanus
TIENE ALPHA...
platense
TIENE ALPHA...
gimnasia
TIENE ALPHA...
estudiantes
TIENE ALPHA...
boca
TIENE ALPHA...
belgrano
TIENE ALPHA...
argentinos
TIENE ALPHA...
sheffield
TIENE ALPHA...
estrella-roja
TIENE ALPHA...
maccabi-haifa
TIENE ALPHA...
parma
TIENE ALPHA...
universitarios
TIENE ALPHA...
sporting-cristal
TIENE ALPHA...
porto
TIENE ALPHA...
ajax
TIENE ALPHA...
deportivo-alaves
TIENE ALPHA...
real-murcia
TIENE ALPHA...
galatasaray
TIENE ALPHA...
galaxy
TIENE ALPHA...
nacional
TIENE ALPHA...
peñarol
TIENE ALPHA...
malaga
TIENE ALPHA...
seattle-sounders
TIENE ALPHA...
ferro
TIENE ALPHA...
brujas
TIENE ALPHA...
colo-colo
TIENE ALPHA...
america
TIENE ALPHA...
blackburn
TIENE ALPHA...
huracan
TIENE ALPHA...
(224, 224, 4)
aston-villa
TIENE ALPHA...
augsburg
TIENE ALPHA...
bremen
TIENE ALPHA...
cadiz
TIENE ALPHA...
celta
TIENE ALPHA...
dusseldorf
TIENE ALPHA...
empoli
TIENE ALPHA...
clermont-foot-63
TIENE ALPHA...
deportivo-alavez
TIENE ALPHA...
udinese
TIENE ALPHA...
paris-saint-germain
TIENE ALPHA...
elche
TIENE ALPHA...
manchester-united
TIENE ALPHA...
ogc-nice
TIENE ALPHA...
levante
TIENE ALPHA...
spezia
TIENE ALPHA...
wolfsburg
TIENE ALPHA...
stuttgart
TIENE ALPHA...
barcelona
TIENE ALPHA...
inter
TIENE ALPHA...
roma
TIENE ALPHA...
villarreal
TIENE ALPHA...
crystal-palace
TIENE ALPHA...
fiorentina
TIENE ALPHA...
moenchengladbach
TIENE ALPHA...
napoli
TIENE ALPHA...
chelsea
TIENE ALPHA...
cagliari
TIENE ALPHA...
sampdoria
TIENE ALPHA...
real-betis
TIENE ALPHA...
olympique-lyonnais
TIENE ALPHA...
juventus
TIENE ALPHA...
getafe
TIENE ALPHA...
brentford
TIENE ALPHA...
watford
TIENE ALPHA...
bologna
TIENE ALPHA...
norwich-city
TIENE ALPHA...
granada
TIENE ALPHA...
valencia
TIENE ALPHA...
mainz
TIENE ALPHA...
bayern
TIENE ALPHA...
fc-lorient
TIENE ALPHA...
west-ham-united
TIENE ALPHA...
genoa
TIENE ALPHA...
hellas-verona
TIENE ALPHA...
liverpool
TIENE ALPHA...
southampton
TIENE ALPHA...
burnley
TIENE ALPHA...
hoffenheim
TIENE ALPHA...
dortmund
TIENE ALPHA...
freiburg
TIENE ALPHA...
espanyol
TIENE ALPHA...
redbull-leipzig
TIENE ALPHA...
schalke
TIENE ALPHA...
ac-milan
TIENE ALPHA...
arsenal
TIENE ALPHA...
real-madrid
TIENE ALPHA...
real-sociedad
TIENE ALPHA...
manchester-city
TIENE ALPHA...
nacional
atlanta
TIENE ALPHA...
peñarol
TIENE ALPHA...
gimnasia
TIENE ALPHA...
atletico-tucuman
TIENE ALPHA...
cerro-largo
TIENE ALPHA...
argentinos-juniors
TIENE ALPHA...
wanders
TIENE ALPHA...
colon
TIENE ALPHA...
danubio
river-plate
TIENE ALPHA...
rampla-juniors
TIENE ALPHA...
bella-vista
TIENE ALPHA...
central-español
TIENE ALPHA...
club-atletico-progreso
TIENE ALPHA...
defensor-sporting
TIENE ALPHA...
aldosivi
TIENE ALPHA...
arsenal-sarandi
TIENE ALPHA...
boca
TIENE ALPHA...
san-lorenzo
TIENE ALPHA...
independiente
TIENE ALPHA...
(224, 224, 4)
```




**[Celda 7 - Código]**
```python
#No se ejecuta
# CARGAR CONSULTAS DIBUJADAS A MANO
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


directorio = r'/content/drive/MyDrive/Grupo de investigación (personal)/BBDD/QueriesDibujadas(224,224,3)'
contenido = os.listdir(directorio)


consultas = []
y_consultas = []
for fichero in contenido:
    if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.jpg'):
        image = directorio + '/' + fichero
        image_gris = cv2.imread(image)
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
['47', '128', '126', '100', '145', '130', '79', '131', '115', '129', '146', '32', '166', '8', '174', '118', '2', '195', '170', '144', '134', '127', '59', '175', '125', '133', '37', '24', '58', '197', '7', '134 (2)', '88', '94', '107', '132', '143', '192', '55', '31']
```


**[Celda 8 - Código]**
```python
def nnk(actuales,nuevo,k=1):
    p = 0
    while (p<len(actuales) and (nuevo[0]>=actuales[p][0])):
        p += 1
    actuales.insert(p, nuevo)
    while len(actuales)>k:
        actuales.pop(len(actuales)-1)
```


**[Celda 9 - Código]**
```python
def nnk2etiquetas(nnklist, y_BD):
    l = []
    for par in nnklist:
        nuevo = [par[0], y_BD[par[1]]]
        l.append(nuevo)
    return l
```


**[Celda 10 - Código]**
```python
from keras.applications.resnet import ResNet50
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Dropout, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_resnet(input_shape=(224, 224, 3), n_classes=10):
    input_layer = Input(shape=input_shape)
    resnet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=input_layer)
    x = GlobalAveragePooling2D()(resnet50.output)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(input_layer, x)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=3e-4), metrics=['accuracy'])
    return model

resnet = build_resnet(n_classes=128)

```


*Salida:*
```text
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
94765736/94765736 [==============================] - 1s 0us/step
```


**[Celda 11 - Código]**
```python
from torchvision.models.resnet import resnet50
#Algoritmo optimizado

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
vectores_elemento = resnet.predict(imagenesBD)
vectores_consulta = resnet.predict(consultas)

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
4/4 [==============================] - 28s 7s/step
2/2 [==============================] - 11s 4s/step
Consulta número:  0
   Imagen de consulta:  aston-villa
   Mas cercanos:  [[0.063724145, 'aston-villa'], [0.07632743, 'nuremberg'], [0.07961632, 'everton'], [0.082219206, 'burnley'], [0.082797825, 'arsenal']]
--> ACIERTO
    Primer Lugar: 1
    Entre los tres primeros: 1


Consulta número:  1
   Imagen de consulta:  augsburg
   Mas cercanos:  [[0.015950995, 'augsburg'], [0.05922222, 'genoa'], [0.06205053, 'wolfsburg'], [0.062254578, 'rc-lens'], [0.07593947, 'getafe']]
--> ACIERTO
    Primer Lugar: 2
    Entre los tres primeros: 2


Consulta número:  2
   Imagen de consulta:  bremen
   Mas cercanos:  [[0.0950202, 'cagliari'], [0.0983085, 'freiburg'], [0.10340732, 'fc-girondins-de-bordeaux'], [0.1090693, 'atalanta'], [0.11138194, 'moenchengladbach']]


Consulta número:  3
   Imagen de consulta:  cadiz
   Mas cercanos:  [[0.07795565, 'lazio'], [0.086062774, 'real-betis'], [0.08691801, 'as-monaco'], [0.08933577, 'granada'], [0.09040915, 'rayo-vallecano']]


Consulta número:  4
   Imagen de consulta:  celta
   Mas cercanos:  [[0.06852272, 'furth'], [0.07597048, 'brentford'], [0.080459826, 'nuremberg'], [0.0825859, 'hannover'], [0.083688684, 'angers-sco']]


Consulta número:  5
   Imagen de consulta:  dusseldorf
   Mas cercanos:  [[0.04519065, 'dusseldorf'], [0.08940552, 'as-monaco'], [0.090149455, 'montpellier-herault'], [0.09243598, 'nuremberg'], [0.09328602, 'brighton']]
--> ACIERTO
    Primer Lugar: 3
    Entre los tres primeros: 3


Consulta número:  6
   Imagen de consulta:  empoli
   Mas cercanos:  [[0.031063525, 'empoli'], [0.08878951, 'stade-brestois-29'], [0.0927783, 'salernitana'], [0.09362309, 'fc-girondins-de-bordeaux'], [0.096602805, 'everton']]
--> ACIERTO
    Primer Lugar: 4
    Entre los tres primeros: 4


Consulta número:  7
   Imagen de consulta:  clermont-foot-63
   Mas cercanos:  [[0.06513387, 'clermont-foot-63'], [0.08004666, 'torino'], [0.082283616, 'stuttgart'], [0.09411122, 'liverpool'], [0.09458236, 'athletic']]
--> ACIERTO
    Primer Lugar: 5
    Entre los tres primeros: 5


Consulta número:  8
   Imagen de consulta:  deportivo-alavez
   Mas cercanos:  [[0.0996674, 'deportivo-alavez'], [0.11593224, 'hertha-bsc-berlin'], [0.11884972, 'salernitana'], [0.1191757, 'empoli'], [0.11948876, 'stade-brestois-29']]
--> ACIERTO
    Primer Lugar: 6
    Entre los tres primeros: 6


Consulta número:  9
   Imagen de consulta:  udinese
   Mas cercanos:  [[0.07381816, 'udinese'], [0.08161936, 'chelsea'], [0.088596515, 'redbull-leipzig'], [0.09698614, 'estac-troyes'], [0.102489084, 'dortmund']]
--> ACIERTO
    Primer Lugar: 7
    Entre los tres primeros: 7


Consulta número:  10
   Imagen de consulta:  paris-saint-germain
   Mas cercanos:  [[0.040265735, 'paris-saint-germain'], [0.08183725, 'chelsea'], [0.091923, 'estac-troyes'], [0.09673352, 'manchester-city'], [0.0978241, 'redbull-leipzig']]
--> ACIERTO
    Primer Lugar: 8
    Entre los tres primeros: 8


Consulta número:  11
   Imagen de consulta:  elche
   Mas cercanos:  [[0.05904543, 'elche'], [0.07131474, 'wolfsburg'], [0.073820725, 'norwich-city'], [0.073928334, 'rc-lens'], [0.076964356, 'getafe']]
--> ACIERTO
    Primer Lugar: 9
    Entre los tres primeros: 9


Consulta número:  12
   Imagen de consulta:  manchester-united
   Mas cercanos:  [[0.07650288, 'manchester-united'], [0.10162432, 'southampton'], [0.13718337, 'mallorca'], [0.14569269, 'leverkusen'], [0.1609989, 'stade-de-reims']]
--> ACIERTO
    Primer Lugar: 10
    Entre los tres primeros: 10


Consulta número:  13
   Imagen de consulta:  ogc-nice
   Mas cercanos:  [[0.08862463, 'barcelona'], [0.09637039, 'ogc-nice'], [0.109042384, 'newcastle-united'], [0.109594814, 'hannover'], [0.11082071, 'rc-lens']]
--> ACIERTO
    Entre los tres primeros: 11


Consulta número:  14
   Imagen de consulta:  levante
   Mas cercanos:  [[0.078315586, 'hannover'], [0.08063135, 'newcastle-united'], [0.09264347, 'venezia'], [0.09375025, 'sassuolo'], [0.09468174, 'roma']]


Consulta número:  15
   Imagen de consulta:  spezia
   Mas cercanos:  [[0.030399209, 'spezia'], [0.06279738, 'nuremberg'], [0.06564181, 'montpellier-herault'], [0.07545368, 'brentford'], [0.07838153, 'everton']]
--> ACIERTO
    Primer Lugar: 11
    Entre los tres primeros: 12


Consulta número:  16
   Imagen de consulta:  wolfsburg
   Mas cercanos:  [[0.02621544, 'wolfsburg'], [0.06047688, 'rc-lens'], [0.07042434, 'as-saint-etienne'], [0.07112945, 'getafe'], [0.07836999, 'augsburg']]
--> ACIERTO
    Primer Lugar: 12
    Entre los tres primeros: 13


Consulta número:  17
   Imagen de consulta:  stuttgart
   Mas cercanos:  [[0.07349101, 'norwich-city'], [0.07923423, 'elche'], [0.081195086, 'furth'], [0.08315104, 'arsenal'], [0.08449882, 'brighton']]


Consulta número:  18
   Imagen de consulta:  barcelona
   Mas cercanos:  [[0.030676067, 'barcelona'], [0.08708615, 'real-madrid'], [0.095613815, 'hannover'], [0.09719349, 'newcastle-united'], [0.110733435, 'leverkusen']]
--> ACIERTO
    Primer Lugar: 13
    Entre los tres primeros: 14


Consulta número:  19
   Imagen de consulta:  inter
   Mas cercanos:  [[0.01840178, 'inter'], [0.09747781, 'chelsea'], [0.09916252, 'estac-troyes'], [0.11058024, 'bayern'], [0.11081036, 'paris-saint-germain']]
--> ACIERTO
    Primer Lugar: 14
    Entre los tres primeros: 15


Consulta número:  20
   Imagen de consulta:  roma
   Mas cercanos:  [[0.022280902, 'roma'], [0.08242099, 'fc-girondins-de-bordeaux'], [0.08694129, 'hannover'], [0.08896939, 'athletic'], [0.08956477, 'norwich-city']]
--> ACIERTO
    Primer Lugar: 15
    Entre los tres primeros: 16


Consulta número:  21
   Imagen de consulta:  villarreal
   Mas cercanos:  [[0.1138162, 'as-saint-etienne'], [0.1291518, 'wolfsburg'], [0.14000465, 'rc-lens'], [0.14077789, 'stade-rennais-fc'], [0.14512366, 'villarreal']]
--> ACIERTO


Consulta número:  22
   Imagen de consulta:  crystal-palace
   Mas cercanos:  [[0.06175636, 'crystal-palace'], [0.114057735, 'as-monaco'], [0.116144344, 'hoffenheim'], [0.11757743, 'hellas-verona'], [0.11814449, 'brighton']]
--> ACIERTO
    Primer Lugar: 16
    Entre los tres primeros: 17


Consulta número:  23
   Imagen de consulta:  machester-city
   Mas cercanos:  [[0.045129474, 'manchester-city'], [0.08987222, 'liverpool'], [0.09139524, 'leicester-city'], [0.10589137, 'brentford'], [0.10726959, 'redbull-leipzig']]


Consulta número:  24
   Imagen de consulta:  fiorentina
   Mas cercanos:  [[0.10400976, 'leeds-united'], [0.10816977, 'bremen'], [0.110534094, 'cadiz'], [0.11369354, 'granada'], [0.11635304, 'rayo-vallecano']]


Consulta número:  25
   Imagen de consulta:  moenchengladbach
   Mas cercanos:  [[0.014726346, 'moenchengladbach'], [0.06076541, 'angers-sco'], [0.06672086, 'hamburg'], [0.07237507, 'fc-girondins-de-bordeaux'], [0.084566064, 'spezia']]
--> ACIERTO
    Primer Lugar: 17
    Entre los tres primeros: 18


Consulta número:  26
   Imagen de consulta:  napoli
   Mas cercanos:  [[0.03621966, 'napoli'], [0.07323371, 'nuremberg'], [0.0740862, 'everton'], [0.07952768, 'leeds-united'], [0.081261955, 'hannover']]
--> ACIERTO
    Primer Lugar: 18
    Entre los tres primeros: 19


Consulta número:  27
   Imagen de consulta:  chelsea
   Mas cercanos:  [[0.01588804, 'chelsea'], [0.062800914, 'estac-troyes'], [0.08332238, 'redbull-leipzig'], [0.086822435, 'paris-saint-germain'], [0.09626918, 'udinese']]
--> ACIERTO
    Primer Lugar: 19
    Entre los tres primeros: 20


Consulta número:  28
   Imagen de consulta:  cagliari
   Mas cercanos:  [[0.016862892, 'cagliari'], [0.10864586, 'freiburg'], [0.12294015, 'atalanta'], [0.12309933, 'fc-girondins-de-bordeaux'], [0.12871122, 'sassuolo']]
--> ACIERTO
    Primer Lugar: 20
    Entre los tres primeros: 21


Consulta número:  29
   Imagen de consulta:  sampdoria
   Mas cercanos:  [[0.09067208, 'sampdoria'], [0.09387696, 'as-monaco'], [0.094590716, 'granada'], [0.09744207, 'real-betis'], [0.0987589, 'brighton']]
--> ACIERTO
    Primer Lugar: 21
    Entre los tres primeros: 22


Consulta número:  30
   Imagen de consulta:  real-betis
   Mas cercanos:  [[0.040784825, 'real-betis'], [0.072940774, 'aston-villa'], [0.07386192, 'nuremberg'], [0.083233915, 'as-monaco'], [0.08370035, 'arsenal']]
--> ACIERTO
    Primer Lugar: 22
    Entre los tres primeros: 23


Consulta número:  31
   Imagen de consulta:  olympique-lyonnais
   Mas cercanos:  [[0.022688206, 'olympique-lyonnais'], [0.07393251, 'aston-villa'], [0.08050383, 'montpellier-herault'], [0.081246205, 'nuremberg'], [0.08128154, 'dortmund']]
--> ACIERTO
    Primer Lugar: 23
    Entre los tres primeros: 24


Consulta número:  32
   Imagen de consulta:  juventus
   Mas cercanos:  [[0.08029324, 'augsburg'], [0.080374144, 'genoa'], [0.08392598, 'juventus'], [0.101659514, 'aston-villa'], [0.10415968, 'fc-metz']]
--> ACIERTO
    Entre los tres primeros: 25


Consulta número:  33
   Imagen de consulta:  getafe
   Mas cercanos:  [[0.07445565, 'wolfsburg'], [0.080792144, 'as-saint-etienne'], [0.08475648, 'rc-lens'], [0.08663554, 'getafe'], [0.10274654, 'augsburg']]
--> ACIERTO


Consulta número:  34
   Imagen de consulta:  brentford
   Mas cercanos:  [[0.042700607, 'brentford'], [0.07539286, 'leverkusen'], [0.07554644, 'spezia'], [0.078728825, 'hannover'], [0.07948924, 'nuremberg']]
--> ACIERTO
    Primer Lugar: 24
    Entre los tres primeros: 26


Consulta número:  35
   Imagen de consulta:  watford
   Mas cercanos:  [[0.0924638, 'moenchengladbach'], [0.10975263, 'spezia'], [0.11122793, 'hamburg'], [0.11199365, 'angers-sco'], [0.112395145, 'brentford']]


Consulta número:  36
   Imagen de consulta:  bologna
   Mas cercanos:  [[0.031997126, 'bologna'], [0.073752485, 'rc-lens'], [0.07732245, 'getafe'], [0.08369435, 'genoa'], [0.08392683, 'wolfsburg']]
--> ACIERTO
    Primer Lugar: 25
    Entre los tres primeros: 27


Consulta número:  37
   Imagen de consulta:  norwich-city
   Mas cercanos:  [[0.10527694, 'wolfsburg'], [0.117448546, 'augsburg'], [0.12035965, 'fc-nantes'], [0.12109111, 'rc-lens'], [0.121208705, 'as-saint-etienne']]


Consulta número:  38
   Imagen de consulta:  granada
   Mas cercanos:  [[0.08530437, 'granada'], [0.102351926, 'lazio'], [0.10354694, 'osasuna'], [0.10409356, 'rayo-vallecano'], [0.10523781, 'stuttgart']]
--> ACIERTO
    Primer Lugar: 26
    Entre los tres primeros: 28


Consulta número:  39
   Imagen de consulta:  valencia
   Mas cercanos:  [[0.04522885, 'valencia'], [0.071584396, 'hannover'], [0.07203962, 'bologna'], [0.0807686, 'leverkusen'], [0.08216371, 'nuremberg']]
--> ACIERTO
    Primer Lugar: 27
    Entre los tres primeros: 29


Consulta número:  40
   Imagen de consulta:  mainz
   Mas cercanos:  [[0.071148485, 'nuremberg'], [0.072544806, 'olympique-lyonnais'], [0.07277094, 'montpellier-herault'], [0.077979125, 'dortmund'], [0.0813917, 'leeds-united']]


Consulta número:  41
   Imagen de consulta:  bayern
   Mas cercanos:  [[0.0074729514, 'bayern'], [0.07811994, 'estac-troyes'], [0.0825473, 'frankfurt'], [0.08845604, 'udinese'], [0.09018757, 'chelsea']]
--> ACIERTO
    Primer Lugar: 28
    Entre los tres primeros: 30


Consulta número:  42
   Imagen de consulta:  fc-lorient
   Mas cercanos:  [[0.04855677, 'fc-lorient'], [0.073359825, 'roma'], [0.08111531, 'freiburg'], [0.090180844, 'fc-girondins-de-bordeaux'], [0.09221339, 'hertha-bsc-berlin']]
--> ACIERTO
    Primer Lugar: 29
    Entre los tres primeros: 31


Consulta número:  43
   Imagen de consulta:  west-ham-united
   Mas cercanos:  [[0.035755455, 'west-ham-united'], [0.122573115, 'fiorentina'], [0.13443299, 'clermont-foot-63'], [0.13948947, 'mainz'], [0.14035246, 'chelsea']]
--> ACIERTO
    Primer Lugar: 30
    Entre los tres primeros: 32


Consulta número:  44
   Imagen de consulta:  genoa
   Mas cercanos:  [[0.044699106, 'genoa'], [0.077808924, 'aston-villa'], [0.078197174, 'augsburg'], [0.09067615, 'getafe'], [0.09268086, 'burnley']]
--> ACIERTO
    Primer Lugar: 31
    Entre los tres primeros: 33


Consulta número:  45
   Imagen de consulta:  hellas-verona
   Mas cercanos:  [[0.09199338, 'celta'], [0.09896359, 'hellas-verona'], [0.10077653, 'burnley'], [0.101448916, 'lazio'], [0.10530109, 'aston-villa']]
--> ACIERTO
    Entre los tres primeros: 34


Consulta número:  46
   Imagen de consulta:  liverpool
   Mas cercanos:  [[0.072590984, 'liverpool'], [0.08409356, 'stuttgart'], [0.09536153, 'clermont-foot-63'], [0.1025301, 'torino'], [0.10493772, 'lazio']]
--> ACIERTO
    Primer Lugar: 32
    Entre los tres primeros: 35


Consulta número:  47
   Imagen de consulta:  southampton
   Mas cercanos:  [[0.12846103, 'southampton'], [0.17905645, 'manchester-united'], [0.21434364, 'mallorca'], [0.23854074, 'leverkusen'], [0.2412529, 'crystal-palace']]
--> ACIERTO
    Primer Lugar: 33
    Entre los tres primeros: 36


Consulta número:  48
   Imagen de consulta:  burnley
   Mas cercanos:  [[0.060761716, 'burnley'], [0.0833318, 'hellas-verona'], [0.08872373, 'sevilla'], [0.092384584, 'aston-villa'], [0.0938506, 'as-monaco']]
--> ACIERTO
    Primer Lugar: 34
    Entre los tres primeros: 37


Consulta número:  49
   Imagen de consulta:  hoffenheim
   Mas cercanos:  [[0.008773083, 'hoffenheim'], [0.07114788, 'everton'], [0.07798121, 'brighton'], [0.083482504, 'fc-metz'], [0.08970667, 'nuremberg']]
--> ACIERTO
    Primer Lugar: 35
    Entre los tres primeros: 38


Consulta número:  50
   Imagen de consulta:  dortmund
   Mas cercanos:  [[0.06107767, 'dortmund'], [0.06550121, 'napoli'], [0.07662184, 'nuremberg'], [0.077321894, 'rc-lens'], [0.07942921, 'getafe']]
--> ACIERTO
    Primer Lugar: 36
    Entre los tres primeros: 39


Consulta número:  51
   Imagen de consulta:  freiburg
   Mas cercanos:  [[0.05509553, 'freiburg'], [0.06629209, 'fc-girondins-de-bordeaux'], [0.0730177, 'spezia'], [0.07437262, 'atalanta'], [0.077428855, 'hertha-bsc-berlin']]
--> ACIERTO
    Primer Lugar: 37
    Entre los tres primeros: 40


Consulta número:  52
   Imagen de consulta:  espanyol
   Mas cercanos:  [[0.028270552, 'espanyol'], [0.08170489, 'real-betis'], [0.08623188, 'olympique-lyonnais'], [0.088901155, 'aston-villa'], [0.09229675, 'as-monaco']]
--> ACIERTO
    Primer Lugar: 38
    Entre los tres primeros: 41


Consulta número:  53
   Imagen de consulta:  redbull-leipzig
   Mas cercanos:  [[0.035956703, 'redbull-leipzig'], [0.086831324, 'chelsea'], [0.09881498, 'liverpool'], [0.09974922, 'udinese'], [0.10452254, 'estac-troyes']]
--> ACIERTO
    Primer Lugar: 39
    Entre los tres primeros: 42


Consulta número:  54
   Imagen de consulta:  schalke
   Mas cercanos:  [[0.038856633, 'schalke'], [0.08043206, 'brighton'], [0.08807805, 'deportivo-alavez'], [0.09117226, 'hertha-bsc-berlin'], [0.09448699, 'hoffenheim']]
--> ACIERTO
    Primer Lugar: 40
    Entre los tres primeros: 43


Porcentaje de aciertos NN1:  0.7272727272727273
Porcentaje de aciertos NN3:  0.7818181818181819
Porcentaje de aciertos NN5:  0.8181818181818182
  Cantidad consultas:  55
```


**[Celda 12 - Código]**
```python
resnet.summary()

```
