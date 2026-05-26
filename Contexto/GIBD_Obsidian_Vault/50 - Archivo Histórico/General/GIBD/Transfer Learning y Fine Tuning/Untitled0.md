---
aliases: [Untitled0]
tags:
  - grupo/general
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2024-05-16
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Transfer Learning y Fine Tuning/Pruebas 2023/Untitled0.ipynb"
tamanio_bytes: 53132
---

# Notebook: Untitled0.ipynb

Ruta interna: `GIBD/Transfer Learning y Fine Tuning/Pruebas 2023/Untitled0.ipynb`

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


**[Celda 5 - Código]**
```python
def nnk(actuales,nuevo,k=1):
    p = 0
    while (p<len(actuales) and (nuevo[0]>=actuales[p][0])):
        p += 1
    actuales.insert(p, nuevo)
    while len(actuales)>k:
        actuales.pop(len(actuales)-1)
```


**[Celda 6 - Código]**
```python
def nnk2etiquetas(nnklist, y_BD):
    l = []
    for par in nnklist:
        nuevo = [par[0], y_BD[par[1]]]
        l.append(nuevo)
    return l
```


**[Celda 7 - Código]**
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
94765736/94765736 [==============================] - 0s 0us/step
```


**[Celda 8 - Código]**
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
