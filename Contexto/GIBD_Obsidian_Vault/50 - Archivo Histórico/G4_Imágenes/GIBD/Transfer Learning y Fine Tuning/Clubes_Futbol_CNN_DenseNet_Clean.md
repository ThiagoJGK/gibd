---
aliases: [Clubes_Futbol_CNN_DenseNet_Clean]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2024-05-16
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Transfer Learning y Fine Tuning/Pruebas 2023/Clubes_Futbol_CNN_DenseNet_Clean.ipynb"
tamanio_bytes: 248215
---

# Notebook: Clubes_Futbol_CNN_DenseNet_Clean.ipynb

Ruta interna: `GIBD/Transfer Learning y Fine Tuning/Pruebas 2023/Clubes_Futbol_CNN_DenseNet_Clean.ipynb`

---

Primer paso: Cargar todas las imagenes de la carpeta de marcas


**[Celda 2 - Código]**
```python
from google.colab import drive
drive.mount('/content/drive')
```


*Salida:*
```text
Mounted at /content/drive
```


**[Celda 3 - Código]**
```python
%cd /content/drive/MyDrive/GIBD/E.C de imágenes a color/E.C archivos/imagenesBD
```


*Salida:*
```text
/content/drive/.shortcut-targets-by-id/12cVA9aUXiq20uRXWRTzC8gXJBHvVHliY/Extraccion de Caracteristicas archivos/imagenesBD
```


**[Celda 4 - Código]**
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


**[Celda 5 - Código]**
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
    if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.png'):
        image = directorio + '/' + fichero
        image_gris = cv2.imread(image)
        image_gris = image_gris / 255
        image_gris = 1 - image_gris
        imagenesBD.append(image_gris)
        n = fichero[:fichero.find('.')]
        y_imagenesBD.append(n)
print(y_imagenesBD)
```


**[Celda 6 - Código]**
```python
## CARGAR Y NORMALIZAR LAS IMAGENES DE LA BD Y LAS CONSULTAS
from PIL import Image, ImageEnhance

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


target_size = (224, 224)  # El tamaño objetivo después de recortar y redimensionar

## CARGAR LAS RUTAS DE CADA IMAGEN DE LA BASE DE DATOS
orig_dir = r'/content/drive/MyDrive/GIBD/E.C de imágenes a color/E.C archivos/imagenesBD'

image_paths = []

for filename in os.listdir(orig_dir):
    image_paths.append(os.path.join(orig_dir, filename))

## NORMALIZAR LAS IMAGENES
imagenesBD, y_imagenesBD = crop_and_normalize_images(image_paths, target_size)



## CARGAR LAS RUTAS DE CADA IMAGEN DE CONSULTA
orig_dir = r'/content/drive/MyDrive/GIBD/E.C de imágenes a color/E.C archivos/consultas'

image_paths = []

for filename in os.listdir(orig_dir):
    image_paths.append(os.path.join(orig_dir, filename))


## NORMALIZAR LAS IMAGENES

consultas, y_consultas = crop_and_normalize_images(image_paths, target_size)

```


*Salida:*
```text
freiburg
TIENE ALPHA...
19 80 0 99
atletico-madrid
TIENE ALPHA...
23 76 14 84
athletic
TIENE ALPHA...
21 78 17 82
estac-troyes
TIENE ALPHA...
9 90 0 98
hamburg
TIENE ALPHA...
0 99 15 84
montpellier-herault
TIENE ALPHA...
1 99 1 99
stade-de-reims
TIENE ALPHA...
22 78 3 97
bremen
TIENE ALPHA...
17 82 1 98
fc-lorient
TIENE ALPHA...
16 85 0 99
clermont-foot-63
TIENE ALPHA...
8 91 1 98
barcelona
TIENE ALPHA...
17 82 17 82
redbull-leipzig
TIENE ALPHA...
3 96 22 77
schalke
TIENE ALPHA...
3 96 3 96
nuremberg
TIENE ALPHA...
2 98 2 98
fc-nantes
TIENE ALPHA...
13 86 1 99
stade-rennais-fc
TIENE ALPHA...
10 89 1 98
deportivo-alavez
TIENE ALPHA...
16 82 18 84
moenchengladbach
TIENE ALPHA...
20 80 1 98
elche
TIENE ALPHA...
22 77 15 83
mainz
TIENE ALPHA...
2 98 1 98
fc-girondins-de-bordeaux
TIENE ALPHA...
9 89 0 99
cadiz
TIENE ALPHA...
25 74 11 87
losc-lille
TIENE ALPHA...
1 98 4 96
olympique-de-marseille
TIENE ALPHA...
12 87 2 99
as-saint-etienne
TIENE ALPHA...
14 85 2 98
rc-strasbourg-alsace
TIENE ALPHA...
1 98 1 98
espanyol
TIENE ALPHA...
23 76 9 85
stuttgart
TIENE ALPHA...
5 94 0 99
stade-brestois-29
TIENE ALPHA...
9 90 0 99
rc-lens
TIENE ALPHA...
14 86 0 98
fc-metz
TIENE ALPHA...
15 82 2 98
hertha-bsc-berlin
TIENE ALPHA...
16 88 17 84
frankfurt
TIENE ALPHA...
0 99 0 99
wolfsburg
TIENE ALPHA...
0 97 0 98
dusseldorf
TIENE ALPHA...
0 99 0 99
dortmund
TIENE ALPHA...
0 99 0 99
augsburg
TIENE ALPHA...
13 86 2 97
as-monaco
TIENE ALPHA...
22 78 0 99
celta
TIENE ALPHA...
27 71 9 88
getafe
TIENE ALPHA...
16 81 15 83
leverkusen
TIENE ALPHA...
0 99 12 87
angers-sco
TIENE ALPHA...
8 89 0 98
paris-saint-germain
TIENE ALPHA...
2 97 2 97
hoffenheim
TIENE ALPHA...
5 94 0 99
furth
TIENE ALPHA...
19 80 3 96
hannover
TIENE ALPHA...
0 98 6 93
ogc-nice
TIENE ALPHA...
10 89 0 98
bayern
TIENE ALPHA...
2 97 2 97
olympique-lyonnais
TIENE ALPHA...
8 91 1 98
leeds-united
TIENE ALPHA...
10 90 0 99
osasuna
TIENE ALPHA...
22 77 14 90
chelsea
TIENE ALPHA...
0 99 0 99
norwich-city
TIENE ALPHA...
5 93 1 99
real-sociedad
TIENE ALPHA...
18 82 17 89
southampton
TIENE ALPHA...
6 92 0 99
mallorca
TIENE ALPHA...
21 77 14 89
atalanta
TIENE ALPHA...
24 75 8 91
hellas-verona
TIENE ALPHA...
6 94 5 94
newcastle-united
TIENE ALPHA...
4 96 4 96
real-madrid
TIENE ALPHA...
22 76 8 83
levante
TIENE ALPHA...
20 79 14 86
napoli
TIENE ALPHA...
12 87 12 87
rayo-vallecano
TIENE ALPHA...
17 82 21 77
lazio
TIENE ALPHA...
1 98 16 82
granada
TIENE ALPHA...
30 69 6 94
udinese
TIENE ALPHA...
5 94 6 93
crystal-palace
TIENE ALPHA...
12 87 1 96
torino
TIENE ALPHA...
15 85 8 92
watford
TIENE ALPHA...
7 90 3 97
salernitana
TIENE ALPHA...
8 91 8 91
liverpool
TIENE ALPHA...
13 86 0 99
west-ham-united
TIENE ALPHA...
5 94 0 99
arsenal
TIENE ALPHA...
9 90 0 99
sassuolo
TIENE ALPHA...
12 87 9 91
bologna
TIENE ALPHA...
18 81 2 97
venezia
TIENE ALPHA...
11 88 11 88
spezia
TIENE ALPHA...
8 91 8 91
genoa
TIENE ALPHA...
22 77 10 89
roma
TIENE ALPHA...
18 81 9 91
everton
TIENE ALPHA...
1 97 1 99
valencia
TIENE ALPHA...
22 77 13 87
real-betis
TIENE ALPHA...
8 91 14 83
manchester-united
TIENE ALPHA...
1 98 0 99
sevilla
TIENE ALPHA...
22 77 15 83
brighton
TIENE ALPHA...
0 99 0 99
aston-villa
TIENE ALPHA...
14 84 0 98
inter
TIENE ALPHA...
9 90 9 90
fiorentina
TIENE ALPHA...
20 78 6 92
tottenham-hotspur
TIENE ALPHA...
26 72 2 97
brentford
TIENE ALPHA...
0 99 0 99
burnley
TIENE ALPHA...
7 92 0 99
sampdoria
TIENE ALPHA...
17 82 9 92
cagliari
TIENE ALPHA...
9 90 1 98
ac-milan
TIENE ALPHA...
23 81 2 97
empoli
TIENE ALPHA...
16 82 4 94
leicester-city
TIENE ALPHA...
0 99 0 99
villarreal
TIENE ALPHA...
19 79 11 86
manchester-city
TIENE ALPHA...
0 99 0 99
juventus
TIENE ALPHA...
22 77 6 93
the-strongest
TIENE ALPHA...
14 166 3 175
mineiro
TIENE ALPHA...
33 146 5 174
botafogo
TIENE ALPHA...
15 164 3 175
corinthians
TIENE ALPHA...
23 156 3 173
cruzeiro
TIENE ALPHA...
7 173 5 172
gremio
TIENE ALPHA...
18 161 5 176
fluminense
TIENE ALPHA...
15 163 3 173
flamengo
TIENE ALPHA...
23 158 5 172
internacional
TIENE ALPHA...
4 175 3 176
palmeiras
TIENE ALPHA...
7 173 5 172
fortaleza
TIENE ALPHA...
7 172 3 174
santos
TIENE ALPHA...
16 163 4 174
millonarios
TIENE ALPHA...
8 171 4 174
deportivo-cali
TIENE ALPHA...
6 175 4 171
atletico-nacional
TIENE ALPHA...
31 148 4 173
luton
TIENE ALPHA...
7 174 4 171
velez
TIENE ALPHA...
23 157 3 174
san-lorenzo
TIENE ALPHA...
14 165 5 175
central
TIENE ALPHA...
18 163 5 172
racing
TIENE ALPHA...
19 160 3 174
independiente
TIENE ALPHA...
7 172 4 175
newells
TIENE ALPHA...
20 159 3 175
lanus
TIENE ALPHA...
6 173 4 172
platense
TIENE ALPHA...
23 156 4 174
gimnasia
TIENE ALPHA...
34 145 3 171
estudiantes
TIENE ALPHA...
6 175 14 162
boca
TIENE ALPHA...
17 161 3 173
belgrano
TIENE ALPHA...
4 175 3 176
argentinos
TIENE ALPHA...
19 159 5 176
sheffield
TIENE ALPHA...
4 175 3 175
estrella-roja
TIENE ALPHA...
4 175 3 176
maccabi-haifa
TIENE ALPHA...
25 156 3 172
parma
TIENE ALPHA...
20 159 4 173
universitarios
TIENE ALPHA...
4 175 3 176
sporting-cristal
TIENE ALPHA...
39 139 3 172
porto
TIENE ALPHA...
27 152 4 172
ajax
TIENE ALPHA...
4 175 3 176
deportivo-alaves
TIENE ALPHA...
7 174 5 172
real-murcia
TIENE ALPHA...
43 135 3 172
galatasaray
TIENE ALPHA...
35 143 3 172
galaxy
TIENE ALPHA...
22 157 4 175
nacional
TIENE ALPHA...
7 173 5 176
peñarol
TIENE ALPHA...
26 153 4 174
malaga
TIENE ALPHA...
14 165 3 173
seattle-sounders
TIENE ALPHA...
5 175 63 116
ferro
TIENE ALPHA...
15 164 3 175
brujas
TIENE ALPHA...
26 153 3 173
colo-colo
TIENE ALPHA...
9 172 4 172
america
TIENE ALPHA...
16 162 3 173
blackburn
TIENE ALPHA...
7 172 3 176
huracan
TIENE ALPHA...
38 141 3 176
(224, 224, 3)
aston-villa
TIENE ALPHA...
14 85 0 99
augsburg
TIENE ALPHA...
13 86 2 97
bremen
TIENE ALPHA...
0 68 0 99
cadiz
TIENE ALPHA...
0 65 0 98
celta
TIENE ALPHA...
0 65 0 99
dusseldorf
TIENE ALPHA...
6 93 6 93
empoli
TIENE ALPHA...
15 84 2 97
clermont-foot-63
TIENE ALPHA...
0 83 0 99
deportivo-alavez
TIENE ALPHA...
0 98 0 99
udinese
TIENE ALPHA...
0 99 1 97
paris-saint-germain
TIENE ALPHA...
3 96 3 96
elche
TIENE ALPHA...
0 81 0 99
manchester-united
TIENE ALPHA...
2 97 2 97
ogc-nice
TIENE ALPHA...
10 89 0 99
levante
TIENE ALPHA...
0 79 0 98
spezia
TIENE ALPHA...
0 99 0 99
wolfsburg
TIENE ALPHA...
2 97 2 97
stuttgart
TIENE ALPHA...
6 93 2 97
barcelona
TIENE ALPHA...
9 90 9 90
inter
TIENE ALPHA...
5 94 5 94
roma
TIENE ALPHA...
16 84 5 94
villarreal
TIENE ALPHA...
10 89 0 98
crystal-palace
TIENE ALPHA...
11 89 0 99
fiorentina
TIENE ALPHA...
3 73 14 85
moenchengladbach
TIENE ALPHA...
19 80 1 98
napoli
TIENE ALPHA...
0 99 0 99
chelsea
TIENE ALPHA...
2 97 2 97
cagliari
TIENE ALPHA...
0 82 0 99
sampdoria
TIENE ALPHA...
12 87 2 97
real-betis
TIENE ALPHA...
1 98 0 83
olympique-lyonnais
TIENE ALPHA...
9 90 2 97
juventus
TIENE ALPHA...
18 78 3 98
getafe
TIENE ALPHA...
6 93 4 95
brentford
TIENE ALPHA...
5 94 6 94
watford
TIENE ALPHA...
0 88 0 99
bologna
TIENE ALPHA...
19 80 2 97
norwich-city
TIENE ALPHA...
0 87 0 99
granada
TIENE ALPHA...
31 68 0 99
valencia
TIENE ALPHA...
20 79 10 89
mainz
TIENE ALPHA...
2 97 3 92
bayern
TIENE ALPHA...
2 98 2 97
fc-lorient
TIENE ALPHA...
1 68 1 98
west-ham-united
TIENE ALPHA...
0 89 0 98
genoa
TIENE ALPHA...
16 83 2 97
hellas-verona
TIENE ALPHA...
19 80 18 80
liverpool
TIENE ALPHA...
13 86 1 99
southampton
TIENE ALPHA...
8 91 2 97
burnley
TIENE ALPHA...
9 90 2 97
hoffenheim
TIENE ALPHA...
7 93 2 97
dortmund
TIENE ALPHA...
2 97 2 97
freiburg
TIENE ALPHA...
0 69 0 99
espanyol
TIENE ALPHA...
17 82 3 95
redbull-leipzig
TIENE ALPHA...
1 98 24 76
schalke
TIENE ALPHA...
0 99 0 99
ac-milan
TIENE ALPHA...
0 1199 0 1880
arsenal
TIENE ALPHA...
1 1198 0 1409
real-madrid
TIENE ALPHA...
80 416 16 482
real-sociedad
TIENE ALPHA...
9 338 6 334
manchester-city
TIENE ALPHA...
11 88 12 88
nacional
0 423 0 429
atlanta
TIENE ALPHA...
59 964 0 1023
peñarol
TIENE ALPHA...
1 336 0 449
gimnasia
TIENE ALPHA...
101 398 22 477
atletico-tucuman
TIENE ALPHA...
12 1191 14 1390
cerro-largo
TIENE ALPHA...
7 192 7 192
argentinos-juniors
TIENE ALPHA...
65 434 24 476
wanders
TIENE ALPHA...
27 172 9 191
colon
TIENE ALPHA...
0 785 0 815
danubio
0 344 0 446
river-plate
TIENE ALPHA...
0 99 0 103
rampla-juniors
TIENE ALPHA...
79 428 8 500
bella-vista
TIENE ALPHA...
0 561 0 599
central-español
TIENE ALPHA...
44 555 0 599
club-atletico-progreso
TIENE ALPHA...
60 452 13 499
defensor-sporting
TIENE ALPHA...
49 303 22 330
aldosivi
TIENE ALPHA...
3 198 1 199
arsenal-sarandi
TIENE ALPHA...
71 427 10 489
boca
TIENE ALPHA...
30 994 26 1162
san-lorenzo
TIENE ALPHA...
52 447 23 477
independiente
TIENE ALPHA...
7 1191 7 1255
(224, 224, 3)
```


**[Celda 7 - Código]**
```python
def crop_and_normalize_images(image_paths, target_size):
    normalized_images = []
    etiquetas = []

    for path in image_paths:
        image = Image.open(path)

        # Obtener la etiqueta de la imagen
        filename = os.path.basename(path)
        nombre = filename[:filename.find(".")]
        etiquetas.append(nombre)
        print(nombre)

        # Verificar si la imagen tiene un canal alfa
        has_alpha = 'A' in image.mode

        # Convertir la imagen en un array numpy
        image_array = np.array(image)

        # Si tiene canal alfa, eliminarlo y convertir a RGB
        if has_alpha:
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

    # Redimensionar todas las imágenes normalizadas al mismo tamaño
    max_shape = max([image.shape for image in normalized_images])
    normalized_images_resized = [np.pad(image, [(0, max_shape[0] - image.shape[0]), (0, max_shape[1] - image.shape[1]), (0, 0)], mode='constant') for image in normalized_images]

    print(normalized_images_resized[0].shape)

    return np.array(normalized_images_resized), np.array(etiquetas)


target_size = (224, 224)  # El tamaño objetivo después de recortar y redimensionar

## CARGAR LAS RUTAS DE CADA IMAGEN DE LA BASE DE DATOS
orig_dir = r'/content/drive/MyDrive/GIBD/E.C de imágenes a color/E.C archivos/imagenesBD'

image_paths = []

for filename in os.listdir(orig_dir):
    image_paths.append(os.path.join(orig_dir, filename))

## NORMALIZAR LAS IMAGENES
imagenesBD, y_imagenesBD = crop_and_normalize_images(image_paths, target_size)



## CARGAR LAS RUTAS DE CADA IMAGEN DE CONSULTA
orig_dir = r'/content/drive/MyDrive/GIBD/E.C de imágenes a color/E.C archivos/consultas'

image_paths = []

for filename in os.listdir(orig_dir):
    image_paths.append(os.path.join(orig_dir, filename))


## NORMALIZAR LAS IMAGENES

consultas, y_consultas = crop_and_normalize_images(image_paths, target_size)


```


*Salida:*
```text
freiburg
atletico-madrid
athletic
estac-troyes
hamburg
montpellier-herault
stade-de-reims
bremen
fc-lorient
clermont-foot-63
barcelona
redbull-leipzig
schalke
nuremberg
fc-nantes
stade-rennais-fc
deportivo-alavez
moenchengladbach
elche
mainz
fc-girondins-de-bordeaux
cadiz
losc-lille
olympique-de-marseille
as-saint-etienne
rc-strasbourg-alsace
espanyol
stuttgart
stade-brestois-29
rc-lens
fc-metz
hertha-bsc-berlin
frankfurt
wolfsburg
dusseldorf
dortmund
augsburg
as-monaco
celta
getafe
leverkusen
angers-sco
paris-saint-germain
hoffenheim
furth
hannover
ogc-nice
bayern
olympique-lyonnais
leeds-united
osasuna
chelsea
norwich-city
real-sociedad
southampton
mallorca
atalanta
hellas-verona
newcastle-united
real-madrid
levante
napoli
rayo-vallecano
lazio
granada
udinese
crystal-palace
torino
watford
salernitana
liverpool
west-ham-united
arsenal
sassuolo
bologna
venezia
spezia
genoa
roma
everton
valencia
real-betis
manchester-united
sevilla
brighton
aston-villa
inter
fiorentina
tottenham-hotspur
brentford
burnley
sampdoria
cagliari
ac-milan
empoli
leicester-city
villarreal
manchester-city
juventus
the-strongest
mineiro
botafogo
corinthians
cruzeiro
gremio
fluminense
flamengo
internacional
palmeiras
fortaleza
santos
millonarios
deportivo-cali
atletico-nacional
luton
velez
san-lorenzo
central
racing
independiente
newells
lanus
platense
gimnasia
estudiantes
boca
belgrano
argentinos
sheffield
estrella-roja
maccabi-haifa
parma
universitarios
sporting-cristal
porto
ajax
deportivo-alaves
real-murcia
galatasaray
galaxy
nacional
peñarol
malaga
seattle-sounders
ferro
brujas
colo-colo
america
blackburn
huracan
(224, 224, 4)
aston-villa
augsburg
bremen
cadiz
celta
dusseldorf
empoli
clermont-foot-63
deportivo-alavez
udinese
paris-saint-germain
elche
manchester-united
ogc-nice
levante
spezia
wolfsburg
stuttgart
barcelona
inter
roma
villarreal
crystal-palace
fiorentina
moenchengladbach
napoli
chelsea
cagliari
sampdoria
real-betis
olympique-lyonnais
juventus
getafe
brentford
watford
bologna
norwich-city
granada
valencia
mainz
bayern
fc-lorient
west-ham-united
genoa
hellas-verona
liverpool
southampton
burnley
hoffenheim
dortmund
freiburg
espanyol
redbull-leipzig
schalke
ac-milan
arsenal
real-madrid
real-sociedad
manchester-city
nacional
atlanta
peñarol
gimnasia
atletico-tucuman
cerro-largo
argentinos-juniors
wanders
colon
danubio
river-plate
rampla-juniors
bella-vista
central-español
club-atletico-progreso
defensor-sporting
aldosivi
arsenal-sarandi
boca
san-lorenzo
independiente
```


**[Celda 8 - Código]**
```python
#No ejecutamos

# CARGAR CONSULTAS DIBUJADAS A MANO
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


directorio = r'/content/drive/MyDrive/GIBD/E.C de imágenes a color/E.C archivos/imagenesBD'
contenido = os.listdir(directorio)


consultas = []
y_consultas = []
for fichero in contenido:
    if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.png'):
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
[]
```


**[Celda 9 - Código]**
```python
def nnk(actuales,nuevo,k=1):
    p = 0
    while (p<len(actuales) and (nuevo[0]>=actuales[p][0])):
        p += 1
    actuales.insert(p, nuevo)
    while len(actuales)>k:
        actuales.pop(len(actuales)-1)
```


**[Celda 10 - Código]**
```python
def nnk2etiquetas(nnklist, y_BD):
    l = []
    for par in nnklist:
        nuevo = [par[0], y_BD[par[1]]]
        l.append(nuevo)
    return l

```


**[Celda 11 - Código]**
```python
#CARGA DE MODELO DENSENET
from keras.applications.densenet import DenseNet121
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Dropout, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_densenet(input_shape=(224, 224, 3), n_classes=10):
    input_layer = Input(shape=input_shape)
    densenet121 = DenseNet121(include_top=False, weights='imagenet', input_tensor=input_layer)
    x = GlobalAveragePooling2D()(densenet121.output)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(input_layer, x)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=3e-4), metrics=['accuracy'])
    return model

densenet = build_densenet(n_classes=128)
```


**[Celda 12 - Código]**
```python
densenet.summary()
```


*Salida:*
```text
Model: "model_1"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_2 (InputLayer)           [(None, 224, 224, 3  0           []                               
                                )]                                                                
                                                                                                  
 zero_padding2d_2 (ZeroPadding2  (None, 230, 230, 3)  0          ['input_2[0][0]']                
 D)                                                                                               
                                                                                                  
 conv1/conv (Conv2D)            (None, 112, 112, 64  9408        ['zero_padding2d_2[0][0]']       
                                )                                                                 
                                                                                                  
 conv1/bn (BatchNormalization)  (None, 112, 112, 64  256         ['conv1/conv[0][0]']             
                                )                                                                 
                                                                                                  
 conv1/relu (Activation)        (None, 112, 112, 64  0           ['conv1/bn[0][0]']               
                                )                                                                 
                                                                                                  
 zero_padding2d_3 (ZeroPadding2  (None, 114, 114, 64  0          ['conv1/relu[0][0]']             
 D)                             )                                                                 
                                                                                                  
 pool1 (MaxPooling2D)           (None, 56, 56, 64)   0           ['zero_padding2d_3[0][0]']       
                                                                                                  
 conv2_block1_0_bn (BatchNormal  (None, 56, 56, 64)  256         ['pool1[0][0]']                  
 ization)                                                                                         
                                                                                                  
 conv2_block1_0_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block1_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv2_block1_1_conv (Conv2D)   (None, 56, 56, 128)  8192        ['conv2_block1_0_relu[0][0]']    
                                                                                                  
 conv2_block1_1_bn (BatchNormal  (None, 56, 56, 128)  512        ['conv2_block1_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block1_1_relu (Activatio  (None, 56, 56, 128)  0          ['conv2_block1_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv2_block1_2_conv (Conv2D)   (None, 56, 56, 32)   36864       ['conv2_block1_1_relu[0][0]']    
                                                                                                  
 conv2_block1_concat (Concatena  (None, 56, 56, 96)  0           ['pool1[0][0]',                  
 te)                                                              'conv2_block1_2_conv[0][0]']    
                                                                                                  
 conv2_block2_0_bn (BatchNormal  (None, 56, 56, 96)  384         ['conv2_block1_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block2_0_relu (Activatio  (None, 56, 56, 96)  0           ['conv2_block2_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv2_block2_1_conv (Conv2D)   (None, 56, 56, 128)  12288       ['conv2_block2_0_relu[0][0]']    
                                                                                                  
 conv2_block2_1_bn (BatchNormal  (None, 56, 56, 128)  512        ['conv2_block2_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block2_1_relu (Activatio  (None, 56, 56, 128)  0          ['conv2_block2_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv2_block2_2_conv (Conv2D)   (None, 56, 56, 32)   36864       ['conv2_block2_1_relu[0][0]']    
                                                                                                  
 conv2_block2_concat (Concatena  (None, 56, 56, 128)  0          ['conv2_block1_concat[0][0]',    
 te)                                                              'conv2_block2_2_conv[0][0]']    
                                                                                                  
 conv2_block3_0_bn (BatchNormal  (None, 56, 56, 128)  512        ['conv2_block2_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block3_0_relu (Activatio  (None, 56, 56, 128)  0          ['conv2_block3_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv2_block3_1_conv (Conv2D)   (None, 56, 56, 128)  16384       ['conv2_block3_0_relu[0][0]']    
                                                                                                  
 conv2_block3_1_bn (BatchNormal  (None, 56, 56, 128)  512        ['conv2_block3_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block3_1_relu (Activatio  (None, 56, 56, 128)  0          ['conv2_block3_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv2_block3_2_conv (Conv2D)   (None, 56, 56, 32)   36864       ['conv2_block3_1_relu[0][0]']    
                                                                                                  
 conv2_block3_concat (Concatena  (None, 56, 56, 160)  0          ['conv2_block2_concat[0][0]',    
 te)                                                              'conv2_block3_2_conv[0][0]']    
                                                                                                  
 conv2_block4_0_bn (BatchNormal  (None, 56, 56, 160)  640        ['conv2_block3_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block4_0_relu (Activatio  (None, 56, 56, 160)  0          ['conv2_block4_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv2_block4_1_conv (Conv2D)   (None, 56, 56, 128)  20480       ['conv2_block4_0_relu[0][0]']    
                                                                                                  
 conv2_block4_1_bn (BatchNormal  (None, 56, 56, 128)  512        ['conv2_block4_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block4_1_relu (Activatio  (None, 56, 56, 128)  0          ['conv2_block4_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv2_block4_2_conv (Conv2D)   (None, 56, 56, 32)   36864       ['conv2_block4_1_relu[0][0]']    
                                                                                                  
 conv2_block4_concat (Concatena  (None, 56, 56, 192)  0          ['conv2_block3_concat[0][0]',    
 te)                                                              'conv2_block4_2_conv[0][0]']    
                                                                                                  
 conv2_block5_0_bn (BatchNormal  (None, 56, 56, 192)  768        ['conv2_block4_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block5_0_relu (Activatio  (None, 56, 56, 192)  0          ['conv2_block5_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv2_block5_1_conv (Conv2D)   (None, 56, 56, 128)  24576       ['conv2_block5_0_relu[0][0]']    
                                                                                                  
 conv2_block5_1_bn (BatchNormal  (None, 56, 56, 128)  512        ['conv2_block5_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block5_1_relu (Activatio  (None, 56, 56, 128)  0          ['conv2_block5_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv2_block5_2_conv (Conv2D)   (None, 56, 56, 32)   36864       ['conv2_block5_1_relu[0][0]']    
                                                                                                  
 conv2_block5_concat (Concatena  (None, 56, 56, 224)  0          ['conv2_block4_concat[0][0]',    
 te)                                                              'conv2_block5_2_conv[0][0]']    
                                                                                                  
 conv2_block6_0_bn (BatchNormal  (None, 56, 56, 224)  896        ['conv2_block5_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block6_0_relu (Activatio  (None, 56, 56, 224)  0          ['conv2_block6_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv2_block6_1_conv (Conv2D)   (None, 56, 56, 128)  28672       ['conv2_block6_0_relu[0][0]']    
                                                                                                  
 conv2_block6_1_bn (BatchNormal  (None, 56, 56, 128)  512        ['conv2_block6_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block6_1_relu (Activatio  (None, 56, 56, 128)  0          ['conv2_block6_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv2_block6_2_conv (Conv2D)   (None, 56, 56, 32)   36864       ['conv2_block6_1_relu[0][0]']    
                                                                                                  
 conv2_block6_concat (Concatena  (None, 56, 56, 256)  0          ['conv2_block5_concat[0][0]',    
 te)                                                              'conv2_block6_2_conv[0][0]']    
                                                                                                  
 pool2_bn (BatchNormalization)  (None, 56, 56, 256)  1024        ['conv2_block6_concat[0][0]']    
                                                                                                  
 pool2_relu (Activation)        (None, 56, 56, 256)  0           ['pool2_bn[0][0]']               
                                                                                                  
 pool2_conv (Conv2D)            (None, 56, 56, 128)  32768       ['pool2_relu[0][0]']             
                                                                                                  
 pool2_pool (AveragePooling2D)  (None, 28, 28, 128)  0           ['pool2_conv[0][0]']             
                                                                                                  
 conv3_block1_0_bn (BatchNormal  (None, 28, 28, 128)  512        ['pool2_pool[0][0]']             
 ization)                                                                                         
                                                                                                  
 conv3_block1_0_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block1_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block1_1_conv (Conv2D)   (None, 28, 28, 128)  16384       ['conv3_block1_0_relu[0][0]']    
                                                                                                  
 conv3_block1_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block1_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block1_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block1_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block1_2_conv (Conv2D)   (None, 28, 28, 32)   36864       ['conv3_block1_1_relu[0][0]']    
                                                                                                  
 conv3_block1_concat (Concatena  (None, 28, 28, 160)  0          ['pool2_pool[0][0]',             
 te)                                                              'conv3_block1_2_conv[0][0]']    
                                                                                                  
 conv3_block2_0_bn (BatchNormal  (None, 28, 28, 160)  640        ['conv3_block1_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block2_0_relu (Activatio  (None, 28, 28, 160)  0          ['conv3_block2_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block2_1_conv (Conv2D)   (None, 28, 28, 128)  20480       ['conv3_block2_0_relu[0][0]']    
                                                                                                  
 conv3_block2_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block2_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block2_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block2_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block2_2_conv (Conv2D)   (None, 28, 28, 32)   36864       ['conv3_block2_1_relu[0][0]']    
                                                                                                  
 conv3_block2_concat (Concatena  (None, 28, 28, 192)  0          ['conv3_block1_concat[0][0]',    
 te)                                                              'conv3_block2_2_conv[0][0]']    
                                                                                                  
 conv3_block3_0_bn (BatchNormal  (None, 28, 28, 192)  768        ['conv3_block2_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block3_0_relu (Activatio  (None, 28, 28, 192)  0          ['conv3_block3_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block3_1_conv (Conv2D)   (None, 28, 28, 128)  24576       ['conv3_block3_0_relu[0][0]']    
                                                                                                  
 conv3_block3_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block3_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block3_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block3_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block3_2_conv (Conv2D)   (None, 28, 28, 32)   36864       ['conv3_block3_1_relu[0][0]']    
                                                                                                  
 conv3_block3_concat (Concatena  (None, 28, 28, 224)  0          ['conv3_block2_concat[0][0]',    
 te)                                                              'conv3_block3_2_conv[0][0]']    
                                                                                                  
 conv3_block4_0_bn (BatchNormal  (None, 28, 28, 224)  896        ['conv3_block3_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block4_0_relu (Activatio  (None, 28, 28, 224)  0          ['conv3_block4_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block4_1_conv (Conv2D)   (None, 28, 28, 128)  28672       ['conv3_block4_0_relu[0][0]']    
                                                                                                  
 conv3_block4_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block4_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block4_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block4_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block4_2_conv (Conv2D)   (None, 28, 28, 32)   36864       ['conv3_block4_1_relu[0][0]']    
                                                                                                  
 conv3_block4_concat (Concatena  (None, 28, 28, 256)  0          ['conv3_block3_concat[0][0]',    
 te)                                                              'conv3_block4_2_conv[0][0]']    
                                                                                                  
 conv3_block5_0_bn (BatchNormal  (None, 28, 28, 256)  1024       ['conv3_block4_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block5_0_relu (Activatio  (None, 28, 28, 256)  0          ['conv3_block5_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block5_1_conv (Conv2D)   (None, 28, 28, 128)  32768       ['conv3_block5_0_relu[0][0]']    
                                                                                                  
 conv3_block5_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block5_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block5_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block5_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block5_2_conv (Conv2D)   (None, 28, 28, 32)   36864       ['conv3_block5_1_relu[0][0]']    
                                                                                                  
 conv3_block5_concat (Concatena  (None, 28, 28, 288)  0          ['conv3_block4_concat[0][0]',    
 te)                                                              'conv3_block5_2_conv[0][0]']    
                                                                                                  
 conv3_block6_0_bn (BatchNormal  (None, 28, 28, 288)  1152       ['conv3_block5_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block6_0_relu (Activatio  (None, 28, 28, 288)  0          ['conv3_block6_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block6_1_conv (Conv2D)   (None, 28, 28, 128)  36864       ['conv3_block6_0_relu[0][0]']    
                                                                                                  
 conv3_block6_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block6_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block6_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block6_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block6_2_conv (Conv2D)   (None, 28, 28, 32)   36864       ['conv3_block6_1_relu[0][0]']    
                                                                                                  
 conv3_block6_concat (Concatena  (None, 28, 28, 320)  0          ['conv3_block5_concat[0][0]',    
 te)                                                              'conv3_block6_2_conv[0][0]']    
                                                                                                  
 conv3_block7_0_bn (BatchNormal  (None, 28, 28, 320)  1280       ['conv3_block6_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block7_0_relu (Activatio  (None, 28, 28, 320)  0          ['conv3_block7_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block7_1_conv (Conv2D)   (None, 28, 28, 128)  40960       ['conv3_block7_0_relu[0][0]']    
                                                                                                  
 conv3_block7_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block7_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block7_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block7_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block7_2_conv (Conv2D)   (None, 28, 28, 32)   36864       ['conv3_block7_1_relu[0][0]']    
                                                                                                  
 conv3_block7_concat (Concatena  (None, 28, 28, 352)  0          ['conv3_block6_concat[0][0]',    
 te)                                                              'conv3_block7_2_conv[0][0]']    
                                                                                                  
 conv3_block8_0_bn (BatchNormal  (None, 28, 28, 352)  1408       ['conv3_block7_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block8_0_relu (Activatio  (None, 28, 28, 352)  0          ['conv3_block8_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block8_1_conv (Conv2D)   (None, 28, 28, 128)  45056       ['conv3_block8_0_relu[0][0]']    
                                                                                                  
 conv3_block8_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block8_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block8_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block8_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block8_2_conv (Conv2D)   (None, 28, 28, 32)   36864       ['conv3_block8_1_relu[0][0]']    
                                                                                                  
 conv3_block8_concat (Concatena  (None, 28, 28, 384)  0          ['conv3_block7_concat[0][0]',    
 te)                                                              'conv3_block8_2_conv[0][0]']    
                                                                                                  
 conv3_block9_0_bn (BatchNormal  (None, 28, 28, 384)  1536       ['conv3_block8_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block9_0_relu (Activatio  (None, 28, 28, 384)  0          ['conv3_block9_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block9_1_conv (Conv2D)   (None, 28, 28, 128)  49152       ['conv3_block9_0_relu[0][0]']    
                                                                                                  
 conv3_block9_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block9_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block9_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block9_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block9_2_conv (Conv2D)   (None, 28, 28, 32)   36864       ['conv3_block9_1_relu[0][0]']    
                                                                                                  
 conv3_block9_concat (Concatena  (None, 28, 28, 416)  0          ['conv3_block8_concat[0][0]',    
 te)                                                              'conv3_block9_2_conv[0][0]']    
                                                                                                  
 conv3_block10_0_bn (BatchNorma  (None, 28, 28, 416)  1664       ['conv3_block9_concat[0][0]']    
 lization)                                                                                        
                                                                                                  
 conv3_block10_0_relu (Activati  (None, 28, 28, 416)  0          ['conv3_block10_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv3_block10_1_conv (Conv2D)  (None, 28, 28, 128)  53248       ['conv3_block10_0_relu[0][0]']   
                                                                                                  
 conv3_block10_1_bn (BatchNorma  (None, 28, 28, 128)  512        ['conv3_block10_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv3_block10_1_relu (Activati  (None, 28, 28, 128)  0          ['conv3_block10_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv3_block10_2_conv (Conv2D)  (None, 28, 28, 32)   36864       ['conv3_block10_1_relu[0][0]']   
                                                                                                  
 conv3_block10_concat (Concaten  (None, 28, 28, 448)  0          ['conv3_block9_concat[0][0]',    
 ate)                                                             'conv3_block10_2_conv[0][0]']   
                                                                                                  
 conv3_block11_0_bn (BatchNorma  (None, 28, 28, 448)  1792       ['conv3_block10_concat[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv3_block11_0_relu (Activati  (None, 28, 28, 448)  0          ['conv3_block11_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv3_block11_1_conv (Conv2D)  (None, 28, 28, 128)  57344       ['conv3_block11_0_relu[0][0]']   
                                                                                                  
 conv3_block11_1_bn (BatchNorma  (None, 28, 28, 128)  512        ['conv3_block11_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv3_block11_1_relu (Activati  (None, 28, 28, 128)  0          ['conv3_block11_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv3_block11_2_conv (Conv2D)  (None, 28, 28, 32)   36864       ['conv3_block11_1_relu[0][0]']   
                                                                                                  
 conv3_block11_concat (Concaten  (None, 28, 28, 480)  0          ['conv3_block10_concat[0][0]',   
 ate)                                                             'conv3_block11_2_conv[0][0]']   
                                                                                                  
 conv3_block12_0_bn (BatchNorma  (None, 28, 28, 480)  1920       ['conv3_block11_concat[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv3_block12_0_relu (Activati  (None, 28, 28, 480)  0          ['conv3_block12_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv3_block12_1_conv (Conv2D)  (None, 28, 28, 128)  61440       ['conv3_block12_0_relu[0][0]']   
                                                                                                  
 conv3_block12_1_bn (BatchNorma  (None, 28, 28, 128)  512        ['conv3_block12_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv3_block12_1_relu (Activati  (None, 28, 28, 128)  0          ['conv3_block12_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv3_block12_2_conv (Conv2D)  (None, 28, 28, 32)   36864       ['conv3_block12_1_relu[0][0]']   
                                                                                                  
 conv3_block12_concat (Concaten  (None, 28, 28, 512)  0          ['conv3_block11_concat[0][0]',   
 ate)                                                             'conv3_block12_2_conv[0][0]']   
                                                                                                  
 pool3_bn (BatchNormalization)  (None, 28, 28, 512)  2048        ['conv3_block12_concat[0][0]']   
                                                                                                  
 pool3_relu (Activation)        (None, 28, 28, 512)  0           ['pool3_bn[0][0]']               
                                                                                                  
 pool3_conv (Conv2D)            (None, 28, 28, 256)  131072      ['pool3_relu[0][0]']             
                                                                                                  
 pool3_pool (AveragePooling2D)  (None, 14, 14, 256)  0           ['pool3_conv[0][0]']             
                                                                                                  
 conv4_block1_0_bn (BatchNormal  (None, 14, 14, 256)  1024       ['pool3_pool[0][0]']             
 ization)                                                                                         
                                                                                                  
 conv4_block1_0_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block1_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block1_1_conv (Conv2D)   (None, 14, 14, 128)  32768       ['conv4_block1_0_relu[0][0]']    
                                                                                                  
 conv4_block1_1_bn (BatchNormal  (None, 14, 14, 128)  512        ['conv4_block1_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block1_1_relu (Activatio  (None, 14, 14, 128)  0          ['conv4_block1_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block1_2_conv (Conv2D)   (None, 14, 14, 32)   36864       ['conv4_block1_1_relu[0][0]']    
                                                                                                  
 conv4_block1_concat (Concatena  (None, 14, 14, 288)  0          ['pool3_pool[0][0]',             
 te)                                                              'conv4_block1_2_conv[0][0]']    
                                                                                                  
 conv4_block2_0_bn (BatchNormal  (None, 14, 14, 288)  1152       ['conv4_block1_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block2_0_relu (Activatio  (None, 14, 14, 288)  0          ['conv4_block2_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block2_1_conv (Conv2D)   (None, 14, 14, 128)  36864       ['conv4_block2_0_relu[0][0]']    
                                                                                                  
 conv4_block2_1_bn (BatchNormal  (None, 14, 14, 128)  512        ['conv4_block2_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block2_1_relu (Activatio  (None, 14, 14, 128)  0          ['conv4_block2_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block2_2_conv (Conv2D)   (None, 14, 14, 32)   36864       ['conv4_block2_1_relu[0][0]']    
                                                                                                  
 conv4_block2_concat (Concatena  (None, 14, 14, 320)  0          ['conv4_block1_concat[0][0]',    
 te)                                                              'conv4_block2_2_conv[0][0]']    
                                                                                                  
 conv4_block3_0_bn (BatchNormal  (None, 14, 14, 320)  1280       ['conv4_block2_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block3_0_relu (Activatio  (None, 14, 14, 320)  0          ['conv4_block3_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block3_1_conv (Conv2D)   (None, 14, 14, 128)  40960       ['conv4_block3_0_relu[0][0]']    
                                                                                                  
 conv4_block3_1_bn (BatchNormal  (None, 14, 14, 128)  512        ['conv4_block3_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block3_1_relu (Activatio  (None, 14, 14, 128)  0          ['conv4_block3_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block3_2_conv (Conv2D)   (None, 14, 14, 32)   36864       ['conv4_block3_1_relu[0][0]']    
                                                                                                  
 conv4_block3_concat (Concatena  (None, 14, 14, 352)  0          ['conv4_block2_concat[0][0]',    
 te)                                                              'conv4_block3_2_conv[0][0]']    
                                                                                                  
 conv4_block4_0_bn (BatchNormal  (None, 14, 14, 352)  1408       ['conv4_block3_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block4_0_relu (Activatio  (None, 14, 14, 352)  0          ['conv4_block4_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block4_1_conv (Conv2D)   (None, 14, 14, 128)  45056       ['conv4_block4_0_relu[0][0]']    
                                                                                                  
 conv4_block4_1_bn (BatchNormal  (None, 14, 14, 128)  512        ['conv4_block4_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block4_1_relu (Activatio  (None, 14, 14, 128)  0          ['conv4_block4_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block4_2_conv (Conv2D)   (None, 14, 14, 32)   36864       ['conv4_block4_1_relu[0][0]']    
                                                                                                  
 conv4_block4_concat (Concatena  (None, 14, 14, 384)  0          ['conv4_block3_concat[0][0]',    
 te)                                                              'conv4_block4_2_conv[0][0]']    
                                                                                                  
 conv4_block5_0_bn (BatchNormal  (None, 14, 14, 384)  1536       ['conv4_block4_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block5_0_relu (Activatio  (None, 14, 14, 384)  0          ['conv4_block5_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block5_1_conv (Conv2D)   (None, 14, 14, 128)  49152       ['conv4_block5_0_relu[0][0]']    
                                                                                                  
 conv4_block5_1_bn (BatchNormal  (None, 14, 14, 128)  512        ['conv4_block5_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block5_1_relu (Activatio  (None, 14, 14, 128)  0          ['conv4_block5_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block5_2_conv (Conv2D)   (None, 14, 14, 32)   36864       ['conv4_block5_1_relu[0][0]']    
                                                                                                  
 conv4_block5_concat (Concatena  (None, 14, 14, 416)  0          ['conv4_block4_concat[0][0]',    
 te)                                                              'conv4_block5_2_conv[0][0]']    
                                                                                                  
 conv4_block6_0_bn (BatchNormal  (None, 14, 14, 416)  1664       ['conv4_block5_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block6_0_relu (Activatio  (None, 14, 14, 416)  0          ['conv4_block6_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block6_1_conv (Conv2D)   (None, 14, 14, 128)  53248       ['conv4_block6_0_relu[0][0]']    
                                                                                                  
 conv4_block6_1_bn (BatchNormal  (None, 14, 14, 128)  512        ['conv4_block6_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block6_1_relu (Activatio  (None, 14, 14, 128)  0          ['conv4_block6_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block6_2_conv (Conv2D)   (None, 14, 14, 32)   36864       ['conv4_block6_1_relu[0][0]']    
                                                                                                  
 conv4_block6_concat (Concatena  (None, 14, 14, 448)  0          ['conv4_block5_concat[0][0]',    
 te)                                                              'conv4_block6_2_conv[0][0]']    
                                                                                                  
 conv4_block7_0_bn (BatchNormal  (None, 14, 14, 448)  1792       ['conv4_block6_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block7_0_relu (Activatio  (None, 14, 14, 448)  0          ['conv4_block7_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block7_1_conv (Conv2D)   (None, 14, 14, 128)  57344       ['conv4_block7_0_relu[0][0]']    
                                                                                                  
 conv4_block7_1_bn (BatchNormal  (None, 14, 14, 128)  512        ['conv4_block7_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block7_1_relu (Activatio  (None, 14, 14, 128)  0          ['conv4_block7_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block7_2_conv (Conv2D)   (None, 14, 14, 32)   36864       ['conv4_block7_1_relu[0][0]']    
                                                                                                  
 conv4_block7_concat (Concatena  (None, 14, 14, 480)  0          ['conv4_block6_concat[0][0]',    
 te)                                                              'conv4_block7_2_conv[0][0]']    
                                                                                                  
 conv4_block8_0_bn (BatchNormal  (None, 14, 14, 480)  1920       ['conv4_block7_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block8_0_relu (Activatio  (None, 14, 14, 480)  0          ['conv4_block8_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block8_1_conv (Conv2D)   (None, 14, 14, 128)  61440       ['conv4_block8_0_relu[0][0]']    
                                                                                                  
 conv4_block8_1_bn (BatchNormal  (None, 14, 14, 128)  512        ['conv4_block8_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block8_1_relu (Activatio  (None, 14, 14, 128)  0          ['conv4_block8_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block8_2_conv (Conv2D)   (None, 14, 14, 32)   36864       ['conv4_block8_1_relu[0][0]']    
                                                                                                  
 conv4_block8_concat (Concatena  (None, 14, 14, 512)  0          ['conv4_block7_concat[0][0]',    
 te)                                                              'conv4_block8_2_conv[0][0]']    
                                                                                                  
 conv4_block9_0_bn (BatchNormal  (None, 14, 14, 512)  2048       ['conv4_block8_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block9_0_relu (Activatio  (None, 14, 14, 512)  0          ['conv4_block9_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block9_1_conv (Conv2D)   (None, 14, 14, 128)  65536       ['conv4_block9_0_relu[0][0]']    
                                                                                                  
 conv4_block9_1_bn (BatchNormal  (None, 14, 14, 128)  512        ['conv4_block9_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block9_1_relu (Activatio  (None, 14, 14, 128)  0          ['conv4_block9_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block9_2_conv (Conv2D)   (None, 14, 14, 32)   36864       ['conv4_block9_1_relu[0][0]']    
                                                                                                  
 conv4_block9_concat (Concatena  (None, 14, 14, 544)  0          ['conv4_block8_concat[0][0]',    
 te)                                                              'conv4_block9_2_conv[0][0]']    
                                                                                                  
 conv4_block10_0_bn (BatchNorma  (None, 14, 14, 544)  2176       ['conv4_block9_concat[0][0]']    
 lization)                                                                                        
                                                                                                  
 conv4_block10_0_relu (Activati  (None, 14, 14, 544)  0          ['conv4_block10_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block10_1_conv (Conv2D)  (None, 14, 14, 128)  69632       ['conv4_block10_0_relu[0][0]']   
                                                                                                  
 conv4_block10_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block10_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block10_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block10_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block10_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block10_1_relu[0][0]']   
                                                                                                  
 conv4_block10_concat (Concaten  (None, 14, 14, 576)  0          ['conv4_block9_concat[0][0]',    
 ate)                                                             'conv4_block10_2_conv[0][0]']   
                                                                                                  
 conv4_block11_0_bn (BatchNorma  (None, 14, 14, 576)  2304       ['conv4_block10_concat[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block11_0_relu (Activati  (None, 14, 14, 576)  0          ['conv4_block11_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block11_1_conv (Conv2D)  (None, 14, 14, 128)  73728       ['conv4_block11_0_relu[0][0]']   
                                                                                                  
 conv4_block11_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block11_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block11_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block11_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block11_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block11_1_relu[0][0]']   
                                                                                                  
 conv4_block11_concat (Concaten  (None, 14, 14, 608)  0          ['conv4_block10_concat[0][0]',   
 ate)                                                             'conv4_block11_2_conv[0][0]']   
                                                                                                  
 conv4_block12_0_bn (BatchNorma  (None, 14, 14, 608)  2432       ['conv4_block11_concat[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block12_0_relu (Activati  (None, 14, 14, 608)  0          ['conv4_block12_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block12_1_conv (Conv2D)  (None, 14, 14, 128)  77824       ['conv4_block12_0_relu[0][0]']   
                                                                                                  
 conv4_block12_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block12_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block12_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block12_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block12_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block12_1_relu[0][0]']   
                                                                                                  
 conv4_block12_concat (Concaten  (None, 14, 14, 640)  0          ['conv4_block11_concat[0][0]',   
 ate)                                                             'conv4_block12_2_conv[0][0]']   
                                                                                                  
 conv4_block13_0_bn (BatchNorma  (None, 14, 14, 640)  2560       ['conv4_block12_concat[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block13_0_relu (Activati  (None, 14, 14, 640)  0          ['conv4_block13_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block13_1_conv (Conv2D)  (None, 14, 14, 128)  81920       ['conv4_block13_0_relu[0][0]']   
                                                                                                  
 conv4_block13_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block13_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block13_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block13_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block13_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block13_1_relu[0][0]']   
                                                                                                  
 conv4_block13_concat (Concaten  (None, 14, 14, 672)  0          ['conv4_block12_concat[0][0]',   
 ate)                                                             'conv4_block13_2_conv[0][0]']   
                                                                                                  
 conv4_block14_0_bn (BatchNorma  (None, 14, 14, 672)  2688       ['conv4_block13_concat[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block14_0_relu (Activati  (None, 14, 14, 672)  0          ['conv4_block14_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block14_1_conv (Conv2D)  (None, 14, 14, 128)  86016       ['conv4_block14_0_relu[0][0]']   
                                                                                                  
 conv4_block14_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block14_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block14_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block14_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block14_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block14_1_relu[0][0]']   
                                                                                                  
 conv4_block14_concat (Concaten  (None, 14, 14, 704)  0          ['conv4_block13_concat[0][0]',   
 ate)                                                             'conv4_block14_2_conv[0][0]']   
                                                                                                  
 conv4_block15_0_bn (BatchNorma  (None, 14, 14, 704)  2816       ['conv4_block14_concat[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block15_0_relu (Activati  (None, 14, 14, 704)  0          ['conv4_block15_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block15_1_conv (Conv2D)  (None, 14, 14, 128)  90112       ['conv4_block15_0_relu[0][0]']   
                                                                                                  
 conv4_block15_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block15_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block15_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block15_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block15_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block15_1_relu[0][0]']   
                                                                                                  
 conv4_block15_concat (Concaten  (None, 14, 14, 736)  0          ['conv4_block14_concat[0][0]',   
 ate)                                                             'conv4_block15_2_conv[0][0]']   
                                                                                                  
 conv4_block16_0_bn (BatchNorma  (None, 14, 14, 736)  2944       ['conv4_block15_concat[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block16_0_relu (Activati  (None, 14, 14, 736)  0          ['conv4_block16_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block16_1_conv (Conv2D)  (None, 14, 14, 128)  94208       ['conv4_block16_0_relu[0][0]']   
                                                                                                  
 conv4_block16_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block16_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block16_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block16_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block16_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block16_1_relu[0][0]']   
                                                                                                  
 conv4_block16_concat (Concaten  (None, 14, 14, 768)  0          ['conv4_block15_concat[0][0]',   
 ate)                                                             'conv4_block16_2_conv[0][0]']   
                                                                                                  
 conv4_block17_0_bn (BatchNorma  (None, 14, 14, 768)  3072       ['conv4_block16_concat[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block17_0_relu (Activati  (None, 14, 14, 768)  0          ['conv4_block17_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block17_1_conv (Conv2D)  (None, 14, 14, 128)  98304       ['conv4_block17_0_relu[0][0]']   
                                                                                                  
 conv4_block17_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block17_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block17_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block17_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block17_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block17_1_relu[0][0]']   
                                                                                                  
 conv4_block17_concat (Concaten  (None, 14, 14, 800)  0          ['conv4_block16_concat[0][0]',   
 ate)                                                             'conv4_block17_2_conv[0][0]']   
                                                                                                  
 conv4_block18_0_bn (BatchNorma  (None, 14, 14, 800)  3200       ['conv4_block17_concat[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block18_0_relu (Activati  (None, 14, 14, 800)  0          ['conv4_block18_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block18_1_conv (Conv2D)  (None, 14, 14, 128)  102400      ['conv4_block18_0_relu[0][0]']   
                                                                                                  
 conv4_block18_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block18_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block18_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block18_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block18_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block18_1_relu[0][0]']   
                                                                                                  
 conv4_block18_concat (Concaten  (None, 14, 14, 832)  0          ['conv4_block17_concat[0][0]',   
 ate)                                                             'conv4_block18_2_conv[0][0]']   
                                                                                                  
 conv4_block19_0_bn (BatchNorma  (None, 14, 14, 832)  3328       ['conv4_block18_concat[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block19_0_relu (Activati  (None, 14, 14, 832)  0          ['conv4_block19_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block19_1_conv (Conv2D)  (None, 14, 14, 128)  106496      ['conv4_block19_0_relu[0][0]']   
                                                                                                  
 conv4_block19_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block19_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block19_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block19_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block19_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block19_1_relu[0][0]']   
                                                                                                  
 conv4_block19_concat (Concaten  (None, 14, 14, 864)  0          ['conv4_block18_concat[0][0]',   
 ate)                                                             'conv4_block19_2_conv[0][0]']   
                                                                                                  
 conv4_block20_0_bn (BatchNorma  (None, 14, 14, 864)  3456       ['conv4_block19_concat[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block20_0_relu (Activati  (None, 14, 14, 864)  0          ['conv4_block20_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block20_1_conv (Conv2D)  (None, 14, 14, 128)  110592      ['conv4_block20_0_relu[0][0]']   
                                                                                                  
 conv4_block20_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block20_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block20_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block20_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block20_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block20_1_relu[0][0]']   
                                                                                                  
 conv4_block20_concat (Concaten  (None, 14, 14, 896)  0          ['conv4_block19_concat[0][0]',   
 ate)                                                             'conv4_block20_2_conv[0][0]']   
                                                                                                  
 conv4_block21_0_bn (BatchNorma  (None, 14, 14, 896)  3584       ['conv4_block20_concat[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block21_0_relu (Activati  (None, 14, 14, 896)  0          ['conv4_block21_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block21_1_conv (Conv2D)  (None, 14, 14, 128)  114688      ['conv4_block21_0_relu[0][0]']   
                                                                                                  
 conv4_block21_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block21_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block21_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block21_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block21_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block21_1_relu[0][0]']   
                                                                                                  
 conv4_block21_concat (Concaten  (None, 14, 14, 928)  0          ['conv4_block20_concat[0][0]',   
 ate)                                                             'conv4_block21_2_conv[0][0]']   
                                                                                                  
 conv4_block22_0_bn (BatchNorma  (None, 14, 14, 928)  3712       ['conv4_block21_concat[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block22_0_relu (Activati  (None, 14, 14, 928)  0          ['conv4_block22_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block22_1_conv (Conv2D)  (None, 14, 14, 128)  118784      ['conv4_block22_0_relu[0][0]']   
                                                                                                  
 conv4_block22_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block22_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block22_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block22_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block22_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block22_1_relu[0][0]']   
                                                                                                  
 conv4_block22_concat (Concaten  (None, 14, 14, 960)  0          ['conv4_block21_concat[0][0]',   
 ate)                                                             'conv4_block22_2_conv[0][0]']   
                                                                                                  
 conv4_block23_0_bn (BatchNorma  (None, 14, 14, 960)  3840       ['conv4_block22_concat[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block23_0_relu (Activati  (None, 14, 14, 960)  0          ['conv4_block23_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block23_1_conv (Conv2D)  (None, 14, 14, 128)  122880      ['conv4_block23_0_relu[0][0]']   
                                                                                                  
 conv4_block23_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block23_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block23_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block23_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block23_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block23_1_relu[0][0]']   
                                                                                                  
 conv4_block23_concat (Concaten  (None, 14, 14, 992)  0          ['conv4_block22_concat[0][0]',   
 ate)                                                             'conv4_block23_2_conv[0][0]']   
                                                                                                  
 conv4_block24_0_bn (BatchNorma  (None, 14, 14, 992)  3968       ['conv4_block23_concat[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block24_0_relu (Activati  (None, 14, 14, 992)  0          ['conv4_block24_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block24_1_conv (Conv2D)  (None, 14, 14, 128)  126976      ['conv4_block24_0_relu[0][0]']   
                                                                                                  
 conv4_block24_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block24_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv4_block24_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block24_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv4_block24_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block24_1_relu[0][0]']   
                                                                                                  
 conv4_block24_concat (Concaten  (None, 14, 14, 1024  0          ['conv4_block23_concat[0][0]',   
 ate)                           )                                 'conv4_block24_2_conv[0][0]']   
                                                                                                  
 pool4_bn (BatchNormalization)  (None, 14, 14, 1024  4096        ['conv4_block24_concat[0][0]']   
                                )                                                                 
                                                                                                  
 pool4_relu (Activation)        (None, 14, 14, 1024  0           ['pool4_bn[0][0]']               
                                )                                                                 
                                                                                                  
 pool4_conv (Conv2D)            (None, 14, 14, 512)  524288      ['pool4_relu[0][0]']             
                                                                                                  
 pool4_pool (AveragePooling2D)  (None, 7, 7, 512)    0           ['pool4_conv[0][0]']             
                                                                                                  
 conv5_block1_0_bn (BatchNormal  (None, 7, 7, 512)   2048        ['pool4_pool[0][0]']             
 ization)                                                                                         
                                                                                                  
 conv5_block1_0_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block1_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block1_1_conv (Conv2D)   (None, 7, 7, 128)    65536       ['conv5_block1_0_relu[0][0]']    
                                                                                                  
 conv5_block1_1_bn (BatchNormal  (None, 7, 7, 128)   512         ['conv5_block1_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block1_1_relu (Activatio  (None, 7, 7, 128)   0           ['conv5_block1_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block1_2_conv (Conv2D)   (None, 7, 7, 32)     36864       ['conv5_block1_1_relu[0][0]']    
                                                                                                  
 conv5_block1_concat (Concatena  (None, 7, 7, 544)   0           ['pool4_pool[0][0]',             
 te)                                                              'conv5_block1_2_conv[0][0]']    
                                                                                                  
 conv5_block2_0_bn (BatchNormal  (None, 7, 7, 544)   2176        ['conv5_block1_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block2_0_relu (Activatio  (None, 7, 7, 544)   0           ['conv5_block2_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block2_1_conv (Conv2D)   (None, 7, 7, 128)    69632       ['conv5_block2_0_relu[0][0]']    
                                                                                                  
 conv5_block2_1_bn (BatchNormal  (None, 7, 7, 128)   512         ['conv5_block2_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block2_1_relu (Activatio  (None, 7, 7, 128)   0           ['conv5_block2_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block2_2_conv (Conv2D)   (None, 7, 7, 32)     36864       ['conv5_block2_1_relu[0][0]']    
                                                                                                  
 conv5_block2_concat (Concatena  (None, 7, 7, 576)   0           ['conv5_block1_concat[0][0]',    
 te)                                                              'conv5_block2_2_conv[0][0]']    
                                                                                                  
 conv5_block3_0_bn (BatchNormal  (None, 7, 7, 576)   2304        ['conv5_block2_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block3_0_relu (Activatio  (None, 7, 7, 576)   0           ['conv5_block3_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block3_1_conv (Conv2D)   (None, 7, 7, 128)    73728       ['conv5_block3_0_relu[0][0]']    
                                                                                                  
 conv5_block3_1_bn (BatchNormal  (None, 7, 7, 128)   512         ['conv5_block3_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block3_1_relu (Activatio  (None, 7, 7, 128)   0           ['conv5_block3_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block3_2_conv (Conv2D)   (None, 7, 7, 32)     36864       ['conv5_block3_1_relu[0][0]']    
                                                                                                  
 conv5_block3_concat (Concatena  (None, 7, 7, 608)   0           ['conv5_block2_concat[0][0]',    
 te)                                                              'conv5_block3_2_conv[0][0]']    
                                                                                                  
 conv5_block4_0_bn (BatchNormal  (None, 7, 7, 608)   2432        ['conv5_block3_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block4_0_relu (Activatio  (None, 7, 7, 608)   0           ['conv5_block4_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block4_1_conv (Conv2D)   (None, 7, 7, 128)    77824       ['conv5_block4_0_relu[0][0]']    
                                                                                                  
 conv5_block4_1_bn (BatchNormal  (None, 7, 7, 128)   512         ['conv5_block4_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block4_1_relu (Activatio  (None, 7, 7, 128)   0           ['conv5_block4_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block4_2_conv (Conv2D)   (None, 7, 7, 32)     36864       ['conv5_block4_1_relu[0][0]']    
                                                                                                  
 conv5_block4_concat (Concatena  (None, 7, 7, 640)   0           ['conv5_block3_concat[0][0]',    
 te)                                                              'conv5_block4_2_conv[0][0]']    
                                                                                                  
 conv5_block5_0_bn (BatchNormal  (None, 7, 7, 640)   2560        ['conv5_block4_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block5_0_relu (Activatio  (None, 7, 7, 640)   0           ['conv5_block5_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block5_1_conv (Conv2D)   (None, 7, 7, 128)    81920       ['conv5_block5_0_relu[0][0]']    
                                                                                                  
 conv5_block5_1_bn (BatchNormal  (None, 7, 7, 128)   512         ['conv5_block5_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block5_1_relu (Activatio  (None, 7, 7, 128)   0           ['conv5_block5_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block5_2_conv (Conv2D)   (None, 7, 7, 32)     36864       ['conv5_block5_1_relu[0][0]']    
                                                                                                  
 conv5_block5_concat (Concatena  (None, 7, 7, 672)   0           ['conv5_block4_concat[0][0]',    
 te)                                                              'conv5_block5_2_conv[0][0]']    
                                                                                                  
 conv5_block6_0_bn (BatchNormal  (None, 7, 7, 672)   2688        ['conv5_block5_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block6_0_relu (Activatio  (None, 7, 7, 672)   0           ['conv5_block6_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block6_1_conv (Conv2D)   (None, 7, 7, 128)    86016       ['conv5_block6_0_relu[0][0]']    
                                                                                                  
 conv5_block6_1_bn (BatchNormal  (None, 7, 7, 128)   512         ['conv5_block6_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block6_1_relu (Activatio  (None, 7, 7, 128)   0           ['conv5_block6_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block6_2_conv (Conv2D)   (None, 7, 7, 32)     36864       ['conv5_block6_1_relu[0][0]']    
                                                                                                  
 conv5_block6_concat (Concatena  (None, 7, 7, 704)   0           ['conv5_block5_concat[0][0]',    
 te)                                                              'conv5_block6_2_conv[0][0]']    
                                                                                                  
 conv5_block7_0_bn (BatchNormal  (None, 7, 7, 704)   2816        ['conv5_block6_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block7_0_relu (Activatio  (None, 7, 7, 704)   0           ['conv5_block7_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block7_1_conv (Conv2D)   (None, 7, 7, 128)    90112       ['conv5_block7_0_relu[0][0]']    
                                                                                                  
 conv5_block7_1_bn (BatchNormal  (None, 7, 7, 128)   512         ['conv5_block7_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block7_1_relu (Activatio  (None, 7, 7, 128)   0           ['conv5_block7_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block7_2_conv (Conv2D)   (None, 7, 7, 32)     36864       ['conv5_block7_1_relu[0][0]']    
                                                                                                  
 conv5_block7_concat (Concatena  (None, 7, 7, 736)   0           ['conv5_block6_concat[0][0]',    
 te)                                                              'conv5_block7_2_conv[0][0]']    
                                                                                                  
 conv5_block8_0_bn (BatchNormal  (None, 7, 7, 736)   2944        ['conv5_block7_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block8_0_relu (Activatio  (None, 7, 7, 736)   0           ['conv5_block8_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block8_1_conv (Conv2D)   (None, 7, 7, 128)    94208       ['conv5_block8_0_relu[0][0]']    
                                                                                                  
 conv5_block8_1_bn (BatchNormal  (None, 7, 7, 128)   512         ['conv5_block8_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block8_1_relu (Activatio  (None, 7, 7, 128)   0           ['conv5_block8_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block8_2_conv (Conv2D)   (None, 7, 7, 32)     36864       ['conv5_block8_1_relu[0][0]']    
                                                                                                  
 conv5_block8_concat (Concatena  (None, 7, 7, 768)   0           ['conv5_block7_concat[0][0]',    
 te)                                                              'conv5_block8_2_conv[0][0]']    
                                                                                                  
 conv5_block9_0_bn (BatchNormal  (None, 7, 7, 768)   3072        ['conv5_block8_concat[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block9_0_relu (Activatio  (None, 7, 7, 768)   0           ['conv5_block9_0_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block9_1_conv (Conv2D)   (None, 7, 7, 128)    98304       ['conv5_block9_0_relu[0][0]']    
                                                                                                  
 conv5_block9_1_bn (BatchNormal  (None, 7, 7, 128)   512         ['conv5_block9_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block9_1_relu (Activatio  (None, 7, 7, 128)   0           ['conv5_block9_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block9_2_conv (Conv2D)   (None, 7, 7, 32)     36864       ['conv5_block9_1_relu[0][0]']    
                                                                                                  
 conv5_block9_concat (Concatena  (None, 7, 7, 800)   0           ['conv5_block8_concat[0][0]',    
 te)                                                              'conv5_block9_2_conv[0][0]']    
                                                                                                  
 conv5_block10_0_bn (BatchNorma  (None, 7, 7, 800)   3200        ['conv5_block9_concat[0][0]']    
 lization)                                                                                        
                                                                                                  
 conv5_block10_0_relu (Activati  (None, 7, 7, 800)   0           ['conv5_block10_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv5_block10_1_conv (Conv2D)  (None, 7, 7, 128)    102400      ['conv5_block10_0_relu[0][0]']   
                                                                                                  
 conv5_block10_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block10_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv5_block10_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block10_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv5_block10_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block10_1_relu[0][0]']   
                                                                                                  
 conv5_block10_concat (Concaten  (None, 7, 7, 832)   0           ['conv5_block9_concat[0][0]',    
 ate)                                                             'conv5_block10_2_conv[0][0]']   
                                                                                                  
 conv5_block11_0_bn (BatchNorma  (None, 7, 7, 832)   3328        ['conv5_block10_concat[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv5_block11_0_relu (Activati  (None, 7, 7, 832)   0           ['conv5_block11_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv5_block11_1_conv (Conv2D)  (None, 7, 7, 128)    106496      ['conv5_block11_0_relu[0][0]']   
                                                                                                  
 conv5_block11_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block11_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv5_block11_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block11_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv5_block11_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block11_1_relu[0][0]']   
                                                                                                  
 conv5_block11_concat (Concaten  (None, 7, 7, 864)   0           ['conv5_block10_concat[0][0]',   
 ate)                                                             'conv5_block11_2_conv[0][0]']   
                                                                                                  
 conv5_block12_0_bn (BatchNorma  (None, 7, 7, 864)   3456        ['conv5_block11_concat[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv5_block12_0_relu (Activati  (None, 7, 7, 864)   0           ['conv5_block12_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv5_block12_1_conv (Conv2D)  (None, 7, 7, 128)    110592      ['conv5_block12_0_relu[0][0]']   
                                                                                                  
 conv5_block12_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block12_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv5_block12_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block12_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv5_block12_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block12_1_relu[0][0]']   
                                                                                                  
 conv5_block12_concat (Concaten  (None, 7, 7, 896)   0           ['conv5_block11_concat[0][0]',   
 ate)                                                             'conv5_block12_2_conv[0][0]']   
                                                                                                  
 conv5_block13_0_bn (BatchNorma  (None, 7, 7, 896)   3584        ['conv5_block12_concat[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv5_block13_0_relu (Activati  (None, 7, 7, 896)   0           ['conv5_block13_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv5_block13_1_conv (Conv2D)  (None, 7, 7, 128)    114688      ['conv5_block13_0_relu[0][0]']   
                                                                                                  
 conv5_block13_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block13_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv5_block13_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block13_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv5_block13_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block13_1_relu[0][0]']   
                                                                                                  
 conv5_block13_concat (Concaten  (None, 7, 7, 928)   0           ['conv5_block12_concat[0][0]',   
 ate)                                                             'conv5_block13_2_conv[0][0]']   
                                                                                                  
 conv5_block14_0_bn (BatchNorma  (None, 7, 7, 928)   3712        ['conv5_block13_concat[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv5_block14_0_relu (Activati  (None, 7, 7, 928)   0           ['conv5_block14_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv5_block14_1_conv (Conv2D)  (None, 7, 7, 128)    118784      ['conv5_block14_0_relu[0][0]']   
                                                                                                  
 conv5_block14_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block14_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv5_block14_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block14_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv5_block14_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block14_1_relu[0][0]']   
                                                                                                  
 conv5_block14_concat (Concaten  (None, 7, 7, 960)   0           ['conv5_block13_concat[0][0]',   
 ate)                                                             'conv5_block14_2_conv[0][0]']   
                                                                                                  
 conv5_block15_0_bn (BatchNorma  (None, 7, 7, 960)   3840        ['conv5_block14_concat[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv5_block15_0_relu (Activati  (None, 7, 7, 960)   0           ['conv5_block15_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv5_block15_1_conv (Conv2D)  (None, 7, 7, 128)    122880      ['conv5_block15_0_relu[0][0]']   
                                                                                                  
 conv5_block15_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block15_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv5_block15_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block15_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv5_block15_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block15_1_relu[0][0]']   
                                                                                                  
 conv5_block15_concat (Concaten  (None, 7, 7, 992)   0           ['conv5_block14_concat[0][0]',   
 ate)                                                             'conv5_block15_2_conv[0][0]']   
                                                                                                  
 conv5_block16_0_bn (BatchNorma  (None, 7, 7, 992)   3968        ['conv5_block15_concat[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv5_block16_0_relu (Activati  (None, 7, 7, 992)   0           ['conv5_block16_0_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv5_block16_1_conv (Conv2D)  (None, 7, 7, 128)    126976      ['conv5_block16_0_relu[0][0]']   
                                                                                                  
 conv5_block16_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block16_1_conv[0][0]']   
 lization)                                                                                        
                                                                                                  
 conv5_block16_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block16_1_bn[0][0]']     
 on)                                                                                              
                                                                                                  
 conv5_block16_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block16_1_relu[0][0]']   
                                                                                                  
 conv5_block16_concat (Concaten  (None, 7, 7, 1024)  0           ['conv5_block15_concat[0][0]',   
 ate)                                                             'conv5_block16_2_conv[0][0]']   
                                                                                                  
 bn (BatchNormalization)        (None, 7, 7, 1024)   4096        ['conv5_block16_concat[0][0]']   
                                                                                                  
 relu (Activation)              (None, 7, 7, 1024)   0           ['bn[0][0]']                     
                                                                                                  
 global_average_pooling2d_1 (Gl  (None, 1024)        0           ['relu[0][0]']                   
 obalAveragePooling2D)                                                                            
                                                                                                  
 dropout_1 (Dropout)            (None, 1024)         0           ['global_average_pooling2d_1[0][0
                                                                 ]']                              
                                                                                                  
 dense_1 (Dense)                (None, 128)          131200      ['dropout_1[0][0]']              
                                                                                                  
==================================================================================================
Total params: 7,168,704
Trainable params: 7,085,056
Non-trainable params: 83,648
__________________________________________________________________________________________________
```


**[Celda 13 - Código]**
```python
nombre_capa = 'conv5_block9_1_conv'

# Obtener el índice de la capa a partir de su nombre
indice_capa = None
for i, layer in enumerate(densenet.layers):
    if layer.name == nombre_capa:
        indice_capa = i
        break

# Verificar si se encontró la capa y mostrar el índice
if indice_capa is not None:
    print(f"El índice de la capa '{nombre_capa}' es: {indice_capa}")
else:
    print(f"No se encontró una capa con el nombre '{nombre_capa}' en el modelo")
```


*Salida:*
```text
El índice de la capa 'conv5_block9_1_conv' es: 371
```


**[Celda 14 - Código]**
```python
# Congelar todas las capas
for layer in densenet.layers:
    layer.trainable = False

# Descongelar las últimas capas
for layer in densenet.layers[371:]:
    layer.trainable = True
```


**[Celda 15 - Código]**
```python
print(densenet.layers[371].trainable)
print(densenet.layers[370].trainable)
```


*Salida:*
```text
True
False
```


**[Celda 16 - Código]**
```python
#Solo para fine tuning

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
batch_size = 32
image_size = (224, 224)
train_generator = data_generator.flow(
    x=array_imagenes,
    y=to_categorical(y_imagenesBD, num_classes=128),  # En caso de tener etiquetas, proporcionarlas aquí
    batch_size=batch_size,
    shuffle=True
)
```


**[Celda 17 - Código]**
```python
len(y_imagenesBD)
```


*Salida:*
```text
99
```


**[Celda 18 - Código]**
```python
x,y=train_generator.next()
print(x.shape)
print(y)
```


*Salida:*
```text
(32, 224, 224, 3)
['97' '186' '127' '85' '178' '29' '134' '147' '302' '306' '6' '105' '60'
 '191' '173' '57' '201' '100' '116' '194' '120' '199' '82' '110' '66' '13'
 '63' '195' '26' '304' '7' '307']
```


**[Celda 19 - Código]**
```python

```


**[Celda 20 - Código]**
```python
history = densenet.fit(train_generator,
                    batch_size=32,
                    epochs=10,
                    validation_data=train_generator)
```


*Salida:*
```text
Epoch 1/10
7/7 [==============================] - 199s 23s/step - loss: 6.4969 - accuracy: 0.0000e+00 - val_loss: 5.6831 - val_accuracy: 0.0099
Epoch 2/10
7/7 [==============================] - 172s 25s/step - loss: 5.9486 - accuracy: 0.0148 - val_loss: 5.6428 - val_accuracy: 0.0000e+00
Epoch 3/10
7/7 [==============================] - 161s 23s/step - loss: 5.7134 - accuracy: 0.0148 - val_loss: 5.5765 - val_accuracy: 0.0049
Epoch 4/10
7/7 [==============================] - 175s 25s/step - loss: 5.3676 - accuracy: 0.0246 - val_loss: 5.4452 - val_accuracy: 0.0099
Epoch 5/10
7/7 [==============================] - 170s 27s/step - loss: 4.9845 - accuracy: 0.0345 - val_loss: 5.2439 - val_accuracy: 0.0197
Epoch 6/10
7/7 [==============================] - 165s 24s/step - loss: 4.8503 - accuracy: 0.0542 - val_loss: 5.0743 - val_accuracy: 0.0246
Epoch 7/10
7/7 [==============================] - 159s 25s/step - loss: 4.3343 - accuracy: 0.1429 - val_loss: 4.9550 - val_accuracy: 0.0394
Epoch 8/10
7/7 [==============================] - 168s 25s/step - loss: 4.2541 - accuracy: 0.1232 - val_loss: 4.7354 - val_accuracy: 0.0837
Epoch 9/10
7/7 [==============================] - 173s 25s/step - loss: 3.7692 - accuracy: 0.2217 - val_loss: 4.5201 - val_accuracy: 0.1034
Epoch 10/10
7/7 [==============================] - 162s 24s/step - loss: 3.6736 - accuracy: 0.2315 - val_loss: 4.4518 - val_accuracy: 0.1232
```


**[Celda 21 - Código]**
```python
#Asignar capa a una variable
global_pooling2d = densenet.get_layer('global_average_pooling2d_1')

# Crear un nuevo modelo que tenga la capa convolucional como salida
new_model_densenet = Model(inputs=densenet.input, outputs=global_pooling2d.output)
```


**[Celda 22 - Código]**
```python
#No se ejecuta
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

directorio = r''
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
['45', '54', '33', '66', '52', '12', '51', '59', '47', '68', '36', '29', '22', '23', '17', '55', '3', '16', '63', '49', '61', '37', '67', '32', '18', '5', '53', '35', '50', '56', '24', '65', '28', '41', '57', '38', '10', '1', '19', '62', '40', '2', '11', '34', '39', '48', '25', '27', '13', '31', '7', '60', '46', '20', '42', '44', '64', '21', '58', '9', '4', '43', '15', '14', '8', '30', '26', '6', '163', '182', '135', '137', '136', '158', '183', '105', '92', '121', '156', '178', '157', '78', '144', '115', '184', '169', '141', '159', '179', '131', '107', '153', '143', '89', '140', '174', '104', '175', '74', '118', '168', '95', '113', '126', '185', '119', '122', '138', '73', '83', '133', '82', '88', '99', '91', '123', '93', '80', '108', '165', '72', '172', '112', '117', '146', '180', '90', '120', '177', '100', '127', '160', '147', '171', '155', '148', '181', '76', '86', '124', '110', '170', '97', '161', '139', '134', '75', '142', '166', '111', '79', '94', '71', '69', '173', '151', '145', '129', '130', '164', '85', '102', '96', '114', '103', '176', '116', '128', '150', '125', '98', '167', '109', '87', '84', '149', '70', '106', '101', '81', '186', '132', '77', '152', '154', '162', '200', '201', '187', '203', '194', '189', '197', '196', '191', '193', '198', '199', '202', '192', '188', '195', '190', '306', '303', '302', '301', '309', '304', '310', '307', '305', '308']
```


**[Celda 23 - Código]**
```python
#Algoritmo no optimizado
# CONSULTAS POR SIMILITUD UTILIZANDO LA BASE DE CONSULTAS DIBUJADAS

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
    print("Consulta número: ", i)
    vector_consulta = new_model_densenet.predict(consultas[i].reshape(1,224,224,3), verbose=0)
    a = np.array(vector_consulta[0])
    min = 2000000
    nnklist = []
    # Inicio del for DB
    for j in range(cantidad_BD):
        vector_elemento = new_model_densenet.predict(imagenesBD[j].reshape(1,224,224,3), verbose=0)
        b = np.array(vector_elemento[0])
        #dist = np.sqrt(np.sum(np.square(a-b)))
        dist = np.sum(np.abs(a-b))
        nnk(nnklist,[dist,j],5)
        if dist < min:
            min = dist
            vector_cercano = b
            posicion_cercano = j


    print("Imagen de BD: ", y_imagenesBD[posicion_cercano], "Imagen de consulta: ", y_consultas[i])
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
Imagen de BD:  west-ham-united Imagen de consulta:  aston-villa
Mas cercanos:  [[1549.8031, 'west-ham-united'], [1578.9729, 'olympique-lyonnais'], [1640.3906, 'as-monaco'], [1719.156, 'arsenal'], [1751.8876, 'roma']]


Consulta número:  1
Imagen de BD:  augsburg Imagen de consulta:  augsburg
Mas cercanos:  [[521.4999, 'augsburg'], [1399.0387, 'torino'], [1437.9252, 'osasuna'], [1496.3474, 'salernitana'], [1535.144, 'stade-rennais-fc']]
--> ACIERTO


Consulta número:  2
Imagen de BD:  real-sociedad Imagen de consulta:  bremen
Mas cercanos:  [[1753.3483, 'real-sociedad'], [1922.0057, 'fiorentina'], [1975.6067, 'ogc-nice'], [1987.9515, 'redbull-leipzig'], [2062.508, 'liverpool']]


Consulta número:  3
Imagen de BD:  cadiz Imagen de consulta:  cadiz
Mas cercanos:  [[1788.9749, 'cadiz'], [1832.0624, 'real-madrid'], [1859.9045, 'real-betis'], [2022.195, 'redbull-leipzig'], [2052.4297, 'elche']]
--> ACIERTO


Consulta número:  4
Imagen de BD:  redbull-leipzig Imagen de consulta:  celta
Mas cercanos:  [[1431.7886, 'redbull-leipzig'], [1502.6128, 'celta'], [1725.662, 'fiorentina'], [1740.8276, 'as-monaco'], [1793.8352, 'real-betis']]
--> ACIERTO


Consulta número:  5
Imagen de BD:  dusseldorf Imagen de consulta:  dusseldorf
Mas cercanos:  [[1225.2018, 'dusseldorf'], [1432.7866, 'nuremberg'], [1447.4025, 'frankfurt'], [1987.0126, 'espanyol'], [2020.9753, 'arsenal']]
--> ACIERTO


Consulta número:  6
Imagen de BD:  empoli Imagen de consulta:  empoli
Mas cercanos:  [[694.1832, 'empoli'], [1975.0471, 'hertha-bsc-berlin'], [2035.0234, 'napoli'], [2136.769, 'estac-troyes'], [2211.3115, 'hoffenheim']]
--> ACIERTO


Consulta número:  7
Imagen de BD:  clermont-foot-63 Imagen de consulta:  clermont-foot-63
Mas cercanos:  [[1103.2991, 'clermont-foot-63'], [1514.7429, 'cagliari'], [1702.9758, 'as-monaco'], [1723.7452, 'stade-brestois-29'], [1724.2299, 'levante']]
--> ACIERTO


Consulta número:  8
Imagen de BD:  crystal-palace Imagen de consulta:  deportivo-alavez
Mas cercanos:  [[1614.9412, 'crystal-palace'], [1796.8672, 'deportivo-alavez'], [1835.1023, 'leicester-city'], [1922.9406, 'liverpool'], [2048.638, 'hoffenheim']]
--> ACIERTO


Consulta número:  9
Imagen de BD:  udinese Imagen de consulta:  udinese
Mas cercanos:  [[1294.5349, 'udinese'], [1746.823, 'ogc-nice'], [1803.9574, 'everton'], [2119.6206, 'venezia'], [2149.9153, 'liverpool']]
--> ACIERTO


Consulta número:  10
Imagen de BD:  paris-saint-germain Imagen de consulta:  paris-saint-germain
Mas cercanos:  [[959.05334, 'paris-saint-germain'], [2152.7683, 'brighton'], [2281.889, 'napoli'], [2314.0771, 'fc-girondins-de-bordeaux'], [2437.3103, 'estac-troyes']]
--> ACIERTO


Consulta número:  11
Imagen de BD:  osasuna Imagen de consulta:  elche
Mas cercanos:  [[1194.5408, 'osasuna'], [1367.8522, 'augsburg'], [1402.0751, 'elche'], [1444.2437, 'fiorentina'], [1593.2031, 'as-monaco']]
--> ACIERTO


Consulta número:  12
Imagen de BD:  manchester-united Imagen de consulta:  manchester-united
Mas cercanos:  [[1250.7029, 'manchester-united'], [1330.781, 'mallorca'], [1580.414, 'watford'], [1620.6797, 'roma'], [1651.8074, 'levante']]
--> ACIERTO


Consulta número:  13
Imagen de BD:  ogc-nice Imagen de consulta:  ogc-nice
Mas cercanos:  [[841.4332, 'ogc-nice'], [1566.1919, 'venezia'], [1703.7534, 'everton'], [1810.1875, 'udinese'], [1856.5566, 'hellas-verona']]
--> ACIERTO


Consulta número:  14
Imagen de BD:  levante Imagen de consulta:  levante
Mas cercanos:  [[1117.2297, 'levante'], [1514.1821, 'southampton'], [1525.6754, 'leverkusen'], [1739.7067, 'clermont-foot-63'], [1972.5256, 'mallorca']]
--> ACIERTO


Consulta número:  15
Imagen de BD:  spezia Imagen de consulta:  spezia
Mas cercanos:  [[2024.7686, 'spezia'], [2200.2507, 'ogc-nice'], [2284.022, 'udinese'], [2324.9734, 'venezia'], [2480.4739, 'liverpool']]
--> ACIERTO


Consulta número:  16
Imagen de BD:  wolfsburg Imagen de consulta:  wolfsburg
Mas cercanos:  [[1323.3715, 'wolfsburg'], [2819.8367, 'as-saint-etienne'], [2831.097, 'getafe'], [3090.376, 'bremen'], [3136.0422, 'valencia']]
--> ACIERTO


Consulta número:  17
Imagen de BD:  roma Imagen de consulta:  stuttgart
Mas cercanos:  [[1529.139, 'roma'], [1563.866, 'as-monaco'], [1563.8724, 'stade-rennais-fc'], [1646.6923, 'fiorentina'], [1705.4515, 'arsenal']]


Consulta número:  18
Imagen de BD:  barcelona Imagen de consulta:  barcelona
Mas cercanos:  [[1009.1971, 'barcelona'], [1868.8519, 'manchester-united'], [1962.7023, 'watford'], [2133.5269, 'burnley'], [2137.3872, 'levante']]
--> ACIERTO


Consulta número:  19
Imagen de BD:  inter Imagen de consulta:  inter
Mas cercanos:  [[670.12994, 'inter'], [2832.1616, 'brighton'], [2982.9858, 'sampdoria'], [3023.0713, 'cagliari'], [3061.6533, 'hannover']]
--> ACIERTO


Consulta número:  20
Imagen de BD:  roma Imagen de consulta:  roma
Mas cercanos:  [[739.7527, 'roma'], [1532.5042, 'cadiz'], [1616.1053, 'mallorca'], [1633.9065, 'rc-lens'], [1678.8246, 'genoa']]
--> ACIERTO


Consulta número:  21
Imagen de BD:  stade-rennais-fc Imagen de consulta:  villarreal
Mas cercanos:  [[1420.262, 'stade-rennais-fc'], [1516.7128, 'villarreal'], [1584.678, 'levante'], [1685.0885, 'mallorca'], [1736.4066, 'roma']]
--> ACIERTO


Consulta número:  22
Imagen de BD:  crystal-palace Imagen de consulta:  crystal-palace
Mas cercanos:  [[1031.8431, 'crystal-palace'], [1797.5524, 'real-sociedad'], [1862.0817, 'sampdoria'], [1897.1588, 'hoffenheim'], [1970.9192, 'liverpool']]
--> ACIERTO


Consulta número:  23
Imagen de BD:  liverpool Imagen de consulta:  machester-city
Mas cercanos:  [[1724.52, 'liverpool'], [1883.6102, 'crystal-palace'], [1890.6616, 'real-sociedad'], [2103.579, 'hoffenheim'], [2203.3716, 'stade-brestois-29']]


Consulta número:  24
Imagen de BD:  fiorentina Imagen de consulta:  fiorentina
Mas cercanos:  [[1790.2493, 'fiorentina'], [1971.5325, 'redbull-leipzig'], [2008.8529, 'real-betis'], [2110.06, 'villarreal'], [2150.1018, 'real-sociedad']]
--> ACIERTO


Consulta número:  25
Imagen de BD:  moenchengladbach Imagen de consulta:  moenchengladbach
Mas cercanos:  [[974.43976, 'moenchengladbach'], [2622.373, 'angers-sco'], [2705.3896, 'fc-lorient'], [2778.7766, 'tottenham-hotspur'], [2892.4043, 'atalanta']]
--> ACIERTO


Consulta número:  26
Imagen de BD:  napoli Imagen de consulta:  napoli
Mas cercanos:  [[1183.8674, 'napoli'], [2004.3035, 'schalke'], [2043.9749, 'estac-troyes'], [2203.099, 'hoffenheim'], [2299.295, 'olympique-de-marseille']]
--> ACIERTO


Consulta número:  27
Imagen de BD:  chelsea Imagen de consulta:  chelsea
Mas cercanos:  [[448.151, 'chelsea'], [1922.4707, 'olympique-lyonnais'], [1924.8633, 'arsenal'], [1970.4415, 'as-monaco'], [2000.3646, 'frankfurt']]
--> ACIERTO


Consulta número:  28
Imagen de BD:  cagliari Imagen de consulta:  cagliari
Mas cercanos:  [[730.91394, 'cagliari'], [1766.8738, 'hoffenheim'], [1871.0608, 'real-sociedad'], [1880.5836, 'losc-lille'], [1893.108, 'augsburg']]
--> ACIERTO


Consulta número:  29
Imagen de BD:  sampdoria Imagen de consulta:  sampdoria
Mas cercanos:  [[1115.5151, 'sampdoria'], [1625.3354, 'losc-lille'], [1745.9669, 'cagliari'], [1812.4215, 'real-sociedad'], [1860.589, 'hoffenheim']]
--> ACIERTO


Consulta número:  30
Imagen de BD:  real-betis Imagen de consulta:  real-betis
Mas cercanos:  [[530.16003, 'real-betis'], [1482.0476, 'as-monaco'], [1557.2274, 'elche'], [1599.996, 'redbull-leipzig'], [1641.6854, 'celta']]
--> ACIERTO


Consulta número:  31
Imagen de BD:  olympique-lyonnais Imagen de consulta:  olympique-lyonnais
Mas cercanos:  [[711.69714, 'olympique-lyonnais'], [1259.6741, 'as-monaco'], [1378.283, 'arsenal'], [1603.6556, 'granada'], [1703.0836, 'west-ham-united']]
--> ACIERTO


Consulta número:  32
Imagen de BD:  juventus Imagen de consulta:  juventus
Mas cercanos:  [[1632.9211, 'juventus'], [2496.8486, 'bologna'], [2681.7556, 'redbull-leipzig'], [2724.4158, 'hertha-bsc-berlin'], [2747.6274, 'tottenham-hotspur']]
--> ACIERTO


Consulta número:  33
Imagen de BD:  hellas-verona Imagen de consulta:  getafe
Mas cercanos:  [[1492.8065, 'hellas-verona'], [1495.063, 'hoffenheim'], [1508.7056, 'everton'], [1603.7673, 'stade-brestois-29'], [1624.3306, 'augsburg']]


Consulta número:  34
Imagen de BD:  brentford Imagen de consulta:  brentford
Mas cercanos:  [[1160.0934, 'brentford'], [1366.6387, 'stade-brestois-29'], [1617.44, 'arsenal'], [1653.265, 'frankfurt'], [1758.7749, 'liverpool']]
--> ACIERTO


Consulta número:  35
Imagen de BD:  angers-sco Imagen de consulta:  watford
Mas cercanos:  [[2367.0676, 'angers-sco'], [2442.2852, 'moenchengladbach'], [2778.6829, 'fc-lorient'], [2893.8662, 'hertha-bsc-berlin'], [2983.169, 'empoli']]


Consulta número:  36
Imagen de BD:  bologna Imagen de consulta:  bologna
Mas cercanos:  [[1332.6853, 'bologna'], [2063.341, 'athletic'], [2365.6445, 'sassuolo'], [2408.5747, 'redbull-leipzig'], [2428.0774, 'hertha-bsc-berlin']]
--> ACIERTO


Consulta número:  37
Imagen de BD:  norwich-city Imagen de consulta:  norwich-city
Mas cercanos:  [[2059.2146, 'norwich-city'], [2962.2107, 'genoa'], [2973.812, 'espanyol'], [2988.504, 'granada'], [3018.9438, 'watford']]
--> ACIERTO


Consulta número:  38
```


**[Celda 24 - Código]**
```python
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
vectores_elemento = new_model_densenet.predict(imagenesBD)
vectores_consulta = new_model_densenet.predict(consultas)

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
4/4 [==============================] - 20s 4s/step
2/2 [==============================] - 10s 4s/step
Consulta número:  0
   Imagen de consulta:  aston-villa
   Mas cercanos:  [[137.35101, 'west-ham-united'], [139.72972, 'roma'], [145.16667, 'augsburg'], [145.26486, 'olympique-lyonnais'], [150.27898, 'as-monaco']]


Consulta número:  1
   Imagen de consulta:  augsburg
   Mas cercanos:  [[45.433125, 'augsburg'], [119.480225, 'osasuna'], [126.15412, 'stade-rennais-fc'], [131.00252, 'torino'], [131.3438, 'as-monaco']]
--> ACIERTO
    Primer Lugar: 1
    Entre los tres primeros: 1


Consulta número:  2
   Imagen de consulta:  bremen
   Mas cercanos:  [[140.81874, 'real-sociedad'], [160.44606, 'fiorentina'], [160.85881, 'liverpool'], [167.73592, 'as-monaco'], [170.20271, 'venezia']]


Consulta número:  3
   Imagen de consulta:  cadiz
   Mas cercanos:  [[146.31024, 'cadiz'], [155.31801, 'real-madrid'], [174.19339, 'rc-lens'], [175.98715, 'rayo-vallecano'], [178.49103, 'real-betis']]
--> ACIERTO
    Primer Lugar: 2
    Entre los tres primeros: 2


Consulta número:  4
   Imagen de consulta:  celta
   Mas cercanos:  [[110.08828, 'redbull-leipzig'], [122.441345, 'celta'], [153.03079, 'as-monaco'], [154.7097, 'clermont-foot-63'], [159.06079, 'real-betis']]
--> ACIERTO
    Entre los tres primeros: 3


Consulta número:  5
   Imagen de consulta:  dusseldorf
   Mas cercanos:  [[125.08061, 'dusseldorf'], [128.17638, 'frankfurt'], [149.72426, 'nuremberg'], [176.06407, 'espanyol'], [194.71277, 'mainz']]
--> ACIERTO
    Primer Lugar: 3
    Entre los tres primeros: 4


Consulta número:  6
   Imagen de consulta:  empoli
   Mas cercanos:  [[56.19085, 'empoli'], [164.56862, 'hertha-bsc-berlin'], [177.38737, 'estac-troyes'], [177.65565, 'napoli'], [182.03787, 'fc-girondins-de-bordeaux']]
--> ACIERTO
    Primer Lugar: 4
    Entre los tres primeros: 5


Consulta número:  7
   Imagen de consulta:  clermont-foot-63
   Mas cercanos:  [[97.46265, 'clermont-foot-63'], [129.07506, 'cagliari'], [132.98831, 'as-monaco'], [138.79735, 'real-sociedad'], [139.7836, 'arsenal']]
--> ACIERTO
    Primer Lugar: 5
    Entre los tres primeros: 6


Consulta número:  8
   Imagen de consulta:  deportivo-alavez
   Mas cercanos:  [[128.06502, 'crystal-palace'], [152.79399, 'leicester-city'], [162.76123, 'deportivo-alavez'], [168.60565, 'hoffenheim'], [179.48459, 'fc-girondins-de-bordeaux']]
--> ACIERTO
    Entre los tres primeros: 7


Consulta número:  9
   Imagen de consulta:  udinese
   Mas cercanos:  [[109.421936, 'udinese'], [143.41884, 'ogc-nice'], [152.34103, 'everton'], [175.42534, 'liverpool'], [175.90479, 'salernitana']]
--> ACIERTO
    Primer Lugar: 6
    Entre los tres primeros: 8


Consulta número:  10
   Imagen de consulta:  paris-saint-germain
   Mas cercanos:  [[81.38798, 'paris-saint-germain'], [196.76436, 'fc-girondins-de-bordeaux'], [201.02414, 'brighton'], [203.27312, 'manchester-city'], [210.59656, 'napoli']]
--> ACIERTO
    Primer Lugar: 7
    Entre los tres primeros: 9


Consulta número:  11
   Imagen de consulta:  elche
   Mas cercanos:  [[95.51645, 'osasuna'], [113.15926, 'augsburg'], [123.1449, 'elche'], [128.52927, 'fiorentina'], [129.31407, 'stade-brestois-29']]
--> ACIERTO
    Entre los tres primeros: 10


Consulta número:  12
   Imagen de consulta:  manchester-united
   Mas cercanos:  [[114.49684, 'manchester-united'], [117.19901, 'mallorca'], [129.65276, 'watford'], [137.76389, 'rc-lens'], [144.82475, 'genoa']]
--> ACIERTO
    Primer Lugar: 8
    Entre los tres primeros: 11


Consulta número:  13
   Imagen de consulta:  ogc-nice
   Mas cercanos:  [[68.8993, 'ogc-nice'], [127.41383, 'venezia'], [139.00371, 'everton'], [143.22809, 'udinese'], [155.26196, 'liverpool']]
--> ACIERTO
    Primer Lugar: 9
    Entre los tres primeros: 12


Consulta número:  14
   Imagen de consulta:  levante
   Mas cercanos:  [[93.19387, 'levante'], [125.36325, 'southampton'], [156.61057, 'leverkusen'], [160.70943, 'clermont-foot-63'], [171.80164, 'rc-lens']]
--> ACIERTO
    Primer Lugar: 10
    Entre los tres primeros: 13


Consulta número:  15
   Imagen de consulta:  spezia
   Mas cercanos:  [[174.51552, 'spezia'], [186.22084, 'venezia'], [195.3657, 'udinese'], [198.60167, 'ogc-nice'], [207.9398, 'liverpool']]
--> ACIERTO
    Primer Lugar: 11
    Entre los tres primeros: 14


Consulta número:  16
   Imagen de consulta:  wolfsburg
   Mas cercanos:  [[120.18054, 'wolfsburg'], [229.77289, 'getafe'], [235.85355, 'as-saint-etienne'], [249.63118, 'valencia'], [254.9981, 'bremen']]
--> ACIERTO
    Primer Lugar: 12
    Entre los tres primeros: 15


Consulta número:  17
   Imagen de consulta:  stuttgart
   Mas cercanos:  [[127.641495, 'stade-rennais-fc'], [133.8858, 'roma'], [148.37395, 'genoa'], [150.57512, 'as-monaco'], [155.85452, 'rc-lens']]


Consulta número:  18
   Imagen de consulta:  barcelona
   Mas cercanos:  [[94.53392, 'barcelona'], [180.03273, 'manchester-united'], [190.24509, 'rc-lens'], [195.51767, 'watford'], [204.30151, 'leverkusen']]
--> ACIERTO
    Primer Lugar: 13
    Entre los tres primeros: 16


Consulta número:  19
   Imagen de consulta:  inter
   Mas cercanos:  [[62.723278, 'inter'], [238.65428, 'brighton'], [290.65247, 'schalke'], [296.05304, 'sampdoria'], [300.69424, 'hannover']]
--> ACIERTO
    Primer Lugar: 14
    Entre los tres primeros: 17


Consulta número:  20
   Imagen de consulta:  roma
   Mas cercanos:  [[82.30326, 'roma'], [125.69617, 'cadiz'], [126.504, 'rc-lens'], [131.35951, 'genoa'], [149.13701, 'mallorca']]
--> ACIERTO
    Primer Lugar: 15
    Entre los tres primeros: 18


Consulta número:  21
   Imagen de consulta:  villarreal
   Mas cercanos:  [[125.70802, 'stade-rennais-fc'], [142.4496, 'villarreal'], [144.09784, 'arsenal'], [152.52571, 'levante'], [153.98248, 'clermont-foot-63']]
--> ACIERTO
    Entre los tres primeros: 19


Consulta número:  22
   Imagen de consulta:  crystal-palace
   Mas cercanos:  [[88.21929, 'crystal-palace'], [150.95245, 'real-sociedad'], [154.76587, 'tottenham-hotspur'], [156.26465, 'sampdoria'], [159.1524, 'hoffenheim']]
--> ACIERTO
    Primer Lugar: 16
    Entre los tres primeros: 20


Consulta número:  23
   Imagen de consulta:  machester-city
   Mas cercanos:  [[133.44374, 'liverpool'], [143.74725, 'real-sociedad'], [165.43085, 'crystal-palace'], [171.01364, 'stade-brestois-29'], [172.51503, 'augsburg']]


Consulta número:  24
   Imagen de consulta:  fiorentina
   Mas cercanos:  [[164.89073, 'real-betis'], [166.94626, 'losc-lille'], [167.78563, 'cagliari'], [172.7247, 'villarreal'], [175.30598, 'fiorentina']]
--> ACIERTO


Consulta número:  25
   Imagen de consulta:  moenchengladbach
   Mas cercanos:  [[81.61375, 'moenchengladbach'], [212.88861, 'fc-lorient'], [224.77762, 'tottenham-hotspur'], [225.82916, 'atalanta'], [227.9369, 'sampdoria']]
--> ACIERTO
    Primer Lugar: 17
    Entre los tres primeros: 21


Consulta número:  26
   Imagen de consulta:  napoli
   Mas cercanos:  [[121.06509, 'napoli'], [171.15, 'schalke'], [186.51323, 'estac-troyes'], [190.80046, 'olympique-de-marseille'], [200.0065, 'hoffenheim']]
--> ACIERTO
    Primer Lugar: 18
    Entre los tres primeros: 22


Consulta número:  27
   Imagen de consulta:  chelsea
   Mas cercanos:  [[35.09001, 'chelsea'], [157.8285, 'as-monaco'], [158.98758, 'arsenal'], [167.06789, 'stade-brestois-29'], [167.35435, 'olympique-lyonnais']]
--> ACIERTO
    Primer Lugar: 19
    Entre los tres primeros: 23


Consulta número:  28
   Imagen de consulta:  cagliari
   Mas cercanos:  [[71.1773, 'cagliari'], [147.87343, 'as-monaco'], [155.05753, 'real-sociedad'], [156.18446, 'losc-lille'], [160.17818, 'hoffenheim']]
--> ACIERTO
    Primer Lugar: 20
    Entre los tres primeros: 24


Consulta número:  29
   Imagen de consulta:  sampdoria
   Mas cercanos:  [[95.653984, 'sampdoria'], [129.27177, 'losc-lille'], [150.32007, 'cagliari'], [157.43767, 'deportivo-alavez'], [160.15149, 'real-sociedad']]
--> ACIERTO
    Primer Lugar: 21
    Entre los tres primeros: 25


Consulta número:  30
   Imagen de consulta:  real-betis
   Mas cercanos:  [[53.625904, 'real-betis'], [129.50468, 'as-monaco'], [139.26588, 'redbull-leipzig'], [142.28523, 'celta'], [145.81876, 'sevilla']]
--> ACIERTO
    Primer Lugar: 22
    Entre los tres primeros: 26


Consulta número:  31
   Imagen de consulta:  olympique-lyonnais
   Mas cercanos:  [[70.44129, 'olympique-lyonnais'], [102.121994, 'as-monaco'], [125.04186, 'arsenal'], [136.27776, 'stade-brestois-29'], [141.85501, 'granada']]
--> ACIERTO
    Primer Lugar: 23
    Entre los tres primeros: 27


Consulta número:  32
   Imagen de consulta:  juventus
   Mas cercanos:  [[140.59485, 'juventus'], [198.04439, 'bologna'], [219.03279, 'hertha-bsc-berlin'], [224.4586, 'tottenham-hotspur'], [229.70226, 'hellas-verona']]
--> ACIERTO
    Primer Lugar: 24
    Entre los tres primeros: 28


Consulta número:  33
   Imagen de consulta:  getafe
   Mas cercanos:  [[125.30669, 'hellas-verona'], [128.10179, 'augsburg'], [130.50748, 'liverpool'], [131.05672, 'stade-brestois-29'], [136.55054, 'everton']]


Consulta número:  34
   Imagen de consulta:  brentford
   Mas cercanos:  [[123.450005, 'stade-brestois-29'], [136.08727, 'arsenal'], [137.96721, 'brentford'], [151.02046, 'stade-rennais-fc'], [160.84712, 'olympique-lyonnais']]
--> ACIERTO
    Entre los tres primeros: 29


Consulta número:  35
   Imagen de consulta:  watford
   Mas cercanos:  [[210.5813, 'moenchengladbach'], [233.18083, 'angers-sco'], [236.47787, 'fc-lorient'], [257.56308, 'fc-girondins-de-bordeaux'], [258.9094, 'venezia']]


Consulta número:  36
   Imagen de consulta:  bologna
   Mas cercanos:  [[129.75977, 'bologna'], [188.94481, 'athletic'], [207.22763, 'wolfsburg'], [209.58481, 'redbull-leipzig'], [210.87381, 'hertha-bsc-berlin']]
--> ACIERTO
    Primer Lugar: 25
    Entre los tres primeros: 30


Consulta número:  37
   Imagen de consulta:  norwich-city
   Mas cercanos:  [[184.09526, 'norwich-city'], [271.09982, 'aston-villa'], [271.4266, 'watford'], [274.1177, 'espanyol'], [279.53372, 'getafe']]
--> ACIERTO
    Primer Lugar: 26
    Entre los tres primeros: 31


Consulta número:  38
   Imagen de consulta:  granada
   Mas cercanos:  [[76.11131, 'granada'], [151.76416, 'as-monaco'], [155.04309, 'clermont-foot-63'], [161.2738, 'arsenal'], [175.37965, 'olympique-lyonnais']]
--> ACIERTO
    Primer Lugar: 27
    Entre los tres primeros: 32


Consulta número:  39
   Imagen de consulta:  valencia
   Mas cercanos:  [[136.6158, 'villarreal'], [146.619, 'sevilla'], [156.3357, 'rc-lens'], [158.2743, 'roma'], [164.03381, 'real-sociedad']]


Consulta número:  40
   Imagen de consulta:  mainz
   Mas cercanos:  [[131.41602, 'mainz'], [175.16109, 'espanyol'], [196.18694, 'frankfurt'], [197.18005, 'dusseldorf'], [220.8171, 'nuremberg']]
--> ACIERTO
    Primer Lugar: 28
    Entre los tres primeros: 33


Consulta número:  41
   Imagen de consulta:  bayern
   Mas cercanos:  [[13.689358, 'bayern'], [134.98244, 'arsenal'], [146.93027, 'clermont-foot-63'], [167.75168, 'stade-brestois-29'], [170.86896, 'losc-lille']]
--> ACIERTO
    Primer Lugar: 29
    Entre los tres primeros: 34


Consulta número:  42
   Imagen de consulta:  fc-lorient
   Mas cercanos:  [[86.74836, 'fc-lorient'], [174.26283, 'athletic'], [191.3807, 'cagliari'], [197.57932, 'sampdoria'], [201.16418, 'as-monaco']]
--> ACIERTO
    Primer Lugar: 30
    Entre los tres primeros: 35


Consulta número:  43
   Imagen de consulta:  west-ham-united
   Mas cercanos:  [[43.719654, 'west-ham-united'], [136.22346, 'arsenal'], [138.48985, 'stade-brestois-29'], [139.14223, 'olympique-lyonnais'], [148.18158, 'augsburg']]
--> ACIERTO
    Primer Lugar: 31
    Entre los tres primeros: 36


Consulta número:  44
   Imagen de consulta:  genoa
   Mas cercanos:  [[127.136955, 'genoa'], [153.33733, 'watford'], [156.21388, 'rc-lens'], [168.73497, 'aston-villa'], [174.83865, 'levante']]
--> ACIERTO
    Primer Lugar: 32
    Entre los tres primeros: 37


Consulta número:  45
   Imagen de consulta:  hellas-verona
   Mas cercanos:  [[231.90053, 'cadiz'], [234.057, 'rayo-vallecano'], [269.8974, 'rc-lens'], [273.1825, 'burnley'], [284.66687, 'real-madrid']]


Consulta número:  46
   Imagen de consulta:  liverpool
   Mas cercanos:  [[153.52292, 'granada'], [170.25, 'clermont-foot-63'], [171.81348, 'as-monaco'], [178.70331, 'elche'], [180.00386, 'real-sociedad']]


Consulta número:  47
   Imagen de consulta:  southampton
   Mas cercanos:  [[131.43153, 'newcastle-united'], [144.5407, 'redbull-leipzig'], [145.8664, 'roma'], [156.24104, 'as-monaco'], [157.39444, 'liverpool']]


Consulta número:  48
   Imagen de consulta:  burnley
   Mas cercanos:  [[95.38013, 'burnley'], [253.27936, 'leeds-united'], [264.38556, 'barcelona'], [271.0974, 'rayo-vallecano'], [277.79376, 'manchester-united']]
--> ACIERTO
    Primer Lugar: 33
    Entre los tres primeros: 38


Consulta número:  49
   Imagen de consulta:  hoffenheim
   Mas cercanos:  [[14.075365, 'hoffenheim'], [129.97087, 'hellas-verona'], [138.04749, 'hamburg'], [156.4472, 'crystal-palace'], [162.02165, 'sampdoria']]
--> ACIERTO
    Primer Lugar: 34
    Entre los tres primeros: 39


Consulta número:  50
   Imagen de consulta:  dortmund
   Mas cercanos:  [[162.31314, 'dortmund'], [257.5092, 'burnley'], [375.19934, 'leeds-united'], [406.7703, 'barcelona'], [409.39062, 'manchester-united']]
--> ACIERTO
    Primer Lugar: 35
    Entre los tres primeros: 40


Consulta número:  51
   Imagen de consulta:  freiburg
   Mas cercanos:  [[140.26344, 'freiburg'], [200.34201, 'venezia'], [206.96225, 'fc-girondins-de-bordeaux'], [212.60368, 'hoffenheim'], [212.84944, 'spezia']]
--> ACIERTO
    Primer Lugar: 36
    Entre los tres primeros: 41


Consulta número:  52
   Imagen de consulta:  redbull-leipzig
   Mas cercanos:  [[24.78521, 'redbull-leipzig'], [122.33429, 'celta'], [132.62956, 'furth'], [137.72348, 'lazio'], [145.07922, 'elche']]
--> ACIERTO
    Primer Lugar: 37
    Entre los tres primeros: 42


Consulta número:  53
   Imagen de consulta:  espanyol
   Mas cercanos:  [[96.274315, 'espanyol'], [191.87901, 'stade-de-reims'], [198.33716, 'watford'], [208.27542, 'mainz'], [227.58557, 'norwich-city']]
--> ACIERTO
    Primer Lugar: 38
    Entre los tres primeros: 43


Consulta número:  54
   Imagen de consulta:  schalke
   Mas cercanos:  [[77.41234, 'schalke'], [173.33244, 'brighton'], [226.11052, 'hoffenheim'], [226.76036, 'bologna'], [229.1192, 'hertha-bsc-berlin']]
--> ACIERTO
    Primer Lugar: 39
    Entre los tres primeros: 44


Porcentaje de aciertos NN1:  0.7090909090909091
Porcentaje de aciertos NN3:  0.8
Porcentaje de aciertos NN5:  0.8181818181818182
  Cantidad consultas:  55
```
