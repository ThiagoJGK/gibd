---
aliases: [TL_FT_Resnet50_logoclubes_08-08-24]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2024-08-08
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Transfer Learning y Fine Tuning/Pruebas 2024/TL_FT_Resnet50_logoclubes_08-08-24.ipynb"
tamanio_bytes: 114319314
---

# Notebook: TL_FT_Resnet50_logoclubes_08-08-24.ipynb

Ruta interna: `GIBD/Transfer Learning y Fine Tuning/Pruebas 2024/TL_FT_Resnet50_logoclubes_08-08-24.ipynb`

---

## Imports


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
import os
print(os.listdir('G:\Mi unidad\GIBD'))
```


*Salida:*
```text
['Papers', 'Analisis de resultados de los intentos fallidos.gdoc', 'Trabajos parecidos', 'Publicaciones Few Shot Learning and Augumentation.gdoc', 'Tareas Iván', 'Copia de PresentacionesConaiisi2022.pptx', 'Presentacion Conaiisi 2022 - Articulo 210.pptx', 'Bases de Datos Mundiales de Prueba de ML.gdoc', 'Código útil', 'Herramientas', 'Tareas y Notas.gdoc', 'Procesamiento del Lenguaje Natural', 'Summary modelTriplet.txt', 'Calendario Congresos - Revistas.gsheet', 'ARTICULOS NUESTROS', 'E.C de imágenes a color', 'Presentaciones', 'Datos facturacion UTN FRCU.gdoc', 'Categorizacion UTN', 'Despedida GIBD.gsheet', 'BERT', 'Transfer Learning y Fine Tuning', 'CONAISII  clubes 2023', '2024', 'CNN Marcas', 'desktop.ini']
```


**[Celda 4 - Código]**
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Dropout, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.layers import Lambda
```

                                                                               
                                                                                
                                                                                 
                                                                                  
                                                                                   
                                                                                    
                                                                                     
                                                                                      
                                                                                       
                                                                                        
                                                                                         
                                                                                          
                                                                                           
                                                                                            
                                                                                             
                                                                                              
                                                                                               
                                                                                                
                                                                                                 
                                                                                                  
                                                                                                   
                                                                                                    
                                                                                                     
                                                                                                      ## Creación del modelo

## Definición de funciones


**[Celda 7 - Código]**
```python
lista=[]
nnk(lista, [3, 1])
nnk(lista, [5, 2])
nnk(lista, [2, 3])
nnk(lista, [7, 4])
nnk(lista, [1, 5])
nnk(lista, [6, 6])
nnk(lista, [4, 7])
print(lista)
nnk2etiquetas(lista,[10,11,12,13,14,15,16,17])
```


*Salida:*
```text
[[1, 5], [2, 3], [3, 1], [4, 7], [5, 2]]
[[1, 15], [2, 13], [3, 11], [4, 17], [5, 12]]
```


**[Celda 8 - Código]**
```python
def nnk(actuales,nuevo,k=5):
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


**[Celda 12 - Código]**
```python
#NO LO USAMOS
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


**[Celda 13 - Código]**
```python
### CARGAR Y NORMALIZAR LAS IMAGENES DE LA BD Y LAS CONSULTAS
from PIL import Image, ImageEnhance

target_size = (224, 224)  # El tamaño objetivo después de recortar y redimensionar

def crop_and_normalize_images(orig_dir, image_filenames, target_size):
    normalized_images = []
    etiquetas = []

    for filename in image_filenames:
        path = os.path.join(orig_dir, filename)
        print(path)
        image = Image.open(path)
        image = image.resize(target_size)
                
        # Convertir a 24 bits (RGB)
        # image = image.convert('RGB')

        # nombre = etiqueta
        filename = os.path.basename(path)
        nombre=filename[:filename.find(".")]

        # Verificar si la imagen tiene un canal alfa
        has_alpha = 'A' in image.mode

        # Convertir la imagen en un array numpy
        image_array = np.array(image)

        # print(image_array)
        if image_array.shape[2] == 4:
            for col in range(image_array.shape[0]):
                for row in range(image_array.shape[1]):
                    if image_array[col, row, 3] < 100:  # Si es transparente
                        image_array[col, row, :3] = [255, 255, 255]  # Convierte a blanco
            image_array = image_array[:, :, :3]
        #if has_alpha:
        #    print('TIENE ALPHA...')
            # Encontrar las filas y columnas donde todos los valores alfa son igual a 0
            # non_empty_rows_alpha = np.where(image_array[:, :, 3] < 20)[0]
            # print(non_empty_rows_alpha)
            # non_empty_columns_alpha = np.where(image_array[:, :, 3] <20)[1]
            # Remover Alpha Channel
        #    for col in range(image_array.shape[0]):
        #        for row in range(image_array.shape[1]):
        #            for chan in range(3):
        #                if image_array[col, row, 3]<100:  # si es transparente
        #                    image_array[col, row, chan] = 255   # lo convierto a blanco, porque luego elimino el canal alpha
            # elimino el canal alpha
        #    image_array = image_array[:,:,:3]
        
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
        cropped_resized_image = image.crop(crop_box).resize(target_size, Image.LANCZOS) # Image.Resampling.LANCZOS

        # Convertir la imagen recortada y redimensionada en un array numpy y normalizar
        # normalized_image = (np.array(cropped_resized_image) - np.mean(cropped_resized_image)) / np.std(cropped_resized_image)
        normalized_image = np.array(cropped_resized_image)
        
        blured_image = cv2.GaussianBlur(normalized_image, (15, 15), 0)

        normalized_image = blured_image / 255.0
        
        if normalized_image.shape[2]==3:
            etiquetas.append(nombre)
            print(nombre, ': ',normalized_image.shape)
            normalized_images.append(normalized_image) 
            

    print(normalized_images[0].shape)
    
    # Verifica que todas las imágenes tengan la misma forma
    formas = [img.shape for img in normalized_images]
    if all(f == formas[0] for f in formas):
        array_conjunto = np.array(normalized_images)
        print("Forma del arreglo combinado:", array_conjunto.shape)
    else:
        print("Las imágenes tienen formas inconsistentes:", formas)
        
    return np.array(normalized_images), np.array(etiquetas)


```


**[Celda 14 - Código]**
```python
## CARGAR LAS RUTAS DE CADA IMAGEN DE CONSULTA
##orig_dir = r'/content/drive/MyDrive/GIBD/E.C de imágenes a color/E.C archivos/consultas_normalizadas'
##orig_dir = r'C:\Users\Usuario\Desktop\Logos\consultas_normalizadas'
orig_dir = r'G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80'

query_paths = []

for filename in sorted(os.listdir(orig_dir)):
    if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
        # Ruta completa del archivo
        query_paths.append(filename)

print(query_paths)
## NORMALIZAR LAS IMAGENES

consultas, y_consultas = crop_and_normalize_images(orig_dir, query_paths, target_size)
print(len(consultas))

```


*Salida:*
```text
['1.png', '102462.png', '102474.png', '102476.png', '102852.png', '103285.png', '103287.png', '108510.png', '1099.png', '1100.png', '1111.png', '1114.png', '1126.png', '1129.png', '1132.png', '1135.png', '1139.png', '1150.png', '1167.png', '1173.png', '1174.png', '1178.png', '1179.png', '121206.png', '1253.png', '1664.png', '1668.png', '1687.png', '1707.png', '1710.png', '1714.png', '1717.png', '1724.png', '1733.png', '1737.png', '1742.png', '1749.png', '1775.png', '1777.png', '1885.png', '1916.png', '1919.png', '1922.png', '1923.png', '1924.png', '1926.png', '1992.png', '2097.png', '2238.png', '602.png', '603.png', '617.png', '622.png', '630.png', '642.png', '676.png', '679.png', '680.png', '691.png', '713.png', '728.png', '732.png', '735.png', '78.png', '82.png', '858.png', '862.png', '865.png', '868.png', '87.png', '89.png', '907.png', '908.png', '918.png', '920.png', '921.png', '93.png', '944.png', '96.png', '961.png']
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1.png
13 210 0 223
1 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\102462.png
0 223 0 223
102462 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\102474.png
2 222 2 220
102474 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\102476.png
32 191 5 218
102476 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\102852.png
25 197 0 223
102852 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\103285.png
0 223 0 223
103285 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\103287.png
4 220 4 219
103287 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\108510.png
4 222 1 223
108510 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1099.png
0 223 0 223
1099 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1100.png
35 189 12 212
1100 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1111.png
42 181 4 219
1111 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1114.png
0 223 0 223
1114 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1126.png
35 189 4 219
1126 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1129.png
5 218 30 193
1129 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1132.png
36 187 4 218
1132 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1135.png
11 213 11 212
1135 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1139.png
41 176 7 220
1139 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1150.png
0 223 0 223
1150 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1167.png
27 196 5 218
1167 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1173.png
0 223 0 223
1173 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1174.png
1 222 1 222
1174 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1178.png
0 223 3 219
1178 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1179.png
42 181 40 181
1179 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\121206.png
13 210 5 218
121206 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1253.png
3 220 3 220
1253 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1664.png
18 206 3 219
1664 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1668.png
20 203 20 203
1668 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1687.png
30 192 3 219
1687 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1707.png
0 223 0 223
1707 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1710.png
14 209 9 214
1710 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1714.png
70 154 0 223
1714 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1717.png
0 223 0 222
1717 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1724.png
0 223 0 223
1724 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1733.png
2 221 0 221
1733 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1737.png
36 186 7 215
1737 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1742.png
5 213 4 211
1742 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1749.png
49 173 2 221
1749 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1775.png
44 179 23 200
1775 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1777.png
22 200 0 221
1777 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1885.png
0 223 0 223
1885 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1916.png
16 207 0 223
1916 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1919.png
31 191 14 208
1919 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1922.png
1 223 0 223
1922 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1923.png
26 197 6 218
1923 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1924.png
34 187 3 218
1924 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1926.png
30 193 10 214
1926 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\1992.png
0 223 0 222
1992 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\2097.png
38 185 6 214
2097 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\2238.png
29 193 4 219
2238 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\602.png
0 223 0 223
602 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\603.png
31 191 0 223
603 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\617.png
12 211 13 212
617 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\622.png
20 203 5 218
622 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\630.png
5 218 5 218
630 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\642.png
23 200 0 223
642 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\676.png
29 193 1 223
676 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\679.png
25 198 27 199
679 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\680.png
5 219 5 218
680 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\691.png
0 223 0 223
691 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\713.png
18 205 4 218
713 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\728.png
77 146 26 197
728 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\732.png
0 223 0 223
732 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\735.png
0 223 0 221
735 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\78.png
29 194 11 213
78 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\82.png
6 217 5 218
82 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\858.png
1 222 1 222
858 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\862.png
22 201 0 223
862 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\865.png
20 203 5 218
865 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\868.png
6 217 6 217
868 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\87.png
45 178 10 213
87 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\89.png
1 222 1 222
89 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\907.png
5 218 5 218
907 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\908.png
42 180 2 221
908 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\918.png
4 219 7 223
918 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\920.png
0 223 0 223
920 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\921.png
14 209 14 209
921 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\93.png
2 221 2 220
93 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\944.png
0 223 0 223
944 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\96.png
23 200 11 213
96 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\consultas80\961.png
5 219 5 218
961 :  (224, 224, 3)
(224, 224, 3)
Forma del arreglo combinado: (80, 224, 224, 3)
80
```


**[Celda 15 - Código]**
```python
#  CARGAR BASE DE DATOS

## CARGAR LAS RUTAS DE CADA IMAGEN DE LA BASE DE DATOS
##orig_dir = r'/content/drive/MyDrive/GIBD/CNN Marcas/Bases de Datos/Escudos Normalizados PNG'

orig_dir = r'G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k'

image_paths = []
i = 0
for filename in sorted(os.listdir(orig_dir)):
    if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
              
        if len(image_paths)<2000:
            image_paths.append(filename)
    i += 1

print(image_paths)
for query in query_paths:
    if query not in image_paths:
        image_paths.append(query)
    
## NORMALIZAR LAS IMAGENES
imagenesBD, y_imagenesBD = crop_and_normalize_images(orig_dir, image_paths, target_size)
print(len(imagenesBD))
print(len(y_imagenesBD))


```


*Salida:*
```text
['1.png', '1000.png', '1001.png', '1002.png', '1003.png', '100349.png', '100351.png', '100352.png', '100353.png', '100354.png', '100355.png', '1004.png', '100473.png', '100479.png', '100488.png', '1005.png', '100604.png', '100605.png', '1007.png', '100729.png', '100795.png', '100796.png', '1008.png', '100806.png', '100818.png', '100881.png', '100882.png', '100883.png', '100884.png', '100885.png', '100893.png', '100896.png', '1009.png', '100904.png', '100909.png', '100911.png', '100912.png', '100913.png', '100914.png', '100915.png', '100916.png', '100918.png', '100921.png', '100922.png', '100931.png', '100933.png', '100936.png', '100939.png', '100940.png', '100952.png', '100987.png', '101.png', '1010.png', '101000.png', '101012.png', '101013.png', '101016.png', '101017.png', '101018.png', '101020.png', '101026.png', '101045.png', '101058.png', '1011.png', '101148.png', '101149.png', '101151.png', '101154.png', '1012.png', '101241.png', '101242.png', '101243.png', '101261.png', '101266.png', '101267.png', '1013.png', '101345.png', '101395.png', '1014.png', '101448.png', '101471.png', '101475.png', '101493.png', '1015.png', '101504.png', '1016.png', '101606.png', '101687.png', '1017.png', '101773.png', '101776.png', '101780.png', '101789.png', '101797.png', '1019.png', '102.png', '1020.png', '102009.png', '102010.png', '102012.png', '102013.png', '102014.png', '102019.png', '102022.png', '102025.png', '102029.png', '102036.png', '102092.png', '1021.png', '102170.png', '1022.png', '1023.png', '102355.png', '102356.png', '102358.png', '102359.png', '102360.png', '102361.png', '102362.png', '102363.png', '102364.png', '102365.png', '102366.png', '102367.png', '102368.png', '102369.png', '102370.png', '102371.png', '102372.png', '102373.png', '102374.png', '102375.png', '102390.png', '1024.png', '102462.png', '102464.png', '102466.png', '102467.png', '102470.png', '102472.png', '102474.png', '102476.png', '102477.png', '102479.png', '102482.png', '102483.png', '102485.png', '102486.png', '102487.png', '102489.png', '102490.png', '102491.png', '102493.png', '102495.png', '1025.png', '102555.png', '1026.png', '1027.png', '1028.png', '102849.png', '102850.png', '102851.png', '102852.png', '102853.png', '102854.png', '102855.png', '102856.png', '102857.png', '102858.png', '102859.png', '102860.png', '102861.png', '102862.png', '102863.png', '102864.png', '102884.png', '102885.png', '102886.png', '102887.png', '102889.png', '102890.png', '1029.png', '102924.png', '102957.png', '102959.png', '103.png', '1030.png', '1031.png', '1032.png', '103278.png', '103279.png', '103282.png', '103283.png', '103284.png', '103285.png', '103286.png', '103287.png', '1033.png', '1034.png', '1036.png', '1037.png', '1038.png', '1039.png', '104.png', '1041.png', '1042.png', '1043.png', '104359.png', '104360.png', '104361.png', '104362.png', '104363.png', '104386.png', '104388.png', '104395.png', '104396.png', '1044.png', '1045.png', '1046.png', '104648.png', '104650.png', '104651.png', '104652.png', '104653.png', '1047.png', '104749.png', '104750.png', '104775.png', '104776.png', '104868.png', '104870.png', '104872.png', '104873.png', '104874.png', '104884.png', '104885.png', '104887.png', '1049.png', '104975.png', '104978.png', '104979.png', '104982.png', '104988.png', '104997.png', '104999.png', '105.png', '1050.png', '105000.png', '105013.png', '105026.png', '105030.png', '1051.png', '105174.png', '105175.png', '105176.png', '105177.png', '105178.png', '105180.png', '105181.png', '105187.png', '105188.png', '1052.png', '1053.png', '1054.png', '1055.png', '1056.png', '105611.png', '105612.png', '105613.png', '105707.png', '1058.png', '105898.png', '1059.png', '105989.png', '105990.png', '105991.png', '105992.png', '105993.png', '105994.png', '105995.png', '105996.png', '105997.png', '105998.png', '105999.png', '1060.png', '106000.png', '106001.png', '106027.png', '106028.png', '106029.png', '106052.png', '106053.png', '106054.png', '106063.png', '106071.png', '106072.png', '1061.png', '106190.png', '1062.png', '1063.png', '106363.png', '106371.png', '1064.png', '106447.png', '1065.png', '106578.png', '106580.png', '106583.png', '1066.png', '106691.png', '106694.png', '106696.png', '106730.png', '106745.png', '106746.png', '106747.png', '106748.png', '106749.png', '106755.png', '106756.png', '106757.png', '106780.png', '106781.png', '1068.png', '106805.png', '106808.png', '106809.png', '106811.png', '106812.png', '106813.png', '106814.png', '106815.png', '106816.png', '106817.png', '106818.png', '106844.png', '106849.png', '106857.png', '106858.png', '106881.png', '1069.png', '1070.png', '1071.png', '1072.png', '107201.png', '107203.png', '107205.png', '107206.png', '107208.png', '107210.png', '107212.png', '107214.png', '107216.png', '107218.png', '107220.png', '107222.png', '107224.png', '107226.png', '107228.png', '107230.png', '107232.png', '107234.png', '107236.png', '107238.png', '107240.png', '107242.png', '107280.png', '107283.png', '107285.png', '107287.png', '107289.png', '107291.png', '107293.png', '107296.png', '107298.png', '1073.png', '107301.png', '107303.png', '107305.png', '107307.png', '107309.png', '107311.png', '107313.png', '107314.png', '107319.png', '107320.png', '107321.png', '107322.png', '107323.png', '107324.png', '107325.png', '107326.png', '107327.png', '107328.png', '107329.png', '107330.png', '107332.png', '107333.png', '107334.png', '107335.png', '107336.png', '107337.png', '107338.png', '107339.png', '107340.png', '107341.png', '107342.png', '107343.png', '107344.png', '107345.png', '107346.png', '107347.png', '107348.png', '107349.png', '107350.png', '107351.png', '107352.png', '107353.png', '107354.png', '107355.png', '107356.png', '107357.png', '107358.png', '107359.png', '107360.png', '107361.png', '107362.png', '107363.png', '107364.png', '107365.png', '107366.png', '107367.png', '107368.png', '107369.png', '107370.png', '1074.png', '1076.png', '107673.png', '107674.png', '107675.png', '107676.png', '107677.png', '107678.png', '107679.png', '107680.png', '107681.png', '107682.png', '107683.png', '107684.png', '1077.png', '1078.png', '1079.png', '1080.png', '1081.png', '1082.png', '1083.png', '1084.png', '1085.png', '108510.png', '108514.png', '108517.png', '108522.png', '108524.png', '108526.png', '108531.png', '108538.png', '108542.png', '108545.png', '108546.png', '108547.png', '108567.png', '108569.png', '108571.png', '108574.png', '108576.png', '108578.png', '1086.png', '1087.png', '1088.png', '108891.png', '108893.png', '108896.png', '108898.png', '1089.png', '108905.png', '108907.png', '108909.png', '108911.png', '108913.png', '108917.png', '108919.png', '108921.png', '108923.png', '108925.png', '108926.png', '108928.png', '108930.png', '108932.png', '108934.png', '108936.png', '108938.png', '108940.png', '108942.png', '108944.png', '108950.png', '108952.png', '108954.png', '108955.png', '108979.png', '108981.png', '108983.png', '108985.png', '108986.png', '108988.png', '108990.png', '108993.png', '108995.png', '108997.png', '1090.png', '109001.png', '109003.png', '109005.png', '109007.png', '109009.png', '109011.png', '109013.png', '109015.png', '109017.png', '109019.png', '109021.png', '109023.png', '109025.png', '109027.png', '109029.png', '109031.png', '109034.png', '109035.png', '109037.png', '109038.png', '109040.png', '109041.png', '109042.png', '109043.png', '109044.png', '1091.png', '109103.png', '109105.png', '109106.png', '109108.png', '109113.png', '109114.png', '109116.png', '109118.png', '109120.png', '109121.png', '109122.png', '109123.png', '109132.png', '109133.png', '109177.png', '109178.png', '109179.png', '109180.png', '109188.png', '109189.png', '109190.png', '109192.png', '109206.png', '109210.png', '109211.png', '109212.png', '109282.png', '109283.png', '109284.png', '109285.png', '109286.png', '1093.png', '109301.png', '109303.png', '109305.png', '1094.png', '1095.png', '1096.png', '1097.png', '1098.png', '1099.png', '1100.png', '1101.png', '1102.png', '1103.png', '1104.png', '1105.png', '1106.png', '1107.png', '1108.png', '1109.png', '1110.png', '1111.png', '1112.png', '1113.png', '1114.png', '1115.png', '1116.png', '1117.png', '1118.png', '1119.png', '1120.png', '112031.png', '1121.png', '1122.png', '1123.png', '1124.png', '1125.png', '1126.png', '1127.png', '1128.png', '1129.png', '1130.png', '1131.png', '1132.png', '1133.png', '1134.png', '1135.png', '113522.png', '1136.png', '1138.png', '1139.png', '1140.png', '1141.png', '1142.png', '1143.png', '1144.png', '1145.png', '114576.png', '114579.png', '114582.png', '1146.png', '1147.png', '1148.png', '114803.png', '114804.png', '114809.png', '114819.png', '114821.png', '1149.png', '114946.png', '114947.png', '114948.png', '114951.png', '114952.png', '1150.png', '115000.png', '115001.png', '115039.png', '1151.png', '1152.png', '115257.png', '1153.png', '1154.png', '1156.png', '1157.png', '115711.png', '115755.png', '1158.png', '1159.png', '115940.png', '115942.png', '115943.png', '115944.png', '115945.png', '115946.png', '115988.png', '1160.png', '1161.png', '116138.png', '116140.png', '116144.png', '116145.png', '116150.png', '116152.png', '116156.png', '116164.png', '116192.png', '116194.png', '1162.png', '116204.png', '116288.png', '1163.png', '116303.png', '116309.png', '116315.png', '116331.png', '116334.png', '116347.png', '116384.png', '1164.png', '116403.png', '1165.png', '1166.png', '1167.png', '1168.png', '1169.png', '1170.png', '117033.png', '1171.png', '117183.png', '117199.png', '1173.png', '117327.png', '1174.png', '1175.png', '1176.png', '117669.png', '1177.png', '117754.png', '1178.png', '1179.png', '1181.png', '1183.png', '1184.png', '1185.png', '1186.png', '1187.png', '1188.png', '1189.png', '1190.png', '1191.png', '1192.png', '1193.png', '1194.png', '1195.png', '1196.png', '1197.png', '119741.png', '1198.png', '1199.png', '1200.png', '120063.png', '120064.png', '120065.png', '120066.png', '120271.png', '120286.png', '120288.png', '120386.png', '1204.png', '120533.png', '120546.png', '120559.png', '1206.png', '1207.png', '120779.png', '120782.png', '120783.png', '1208.png', '120921.png', '120924.png', '120936.png', '120939.png', '120942.png', '120946.png', '120955.png', '120966.png', '120974.png', '120979.png', '120991.png', '120994.png', '120998.png', '1210.png', '121002.png', '1211.png', '121106.png', '121135.png', '121170.png', '121171.png', '121172.png', '121173.png', '121174.png', '121175.png', '121179.png', '121181.png', '121182.png', '121183.png', '121184.png', '121185.png', '121186.png', '121187.png', '121188.png', '121189.png', '121190.png', '121191.png', '121192.png', '121196.png', '121197.png', '121198.png', '1212.png', '121200.png', '121201.png', '121202.png', '121203.png', '121204.png', '121205.png', '121206.png', '121207.png', '121208.png', '121209.png', '121212.png', '121214.png', '121215.png', '121216.png', '121218.png', '121219.png', '121220.png', '121262.png', '121265.png', '121267.png', '1213.png', '1215.png', '1216.png', '1218.png', '1219.png', '1220.png', '1221.png', '1222.png', '1223.png', '1224.png', '1225.png', '1226.png', '1227.png', '1228.png', '1229.png', '122985.png', '122986.png', '122988.png', '122990.png', '122992.png', '122993.png', '122994.png', '122995.png', '122997.png', '122999.png', '1230.png', '123001.png', '123004.png', '123005.png', '123007.png', '123009.png', '1231.png', '1232.png', '1233.png', '1235.png', '1236.png', '1237.png', '1238.png', '1239.png', '1240.png', '1241.png', '1242.png', '1243.png', '1244.png', '1245.png', '1246.png', '1247.png', '1248.png', '1249.png', '1250.png', '1251.png', '1252.png', '1253.png', '1254.png', '1255.png', '1257.png', '1258.png', '1259.png', '1260.png', '1261.png', '1262.png', '126302.png', '1264.png', '126415.png', '126416.png', '126417.png', '126419.png', '126421.png', '126422.png', '126423.png', '126424.png', '126426.png', '126427.png', '1265.png', '1266.png', '1267.png', '1268.png', '1272.png', '1273.png', '127397.png', '1274.png', '1275.png', '1276.png', '1277.png', '127742.png', '127787.png', '1278.png', '1279.png', '127969.png', '127999.png', '1280.png', '128000.png', '128051.png', '1281.png', '128172.png', '1282.png', '128271.png', '128273.png', '128290.png', '1283.png', '1284.png', '128495.png', '1285.png', '128553.png', '128554.png', '128650.png', '128652.png', '128654.png', '128656.png', '1287.png', '1288.png', '1289.png', '1290.png', '1291.png', '129129.png', '129132.png', '129135.png', '129138.png', '129140.png', '129143.png', '129146.png', '129149.png', '129152.png', '129153.png', '129156.png', '129159.png', '129162.png', '129165.png', '129166.png', '129167.png', '129168.png', '129169.png', '129170.png', '129171.png', '129172.png', '129173.png', '129174.png', '129175.png', '129176.png', '129177.png', '129178.png', '129179.png', '129180.png', '129181.png', '129182.png', '129183.png', '129184.png', '129185.png', '129186.png', '129187.png', '129188.png', '129189.png', '129190.png', '129191.png', '129192.png', '129193.png', '129194.png', '129195.png', '129196.png', '129197.png', '129198.png', '129199.png', '1292.png', '129200.png', '129201.png', '129202.png', '129203.png', '129204.png', '129205.png', '129206.png', '129208.png', '129209.png', '129210.png', '129211.png', '129212.png', '1293.png', '129380.png', '129384.png', '129389.png', '129390.png', '129391.png', '129392.png', '129393.png', '129395.png', '129396.png', '129397.png', '129398.png', '129399.png', '1294.png', '129400.png', '129401.png', '129402.png', '129403.png', '129404.png', '129405.png', '129406.png', '129407.png', '129408.png', '129409.png', '129410.png', '129411.png', '129412.png', '129413.png', '129414.png', '129415.png', '129416.png', '129417.png', '129418.png', '129419.png', '129420.png', '129421.png', '129423.png', '129424.png', '129425.png', '129426.png', '129427.png', '129428.png', '129429.png', '129430.png', '129431.png', '129432.png', '129433.png', '129434.png', '129435.png', '1295.png', '129563.png', '129564.png', '129565.png', '129567.png', '129568.png', '129569.png', '129570.png', '129571.png', '129572.png', '129573.png', '129575.png', '129577.png', '129580.png', '129583.png', '129584.png', '129585.png', '1296.png', '129653.png', '129656.png', '129657.png', '129658.png', '129659.png', '129660.png', '129661.png', '129662.png', '129663.png', '129664.png', '129665.png', '129666.png', '129667.png', '129668.png', '129682.png', '129683.png', '129685.png', '129686.png', '129688.png', '129689.png', '129690.png', '129691.png', '129692.png', '129693.png', '129694.png', '129695.png', '129696.png', '129697.png', '129698.png', '129699.png', '1297.png', '129700.png', '129701.png', '129702.png', '129703.png', '129704.png', '129705.png', '129706.png', '129707.png', '129708.png', '129709.png', '129735.png', '129738.png', '129739.png', '129740.png', '129741.png', '1298.png', '129824.png', '129825.png', '129826.png', '129827.png', '129828.png', '129829.png', '129830.png', '129831.png', '129832.png', '129833.png', '129834.png', '129835.png', '129836.png', '129837.png', '129838.png', '129839.png', '129840.png', '129841.png', '129842.png', '129843.png', '129844.png', '129845.png', '129846.png', '129847.png', '129848.png', '129849.png', '129850.png', '129851.png', '129852.png', '129853.png', '129854.png', '129855.png', '129856.png', '129857.png', '129858.png', '129859.png', '129860.png', '129861.png', '129862.png', '129863.png', '129864.png', '129865.png', '129866.png', '129867.png', '129868.png', '129869.png', '129870.png', '129871.png', '129872.png', '129873.png', '129874.png', '129875.png', '129876.png', '129877.png', '129878.png', '129879.png', '129880.png', '129881.png', '129882.png', '129883.png', '129884.png', '129885.png', '129886.png', '129887.png', '129888.png', '129889.png', '129890.png', '129891.png', '129892.png', '129893.png', '129894.png', '129895.png', '129896.png', '1299.png', '1300.png', '130031.png', '130032.png', '130035.png', '130036.png', '130037.png', '130038.png', '130041.png', '130042.png', '130043.png', '130044.png', '130045.png', '130046.png', '130047.png', '130048.png', '130049.png', '130050.png', '130051.png', '130052.png', '130053.png', '130054.png', '130055.png', '130056.png', '130057.png', '130058.png', '130059.png', '130060.png', '130061.png', '130062.png', '130063.png', '130064.png', '130065.png', '130066.png', '130067.png', '1301.png', '130127.png', '130153.png', '130158.png', '130159.png', '130160.png', '130161.png', '130162.png', '130163.png', '130164.png', '130165.png', '130166.png', '130167.png', '130168.png', '130169.png', '130170.png', '130171.png', '130172.png', '130173.png', '130174.png', '130175.png', '130176.png', '130177.png', '130193.png', '130194.png', '130199.png', '1302.png', '130202.png', '130204.png', '130209.png', '130211.png', '130212.png', '130218.png', '130219.png', '130220.png', '130229.png', '130239.png', '130241.png', '130260.png', '130261.png', '130262.png', '130263.png', '130266.png', '130270.png', '130274.png', '130276.png', '130287.png', '130288.png', '130289.png', '130290.png', '130292.png', '130293.png', '130296.png', '130297.png', '130298.png', '130299.png', '1303.png', '130300.png', '130304.png', '130337.png', '130338.png', '130339.png', '130340.png', '130341.png', '130342.png', '130343.png', '130344.png', '130345.png', '130346.png', '130347.png', '130348.png', '130349.png', '130350.png', '130351.png', '130352.png', '130353.png', '130354.png', '130355.png', '130356.png', '130357.png', '130358.png', '130359.png', '130360.png', '130361.png', '130362.png', '130363.png', '130364.png', '130365.png', '130366.png', '130367.png', '130368.png', '130369.png', '130370.png', '130371.png', '130372.png', '130373.png', '130374.png', '130375.png', '130376.png', '130377.png', '130378.png', '130379.png', '130380.png', '130381.png', '130382.png', '1304.png', '130495.png', '130496.png', '130497.png', '130498.png', '130499.png', '1305.png', '130500.png', '130501.png', '130502.png', '130503.png', '130504.png', '130505.png', '130506.png', '130507.png', '130508.png', '130509.png', '130510.png', '130511.png', '130512.png', '130513.png', '130514.png', '130515.png', '130516.png', '130517.png', '130518.png', '130519.png', '130520.png', '130521.png', '130522.png', '130523.png', '130524.png', '130525.png', '130526.png', '130527.png', '130528.png', '130529.png', '130530.png', '130531.png', '130532.png', '130534.png', '130535.png', '130536.png', '130537.png', '130538.png', '130539.png', '130540.png', '130541.png', '130542.png', '130543.png', '130544.png', '130545.png', '130546.png', '130547.png', '130548.png', '130549.png', '130550.png', '130551.png', '130552.png', '130553.png', '130554.png', '130555.png', '130556.png', '130557.png', '130558.png', '130559.png', '130560.png', '130561.png', '130562.png', '130563.png', '130591.png', '130592.png', '130593.png', '130594.png', '130595.png', '1306.png', '130637.png', '1307.png', '130762.png', '130763.png', '130764.png', '130765.png', '130766.png', '130767.png', '130768.png', '130769.png', '130770.png', '130771.png', '130772.png', '130773.png', '130774.png', '130775.png', '130776.png', '130777.png', '130778.png', '130779.png', '130780.png', '130781.png', '130782.png', '130783.png', '130784.png', '130785.png', '130786.png', '130787.png', '130788.png', '130789.png', '130790.png', '130791.png', '130792.png', '130793.png', '130794.png', '130795.png', '130796.png', '130797.png', '130798.png', '130799.png', '1308.png', '130800.png', '130801.png', '130802.png', '130803.png', '130804.png', '130805.png', '130808.png', '130809.png', '130810.png', '130811.png', '130812.png', '130814.png', '130815.png', '130816.png', '130817.png', '130820.png', '130821.png', '130823.png', '130824.png', '130825.png', '130826.png', '130827.png', '130828.png', '130834.png', '130835.png', '130836.png', '130837.png', '130838.png', '130839.png', '130840.png', '130844.png', '130845.png', '130846.png', '130847.png', '130848.png', '130849.png', '130850.png', '130851.png', '130852.png', '130853.png', '130854.png', '130855.png', '130856.png', '130857.png', '130859.png', '130861.png', '130862.png', '130863.png', '130864.png', '130865.png', '130866.png', '130868.png', '130869.png', '130870.png', '130871.png', '130872.png', '130873.png', '130874.png', '130875.png', '130877.png', '130879.png', '130881.png', '130882.png', '1309.png', '130939.png', '130941.png', '130943.png', '1310.png', '1311.png', '131108.png', '131111.png', '131120.png', '131125.png', '131127.png', '131129.png', '131135.png', '131137.png', '131139.png', '131140.png', '131142.png', '131147.png', '131152.png', '131154.png', '131155.png', '131158.png', '131159.png', '131162.png', '131166.png', '131169.png', '131178.png', '131183.png', '131190.png', '131195.png', '131197.png', '131199.png', '1312.png', '131200.png', '131205.png', '131209.png', '131210.png', '131212.png', '131217.png', '131219.png', '131222.png', '131225.png', '131229.png', '131240.png', '131246.png', '131249.png', '131250.png', '131252.png', '131255.png', '131256.png', '131258.png', '131261.png', '131264.png', '131266.png', '131267.png', '131270.png', '131280.png', '131289.png', '1313.png', '1314.png', '1315.png', '1316.png', '1317.png', '1318.png', '1319.png', '1320.png', '1323.png', '1324.png', '1325.png', '1326.png', '1327.png', '1328.png', '1330.png', '1331.png', '133178.png', '133179.png', '1332.png', '133235.png', '1333.png', '1334.png', '1335.png', '1336.png', '133669.png', '133670.png', '133671.png', '133672.png', '133673.png', '133674.png', '133675.png', '133676.png', '133677.png', '133678.png', '133679.png', '133680.png', '133681.png', '1337.png', '133765.png', '1338.png', '1339.png', '1340.png', '1341.png', '1343.png', '1344.png', '134636.png', '134637.png', '1347.png', '134732.png', '1348.png', '134886.png', '134887.png', '134888.png', '134889.png', '134890.png', '134891.png', '134892.png', '134893.png', '134894.png', '134896.png', '134898.png', '1349.png', '134900.png', '134901.png', '134903.png', '134904.png', '134906.png', '134908.png', '134910.png', '134913.png', '134916.png', '134919.png', '134920.png', '134923.png', '1350.png', '1351.png', '135139.png', '135214.png', '135215.png', '135216.png', '135217.png', '135245.png', '135246.png', '135247.png', '135248.png', '135249.png', '135250.png', '135251.png', '135252.png', '135253.png', '135254.png', '135255.png', '135256.png', '135257.png', '135258.png', '135292.png', '135297.png', '1353.png', '135301.png', '135303.png', '135304.png', '135305.png', '135306.png', '135308.png', '135311.png', '135312.png', '135313.png', '135315.png', '135316.png', '135317.png', '135318.png', '135325.png', '135326.png', '135327.png', '135328.png', '135329.png', '135330.png', '135331.png', '135332.png', '135333.png', '135336.png', '135337.png', '135338.png', '135339.png', '135340.png', '135342.png', '135344.png', '135345.png', '135346.png', '135347.png', '135348.png', '135349.png', '135351.png', '135353.png', '135356.png', '135359.png', '135362.png', '135365.png', '135368.png', '135371.png', '135372.png', '135377.png', '135382.png', '135388.png', '135389.png', '135390.png', '135391.png', '135392.png', '135393.png', '135394.png', '135395.png', '135396.png', '135397.png', '135399.png', '135400.png', '135401.png', '135402.png', '135403.png', '135404.png', '135405.png', '135406.png', '135407.png', '135408.png', '135409.png', '135410.png', '135411.png', '135413.png', '135414.png', '135423.png', '135424.png', '135425.png', '135426.png', '135427.png', '135428.png', '135430.png', '135432.png', '135433.png', '135434.png', '135435.png', '135436.png', '135439.png', '135442.png', '135443.png', '135448.png', '135449.png', '135450.png', '135452.png', '135453.png', '135454.png', '135455.png', '135467.png', '135468.png', '135469.png', '135470.png', '135471.png', '135472.png', '135473.png', '135474.png', '135475.png', '135476.png', '135477.png', '135478.png', '135479.png', '135480.png', '135481.png', '135482.png', '135485.png', '135488.png', '135491.png', '135494.png', '135497.png', '1355.png', '135500.png', '135503.png', '135506.png', '135509.png', '135512.png', '135515.png', '135518.png', '135521.png', '135523.png', '135526.png', '135527.png', '135528.png', '135531.png', '135534.png', '135535.png', '135540.png', '1356.png', '1357.png', '1358.png', '135853.png', '1359.png', '135943.png', '135944.png', '135945.png', '135946.png', '1360.png', '136007.png', '136008.png', '136009.png', '136010.png', '136011.png', '136012.png', '136013.png', '136014.png', '136015.png', '136016.png', '136017.png', '136018.png', '136019.png', '136020.png', '136021.png', '136022.png', '136023.png', '136024.png', '1361.png', '136150.png', '136151.png', '136152.png', '136153.png', '136154.png', '136155.png', '136156.png', '136157.png', '136158.png', '136159.png', '136160.png', '136161.png', '136174.png', '136175.png', '136176.png', '136177.png', '136178.png', '136179.png', '136180.png', '136181.png', '136182.png', '136183.png', '136184.png', '136185.png', '136186.png', '136187.png', '136188.png', '136189.png', '136190.png', '136196.png', '136197.png', '136198.png', '136199.png', '136200.png', '136201.png', '136202.png', '136203.png', '136204.png', '136205.png', '136206.png', '136207.png', '136208.png', '136209.png', '136210.png', '136211.png', '136212.png', '136227.png', '136228.png', '136229.png', '136230.png', '136231.png', '136232.png', '136233.png', '136234.png', '136235.png', '136236.png', '136237.png', '136238.png', '136239.png', '136240.png', '136241.png', '136242.png', '136243.png', '136244.png', '136245.png', '136246.png', '136247.png', '136248.png', '136249.png', '136250.png', '136251.png', '136252.png', '136253.png', '136254.png', '136255.png', '136256.png', '136257.png', '136258.png', '136259.png', '136260.png', '136261.png', '136262.png', '136263.png', '136264.png', '136265.png', '136266.png', '136267.png', '136268.png', '136269.png', '136270.png', '136271.png', '136272.png', '136273.png', '136274.png', '136276.png', '136277.png', '136278.png', '136279.png', '136280.png', '136281.png', '136282.png', '136283.png', '136284.png', '136285.png', '136286.png', '136287.png', '136288.png', '136289.png', '136290.png', '136291.png', '136292.png', '136293.png', '136294.png', '136295.png', '136297.png', '136298.png', '136299.png', '1363.png', '136300.png', '136301.png', '136302.png', '136303.png', '136304.png', '136305.png', '136306.png', '1364.png', '136416.png', '136417.png', '136418.png', '136420.png', '136421.png', '136422.png', '136423.png', '136426.png', '136427.png', '136429.png', '136430.png', '136431.png', '136432.png', '136433.png', '136434.png', '136435.png', '136436.png', '136437.png', '136459.png', '136460.png', '136461.png', '136462.png', '136463.png', '136464.png', '136465.png', '136466.png', '136467.png', '136468.png', '136469.png', '136470.png', '136471.png', '136472.png', '136473.png', '136474.png', '136475.png', '136476.png', '136477.png', '1365.png', '136531.png', '136548.png', '136549.png', '136550.png', '136583.png', '136587.png', '136588.png', '1366.png', '136610.png', '136612.png', '136614.png', '136615.png', '136620.png', '136624.png', '136625.png', '136636.png', '136637.png', '136639.png', '136647.png', '1367.png', '1369.png', '1370.png', '1371.png']
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1.png
19 204 1 222
1 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1000.png
5 218 4 219
1000 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1001.png
10 213 4 219
1001 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1002.png
14 209 6 214
1002 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1003.png
30 194 4 216
1003 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100349.png
35 188 6 217
100349 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100351.png
6 218 22 199
100351 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100352.png
5 218 11 210
100352 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100353.png
5 218 20 201
100353 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100354.png
35 186 5 219
100354 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100355.png
15 208 4 218
100355 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1004.png
8 215 6 214
1004 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100473.png
26 197 14 215
100473 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100479.png
46 177 4 218
100479 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100488.png
5 218 14 209
100488 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1005.png
5 218 4 219
1005 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100604.png
5 218 4 219
100604 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100605.png
5 218 4 219
100605 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1007.png
5 218 4 219
1007 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100729.png
32 191 4 217
100729 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100795.png
26 197 4 217
100795 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100796.png
5 218 4 219
100796 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1008.png
6 218 4 215
1008 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100806.png
23 200 4 217
100806 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100818.png
5 218 9 213
100818 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100881.png
25 198 4 218
100881 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100882.png
19 205 5 217
100882 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100883.png
11 212 11 213
100883 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100884.png
55 168 5 216
100884 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100885.png
26 196 5 216
100885 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100893.png
24 198 8 216
100893 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100896.png
24 199 5 218
100896 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1009.png
27 196 4 219
1009 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100904.png
4 219 11 217
100904 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100909.png
25 197 4 216
100909 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100911.png
5 218 4 219
100911 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100912.png
16 207 5 216
100912 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100913.png
34 189 4 219
100913 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100914.png
34 189 7 216
100914 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100915.png
5 218 4 219
100915 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100916.png
7 217 4 217
100916 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100918.png
5 218 4 219
100918 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100921.png
5 218 9 212
100921 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100922.png
6 217 5 216
100922 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100931.png
11 212 4 219
100931 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100933.png
5 218 4 219
100933 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100936.png
11 212 5 217
100936 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100939.png
5 218 5 218
100939 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100940.png
5 218 7 215
100940 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100952.png
5 218 6 216
100952 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\100987.png
6 218 61 160
100987 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101.png
18 205 9 216
101 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1010.png
16 207 6 214
1010 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101000.png
39 185 4 219
101000 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101012.png
6 218 5 219
101012 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101013.png
18 205 4 216
101013 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101016.png
5 217 12 209
101016 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101017.png
6 217 5 216
101017 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101018.png
29 194 4 219
101018 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101020.png
5 218 4 219
101020 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101026.png
6 217 5 216
101026 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101045.png
5 218 37 184
101045 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101058.png
5 218 4 219
101058 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1011.png
14 208 5 216
1011 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101148.png
6 217 5 216
101148 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101149.png
37 185 4 219
101149 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101151.png
40 185 6 218
101151 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101154.png
6 217 5 216
101154 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1012.png
25 198 4 219
1012 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101241.png
5 218 4 218
101241 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101242.png
6 218 16 208
101242 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101243.png
27 196 5 218
101243 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101261.png
6 217 5 219
101261 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101266.png
33 190 4 219
101266 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101267.png
5 218 4 219
101267 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1013.png
5 218 4 219
1013 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101345.png
31 193 5 214
101345 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101395.png
19 203 5 218
101395 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1014.png
20 202 4 219
1014 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101448.png
5 218 4 219
101448 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101471.png
31 192 4 218
101471 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101475.png
12 211 4 217
101475 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101493.png
26 197 4 216
101493 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1015.png
21 202 4 217
1015 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101504.png
5 218 70 151
101504 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1016.png
5 218 4 219
1016 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101606.png
49 174 6 218
101606 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101687.png
21 202 4 219
101687 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1017.png
44 181 5 214
1017 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101773.png
6 216 7 216
101773 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101776.png
24 198 5 216
101776 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101780.png
5 217 4 219
101780 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101789.png
5 218 4 218
101789 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\101797.png
5 218 4 219
101797 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1019.png
5 218 32 190
1019 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102.png
31 193 5 219
102 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1020.png
5 217 5 217
1020 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102009.png
5 218 16 207
102009 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102010.png
37 185 6 218
102010 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102012.png
8 216 4 219
102012 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102013.png
5 218 5 220
102013 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102014.png
13 210 4 217
102014 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102019.png
40 183 6 217
102019 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102022.png
12 210 4 217
102022 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102025.png
21 200 8 217
102025 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102029.png
16 206 4 216
102029 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102036.png
9 215 13 210
102036 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102092.png
9 214 5 217
102092 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1021.png
24 200 5 219
1021 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102170.png
5 218 17 205
102170 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1022.png
37 186 4 216
1022 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1023.png
21 202 4 218
1023 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102355.png
6 217 5 215
102355 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102356.png
30 192 4 214
102356 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102358.png
42 181 5 216
102358 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102359.png
25 197 4 216
102359 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102360.png
8 215 6 214
102360 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102361.png
28 196 6 214
102361 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102362.png
22 200 6 214
102362 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102363.png
18 207 15 205
102363 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102364.png
16 208 5 215
102364 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102365.png
36 187 16 219
102365 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102366.png
22 200 5 214
102366 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102367.png
5 218 4 219
102367 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102368.png
20 204 4 216
102368 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102369.png
19 204 4 219
102369 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102370.png
33 191 3 216
102370 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102371.png
5 218 4 219
102371 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102372.png
6 216 4 216
102372 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102373.png
14 208 5 214
102373 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102374.png
17 206 4 218
102374 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102375.png
13 207 6 218
102375 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102390.png
14 209 3 216
102390 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1024.png
15 208 4 218
1024 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102462.png
23 201 4 214
102462 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102464.png
15 208 5 218
102464 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102466.png
30 193 6 218
102466 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102467.png
16 207 4 218
102467 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102470.png
31 192 4 216
102470 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102472.png
26 197 4 218
102472 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102474.png
20 204 4 219
102474 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102476.png
41 182 6 217
102476 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102477.png
10 212 11 215
102477 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102479.png
19 204 4 217
102479 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102482.png
38 186 4 214
102482 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102483.png
17 205 3 216
102483 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102485.png
5 218 4 219
102485 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102486.png
24 199 5 216
102486 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102487.png
18 205 5 217
102487 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102489.png
25 199 4 217
102489 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102490.png
6 217 4 217
102490 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102491.png
8 215 6 214
102491 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102493.png
32 191 5 217
102493 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102495.png
12 211 5 215
102495 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1025.png
5 218 5 217
1025 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102555.png
8 215 6 214
102555 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1026.png
27 196 4 218
1026 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1027.png
11 212 4 217
1027 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1028.png
5 218 25 199
1028 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102849.png
26 198 6 214
102849 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102850.png
35 190 5 214
102850 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102851.png
15 209 20 200
102851 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102852.png
31 191 6 214
102852 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102853.png
43 179 4 213
102853 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102854.png
5 218 31 192
102854 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102855.png
5 218 4 218
102855 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102856.png
30 192 3 216
102856 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102857.png
6 218 5 218
102857 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102858.png
29 196 5 218
102858 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102859.png
8 216 6 214
102859 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102860.png
20 202 4 214
102860 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102861.png
26 196 4 216
102861 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102862.png
33 190 4 214
102862 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102863.png
29 193 3 216
102863 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102864.png
5 218 4 219
102864 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102884.png
6 217 5 216
102884 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102885.png
5 218 4 218
102885 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102886.png
25 198 4 219
102886 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102887.png
5 218 47 175
102887 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102889.png
22 200 4 216
102889 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102890.png
47 176 4 217
102890 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1029.png
9 215 4 219
1029 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102924.png
5 218 4 219
102924 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102957.png
20 202 5 214
102957 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\102959.png
29 194 6 217
102959 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\103.png
33 190 4 218
103 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1030.png
30 193 4 218
1030 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1031.png
31 192 4 218
1031 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1032.png
32 191 4 217
1032 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\103278.png
16 206 5 216
103278 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\103279.png
7 217 49 175
103279 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\103282.png
1 222 45 180
103282 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\103283.png
8 215 6 214
103283 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\103284.png
6 217 5 216
103284 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\103285.png
41 182 5 216
103285 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\103286.png
27 196 4 219
103286 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\103287.png
6 217 5 216
103287 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1033.png
5 218 5 218
1033 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1034.png
5 218 4 219
1034 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1036.png
13 211 5 217
1036 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1037.png
34 189 4 217
1037 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1038.png
12 213 6 214
1038 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1039.png
8 215 6 214
1039 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104.png
5 218 4 218
104 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1041.png
6 217 5 216
1041 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1042.png
49 175 6 214
1042 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1043.png
24 199 4 219
1043 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104359.png
7 216 8 217
104359 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104360.png
41 183 6 214
104360 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104361.png
17 204 5 216
104361 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104362.png
14 209 5 218
104362 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104363.png
6 217 5 216
104363 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104386.png
12 211 10 214
104386 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104388.png
25 198 4 217
104388 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104395.png
6 217 5 216
104395 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104396.png
8 215 6 214
104396 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1044.png
14 209 4 217
1044 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1045.png
12 211 4 217
1045 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1046.png
50 172 6 214
1046 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104648.png
5 218 4 219
104648 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104650.png
5 218 11 212
104650 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104651.png
41 182 9 215
104651 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104652.png
4 219 50 174
104652 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104653.png
30 193 4 217
104653 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1047.png
51 173 5 214
1047 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104749.png
11 212 4 218
104749 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104750.png
31 191 4 214
104750 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104775.png
12 211 5 217
104775 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104776.png
27 196 6 219
104776 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104868.png
37 186 6 216
104868 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104870.png
29 195 5 214
104870 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104872.png
6 217 5 216
104872 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104873.png
6 217 5 216
104873 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104874.png
17 208 5 214
104874 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104884.png
6 217 5 216
104884 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104885.png
6 217 5 215
104885 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104887.png
41 183 4 214
104887 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1049.png
32 193 6 214
1049 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104975.png
5 218 7 216
104975 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104978.png
8 215 8 217
104978 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104979.png
37 188 4 214
104979 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104982.png
31 194 6 214
104982 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104988.png
31 192 5 216
104988 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104997.png
5 218 4 219
104997 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\104999.png
38 185 6 214
104999 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105.png
21 202 4 219
105 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1050.png
16 207 4 217
1050 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105000.png
6 216 6 217
105000 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105013.png
26 196 4 214
105013 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105026.png
5 218 4 219
105026 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105030.png
20 203 4 218
105030 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1051.png
22 201 4 219
1051 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105174.png
5 218 17 206
105174 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105175.png
25 198 4 219
105175 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105176.png
5 218 40 183
105176 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105177.png
5 218 40 183
105177 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105178.png
5 218 4 219
105178 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105180.png
19 204 6 214
105180 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105181.png
24 198 5 216
105181 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105187.png
35 189 4 218
105187 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105188.png
7 216 5 214
105188 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1052.png
23 200 5 216
1052 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1053.png
32 190 5 216
1053 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1054.png
8 215 4 219
1054 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1055.png
28 194 4 214
1055 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1056.png
29 193 4 216
1056 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105611.png
21 202 4 218
105611 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105612.png
26 196 4 217
105612 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105613.png
25 198 4 216
105613 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105707.png
33 190 4 217
105707 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1058.png
24 199 5 219
1058 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105898.png
8 215 6 214
105898 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1059.png
24 199 4 214
1059 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105989.png
8 215 6 214
105989 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105990.png
12 211 6 214
105990 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105991.png
4 219 37 183
105991 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105992.png
12 209 4 216
105992 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105993.png
26 197 4 219
105993 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105994.png
29 194 4 216
105994 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105995.png
5 218 4 219
105995 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105996.png
5 218 4 219
105996 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105997.png
52 172 5 213
105997 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105998.png
28 196 3 216
105998 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\105999.png
27 196 6 213
105999 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1060.png
12 211 3 216
1060 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106000.png
7 216 5 214
106000 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106001.png
7 216 9 211
106001 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106027.png
18 204 4 214
106027 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106028.png
22 201 5 214
106028 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106029.png
33 192 6 214
106029 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106052.png
6 217 19 202
106052 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106053.png
29 194 4 219
106053 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106054.png
51 172 3 216
106054 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106063.png
21 202 5 216
106063 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106071.png
33 192 5 214
106071 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106072.png
21 202 4 218
106072 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1061.png
31 193 6 214
1061 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106190.png
24 198 5 214
106190 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1062.png
47 176 7 219
1062 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1063.png
6 217 4 217
1063 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106363.png
41 182 5 216
106363 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106371.png
6 217 5 216
106371 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1064.png
23 199 4 214
1064 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106447.png
6 217 9 210
106447 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1065.png
6 217 13 207
1065 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106578.png
9 214 4 218
106578 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106580.png
5 218 4 219
106580 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106583.png
23 200 4 218
106583 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1066.png
6 217 5 216
1066 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106691.png
24 199 4 219
106691 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106694.png
5 218 13 210
106694 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106696.png
8 216 6 214
106696 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106730.png
21 201 8 215
106730 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106745.png
10 214 4 217
106745 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106746.png
6 217 16 207
106746 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106747.png
5 218 4 218
106747 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106748.png
17 206 4 218
106748 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106749.png
6 217 4 219
106749 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106755.png
34 189 6 216
106755 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106756.png
39 183 4 213
106756 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106757.png
6 217 5 215
106757 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106780.png
20 202 6 214
106780 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106781.png
15 207 5 214
106781 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1068.png
46 176 6 214
1068 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106805.png
8 215 6 214
106805 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106808.png
24 198 5 214
106808 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106809.png
23 201 5 214
106809 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106811.png
6 217 5 216
106811 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106812.png
28 194 6 214
106812 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106813.png
31 192 5 214
106813 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106814.png
7 216 8 217
106814 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106815.png
5 218 10 213
106815 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106816.png
5 218 30 191
106816 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106817.png
33 190 5 218
106817 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106818.png
14 208 5 214
106818 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106844.png
30 194 4 218
106844 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106849.png
20 202 6 217
106849 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106857.png
0 223 0 223
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106858.png
5 218 4 219
106858 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\106881.png
40 184 3 216
106881 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1069.png
29 194 6 214
1069 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1070.png
14 209 4 219
1070 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1071.png
21 202 4 216
1071 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1072.png
8 215 6 214
1072 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107201.png
50 171 9 218
107201 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107203.png
32 191 3 216
107203 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107205.png
38 185 6 217
107205 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107206.png
32 192 5 217
107206 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107208.png
27 196 6 218
107208 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107210.png
43 179 5 217
107210 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107212.png
14 209 5 217
107212 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107214.png
24 200 6 214
107214 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107216.png
30 193 3 216
107216 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107218.png
6 217 5 216
107218 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107220.png
12 211 5 218
107220 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107222.png
12 211 5 217
107222 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107224.png
17 206 4 215
107224 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107226.png
21 202 5 217
107226 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107228.png
5 218 21 201
107228 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107230.png
33 190 4 213
107230 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107232.png
7 216 13 206
107232 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107234.png
25 199 4 214
107234 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107236.png
23 200 6 214
107236 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107238.png
38 185 5 218
107238 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107240.png
41 183 4 214
107240 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107242.png
34 189 3 216
107242 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107280.png
37 187 6 217
107280 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107283.png
5 218 13 209
107283 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107285.png
23 201 8 218
107285 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107287.png
6 218 91 132
107287 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107289.png
5 218 4 219
107289 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107291.png
5 218 4 219
107291 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107293.png
7 216 76 143
107293 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107296.png
15 208 5 219
107296 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107298.png
30 194 8 210
107298 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1073.png
12 211 4 219
1073 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107301.png
28 194 5 214
107301 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107303.png
21 202 4 219
107303 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107305.png
29 194 4 217
107305 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107307.png
20 205 7 211
107307 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107309.png
6 217 43 179
107309 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107311.png
55 168 5 216
107311 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107313.png
22 201 5 218
107313 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107314.png
17 207 5 217
107314 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107319.png
2 221 5 220
107319 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107320.png
5 218 5 220
107320 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107321.png
5 218 5 220
107321 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107322.png
14 209 5 218
107322 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107323.png
5 218 22 200
107323 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107324.png
5 217 35 187
107324 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107325.png
6 218 4 219
107325 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107326.png
14 210 6 214
107326 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107327.png
30 193 4 218
107327 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107328.png
12 211 4 217
107328 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107329.png
9 213 5 216
107329 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107330.png
6 218 6 216
107330 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107332.png
6 217 5 216
107332 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107333.png
19 204 4 217
107333 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107334.png
24 199 4 216
107334 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107335.png
5 218 4 219
107335 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107336.png
16 206 4 214
107336 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107337.png
5 218 34 189
107337 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107338.png
6 217 5 216
107338 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107339.png
23 200 4 218
107339 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107340.png
5 218 7 216
107340 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107341.png
5 218 4 218
107341 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107342.png
12 211 5 217
107342 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107343.png
6 218 57 166
107343 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107344.png
26 198 4 218
107344 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107345.png
5 218 21 202
107345 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107346.png
12 211 5 218
107346 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107347.png
20 203 5 217
107347 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107348.png
34 188 4 217
107348 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107349.png
5 218 60 161
107349 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107350.png
5 217 26 196
107350 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107351.png
16 207 4 218
107351 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107352.png
5 218 4 219
107352 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107353.png
5 218 86 136
107353 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107354.png
5 218 33 190
107354 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107355.png
7 216 4 214
107355 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107356.png
20 203 4 217
107356 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107357.png
5 217 60 164
107357 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107358.png
28 195 5 219
107358 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107359.png
5 218 81 142
107359 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107360.png
5 218 5 218
107360 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107361.png
5 218 5 217
107361 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107362.png
6 216 10 210
107362 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107363.png
5 218 90 133
107363 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107364.png
8 215 4 218
107364 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107365.png
27 196 5 217
107365 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107366.png
37 186 6 216
107366 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107367.png
6 217 5 216
107367 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107368.png
5 218 4 218
107368 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107369.png
31 192 4 218
107369 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107370.png
3 220 98 122
107370 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1074.png
5 218 13 209
1074 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1076.png
5 218 5 220
1076 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107673.png
10 213 4 219
107673 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107674.png
6 216 38 185
107674 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107675.png
6 218 23 200
107675 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107676.png
34 189 5 214
107676 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107677.png
25 193 16 210
107677 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107678.png
47 175 5 215
107678 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107679.png
24 198 3 213
107679 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107680.png
6 218 11 212
107680 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107681.png
34 189 4 214
107681 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107682.png
11 213 9 215
107682 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107683.png
5 218 74 149
107683 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\107684.png
7 216 6 216
107684 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1077.png
22 202 5 218
1077 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1078.png
36 186 6 214
1078 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1079.png
17 206 5 216
1079 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1080.png
21 203 5 217
1080 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1081.png
12 211 5 216
1081 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1082.png
7 217 19 200
1082 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1083.png
37 185 4 214
1083 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1084.png
21 202 4 218
1084 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1085.png
25 198 3 216
1085 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108510.png
10 214 4 219
108510 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108514.png
16 207 5 217
108514 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108517.png
39 188 5 218
108517 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108522.png
8 215 10 211
108522 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108524.png
29 195 4 218
108524 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108526.png
22 202 4 214
108526 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108531.png
21 202 4 218
108531 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108538.png
15 207 4 217
108538 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108542.png
28 195 4 217
108542 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108545.png
20 203 4 217
108545 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108546.png
7 216 4 217
108546 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108547.png
37 186 4 216
108547 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108567.png
15 208 5 216
108567 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108569.png
20 203 5 218
108569 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108571.png
18 205 5 218
108571 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108574.png
19 205 5 214
108574 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108576.png
6 217 5 216
108576 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108578.png
21 202 4 217
108578 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1086.png
6 217 29 194
1086 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1087.png
5 218 6 214
1087 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1088.png
5 217 4 216
1088 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108891.png
5 218 49 173
108891 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108893.png
8 215 6 214
108893 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108896.png
5 218 60 162
108896 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108898.png
9 218 4 218
108898 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1089.png
6 218 5 216
1089 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108905.png
5 218 4 219
108905 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108907.png
15 208 4 217
108907 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108909.png
19 204 4 217
108909 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108911.png
28 195 4 217
108911 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108913.png
6 218 7 215
108913 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108917.png
6 218 31 191
108917 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108919.png
9 214 4 217
108919 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108921.png
7 218 44 180
108921 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108923.png
5 218 4 219
108923 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108925.png
5 218 4 219
108925 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108926.png
5 218 41 180
108926 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108928.png
27 199 6 218
108928 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108930.png
5 218 10 213
108930 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108932.png
5 218 34 188
108932 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108934.png
5 218 36 185
108934 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108936.png
18 205 4 218
108936 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108938.png
27 196 4 219
108938 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108940.png
6 218 39 183
108940 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108942.png
5 217 6 220
108942 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108944.png
5 218 26 195
108944 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108950.png
6 219 45 178
108950 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108952.png
6 219 43 182
108952 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108954.png
27 196 4 218
108954 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108955.png
13 210 4 219
108955 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108979.png
5 218 9 213
108979 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108981.png
8 215 5 213
108981 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108983.png
43 180 4 218
108983 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108985.png
32 191 5 214
108985 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108986.png
30 195 6 214
108986 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108988.png
49 173 7 218
108988 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108990.png
22 199 5 216
108990 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108993.png
5 218 5 218
108993 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108995.png
21 203 5 216
108995 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\108997.png
5 218 4 219
108997 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1090.png
19 204 3 216
1090 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109001.png
25 199 5 216
109001 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109003.png
20 203 5 217
109003 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109005.png
45 179 5 214
109005 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109007.png
33 190 4 216
109007 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109009.png
35 188 4 216
109009 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109011.png
40 184 4 217
109011 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109013.png
22 201 4 216
109013 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109015.png
38 185 4 216
109015 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109017.png
24 199 5 214
109017 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109019.png
32 191 4 217
109019 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109021.png
28 195 4 216
109021 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109023.png
21 202 4 217
109023 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109025.png
5 218 27 195
109025 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109027.png
41 182 5 216
109027 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109029.png
7 215 7 217
109029 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109031.png
38 184 5 216
109031 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109034.png
5 218 74 145
109034 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109035.png
7 216 4 217
109035 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109037.png
39 185 5 214
109037 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109038.png
47 175 8 216
109038 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109040.png
25 198 4 219
109040 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109041.png
36 187 4 216
109041 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109042.png
22 201 5 214
109042 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109043.png
53 170 4 217
109043 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109044.png
42 181 4 218
109044 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1091.png
6 217 7 214
1091 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109103.png
22 201 5 218
109103 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109105.png
5 218 54 168
109105 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109106.png
5 218 4 219
109106 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109108.png
5 218 4 219
109108 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109113.png
19 204 4 218
109113 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109114.png
5 218 19 204
109114 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109116.png
5 218 4 219
109116 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109118.png
5 218 13 206
109118 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109120.png
5 218 4 219
109120 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109121.png
15 208 4 217
109121 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109122.png
6 217 5 215
109122 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109123.png
31 193 4 219
109123 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109132.png
36 187 4 217
109132 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109133.png
33 190 4 217
109133 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109177.png
20 203 7 217
109177 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109178.png
45 178 10 213
109178 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109179.png
6 217 5 216
109179 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109180.png
12 212 5 214
109180 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109188.png
40 183 4 216
109188 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109189.png
35 187 3 216
109189 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109190.png
25 198 4 218
109190 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109192.png
26 198 4 214
109192 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109206.png
5 218 4 219
109206 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109210.png
5 218 4 219
109210 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109211.png
38 185 4 218
109211 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109212.png
5 218 5 218
109212 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109282.png
47 177 4 219
109282 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109283.png
18 205 3 216
109283 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109284.png
26 198 5 214
109284 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109285.png
7 216 7 218
109285 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109286.png
5 218 4 219
109286 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1093.png
31 194 4 214
1093 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109301.png
49 173 4 218
109301 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109303.png
35 188 4 219
109303 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\109305.png
32 191 5 217
109305 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1094.png
16 207 4 216
1094 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1095.png
30 192 4 214
1095 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1096.png
7 216 9 210
1096 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1097.png
21 204 4 214
1097 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1098.png
35 186 4 217
1098 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1099.png
43 180 4 219
1099 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1100.png
30 193 4 216
1100 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1101.png
6 218 7 217
1101 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1102.png
5 218 4 219
1102 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1103.png
32 190 6 214
1103 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1104.png
24 199 5 218
1104 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1105.png
46 177 4 220
1105 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1106.png
48 174 6 214
1106 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1107.png
18 205 6 214
1107 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1108.png
9 214 8 215
1108 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1109.png
6 217 17 203
1109 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1110.png
47 176 4 218
1110 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1111.png
42 182 4 219
1111 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1112.png
29 193 5 216
1112 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1113.png
26 196 5 216
1113 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1114.png
23 200 4 217
1114 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1115.png
22 200 6 214
1115 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1116.png
14 210 5 214
1116 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1117.png
39 184 6 214
1117 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1118.png
26 196 6 209
1118 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1119.png
29 194 4 216
1119 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1120.png
33 189 4 216
1120 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\112031.png
20 202 5 216
112031 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1121.png
41 182 5 216
1121 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1122.png
5 218 7 215
1122 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1123.png
14 208 12 214
1123 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1124.png
31 189 5 219
1124 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1125.png
8 216 9 210
1125 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1126.png
37 187 6 213
1126 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1127.png
13 210 4 217
1127 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1128.png
5 218 4 219
1128 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1129.png
7 216 5 214
1129 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1130.png
54 170 4 214
1130 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1131.png
8 215 6 214
1131 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1132.png
33 189 6 214
1132 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1133.png
29 194 6 214
1133 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1134.png
19 204 5 214
1134 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1135.png
8 215 6 214
1135 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\113522.png
20 204 4 218
113522 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1136.png
30 193 6 212
1136 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1138.png
34 189 5 214
1138 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1139.png
47 176 7 212
1139 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1140.png
5 219 37 182
1140 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1141.png
28 196 5 214
1141 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1142.png
12 210 4 219
1142 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1143.png
24 198 5 216
1143 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1144.png
37 186 4 219
1144 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1145.png
30 193 5 214
1145 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\114576.png
41 182 4 219
114576 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\114579.png
7 216 7 218
114579 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\114582.png
5 218 4 219
114582 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1146.png
37 184 6 219
1146 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1147.png
44 178 4 213
1147 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1148.png
13 210 4 219
1148 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\114803.png
8 216 8 211
114803 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\114804.png
38 185 6 214
114804 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\114809.png
6 218 5 218
114809 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\114819.png
6 218 5 219
114819 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\114821.png
5 218 35 185
114821 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1149.png
39 184 9 215
1149 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\114946.png
21 202 4 217
114946 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\114947.png
22 201 4 216
114947 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\114948.png
29 195 4 216
114948 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\114951.png
0 223 0 223
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\114952.png
6 216 4 216
114952 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1150.png
8 215 6 214
1150 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\115000.png
6 217 5 219
115000 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\115001.png
5 218 22 200
115001 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\115039.png
5 218 7 215
115039 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1151.png
28 194 6 214
1151 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1152.png
47 176 8 217
1152 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\115257.png
8 215 4 216
115257 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1153.png
28 194 5 214
1153 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1154.png
36 187 5 218
1154 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1156.png
25 198 5 216
1156 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1157.png
15 209 4 217
1157 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\115711.png
30 193 5 216
115711 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\115755.png
28 195 4 219
115755 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1158.png
34 189 6 214
1158 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1159.png
6 217 5 217
1159 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\115940.png
40 182 5 214
115940 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\115942.png
28 196 3 216
115942 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\115943.png
34 190 4 218
115943 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\115944.png
23 199 4 216
115944 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\115945.png
27 196 5 219
115945 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\115946.png
19 205 4 218
115946 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\115988.png
21 202 4 218
115988 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1160.png
8 215 6 214
1160 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1161.png
29 193 6 214
1161 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\116138.png
11 211 5 216
116138 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\116140.png
39 184 3 216
116140 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\116144.png
28 195 6 219
116144 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\116145.png
24 198 5 216
116145 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\116150.png
15 209 4 217
116150 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\116152.png
7 216 4 217
116152 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\116156.png
5 218 4 219
116156 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\116164.png
14 209 5 217
116164 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\116192.png
14 209 4 217
116192 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\116194.png
16 206 4 217
116194 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1162.png
38 187 6 214
1162 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\116204.png
46 177 5 214
116204 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\116288.png
17 206 4 216
116288 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1163.png
41 182 4 217
1163 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\116303.png
5 218 4 219
116303 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\116309.png
32 190 5 214
116309 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\116315.png
21 202 7 219
116315 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\116331.png
36 187 4 219
116331 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\116334.png
5 218 5 218
116334 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\116347.png
6 217 8 213
116347 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\116384.png
5 218 47 172
116384 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1164.png
32 191 4 218
1164 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\116403.png
34 189 7 211
116403 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1165.png
12 211 8 217
1165 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1166.png
5 218 6 216
1166 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1167.png
29 195 5 216
1167 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1168.png
44 180 4 216
1168 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1169.png
18 205 6 214
1169 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1170.png
39 183 6 214
1170 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\117033.png
32 191 5 218
117033 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1171.png
28 196 11 212
1171 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\117183.png
10 213 4 219
117183 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\117199.png
31 193 4 218
117199 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1173.png
6 217 5 216
1173 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\117327.png
6 217 12 208
117327 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1174.png
24 199 5 216
1174 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1175.png
22 201 5 214
1175 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1176.png
68 156 9 214
1176 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\117669.png
5 218 4 219
117669 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1177.png
25 196 5 220
1177 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\117754.png
24 199 4 219
117754 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1178.png
6 218 7 213
1178 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1179.png
6 217 27 194
1179 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1181.png
13 209 4 220
1181 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1183.png
30 193 6 216
1183 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1184.png
31 193 5 217
1184 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1185.png
10 213 4 218
1185 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1186.png
31 193 6 214
1186 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1187.png
11 214 6 215
1187 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1188.png
20 203 4 219
1188 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1189.png
9 214 6 217
1189 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1190.png
5 218 20 202
1190 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1191.png
11 212 4 219
1191 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1192.png
36 187 4 217
1192 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1193.png
18 206 5 218
1193 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1194.png
31 195 7 218
1194 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1195.png
31 192 6 217
1195 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1196.png
30 193 6 218
1196 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1197.png
8 215 4 215
1197 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\119741.png
6 217 6 218
119741 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1198.png
6 217 36 187
1198 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1199.png
12 211 5 218
1199 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1200.png
7 216 37 186
1200 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120063.png
14 209 5 217
120063 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120064.png
23 200 5 214
120064 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120065.png
23 199 3 216
120065 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120066.png
5 218 41 177
120066 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120271.png
1 222 28 195
120271 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120286.png
29 193 5 216
120286 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120288.png
5 218 13 209
120288 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120386.png
28 195 4 217
120386 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1204.png
20 203 5 219
1204 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120533.png
6 217 5 216
120533 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120546.png
20 202 5 216
120546 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120559.png
5 218 49 173
120559 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1206.png
8 215 6 214
1206 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1207.png
5 218 44 178
1207 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120779.png
22 201 4 217
120779 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120782.png
17 206 5 216
120782 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120783.png
42 181 4 216
120783 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1208.png
14 210 4 216
1208 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120921.png
50 175 6 214
120921 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120924.png
57 166 4 216
120924 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120936.png
23 200 6 214
120936 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120939.png
6 217 5 216
120939 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120942.png
8 215 6 214
120942 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120946.png
20 203 8 216
120946 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120955.png
26 197 4 217
120955 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120966.png
5 218 4 218
120966 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120974.png
26 198 6 217
120974 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120979.png
39 183 3 216
120979 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120991.png
43 179 4 214
120991 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120994.png
6 218 21 200
120994 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\120998.png
42 182 5 214
120998 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1210.png
6 217 12 208
1210 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121002.png
5 218 47 175
121002 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1211.png
5 218 4 219
1211 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121106.png
29 193 4 216
121106 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121135.png
5 218 32 186
121135 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121170.png
5 218 4 219
121170 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121171.png
4 220 40 197
121171 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121172.png
17 206 4 218
121172 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121173.png
36 188 4 219
121173 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121174.png
43 180 4 217
121174 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121175.png
13 209 4 219
121175 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121179.png
8 215 6 214
121179 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121181.png
5 218 4 219
121181 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121182.png
5 218 71 150
121182 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121183.png
5 218 6 215
121183 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121184.png
5 218 4 219
121184 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121185.png
23 200 4 219
121185 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121186.png
6 217 6 219
121186 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121187.png
5 218 4 219
121187 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121188.png
18 205 4 218
121188 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121189.png
12 211 4 216
121189 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121190.png
5 218 11 211
121190 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121191.png
5 218 4 219
121191 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121192.png
5 218 4 218
121192 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121196.png
21 203 6 214
121196 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121197.png
27 196 4 219
121197 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121198.png
5 218 38 185
121198 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1212.png
3 218 8 214
1212 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121200.png
8 215 6 214
121200 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121201.png
5 218 4 219
121201 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121202.png
5 218 4 219
121202 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121203.png
11 212 4 217
121203 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121204.png
5 218 4 219
121204 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121205.png
5 218 4 217
121205 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121206.png
16 207 4 217
121206 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121207.png
11 211 5 216
121207 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121208.png
5 218 8 215
121208 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121209.png
5 218 5 217
121209 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121212.png
8 215 9 216
121212 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121214.png
16 206 5 216
121214 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121215.png
5 218 4 219
121215 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121216.png
7 216 7 218
121216 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121218.png
28 195 4 217
121218 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121219.png
5 218 4 218
121219 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121220.png
5 218 4 219
121220 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121262.png
51 174 4 214
121262 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121265.png
46 179 4 214
121265 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\121267.png
19 204 4 218
121267 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1213.png
41 182 6 214
1213 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1215.png
8 215 11 213
1215 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1216.png
5 218 64 159
1216 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1218.png
29 196 5 214
1218 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1219.png
29 196 5 214
1219 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1220.png
14 210 4 217
1220 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1221.png
44 179 4 219
1221 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1222.png
5 218 4 219
1222 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1223.png
5 218 25 195
1223 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1224.png
48 175 4 214
1224 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1225.png
6 217 5 216
1225 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1226.png
26 197 4 218
1226 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1227.png
6 218 3 215
1227 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1228.png
14 209 4 219
1228 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1229.png
6 218 11 210
1229 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\122985.png
24 199 4 217
122985 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\122986.png
16 207 6 217
122986 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\122988.png
5 218 13 208
122988 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\122990.png
21 203 5 217
122990 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\122992.png
6 218 50 173
122992 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\122993.png
5 214 69 154
122993 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\122994.png
6 217 5 216
122994 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\122995.png
5 218 16 206
122995 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\122997.png
24 199 5 217
122997 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\122999.png
5 218 7 213
122999 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1230.png
31 195 6 214
1230 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\123001.png
31 187 4 212
123001 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\123004.png
26 197 6 218
123004 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\123005.png
6 217 5 216
123005 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\123007.png
9 214 4 219
123007 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\123009.png
5 218 37 182
123009 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1231.png
28 194 6 214
1231 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1232.png
5 218 4 219
1232 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1233.png
44 179 5 217
1233 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1235.png
18 205 5 217
1235 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1236.png
33 191 5 216
1236 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1237.png
5 218 4 219
1237 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1238.png
5 218 7 215
1238 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1239.png
46 178 5 214
1239 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1240.png
37 186 4 219
1240 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1241.png
51 172 5 217
1241 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1242.png
33 191 5 216
1242 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1243.png
26 197 4 214
1243 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1244.png
23 200 4 216
1244 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1245.png
47 176 8 218
1245 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1246.png
35 188 4 219
1246 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1247.png
22 201 4 214
1247 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1248.png
6 217 5 216
1248 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1249.png
22 200 5 217
1249 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1250.png
32 192 4 213
1250 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1251.png
8 215 6 214
1251 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1252.png
10 218 7 211
1252 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1253.png
8 215 6 214
1253 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1254.png
8 215 6 214
1254 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1255.png
32 191 8 218
1255 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1257.png
41 182 4 214
1257 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1258.png
25 198 5 217
1258 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1259.png
45 179 4 213
1259 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1260.png
6 217 19 202
1260 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1261.png
13 210 4 216
1261 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1262.png
37 186 4 217
1262 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\126302.png
34 189 4 217
126302 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1264.png
6 217 5 216
1264 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\126415.png
28 195 4 219
126415 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\126416.png
6 217 6 220
126416 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\126417.png
19 203 4 216
126417 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\126419.png
17 206 4 217
126419 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\126421.png
5 218 4 219
126421 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\126422.png
27 197 4 215
126422 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\126423.png
11 214 5 214
126423 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\126424.png
5 218 4 218
126424 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\126426.png
25 197 5 216
126426 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\126427.png
26 197 4 218
126427 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1265.png
36 187 4 216
1265 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1266.png
42 182 4 218
1266 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1267.png
5 218 4 219
1267 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1268.png
39 185 5 214
1268 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1272.png
10 213 5 217
1272 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1273.png
8 215 6 214
1273 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\127397.png
5 218 4 219
127397 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1274.png
12 211 5 216
1274 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1275.png
5 218 4 219
1275 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1276.png
5 218 4 219
1276 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1277.png
5 218 4 219
1277 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\127742.png
6 218 5 216
127742 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\127787.png
28 194 3 216
127787 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1278.png
33 190 4 217
1278 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1279.png
5 218 20 203
1279 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\127969.png
7 218 8 219
127969 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\127999.png
4 219 6 214
127999 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1280.png
8 215 6 214
1280 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\128000.png
26 197 6 214
128000 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\128051.png
5 218 4 219
128051 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1281.png
6 217 29 194
1281 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\128172.png
13 210 4 219
128172 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1282.png
39 184 4 219
1282 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\128271.png
8 215 6 214
128271 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\128273.png
27 195 6 214
128273 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\128290.png
5 218 27 193
128290 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1283.png
8 215 8 218
1283 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1284.png
10 213 5 214
1284 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\128495.png
51 173 6 211
128495 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1285.png
23 202 8 213
1285 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\128553.png
44 178 4 219
128553 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\128554.png
10 213 4 216
128554 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\128650.png
5 218 4 218
128650 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\128652.png
24 198 5 216
128652 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\128654.png
11 212 4 219
128654 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\128656.png
16 207 4 216
128656 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1287.png
5 218 4 219
1287 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1288.png
6 217 5 216
1288 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1289.png
16 207 5 218
1289 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1290.png
10 213 4 219
1290 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1291.png
28 195 4 218
1291 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129129.png
21 203 5 214
129129 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129132.png
9 213 5 216
129132 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129135.png
21 202 5 219
129135 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129138.png
26 197 7 216
129138 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129140.png
22 201 5 216
129140 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129143.png
39 184 4 217
129143 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129146.png
42 181 4 217
129146 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129149.png
5 218 38 181
129149 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129152.png
5 218 4 219
129152 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129153.png
5 218 4 219
129153 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129156.png
6 217 52 171
129156 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129159.png
32 191 4 216
129159 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129162.png
15 208 5 217
129162 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129165.png
28 196 4 219
129165 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129166.png
5 218 26 193
129166 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129167.png
7 215 3 218
129167 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129168.png
20 203 4 217
129168 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129169.png
6 219 6 220
129169 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129170.png
26 197 7 219
129170 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129171.png
5 218 7 214
129171 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129172.png
5 218 4 219
129172 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129173.png
5 218 4 218
129173 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129174.png
10 213 8 214
129174 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129175.png
16 206 4 216
129175 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129176.png
19 203 4 219
129176 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129177.png
6 218 14 207
129177 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129178.png
6 217 5 216
129178 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129179.png
11 211 5 214
129179 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129180.png
5 218 4 219
129180 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129181.png
6 218 13 205
129181 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129182.png
5 218 4 219
129182 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129183.png
6 217 5 216
129183 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129184.png
5 219 5 214
129184 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129185.png
17 205 4 217
129185 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129186.png
5 218 4 219
129186 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129187.png
5 219 32 188
129187 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129188.png
19 204 4 217
129188 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129189.png
26 197 5 218
129189 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129190.png
27 198 4 217
129190 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129191.png
7 216 9 210
129191 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129192.png
17 206 4 218
129192 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129193.png
6 217 5 216
129193 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129194.png
16 207 7 218
129194 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129195.png
25 197 5 218
129195 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129196.png
5 218 73 146
129196 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129197.png
5 218 6 220
129197 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129198.png
12 212 5 216
129198 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129199.png
6 217 6 217
129199 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1292.png
5 218 25 195
1292 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129200.png
6 218 9 210
129200 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129201.png
22 200 5 214
129201 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129202.png
5 218 15 207
129202 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129203.png
8 215 6 214
129203 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129204.png
17 206 4 217
129204 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129205.png
22 201 5 216
129205 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129206.png
6 217 11 213
129206 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129208.png
9 215 6 214
129208 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129209.png
8 215 6 214
129209 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129210.png
6 217 15 204
129210 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129211.png
34 189 4 217
129211 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129212.png
5 218 5 217
129212 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1293.png
11 213 6 214
1293 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129380.png
6 217 5 216
129380 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129384.png
26 198 4 216
129384 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129389.png
21 203 6 218
129389 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129390.png
31 193 4 216
129390 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129391.png
14 209 4 219
129391 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129392.png
7 216 6 219
129392 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129393.png
6 217 5 216
129393 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129395.png
12 210 4 216
129395 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129396.png
10 212 5 216
129396 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129397.png
24 200 5 216
129397 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129398.png
6 217 5 216
129398 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129399.png
26 197 5 216
129399 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1294.png
16 207 4 216
1294 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129400.png
5 218 10 215
129400 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129401.png
26 197 4 217
129401 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129402.png
12 211 4 217
129402 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129403.png
5 218 5 218
129403 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129404.png
5 218 13 206
129404 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129405.png
39 184 4 218
129405 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129406.png
7 217 26 197
129406 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129407.png
5 218 4 219
129407 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129408.png
23 200 5 218
129408 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129409.png
15 208 4 219
129409 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129410.png
20 202 5 216
129410 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129411.png
7 216 8 218
129411 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129412.png
10 213 5 216
129412 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129413.png
17 207 4 218
129413 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129414.png
7 216 7 220
129414 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129415.png
5 218 24 199
129415 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129416.png
6 217 14 209
129416 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129417.png
5 218 4 219
129417 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129418.png
35 187 3 216
129418 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129419.png
5 218 4 219
129419 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129420.png
8 215 7 216
129420 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129421.png
19 205 4 218
129421 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129423.png
10 214 4 218
129423 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129424.png
5 218 30 193
129424 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129425.png
6 217 16 203
129425 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129426.png
6 217 10 213
129426 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129427.png
5 218 7 215
129427 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129428.png
15 208 4 219
129428 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129429.png
31 192 6 214
129429 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129430.png
41 181 1 220
129430 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129431.png
16 207 4 219
129431 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129432.png
25 198 4 216
129432 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129433.png
9 214 4 219
129433 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129434.png
22 201 4 216
129434 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129435.png
35 188 7 217
129435 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1295.png
5 218 13 208
1295 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129563.png
24 199 4 219
129563 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129564.png
16 206 4 218
129564 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129565.png
8 215 4 219
129565 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129567.png
13 210 8 215
129567 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129568.png
5 218 29 193
129568 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129569.png
15 209 4 217
129569 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129570.png
8 216 8 218
129570 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129571.png
26 199 6 214
129571 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129572.png
24 198 4 216
129572 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129573.png
49 174 4 218
129573 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129575.png
26 197 5 216
129575 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129577.png
14 210 4 218
129577 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129580.png
25 198 4 219
129580 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129583.png
6 218 6 213
129583 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129584.png
5 218 4 219
129584 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129585.png
15 208 5 216
129585 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1296.png
26 197 5 217
1296 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129653.png
7 216 8 218
129653 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129656.png
5 218 4 219
129656 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129657.png
12 211 5 214
129657 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129658.png
8 214 6 214
129658 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129659.png
22 199 4 216
129659 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129660.png
23 200 5 214
129660 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129661.png
28 194 4 216
129661 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129662.png
11 212 5 216
129662 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129663.png
8 215 6 214
129663 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129664.png
7 216 8 218
129664 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129665.png
28 196 4 217
129665 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129666.png
24 199 6 214
129666 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129667.png
25 199 5 214
129667 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129668.png
5 218 17 206
129668 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129682.png
8 215 6 214
129682 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129683.png
5 218 5 217
129683 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129685.png
5 218 4 218
129685 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129686.png
16 207 4 218
129686 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129688.png
28 195 5 216
129688 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129689.png
6 217 5 216
129689 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129690.png
5 218 4 219
129690 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129691.png
27 196 4 217
129691 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129692.png
5 218 4 218
129692 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129693.png
24 199 4 217
129693 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129694.png
12 211 4 216
129694 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129695.png
5 218 7 214
129695 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129696.png
5 218 4 219
129696 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129697.png
15 208 8 215
129697 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129698.png
5 218 4 219
129698 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129699.png
38 187 6 214
129699 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1297.png
6 217 4 216
1297 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129700.png
24 200 4 216
129700 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129701.png
5 218 4 219
129701 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129702.png
5 218 4 219
129702 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129703.png
30 193 4 218
129703 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129704.png
8 215 6 214
129704 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129705.png
5 218 4 219
129705 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129706.png
5 218 4 219
129706 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129707.png
20 203 5 219
129707 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129708.png
5 218 5 219
129708 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129709.png
5 218 4 219
129709 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129735.png
31 192 4 216
129735 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129738.png
17 206 4 217
129738 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129739.png
8 214 6 214
129739 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129740.png
25 198 5 216
129740 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129741.png
33 191 4 217
129741 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1298.png
21 203 4 217
1298 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129824.png
23 201 4 218
129824 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129825.png
46 177 4 219
129825 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129826.png
5 218 4 219
129826 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129827.png
12 211 4 217
129827 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129828.png
5 218 4 218
129828 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129829.png
33 190 4 217
129829 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129830.png
27 196 4 216
129830 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129831.png
48 175 8 216
129831 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129832.png
5 218 4 218
129832 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129833.png
5 218 11 210
129833 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129834.png
24 199 4 218
129834 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129835.png
18 205 4 219
129835 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129836.png
12 211 4 217
129836 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129837.png
8 214 5 214
129837 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129838.png
5 218 4 219
129838 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129839.png
21 203 4 219
129839 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129840.png
24 198 5 216
129840 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129841.png
37 182 8 214
129841 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129842.png
5 217 12 210
129842 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129843.png
23 200 4 217
129843 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129844.png
27 196 4 218
129844 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129845.png
42 180 8 215
129845 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129846.png
32 192 5 214
129846 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129847.png
45 177 5 216
129847 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129848.png
32 191 5 217
129848 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129849.png
20 202 5 216
129849 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129850.png
14 208 4 216
129850 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129851.png
22 201 4 216
129851 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129852.png
6 217 5 218
129852 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129853.png
38 187 9 215
129853 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129854.png
33 190 5 216
129854 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129855.png
8 216 4 218
129855 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129856.png
5 218 4 218
129856 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129857.png
19 203 4 216
129857 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129858.png
35 188 4 217
129858 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129859.png
37 186 4 218
129859 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129860.png
16 202 8 217
129860 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129861.png
26 196 4 216
129861 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129862.png
6 217 5 216
129862 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129863.png
18 205 5 218
129863 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129864.png
21 202 4 217
129864 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129865.png
20 203 4 217
129865 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129866.png
26 197 5 216
129866 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129867.png
30 193 4 219
129867 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129868.png
8 215 6 214
129868 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129869.png
43 180 4 217
129869 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129870.png
21 202 4 217
129870 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129871.png
25 198 4 218
129871 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129872.png
5 218 4 219
129872 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129873.png
51 172 5 218
129873 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129874.png
5 218 4 218
129874 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129875.png
6 218 4 217
129875 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129876.png
42 181 4 217
129876 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129877.png
5 218 4 219
129877 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129878.png
37 186 5 216
129878 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129879.png
42 181 5 216
129879 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129880.png
5 218 4 219
129880 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129881.png
24 199 4 219
129881 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129882.png
18 205 5 217
129882 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129883.png
30 193 4 217
129883 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129884.png
17 206 5 217
129884 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129885.png
8 215 6 214
129885 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129886.png
7 216 8 218
129886 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129887.png
18 205 4 219
129887 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129888.png
5 218 4 219
129888 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129889.png
8 216 4 218
129889 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129890.png
28 195 4 219
129890 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129891.png
29 195 4 216
129891 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129892.png
5 218 5 218
129892 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129893.png
20 203 4 218
129893 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129894.png
9 214 4 216
129894 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129895.png
43 180 4 217
129895 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\129896.png
22 202 5 219
129896 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1299.png
3 220 11 211
1299 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1300.png
16 207 4 218
1300 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130031.png
6 217 21 197
130031 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130032.png
19 203 6 214
130032 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130035.png
5 218 4 219
130035 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130036.png
46 177 5 216
130036 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130037.png
37 185 4 216
130037 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130038.png
5 218 9 214
130038 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130041.png
6 218 6 216
130041 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130042.png
6 217 5 216
130042 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130043.png
5 218 40 182
130043 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130044.png
26 197 4 219
130044 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130045.png
6 217 5 216
130045 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130046.png
30 193 4 216
130046 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130047.png
6 215 3 216
130047 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130048.png
23 200 8 216
130048 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130049.png
6 217 5 216
130049 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130050.png
24 198 5 216
130050 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130051.png
6 217 5 216
130051 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130052.png
6 218 14 208
130052 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130053.png
27 196 5 218
130053 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130054.png
6 217 25 193
130054 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130055.png
5 218 4 219
130055 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130056.png
24 199 5 218
130056 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130057.png
5 218 28 194
130057 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130058.png
5 218 4 219
130058 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130059.png
7 218 7 219
130059 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130060.png
5 218 5 220
130060 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130061.png
6 217 5 216
130061 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130062.png
5 218 32 190
130062 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130063.png
5 218 4 219
130063 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130064.png
5 218 23 198
130064 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130065.png
20 203 4 216
130065 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130066.png
5 218 10 213
130066 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130067.png
5 219 6 214
130067 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1301.png
43 180 4 219
1301 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130127.png
2 221 25 201
130127 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130153.png
5 218 4 219
130153 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130158.png
23 198 3 216
130158 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130159.png
38 187 5 214
130159 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130160.png
29 194 4 217
130160 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130161.png
8 214 9 216
130161 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130162.png
5 218 4 218
130162 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130163.png
20 202 4 216
130163 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130164.png
44 179 3 216
130164 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130165.png
16 207 5 218
130165 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130166.png
16 206 5 216
130166 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130167.png
5 218 21 201
130167 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130168.png
38 185 5 216
130168 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130169.png
5 218 4 219
130169 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130170.png
8 215 6 214
130170 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130171.png
20 203 4 218
130171 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130172.png
8 215 6 214
130172 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130173.png
5 218 4 219
130173 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130174.png
6 217 9 210
130174 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130175.png
5 218 11 211
130175 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130176.png
5 218 15 208
130176 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130177.png
8 215 6 214
130177 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130193.png
5 218 14 207
130193 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130194.png
10 213 4 217
130194 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130199.png
21 202 6 218
130199 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1302.png
30 193 4 217
1302 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130202.png
42 182 5 219
130202 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130204.png
13 210 5 217
130204 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130209.png
34 190 4 219
130209 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130211.png
20 202 5 216
130211 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130212.png
17 206 5 214
130212 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130218.png
34 189 4 216
130218 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130219.png
32 190 5 216
130219 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130220.png
33 190 4 218
130220 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130229.png
25 198 3 216
130229 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130239.png
51 172 5 217
130239 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130241.png
6 217 7 216
130241 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130260.png
52 170 4 217
130260 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130261.png
12 211 4 218
130261 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130262.png
5 217 63 159
130262 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130263.png
6 217 5 215
130263 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130266.png
25 198 4 217
130266 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130270.png
10 216 63 160
130270 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130274.png
37 186 5 218
130274 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130276.png
28 195 5 216
130276 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130287.png
6 217 5 216
130287 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130288.png
5 218 4 219
130288 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130289.png
12 212 5 214
130289 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130290.png
21 202 4 216
130290 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130292.png
3 220 6 217
130292 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130293.png
26 197 4 218
130293 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130296.png
13 211 5 216
130296 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130297.png
5 218 4 219
130297 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130298.png
42 182 6 214
130298 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130299.png
16 207 5 217
130299 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1303.png
18 205 4 217
1303 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130300.png
48 174 8 216
130300 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130304.png
10 213 4 217
130304 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130337.png
6 217 4 215
130337 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130338.png
5 218 20 202
130338 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130339.png
28 196 4 216
130339 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130340.png
6 218 6 216
130340 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130341.png
22 199 5 216
130341 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130342.png
17 205 5 214
130342 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130343.png
31 192 4 216
130343 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130344.png
6 217 5 216
130344 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130345.png
12 211 4 219
130345 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130346.png
11 211 6 214
130346 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130347.png
32 191 4 216
130347 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130348.png
27 197 5 217
130348 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130349.png
8 215 5 214
130349 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130350.png
5 218 4 219
130350 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130351.png
22 201 4 216
130351 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130352.png
5 218 4 218
130352 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130353.png
5 218 4 218
130353 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130354.png
18 206 5 217
130354 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130355.png
18 205 6 214
130355 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130356.png
10 213 8 216
130356 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130357.png
6 217 5 216
130357 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130358.png
37 186 4 218
130358 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130359.png
5 218 5 217
130359 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130360.png
5 218 18 203
130360 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130361.png
28 195 4 219
130361 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130362.png
21 203 4 217
130362 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130363.png
41 182 4 217
130363 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130364.png
18 204 8 217
130364 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130365.png
15 208 4 219
130365 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130366.png
5 218 4 219
130366 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130367.png
5 218 4 218
130367 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130368.png
26 197 5 217
130368 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130369.png
37 186 4 216
130369 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130370.png
15 208 4 218
130370 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130371.png
6 217 5 216
130371 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130372.png
5 218 4 219
130372 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130373.png
37 186 6 216
130373 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130374.png
5 218 4 219
130374 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130375.png
5 218 6 217
130375 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130376.png
8 215 6 214
130376 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130377.png
26 197 4 217
130377 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130378.png
31 192 4 217
130378 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130379.png
11 212 4 218
130379 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130380.png
5 218 4 219
130380 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130381.png
22 201 4 218
130381 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130382.png
32 191 5 216
130382 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1304.png
5 218 4 218
1304 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130495.png
26 197 4 218
130495 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130496.png
18 205 5 217
130496 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130497.png
5 218 9 212
130497 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130498.png
7 216 9 211
130498 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130499.png
19 203 6 214
130499 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1305.png
5 218 37 185
1305 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130500.png
55 169 5 214
130500 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130501.png
5 218 6 217
130501 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130502.png
14 208 9 217
130502 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130503.png
44 179 4 218
130503 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130504.png
52 170 5 216
130504 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130505.png
22 201 4 219
130505 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130506.png
27 196 4 218
130506 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130507.png
8 215 6 214
130507 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130508.png
6 218 5 217
130508 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130509.png
33 190 6 218
130509 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130510.png
38 184 6 214
130510 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130511.png
36 187 4 218
130511 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130512.png
41 182 4 216
130512 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130513.png
51 172 4 218
130513 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130514.png
30 193 4 218
130514 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130515.png
7 216 6 217
130515 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130516.png
16 207 7 218
130516 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130517.png
33 189 6 214
130517 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130518.png
39 185 5 214
130518 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130519.png
7 216 5 219
130519 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130520.png
5 218 4 219
130520 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130521.png
20 203 4 217
130521 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130522.png
30 193 8 218
130522 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130523.png
11 212 5 216
130523 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130524.png
41 182 7 216
130524 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130525.png
34 190 4 218
130525 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130526.png
8 215 6 214
130526 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130527.png
22 201 4 217
130527 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130528.png
6 217 9 211
130528 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130529.png
5 218 11 210
130529 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130530.png
7 216 7 217
130530 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130531.png
5 218 4 219
130531 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130532.png
58 165 5 216
130532 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130534.png
5 219 21 198
130534 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130535.png
11 213 5 217
130535 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130536.png
5 218 4 219
130536 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130537.png
46 178 4 214
130537 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130538.png
8 214 7 217
130538 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130539.png
5 218 4 219
130539 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130540.png
5 218 61 162
130540 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130541.png
44 179 10 207
130541 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130542.png
21 202 5 219
130542 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130543.png
5 218 4 219
130543 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130544.png
51 173 4 217
130544 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130545.png
25 198 4 217
130545 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130546.png
21 202 8 218
130546 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130547.png
14 209 5 217
130547 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130548.png
10 215 7 216
130548 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130549.png
30 193 4 217
130549 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130550.png
12 210 5 214
130550 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130551.png
22 203 6 214
130551 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130552.png
6 218 8 213
130552 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130553.png
6 218 5 215
130553 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130554.png
10 214 4 218
130554 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130555.png
23 200 5 215
130555 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130556.png
5 218 53 168
130556 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130557.png
23 199 6 218
130557 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130558.png
14 208 8 215
130558 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130559.png
48 175 4 219
130559 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130560.png
5 218 53 168
130560 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130561.png
5 218 8 211
130561 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130562.png
6 216 13 208
130562 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130563.png
6 218 34 189
130563 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130591.png
5 218 5 217
130591 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130592.png
36 188 4 217
130592 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130593.png
8 215 6 214
130593 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130594.png
5 218 4 219
130594 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130595.png
49 174 6 215
130595 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1306.png
20 204 6 217
1306 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130637.png
8 215 6 214
130637 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1307.png
5 218 65 156
1307 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130762.png
12 211 5 216
130762 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130763.png
22 201 4 218
130763 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130764.png
32 191 5 217
130764 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130765.png
10 213 5 216
130765 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130766.png
5 218 4 219
130766 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130767.png
5 218 4 219
130767 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130768.png
5 218 4 219
130768 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130769.png
6 218 8 215
130769 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130770.png
5 218 4 219
130770 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130771.png
6 217 21 202
130771 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130772.png
45 178 4 218
130772 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130773.png
5 218 58 161
130773 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130774.png
42 181 5 218
130774 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130775.png
21 201 6 214
130775 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130776.png
18 205 4 214
130776 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130777.png
21 202 7 219
130777 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130778.png
8 215 6 214
130778 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130779.png
57 167 4 214
130779 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130780.png
31 191 3 216
130780 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130781.png
6 217 5 216
130781 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130782.png
23 201 4 215
130782 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130783.png
5 218 4 218
130783 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130784.png
20 202 4 214
130784 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130785.png
22 200 6 214
130785 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130786.png
58 165 4 216
130786 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130787.png
28 195 4 218
130787 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130788.png
25 198 4 216
130788 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130789.png
34 189 6 214
130789 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130790.png
26 197 4 216
130790 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130791.png
5 218 4 218
130791 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130792.png
12 211 4 217
130792 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130793.png
7 215 5 216
130793 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130794.png
8 215 7 213
130794 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130795.png
29 194 4 219
130795 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130796.png
43 180 4 219
130796 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130797.png
23 200 4 214
130797 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130798.png
5 218 12 210
130798 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130799.png
7 216 8 217
130799 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1308.png
6 217 4 219
1308 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130800.png
6 218 6 214
130800 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130801.png
41 182 4 218
130801 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130802.png
17 206 4 216
130802 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130803.png
16 207 3 216
130803 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130804.png
7 216 4 217
130804 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130805.png
5 218 4 219
130805 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130808.png
23 201 4 219
130808 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130809.png
39 184 4 219
130809 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130810.png
5 218 21 200
130810 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130811.png
36 187 4 216
130811 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130812.png
5 218 4 219
130812 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130814.png
17 206 4 217
130814 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130815.png
25 198 4 217
130815 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130816.png
31 192 4 217
130816 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130817.png
47 176 4 219
130817 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130820.png
5 218 5 216
130820 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130821.png
15 208 4 217
130821 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130823.png
6 218 4 217
130823 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130824.png
7 216 8 218
130824 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130825.png
41 181 5 216
130825 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130826.png
19 204 4 217
130826 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130827.png
26 199 6 214
130827 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130828.png
27 196 4 219
130828 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130834.png
5 218 37 185
130834 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130835.png
16 206 4 217
130835 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130836.png
21 202 4 217
130836 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130837.png
7 216 8 218
130837 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130838.png
9 214 7 214
130838 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130839.png
5 218 4 219
130839 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130840.png
26 197 4 217
130840 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130844.png
29 194 4 219
130844 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130845.png
27 196 4 218
130845 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130846.png
7 216 34 189
130846 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130847.png
8 215 7 212
130847 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130848.png
37 186 5 217
130848 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130849.png
34 189 4 219
130849 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130850.png
6 217 14 209
130850 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130851.png
21 203 6 214
130851 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130852.png
29 194 4 216
130852 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130853.png
6 218 7 215
130853 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130854.png
6 216 4 219
130854 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130855.png
5 218 6 220
130855 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130856.png
22 201 4 219
130856 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130857.png
26 197 5 216
130857 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130859.png
29 194 4 217
130859 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130861.png
23 199 5 214
130861 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130862.png
25 197 5 216
130862 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130863.png
15 207 4 217
130863 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130864.png
18 205 4 217
130864 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130865.png
39 183 5 216
130865 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130866.png
44 179 5 217
130866 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130868.png
26 197 5 218
130868 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130869.png
5 218 4 219
130869 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130870.png
30 193 5 216
130870 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130871.png
19 204 4 216
130871 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130872.png
9 213 5 214
130872 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130873.png
37 186 5 216
130873 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130874.png
5 218 4 219
130874 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130875.png
29 195 6 219
130875 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130877.png
23 200 4 216
130877 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130879.png
34 189 6 214
130879 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130881.png
7 217 26 194
130881 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130882.png
15 208 4 219
130882 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1309.png
5 218 42 180
1309 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130939.png
28 195 4 219
130939 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130941.png
17 207 4 217
130941 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\130943.png
19 205 6 214
130943 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1310.png
5 218 25 195
1310 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1311.png
5 218 52 167
1311 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131108.png
23 202 5 214
131108 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131111.png
35 189 4 213
131111 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131120.png
21 203 5 214
131120 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131125.png
24 199 4 216
131125 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131127.png
23 200 4 218
131127 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131129.png
22 201 4 218
131129 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131135.png
46 177 6 218
131135 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131137.png
30 192 5 219
131137 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131139.png
8 215 7 213
131139 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131140.png
6 217 6 217
131140 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131142.png
15 208 4 219
131142 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131147.png
5 218 4 219
131147 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131152.png
12 211 4 219
131152 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131154.png
13 211 22 200
131154 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131155.png
5 218 29 192
131155 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131158.png
25 198 4 218
131158 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131159.png
6 217 32 191
131159 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131162.png
7 216 12 208
131162 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131166.png
5 217 80 142
131166 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131169.png
31 192 5 216
131169 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131178.png
29 194 4 218
131178 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131183.png
10 214 4 219
131183 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131190.png
31 192 5 217
131190 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131195.png
10 213 5 218
131195 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131197.png
35 188 4 217
131197 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131199.png
5 218 4 219
131199 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1312.png
5 218 33 189
1312 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131200.png
57 165 9 218
131200 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131205.png
5 216 28 194
131205 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131209.png
33 190 4 218
131209 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131210.png
16 211 5 219
131210 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131212.png
5 218 40 183
131212 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131217.png
6 217 38 184
131217 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131219.png
48 175 4 217
131219 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131222.png
5 218 9 211
131222 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131225.png
5 218 5 218
131225 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131229.png
8 215 6 214
131229 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131240.png
15 208 6 219
131240 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131246.png
31 191 5 216
131246 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131249.png
5 218 8 215
131249 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131250.png
8 215 6 214
131250 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131252.png
39 183 8 218
131252 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131255.png
32 191 6 218
131255 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131256.png
5 217 5 219
131256 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131258.png
16 207 3 216
131258 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131261.png
24 199 7 218
131261 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131264.png
22 201 5 217
131264 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131266.png
39 183 6 218
131266 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131267.png
15 208 4 218
131267 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131270.png
34 189 4 217
131270 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131280.png
22 199 5 216
131280 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\131289.png
5 218 4 219
131289 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1313.png
25 198 4 219
1313 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1314.png
31 191 4 217
1314 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1315.png
15 208 5 216
1315 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1316.png
5 218 45 176
1316 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1317.png
38 184 8 216
1317 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1318.png
5 218 42 180
1318 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1319.png
5 218 4 219
1319 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1320.png
10 213 5 217
1320 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1323.png
23 200 5 216
1323 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1324.png
5 218 13 206
1324 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1325.png
10 214 4 218
1325 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1326.png
5 218 50 171
1326 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1327.png
5 218 4 218
1327 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1328.png
32 192 4 219
1328 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1330.png
8 215 6 214
1330 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1331.png
7 216 4 217
1331 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\133178.png
10 213 5 214
133178 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\133179.png
35 188 4 218
133179 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1332.png
22 201 4 219
1332 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\133235.png
32 190 5 216
133235 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1333.png
12 211 4 218
1333 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1334.png
27 196 4 217
1334 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1335.png
6 217 5 216
1335 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1336.png
6 217 7 213
1336 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\133669.png
5 218 4 219
133669 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\133670.png
5 218 4 219
133670 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\133671.png
5 218 5 218
133671 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\133672.png
5 218 26 196
133672 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\133673.png
5 218 4 218
133673 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\133674.png
7 216 17 203
133674 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\133675.png
6 218 35 188
133675 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\133676.png
8 215 6 214
133676 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\133677.png
6 217 5 216
133677 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\133678.png
6 216 7 218
133678 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\133679.png
6 218 33 189
133679 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\133680.png
5 218 4 216
133680 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\133681.png
6 217 5 216
133681 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1337.png
5 218 4 219
1337 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\133765.png
15 208 4 219
133765 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1338.png
6 217 6 218
1338 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1339.png
16 208 4 217
1339 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1340.png
40 184 4 214
1340 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1341.png
6 217 5 216
1341 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1343.png
6 218 8 212
1343 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1344.png
6 218 6 216
1344 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134636.png
5 219 30 190
134636 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134637.png
18 204 6 214
134637 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1347.png
23 200 4 219
1347 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134732.png
5 218 4 219
134732 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1348.png
18 205 4 217
1348 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134886.png
21 203 4 217
134886 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134887.png
5 218 4 219
134887 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134888.png
5 218 4 219
134888 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134889.png
5 218 4 219
134889 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134890.png
22 201 4 217
134890 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134891.png
5 218 7 216
134891 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134892.png
6 217 8 218
134892 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134893.png
19 204 5 218
134893 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134894.png
45 178 4 218
134894 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134896.png
13 211 4 217
134896 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134898.png
5 218 6 220
134898 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1349.png
3 220 52 173
1349 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134900.png
5 218 4 219
134900 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134901.png
15 207 4 216
134901 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134903.png
15 207 6 219
134903 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134904.png
5 218 6 220
134904 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134906.png
11 211 5 218
134906 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134908.png
33 190 6 217
134908 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134910.png
33 190 6 217
134910 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134913.png
6 217 21 200
134913 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134916.png
11 212 5 217
134916 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134919.png
21 202 4 218
134919 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134920.png
44 180 3 216
134920 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\134923.png
32 191 4 218
134923 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1350.png
5 218 7 215
1350 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1351.png
5 218 4 219
1351 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135139.png
26 197 8 216
135139 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135214.png
34 188 5 214
135214 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135215.png
7 216 13 208
135215 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135216.png
28 194 6 214
135216 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135217.png
8 215 6 214
135217 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135245.png
13 210 4 216
135245 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135246.png
12 211 5 216
135246 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135247.png
17 206 4 217
135247 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135248.png
18 205 5 216
135248 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135249.png
28 195 6 214
135249 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135250.png
24 199 4 217
135250 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135251.png
15 208 4 219
135251 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135252.png
13 211 5 217
135252 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135253.png
5 218 10 209
135253 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135254.png
6 217 4 219
135254 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135255.png
16 208 9 215
135255 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135256.png
12 214 9 216
135256 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135257.png
11 214 10 216
135257 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135258.png
33 189 6 219
135258 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135292.png
8 215 8 218
135292 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135297.png
5 218 10 212
135297 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1353.png
5 218 54 168
1353 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135301.png
23 201 5 216
135301 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135303.png
8 215 4 216
135303 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135304.png
30 193 4 219
135304 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135305.png
22 201 5 218
135305 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135306.png
24 198 5 214
135306 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135308.png
27 196 4 218
135308 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135311.png
37 186 5 218
135311 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135312.png
28 195 5 216
135312 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135313.png
11 212 4 217
135313 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135315.png
5 218 12 211
135315 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135316.png
7 216 14 210
135316 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135317.png
17 206 5 217
135317 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135318.png
19 206 8 216
135318 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135325.png
23 200 4 219
135325 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135326.png
23 201 5 216
135326 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135327.png
6 217 4 217
135327 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135328.png
27 196 6 214
135328 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135329.png
20 203 4 214
135329 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135330.png
26 198 6 218
135330 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135331.png
30 194 7 217
135331 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135332.png
6 218 5 217
135332 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135333.png
6 218 4 218
135333 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135336.png
5 218 29 194
135336 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135337.png
5 218 77 145
135337 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135338.png
14 209 3 216
135338 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135339.png
8 215 39 185
135339 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135340.png
33 189 8 218
135340 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135342.png
32 190 5 214
135342 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135344.png
6 217 38 185
135344 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135345.png
5 218 6 217
135345 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135346.png
5 218 4 218
135346 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135347.png
5 218 4 219
135347 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135348.png
21 202 5 216
135348 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135349.png
20 203 5 216
135349 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135351.png
50 174 6 214
135351 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135353.png
13 210 4 219
135353 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135356.png
8 215 6 214
135356 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135359.png
17 206 6 214
135359 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135362.png
8 215 6 214
135362 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135365.png
35 187 8 217
135365 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135368.png
27 194 5 216
135368 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135371.png
21 203 4 214
135371 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135372.png
40 183 6 217
135372 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135377.png
18 204 6 214
135377 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135382.png
29 193 5 214
135382 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135388.png
34 189 4 219
135388 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135389.png
37 184 4 216
135389 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135390.png
25 198 5 216
135390 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135391.png
6 217 5 216
135391 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135392.png
18 205 5 217
135392 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135393.png
38 184 6 214
135393 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135394.png
19 205 5 216
135394 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135395.png
35 188 4 218
135395 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135396.png
8 215 6 214
135396 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135397.png
26 197 4 218
135397 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135399.png
32 191 3 216
135399 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135400.png
22 201 4 216
135400 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135401.png
5 218 4 219
135401 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135402.png
31 192 4 218
135402 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135403.png
14 209 4 218
135403 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135404.png
15 208 5 216
135404 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135405.png
14 208 5 216
135405 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135406.png
20 203 5 216
135406 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135407.png
9 214 4 216
135407 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135408.png
11 212 5 214
135408 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135409.png
5 218 4 219
135409 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135410.png
17 205 5 214
135410 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135411.png
28 194 6 217
135411 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135413.png
5 218 4 219
135413 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135414.png
8 214 5 220
135414 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135423.png
32 192 5 214
135423 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135424.png
32 191 4 219
135424 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135425.png
5 218 16 206
135425 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135426.png
11 212 7 218
135426 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135427.png
6 217 5 216
135427 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135428.png
8 215 33 177
135428 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135430.png
31 195 7 216
135430 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135432.png
25 198 4 216
135432 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135433.png
24 198 4 216
135433 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135434.png
17 206 6 219
135434 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135435.png
20 203 5 216
135435 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135436.png
20 203 8 216
135436 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135439.png
10 213 4 218
135439 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135442.png
20 203 4 216
135442 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135443.png
5 218 4 219
135443 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135448.png
36 188 6 214
135448 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135449.png
20 203 5 216
135449 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135450.png
8 215 6 214
135450 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135452.png
8 215 6 214
135452 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135453.png
41 181 5 217
135453 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135454.png
46 177 6 217
135454 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135455.png
8 215 6 214
135455 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135467.png
8 215 6 214
135467 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135468.png
8 214 4 214
135468 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135469.png
8 214 6 214
135469 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135470.png
25 197 5 214
135470 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135471.png
24 198 5 214
135471 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135472.png
8 214 6 214
135472 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135473.png
5 218 5 217
135473 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135474.png
4 219 52 167
135474 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135475.png
11 212 4 218
135475 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135476.png
27 195 4 214
135476 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135477.png
5 218 20 201
135477 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135478.png
6 217 8 214
135478 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135479.png
18 205 5 217
135479 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135480.png
46 177 4 217
135480 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135481.png
32 190 4 214
135481 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135482.png
11 213 11 214
135482 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135485.png
6 218 5 218
135485 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135488.png
21 202 4 219
135488 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135491.png
26 197 4 217
135491 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135494.png
19 203 5 216
135494 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135497.png
44 178 4 214
135497 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1355.png
11 213 4 217
1355 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135500.png
29 194 4 219
135500 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135503.png
5 218 4 219
135503 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135506.png
22 201 4 219
135506 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135509.png
24 199 5 216
135509 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135512.png
5 218 4 219
135512 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135515.png
18 206 5 216
135515 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135518.png
8 217 3 217
135518 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135521.png
27 196 8 217
135521 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135523.png
33 190 5 217
135523 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135526.png
6 218 6 220
135526 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135527.png
39 184 6 220
135527 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135528.png
6 218 15 207
135528 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135531.png
5 218 16 206
135531 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135534.png
32 189 3 216
135534 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135535.png
5 218 5 219
135535 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135540.png
25 198 4 219
135540 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1356.png
6 217 49 173
1356 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1357.png
37 186 4 216
1357 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1358.png
5 218 56 165
1358 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135853.png
5 218 15 208
135853 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1359.png
5 218 15 208
1359 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135943.png
28 195 4 217
135943 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135944.png
13 210 4 217
135944 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135945.png
5 217 47 176
135945 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\135946.png
8 215 4 219
135946 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1360.png
5 218 39 184
1360 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136007.png
20 201 6 215
136007 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136008.png
17 206 6 216
136008 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136009.png
21 202 4 217
136009 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136010.png
49 175 4 217
136010 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136011.png
39 184 4 214
136011 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136012.png
5 218 4 219
136012 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136013.png
5 218 38 181
136013 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136014.png
29 194 4 219
136014 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136015.png
5 218 6 216
136015 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136016.png
22 201 6 214
136016 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136017.png
18 206 3 216
136017 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136018.png
30 193 5 216
136018 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136019.png
29 194 4 219
136019 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136020.png
37 186 5 216
136020 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136021.png
26 197 3 216
136021 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136022.png
15 207 4 216
136022 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136023.png
5 217 16 207
136023 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136024.png
41 183 6 214
136024 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1361.png
27 198 5 217
1361 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136150.png
6 217 5 215
136150 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136151.png
5 218 4 219
136151 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136152.png
24 198 5 216
136152 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136153.png
18 205 6 214
136153 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136154.png
26 197 4 218
136154 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136155.png
18 205 4 219
136155 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136156.png
45 178 4 219
136156 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136157.png
8 215 6 214
136157 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136158.png
19 204 4 219
136158 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136159.png
5 218 4 219
136159 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136160.png
19 203 11 214
136160 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136161.png
31 192 4 218
136161 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136174.png
33 191 4 214
136174 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136175.png
6 217 5 216
136175 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136176.png
19 204 5 216
136176 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136177.png
8 215 8 217
136177 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136178.png
8 213 6 218
136178 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136179.png
19 204 4 219
136179 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136180.png
33 190 4 219
136180 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136181.png
27 196 4 218
136181 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136182.png
5 218 4 219
136182 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136183.png
35 187 8 216
136183 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136184.png
19 204 4 219
136184 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136185.png
5 218 4 218
136185 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136186.png
5 218 4 217
136186 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136187.png
5 217 7 215
136187 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136188.png
31 193 4 217
136188 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136189.png
8 215 6 214
136189 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136190.png
5 218 4 218
136190 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136196.png
41 182 5 216
136196 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136197.png
9 214 4 219
136197 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136198.png
5 218 18 202
136198 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136199.png
7 216 8 218
136199 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136200.png
5 218 8 214
136200 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136201.png
5 218 4 219
136201 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136202.png
36 187 5 216
136202 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136203.png
6 217 5 216
136203 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136204.png
5 218 4 219
136204 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136205.png
5 218 4 219
136205 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136206.png
5 218 4 219
136206 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136207.png
17 205 4 214
136207 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136208.png
5 218 4 219
136208 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136209.png
5 218 4 219
136209 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136210.png
35 187 6 214
136210 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136211.png
5 218 40 182
136211 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136212.png
25 198 8 217
136212 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136227.png
5 218 4 219
136227 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136228.png
5 218 4 219
136228 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136229.png
26 197 5 217
136229 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136230.png
5 218 7 215
136230 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136231.png
36 187 5 216
136231 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136232.png
38 186 4 219
136232 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136233.png
8 216 5 216
136233 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136234.png
14 208 5 216
136234 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136235.png
6 217 7 213
136235 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136236.png
5 218 7 216
136236 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136237.png
5 218 7 215
136237 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136238.png
5 218 4 218
136238 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136239.png
36 186 4 216
136239 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136240.png
26 196 4 219
136240 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136241.png
5 218 28 194
136241 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136242.png
37 186 5 216
136242 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136243.png
6 217 5 216
136243 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136244.png
6 217 5 216
136244 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136245.png
32 190 5 216
136245 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136246.png
6 217 4 215
136246 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136247.png
15 207 4 217
136247 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136248.png
5 218 4 219
136248 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136249.png
10 213 4 216
136249 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136250.png
5 218 4 218
136250 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136251.png
5 218 4 219
136251 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136252.png
56 168 4 216
136252 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136253.png
9 215 7 215
136253 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136254.png
5 218 13 209
136254 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136255.png
46 177 4 218
136255 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136256.png
21 202 5 216
136256 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136257.png
6 217 5 216
136257 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136258.png
32 191 4 218
136258 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136259.png
26 197 4 218
136259 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136260.png
5 218 4 218
136260 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136261.png
51 172 4 218
136261 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136262.png
5 218 4 219
136262 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136263.png
7 216 5 217
136263 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136264.png
21 202 4 219
136264 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136265.png
6 217 5 216
136265 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136266.png
18 205 4 219
136266 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136267.png
7 216 4 217
136267 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136268.png
47 176 5 216
136268 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136269.png
33 190 4 219
136269 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136270.png
8 215 20 203
136270 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136271.png
6 217 5 216
136271 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136272.png
6 217 32 191
136272 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136273.png
7 215 8 218
136273 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136274.png
26 197 5 214
136274 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136276.png
5 218 4 218
136276 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136277.png
27 196 4 219
136277 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136278.png
29 193 5 216
136278 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136279.png
8 215 6 214
136279 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136280.png
33 190 4 219
136280 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136281.png
27 196 4 218
136281 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136282.png
5 218 4 219
136282 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136283.png
6 217 11 211
136283 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136284.png
29 194 5 216
136284 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136285.png
5 218 4 218
136285 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136286.png
5 218 4 219
136286 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136287.png
36 187 4 218
136287 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136288.png
7 216 8 218
136288 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136289.png
14 209 4 219
136289 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136290.png
18 205 4 217
136290 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136291.png
6 218 5 218
136291 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136292.png
7 216 8 218
136292 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136293.png
5 218 10 213
136293 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136294.png
7 216 8 218
136294 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136295.png
6 216 6 220
136295 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136297.png
30 193 4 217
136297 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136298.png
13 210 5 216
136298 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136299.png
31 193 4 219
136299 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1363.png
30 192 5 217
1363 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136300.png
5 218 4 219
136300 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136301.png
6 217 5 216
136301 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136302.png
13 211 4 216
136302 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136303.png
22 201 5 216
136303 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136304.png
6 217 5 216
136304 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136305.png
45 178 4 216
136305 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136306.png
17 207 4 218
136306 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1364.png
5 218 44 175
1364 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136416.png
21 202 4 218
136416 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136417.png
34 189 4 216
136417 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136418.png
29 195 15 210
136418 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136420.png
42 181 4 217
136420 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136421.png
17 205 3 216
136421 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136422.png
39 184 3 216
136422 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136423.png
22 201 4 214
136423 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136426.png
16 207 4 217
136426 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136427.png
32 190 4 216
136427 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136429.png
6 218 4 216
136429 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136430.png
6 217 5 216
136430 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136431.png
6 218 15 205
136431 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136432.png
5 216 3 216
136432 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136433.png
7 216 8 218
136433 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136434.png
5 218 4 219
136434 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136435.png
5 218 86 136
136435 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136436.png
5 218 5 217
136436 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136437.png
6 217 5 216
136437 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136459.png
24 198 4 216
136459 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136460.png
43 180 5 214
136460 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136461.png
5 218 61 159
136461 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136462.png
16 208 4 214
136462 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136463.png
34 189 4 217
136463 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136464.png
9 213 5 216
136464 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136465.png
8 216 8 217
136465 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136466.png
22 201 5 218
136466 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136467.png
41 182 5 216
136467 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136468.png
34 188 4 216
136468 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136469.png
18 205 4 219
136469 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136470.png
29 194 4 217
136470 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136471.png
5 218 23 196
136471 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136472.png
5 218 6 220
136472 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136473.png
18 205 5 217
136473 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136474.png
20 203 4 217
136474 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136475.png
5 218 4 219
136475 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136476.png
16 206 6 214
136476 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136477.png
12 211 4 217
136477 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1365.png
20 203 5 219
1365 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136531.png
5 218 4 219
136531 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136548.png
27 196 7 219
136548 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136549.png
10 214 8 217
136549 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136550.png
5 218 24 196
136550 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136583.png
5 218 4 219
136583 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136587.png
6 217 37 187
136587 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136588.png
6 218 44 177
136588 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1366.png
6 217 5 214
1366 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136610.png
37 186 7 217
136610 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136612.png
17 205 5 214
136612 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136614.png
5 218 4 218
136614 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136615.png
5 218 12 210
136615 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136620.png
25 198 11 213
136620 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136624.png
50 174 3 214
136624 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136625.png
8 214 6 214
136625 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136636.png
7 217 47 175
136636 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136637.png
8 216 7 210
136637 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136639.png
9 214 10 217
136639 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\136647.png
5 218 54 169
136647 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1367.png
27 196 4 217
1367 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1369.png
6 217 30 191
1369 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1370.png
5 218 4 219
1370 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1371.png
14 209 4 217
1371 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1664.png
20 204 6 214
1664 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1668.png
7 216 4 217
1668 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1687.png
30 192 5 216
1687 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1707.png
24 199 4 217
1707 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1710.png
10 213 4 219
1710 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1714.png
65 158 5 214
1714 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1717.png
23 199 4 218
1717 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1724.png
53 170 5 214
1724 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1733.png
6 217 18 200
1733 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1737.png
35 188 4 219
1737 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1742.png
19 203 4 218
1742 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1749.png
49 174 4 217
1749 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1775.png
31 192 4 219
1775 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1777.png
27 195 5 214
1777 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1885.png
47 176 5 218
1885 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1916.png
22 201 5 214
1916 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1919.png
35 186 6 214
1919 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1922.png
32 191 5 216
1922 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1923.png
8 215 6 214
1923 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1924.png
34 189 4 217
1924 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1926.png
28 195 5 216
1926 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\1992.png
21 202 4 217
1992 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\2097.png
37 186 5 219
2097 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\2238.png
30 193 4 219
2238 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\602.png
21 202 4 217
602 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\603.png
37 186 5 216
603 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\617.png
6 217 5 216
617 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\622.png
21 202 5 216
622 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\630.png
8 215 6 214
630 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\642.png
28 195 4 214
642 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\676.png
35 188 5 214
676 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\679.png
6 217 5 216
679 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\680.png
8 214 5 214
680 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\691.png
19 203 6 214
691 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\713.png
18 205 4 219
713 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\728.png
61 162 4 217
728 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\732.png
19 204 4 217
732 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\735.png
16 206 5 216
735 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\78.png
24 199 6 219
78 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\82.png
21 201 4 216
82 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\858.png
5 218 9 214
858 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\862.png
25 197 5 219
862 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\865.png
22 201 6 214
865 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\868.png
5 218 4 219
868 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\87.png
43 181 4 213
87 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\89.png
9 214 5 218
89 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\907.png
5 218 4 219
907 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\908.png
45 177 3 216
908 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\918.png
7 216 10 215
918 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\920.png
6 217 5 216
920 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\921.png
5 218 4 219
921 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\93.png
24 199 4 217
93 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\944.png
45 178 4 219
944 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\96.png
17 206 6 218
96 :  (224, 224, 3)
G:\GIBD\Investigacion\Similitud Logos Clubes\imagenesBD5k\961.png
5 218 4 219
961 :  (224, 224, 3)
(224, 224, 3)
Forma del arreglo combinado: (2053, 224, 224, 3)
2053
2053
```


**[Celda 16 - Código]**
```python
## MOSTRAR LAS IMAGENES DE LA BASE DE DATOS Y DE LAS CONSULTAS
# base de datos
"""i = 0
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
"""    
idxBD = 2489
idxCons = 24
plt.imshow(imagenesBD[idxBD])
plt.show()    
plt.imshow(consultas[idxCons])
plt.show()    
print('Distancia: ',np.sqrt(np.sum(np.square(vectores_elemento[idxBD]-vectores_consulta[idxCons]))))
    
```


*Salida:*
```text
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Distancia:  97.19707
```

# Transfer Learning y Fine Tuning

# Búsqueda por Similitud de los Logos en base a los vectores de Características extraídos mediante Transfer Learning


**[Celda 19 - Código]**
```python
def show_search_images(search_images, search_id):
    # Crear una figura con una fila y 6 columnas (1 entrada + 5 resultados)
    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    
    # Mostrar la imagen de entrada
    axes[0].imshow(search_images["input_image"])
    axes[0].set_title(f'Search {search_id} - Query')
    axes[0].axis('off')

    # Mostrar las 5 imágenes resultantes
    for i in range(5):
        axes[i + 1].imshow(search_images["result_images"][i])
        axes[i + 1].set_title(f'Result {i + 1}')
        axes[i + 1].axis('off')

    # Ajustar el diseño para que las imágenes no se sobrepongan
    plt.tight_layout()
    plt.show()
```


**[Celda 20 - Código]**
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from PIL import Image

#### DICCIONARIO DE CONSULTA Y RESULTADOS PARA MOSTRAR LAS IMAGENES
searches =  {"input_image": np.random.rand(224, 224, 3), "result_images": [np.random.rand(224, 224, 3) for _ in range(5)] } 



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
### Resnet
# Inicialización de variables
vectores_elemento = []
tamanoLote = 50
desde = 0
hasta = tamanoLote

# Procesamiento en lotes
while hasta < len(imagenesBD):
    lote = imagenesBD[desde:hasta]
    vects = new_model_resnet.predict(lote)
    vectores_elemento.extend(vects)
    desde = hasta
    hasta += tamanoLote

# Procesamiento del lote final
lote = imagenesBD[desde:]
vects = new_model_resnet.predict(lote)
vectores_elemento.extend(vects)

vectores_consulta = new_model_resnet.predict(consultas)

print(vectores_elemento[0].shape)


#Inicio del for consulta
no_encontrados = []
i = 0
for cons,imgQ in zip(vectores_consulta,consultas):
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
        nnk(nnklist,[dist,j],10)
        j += 1

    searches["input_image"] = imgQ
    searches["result_images"] = []
    print("Consulta número: ", i)
    print("   Imagen de consulta: ", y_consultas[i])
    print('   Mas cercanos: ', nnk2etiquetas(nnklist, y_imagenesBD))
    encontrado = False
    ps = 1
    for par in nnklist:
        searches["result_images"].append(imagenesBD[par[1]])
        if y_imagenesBD[par[1]] == y_consultas[i]:
            encontrado = True
            contador_aciertos += 1
            print('--> ACIERTO')
            if ps == 1:
                contador_aciertosNN1 += 1
                print(f'    Primer Lugar: {contador_aciertosNN1}')
            if ps <= 3:
                contador_aciertosNN3 += 1
                print(f'    Entre los tres primeros: {contador_aciertosNN3}')
        ps += 1
    show_search_images(searches, i)
    print("\n")
    print("\n")
    if not encontrado:
        no_encontrados.append(y_consultas[i])
    i += 1

print("Porcentaje de aciertos NN1: ", contador_aciertosNN1/cantidad_consultas)
print("Porcentaje de aciertos NN3: ", contador_aciertosNN3/cantidad_consultas)
print("Porcentaje de aciertos NN5: ", contador_aciertos/cantidad_consultas)
print('  Cantidad consultas: ', cantidad_consultas)
```


*Salida:*
```text
<class 'numpy.ndarray'>
(56, 56, 64)
Consulta número:  0
   Imagen de consulta:  1
   Mas cercanos:  [[1.6581974, '1'], [2.0229897, '136160'], [2.07456, '1306'], [2.0778396, '105611'], [2.0968313, '135252'], [2.0980902, '135318'], [2.11091, '108913'], [2.1160793, '108981'], [2.127131, '105026'], [2.1400628, '102858']]
--> ACIERTO
    Primer Lugar: 1
    Entre los tres primeros: 1
<Figure size 2000x400 with 6 Axes>



Consulta número:  1
   Imagen de consulta:  102462
   Mas cercanos:  [[1.8231717, '102462'], [1.9719492, '1063'], [2.1800888, '135430'], [2.1809025, '130342'], [2.2621498, '106027'], [2.2814858, '1371'], [2.3193002, '1124'], [2.3228488, '102860'], [2.3328178, '120946'], [2.3515196, '1165']]
--> ACIERTO
    Primer Lugar: 2
    Entre los tres primeros: 2
<Figure size 2000x400 with 6 Axes>



Consulta número:  2
   Imagen de consulta:  102474
   Mas cercanos:  [[1.0749975, '102474'], [1.5595489, '1139'], [1.5980272, '102464'], [1.6041908, '1306'], [1.6461002, '131266'], [1.6909717, '1140'], [1.6980066, '102854'], [1.7088453, '126416'], [1.7146984, '130557'], [1.7182417, '101058']]
--> ACIERTO
    Primer Lugar: 3
    Entre los tres primeros: 3
<Figure size 2000x400 with 6 Axes>



Consulta número:  3
   Imagen de consulta:  102476
   Mas cercanos:  [[2.570239, '1317'], [2.5829403, '135217'], [2.5918355, '107332'], [2.602193, '105026'], [2.6044676, '130799'], [2.6225517, '130199'], [2.6246412, '107366'], [2.6335711, '134886'], [2.6336188, '106814'], [2.6341205, '135318']]
<Figure size 2000x400 with 6 Axes>



Consulta número:  4
   Imagen de consulta:  102852
   Mas cercanos:  [[0.90979666, '102852'], [2.2077801, '102359'], [2.2844398, '130065'], [2.430184, '130814'], [2.4912372, '130050'], [2.5077436, '130199'], [2.5125592, '131261'], [2.518484, '129873'], [2.5201697, '130370'], [2.5231547, '131129']]
--> ACIERTO
    Primer Lugar: 4
    Entre los tres primeros: 4
<Figure size 2000x400 with 6 Axes>



Consulta número:  5
   Imagen de consulta:  103285
   Mas cercanos:  [[1.3391658, '103285'], [1.4457603, '101151'], [1.4919882, '108917'], [1.5291246, '130776'], [1.5594068, '908'], [1.5660452, '130532'], [1.5762553, '136283'], [1.5849391, '108913'], [1.5887637, '135332'], [1.6058885, '1349']]
--> ACIERTO
    Primer Lugar: 5
    Entre los tres primeros: 5
<Figure size 2000x400 with 6 Axes>



Consulta número:  6
   Imagen de consulta:  103287
   Mas cercanos:  [[1.152241, '103287'], [1.8697165, '129683'], [1.9397562, '121204'], [1.991231, '109029'], [1.998429, '102859'], [2.0485156, '136157'], [2.0802186, '130839'], [2.0870676, '130059'], [2.0922637, '617'], [2.0980287, '101448']]
--> ACIERTO
    Primer Lugar: 6
    Entre los tres primeros: 6
<Figure size 2000x400 with 6 Axes>



Consulta número:  7
   Imagen de consulta:  108510
   Mas cercanos:  [[1.3310466, '108510'], [1.9559381, '102857'], [2.0734217, '130169'], [2.076806, '107325'], [2.154751, '108526'], [2.1601381, '128650'], [2.1682448, '102375'], [2.1736674, '135292'], [2.1782815, '104650'], [2.1853092, '105611']]
--> ACIERTO
    Primer Lugar: 7
    Entre los tres primeros: 7
<Figure size 2000x400 with 6 Axes>



Consulta número:  8
   Imagen de consulta:  1099
   Mas cercanos:  [[0.80277133, '1099'], [2.004188, '101448'], [2.1027267, '136012'], [2.1037836, '136246'], [2.16203, '129411'], [2.171062, '130767'], [2.182422, '114579'], [2.1856139, '1283'], [2.191043, '131250'], [2.1944778, '135217']]
--> ACIERTO
    Primer Lugar: 8
    Entre los tres primeros: 8
<Figure size 2000x400 with 6 Axes>



Consulta número:  9
   Imagen de consulta:  1100
   Mas cercanos:  [[0.48015258, '1100'], [1.8817147, '135430'], [1.9496537, '135454'], [1.9763577, '130499'], [1.9825565, '115946'], [1.9842418, '1239'], [1.9880035, '1145'], [2.034634, '102858'], [2.0480769, '1236'], [2.0558052, '121106']]
--> ACIERTO
    Primer Lugar: 9
    Entre los tres primeros: 9
<Figure size 2000x400 with 6 Axes>



Consulta número:  10
   Imagen de consulta:  1111
   Mas cercanos:  [[0.9882323, '1111'], [1.9923295, '135423'], [2.0019498, '135430'], [2.0494075, '101448'], [2.0640652, '108988'], [2.0977607, '130547'], [2.1127408, '1351'], [2.1128721, '107344'], [2.1260502, '102372'], [2.1312933, '1191']]
--> ACIERTO
    Primer Lugar: 10
    Entre los tres primeros: 10
<Figure size 2000x400 with 6 Axes>



Consulta número:  11
   Imagen de consulta:  1114
   Mas cercanos:  [[0.8705836, '1114'], [1.8875234, '1050'], [1.8919063, '120546'], [1.9827543, '1278'], [1.9913464, '1367'], [2.0079691, '130557'], [2.0087693, '108545'], [2.016442, '130377'], [2.0220432, '102487'], [2.038634, '135252']]
--> ACIERTO
    Primer Lugar: 11
    Entre los tres primeros: 11
<Figure size 2000x400 with 6 Axes>



Consulta número:  12
   Imagen de consulta:  1126
   Mas cercanos:  [[0.6491373, '1126'], [1.5782896, '131129'], [1.6281163, '109132'], [1.6545285, '136274'], [1.6621734, '129129'], [1.6833179, '105180'], [1.691683, '136302'], [1.6923211, '121188'], [1.696059, '136637'], [1.7092787, '130199']]
--> ACIERTO
    Primer Lugar: 12
    Entre los tres primeros: 12
<Figure size 2000x400 with 6 Axes>



Consulta número:  13
   Imagen de consulta:  1129
   Mas cercanos:  [[0.6413736, '1129'], [1.6520747, '136283'], [1.7408919, '130823'], [1.764219, '109035'], [1.7650446, '130595'], [1.7700664, '130549'], [1.7922885, '1306'], [1.8081784, '130524'], [1.816174, '134892'], [1.8223912, '135332']]
--> ACIERTO
    Primer Lugar: 13
    Entre los tres primeros: 13
<Figure size 2000x400 with 6 Axes>



Consulta número:  14
   Imagen de consulta:  1132
   Mas cercanos:  [[1.4284332, '1132'], [2.3535001, '135432'], [2.5016787, '1130'], [2.5053258, '136624'], [2.5150375, '1062'], [2.531306, '130369'], [2.5509536, '135308'], [2.5511038, '101448'], [2.5789278, '1083'], [2.5793164, '1151']]
--> ACIERTO
    Primer Lugar: 14
    Entre los tres primeros: 14
<Figure size 2000x400 with 6 Axes>



Consulta número:  15
   Imagen de consulta:  1135
   Mas cercanos:  [[0.60612524, '1135'], [1.8376462, '129417'], [1.8463907, '106811'], [1.858766, '129692'], [1.8658336, '102013'], [1.8999108, '121209'], [1.910658, '1150'], [1.9231609, '1304'], [1.9285477, '630'], [1.9421343, '136434']]
--> ACIERTO
    Primer Lugar: 15
    Entre los tres primeros: 15
<Figure size 2000x400 with 6 Axes>



Consulta número:  16
   Imagen de consulta:  1139
   Mas cercanos:  [[2.106203, '130551'], [2.2514672, '129429'], [2.2914686, '1139'], [2.3415554, '130546'], [2.3417175, '1196'], [2.373881, '100896'], [2.3817484, '130199'], [2.3852468, '135411'], [2.405612, '134886'], [2.4153457, '130763']]
--> ACIERTO
    Entre los tres primeros: 16
<Figure size 2000x400 with 6 Axes>



Consulta número:  17
   Imagen de consulta:  1150
   Mas cercanos:  [[0.74642175, '1150'], [1.4435836, '108942'], [1.4682087, '109286'], [1.5362346, '1000'], [1.537107, '114582'], [1.5687964, '136286'], [1.574524, '129692'], [1.5778091, '129885'], [1.5857012, '102491'], [1.5953021, '1336']]
--> ACIERTO
    Primer Lugar: 16
    Entre los tres primeros: 17
<Figure size 2000x400 with 6 Axes>



Consulta número:  18
   Imagen de consulta:  1167
   Mas cercanos:  [[1.1488905, '1167'], [1.9575453, '129400'], [2.00771, '130199'], [2.0692358, '105026'], [2.1069043, '130799'], [2.1108093, '1196'], [2.1219873, '129856'], [2.123718, '135347'], [2.1289752, '107325'], [2.1290617, '134886']]
--> ACIERTO
    Primer Lugar: 17
    Entre los tres primeros: 18
<Figure size 2000x400 with 6 Axes>



Consulta número:  19
   Imagen de consulta:  1173
   Mas cercanos:  [[1.4495302, '1173'], [1.5338427, '134886'], [1.5392505, '1280'], [1.5408394, '134892'], [1.5429214, '130791'], [1.5855088, '135426'], [1.586559, '136472'], [1.5995991, '105026'], [1.6053497, '1168'], [1.6065418, '135318']]
--> ACIERTO
    Primer Lugar: 18
    Entre los tres primeros: 19
<Figure size 2000x400 with 6 Axes>



Consulta número:  20
   Imagen de consulta:  1174
   Mas cercanos:  [[0.7667862, '1174'], [1.767643, '1166'], [1.8908263, '1371'], [1.9998735, '120946'], [2.0100293, '135430'], [2.0149252, '130509'], [2.026547, '135432'], [2.028242, '105030'], [2.0299282, '130342'], [2.0450368, '131120']]
--> ACIERTO
    Primer Lugar: 19
    Entre los tres primeros: 20
<Figure size 2000x400 with 6 Axes>



Consulta número:  21
   Imagen de consulta:  1178
   Mas cercanos:  [[1.0736905, '1178'], [1.276985, '134886'], [1.3292538, '102854'], [1.3656809, '129664'], [1.3763258, '105026'], [1.3922281, '1306'], [1.409906, '135257'], [1.4107076, '135469'], [1.4189922, '1280'], [1.4225286, '133677']]
--> ACIERTO
    Primer Lugar: 20
    Entre los tres primeros: 21
<Figure size 2000x400 with 6 Axes>



Consulta número:  22
   Imagen de consulta:  1179
   Mas cercanos:  [[2.1689613, '1306'], [2.1763754, '135332'], [2.2217865, '108913'], [2.245737, '130594'], [2.2598917, '102854'], [2.2693787, '107325'], [2.2785048, '102464'], [2.288847, '133671'], [2.2935798, '106881'], [2.29527, '130058']]
<Figure size 2000x400 with 6 Axes>



Consulta número:  23
   Imagen de consulta:  121206
   Mas cercanos:  [[1.8401976, '121206'], [1.9519324, '1306'], [2.0118802, '102854'], [2.0479074, '1724'], [2.0589619, '135401'], [2.063317, '108576'], [2.0643802, '109031'], [2.0718875, '108981'], [2.07471, '1140'], [2.080759, '135332']]
--> ACIERTO
    Primer Lugar: 21
    Entre los tres primeros: 22
<Figure size 2000x400 with 6 Axes>



Consulta número:  24
   Imagen de consulta:  1253
   Mas cercanos:  [[1.9147372, '1253'], [2.1572144, '135362'], [2.2071745, '121201'], [2.2071745, '907'], [2.230686, '136583'], [2.2920551, '130287'], [2.3451679, '130827'], [2.345615, '130768'], [2.3718774, '102857'], [2.3821409, '1923']]
--> ACIERTO
    Primer Lugar: 22
    Entre los tres primeros: 23
<Figure size 2000x400 with 6 Axes>



Consulta número:  25
   Imagen de consulta:  1664
   Mas cercanos:  [[1.542905, '1664'], [1.8425311, '1306'], [1.8739346, '130557'], [1.8822784, '134892'], [1.9050177, '136198'], [1.9077948, '136177'], [1.9110861, '1140'], [1.9340422, '102854'], [1.9369835, '109035'], [1.938322, '101242']]
--> ACIERTO
    Primer Lugar: 23
    Entre los tres primeros: 24
<Figure size 2000x400 with 6 Axes>



Consulta número:  26
   Imagen de consulta:  1668
   Mas cercanos:  [[0.8193996, '1668'], [2.0737138, '133178'], [2.315735, '109180'], [2.379062, '106028'], [2.4021564, '135432'], [2.4028225, '130499'], [2.404111, '136022'], [2.4101977, '1325'], [2.4140215, '108990'], [2.4153094, '129863']]
--> ACIERTO
    Primer Lugar: 24
    Entre los tres primeros: 25
<Figure size 2000x400 with 6 Axes>



Consulta número:  27
   Imagen de consulta:  1687
   Mas cercanos:  [[1.1982138, '1687'], [2.040866, '129831'], [2.0696907, '101687'], [2.1671684, '120936'], [2.1764112, '735'], [2.1774206, '130763'], [2.189812, '108990'], [2.2063699, '130365'], [2.2073658, '130199'], [2.2114515, '130836']]
--> ACIERTO
    Primer Lugar: 25
    Entre los tres primeros: 26
<Figure size 2000x400 with 6 Axes>



Consulta número:  28
   Imagen de consulta:  1707
   Mas cercanos:  [[1.0141264, '1707'], [1.7937663, '130547'], [1.8283187, '134892'], [1.8309155, '1306'], [1.8413646, '109035'], [1.8495096, '135430'], [1.8533211, '108913'], [1.8590848, '130212'], [1.859151, '105026'], [1.859747, '107325']]
--> ACIERTO
    Primer Lugar: 26
    Entre los tres primeros: 27
<Figure size 2000x400 with 6 Axes>



Consulta número:  29
   Imagen de consulta:  1710
   Mas cercanos:  [[1.347916, '1710'], [1.7293022, '1222'], [1.7360128, '129414'], [1.7801917, '107325'], [1.7895279, '136437'], [1.7994037, '107289'], [1.807135, '129194'], [1.8227761, '136434'], [1.8268491, '129419'], [1.8325543, '136158']]
--> ACIERTO
    Primer Lugar: 27
    Entre los tres primeros: 28
<Figure size 2000x400 with 6 Axes>



Consulta número:  30
   Imagen de consulta:  1714
   Mas cercanos:  [[1.4990286, '1714'], [2.3260407, '130377'], [2.3806589, '1026'], [2.393798, '109285'], [2.3990564, '108576'], [2.4182334, '1283'], [2.4244413, '120779'], [2.429401, '130382'], [2.4359612, '108545'], [2.4370928, '107222']]
--> ACIERTO
    Primer Lugar: 28
    Entre los tres primeros: 29
<Figure size 2000x400 with 6 Axes>



Consulta número:  31
   Imagen de consulta:  1717
   Mas cercanos:  [[1.8782587, '1717'], [1.9351238, '135430'], [1.9480035, '1371'], [2.034181, '120946'], [2.056471, '1124'], [2.0683815, '108911'], [2.1028278, '105181'], [2.1050928, '1165'], [2.1195545, '106809'], [2.1198428, '1265']]
--> ACIERTO
    Primer Lugar: 29
    Entre los tres primeros: 30
<Figure size 2000x400 with 6 Axes>



Consulta número:  32
   Imagen de consulta:  1724
   Mas cercanos:  [[1.3067707, '1724'], [1.8645751, '102854'], [1.8862565, '136463'], [1.905922, '136198'], [1.913484, '136588'], [1.928063, '1306'], [1.939227, '109035'], [1.9612406, '134892'], [1.9691373, '130871'], [1.9800898, '106881']]
--> ACIERTO
    Primer Lugar: 30
    Entre los tres primeros: 31
<Figure size 2000x400 with 6 Axes>



Consulta número:  33
   Imagen de consulta:  1733
   Mas cercanos:  [[0.68399805, '1733'], [1.7215216, '135332'], [1.7847418, '1306'], [1.8211933, '108913'], [1.8232535, '102854'], [1.8468741, '136283'], [1.8561213, '136463'], [1.8629595, '130776'], [1.8887036, '108981'], [1.8892726, '130270']]
--> ACIERTO
    Primer Lugar: 31
    Entre los tres primeros: 32
<Figure size 2000x400 with 6 Axes>



Consulta número:  34
   Imagen de consulta:  1737
   Mas cercanos:  [[1.0527227, '1737'], [2.2339003, '102862'], [2.244024, '107325'], [2.2476125, '105026'], [2.2581165, '130594'], [2.2727304, '102854'], [2.2757592, '133671'], [2.2866066, '135316'], [2.295413, '134886'], [2.3103766, '1243']]
--> ACIERTO
    Primer Lugar: 32
    Entre los tres primeros: 33
<Figure size 2000x400 with 6 Axes>



Consulta número:  35
   Imagen de consulta:  1742
   Mas cercanos:  [[0.98474145, '1742'], [1.9644951, '105026'], [2.0199318, '134886'], [2.0281067, '102854'], [2.0569983, '130199'], [2.06637, '134892'], [2.0821934, '107325'], [2.0855527, '135318'], [2.0860999, '1222'], [2.0930758, '108923']]
--> ACIERTO
    Primer Lugar: 33
    Entre los tres primeros: 34
<Figure size 2000x400 with 6 Axes>



Consulta número:  36
   Imagen de consulta:  1749
   Mas cercanos:  [[1.7509031, '1749'], [1.9867336, '101606'], [2.055196, '1371'], [2.0659406, '130342'], [2.0835423, '135433'], [2.0949354, '135430'], [2.1069028, '135473'], [2.1170862, '1236'], [2.1197097, '1090'], [2.123294, '121106']]
--> ACIERTO
    Primer Lugar: 34
    Entre los tres primeros: 35
<Figure size 2000x400 with 6 Axes>



Consulta número:  37
   Imagen de consulta:  1775
   Mas cercanos:  [[1.1218622, '1775'], [1.9708302, '102858'], [1.9769716, '104868'], [1.9953053, '134886'], [2.0054677, '135318'], [2.0068078, '1306'], [2.0143297, '107325'], [2.024403, '1280'], [2.0272841, '105026'], [2.033923, '135430']]
--> ACIERTO
    Primer Lugar: 35
    Entre los tres primeros: 36
<Figure size 2000x400 with 6 Axes>



Consulta número:  38
   Imagen de consulta:  1777
   Mas cercanos:  [[0.7887545, '1777'], [1.9097204, '1316'], [1.9330207, '1191'], [1.942749, '1255'], [1.9736778, '134886'], [1.9955418, '130799'], [1.9958392, '1351'], [1.9983515, '105026'], [2.007728, '129199'], [2.0131433, '107325']]
--> ACIERTO
    Primer Lugar: 36
    Entre los tres primeros: 37
<Figure size 2000x400 with 6 Axes>



Consulta número:  39
   Imagen de consulta:  1885
   Mas cercanos:  [[0.69223285, '1885'], [1.5905257, '102464'], [1.6215695, '1306'], [1.6325065, '130270'], [1.6374447, '108981'], [1.6572542, '130557'], [1.6653631, '136463'], [1.6675947, '1238'], [1.6697645, '134886'], [1.6885821, '135469']]
--> ACIERTO
    Primer Lugar: 37
    Entre los tres primeros: 38
<Figure size 2000x400 with 6 Axes>



Consulta número:  40
   Imagen de consulta:  1916
   Mas cercanos:  [[0.760703, '1916'], [2.0655067, '101448'], [2.1273446, '1165'], [2.1411562, '102859'], [2.1972294, '136157'], [2.20856, '135432'], [2.2104175, '102861'], [2.2306983, '1335'], [2.2312458, '121106'], [2.2473066, '1022']]
--> ACIERTO
    Primer Lugar: 38
    Entre los tres primeros: 39
<Figure size 2000x400 with 6 Axes>



Consulta número:  41
   Imagen de consulta:  1919
   Mas cercanos:  [[1.4362655, '1919'], [1.8047535, '135331'], [1.9136759, '129667'], [1.9217389, '131266'], [1.9257793, '106814'], [1.9261057, '109029'], [1.9281904, '130199'], [1.9351901, '129664'], [1.937726, '135411'], [1.9417989, '136007']]
--> ACIERTO
    Primer Lugar: 39
    Entre los tres primeros: 40
<Figure size 2000x400 with 6 Axes>



Consulta número:  42
   Imagen de consulta:  1922
   Mas cercanos:  [[1.6655909, '1922'], [2.2378955, '109301'], [2.4363992, '107334'], [2.5168974, '106812'], [2.5280514, '1196'], [2.5331523, '130378'], [2.5376227, '1033'], [2.5452263, '108911'], [2.5563233, '130816'], [2.5580032, '135432']]
--> ACIERTO
    Primer Lugar: 40
    Entre los tres primeros: 41
<Figure size 2000x400 with 6 Axes>



Consulta número:  43
   Imagen de consulta:  1923
   Mas cercanos:  [[1.8708854, '1243'], [1.9877093, '130594'], [1.9943534, '1051'], [2.0204196, '134903'], [2.048696, '135382'], [2.0493808, '1306'], [2.052005, '130522'], [2.0575943, '102854'], [2.060717, '135481'], [2.068313, '107325']]
<Figure size 2000x400 with 6 Axes>



Consulta número:  44
   Imagen de consulta:  1924
   Mas cercanos:  [[0.81034166, '1924'], [1.9268545, '107325'], [1.9295566, '129853'], [1.9658206, '129685'], [1.9847517, '1196'], [1.9894717, '129701'], [2.002635, '130297'], [2.018722, '109108'], [2.0190797, '108913'], [2.0241342, '129401']]
--> ACIERTO
    Primer Lugar: 41
    Entre los tres primeros: 42
<Figure size 2000x400 with 6 Axes>



Consulta número:  45
   Imagen de consulta:  1926
   Mas cercanos:  [[1.2095982, '1926'], [1.8056426, '136007'], [1.8990936, '133677'], [1.9086411, '135332'], [1.9150292, '134886'], [1.9197606, '129664'], [1.9256511, '102854'], [1.9262404, '129697'], [1.9308908, '135449'], [1.9449145, '135331']]
--> ACIERTO
    Primer Lugar: 42
    Entre los tres primeros: 43
<Figure size 2000x400 with 6 Axes>



Consulta número:  46
   Imagen de consulta:  1992
   Mas cercanos:  [[0.9566808, '1992'], [1.7467555, '1371'], [1.7986486, '135430'], [1.8625697, '1124'], [1.8978205, '1165'], [1.9460084, '108911'], [1.9468831, '120946'], [1.9479312, '127969'], [1.9596837, '130342'], [1.9612274, '105181']]
--> ACIERTO
    Primer Lugar: 43
    Entre los tres primeros: 44
<Figure size 2000x400 with 6 Axes>



Consulta número:  47
   Imagen de consulta:  2097
   Mas cercanos:  [[1.4927146, '2097'], [2.3370042, '130522'], [2.4155164, '102365'], [2.4340363, '109005'], [2.4608839, '116144'], [2.5009668, '102010'], [2.5671039, '129391'], [2.6206205, '1243'], [2.6391685, '131255'], [2.639957, '1091']]
--> ACIERTO
    Primer Lugar: 44
    Entre los tres primeros: 45
<Figure size 2000x400 with 6 Axes>



Consulta número:  48
   Imagen de consulta:  2238
   Mas cercanos:  [[0.8462504, '2238'], [2.0714636, '136203'], [2.0906992, '136281'], [2.1024134, '130505'], [2.105578, '102854'], [2.1131904, '1328'], [2.1231825, '130594'], [2.127523, '105026'], [2.1442854, '108923'], [2.1461725, '1102']]
--> ACIERTO
    Primer Lugar: 45
    Entre los tres primeros: 46
<Figure size 2000x400 with 6 Axes>



Consulta número:  49
   Imagen de consulta:  602
   Mas cercanos:  [[0.9115925, '602'], [2.0050871, '134919'], [2.0053694, '1165'], [2.0135245, '135494'], [2.0300126, '100893'], [2.057287, '107236'], [2.0863974, '129893'], [2.0878968, '1157'], [2.090157, '1371'], [2.1323872, '135430']]
--> ACIERTO
    Primer Lugar: 46
    Entre los tres primeros: 47
<Figure size 2000x400 with 6 Axes>



Consulta número:  50
   Imagen de consulta:  603
   Mas cercanos:  [[1.8240482, '130199'], [1.9138631, '1196'], [1.9139384, '603'], [1.9623916, '129429'], [1.9973089, '107325'], [2.0084498, '131266'], [2.0433767, '135391'], [2.0545862, '129699'], [2.0616765, '105026'], [2.0632274, '135336']]
--> ACIERTO
    Entre los tres primeros: 48
<Figure size 2000x400 with 6 Axes>



Consulta número:  51
   Imagen de consulta:  617
   Mas cercanos:  [[1.1425886, '617'], [1.615447, '101448'], [1.6468694, '136157'], [1.6484361, '121204'], [1.6539621, '136430'], [1.721886, '130805'], [1.7481476, '102859'], [1.7559887, '129174'], [1.7657392, '121216'], [1.7657763, '101789']]
--> ACIERTO
    Primer Lugar: 47
    Entre los tres primeros: 49
<Figure size 2000x400 with 6 Axes>



Consulta número:  52
   Imagen de consulta:  622
   Mas cercanos:  [[1.8678324, '622'], [2.3382132, '107325'], [2.3766131, '130594'], [2.3848279, '101017'], [2.387137, '130199'], [2.3875427, '102472'], [2.3899202, '135391'], [2.3950753, '102854'], [2.3991334, '1306'], [2.4086037, '133671']]
--> ACIERTO
    Primer Lugar: 48
    Entre los tres primeros: 50
<Figure size 2000x400 with 6 Axes>



Consulta número:  53
   Imagen de consulta:  630
   Mas cercanos:  [[0.96294695, '630'], [1.5951791, '134887'], [1.6304737, '109120'], [1.6706386, '136434'], [1.7170953, '106816'], [1.7356615, '129193'], [1.7412926, '136185'], [1.7414703, '108942'], [1.7595828, '136276'], [1.7652011, '102013']]
--> ACIERTO
    Primer Lugar: 49
    Entre los tres primeros: 51
<Figure size 2000x400 with 6 Axes>



Consulta número:  54
   Imagen de consulta:  642
   Mas cercanos:  [[0.84876513, '642'], [1.5688478, '102854'], [1.5870439, '1306'], [1.618518, '102464'], [1.6512533, '1139'], [1.6699092, '136588'], [1.6709182, '1326'], [1.6737113, '106816'], [1.6746384, '133677'], [1.6747041, '126416']]
--> ACIERTO
    Primer Lugar: 50
    Entre los tres primeros: 52
<Figure size 2000x400 with 6 Axes>



Consulta número:  55
   Imagen de consulta:  676
   Mas cercanos:  [[1.27851, '676'], [1.8426132, '134892'], [1.8518435, '102854'], [1.8804133, '129664'], [1.8841163, '1310'], [1.9029034, '130594'], [1.9067061, '1306'], [1.9082006, '108905'], [1.9151152, '134886'], [1.9210174, '135331']]
--> ACIERTO
    Primer Lugar: 51
    Entre los tres primeros: 53
<Figure size 2000x400 with 6 Axes>



Consulta número:  56
   Imagen de consulta:  679
   Mas cercanos:  [[1.1970438, '679'], [1.3678446, '102854'], [1.426918, '129832'], [1.4442134, '135401'], [1.4777263, '129420'], [1.4905854, '136177'], [1.5038221, '136199'], [1.516875, '105026'], [1.5274266, '105898'], [1.5283748, '129664']]
--> ACIERTO
    Primer Lugar: 52
    Entre los tres primeros: 54
<Figure size 2000x400 with 6 Axes>



Consulta número:  57
   Imagen de consulta:  680
   Mas cercanos:  [[0.9445062, '680'], [1.9688714, '101448'], [2.0473287, '102859'], [2.1039426, '121204'], [2.1072185, '136430'], [2.1227527, '107320'], [2.1249096, '136157'], [2.1449535, '617'], [2.1480885, '102861'], [2.1700304, '130805']]
--> ACIERTO
    Primer Lugar: 53
    Entre los tres primeros: 55
<Figure size 2000x400 with 6 Axes>



Consulta número:  58
   Imagen de consulta:  691
   Mas cercanos:  [[0.8243718, '691'], [2.3550997, '130793'], [2.360313, '129401'], [2.3918371, '113522'], [2.4057448, '131270'], [2.4343712, '136290'], [2.450375, '135406'], [2.4669454, '102857'], [2.5259073, '130816'], [2.558012, '128650']]
--> ACIERTO
    Primer Lugar: 54
    Entre los tres primeros: 56
<Figure size 2000x400 with 6 Axes>



Consulta número:  59
   Imagen de consulta:  713
   Mas cercanos:  [[0.16588326, '713'], [1.9213011, '1280'], [1.9501799, '106814'], [1.9621265, '102854'], [1.994251, '105026'], [1.9970211, '129664'], [2.0108955, '134886'], [2.011228, '129697'], [2.0247493, '1039'], [2.0292888, '101242']]
--> ACIERTO
    Primer Lugar: 55
    Entre los tres primeros: 57
<Figure size 2000x400 with 6 Axes>



Consulta número:  60
   Imagen de consulta:  728
   Mas cercanos:  [[1.9303681, '102854'], [1.9580845, '136433'], [1.967478, '106814'], [1.9936428, '135327'], [1.9978639, '134886'], [2.0059915, '106816'], [2.0065117, '134892'], [2.007142, '136293'], [2.0100546, '130177'], [2.0123384, '105026']]
<Figure size 2000x400 with 6 Axes>



Consulta número:  61
   Imagen de consulta:  732
   Mas cercanos:  [[2.3248475, '105181'], [2.3378935, '134886'], [2.3509033, '1367'], [2.351227, '1109'], [2.3553934, '130811'], [2.3567555, '732'], [2.3583932, '107357'], [2.3585014, '109042'], [2.3702805, '1349'], [2.374113, '130763']]
--> ACIERTO
<Figure size 2000x400 with 6 Axes>



Consulta número:  62
   Imagen de consulta:  735
   Mas cercanos:  [[0.79173076, '735'], [1.5447398, '108990'], [1.6721762, '135430'], [1.7130822, '1124'], [1.7694877, '1371'], [1.7927953, '1166'], [1.7977053, '1165'], [1.8033477, '135432'], [1.8379439, '130509'], [1.853108, '1189']]
--> ACIERTO
    Primer Lugar: 56
    Entre los tres primeros: 58
<Figure size 2000x400 with 6 Axes>



Consulta número:  63
   Imagen de consulta:  78
   Mas cercanos:  [[0.9021489, '78'], [1.7856393, '105013'], [1.8961657, '136177'], [1.9046044, '134892'], [1.9157133, '135469'], [1.9202932, '1064'], [1.9307694, '135304'], [1.9372286, '109192'], [1.9429111, '133677'], [1.9512974, '102854']]
--> ACIERTO
    Primer Lugar: 57
    Entre los tres primeros: 59
<Figure size 2000x400 with 6 Axes>



Consulta número:  64
   Imagen de consulta:  82
   Mas cercanos:  [[1.6399893, '82'], [2.2729409, '129735'], [2.3752067, '105611'], [2.411026, '129201'], [2.4189558, '108995'], [2.4275336, '109133'], [2.429684, '130348'], [2.4397097, '1141'], [2.4458296, '136160'], [2.472629, '131120']]
--> ACIERTO
    Primer Lugar: 58
    Entre los tres primeros: 60
<Figure size 2000x400 with 6 Axes>



Consulta número:  65
   Imagen de consulta:  858
   Mas cercanos:  [[0.8285047, '858'], [2.1283674, '130522'], [2.1467988, '106814'], [2.1658971, '135257'], [2.175201, '109029'], [2.1798613, '136471'], [2.1902456, '1028'], [2.1906435, '1064'], [2.1982026, '136465'], [2.1999826, '100939']]
--> ACIERTO
    Primer Lugar: 59
    Entre los tres primeros: 61
<Figure size 2000x400 with 6 Axes>



Consulta número:  66
   Imagen de consulta:  862
   Mas cercanos:  [[0.635864, '862'], [1.6215347, '130799'], [1.7029775, '135430'], [1.7054349, '105026'], [1.7284343, '134892'], [1.7367119, '129414'], [1.7407922, '135333'], [1.7525953, '1280'], [1.759027, '1351'], [1.7593346, '130349']]
--> ACIERTO
    Primer Lugar: 60
    Entre los tres primeros: 62
<Figure size 2000x400 with 6 Axes>



Consulta número:  67
   Imagen de consulta:  865
   Mas cercanos:  [[1.6740675, '865'], [2.36319, '129883'], [2.377594, '1919'], [2.4409113, '130509'], [2.4482176, '130199'], [2.4549205, '129699'], [2.4639814, '136476'], [2.4745426, '130260'], [2.4749956, '129831'], [2.4757688, '108919']]
--> ACIERTO
    Primer Lugar: 61
    Entre los tres primeros: 63
<Figure size 2000x400 with 6 Axes>



Consulta número:  68
   Imagen de consulta:  868
   Mas cercanos:  [[0.63771933, '868'], [1.5774463, '1150'], [1.5817788, '114582'], [1.5866226, '107366'], [1.5921043, '129419'], [1.6069561, '107341'], [1.6305665, '133676'], [1.6530653, '129178'], [1.6585736, '130855'], [1.6586823, '136437']]
--> ACIERTO
    Primer Lugar: 62
    Entre los tres primeros: 64
<Figure size 2000x400 with 6 Axes>



Consulta número:  69
   Imagen de consulta:  87
   Mas cercanos:  [[1.0191731, '87'], [1.6542125, '1306'], [1.6702396, '134886'], [1.6707615, '102854'], [1.6849147, '102464'], [1.6923397, '135252'], [1.7086987, '133677'], [1.7122041, '1336'], [1.7191544, '136463'], [1.7201099, '134892']]
--> ACIERTO
    Primer Lugar: 63
    Entre los tres primeros: 65
<Figure size 2000x400 with 6 Axes>



Consulta número:  70
   Imagen de consulta:  89
   Mas cercanos:  [[1.312227, '89'], [2.1985834, '107222'], [2.23607, '1371'], [2.2666428, '120779'], [2.269759, '1149'], [2.2754369, '135433'], [2.3001218, '129700'], [2.3026266, '135430'], [2.3180156, '115946'], [2.3190577, '121106']]
--> ACIERTO
    Primer Lugar: 64
    Entre los tres primeros: 66
<Figure size 2000x400 with 6 Axes>



Consulta número:  71
   Imagen de consulta:  907
   Mas cercanos:  [[0.89033026, '121201'], [0.89033026, '907'], [2.206112, '130357'], [2.214966, '130287'], [2.2193859, '102857'], [2.2386537, '135362'], [2.248505, '130827'], [2.2544217, '130768'], [2.2616541, '136583'], [2.2892141, '129209']]
--> ACIERTO
    Entre los tres primeros: 67
<Figure size 2000x400 with 6 Axes>



Consulta número:  72
   Imagen de consulta:  908
   Mas cercanos:  [[1.3365369, '908'], [1.684904, '1129'], [1.6857213, '101151'], [1.7510766, '108917'], [1.7709103, '107311'], [1.8272814, '130212'], [1.8393844, '136463'], [1.840034, '1349'], [1.840316, '1297'], [1.8559617, '106849']]
--> ACIERTO
    Primer Lugar: 65
    Entre los tres primeros: 68
<Figure size 2000x400 with 6 Axes>



Consulta número:  73
   Imagen de consulta:  918
   Mas cercanos:  [[1.3430991, '918'], [2.116203, '1331'], [2.1514971, '121204'], [2.1854203, '121170'], [2.1900733, '1236'], [2.2596333, '101448'], [2.2693074, '1086'], [2.290687, '135407'], [2.2960594, '102861'], [2.2998886, '134896']]
--> ACIERTO
    Primer Lugar: 66
    Entre los tres primeros: 69
<Figure size 2000x400 with 6 Axes>



Consulta número:  74
   Imagen de consulta:  920
   Mas cercanos:  [[1.349875, '920'], [1.7595872, '130515'], [1.7622718, '1366'], [1.770122, '1308'], [1.7848767, '134886'], [1.7893155, '102170'], [1.7893593, '129173'], [1.7996436, '106816'], [1.8079444, '129653'], [1.8187957, '136434']]
--> ACIERTO
    Primer Lugar: 67
    Entre los tres primeros: 70
<Figure size 2000x400 with 6 Axes>



Consulta número:  75
   Imagen de consulta:  921
   Mas cercanos:  [[1.1305943, '921'], [1.9940606, '102859'], [2.097588, '101448'], [2.1044154, '129683'], [2.2049744, '1225'], [2.260411, '121204'], [2.2711437, '1273'], [2.282275, '107321'], [2.3184664, '918'], [2.3315945, '129182']]
--> ACIERTO
    Primer Lugar: 68
    Entre los tres primeros: 71
<Figure size 2000x400 with 6 Axes>



Consulta número:  76
   Imagen de consulta:  93
   Mas cercanos:  [[1.6190615, '93'], [1.9988962, '130815'], [1.9993023, '130199'], [2.0230057, '102464'], [2.0300753, '133676'], [2.0633705, '101058'], [2.0956109, '130058'], [2.1013575, '1238'], [2.1295693, '135434'], [2.1331978, '130855']]
--> ACIERTO
    Primer Lugar: 69
    Entre los tres primeros: 72
<Figure size 2000x400 with 6 Axes>



Consulta número:  77
   Imagen de consulta:  944
   Mas cercanos:  [[1.3279381, '944'], [1.779758, '129414'], [1.7812438, '134886'], [1.7935611, '129888'], [1.8077606, '1039'], [1.8088697, '102854'], [1.8165084, '136271'], [1.8233529, '134892'], [1.8237436, '105026'], [1.8262876, '129664']]
--> ACIERTO
    Primer Lugar: 70
    Entre los tres primeros: 73
<Figure size 2000x400 with 6 Axes>



Consulta número:  78
   Imagen de consulta:  96
   Mas cercanos:  [[0.77517384, '96'], [1.5566792, '120946'], [1.6332183, '135430'], [1.6811324, '1124'], [1.7275399, '108911'], [1.7781105, '130368'], [1.7961645, '131190'], [1.8001732, '1165'], [1.8156474, '121106'], [1.8415664, '108990']]
--> ACIERTO
    Primer Lugar: 71
    Entre los tres primeros: 74
<Figure size 2000x400 with 6 Axes>



Consulta número:  79
   Imagen de consulta:  961
   Mas cercanos:  [[0.9070459, '961'], [1.7112437, '130058'], [1.778848, '103283'], [1.8151768, '130367'], [1.8162993, '105898'], [1.8195174, '107325'], [1.8281177, '136199'], [1.8306444, '130297'], [1.8502533, '134898'], [1.8560432, '102854']]
--> ACIERTO
    Primer Lugar: 72
    Entre los tres primeros: 75
<Figure size 2000x400 with 6 Axes>



Porcentaje de aciertos NN1:  0.9
Porcentaje de aciertos NN3:  0.9375
Porcentaje de aciertos NN5:  0.95
  Cantidad consultas:  80
```


**[Celda 21 - Código]**
```python
##################### PROMEDIO DE DOS CAPAS ###########################

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from PIL import Image

#### DICCIONARIO DE CONSULTA Y RESULTADOS PARA MOSTRAR LAS IMAGENES
searches =  {"input_image": np.random.rand(224, 224, 3), "result_images": [np.random.rand(224, 224, 3) for _ in range(5)] } 



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
### Resnet
# Inicialización de variables
vectores_elemento = []
vectores_elemento2 = []
tamanoLote = 50
desde = 0
hasta = tamanoLote

# Procesamiento en lotes
while hasta < len(imagenesBD):
    lote = imagenesBD[desde:hasta]
    vects = new_model_resnet.predict(lote)
    vects2 = new_model_resnet2.predict(lote)
    vectores_elemento.extend(vects)
    vectores_elemento2.extend(vects2)
    desde = hasta
    hasta += tamanoLote

# Procesamiento del lote final
lote = imagenesBD[desde:]
vects = new_model_resnet.predict(lote)
vectores_elemento.extend(vects)
vects2 = new_model_resnet2.predict(lote)
vectores_elemento2.extend(vects2)

vectores_consulta = new_model_resnet.predict(consultas)
vectores_consulta2 = new_model_resnet2.predict(consultas)

print(vectores_elemento[0].shape)


#Inicio del for consulta
no_encontrados = []
i = 0
for cons, cons2,imgQ in zip(vectores_consulta,vectores_consulta2,consultas):
    a = np.array(cons)
    c = np.array(cons2)
    # print(np.sum(a))
    min = 2000000
    nnklist = []
    # Inicio del for DB
    j = 0
    for elem, elem2 in zip(vectores_elemento, vectores_elemento2):
        b = np.array(elem)
        d = np.array(elem2)
        # print(np.sum(b))
        dist = np.sqrt(np.sum(np.square(a-b)))
        dist2 = np.sqrt(np.sum(np.square(c-d)))
        dist = (dist+dist2/2)/2
        # dist = np.sum(np.abs(a-b))
        # print('distancia: ',dist)
        nnk(nnklist,[dist,j],10)
        j += 1

    searches["input_image"] = imgQ
    searches["result_images"] = []
    print("Consulta número: ", i)
    print("   Imagen de consulta: ", y_consultas[i])
    print('   Mas cercanos: ', nnk2etiquetas(nnklist, y_imagenesBD))
    encontrado = False
    ps = 1
    for par in nnklist:
        searches["result_images"].append(imagenesBD[par[1]])
        if y_imagenesBD[par[1]] == y_consultas[i]:
            encontrado = True
            contador_aciertos += 1
            print('--> ACIERTO')
            if ps == 1:
                contador_aciertosNN1 += 1
                print(f'    Primer Lugar: {contador_aciertosNN1}')
            if ps <= 3:
                contador_aciertosNN3 += 1
                print(f'    Entre los tres primeros: {contador_aciertosNN3}')
        ps += 1
    show_search_images(searches, i)
    print("\n")
    print("\n")
    if not encontrado:
        no_encontrados.append(y_consultas[i])
    i += 1

print("Porcentaje de aciertos NN1: ", contador_aciertosNN1/cantidad_consultas)
print("Porcentaje de aciertos NN3: ", contador_aciertosNN3/cantidad_consultas)
print("Porcentaje de aciertos NN5: ", contador_aciertos/cantidad_consultas)
print('  Cantidad consultas: ', cantidad_consultas)
```


*Salida:*
```text
<class 'numpy.ndarray'>
(56, 56, 64)
Consulta número:  0
   Imagen de consulta:  1
   Mas cercanos:  [[7.652832984924316, '1'], [9.887059450149536, '82'], [10.220504999160767, '129201'], [10.362266778945923, '129735'], [10.499680399894714, '1306'], [10.504860877990723, '135434'], [10.609899282455444, '131270'], [10.674402236938477, '102464'], [10.675502061843872, '1116'], [10.736870050430298, '136274']]
--> ACIERTO
    Primer Lugar: 1
    Entre los tres primeros: 1
<Figure size 2000x400 with 6 Axes>



Consulta número:  1
   Imagen de consulta:  102462
   Mas cercanos:  [[15.387127876281738, '130342'], [15.558541893959045, '1371'], [16.415911436080933, '109122'], [16.436689019203186, '1749'], [16.543155074119568, '120946'], [16.69887089729309, '1050'], [16.70053768157959, '136637'], [16.716846585273743, '117199'], [16.72969251871109, '102462'], [16.7314555644989, '109305']]
--> ACIERTO
<Figure size 2000x400 with 6 Axes>



Consulta número:  2
   Imagen de consulta:  102474
   Mas cercanos:  [[7.001036465167999, '102474'], [9.835492491722107, '1233'], [10.525997519493103, '135527'], [10.62369978427887, '130199'], [10.630661845207214, '130594'], [10.651952743530273, '102857'], [10.713443338871002, '131249'], [10.792846262454987, '133671'], [10.810606956481934, '1306'], [10.884988188743591, '136272']]
--> ACIERTO
    Primer Lugar: 2
    Entre los tres primeros: 2
<Figure size 2000x400 with 6 Axes>



Consulta número:  3
   Imagen de consulta:  102476
   Mas cercanos:  [[14.129337906837463, '102857'], [14.37866222858429, '129570'], [14.38511061668396, '106814'], [14.39793336391449, '133671'], [14.446570038795471, '105026'], [14.453819870948792, '1117'], [14.456788897514343, '1233'], [14.456934928894043, '130594'], [14.484498858451843, '135467'], [14.49096155166626, '133678']]
<Figure size 2000x400 with 6 Axes>



Consulta número:  4
   Imagen de consulta:  102852
   Mas cercanos:  [[6.415488213300705, '102852'], [14.115700244903564, '136160'], [14.212758183479309, '1124'], [14.226674914360046, '130199'], [14.390107035636902, '108990'], [14.423141837120056, '1306'], [14.437923192977905, '136272'], [14.453348398208618, '136289'], [14.495728611946106, '102464'], [14.525388479232788, '135430']]
--> ACIERTO
    Primer Lugar: 3
    Entre los tres primeros: 3
<Figure size 2000x400 with 6 Axes>



Consulta número:  5
   Imagen de consulta:  103285
   Mas cercanos:  [[14.030829966068268, '103285'], [14.987931966781616, '101151'], [15.282701253890991, '130824'], [16.475097358226776, '130532'], [16.52119356393814, '130176'], [16.738742172718048, '1297'], [16.744086384773254, '135330'], [16.821231245994568, '135469'], [16.85574322938919, '129685'], [16.857598304748535, '135391']]
--> ACIERTO
    Primer Lugar: 4
    Entre los tres primeros: 4
<Figure size 2000x400 with 6 Axes>



Consulta número:  6
   Imagen de consulta:  103287
   Mas cercanos:  [[8.476698517799377, '103287'], [14.2522252202034, '102859'], [14.315253734588623, '123005'], [14.992376267910004, '129683'], [15.185822486877441, '1338'], [15.190554738044739, '135409'], [15.238491177558899, '107289'], [15.316598892211914, '100939'], [15.32009768486023, '136286'], [15.333447217941284, '130840']]
--> ACIERTO
    Primer Lugar: 5
    Entre los tres primeros: 5
<Figure size 2000x400 with 6 Axes>



Consulta número:  7
   Imagen de consulta:  108510
   Mas cercanos:  [[11.327596426010132, '108510'], [15.760328531265259, '136274'], [15.78248119354248, '1108'], [15.845080137252808, '102375'], [15.871862769126892, '1280'], [15.927699446678162, '136286'], [15.933233380317688, '105026'], [15.959949135780334, '136182'], [16.018259286880493, '130172'], [16.037172317504883, '130502']]
--> ACIERTO
    Primer Lugar: 6
    Entre los tres primeros: 6
<Figure size 2000x400 with 6 Axes>



Consulta número:  8
   Imagen de consulta:  1099
   Mas cercanos:  [[6.3248432874679565, '1099'], [19.329706072807312, '109212'], [19.456292748451233, '1304'], [19.603004693984985, '129182'], [19.724098682403564, '130366'], [19.76587975025177, '679'], [20.061301469802856, '116334'], [20.08196258544922, '130042'], [20.12458562850952, '131250'], [20.155791997909546, '1173']]
--> ACIERTO
    Primer Lugar: 7
    Entre los tres primeros: 7
<Figure size 2000x400 with 6 Axes>



Consulta número:  9
   Imagen de consulta:  1100
   Mas cercanos:  [[3.3269834369421005, '1100'], [11.068107604980469, '1157'], [11.325958490371704, '102858'], [11.62754738330841, '130289'], [11.648979663848877, '1124'], [11.667147099971771, '135430'], [11.787523448467255, '1239'], [11.85387897491455, '130368'], [12.04778516292572, '117199'], [12.093070268630981, '120946']]
--> ACIERTO
    Primer Lugar: 8
    Entre los tres primeros: 8
<Figure size 2000x400 with 6 Axes>



Consulta número:  10
   Imagen de consulta:  1111
   Mas cercanos:  [[8.498166412115097, '1111'], [18.153605222702026, '1117'], [18.347848176956177, '130172'], [18.371766448020935, '123004'], [18.392446637153625, '1317'], [18.430386662483215, '129206'], [18.463334321975708, '1084'], [18.52127516269684, '130171'], [18.59287703037262, '108988'], [18.60783541202545, '1280']]
--> ACIERTO
    Primer Lugar: 9
    Entre los tres primeros: 9
<Figure size 2000x400 with 6 Axes>



Consulta número:  11
   Imagen de consulta:  1114
   Mas cercanos:  [[11.693656474351883, '1114'], [22.48325514793396, '120546'], [22.486661553382874, '108545'], [23.165088415145874, '130866'], [23.281631767749786, '1278'], [23.798093795776367, '130821'], [23.812857389450073, '107222'], [23.88706910610199, '129132'], [23.919891119003296, '1334'], [23.938875794410706, '109132']]
--> ACIERTO
    Primer Lugar: 10
    Entre los tres primeros: 10
<Figure size 2000x400 with 6 Axes>



Consulta número:  12
   Imagen de consulta:  1126
   Mas cercanos:  [[5.259649187326431, '1126'], [11.751718997955322, '129401'], [11.90172952413559, '1124'], [11.917831003665924, '1161'], [12.21701717376709, '1306'], [12.285913705825806, '129201'], [12.320175170898438, '82'], [12.323369264602661, '131270'], [12.328907132148743, '129870'], [12.372989356517792, '102464']]
--> ACIERTO
    Primer Lugar: 11
    Entre los tres primeros: 11
<Figure size 2000x400 with 6 Axes>



Consulta número:  13
   Imagen de consulta:  1129
   Mas cercanos:  [[6.056091278791428, '1129'], [16.884041368961334, '130532'], [18.219076335430145, '107311'], [18.334690630435944, '130549'], [18.482073962688446, '908'], [18.499030113220215, '136235'], [18.571483314037323, '130595'], [18.652622997760773, '1159'], [18.79156368970871, '1297'], [18.802271127700806, '130524']]
--> ACIERTO
    Primer Lugar: 12
    Entre los tres primeros: 12
<Figure size 2000x400 with 6 Axes>



Consulta número:  14
   Imagen de consulta:  1132
   Mas cercanos:  [[8.954690337181091, '1132'], [15.326098918914795, '106748'], [15.65570080280304, '131266'], [15.684257984161377, '101149'], [16.152429461479187, '135432'], [16.181572914123535, '1062'], [16.322925806045532, '1233'], [16.35671901702881, '1117'], [16.464367389678955, '131219'], [16.588163256645203, '130199']]
--> ACIERTO
    Primer Lugar: 13
    Entre los tres primeros: 13
<Figure size 2000x400 with 6 Axes>



Consulta número:  15
   Imagen de consulta:  1135
   Mas cercanos:  [[7.3672816157341, '1135'], [20.676344990730286, '136185'], [20.745588541030884, '100605'], [20.973713874816895, '136262'], [21.082005739212036, '1160'], [21.370445728302002, '106811'], [21.379848897457123, '121209'], [21.80694854259491, '133670'], [22.01223075389862, '136151'], [22.017041325569153, '116334']]
--> ACIERTO
    Primer Lugar: 14
    Entre los tres primeros: 14
<Figure size 2000x400 with 6 Axes>



Consulta número:  16
   Imagen de consulta:  1139
   Mas cercanos:  [[15.147112369537354, '1139'], [18.502389788627625, '108547'], [19.312389254570007, '136286'], [19.353598952293396, '1138'], [19.355997323989868, '735'], [19.448174357414246, '130877'], [19.459977865219116, '109179'], [19.615748047828674, '136183'], [19.623730540275574, '1056'], [19.638827800750732, '108990']]
--> ACIERTO
    Primer Lugar: 15
    Entre los tres primeros: 15
<Figure size 2000x400 with 6 Axes>



Consulta número:  17
   Imagen de consulta:  1150
   Mas cercanos:  [[4.9497866332530975, '1150'], [9.118017196655273, '134900'], [9.225060641765594, '109286'], [9.238099336624146, '102857'], [9.385497808456421, '130847'], [9.402065396308899, '1319'], [9.539770603179932, '101448'], [9.547255277633667, '135217'], [9.553136706352234, '106814'], [9.704995274543762, '114809']]
--> ACIERTO
    Primer Lugar: 16
    Entre los tres primeros: 16
<Figure size 2000x400 with 6 Axes>



Consulta número:  18
   Imagen de consulta:  1167
   Mas cercanos:  [[9.625698566436768, '1167'], [15.997869372367859, '136471'], [16.04235351085663, '1924'], [16.109766364097595, '102474'], [16.115915060043335, '1317'], [16.186524510383606, '136182'], [16.194326877593994, '105026'], [16.24489974975586, '106814'], [16.24974822998047, '1233'], [16.270262956619263, '136289']]
--> ACIERTO
    Primer Lugar: 17
    Entre los tres primeros: 17
<Figure size 2000x400 with 6 Axes>



Consulta número:  19
   Imagen de consulta:  1173
   Mas cercanos:  [[13.806203424930573, '1273'], [14.36620044708252, '102859'], [14.378419637680054, '123005'], [14.42669004201889, '130172'], [14.553736746311188, '133678'], [14.60555350780487, '136472'], [14.651932299137115, '1168'], [14.678598999977112, '130593'], [14.691332221031189, '131129'], [14.725819289684296, '1304']]
<Figure size 2000x400 with 6 Axes>



Consulta número:  20
   Imagen de consulta:  1174
   Mas cercanos:  [[6.86212283372879, '1174'], [19.952153086662292, '1113'], [20.15688681602478, '1063'], [20.32674217224121, '1166'], [20.356155395507812, '130821'], [20.372102975845337, '102014'], [20.415014266967773, '102887'], [20.517735958099365, '1062'], [20.525392413139343, '101345'], [20.558359622955322, '101475']]
--> ACIERTO
    Primer Lugar: 18
    Entre los tres primeros: 18
<Figure size 2000x400 with 6 Axes>



Consulta número:  21
   Imagen de consulta:  1178
   Mas cercanos:  [[10.602867186069489, '1178'], [14.202077090740204, '130526'], [14.358408749103546, '136286'], [14.412107586860657, '130855'], [14.485334694385529, '136465'], [14.648776292800903, '109179'], [14.676574170589447, '130502'], [14.682904779911041, '1280'], [14.773177683353424, '136182'], [14.813586294651031, '106814']]
--> ACIERTO
    Primer Lugar: 19
    Entre los tres primeros: 19
<Figure size 2000x400 with 6 Axes>



Consulta número:  22
   Imagen de consulta:  1179
   Mas cercanos:  [[10.345093011856079, '136280'], [11.04150640964508, '130851'], [11.093942999839783, '1306'], [11.100153923034668, '133671'], [11.129230976104736, '135527'], [11.146345853805542, '102857'], [11.159262776374817, '136272'], [11.228484869003296, '130199'], [11.238738059997559, '131270'], [11.411070704460144, '129401']]
<Figure size 2000x400 with 6 Axes>



Consulta número:  23
   Imagen de consulta:  121206
   Mas cercanos:  [[14.277055263519287, '121206'], [18.02292561531067, '135433'], [18.158453941345215, '1077'], [18.24224889278412, '129739'], [18.453954458236694, '108981'], [18.46734583377838, '136473'], [18.52289581298828, '130527'], [18.78424060344696, '107309'], [18.79994249343872, '105180'], [18.820685148239136, '1080']]
--> ACIERTO
    Primer Lugar: 20
    Entre los tres primeros: 20
<Figure size 2000x400 with 6 Axes>



Consulta número:  24
   Imagen de consulta:  1253
   Mas cercanos:  [[9.7893226146698, '1253'], [10.797643542289734, '136583'], [11.13191294670105, '101448'], [11.17420482635498, '130287'], [11.185698866844177, '101154'], [11.269636988639832, '102857'], [11.295968651771545, '107684'], [11.40535032749176, '129153'], [11.415025353431702, '130799'], [11.55728006362915, '107325']]
--> ACIERTO
    Primer Lugar: 21
    Entre los tres primeros: 21
<Figure size 2000x400 with 6 Axes>



Consulta número:  25
   Imagen de consulta:  1664
   Mas cercanos:  [[16.387057840824127, '1664'], [19.027794361114502, '107222'], [20.06601309776306, '136238'], [20.329038739204407, '1334'], [20.476900696754456, '117754'], [21.049337327480316, '136198'], [21.513996839523315, '120921'], [21.616591572761536, '102890'], [21.71039581298828, '136185'], [21.790175795555115, '1923']]
--> ACIERTO
    Primer Lugar: 22
    Entre los tres primeros: 22
<Figure size 2000x400 with 6 Axes>



Consulta número:  26
   Imagen de consulta:  1668
   Mas cercanos:  [[5.746437430381775, '1668'], [12.803146600723267, '133178'], [14.58399486541748, '131125'], [15.081811666488647, '129863'], [15.08678412437439, '129572'], [15.264243364334106, '135318'], [15.332423448562622, '1289'], [15.34683072566986, '129694'], [15.350821614265442, '129827'], [15.39593231678009, '129429']]
--> ACIERTO
    Primer Lugar: 23
    Entre los tres primeros: 23
<Figure size 2000x400 with 6 Axes>



Consulta número:  27
   Imagen de consulta:  1687
   Mas cercanos:  [[10.351965069770813, '1687'], [17.23648691177368, '108571'], [17.991281867027283, '120936'], [18.52251636981964, '109284'], [18.59026277065277, '114947'], [18.614455103874207, '129840'], [19.146071314811707, '135252'], [19.265434980392456, '136160'], [19.32176375389099, '130555'], [19.45122516155243, '106363']]
--> ACIERTO
    Primer Lugar: 24
    Entre los tres primeros: 24
<Figure size 2000x400 with 6 Axes>



Consulta número:  28
   Imagen de consulta:  1707
   Mas cercanos:  [[6.618589699268341, '1707'], [11.430362462997437, '129168'], [11.73783802986145, '116347'], [11.819576799869537, '134892'], [11.845782041549683, '136274'], [11.860941410064697, '131270'], [11.887961030006409, '102854'], [11.926906943321228, '130502'], [11.928489565849304, '1080'], [11.932549118995667, '1317']]
--> ACIERTO
    Primer Lugar: 25
    Entre los tres primeros: 25
<Figure size 2000x400 with 6 Axes>



Consulta número:  29
   Imagen de consulta:  1710
   Mas cercanos:  [[14.317286670207977, '1710'], [17.77138078212738, '117183'], [17.927465796470642, '1222'], [18.000937044620514, '136276'], [18.030980110168457, '135356'], [18.087223291397095, '106696'], [18.114985167980194, '129414'], [18.167414367198944, '100939'], [18.177926778793335, '107289'], [18.201019048690796, '136244']]
--> ACIERTO
    Primer Lugar: 26
    Entre los tres primeros: 26
<Figure size 2000x400 with 6 Axes>



Consulta número:  30
   Imagen de consulta:  1714
   Mas cercanos:  [[12.337124526500702, '1714'], [19.91680073738098, '107201'], [20.076475381851196, '134888'], [20.191088438034058, '129408'], [20.22554588317871, '130855'], [20.275933504104614, '106814'], [20.349745869636536, '129420'], [20.381013989448547, '136465'], [20.42736566066742, '102854'], [20.44165599346161, '135257']]
--> ACIERTO
    Primer Lugar: 27
    Entre los tres primeros: 27
<Figure size 2000x400 with 6 Axes>



Consulta número:  31
   Imagen de consulta:  1717
   Mas cercanos:  [[16.121859073638916, '1717'], [16.218510031700134, '82'], [16.325454592704773, '130508'], [16.32834041118622, '1371'], [16.735196113586426, '101475'], [16.92128562927246, '101395'], [16.94486403465271, '120946'], [16.987059116363525, '130348'], [16.987237095832825, '1175'], [17.060892820358276, '130266']]
--> ACIERTO
    Primer Lugar: 28
    Entre los tres primeros: 28
<Figure size 2000x400 with 6 Axes>



Consulta número:  32
   Imagen de consulta:  1724
   Mas cercanos:  [[10.678421199321747, '1724'], [14.660337805747986, '102854'], [14.707985997200012, '129841'], [14.711383581161499, '130811'], [14.89566683769226, '130502'], [14.943395733833313, '134893'], [14.964501321315765, '134892'], [14.969112634658813, '130199'], [15.090151846408844, '136198'], [15.095284223556519, '131249']]
--> ACIERTO
    Primer Lugar: 29
    Entre los tres primeros: 29
<Figure size 2000x400 with 6 Axes>



Consulta número:  33
   Imagen de consulta:  1733
   Mas cercanos:  [[6.2097868621349335, '1733'], [16.61811488866806, '101151'], [16.721712946891785, '103285'], [16.91959571838379, '129841'], [16.931257605552673, '136265'], [16.93766462802887, '130177'], [17.045619428157806, '135469'], [17.11065411567688, '1710'], [17.131803810596466, '133677'], [17.197834849357605, '1297']]
--> ACIERTO
    Primer Lugar: 30
    Entre los tres primeros: 30
<Figure size 2000x400 with 6 Axes>



Consulta número:  34
   Imagen de consulta:  1737
   Mas cercanos:  [[6.6692036390304565, '1737'], [12.64210832118988, '130502'], [12.66057288646698, '130199'], [12.750509858131409, '102019'], [12.832952737808228, '102854'], [12.856863856315613, '134892'], [12.901248216629028, '105026'], [12.911281824111938, '136272'], [12.921626448631287, '130526'], [12.92321252822876, '133678']]
--> ACIERTO
    Primer Lugar: 31
    Entre los tres primeros: 31
<Figure size 2000x400 with 6 Axes>



Consulta número:  35
   Imagen de consulta:  1742
   Mas cercanos:  [[5.655642151832581, '1742'], [17.414242267608643, '129416'], [17.93628966808319, '102019'], [17.965649247169495, '129399'], [17.989693999290466, '136176'], [18.000372529029846, '108911'], [18.050549626350403, '129575'], [18.081570744514465, '1222'], [18.107390522956848, '130352'], [18.13758772611618, '105026']]
--> ACIERTO
    Primer Lugar: 32
    Entre los tres primeros: 32
<Figure size 2000x400 with 6 Axes>



Consulta número:  36
   Imagen de consulta:  1749
   Mas cercanos:  [[10.64304769039154, '129735'], [10.974472880363464, '130342'], [11.103824138641357, '1124'], [11.249004125595093, '129168'], [11.261950492858887, '129201'], [11.343178272247314, '1749'], [11.365780711174011, '82'], [11.417550086975098, '120946'], [11.574684143066406, '1080'], [11.597774624824524, '116347']]
--> ACIERTO
<Figure size 2000x400 with 6 Axes>



Consulta número:  37
   Imagen de consulta:  1775
   Mas cercanos:  [[9.072097659111023, '1775'], [13.53890347480774, '136274'], [14.083515882492065, '1238'], [14.221127986907959, '130172'], [14.520284295082092, '135318'], [14.582157492637634, '102372'], [14.68742048740387, '135217'], [14.694976091384888, '133678'], [14.807462692260742, '109029'], [14.860350012779236, '135430']]
--> ACIERTO
    Primer Lugar: 33
    Entre los tres primeros: 33
<Figure size 2000x400 with 6 Axes>



Consulta número:  38
   Imagen de consulta:  1777
   Mas cercanos:  [[7.75236514210701, '1777'], [15.4374178647995, '104776'], [15.582584500312805, '107314'], [15.60972774028778, '136286'], [15.797244668006897, '130172'], [15.819960355758667, '136471'], [15.825151681900024, '100488'], [15.835433721542358, '135448'], [15.85348665714264, '130229'], [15.912088632583618, '130506']]
--> ACIERTO
    Primer Lugar: 34
    Entre los tres primeros: 34
<Figure size 2000x400 with 6 Axes>



Consulta número:  39
   Imagen de consulta:  1885
   Mas cercanos:  [[5.875598788261414, '1885'], [11.784425616264343, '101151'], [11.79670786857605, '136272'], [12.010274887084961, '129401'], [12.066861689090729, '130199'], [12.185065388679504, '130502'], [12.223091661930084, '102464'], [12.271824836730957, '129201'], [12.299468040466309, '134892'], [12.351540982723236, '1306']]
--> ACIERTO
    Primer Lugar: 35
    Entre los tres primeros: 35
<Figure size 2000x400 with 6 Axes>



Consulta número:  40
   Imagen de consulta:  1916
   Mas cercanos:  [[6.732964485883713, '1916'], [19.897124886512756, '135345'], [20.055351972579956, '129569'], [20.140297293663025, '1923'], [20.309803247451782, '136306'], [20.39133846759796, '136286'], [20.460002183914185, '135448'], [20.500001311302185, '121106'], [20.516180753707886, '136262'], [20.536266565322876, '107314']]
--> ACIERTO
    Primer Lugar: 36
    Entre los tres primeros: 36
<Figure size 2000x400 with 6 Axes>



Consulta número:  41
   Imagen de consulta:  1919
   Mas cercanos:  [[12.440356016159058, '1919'], [15.362513065338135, '129830'], [15.365527629852295, '135411'], [15.464618921279907, '130545'], [15.468209385871887, '130845'], [15.517006993293762, '1306'], [15.526419997215271, '1289'], [15.555464506149292, '1126'], [15.590194582939148, '135432'], [15.591602683067322, '1124']]
--> ACIERTO
    Primer Lugar: 37
    Entre los tres primeros: 37
<Figure size 2000x400 with 6 Axes>



Consulta número:  42
   Imagen de consulta:  1922
   Mas cercanos:  [[14.337208092212677, '1922'], [17.548864126205444, '109301'], [18.30406630039215, '107208'], [18.77587068080902, '120063'], [19.012640237808228, '136240'], [19.37779402732849, '102470'], [19.414570689201355, '107334'], [19.456023931503296, '109177'], [19.458038449287415, '102467'], [19.751022338867188, '123004']]
--> ACIERTO
    Primer Lugar: 38
    Entre los tres primeros: 38
<Figure size 2000x400 with 6 Axes>



Consulta número:  43
   Imagen de consulta:  1923
   Mas cercanos:  [[11.550927758216858, '104776'], [13.094675779342651, '1923'], [13.212565302848816, '130502'], [13.258134603500366, '135382'], [13.263745903968811, '100488'], [13.281420707702637, '130199'], [13.485091209411621, '1133'], [13.5338773727417, '130229'], [13.585374116897583, '101151'], [13.634775757789612, '106814']]
--> ACIERTO
    Entre los tres primeros: 39
<Figure size 2000x400 with 6 Axes>



Consulta número:  44
   Imagen de consulta:  1924
   Mas cercanos:  [[5.652427583932877, '1924'], [10.619499444961548, '129401'], [10.683126449584961, '1306'], [10.796486735343933, '133671'], [10.870205998420715, '102857'], [10.932117462158203, '136272'], [10.952689170837402, '133678'], [11.111360430717468, '107320'], [11.15127956867218, '135249'], [11.159831285476685, '131270']]
--> ACIERTO
    Primer Lugar: 39
    Entre los tres primeros: 40
<Figure size 2000x400 with 6 Axes>



Consulta número:  45
   Imagen de consulta:  1926
   Mas cercanos:  [[13.275643169879913, '1926'], [22.27188003063202, '108522'], [22.360029101371765, '135349'], [22.372679948806763, '101493'], [22.39889359474182, '1294'], [22.448642253875732, '131264'], [22.64794611930847, '135248'], [22.680270075798035, '135449'], [22.71182954311371, '130790'], [22.75346291065216, '108995']]
--> ACIERTO
    Primer Lugar: 40
    Entre los tres primeros: 41
<Figure size 2000x400 with 6 Axes>



Consulta número:  46
   Imagen de consulta:  1992
   Mas cercanos:  [[9.265371561050415, '1992'], [14.99432748556137, '1371'], [15.070064783096313, '1050'], [15.241854071617126, '101475'], [15.26987338066101, '130508'], [15.299675583839417, '130342'], [15.598389387130737, '130348'], [15.904709458351135, '130377'], [15.949660539627075, '129735'], [15.962887525558472, '116138']]
--> ACIERTO
    Primer Lugar: 41
    Entre los tres primeros: 42
<Figure size 2000x400 with 6 Axes>



Consulta número:  47
   Imagen de consulta:  2097
   Mas cercanos:  [[9.446678519248962, '2097'], [14.8499516248703, '102010'], [15.205232977867126, '102019'], [15.30216372013092, '1737'], [15.496350526809692, '130522'], [15.562003254890442, '135534'], [15.887035846710205, '102862'], [15.986714601516724, '135316'], [16.008240818977356, '130851'], [16.024129629135132, '107340']]
--> ACIERTO
    Primer Lugar: 42
    Entre los tres primeros: 43
<Figure size 2000x400 with 6 Axes>



Consulta número:  48
   Imagen de consulta:  2238
   Mas cercanos:  [[6.36750453710556, '2238'], [16.061869859695435, '135249'], [16.211435556411743, '102854'], [16.340815663337708, '130594'], [16.359856367111206, '107673'], [16.426778197288513, '1117'], [16.48083794116974, '1280'], [16.53054964542389, '130768'], [16.550312757492065, '126416'], [16.60403263568878, '130502']]
--> ACIERTO
    Primer Lugar: 43
    Entre los tres primeros: 44
<Figure size 2000x400 with 6 Axes>



Consulta número:  49
   Imagen de consulta:  602
   Mas cercanos:  [[7.318150520324707, '602'], [14.288129329681396, '1124'], [14.506348133087158, '1231'], [14.677678108215332, '135430'], [14.68820059299469, '120782'], [14.703824162483215, '130266'], [14.8368821144104, '108578'], [14.884440183639526, '129665'], [14.897590279579163, '130499'], [14.960530400276184, '108990']]
--> ACIERTO
    Primer Lugar: 44
    Entre los tres primeros: 45
<Figure size 2000x400 with 6 Axes>



Consulta número:  50
   Imagen de consulta:  603
   Mas cercanos:  [[10.177733898162842, '135365'], [11.357524812221527, '603'], [11.375930190086365, '131169'], [11.384572744369507, '135393'], [11.503473043441772, '1233'], [11.56798791885376, '126417'], [11.62687349319458, '130845'], [11.642546772956848, '108990'], [11.69806170463562, '135432'], [11.735713124275208, '1032']]
--> ACIERTO
    Entre los tres primeros: 46
<Figure size 2000x400 with 6 Axes>



Consulta número:  51
   Imagen de consulta:  617
   Mas cercanos:  [[11.974755764007568, '617'], [15.164913058280945, '136157'], [15.238269329071045, '136430'], [15.312967658042908, '136286'], [15.340859174728394, '123005'], [15.57683265209198, '136276'], [15.798658668994904, '101789'], [15.91408258676529, '135396'], [16.035731196403503, '121216'], [16.037410855293274, '121209']]
--> ACIERTO
    Primer Lugar: 45
    Entre los tres primeros: 47
<Figure size 2000x400 with 6 Axes>



Consulta número:  52
   Imagen de consulta:  622
   Mas cercanos:  [[12.199498295783997, '622'], [15.651965022087097, '130352'], [15.7134690284729, '135391'], [15.739050149917603, '136471'], [15.83806562423706, '106814'], [15.86995804309845, '136182'], [15.89772343635559, '107353'], [15.905786395072937, '1222'], [15.92667806148529, '135527'], [15.934474110603333, '130502']]
--> ACIERTO
    Primer Lugar: 46
    Entre los tres primeros: 48
<Figure size 2000x400 with 6 Axes>



Consulta número:  53
   Imagen de consulta:  630
   Mas cercanos:  [[5.0044040977954865, '630'], [13.942125618457794, '129868'], [14.05857503414154, '136430'], [14.22900116443634, '136276'], [14.26041865348816, '136434'], [14.279647648334503, '129852'], [14.43185842037201, '121216'], [14.498786211013794, '121209'], [14.507378339767456, '135356'], [14.508490085601807, '102477']]
--> ACIERTO
    Primer Lugar: 47
    Entre los tres primeros: 49
<Figure size 2000x400 with 6 Axes>



Consulta número:  54
   Imagen de consulta:  642
   Mas cercanos:  [[8.837047934532166, '642'], [11.695130586624146, '130199'], [12.001793205738068, '130502'], [12.047494947910309, '102854'], [12.081633925437927, '136272'], [12.134522914886475, '1140'], [12.145786941051483, '133671'], [12.21035772562027, '1306'], [12.247726619243622, '130594'], [12.271779298782349, '1233']]
--> ACIERTO
    Primer Lugar: 48
    Entre los tres primeros: 50
<Figure size 2000x400 with 6 Axes>



Consulta número:  55
   Imagen de consulta:  676
   Mas cercanos:  [[9.923581540584564, '676'], [12.873779714107513, '130594'], [13.027934074401855, '134892'], [13.10313594341278, '1274'], [13.13098692893982, '133678'], [13.145153045654297, '106814'], [13.205330967903137, '135316'], [13.21186912059784, '130502'], [13.377500057220459, '133671'], [13.391045212745667, '105026']]
--> ACIERTO
    Primer Lugar: 49
    Entre los tres primeros: 51
<Figure size 2000x400 with 6 Axes>



Consulta número:  56
   Imagen de consulta:  679
   Mas cercanos:  [[11.290247619152069, '679'], [12.263178586959839, '135401'], [12.636808812618256, '1066'], [12.748082995414734, '129414'], [12.926599204540253, '105995'], [12.950976490974426, '130824'], [13.359371185302734, '129832'], [13.439621448516846, '102854'], [13.563106894493103, '136260'], [13.75035035610199, '105026']]
--> ACIERTO
    Primer Lugar: 50
    Entre los tres primeros: 52
<Figure size 2000x400 with 6 Axes>



Consulta número:  57
   Imagen de consulta:  680
   Mas cercanos:  [[8.978023797273636, '680'], [17.049796104431152, '106858'], [17.350905299186707, '136286'], [17.411797642707825, '1710'], [17.479527354240417, '130805'], [17.511802077293396, '123005'], [17.583402156829834, '136276'], [17.841649413108826, '136430'], [17.86796247959137, '136157'], [17.915636897087097, '1051']]
--> ACIERTO
    Primer Lugar: 51
    Entre los tres primeros: 53
<Figure size 2000x400 with 6 Axes>



Consulta número:  58
   Imagen de consulta:  691
   Mas cercanos:  [[3.5322186946868896, '691'], [8.971526265144348, '108990'], [9.12481141090393, '1124'], [9.157012462615967, '135430'], [9.311515808105469, '1306'], [9.529293537139893, '1122'], [10.064883351325989, '1161'], [10.096784830093384, '129665'], [10.108513951301575, '131270'], [10.127933025360107, '113522']]
--> ACIERTO
    Primer Lugar: 52
    Entre los tres primeros: 54
<Figure size 2000x400 with 6 Axes>



Consulta número:  59
   Imagen de consulta:  713
   Mas cercanos:  [[1.4626530185341835, '713'], [18.749972820281982, '130541'], [18.955222606658936, '1222'], [19.013768196105957, '130344'], [19.030195593833923, '130522'], [19.030423879623413, '135534'], [19.057205379009247, '129664'], [19.091899156570435, '1225'], [19.13635540008545, '107332'], [19.140767097473145, '136212']]
--> ACIERTO
    Primer Lugar: 53
    Entre los tres primeros: 55
<Figure size 2000x400 with 6 Axes>



Consulta número:  60
   Imagen de consulta:  728
   Mas cercanos:  [[16.390412151813507, '102854'], [16.43559753894806, '130199'], [16.51046323776245, '107366'], [16.577951192855835, '130594'], [16.610233068466187, '130502'], [16.6118346452713, '1280'], [16.612215995788574, '105898'], [16.626583218574524, '106814'], [16.63258707523346, '135467'], [16.64370882511139, '133671']]
<Figure size 2000x400 with 6 Axes>



Consulta número:  61
   Imagen de consulta:  732
   Mas cercanos:  [[25.358164429664612, '732'], [27.067147493362427, '107357'], [27.295488476753235, '1114'], [27.97553551197052, '102014'], [27.9900141954422, '1183'], [28.043824791908264, '104388'], [28.07925236225128, '130866'], [28.132863402366638, '130821'], [28.14563751220703, '1079'], [28.162513256072998, '1123']]
--> ACIERTO
    Primer Lugar: 54
    Entre los tres primeros: 56
<Figure size 2000x400 with 6 Axes>



Consulta número:  62
   Imagen de consulta:  735
   Mas cercanos:  [[5.285388886928558, '735'], [11.526618540287018, '108990'], [12.106644093990326, '1124'], [12.516819894313812, '135430'], [12.573967576026917, '1161'], [12.66781759262085, '107356'], [12.766757369041443, '1126'], [12.831379175186157, '1306'], [12.926431238651276, '135432'], [12.980287909507751, '120782']]
--> ACIERTO
    Primer Lugar: 55
    Entre los tres primeros: 57
<Figure size 2000x400 with 6 Axes>



Consulta número:  63
   Imagen de consulta:  78
   Mas cercanos:  [[7.940292209386826, '78'], [14.986706137657166, '104776'], [15.224127650260925, '130229'], [15.327361583709717, '1095'], [15.352619409561157, '105013'], [15.527525067329407, '130502'], [15.572686672210693, '104395'], [15.596728444099426, '136021'], [15.663859724998474, '130855'], [15.671100974082947, '135534']]
--> ACIERTO
    Primer Lugar: 56
    Entre los tres primeros: 58
<Figure size 2000x400 with 6 Axes>



Consulta número:  64
   Imagen de consulta:  82
   Mas cercanos:  [[6.372326552867889, '82'], [8.733253598213196, '129735'], [9.579531788825989, '129168'], [9.802386045455933, '116347'], [9.866766929626465, '129201'], [9.920103430747986, '1116'], [10.079073071479797, '135434'], [10.08866024017334, '131270'], [10.302116394042969, '1306'], [10.34554660320282, '1080']]
--> ACIERTO
    Primer Lugar: 57
    Entre los tres primeros: 59
<Figure size 2000x400 with 6 Axes>



Consulta número:  65
   Imagen de consulta:  858
   Mas cercanos:  [[7.978060781955719, '858'], [22.959402203559875, '123005'], [23.025834918022156, '1058'], [23.050556659698486, '107289'], [23.123098492622375, '135371'], [23.13552415370941, '131183'], [23.21694815158844, '1349'], [23.339951753616333, '135258'], [23.364532828330994, '136205'], [23.443392515182495, '103287']]
--> ACIERTO
    Primer Lugar: 58
    Entre los tres primeros: 60
<Figure size 2000x400 with 6 Axes>



Consulta número:  66
   Imagen de consulta:  862
   Mas cercanos:  [[6.183857321739197, '862'], [14.284044325351715, '105026'], [14.381043016910553, '106814'], [14.596391379833221, '1280'], [14.618592858314514, '135434'], [14.683040618896484, '136182'], [14.739821434020996, '136274'], [14.755603671073914, '135469'], [14.760741710662842, '133678'], [14.767910361289978, '134892']]
--> ACIERTO
    Primer Lugar: 59
    Entre los tres primeros: 61
<Figure size 2000x400 with 6 Axes>



Consulta número:  67
   Imagen de consulta:  865
   Mas cercanos:  [[12.648130893707275, '865'], [16.228736400604248, '108547'], [16.469423174858093, '735'], [16.72526478767395, '130545'], [16.925682067871094, '136473'], [17.019522309303284, '106814'], [17.06144070625305, '116288'], [17.06645667552948, '136007'], [17.08359920978546, '130797'], [17.085219025611877, '136464']]
--> ACIERTO
    Primer Lugar: 60
    Entre los tres primeros: 62
<Figure size 2000x400 with 6 Axes>



Consulta número:  68
   Imagen de consulta:  868
   Mas cercanos:  [[4.862064450979233, '868'], [11.407978475093842, '133676'], [11.440744757652283, '130058'], [11.545189321041107, '136465'], [11.585348904132843, '136433'], [11.844626665115356, '130855'], [11.917518198490143, '129832'], [11.9376882314682, '129874'], [12.003294229507446, '136199'], [12.023762941360474, '135346']]
--> ACIERTO
    Primer Lugar: 61
    Entre los tres primeros: 63
<Figure size 2000x400 with 6 Axes>



Consulta número:  69
   Imagen de consulta:  87
   Mas cercanos:  [[10.194679498672485, '87'], [16.166340231895447, '135252'], [16.236706852912903, '130763'], [16.250885248184204, '133677'], [16.36498510837555, '130840'], [16.383586168289185, '131129'], [16.51004207134247, '136160'], [16.629215240478516, '1317'], [16.63729476928711, '104395'], [16.653055608272552, '1885']]
--> ACIERTO
    Primer Lugar: 62
    Entre los tres primeros: 64
<Figure size 2000x400 with 6 Axes>



Consulta número:  70
   Imagen de consulta:  89
   Mas cercanos:  [[15.115935206413269, '89'], [22.765750527381897, '120779'], [22.864940285682678, '106755'], [22.941460371017456, '102368'], [23.120044946670532, '135439'], [23.134526133537292, '1348'], [23.227131009101868, '130866'], [23.404680490493774, '130266'], [23.44375503063202, '107222'], [23.520824790000916, '1749']]
--> ACIERTO
    Primer Lugar: 63
    Entre los tres primeros: 65
<Figure size 2000x400 with 6 Axes>



Consulta número:  71
   Imagen de consulta:  907
   Mas cercanos:  [[5.309684723615646, '121201'], [5.309684723615646, '907'], [15.078906416893005, '136150'], [15.137284636497498, '106858'], [15.509800910949707, '136276'], [15.5750070810318, '136286'], [15.642411589622498, '1253'], [15.657328128814697, '129209'], [15.711740374565125, '130366'], [15.774551033973694, '101154']]
--> ACIERTO
    Entre los tres primeros: 66
<Figure size 2000x400 with 6 Axes>



Consulta número:  72
   Imagen de consulta:  908
   Mas cercanos:  [[13.813987016677856, '908'], [17.65608686208725, '1129'], [21.172281742095947, '130532'], [22.25783061981201, '107311'], [22.835329055786133, '136235'], [23.003249049186707, '130595'], [23.13448029756546, '130549'], [23.146735072135925, '1159'], [23.35420650243759, '108917'], [23.35687291622162, '130524']]
--> ACIERTO
    Primer Lugar: 64
    Entre los tres primeros: 67
<Figure size 2000x400 with 6 Axes>



Consulta número:  73
   Imagen de consulta:  918
   Mas cercanos:  [[13.844865560531616, '918'], [23.899436831474304, '121219'], [25.37635827064514, '1334'], [25.440362453460693, '130838'], [25.832783579826355, '102467'], [25.943926095962524, '107222'], [26.139012098312378, '1131'], [26.14477789402008, '136257'], [26.173611283302307, '1664'], [26.21904230117798, '130866']]
--> ACIERTO
    Primer Lugar: 65
    Entre los tres primeros: 68
<Figure size 2000x400 with 6 Axes>



Consulta número:  74
   Imagen de consulta:  920
   Mas cercanos:  [[12.895130395889282, '920'], [16.708801984786987, '129169'], [17.573147416114807, '130791'], [17.71064269542694, '102372'], [17.891080021858215, '1353'], [18.209316194057465, '1366'], [18.434913635253906, '121212'], [18.593817591667175, '129182'], [18.59812867641449, '130061'], [18.769800901412964, '130366']]
--> ACIERTO
    Primer Lugar: 66
    Entre los tres primeros: 69
<Figure size 2000x400 with 6 Axes>



Consulta número:  75
   Imagen de consulta:  921
   Mas cercanos:  [[7.77647066116333, '921'], [20.430341482162476, '135443'], [21.086667895317078, '130507'], [21.192647576332092, '102555'], [21.26897644996643, '114819'], [21.72930109500885, '136472'], [21.73882257938385, '136205'], [22.222596883773804, '129683'], [22.584725975990295, '1287'], [22.66490387916565, '109210']]
--> ACIERTO
    Primer Lugar: 67
    Entre los tres primeros: 70
<Figure size 2000x400 with 6 Axes>



Consulta número:  76
   Imagen de consulta:  93
   Mas cercanos:  [[10.51577776670456, '130199'], [10.774484753608704, '135249'], [10.831052541732788, '1140'], [10.95486605167389, '136272'], [10.970808148384094, '136280'], [11.010870099067688, '134892'], [11.118045330047607, '93'], [11.121821403503418, '1306'], [11.211461544036865, '130817'], [11.275428533554077, '107363']]
--> ACIERTO
<Figure size 2000x400 with 6 Axes>



Consulta número:  77
   Imagen de consulta:  944
   Mas cercanos:  [[13.783286094665527, '944'], [17.629611432552338, '130177'], [17.65601933002472, '129874'], [17.67401385307312, '1108'], [17.696417093276978, '133677'], [17.758865237236023, '119741'], [17.76289927959442, '136286'], [17.795828878879547, '130502'], [17.839009404182434, '104395'], [17.84133470058441, '1280']]
--> ACIERTO
    Primer Lugar: 68
    Entre los tres primeros: 71
<Figure size 2000x400 with 6 Axes>



Consulta número:  78
   Imagen de consulta:  96
   Mas cercanos:  [[6.963617652654648, '96'], [11.814855098724365, '131190'], [12.1478853225708, '1194'], [12.23899781703949, '1126'], [12.357704877853394, '135430'], [12.525445520877838, '1124'], [12.580196380615234, '130368'], [12.607241094112396, '117199'], [12.638088345527649, '105612'], [12.734065413475037, '1100']]
--> ACIERTO
    Primer Lugar: 69
    Entre los tres primeros: 72
<Figure size 2000x400 with 6 Axes>



Consulta número:  79
   Imagen de consulta:  961
   Mas cercanos:  [[5.261099606752396, '961'], [12.072373807430267, '105026'], [12.311771094799042, '102857'], [12.459428369998932, '130058'], [12.531982660293579, '1150'], [12.532101035118103, '106814'], [12.604421496391296, '136583'], [12.655323147773743, '130526'], [12.683288097381592, '133671'], [12.69603407382965, '129209']]
--> ACIERTO
    Primer Lugar: 70
    Entre los tres primeros: 73
<Figure size 2000x400 with 6 Axes>



Porcentaje de aciertos NN1:  0.875
Porcentaje de aciertos NN3:  0.9125
Porcentaje de aciertos NN5:  0.95
  Cantidad consultas:  80
```


**[Celda 22 - Código]**
```python
print(no_encontrados)

no_posicion = [np.where(y_imagenesBD==etiqueta)[0][0] for etiqueta in no_encontrados]
no_query = [np.where(y_consultas==etiqueta)[0][0] for etiqueta in no_encontrados]

print(no_posicion)

for idx in range(len(no_encontrados)):
    plt.imshow(imagenesBD[no_posicion[idx]])
    plt.show()
    plt.imshow(consultas[no_query[idx]])
    plt.show()
                        
    
```


*Salida:*
```text
['102476', '1179', '1923', '728']
[141, 714, 2016, 2033]
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>
```


**[Celda 23 - Código]**
```python
mat = vectores_elemento[139][:,:,511]
for i in range(511):
    mat = mat + vectores_elemento[139][:,:,i]
plt.imshow(mat)
plt.show()
```


*Salida:*
```text
<Figure size 640x480 with 1 Axes>
```


**[Celda 24 - Código]**
```python
print(len(imagenesBD))
```


*Salida:*
```text
4554
```

## Resnet



**[Celda 26 - Código]**
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

resnet = build_resnet(n_classes=10388)
#cambiar las clases segun la cantidad de imagenes en la base de datos
```


**[Celda 27 - Código]**
```python
resnet.summary()
```


*Salida:*
```text
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 224, 224, 3) 0                                            
__________________________________________________________________________________________________
conv1_pad (ZeroPadding2D)       (None, 230, 230, 3)  0           input_1[0][0]                    
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
global_average_pooling2d (Globa (None, 2048)         0           conv5_block3_out[0][0]           
__________________________________________________________________________________________________
dropout (Dropout)               (None, 2048)         0           global_average_pooling2d[0][0]   
__________________________________________________________________________________________________
dense (Dense)                   (None, 10388)        21285012    dropout[0][0]                    
==================================================================================================
Total params: 44,872,724
Trainable params: 44,819,604
Non-trainable params: 53,120
__________________________________________________________________________________________________
```


**[Celda 28 - Código]**
```python
salida = resnet.get_layer('pool1_pool')
salida2 = resnet.get_layer('conv3_block4_out')
new_model_resnet = Model(inputs=resnet.input, outputs=salida.output)
new_model_resnet2 = Model(inputs=resnet.input, outputs=salida2.output)

# Porcentaje de aciertos NN1:  0.8625
# Porcentaje de aciertos NN3:  0.9
# Porcentaje de aciertos NN5:  0.9375

# promedio ponderado
# Porcentaje de aciertos NN1:  0.875
# Porcentaje de aciertos NN3:  0.9125
# Porcentaje de aciertos NN5:  0.95
```


**[Celda 29 - Código]**
```python
# Asignar la capa de pooling global a una variable
# salida = resnet.get_layer('conv4_block5_add')
# salida = resnet.get_layer('conv4_block2_add')
salida = resnet.get_layer('pool1_pool')



### RESULTADO conv1_bn
### Porcentaje de aciertos NN1:  0.875
### Porcentaje de aciertos NN3:  0.9125
### Porcentaje de aciertos NN5:  0.9375

### RESULTADO conv1_relu
### Porcentaje de aciertos NN1:  0.875
### Porcentaje de aciertos NN3:  0.925
### Porcentaje de aciertos NN5:  0.9375

### RESULTADO conv1_conv
### Porcentaje de aciertos NN1:  0.9
### Porcentaje de aciertos NN3:  0.925
### Porcentaje de aciertos NN5:  0.9375

### RESULTADO pool1_pool
### Porcentaje de aciertos NN1:  0.9
### Porcentaje de aciertos NN3:  0.9375
### Porcentaje de aciertos NN5:  0.95


### RESULTADO conv2_block1_out
### Porcentaje de aciertos NN1:  0.9
### Porcentaje de aciertos NN3:  0.9125
### Porcentaje de aciertos NN5:  0.9375

### RESULTADO conv2_block3_out
### Porcentaje de aciertos NN1:  0.8
### Porcentaje de aciertos NN3:  0.8625
### Porcentaje de aciertos NN5:  0.8625

### RESULTADO conv3_block2_out
### Porcentaje de aciertos NN1:  0.7125
### Porcentaje de aciertos NN3:  0.7625
### Porcentaje de aciertos NN5:  0.7875

### RESULTADO conv3_block3_out
### Porcentaje de aciertos NN1:  0.7625
### Porcentaje de aciertos NN3:  0.8
### Porcentaje de aciertos NN5:  0.85
### CON BLUR:
### Porcentaje de aciertos NN1:  0.8625
### Porcentaje de aciertos NN3:  0.875
### Porcentaje de aciertos NN5:  0.9125
### Arreglando algunos numeros de consulta que estaban mal
### Porcentaje de aciertos NN1:  0.875
### Porcentaje de aciertos NN3:  0.8875
### Porcentaje de aciertos NN5:  0.925

### RESULTADO conv3_block4_out
# Porcentaje de aciertos NN1:  0.8625
# Porcentaje de aciertos NN3:  0.9
# Porcentaje de aciertos NN5:  0.9125

### RESULTADO conv4_block1_out
### Porcentaje de aciertos NN1:  0.65
### Porcentaje de aciertos NN3:  0.775
### Porcentaje de aciertos NN10:  0.825

### RESULTADO conv4_block4_out
### Porcentaje de aciertos NN1:  0.625
### Porcentaje de aciertos NN3:  0.6625
### Porcentaje de aciertos NN5:  0.725 

### RESULTADO conv5_block3_out
### Porcentaje de aciertos NN1:  0.7625
### Porcentaje de aciertos NN3:  0.8125
### Porcentaje de aciertos NN5:  0.85

### RESULTADO global_average_pooling2d
### Porcentaje de aciertos NN1:  0.3875
### Porcentaje de aciertos NN3:  0.4875
### Porcentaje de aciertos NN5:  0.575


# salida = resnet.get_layer('conv5_block3_out')

# Crear un nuevo modelo que tenga la capa de pooling global como salida
new_model_resnet = Model(inputs=resnet.input, outputs=salida.output)
```


**[Celda 30 - Código]**
```python

```

## Fine Tuning


**[Celda 32 - Código]**
```python
imagenesBD[0].shape
print(len(imagenesBD))
```


*Salida:*
```text
10388
```


**[Celda 33 - Código]**
```python
 np.where(y_imagenesBD == '131152')
```


*Salida:*
```text
(array([2285], dtype=int64),)
```


**[Celda 34 - Código]**
```python
np.where(y_consultas == '1708')
```


*Salida:*
```text
(array([1], dtype=int64),)
```


**[Celda 35 - Código]**
```python
dist = np.sqrt(np.sum(np.square(vectores_consulta[0]-vectores_elemento[133])))
print(dist)
```


*Salida:*
```text
118.8077
```


**[Celda 36 - Código]**
```python
import numpy as np

# Crear un ndarray

# Buscar el índice del elemento 82
indices = np.where(y_imagenesBD == "82")

print(f'Los índices del elemento 82 son: {indices[0]}')


```


*Salida:*
```text
Los índices del elemento 82 son: [1261]
```


**[Celda 37 - Código]**
```python
type(imagenesBD[0])
```


*Salida:*
```text
numpy.ndarray
```


**[Celda 38 - Código]**
```python
y_consultas[77]
```


*Salida:*
```text
'boca'
```


**[Celda 39 - Código]**
```python
type(consultas[0])
```


*Salida:*
```text
numpy.ndarray
```


**[Celda 40 - Código]**
```python
plt.imshow(imagenesBD[1261])
plt.show()
print(imagenesBD[1261])
```


*Salida:*
```text
<Figure size 640x480 with 1 Axes>[[[1.         1.         0.99215686]
  [1.         1.         0.99215686]
  [1.         1.         0.99607843]
  ...
  [1.         1.         1.        ]
  [1.         1.         1.        ]
  [1.         1.         1.        ]]

 [[1.         1.         0.99215686]
  [1.         1.         0.99215686]
  [1.         1.         0.99607843]
  ...
  [1.         1.         1.        ]
  [1.         1.         1.        ]
  [1.         1.         1.        ]]

 [[1.         1.         1.        ]
  [1.         1.         1.        ]
  [1.         1.         1.        ]
  ...
  [1.         1.         1.        ]
  [1.         1.         1.        ]
  [1.         1.         1.        ]]

 ...

 [[1.         1.         1.        ]
  [1.         1.         1.        ]
  [1.         1.         1.        ]
  ...
  [1.         1.         1.        ]
  [1.         1.         1.        ]
  [1.         1.         1.        ]]

 [[1.         1.         1.        ]
  [1.         1.         1.        ]
  [1.         1.         1.        ]
  ...
  [1.         1.         1.        ]
  [1.         1.         1.        ]
  [1.         1.         1.        ]]

 [[1.         1.         1.        ]
  [1.         1.         1.        ]
  [1.         1.         1.        ]
  ...
  [1.         1.         1.        ]
  [1.         1.         1.        ]
  [1.         1.         1.        ]]]
```


**[Celda 41 - Código]**
```python
plt.imshow(consultas[4])
plt.show()
print(consultas[4])
```


*Salida:*
```text
<Figure size 640x480 with 1 Axes>[[[0. 0. 0.]
  [0. 0. 0.]
  [0. 0. 0.]
  ...
  [0. 0. 0.]
  [0. 0. 0.]
  [0. 0. 0.]]

 [[0. 0. 0.]
  [0. 0. 0.]
  [0. 0. 0.]
  ...
  [0. 0. 0.]
  [0. 0. 0.]
  [0. 0. 0.]]

 [[0. 0. 0.]
  [0. 0. 0.]
  [0. 0. 0.]
  ...
  [0. 0. 0.]
  [0. 0. 0.]
  [0. 0. 0.]]

 ...

 [[0. 0. 0.]
  [0. 0. 0.]
  [0. 0. 0.]
  ...
  [0. 0. 0.]
  [0. 0. 0.]
  [0. 0. 0.]]

 [[0. 0. 0.]
  [0. 0. 0.]
  [0. 0. 0.]
  ...
  [0. 0. 0.]
  [0. 0. 0.]
  [0. 0. 0.]]

 [[0. 0. 0.]
  [0. 0. 0.]
  [0. 0. 0.]
  ...
  [0. 0. 0.]
  [0. 0. 0.]
  [0. 0. 0.]]]
```


**[Celda 42 - Código]**
```python
a=vectores_elemento[1261]
b=vectores_consulta[77]
dist = np.sqrt(np.sum(np.square(a-b)))
print(dist)
```


*Salida:*
```text
988.7241
```


**[Celda 43 - Código]**
```python
#consultas[77][100]
imagenesBD[1261][100]
```


*Salida:*
```text
array([[1.        , 1.        , 1.        ],
       [1.        , 1.        , 1.        ],
       [1.        , 1.        , 1.        ],
       [1.        , 1.        , 1.        ],
       [0.98823529, 0.99215686, 0.98823529],
       [0.97254902, 0.97647059, 0.97647059],
       [0.94117647, 0.94509804, 0.94509804],
       [0.8745098 , 0.87843137, 0.87843137],
       [0.8627451 , 0.85882353, 0.84705882],
       [0.79215686, 0.77254902, 0.71764706],
       [0.58431373, 0.54117647, 0.42745098],
       [0.46666667, 0.38823529, 0.19215686],
       [0.51764706, 0.39607843, 0.0745098 ],
       [0.61960784, 0.45882353, 0.07843137],
       [0.58431373, 0.38039216, 0.0745098 ],
       [0.43921569, 0.18431373, 0.01960784],
       [0.34509804, 0.05882353, 0.        ],
       [0.34901961, 0.03529412, 0.        ],
       [0.38039216, 0.05098039, 0.03137255],
       [0.38823529, 0.05882353, 0.03921569],
       [0.38823529, 0.05490196, 0.03529412],
       [0.4       , 0.06666667, 0.04705882],
       [0.41176471, 0.06666667, 0.05098039],
       [0.41568627, 0.07058824, 0.05490196],
       [0.41568627, 0.0745098 , 0.05098039],
       [0.42352941, 0.07843137, 0.05098039],
       [0.42352941, 0.07843137, 0.05490196],
       [0.41568627, 0.06666667, 0.04313725],
       [0.41568627, 0.06666667, 0.04705882],
       [0.41176471, 0.06666667, 0.05098039],
       [0.41176471, 0.06666667, 0.05098039],
       [0.41568627, 0.07058824, 0.05490196],
       [0.41960784, 0.07058824, 0.05490196],
       [0.41568627, 0.06666667, 0.05490196],
       [0.41568627, 0.06666667, 0.05490196],
       [0.41568627, 0.07058824, 0.05490196],
       [0.41960784, 0.07058824, 0.05490196],
       [0.42745098, 0.07058824, 0.05882353],
       [0.42352941, 0.07058824, 0.05882353],
       [0.42352941, 0.07058824, 0.05882353],
       [0.42352941, 0.0745098 , 0.05882353],
       [0.43529412, 0.0745098 , 0.06666667],
       [0.43529412, 0.07843137, 0.06666667],
       [0.43137255, 0.08235294, 0.06666667],
       [0.42745098, 0.08235294, 0.0745098 ],
       [0.42745098, 0.08235294, 0.0745098 ],
       [0.42352941, 0.07843137, 0.07058824],
       [0.41960784, 0.0745098 , 0.0627451 ],
       [0.42745098, 0.08235294, 0.0745098 ],
       [0.43921569, 0.09019608, 0.0745098 ],
       [0.43921569, 0.09019608, 0.0745098 ],
       [0.43921569, 0.08627451, 0.07843137],
       [0.43921569, 0.08627451, 0.08235294],
       [0.43921569, 0.08627451, 0.07843137],
       [0.43529412, 0.09019608, 0.07843137],
       [0.43921569, 0.09411765, 0.08627451],
       [0.44313725, 0.09803922, 0.08627451],
       [0.44313725, 0.09803922, 0.09019608],
       [0.44705882, 0.10196078, 0.09019608],
       [0.45098039, 0.10588235, 0.09019608],
       [0.45098039, 0.10588235, 0.09019608],
       [0.45098039, 0.10196078, 0.09411765],
       [0.44313725, 0.09019608, 0.08627451],
       [0.43921569, 0.09019608, 0.08235294],
       [0.44705882, 0.10588235, 0.09019608],
       [0.45490196, 0.10588235, 0.09411765],
       [0.45882353, 0.11372549, 0.10196078],
       [0.46666667, 0.12156863, 0.10588235],
       [0.47058824, 0.1254902 , 0.11372549],
       [0.4745098 , 0.1254902 , 0.11372549],
       [0.47843137, 0.12941176, 0.11764706],
       [0.48627451, 0.1372549 , 0.12156863],
       [0.48627451, 0.1372549 , 0.1254902 ],
       [0.49019608, 0.14117647, 0.1254902 ],
       [0.49019608, 0.1372549 , 0.1254902 ],
       [0.48627451, 0.13333333, 0.12156863],
       [0.49019608, 0.13333333, 0.1254902 ],
       [0.49019608, 0.13333333, 0.12156863],
       [0.49411765, 0.1372549 , 0.1254902 ],
       [0.49803922, 0.14117647, 0.1254902 ],
       [0.49411765, 0.14117647, 0.12941176],
       [0.49803922, 0.14509804, 0.13333333],
       [0.50980392, 0.16078431, 0.14509804],
       [0.50196078, 0.15294118, 0.1372549 ],
       [0.50196078, 0.16078431, 0.14509804],
       [0.52941176, 0.21960784, 0.20784314],
       [0.56862745, 0.30196078, 0.29411765],
       [0.61960784, 0.37647059, 0.36862745],
       [0.68235294, 0.44313725, 0.43921569],
       [0.88627451, 0.66666667, 0.6627451 ],
       [1.        , 0.8627451 , 0.87058824],
       [1.        , 0.91764706, 0.91764706],
       [1.        , 0.90588235, 0.90196078],
       [0.99215686, 0.85098039, 0.85098039],
       [0.69411765, 0.50196078, 0.50196078],
       [0.45098039, 0.19215686, 0.18431373],
       [0.44313725, 0.13333333, 0.11764706],
       [0.50980392, 0.18039216, 0.16078431],
       [0.52941176, 0.18823529, 0.16862745],
       [0.51372549, 0.18823529, 0.17254902],
       [0.50196078, 0.22745098, 0.21568627],
       [0.7254902 , 0.51372549, 0.50980392],
       [0.99215686, 0.87843137, 0.87843137],
       [1.        , 0.99215686, 0.99607843],
       [0.95294118, 0.95294118, 0.95686275],
       [0.95294118, 0.94901961, 0.95294118],
       [0.96470588, 0.96470588, 0.96470588],
       [0.97254902, 0.97254902, 0.97647059],
       [0.98431373, 0.98431373, 0.99215686],
       [0.98823529, 0.98823529, 0.99607843],
       [0.99215686, 0.99215686, 0.99607843],
       [0.99607843, 0.99607843, 1.        ],
       [0.99607843, 0.99607843, 1.        ],
       [0.99607843, 0.99607843, 0.99607843],
       [0.99607843, 0.99607843, 0.99607843],
       [0.99607843, 0.99607843, 0.99607843],
       [1.        , 1.        , 1.        ],
       [1.        , 1.        , 1.        ],
       [1.        , 1.        , 1.        ],
       [1.        , 1.        , 1.        ],
       [1.        , 1.        , 1.        ],
       [1.        , 1.        , 1.        ],
       [1.        , 1.        , 1.        ],
       [1.        , 1.        , 1.        ],
       [1.        , 1.        , 1.        ],
       [1.        , 1.        , 1.        ],
       [1.        , 1.        , 1.        ],
       [1.        , 1.        , 1.        ],
       [1.        , 1.        , 1.        ],
       [1.        , 1.        , 1.        ],
       [0.99607843, 0.99607843, 0.99607843],
       [0.99607843, 0.99607843, 0.99607843],
       [0.98431373, 0.98039216, 0.98039216],
       [1.        , 1.        , 1.        ],
       [0.98039216, 0.90196078, 0.89803922],
       [0.78823529, 0.54901961, 0.54901961],
       [0.51764706, 0.22352941, 0.21176471],
       [0.54901961, 0.22352941, 0.20784314],
       [0.59215686, 0.24705882, 0.23137255],
       [0.59215686, 0.23921569, 0.22352941],
       [0.58823529, 0.24313725, 0.22352941],
       [0.58431373, 0.23529412, 0.21960784],
       [0.58823529, 0.23529412, 0.21960784],
       [0.59215686, 0.23529412, 0.22352941],
       [0.59215686, 0.23921569, 0.22352941],
       [0.58039216, 0.23529412, 0.22352941],
       [0.57254902, 0.21960784, 0.20784314],
       [0.56470588, 0.20784314, 0.19607843],
       [0.55686275, 0.19215686, 0.17647059],
       [0.55294118, 0.18431373, 0.17254902],
       [0.56078431, 0.18823529, 0.18823529],
       [0.56862745, 0.2       , 0.19607843],
       [0.56862745, 0.21176471, 0.20392157],
       [0.58039216, 0.22352941, 0.20784314],
       [0.57647059, 0.22745098, 0.20392157],
       [0.57254902, 0.22352941, 0.2       ],
       [0.57647059, 0.23137255, 0.20392157],
       [0.57647059, 0.23137255, 0.20784314],
       [0.57647059, 0.23529412, 0.20784314],
       [0.57254902, 0.22352941, 0.2       ],
       [0.56470588, 0.21568627, 0.19215686],
       [0.55686275, 0.20784314, 0.18431373],
       [0.55294118, 0.20392157, 0.18823529],
       [0.55294118, 0.20392157, 0.18823529],
       [0.54901961, 0.19607843, 0.18823529],
       [0.54509804, 0.19215686, 0.19215686],
       [0.55294118, 0.20392157, 0.19215686],
       [0.55294118, 0.20392157, 0.18823529],
       [0.54509804, 0.19607843, 0.17647059],
       [0.54509804, 0.18823529, 0.16862745],
       [0.5372549 , 0.18039216, 0.16078431],
       [0.52941176, 0.17647059, 0.15686275],
       [0.50980392, 0.16470588, 0.14117647],
       [0.49411765, 0.14117647, 0.12156863],
       [0.48627451, 0.12941176, 0.11764706],
       [0.48627451, 0.13333333, 0.12156863],
       [0.48235294, 0.12941176, 0.11372549],
       [0.48235294, 0.13333333, 0.11764706],
       [0.48627451, 0.1372549 , 0.12156863],
       [0.49019608, 0.14117647, 0.1254902 ],
       [0.49411765, 0.14509804, 0.12941176],
       [0.48627451, 0.1372549 , 0.12941176],
       [0.48627451, 0.13333333, 0.1254902 ],
       [0.48627451, 0.12941176, 0.12156863],
       [0.4745098 , 0.11764706, 0.10980392],
       [0.46666667, 0.10980392, 0.10196078],
       [0.4627451 , 0.11372549, 0.10196078],
       [0.4627451 , 0.11372549, 0.10196078],
       [0.4627451 , 0.11372549, 0.10196078],
       [0.4745098 , 0.1254902 , 0.10980392],
       [0.48235294, 0.13333333, 0.11764706],
       [0.48627451, 0.13333333, 0.1254902 ],
       [0.48235294, 0.12941176, 0.12156863],
       [0.48235294, 0.1254902 , 0.12156863],
       [0.4745098 , 0.11764706, 0.10980392],
       [0.45882353, 0.10980392, 0.09411765],
       [0.45098039, 0.10196078, 0.09019608],
       [0.45098039, 0.10588235, 0.09019608],
       [0.45098039, 0.10196078, 0.08627451],
       [0.44705882, 0.09803922, 0.08235294],
       [0.43921569, 0.09019608, 0.0745098 ],
       [0.43137255, 0.08627451, 0.07058824],
       [0.41960784, 0.0745098 , 0.05882353],
       [0.40784314, 0.0627451 , 0.04705882],
       [0.40392157, 0.06666667, 0.04705882],
       [0.40392157, 0.0745098 , 0.05882353],
       [0.40392157, 0.07843137, 0.0627451 ],
       [0.38039216, 0.06666667, 0.04313725],
       [0.33333333, 0.03529412, 0.00392157],
       [0.36862745, 0.09411765, 0.        ],
       [0.51372549, 0.28235294, 0.04705882],
       [0.60784314, 0.42745098, 0.08235294],
       [0.54117647, 0.40392157, 0.07058824],
       [0.47058824, 0.36470588, 0.1254902 ],
       [0.49803922, 0.42745098, 0.27058824],
       [0.70588235, 0.6627451 , 0.58431373],
       [0.84313725, 0.82352941, 0.80392157],
       [0.84705882, 0.84313725, 0.84705882],
       [0.92156863, 0.92156863, 0.92156863],
       [0.96078431, 0.96078431, 0.96078431],
       [0.98431373, 0.98431373, 0.98431373],
       [0.99607843, 0.99607843, 0.99607843],
       [1.        , 1.        , 1.        ],
       [1.        , 1.        , 1.        ]])
```


**[Celda 44 - Código]**
```python

```
