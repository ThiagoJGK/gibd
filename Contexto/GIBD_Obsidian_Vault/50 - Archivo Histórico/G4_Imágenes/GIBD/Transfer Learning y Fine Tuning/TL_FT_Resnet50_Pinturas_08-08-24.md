---
aliases: [TL_FT_Resnet50_Pinturas_08-08-24]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2024-08-08
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Transfer Learning y Fine Tuning/Pruebas 2024/TL_FT_Resnet50_Pinturas_08-08-24.ipynb"
tamanio_bytes: 118955549
---

# Notebook: TL_FT_Resnet50_Pinturas_08-08-24.ipynb

Ruta interna: `GIBD/Transfer Learning y Fine Tuning/Pruebas 2024/TL_FT_Resnet50_Pinturas_08-08-24.ipynb`

---

## Imports


**[Celda 2 - Código]**
```python
import os
print(os.listdir('G:\Mi unidad\GIBD'))
```


*Salida:*
```text
['Papers', 'Analisis de resultados de los intentos fallidos.gdoc', 'Trabajos parecidos', 'Publicaciones Few Shot Learning and Augumentation.gdoc', 'Tareas Iván', 'Copia de PresentacionesConaiisi2022.pptx', 'Presentacion Conaiisi 2022 - Articulo 210.pptx', 'Bases de Datos Mundiales de Prueba de ML.gdoc', 'Código útil', 'Herramientas', 'Tareas y Notas.gdoc', 'Procesamiento del Lenguaje Natural', 'Summary modelTriplet.txt', 'Calendario Congresos - Revistas.gsheet', 'ARTICULOS NUESTROS', 'E.C de imágenes a color', 'Presentaciones', 'Datos facturacion UTN FRCU.gdoc', 'Categorizacion UTN', 'Despedida GIBD.gsheet', 'BERT', 'Transfer Learning y Fine Tuning', 'CONAISII  clubes 2023', '2024', 'CNN Marcas', 'desktop.ini']
```


**[Celda 3 - Código]**
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


**[Celda 6 - Código]**
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


**[Celda 7 - Código]**
```python
def nnk(actuales,nuevo,k=5):
    p = 0
    while (p<len(actuales) and (nuevo[0]>=actuales[p][0])):
        p += 1
    actuales.insert(p, nuevo)
    while len(actuales)>k:
        actuales.pop(len(actuales)-1)
```


**[Celda 8 - Código]**
```python
def nnk2etiquetas(nnklist, y_BD):
    l = []
    for par in nnklist:
        nuevo = [par[0], y_BD[par[1]]]
        l.append(nuevo)
    return l
```


**[Celda 9 - Código]**
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


**[Celda 11 - Código]**
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


**[Celda 12 - Código]**
```python
### CARGAR Y NORMALIZAR LAS IMAGENES DE LA BD Y LAS CONSULTAS
from PIL import Image, ImageEnhance

target_size = (224, 224)  # El tamaño objetivo después de recortar y redimensionar

def crop_and_normalize_images(orig_dir, image_filenames, target_size):
    normalized_images = []
    etiquetas = []

    for filename in image_filenames:
        path = os.path.join(orig_dir, filename)
        image = Image.open(path)
        image = image.resize(target_size)
                
        # Convertir a 24 bits (RGB)
        image = image.convert('RGB')

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

        normalized_image = normalized_image / 255.0
        
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


**[Celda 13 - Código]**
```python
## CARGAR LAS RUTAS DE CADA IMAGEN DE CONSULTA
##orig_dir = r'/content/drive/MyDrive/GIBD/E.C de imágenes a color/E.C archivos/consultas_normalizadas'
##orig_dir = r'C:\Users\Usuario\Desktop\Logos\consultas_normalizadas'
orig_dir = r'G:\GIBD\Investigacion\Similitud Arte\Consultas'

query_paths = []

for filename in sorted(os.listdir(orig_dir)):
    if filename.lower().endswith('.jpg'):
        # Ruta completa del archivo
        query_paths.append(filename)


## NORMALIZAR LAS IMAGENES

consultas, y_consultas = crop_and_normalize_images(orig_dir, query_paths, target_size)
print(len(consultas))

```


*Salida:*
```text
0 223 0 223
Alfred_Sisley_110 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_84 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_95 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_114 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_172 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_26 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_5 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_52 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_94 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_36 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_53 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_64 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_81 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_119 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_17 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_28 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_69 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_77 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_53 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_84 :  (224, 224, 3)
0 223 0 223
Caravaggio_14 :  (224, 224, 3)
0 223 0 223
Claude_Monet_39 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_1 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_20 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_22 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_62 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_103 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_14 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_115 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_15 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_179 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_24 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_275 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_293 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_297 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_328 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_35 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_366 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_37 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_438 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_44 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_540 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_584 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_639 :  (224, 224, 3)
0 223 0 223
Kazimir_Malevich_100 :  (224, 224, 3)
0 223 0 223
Kazimir_Malevich_115 :  (224, 224, 3)
0 223 0 223
Kazimir_Malevich_124 :  (224, 224, 3)
0 223 0 223
Kazimir_Malevich_72 :  (224, 224, 3)
0 223 0 223
Leonardo_da_Vinci_11 :  (224, 224, 3)
0 223 0 223
Leonardo_da_Vinci_110 :  (224, 224, 3)
0 223 0 223
Leonardo_da_Vinci_121 :  (224, 224, 3)
0 223 0 223
Leonardo_da_Vinci_122 :  (224, 224, 3)
0 223 0 223
Leonardo_da_Vinci_124 :  (224, 224, 3)
0 223 0 223
Leonardo_da_Vinci_45 :  (224, 224, 3)
0 223 0 223
Leonardo_da_Vinci_89 :  (224, 224, 3)
0 223 0 223
Marc_Chagall_123 :  (224, 224, 3)
0 223 0 223
Marc_Chagall_152 :  (224, 224, 3)
0 223 0 223
Marc_Chagall_170 :  (224, 224, 3)
0 223 0 223
Marc_Chagall_178 :  (224, 224, 3)
0 223 0 223
Marc_Chagall_183 :  (224, 224, 3)
0 223 0 223
Marc_Chagall_2 :  (224, 224, 3)
0 223 0 223
Marc_Chagall_52 :  (224, 224, 3)
0 223 0 223
Marc_Chagall_56 :  (224, 224, 3)
0 223 0 222
Marc_Chagall_8 :  (224, 224, 3)
0 223 0 223
Marc_Chagall_92 :  (224, 224, 3)
0 223 0 223
Michelangelo_16 :  (224, 224, 3)
0 223 0 223
Michelangelo_21 :  (224, 224, 3)
0 223 0 223
Mikhail_Vrubel_140 :  (224, 224, 3)
0 223 0 223
Mikhail_Vrubel_16 :  (224, 224, 3)
0 223 0 223
Mikhail_Vrubel_36 :  (224, 224, 3)
0 223 0 223
Mikhail_Vrubel_5 :  (224, 224, 3)
0 223 0 223
Mikhail_Vrubel_67 :  (224, 224, 3)
0 223 0 223
Mikhail_Vrubel_96 :  (224, 224, 3)
0 223 0 223
Pablo_Picasso_249 :  (224, 224, 3)
0 223 0 223
Pablo_Picasso_305 :  (224, 224, 3)
0 223 0 223
Pablo_Picasso_322 :  (224, 224, 3)
0 223 0 223
Pablo_Picasso_381 :  (224, 224, 3)
0 223 0 223
Pablo_Picasso_438 :  (224, 224, 3)
0 223 0 223
Pablo_Picasso_46 :  (224, 224, 3)
0 223 0 223
Paul_Gauguin_107 :  (224, 224, 3)
0 223 0 223
Paul_Gauguin_116 :  (224, 224, 3)
0 223 0 223
Paul_Gauguin_290 :  (224, 224, 3)
0 223 0 223
Paul_Gauguin_49 :  (224, 224, 3)
0 223 0 223
Paul_Klee_1 :  (224, 224, 3)
0 223 0 223
Paul_Klee_14 :  (224, 224, 3)
0 223 0 223
Paul_Klee_23 :  (224, 224, 3)
0 223 0 223
Paul_Klee_38 :  (224, 224, 3)
0 223 0 223
Pierre-Auguste_Renoir_142 :  (224, 224, 3)
0 223 0 223
Pierre-Auguste_Renoir_257 :  (224, 224, 3)
0 223 0 223
Rene_Magritte_138 :  (224, 224, 3)
0 223 0 223
Salvador_Dali_35 :  (224, 224, 3)
0 223 0 223
Salvador_Dali_61 :  (224, 224, 3)
0 223 0 223
Salvador_Dali_8 :  (224, 224, 3)
0 223 0 223
Sandro_Botticelli_74 :  (224, 224, 3)
0 223 0 223
Titian_81 :  (224, 224, 3)
0 223 0 223
Vincent_van_Gogh_276 :  (224, 224, 3)
0 223 0 223
Vincent_van_Gogh_281 :  (224, 224, 3)
0 223 0 223
Vincent_van_Gogh_309 :  (224, 224, 3)
0 223 0 223
Vincent_van_Gogh_645 :  (224, 224, 3)
0 223 0 223
William_Turner_65 :  (224, 224, 3)
(224, 224, 3)
Forma del arreglo combinado: (100, 224, 224, 3)
100
```


**[Celda 14 - Código]**
```python
#  CARGAR BASE DE DATOS
## CARGAR LAS RUTAS DE CADA IMAGEN DE LA BASE DE DATOS
##orig_dir = r'/content/drive/MyDrive/GIBD/CNN Marcas/Bases de Datos/Escudos Normalizados PNG'
orig_dir = r'G:\GIBD\Investigacion\Similitud Arte\BD'

image_paths = []
i = 0
for filename in sorted(os.listdir(orig_dir)):
    if filename.lower().endswith('.jpg'):
              
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
['Albrecht_DuÔòá├¬rer_1.jpg', 'Albrecht_DuÔòá├¬rer_10.jpg', 'Albrecht_DuÔòá├¬rer_100.jpg', 'Albrecht_DuÔòá├¬rer_101.jpg', 'Albrecht_DuÔòá├¬rer_102.jpg', 'Albrecht_DuÔòá├¬rer_103.jpg', 'Albrecht_DuÔòá├¬rer_104.jpg', 'Albrecht_DuÔòá├¬rer_105.jpg', 'Albrecht_DuÔòá├¬rer_106.jpg', 'Albrecht_DuÔòá├¬rer_107.jpg', 'Albrecht_DuÔòá├¬rer_108.jpg', 'Albrecht_DuÔòá├¬rer_109.jpg', 'Albrecht_DuÔòá├¬rer_11.jpg', 'Albrecht_DuÔòá├¬rer_110.jpg', 'Albrecht_DuÔòá├¬rer_111.jpg', 'Albrecht_DuÔòá├¬rer_112.jpg', 'Albrecht_DuÔòá├¬rer_113.jpg', 'Albrecht_DuÔòá├¬rer_114.jpg', 'Albrecht_DuÔòá├¬rer_115.jpg', 'Albrecht_DuÔòá├¬rer_116.jpg', 'Albrecht_DuÔòá├¬rer_117.jpg', 'Albrecht_DuÔòá├¬rer_118.jpg', 'Albrecht_DuÔòá├¬rer_119.jpg', 'Albrecht_DuÔòá├¬rer_12.jpg', 'Albrecht_DuÔòá├¬rer_120.jpg', 'Albrecht_DuÔòá├¬rer_121.jpg', 'Albrecht_DuÔòá├¬rer_122.jpg', 'Albrecht_DuÔòá├¬rer_123.jpg', 'Albrecht_DuÔòá├¬rer_124.jpg', 'Albrecht_DuÔòá├¬rer_125.jpg', 'Albrecht_DuÔòá├¬rer_126.jpg', 'Albrecht_DuÔòá├¬rer_127.jpg', 'Albrecht_DuÔòá├¬rer_128.jpg', 'Albrecht_DuÔòá├¬rer_129.jpg', 'Albrecht_DuÔòá├¬rer_13.jpg', 'Albrecht_DuÔòá├¬rer_130.jpg', 'Albrecht_DuÔòá├¬rer_131.jpg', 'Albrecht_DuÔòá├¬rer_132.jpg', 'Albrecht_DuÔòá├¬rer_133.jpg', 'Albrecht_DuÔòá├¬rer_134.jpg', 'Albrecht_DuÔòá├¬rer_135.jpg', 'Albrecht_DuÔòá├¬rer_136.jpg', 'Albrecht_DuÔòá├¬rer_137.jpg', 'Albrecht_DuÔòá├¬rer_138.jpg', 'Albrecht_DuÔòá├¬rer_139.jpg', 'Albrecht_DuÔòá├¬rer_14.jpg', 'Albrecht_DuÔòá├¬rer_140.jpg', 'Albrecht_DuÔòá├¬rer_141.jpg', 'Albrecht_DuÔòá├¬rer_142.jpg', 'Albrecht_DuÔòá├¬rer_143.jpg', 'Albrecht_DuÔòá├¬rer_144.jpg', 'Albrecht_DuÔòá├¬rer_145.jpg', 'Albrecht_DuÔòá├¬rer_146.jpg', 'Albrecht_DuÔòá├¬rer_147.jpg', 'Albrecht_DuÔòá├¬rer_148.jpg', 'Albrecht_DuÔòá├¬rer_149.jpg', 'Albrecht_DuÔòá├¬rer_15.jpg', 'Albrecht_DuÔòá├¬rer_150.jpg', 'Albrecht_DuÔòá├¬rer_151.jpg', 'Albrecht_DuÔòá├¬rer_152.jpg', 'Albrecht_DuÔòá├¬rer_153.jpg', 'Albrecht_DuÔòá├¬rer_154.jpg', 'Albrecht_DuÔòá├¬rer_155.jpg', 'Albrecht_DuÔòá├¬rer_156.jpg', 'Albrecht_DuÔòá├¬rer_157.jpg', 'Albrecht_DuÔòá├¬rer_158.jpg', 'Albrecht_DuÔòá├¬rer_159.jpg', 'Albrecht_DuÔòá├¬rer_16.jpg', 'Albrecht_DuÔòá├¬rer_160.jpg', 'Albrecht_DuÔòá├¬rer_161.jpg', 'Albrecht_DuÔòá├¬rer_162.jpg', 'Albrecht_DuÔòá├¬rer_163.jpg', 'Albrecht_DuÔòá├¬rer_164.jpg', 'Albrecht_DuÔòá├¬rer_165.jpg', 'Albrecht_DuÔòá├¬rer_166.jpg', 'Albrecht_DuÔòá├¬rer_167.jpg', 'Albrecht_DuÔòá├¬rer_168.jpg', 'Albrecht_DuÔòá├¬rer_169.jpg', 'Albrecht_DuÔòá├¬rer_17.jpg', 'Albrecht_DuÔòá├¬rer_170.jpg', 'Albrecht_DuÔòá├¬rer_171.jpg', 'Albrecht_DuÔòá├¬rer_172.jpg', 'Albrecht_DuÔòá├¬rer_173.jpg', 'Albrecht_DuÔòá├¬rer_174.jpg', 'Albrecht_DuÔòá├¬rer_175.jpg', 'Albrecht_DuÔòá├¬rer_176.jpg', 'Albrecht_DuÔòá├¬rer_177.jpg', 'Albrecht_DuÔòá├¬rer_178.jpg', 'Albrecht_DuÔòá├¬rer_179.jpg', 'Albrecht_DuÔòá├¬rer_18.jpg', 'Albrecht_DuÔòá├¬rer_180.jpg', 'Albrecht_DuÔòá├¬rer_181.jpg', 'Albrecht_DuÔòá├¬rer_182.jpg', 'Albrecht_DuÔòá├¬rer_183.jpg', 'Albrecht_DuÔòá├¬rer_184.jpg', 'Albrecht_DuÔòá├¬rer_185.jpg', 'Albrecht_DuÔòá├¬rer_186.jpg', 'Albrecht_DuÔòá├¬rer_187.jpg', 'Albrecht_DuÔòá├¬rer_188.jpg', 'Albrecht_DuÔòá├¬rer_189.jpg', 'Albrecht_DuÔòá├¬rer_19.jpg', 'Albrecht_DuÔòá├¬rer_190.jpg', 'Albrecht_DuÔòá├¬rer_191.jpg', 'Albrecht_DuÔòá├¬rer_192.jpg', 'Albrecht_DuÔòá├¬rer_193.jpg', 'Albrecht_DuÔòá├¬rer_194.jpg', 'Albrecht_DuÔòá├¬rer_195.jpg', 'Albrecht_DuÔòá├¬rer_196.jpg', 'Albrecht_DuÔòá├¬rer_197.jpg', 'Albrecht_DuÔòá├¬rer_198.jpg', 'Albrecht_DuÔòá├¬rer_199.jpg', 'Albrecht_DuÔòá├¬rer_2.jpg', 'Albrecht_DuÔòá├¬rer_20.jpg', 'Albrecht_DuÔòá├¬rer_200.jpg', 'Albrecht_DuÔòá├¬rer_201.jpg', 'Albrecht_DuÔòá├¬rer_202.jpg', 'Albrecht_DuÔòá├¬rer_203.jpg', 'Albrecht_DuÔòá├¬rer_204.jpg', 'Albrecht_DuÔòá├¬rer_205.jpg', 'Albrecht_DuÔòá├¬rer_206.jpg', 'Albrecht_DuÔòá├¬rer_207.jpg', 'Albrecht_DuÔòá├¬rer_208.jpg', 'Albrecht_DuÔòá├¬rer_209.jpg', 'Albrecht_DuÔòá├¬rer_21.jpg', 'Albrecht_DuÔòá├¬rer_210.jpg', 'Albrecht_DuÔòá├¬rer_211.jpg', 'Albrecht_DuÔòá├¬rer_212.jpg', 'Albrecht_DuÔòá├¬rer_213.jpg', 'Albrecht_DuÔòá├¬rer_214.jpg', 'Albrecht_DuÔòá├¬rer_215.jpg', 'Albrecht_DuÔòá├¬rer_216.jpg', 'Albrecht_DuÔòá├¬rer_217.jpg', 'Albrecht_DuÔòá├¬rer_218.jpg', 'Albrecht_DuÔòá├¬rer_219.jpg', 'Albrecht_DuÔòá├¬rer_22.jpg', 'Albrecht_DuÔòá├¬rer_220.jpg', 'Albrecht_DuÔòá├¬rer_221.jpg', 'Albrecht_DuÔòá├¬rer_222.jpg', 'Albrecht_DuÔòá├¬rer_223.jpg', 'Albrecht_DuÔòá├¬rer_224.jpg', 'Albrecht_DuÔòá├¬rer_225.jpg', 'Albrecht_DuÔòá├¬rer_226.jpg', 'Albrecht_DuÔòá├¬rer_227.jpg', 'Albrecht_DuÔòá├¬rer_228.jpg', 'Albrecht_DuÔòá├¬rer_229.jpg', 'Albrecht_DuÔòá├¬rer_23.jpg', 'Albrecht_DuÔòá├¬rer_230.jpg', 'Albrecht_DuÔòá├¬rer_231.jpg', 'Albrecht_DuÔòá├¬rer_232.jpg', 'Albrecht_DuÔòá├¬rer_233.jpg', 'Albrecht_DuÔòá├¬rer_234.jpg', 'Albrecht_DuÔòá├¬rer_235.jpg', 'Albrecht_DuÔòá├¬rer_236.jpg', 'Albrecht_DuÔòá├¬rer_237.jpg', 'Albrecht_DuÔòá├¬rer_238.jpg', 'Albrecht_DuÔòá├¬rer_239.jpg', 'Albrecht_DuÔòá├¬rer_24.jpg', 'Albrecht_DuÔòá├¬rer_240.jpg', 'Albrecht_DuÔòá├¬rer_241.jpg', 'Albrecht_DuÔòá├¬rer_242.jpg', 'Albrecht_DuÔòá├¬rer_243.jpg', 'Albrecht_DuÔòá├¬rer_244.jpg', 'Albrecht_DuÔòá├¬rer_245.jpg', 'Albrecht_DuÔòá├¬rer_246.jpg', 'Albrecht_DuÔòá├¬rer_247.jpg', 'Albrecht_DuÔòá├¬rer_248.jpg', 'Albrecht_DuÔòá├¬rer_249.jpg', 'Albrecht_DuÔòá├¬rer_25.jpg', 'Albrecht_DuÔòá├¬rer_250.jpg', 'Albrecht_DuÔòá├¬rer_251.jpg', 'Albrecht_DuÔòá├¬rer_252.jpg', 'Albrecht_DuÔòá├¬rer_253.jpg', 'Albrecht_DuÔòá├¬rer_254.jpg', 'Albrecht_DuÔòá├¬rer_255.jpg', 'Albrecht_DuÔòá├¬rer_256.jpg', 'Albrecht_DuÔòá├¬rer_257.jpg', 'Albrecht_DuÔòá├¬rer_258.jpg', 'Albrecht_DuÔòá├¬rer_259.jpg', 'Albrecht_DuÔòá├¬rer_26.jpg', 'Albrecht_DuÔòá├¬rer_260.jpg', 'Albrecht_DuÔòá├¬rer_261.jpg', 'Albrecht_DuÔòá├¬rer_262.jpg', 'Albrecht_DuÔòá├¬rer_263.jpg', 'Albrecht_DuÔòá├¬rer_264.jpg', 'Albrecht_DuÔòá├¬rer_265.jpg', 'Albrecht_DuÔòá├¬rer_266.jpg', 'Albrecht_DuÔòá├¬rer_267.jpg', 'Albrecht_DuÔòá├¬rer_268.jpg', 'Albrecht_DuÔòá├¬rer_269.jpg', 'Albrecht_DuÔòá├¬rer_27.jpg', 'Albrecht_DuÔòá├¬rer_270.jpg', 'Albrecht_DuÔòá├¬rer_271.jpg', 'Albrecht_DuÔòá├¬rer_272.jpg', 'Albrecht_DuÔòá├¬rer_273.jpg', 'Albrecht_DuÔòá├¬rer_274.jpg', 'Albrecht_DuÔòá├¬rer_275.jpg', 'Albrecht_DuÔòá├¬rer_276.jpg', 'Albrecht_DuÔòá├¬rer_277.jpg', 'Albrecht_DuÔòá├¬rer_278.jpg', 'Albrecht_DuÔòá├¬rer_279.jpg', 'Albrecht_DuÔòá├¬rer_28.jpg', 'Albrecht_DuÔòá├¬rer_280.jpg', 'Albrecht_DuÔòá├¬rer_281.jpg', 'Albrecht_DuÔòá├¬rer_282.jpg', 'Albrecht_DuÔòá├¬rer_283.jpg', 'Albrecht_DuÔòá├¬rer_284.jpg', 'Albrecht_DuÔòá├¬rer_285.jpg', 'Albrecht_DuÔòá├¬rer_286.jpg', 'Albrecht_DuÔòá├¬rer_287.jpg', 'Albrecht_DuÔòá├¬rer_288.jpg', 'Albrecht_DuÔòá├¬rer_289.jpg', 'Albrecht_DuÔòá├¬rer_29.jpg', 'Albrecht_DuÔòá├¬rer_290.jpg', 'Albrecht_DuÔòá├¬rer_291.jpg', 'Albrecht_DuÔòá├¬rer_292.jpg', 'Albrecht_DuÔòá├¬rer_293.jpg', 'Albrecht_DuÔòá├¬rer_294.jpg', 'Albrecht_DuÔòá├¬rer_295.jpg', 'Albrecht_DuÔòá├¬rer_296.jpg', 'Albrecht_DuÔòá├¬rer_297.jpg', 'Albrecht_DuÔòá├¬rer_298.jpg', 'Albrecht_DuÔòá├¬rer_299.jpg', 'Albrecht_DuÔòá├¬rer_3.jpg', 'Albrecht_DuÔòá├¬rer_30.jpg', 'Albrecht_DuÔòá├¬rer_300.jpg', 'Albrecht_DuÔòá├¬rer_301.jpg', 'Albrecht_DuÔòá├¬rer_302.jpg', 'Albrecht_DuÔòá├¬rer_303.jpg', 'Albrecht_DuÔòá├¬rer_304.jpg', 'Albrecht_DuÔòá├¬rer_305.jpg', 'Albrecht_DuÔòá├¬rer_306.jpg', 'Albrecht_DuÔòá├¬rer_307.jpg', 'Albrecht_DuÔòá├¬rer_308.jpg', 'Albrecht_DuÔòá├¬rer_309.jpg', 'Albrecht_DuÔòá├¬rer_31.jpg', 'Albrecht_DuÔòá├¬rer_310.jpg', 'Albrecht_DuÔòá├¬rer_311.jpg', 'Albrecht_DuÔòá├¬rer_312.jpg', 'Albrecht_DuÔòá├¬rer_313.jpg', 'Albrecht_DuÔòá├¬rer_314.jpg', 'Albrecht_DuÔòá├¬rer_315.jpg', 'Albrecht_DuÔòá├¬rer_316.jpg', 'Albrecht_DuÔòá├¬rer_317.jpg', 'Albrecht_DuÔòá├¬rer_318.jpg', 'Albrecht_DuÔòá├¬rer_319.jpg', 'Albrecht_DuÔòá├¬rer_32.jpg', 'Albrecht_DuÔòá├¬rer_320.jpg', 'Albrecht_DuÔòá├¬rer_321.jpg', 'Albrecht_DuÔòá├¬rer_322.jpg', 'Albrecht_DuÔòá├¬rer_323.jpg', 'Albrecht_DuÔòá├¬rer_324.jpg', 'Albrecht_DuÔòá├¬rer_325.jpg', 'Albrecht_DuÔòá├¬rer_326.jpg', 'Albrecht_DuÔòá├¬rer_327.jpg', 'Albrecht_DuÔòá├¬rer_328.jpg', 'Albrecht_DuÔòá├¬rer_33.jpg', 'Albrecht_DuÔòá├¬rer_34.jpg', 'Albrecht_DuÔòá├¬rer_35.jpg', 'Albrecht_DuÔòá├¬rer_36.jpg', 'Albrecht_DuÔòá├¬rer_37.jpg', 'Albrecht_DuÔòá├¬rer_38.jpg', 'Albrecht_DuÔòá├¬rer_39.jpg', 'Albrecht_DuÔòá├¬rer_4.jpg', 'Albrecht_DuÔòá├¬rer_40.jpg', 'Albrecht_DuÔòá├¬rer_41.jpg', 'Albrecht_DuÔòá├¬rer_42.jpg', 'Albrecht_DuÔòá├¬rer_43.jpg', 'Albrecht_DuÔòá├¬rer_44.jpg', 'Albrecht_DuÔòá├¬rer_45.jpg', 'Albrecht_DuÔòá├¬rer_46.jpg', 'Albrecht_DuÔòá├¬rer_47.jpg', 'Albrecht_DuÔòá├¬rer_48.jpg', 'Albrecht_DuÔòá├¬rer_49.jpg', 'Albrecht_DuÔòá├¬rer_5.jpg', 'Albrecht_DuÔòá├¬rer_50.jpg', 'Albrecht_DuÔòá├¬rer_51.jpg', 'Albrecht_DuÔòá├¬rer_52.jpg', 'Albrecht_DuÔòá├¬rer_53.jpg', 'Albrecht_DuÔòá├¬rer_54.jpg', 'Albrecht_DuÔòá├¬rer_55.jpg', 'Albrecht_DuÔòá├¬rer_56.jpg', 'Albrecht_DuÔòá├¬rer_57.jpg', 'Albrecht_DuÔòá├¬rer_58.jpg', 'Albrecht_DuÔòá├¬rer_59.jpg', 'Albrecht_DuÔòá├¬rer_6.jpg', 'Albrecht_DuÔòá├¬rer_60.jpg', 'Albrecht_DuÔòá├¬rer_61.jpg', 'Albrecht_DuÔòá├¬rer_62.jpg', 'Albrecht_DuÔòá├¬rer_63.jpg', 'Albrecht_DuÔòá├¬rer_64.jpg', 'Albrecht_DuÔòá├¬rer_65.jpg', 'Albrecht_DuÔòá├¬rer_66.jpg', 'Albrecht_DuÔòá├¬rer_67.jpg', 'Albrecht_DuÔòá├¬rer_68.jpg', 'Albrecht_DuÔòá├¬rer_69.jpg', 'Albrecht_DuÔòá├¬rer_7.jpg', 'Albrecht_DuÔòá├¬rer_70.jpg', 'Albrecht_DuÔòá├¬rer_71.jpg', 'Albrecht_DuÔòá├¬rer_72.jpg', 'Albrecht_DuÔòá├¬rer_73.jpg', 'Albrecht_DuÔòá├¬rer_74.jpg', 'Albrecht_DuÔòá├¬rer_75.jpg', 'Albrecht_DuÔòá├¬rer_76.jpg', 'Albrecht_DuÔòá├¬rer_77.jpg', 'Albrecht_DuÔòá├¬rer_78.jpg', 'Albrecht_DuÔòá├¬rer_79.jpg', 'Albrecht_DuÔòá├¬rer_8.jpg', 'Albrecht_DuÔòá├¬rer_80.jpg', 'Albrecht_DuÔòá├¬rer_81.jpg', 'Albrecht_DuÔòá├¬rer_82.jpg', 'Albrecht_DuÔòá├¬rer_83.jpg', 'Albrecht_DuÔòá├¬rer_84.jpg', 'Albrecht_DuÔòá├¬rer_85.jpg', 'Albrecht_DuÔòá├¬rer_86.jpg', 'Albrecht_DuÔòá├¬rer_87.jpg', 'Albrecht_DuÔòá├¬rer_88.jpg', 'Albrecht_DuÔòá├¬rer_89.jpg', 'Albrecht_DuÔòá├¬rer_9.jpg', 'Albrecht_DuÔòá├¬rer_90.jpg', 'Albrecht_DuÔòá├¬rer_91.jpg', 'Albrecht_DuÔòá├¬rer_92.jpg', 'Albrecht_DuÔòá├¬rer_93.jpg', 'Albrecht_DuÔòá├¬rer_94.jpg', 'Albrecht_DuÔòá├¬rer_95.jpg', 'Albrecht_DuÔòá├¬rer_96.jpg', 'Albrecht_DuÔòá├¬rer_97.jpg', 'Albrecht_DuÔòá├¬rer_98.jpg', 'Albrecht_DuÔòá├¬rer_99.jpg', 'Albrecht_Du╠êrer_1.jpg', 'Albrecht_Du╠êrer_10.jpg', 'Albrecht_Du╠êrer_100.jpg', 'Albrecht_Du╠êrer_101.jpg', 'Albrecht_Du╠êrer_102.jpg', 'Albrecht_Du╠êrer_103.jpg', 'Albrecht_Du╠êrer_104.jpg', 'Albrecht_Du╠êrer_105.jpg', 'Albrecht_Du╠êrer_106.jpg', 'Albrecht_Du╠êrer_107.jpg', 'Albrecht_Du╠êrer_108.jpg', 'Albrecht_Du╠êrer_109.jpg', 'Albrecht_Du╠êrer_11.jpg', 'Albrecht_Du╠êrer_110.jpg', 'Albrecht_Du╠êrer_111.jpg', 'Albrecht_Du╠êrer_112.jpg', 'Albrecht_Du╠êrer_113.jpg', 'Albrecht_Du╠êrer_114.jpg', 'Albrecht_Du╠êrer_115.jpg', 'Albrecht_Du╠êrer_116.jpg', 'Albrecht_Du╠êrer_117.jpg', 'Albrecht_Du╠êrer_118.jpg', 'Albrecht_Du╠êrer_119.jpg', 'Albrecht_Du╠êrer_12.jpg', 'Albrecht_Du╠êrer_120.jpg', 'Albrecht_Du╠êrer_121.jpg', 'Albrecht_Du╠êrer_122.jpg', 'Albrecht_Du╠êrer_123.jpg', 'Albrecht_Du╠êrer_124.jpg', 'Albrecht_Du╠êrer_125.jpg', 'Albrecht_Du╠êrer_126.jpg', 'Albrecht_Du╠êrer_127.jpg', 'Albrecht_Du╠êrer_128.jpg', 'Albrecht_Du╠êrer_129.jpg', 'Albrecht_Du╠êrer_13.jpg', 'Albrecht_Du╠êrer_130.jpg', 'Albrecht_Du╠êrer_131.jpg', 'Albrecht_Du╠êrer_132.jpg', 'Albrecht_Du╠êrer_133.jpg', 'Albrecht_Du╠êrer_134.jpg', 'Albrecht_Du╠êrer_135.jpg', 'Albrecht_Du╠êrer_136.jpg', 'Albrecht_Du╠êrer_137.jpg', 'Albrecht_Du╠êrer_138.jpg', 'Albrecht_Du╠êrer_139.jpg', 'Albrecht_Du╠êrer_14.jpg', 'Albrecht_Du╠êrer_140.jpg', 'Albrecht_Du╠êrer_141.jpg', 'Albrecht_Du╠êrer_142.jpg', 'Albrecht_Du╠êrer_143.jpg', 'Albrecht_Du╠êrer_144.jpg', 'Albrecht_Du╠êrer_145.jpg', 'Albrecht_Du╠êrer_146.jpg', 'Albrecht_Du╠êrer_147.jpg', 'Albrecht_Du╠êrer_148.jpg', 'Albrecht_Du╠êrer_149.jpg', 'Albrecht_Du╠êrer_15.jpg', 'Albrecht_Du╠êrer_150.jpg', 'Albrecht_Du╠êrer_151.jpg', 'Albrecht_Du╠êrer_152.jpg', 'Albrecht_Du╠êrer_153.jpg', 'Albrecht_Du╠êrer_154.jpg', 'Albrecht_Du╠êrer_155.jpg', 'Albrecht_Du╠êrer_156.jpg', 'Albrecht_Du╠êrer_157.jpg', 'Albrecht_Du╠êrer_158.jpg', 'Albrecht_Du╠êrer_159.jpg', 'Albrecht_Du╠êrer_16.jpg', 'Albrecht_Du╠êrer_160.jpg', 'Albrecht_Du╠êrer_161.jpg', 'Albrecht_Du╠êrer_162.jpg', 'Albrecht_Du╠êrer_163.jpg', 'Albrecht_Du╠êrer_164.jpg', 'Albrecht_Du╠êrer_165.jpg', 'Albrecht_Du╠êrer_166.jpg', 'Albrecht_Du╠êrer_167.jpg', 'Albrecht_Du╠êrer_168.jpg', 'Albrecht_Du╠êrer_169.jpg', 'Albrecht_Du╠êrer_17.jpg', 'Albrecht_Du╠êrer_170.jpg', 'Albrecht_Du╠êrer_171.jpg', 'Albrecht_Du╠êrer_172.jpg', 'Albrecht_Du╠êrer_173.jpg', 'Albrecht_Du╠êrer_174.jpg', 'Albrecht_Du╠êrer_175.jpg', 'Albrecht_Du╠êrer_176.jpg', 'Albrecht_Du╠êrer_177.jpg', 'Albrecht_Du╠êrer_178.jpg', 'Albrecht_Du╠êrer_179.jpg', 'Albrecht_Du╠êrer_18.jpg', 'Albrecht_Du╠êrer_180.jpg', 'Albrecht_Du╠êrer_181.jpg', 'Albrecht_Du╠êrer_182.jpg', 'Albrecht_Du╠êrer_183.jpg', 'Albrecht_Du╠êrer_184.jpg', 'Albrecht_Du╠êrer_185.jpg', 'Albrecht_Du╠êrer_186.jpg', 'Albrecht_Du╠êrer_187.jpg', 'Albrecht_Du╠êrer_188.jpg', 'Albrecht_Du╠êrer_189.jpg', 'Albrecht_Du╠êrer_19.jpg', 'Albrecht_Du╠êrer_190.jpg', 'Albrecht_Du╠êrer_191.jpg', 'Albrecht_Du╠êrer_192.jpg', 'Albrecht_Du╠êrer_193.jpg', 'Albrecht_Du╠êrer_194.jpg', 'Albrecht_Du╠êrer_195.jpg', 'Albrecht_Du╠êrer_196.jpg', 'Albrecht_Du╠êrer_197.jpg', 'Albrecht_Du╠êrer_198.jpg', 'Albrecht_Du╠êrer_199.jpg', 'Albrecht_Du╠êrer_2.jpg', 'Albrecht_Du╠êrer_20.jpg', 'Albrecht_Du╠êrer_200.jpg', 'Albrecht_Du╠êrer_201.jpg', 'Albrecht_Du╠êrer_202.jpg', 'Albrecht_Du╠êrer_203.jpg', 'Albrecht_Du╠êrer_204.jpg', 'Albrecht_Du╠êrer_205.jpg', 'Albrecht_Du╠êrer_206.jpg', 'Albrecht_Du╠êrer_207.jpg', 'Albrecht_Du╠êrer_208.jpg', 'Albrecht_Du╠êrer_209.jpg', 'Albrecht_Du╠êrer_21.jpg', 'Albrecht_Du╠êrer_210.jpg', 'Albrecht_Du╠êrer_211.jpg', 'Albrecht_Du╠êrer_212.jpg', 'Albrecht_Du╠êrer_213.jpg', 'Albrecht_Du╠êrer_214.jpg', 'Albrecht_Du╠êrer_215.jpg', 'Albrecht_Du╠êrer_216.jpg', 'Albrecht_Du╠êrer_217.jpg', 'Albrecht_Du╠êrer_218.jpg', 'Albrecht_Du╠êrer_219.jpg', 'Albrecht_Du╠êrer_22.jpg', 'Albrecht_Du╠êrer_220.jpg', 'Albrecht_Du╠êrer_221.jpg', 'Albrecht_Du╠êrer_222.jpg', 'Albrecht_Du╠êrer_223.jpg', 'Albrecht_Du╠êrer_224.jpg', 'Albrecht_Du╠êrer_225.jpg', 'Albrecht_Du╠êrer_226.jpg', 'Albrecht_Du╠êrer_227.jpg', 'Albrecht_Du╠êrer_228.jpg', 'Albrecht_Du╠êrer_229.jpg', 'Albrecht_Du╠êrer_23.jpg', 'Albrecht_Du╠êrer_230.jpg', 'Albrecht_Du╠êrer_231.jpg', 'Albrecht_Du╠êrer_232.jpg', 'Albrecht_Du╠êrer_233.jpg', 'Albrecht_Du╠êrer_234.jpg', 'Albrecht_Du╠êrer_235.jpg', 'Albrecht_Du╠êrer_236.jpg', 'Albrecht_Du╠êrer_237.jpg', 'Albrecht_Du╠êrer_238.jpg', 'Albrecht_Du╠êrer_239.jpg', 'Albrecht_Du╠êrer_24.jpg', 'Albrecht_Du╠êrer_240.jpg', 'Albrecht_Du╠êrer_241.jpg', 'Albrecht_Du╠êrer_242.jpg', 'Albrecht_Du╠êrer_243.jpg', 'Albrecht_Du╠êrer_244.jpg', 'Albrecht_Du╠êrer_245.jpg', 'Albrecht_Du╠êrer_246.jpg', 'Albrecht_Du╠êrer_247.jpg', 'Albrecht_Du╠êrer_248.jpg', 'Albrecht_Du╠êrer_249.jpg', 'Albrecht_Du╠êrer_25.jpg', 'Albrecht_Du╠êrer_250.jpg', 'Albrecht_Du╠êrer_251.jpg', 'Albrecht_Du╠êrer_252.jpg', 'Albrecht_Du╠êrer_253.jpg', 'Albrecht_Du╠êrer_254.jpg', 'Albrecht_Du╠êrer_255.jpg', 'Albrecht_Du╠êrer_256.jpg', 'Albrecht_Du╠êrer_257.jpg', 'Albrecht_Du╠êrer_258.jpg', 'Albrecht_Du╠êrer_259.jpg', 'Albrecht_Du╠êrer_26.jpg', 'Albrecht_Du╠êrer_260.jpg', 'Albrecht_Du╠êrer_261.jpg', 'Albrecht_Du╠êrer_262.jpg', 'Albrecht_Du╠êrer_263.jpg', 'Albrecht_Du╠êrer_264.jpg', 'Albrecht_Du╠êrer_265.jpg', 'Albrecht_Du╠êrer_266.jpg', 'Albrecht_Du╠êrer_267.jpg', 'Albrecht_Du╠êrer_268.jpg', 'Albrecht_Du╠êrer_269.jpg', 'Albrecht_Du╠êrer_27.jpg', 'Albrecht_Du╠êrer_270.jpg', 'Albrecht_Du╠êrer_271.jpg', 'Albrecht_Du╠êrer_272.jpg', 'Albrecht_Du╠êrer_273.jpg', 'Albrecht_Du╠êrer_274.jpg', 'Albrecht_Du╠êrer_275.jpg', 'Albrecht_Du╠êrer_276.jpg', 'Albrecht_Du╠êrer_277.jpg', 'Albrecht_Du╠êrer_278.jpg', 'Albrecht_Du╠êrer_279.jpg', 'Albrecht_Du╠êrer_28.jpg', 'Albrecht_Du╠êrer_280.jpg', 'Albrecht_Du╠êrer_281.jpg', 'Albrecht_Du╠êrer_282.jpg', 'Albrecht_Du╠êrer_283.jpg', 'Albrecht_Du╠êrer_284.jpg', 'Albrecht_Du╠êrer_285.jpg', 'Albrecht_Du╠êrer_286.jpg', 'Albrecht_Du╠êrer_287.jpg', 'Albrecht_Du╠êrer_288.jpg', 'Albrecht_Du╠êrer_289.jpg', 'Albrecht_Du╠êrer_29.jpg', 'Albrecht_Du╠êrer_290.jpg', 'Albrecht_Du╠êrer_291.jpg', 'Albrecht_Du╠êrer_292.jpg', 'Albrecht_Du╠êrer_293.jpg', 'Albrecht_Du╠êrer_294.jpg', 'Albrecht_Du╠êrer_295.jpg', 'Albrecht_Du╠êrer_296.jpg', 'Albrecht_Du╠êrer_297.jpg', 'Albrecht_Du╠êrer_298.jpg', 'Albrecht_Du╠êrer_299.jpg', 'Albrecht_Du╠êrer_3.jpg', 'Albrecht_Du╠êrer_30.jpg', 'Albrecht_Du╠êrer_300.jpg', 'Albrecht_Du╠êrer_301.jpg', 'Albrecht_Du╠êrer_302.jpg', 'Albrecht_Du╠êrer_303.jpg', 'Albrecht_Du╠êrer_304.jpg', 'Albrecht_Du╠êrer_305.jpg', 'Albrecht_Du╠êrer_306.jpg', 'Albrecht_Du╠êrer_307.jpg', 'Albrecht_Du╠êrer_308.jpg', 'Albrecht_Du╠êrer_309.jpg', 'Albrecht_Du╠êrer_31.jpg', 'Albrecht_Du╠êrer_310.jpg', 'Albrecht_Du╠êrer_311.jpg', 'Albrecht_Du╠êrer_312.jpg', 'Albrecht_Du╠êrer_313.jpg', 'Albrecht_Du╠êrer_314.jpg', 'Albrecht_Du╠êrer_315.jpg', 'Albrecht_Du╠êrer_316.jpg', 'Albrecht_Du╠êrer_317.jpg', 'Albrecht_Du╠êrer_318.jpg', 'Albrecht_Du╠êrer_319.jpg', 'Albrecht_Du╠êrer_32.jpg', 'Albrecht_Du╠êrer_320.jpg', 'Albrecht_Du╠êrer_321.jpg', 'Albrecht_Du╠êrer_322.jpg', 'Albrecht_Du╠êrer_323.jpg', 'Albrecht_Du╠êrer_324.jpg', 'Albrecht_Du╠êrer_325.jpg', 'Albrecht_Du╠êrer_326.jpg', 'Albrecht_Du╠êrer_327.jpg', 'Albrecht_Du╠êrer_328.jpg', 'Albrecht_Du╠êrer_33.jpg', 'Albrecht_Du╠êrer_34.jpg', 'Albrecht_Du╠êrer_35.jpg', 'Albrecht_Du╠êrer_36.jpg', 'Albrecht_Du╠êrer_37.jpg', 'Albrecht_Du╠êrer_38.jpg', 'Albrecht_Du╠êrer_39.jpg', 'Albrecht_Du╠êrer_4.jpg', 'Albrecht_Du╠êrer_40.jpg', 'Albrecht_Du╠êrer_41.jpg', 'Albrecht_Du╠êrer_42.jpg', 'Albrecht_Du╠êrer_43.jpg', 'Albrecht_Du╠êrer_44.jpg', 'Albrecht_Du╠êrer_45.jpg', 'Albrecht_Du╠êrer_46.jpg', 'Albrecht_Du╠êrer_47.jpg', 'Albrecht_Du╠êrer_48.jpg', 'Albrecht_Du╠êrer_49.jpg', 'Albrecht_Du╠êrer_5.jpg', 'Albrecht_Du╠êrer_50.jpg', 'Albrecht_Du╠êrer_51.jpg', 'Albrecht_Du╠êrer_52.jpg', 'Albrecht_Du╠êrer_53.jpg', 'Albrecht_Du╠êrer_54.jpg', 'Albrecht_Du╠êrer_55.jpg', 'Albrecht_Du╠êrer_56.jpg', 'Albrecht_Du╠êrer_57.jpg', 'Albrecht_Du╠êrer_58.jpg', 'Albrecht_Du╠êrer_59.jpg', 'Albrecht_Du╠êrer_6.jpg', 'Albrecht_Du╠êrer_60.jpg', 'Albrecht_Du╠êrer_61.jpg', 'Albrecht_Du╠êrer_62.jpg', 'Albrecht_Du╠êrer_63.jpg', 'Albrecht_Du╠êrer_64.jpg', 'Albrecht_Du╠êrer_65.jpg', 'Albrecht_Du╠êrer_66.jpg', 'Albrecht_Du╠êrer_67.jpg', 'Albrecht_Du╠êrer_68.jpg', 'Albrecht_Du╠êrer_69.jpg', 'Albrecht_Du╠êrer_7.jpg', 'Albrecht_Du╠êrer_70.jpg', 'Albrecht_Du╠êrer_71.jpg', 'Albrecht_Du╠êrer_72.jpg', 'Albrecht_Du╠êrer_73.jpg', 'Albrecht_Du╠êrer_74.jpg', 'Albrecht_Du╠êrer_75.jpg', 'Albrecht_Du╠êrer_76.jpg', 'Albrecht_Du╠êrer_77.jpg', 'Albrecht_Du╠êrer_78.jpg', 'Albrecht_Du╠êrer_79.jpg', 'Albrecht_Du╠êrer_8.jpg', 'Albrecht_Du╠êrer_80.jpg', 'Albrecht_Du╠êrer_81.jpg', 'Albrecht_Du╠êrer_82.jpg', 'Albrecht_Du╠êrer_83.jpg', 'Albrecht_Du╠êrer_84.jpg', 'Albrecht_Du╠êrer_85.jpg', 'Albrecht_Du╠êrer_86.jpg', 'Albrecht_Du╠êrer_87.jpg', 'Albrecht_Du╠êrer_88.jpg', 'Albrecht_Du╠êrer_89.jpg', 'Albrecht_Du╠êrer_9.jpg', 'Albrecht_Du╠êrer_90.jpg', 'Albrecht_Du╠êrer_91.jpg', 'Albrecht_Du╠êrer_92.jpg', 'Albrecht_Du╠êrer_93.jpg', 'Albrecht_Du╠êrer_94.jpg', 'Albrecht_Du╠êrer_95.jpg', 'Albrecht_Du╠êrer_96.jpg', 'Albrecht_Du╠êrer_97.jpg', 'Albrecht_Du╠êrer_98.jpg', 'Albrecht_Du╠êrer_99.jpg', 'Alfred_Sisley_1.jpg', 'Alfred_Sisley_10.jpg', 'Alfred_Sisley_100.jpg', 'Alfred_Sisley_101.jpg', 'Alfred_Sisley_102.jpg', 'Alfred_Sisley_103.jpg', 'Alfred_Sisley_104.jpg', 'Alfred_Sisley_105.jpg', 'Alfred_Sisley_106.jpg', 'Alfred_Sisley_107.jpg', 'Alfred_Sisley_108.jpg', 'Alfred_Sisley_109.jpg', 'Alfred_Sisley_11.jpg', 'Alfred_Sisley_110.jpg', 'Alfred_Sisley_111.jpg', 'Alfred_Sisley_112.jpg', 'Alfred_Sisley_113.jpg', 'Alfred_Sisley_114.jpg', 'Alfred_Sisley_115.jpg', 'Alfred_Sisley_116.jpg', 'Alfred_Sisley_117.jpg', 'Alfred_Sisley_118.jpg', 'Alfred_Sisley_119.jpg', 'Alfred_Sisley_12.jpg', 'Alfred_Sisley_120.jpg', 'Alfred_Sisley_121.jpg', 'Alfred_Sisley_122.jpg', 'Alfred_Sisley_123.jpg', 'Alfred_Sisley_124.jpg', 'Alfred_Sisley_125.jpg', 'Alfred_Sisley_126.jpg', 'Alfred_Sisley_127.jpg', 'Alfred_Sisley_128.jpg', 'Alfred_Sisley_129.jpg', 'Alfred_Sisley_13.jpg', 'Alfred_Sisley_130.jpg', 'Alfred_Sisley_131.jpg', 'Alfred_Sisley_132.jpg', 'Alfred_Sisley_133.jpg', 'Alfred_Sisley_134.jpg', 'Alfred_Sisley_135.jpg', 'Alfred_Sisley_136.jpg', 'Alfred_Sisley_137.jpg', 'Alfred_Sisley_138.jpg', 'Alfred_Sisley_139.jpg', 'Alfred_Sisley_14.jpg', 'Alfred_Sisley_140.jpg', 'Alfred_Sisley_141.jpg', 'Alfred_Sisley_142.jpg', 'Alfred_Sisley_143.jpg', 'Alfred_Sisley_144.jpg', 'Alfred_Sisley_145.jpg', 'Alfred_Sisley_146.jpg', 'Alfred_Sisley_147.jpg', 'Alfred_Sisley_148.jpg', 'Alfred_Sisley_149.jpg', 'Alfred_Sisley_15.jpg', 'Alfred_Sisley_150.jpg', 'Alfred_Sisley_151.jpg', 'Alfred_Sisley_152.jpg', 'Alfred_Sisley_153.jpg', 'Alfred_Sisley_154.jpg', 'Alfred_Sisley_155.jpg', 'Alfred_Sisley_156.jpg', 'Alfred_Sisley_157.jpg', 'Alfred_Sisley_158.jpg', 'Alfred_Sisley_159.jpg', 'Alfred_Sisley_16.jpg', 'Alfred_Sisley_160.jpg', 'Alfred_Sisley_161.jpg', 'Alfred_Sisley_162.jpg', 'Alfred_Sisley_163.jpg', 'Alfred_Sisley_164.jpg', 'Alfred_Sisley_165.jpg', 'Alfred_Sisley_166.jpg', 'Alfred_Sisley_167.jpg', 'Alfred_Sisley_168.jpg', 'Alfred_Sisley_169.jpg', 'Alfred_Sisley_17.jpg', 'Alfred_Sisley_170.jpg', 'Alfred_Sisley_171.jpg', 'Alfred_Sisley_172.jpg', 'Alfred_Sisley_173.jpg', 'Alfred_Sisley_174.jpg', 'Alfred_Sisley_175.jpg', 'Alfred_Sisley_176.jpg', 'Alfred_Sisley_177.jpg', 'Alfred_Sisley_178.jpg', 'Alfred_Sisley_179.jpg', 'Alfred_Sisley_18.jpg', 'Alfred_Sisley_180.jpg', 'Alfred_Sisley_181.jpg', 'Alfred_Sisley_182.jpg', 'Alfred_Sisley_183.jpg', 'Alfred_Sisley_184.jpg', 'Alfred_Sisley_185.jpg', 'Alfred_Sisley_186.jpg', 'Alfred_Sisley_187.jpg', 'Alfred_Sisley_188.jpg', 'Alfred_Sisley_189.jpg', 'Alfred_Sisley_19.jpg', 'Alfred_Sisley_190.jpg', 'Alfred_Sisley_191.jpg', 'Alfred_Sisley_192.jpg', 'Alfred_Sisley_193.jpg', 'Alfred_Sisley_194.jpg', 'Alfred_Sisley_195.jpg', 'Alfred_Sisley_196.jpg', 'Alfred_Sisley_197.jpg', 'Alfred_Sisley_198.jpg', 'Alfred_Sisley_199.jpg', 'Alfred_Sisley_2.jpg', 'Alfred_Sisley_20.jpg', 'Alfred_Sisley_200.jpg', 'Alfred_Sisley_201.jpg', 'Alfred_Sisley_202.jpg', 'Alfred_Sisley_203.jpg', 'Alfred_Sisley_204.jpg', 'Alfred_Sisley_205.jpg', 'Alfred_Sisley_206.jpg', 'Alfred_Sisley_207.jpg', 'Alfred_Sisley_208.jpg', 'Alfred_Sisley_209.jpg', 'Alfred_Sisley_21.jpg', 'Alfred_Sisley_210.jpg', 'Alfred_Sisley_211.jpg', 'Alfred_Sisley_212.jpg', 'Alfred_Sisley_213.jpg', 'Alfred_Sisley_214.jpg', 'Alfred_Sisley_215.jpg', 'Alfred_Sisley_216.jpg', 'Alfred_Sisley_217.jpg', 'Alfred_Sisley_218.jpg', 'Alfred_Sisley_219.jpg', 'Alfred_Sisley_22.jpg', 'Alfred_Sisley_220.jpg', 'Alfred_Sisley_221.jpg', 'Alfred_Sisley_222.jpg', 'Alfred_Sisley_223.jpg', 'Alfred_Sisley_224.jpg', 'Alfred_Sisley_225.jpg', 'Alfred_Sisley_226.jpg', 'Alfred_Sisley_227.jpg', 'Alfred_Sisley_228.jpg', 'Alfred_Sisley_229.jpg', 'Alfred_Sisley_23.jpg', 'Alfred_Sisley_230.jpg', 'Alfred_Sisley_231.jpg', 'Alfred_Sisley_232.jpg', 'Alfred_Sisley_233.jpg', 'Alfred_Sisley_234.jpg', 'Alfred_Sisley_235.jpg', 'Alfred_Sisley_236.jpg', 'Alfred_Sisley_237.jpg', 'Alfred_Sisley_238.jpg', 'Alfred_Sisley_239.jpg', 'Alfred_Sisley_24.jpg', 'Alfred_Sisley_240.jpg', 'Alfred_Sisley_241.jpg', 'Alfred_Sisley_242.jpg', 'Alfred_Sisley_243.jpg', 'Alfred_Sisley_244.jpg', 'Alfred_Sisley_245.jpg', 'Alfred_Sisley_246.jpg', 'Alfred_Sisley_247.jpg', 'Alfred_Sisley_248.jpg', 'Alfred_Sisley_249.jpg', 'Alfred_Sisley_25.jpg', 'Alfred_Sisley_250.jpg', 'Alfred_Sisley_251.jpg', 'Alfred_Sisley_252.jpg', 'Alfred_Sisley_253.jpg', 'Alfred_Sisley_254.jpg', 'Alfred_Sisley_255.jpg', 'Alfred_Sisley_256.jpg', 'Alfred_Sisley_257.jpg', 'Alfred_Sisley_258.jpg', 'Alfred_Sisley_259.jpg', 'Alfred_Sisley_26.jpg', 'Alfred_Sisley_27.jpg', 'Alfred_Sisley_28.jpg', 'Alfred_Sisley_29.jpg', 'Alfred_Sisley_3.jpg', 'Alfred_Sisley_30.jpg', 'Alfred_Sisley_31.jpg', 'Alfred_Sisley_32.jpg', 'Alfred_Sisley_33.jpg', 'Alfred_Sisley_34.jpg', 'Alfred_Sisley_35.jpg', 'Alfred_Sisley_36.jpg', 'Alfred_Sisley_37.jpg', 'Alfred_Sisley_38.jpg', 'Alfred_Sisley_39.jpg', 'Alfred_Sisley_4.jpg', 'Alfred_Sisley_40.jpg', 'Alfred_Sisley_41.jpg', 'Alfred_Sisley_42.jpg', 'Alfred_Sisley_43.jpg', 'Alfred_Sisley_44.jpg', 'Alfred_Sisley_45.jpg', 'Alfred_Sisley_46.jpg', 'Alfred_Sisley_47.jpg', 'Alfred_Sisley_48.jpg', 'Alfred_Sisley_49.jpg', 'Alfred_Sisley_5.jpg', 'Alfred_Sisley_50.jpg', 'Alfred_Sisley_51.jpg', 'Alfred_Sisley_52.jpg', 'Alfred_Sisley_53.jpg', 'Alfred_Sisley_54.jpg', 'Alfred_Sisley_55.jpg', 'Alfred_Sisley_56.jpg', 'Alfred_Sisley_57.jpg', 'Alfred_Sisley_58.jpg', 'Alfred_Sisley_59.jpg', 'Alfred_Sisley_6.jpg', 'Alfred_Sisley_60.jpg', 'Alfred_Sisley_61.jpg', 'Alfred_Sisley_62.jpg', 'Alfred_Sisley_63.jpg', 'Alfred_Sisley_64.jpg', 'Alfred_Sisley_65.jpg', 'Alfred_Sisley_66.jpg', 'Alfred_Sisley_67.jpg', 'Alfred_Sisley_68.jpg', 'Alfred_Sisley_69.jpg', 'Alfred_Sisley_7.jpg', 'Alfred_Sisley_70.jpg', 'Alfred_Sisley_71.jpg', 'Alfred_Sisley_72.jpg', 'Alfred_Sisley_73.jpg', 'Alfred_Sisley_74.jpg', 'Alfred_Sisley_75.jpg', 'Alfred_Sisley_76.jpg', 'Alfred_Sisley_77.jpg', 'Alfred_Sisley_78.jpg', 'Alfred_Sisley_79.jpg', 'Alfred_Sisley_8.jpg', 'Alfred_Sisley_80.jpg', 'Alfred_Sisley_81.jpg', 'Alfred_Sisley_82.jpg', 'Alfred_Sisley_83.jpg', 'Alfred_Sisley_84.jpg', 'Alfred_Sisley_85.jpg', 'Alfred_Sisley_86.jpg', 'Alfred_Sisley_87.jpg', 'Alfred_Sisley_88.jpg', 'Alfred_Sisley_89.jpg', 'Alfred_Sisley_9.jpg', 'Alfred_Sisley_90.jpg', 'Alfred_Sisley_91.jpg', 'Alfred_Sisley_92.jpg', 'Alfred_Sisley_93.jpg', 'Alfred_Sisley_94.jpg', 'Alfred_Sisley_95.jpg', 'Alfred_Sisley_96.jpg', 'Alfred_Sisley_97.jpg', 'Alfred_Sisley_98.jpg', 'Alfred_Sisley_99.jpg', 'Amedeo_Modigliani_1.jpg', 'Amedeo_Modigliani_10.jpg', 'Amedeo_Modigliani_100.jpg', 'Amedeo_Modigliani_101.jpg', 'Amedeo_Modigliani_102.jpg', 'Amedeo_Modigliani_103.jpg', 'Amedeo_Modigliani_104.jpg', 'Amedeo_Modigliani_105.jpg', 'Amedeo_Modigliani_106.jpg', 'Amedeo_Modigliani_107.jpg', 'Amedeo_Modigliani_108.jpg', 'Amedeo_Modigliani_109.jpg', 'Amedeo_Modigliani_11.jpg', 'Amedeo_Modigliani_110.jpg', 'Amedeo_Modigliani_111.jpg', 'Amedeo_Modigliani_112.jpg', 'Amedeo_Modigliani_113.jpg', 'Amedeo_Modigliani_114.jpg', 'Amedeo_Modigliani_115.jpg', 'Amedeo_Modigliani_116.jpg', 'Amedeo_Modigliani_117.jpg', 'Amedeo_Modigliani_118.jpg', 'Amedeo_Modigliani_119.jpg', 'Amedeo_Modigliani_12.jpg', 'Amedeo_Modigliani_120.jpg', 'Amedeo_Modigliani_121.jpg', 'Amedeo_Modigliani_122.jpg', 'Amedeo_Modigliani_123.jpg', 'Amedeo_Modigliani_124.jpg', 'Amedeo_Modigliani_125.jpg', 'Amedeo_Modigliani_126.jpg', 'Amedeo_Modigliani_127.jpg', 'Amedeo_Modigliani_128.jpg', 'Amedeo_Modigliani_129.jpg', 'Amedeo_Modigliani_13.jpg', 'Amedeo_Modigliani_130.jpg', 'Amedeo_Modigliani_131.jpg', 'Amedeo_Modigliani_132.jpg', 'Amedeo_Modigliani_133.jpg', 'Amedeo_Modigliani_134.jpg', 'Amedeo_Modigliani_135.jpg', 'Amedeo_Modigliani_136.jpg', 'Amedeo_Modigliani_137.jpg', 'Amedeo_Modigliani_138.jpg', 'Amedeo_Modigliani_139.jpg', 'Amedeo_Modigliani_14.jpg', 'Amedeo_Modigliani_140.jpg', 'Amedeo_Modigliani_141.jpg', 'Amedeo_Modigliani_142.jpg', 'Amedeo_Modigliani_143.jpg', 'Amedeo_Modigliani_144.jpg', 'Amedeo_Modigliani_145.jpg', 'Amedeo_Modigliani_146.jpg', 'Amedeo_Modigliani_147.jpg', 'Amedeo_Modigliani_148.jpg', 'Amedeo_Modigliani_149.jpg', 'Amedeo_Modigliani_15.jpg', 'Amedeo_Modigliani_150.jpg', 'Amedeo_Modigliani_151.jpg', 'Amedeo_Modigliani_152.jpg', 'Amedeo_Modigliani_153.jpg', 'Amedeo_Modigliani_154.jpg', 'Amedeo_Modigliani_155.jpg', 'Amedeo_Modigliani_156.jpg', 'Amedeo_Modigliani_157.jpg', 'Amedeo_Modigliani_158.jpg', 'Amedeo_Modigliani_159.jpg', 'Amedeo_Modigliani_16.jpg', 'Amedeo_Modigliani_160.jpg', 'Amedeo_Modigliani_161.jpg', 'Amedeo_Modigliani_162.jpg', 'Amedeo_Modigliani_163.jpg', 'Amedeo_Modigliani_164.jpg', 'Amedeo_Modigliani_165.jpg', 'Amedeo_Modigliani_166.jpg', 'Amedeo_Modigliani_167.jpg', 'Amedeo_Modigliani_168.jpg', 'Amedeo_Modigliani_169.jpg', 'Amedeo_Modigliani_17.jpg', 'Amedeo_Modigliani_170.jpg', 'Amedeo_Modigliani_171.jpg', 'Amedeo_Modigliani_172.jpg', 'Amedeo_Modigliani_173.jpg', 'Amedeo_Modigliani_174.jpg', 'Amedeo_Modigliani_175.jpg', 'Amedeo_Modigliani_176.jpg', 'Amedeo_Modigliani_177.jpg', 'Amedeo_Modigliani_178.jpg', 'Amedeo_Modigliani_179.jpg', 'Amedeo_Modigliani_18.jpg', 'Amedeo_Modigliani_180.jpg', 'Amedeo_Modigliani_181.jpg', 'Amedeo_Modigliani_182.jpg', 'Amedeo_Modigliani_183.jpg', 'Amedeo_Modigliani_184.jpg', 'Amedeo_Modigliani_185.jpg', 'Amedeo_Modigliani_186.jpg', 'Amedeo_Modigliani_187.jpg', 'Amedeo_Modigliani_188.jpg', 'Amedeo_Modigliani_189.jpg', 'Amedeo_Modigliani_19.jpg', 'Amedeo_Modigliani_190.jpg', 'Amedeo_Modigliani_191.jpg', 'Amedeo_Modigliani_192.jpg', 'Amedeo_Modigliani_193.jpg', 'Amedeo_Modigliani_2.jpg', 'Amedeo_Modigliani_20.jpg', 'Amedeo_Modigliani_21.jpg', 'Amedeo_Modigliani_22.jpg', 'Amedeo_Modigliani_23.jpg', 'Amedeo_Modigliani_24.jpg', 'Amedeo_Modigliani_25.jpg', 'Amedeo_Modigliani_26.jpg', 'Amedeo_Modigliani_27.jpg', 'Amedeo_Modigliani_28.jpg', 'Amedeo_Modigliani_29.jpg', 'Amedeo_Modigliani_3.jpg', 'Amedeo_Modigliani_30.jpg', 'Amedeo_Modigliani_31.jpg', 'Amedeo_Modigliani_32.jpg', 'Amedeo_Modigliani_33.jpg', 'Amedeo_Modigliani_34.jpg', 'Amedeo_Modigliani_35.jpg', 'Amedeo_Modigliani_36.jpg', 'Amedeo_Modigliani_37.jpg', 'Amedeo_Modigliani_38.jpg', 'Amedeo_Modigliani_39.jpg', 'Amedeo_Modigliani_4.jpg', 'Amedeo_Modigliani_40.jpg', 'Amedeo_Modigliani_41.jpg', 'Amedeo_Modigliani_42.jpg', 'Amedeo_Modigliani_43.jpg', 'Amedeo_Modigliani_44.jpg', 'Amedeo_Modigliani_45.jpg', 'Amedeo_Modigliani_46.jpg', 'Amedeo_Modigliani_47.jpg', 'Amedeo_Modigliani_48.jpg', 'Amedeo_Modigliani_49.jpg', 'Amedeo_Modigliani_5.jpg', 'Amedeo_Modigliani_50.jpg', 'Amedeo_Modigliani_51.jpg', 'Amedeo_Modigliani_52.jpg', 'Amedeo_Modigliani_53.jpg', 'Amedeo_Modigliani_54.jpg', 'Amedeo_Modigliani_55.jpg', 'Amedeo_Modigliani_56.jpg', 'Amedeo_Modigliani_57.jpg', 'Amedeo_Modigliani_58.jpg', 'Amedeo_Modigliani_59.jpg', 'Amedeo_Modigliani_6.jpg', 'Amedeo_Modigliani_60.jpg', 'Amedeo_Modigliani_61.jpg', 'Amedeo_Modigliani_62.jpg', 'Amedeo_Modigliani_63.jpg', 'Amedeo_Modigliani_64.jpg', 'Amedeo_Modigliani_65.jpg', 'Amedeo_Modigliani_66.jpg', 'Amedeo_Modigliani_67.jpg', 'Amedeo_Modigliani_68.jpg', 'Amedeo_Modigliani_69.jpg', 'Amedeo_Modigliani_7.jpg', 'Amedeo_Modigliani_70.jpg', 'Amedeo_Modigliani_71.jpg', 'Amedeo_Modigliani_72.jpg', 'Amedeo_Modigliani_73.jpg', 'Amedeo_Modigliani_74.jpg', 'Amedeo_Modigliani_75.jpg', 'Amedeo_Modigliani_76.jpg', 'Amedeo_Modigliani_77.jpg', 'Amedeo_Modigliani_78.jpg', 'Amedeo_Modigliani_79.jpg', 'Amedeo_Modigliani_8.jpg', 'Amedeo_Modigliani_80.jpg', 'Amedeo_Modigliani_81.jpg', 'Amedeo_Modigliani_82.jpg', 'Amedeo_Modigliani_83.jpg', 'Amedeo_Modigliani_84.jpg', 'Amedeo_Modigliani_85.jpg', 'Amedeo_Modigliani_86.jpg', 'Amedeo_Modigliani_87.jpg', 'Amedeo_Modigliani_88.jpg', 'Amedeo_Modigliani_89.jpg', 'Amedeo_Modigliani_9.jpg', 'Amedeo_Modigliani_90.jpg', 'Amedeo_Modigliani_91.jpg', 'Amedeo_Modigliani_92.jpg', 'Amedeo_Modigliani_93.jpg', 'Amedeo_Modigliani_94.jpg', 'Amedeo_Modigliani_95.jpg', 'Amedeo_Modigliani_96.jpg', 'Amedeo_Modigliani_97.jpg', 'Amedeo_Modigliani_98.jpg', 'Amedeo_Modigliani_99.jpg', 'Andrei_Rublev_1.jpg', 'Andrei_Rublev_10.jpg', 'Andrei_Rublev_11.jpg', 'Andrei_Rublev_12.jpg', 'Andrei_Rublev_13.jpg', 'Andrei_Rublev_14.jpg', 'Andrei_Rublev_15.jpg', 'Andrei_Rublev_16.jpg', 'Andrei_Rublev_17.jpg', 'Andrei_Rublev_18.jpg', 'Andrei_Rublev_19.jpg', 'Andrei_Rublev_2.jpg', 'Andrei_Rublev_20.jpg', 'Andrei_Rublev_21.jpg', 'Andrei_Rublev_22.jpg', 'Andrei_Rublev_23.jpg', 'Andrei_Rublev_24.jpg', 'Andrei_Rublev_25.jpg', 'Andrei_Rublev_26.jpg', 'Andrei_Rublev_27.jpg', 'Andrei_Rublev_28.jpg', 'Andrei_Rublev_29.jpg', 'Andrei_Rublev_3.jpg', 'Andrei_Rublev_30.jpg', 'Andrei_Rublev_31.jpg', 'Andrei_Rublev_32.jpg', 'Andrei_Rublev_33.jpg', 'Andrei_Rublev_34.jpg', 'Andrei_Rublev_35.jpg', 'Andrei_Rublev_36.jpg', 'Andrei_Rublev_37.jpg', 'Andrei_Rublev_38.jpg', 'Andrei_Rublev_39.jpg', 'Andrei_Rublev_4.jpg', 'Andrei_Rublev_40.jpg', 'Andrei_Rublev_41.jpg', 'Andrei_Rublev_42.jpg', 'Andrei_Rublev_43.jpg', 'Andrei_Rublev_44.jpg', 'Andrei_Rublev_45.jpg', 'Andrei_Rublev_46.jpg', 'Andrei_Rublev_47.jpg', 'Andrei_Rublev_48.jpg', 'Andrei_Rublev_49.jpg', 'Andrei_Rublev_5.jpg', 'Andrei_Rublev_50.jpg', 'Andrei_Rublev_51.jpg', 'Andrei_Rublev_52.jpg', 'Andrei_Rublev_53.jpg', 'Andrei_Rublev_54.jpg', 'Andrei_Rublev_55.jpg', 'Andrei_Rublev_56.jpg', 'Andrei_Rublev_57.jpg', 'Andrei_Rublev_58.jpg', 'Andrei_Rublev_59.jpg', 'Andrei_Rublev_6.jpg', 'Andrei_Rublev_60.jpg', 'Andrei_Rublev_61.jpg', 'Andrei_Rublev_62.jpg', 'Andrei_Rublev_63.jpg', 'Andrei_Rublev_64.jpg', 'Andrei_Rublev_65.jpg', 'Andrei_Rublev_66.jpg', 'Andrei_Rublev_67.jpg', 'Andrei_Rublev_68.jpg', 'Andrei_Rublev_69.jpg', 'Andrei_Rublev_7.jpg', 'Andrei_Rublev_70.jpg', 'Andrei_Rublev_71.jpg', 'Andrei_Rublev_72.jpg', 'Andrei_Rublev_73.jpg', 'Andrei_Rublev_74.jpg', 'Andrei_Rublev_75.jpg', 'Andrei_Rublev_76.jpg', 'Andrei_Rublev_77.jpg', 'Andrei_Rublev_78.jpg', 'Andrei_Rublev_79.jpg', 'Andrei_Rublev_8.jpg', 'Andrei_Rublev_80.jpg', 'Andrei_Rublev_81.jpg', 'Andrei_Rublev_82.jpg', 'Andrei_Rublev_83.jpg', 'Andrei_Rublev_84.jpg', 'Andrei_Rublev_85.jpg', 'Andrei_Rublev_86.jpg', 'Andrei_Rublev_87.jpg', 'Andrei_Rublev_88.jpg', 'Andrei_Rublev_89.jpg', 'Andrei_Rublev_9.jpg', 'Andrei_Rublev_90.jpg', 'Andrei_Rublev_91.jpg', 'Andrei_Rublev_92.jpg', 'Andrei_Rublev_93.jpg', 'Andrei_Rublev_94.jpg', 'Andrei_Rublev_95.jpg', 'Andrei_Rublev_96.jpg', 'Andrei_Rublev_97.jpg', 'Andrei_Rublev_98.jpg', 'Andrei_Rublev_99.jpg', 'Andy_Warhol_1.jpg', 'Andy_Warhol_10.jpg', 'Andy_Warhol_100.jpg', 'Andy_Warhol_101.jpg', 'Andy_Warhol_102.jpg', 'Andy_Warhol_103.jpg', 'Andy_Warhol_104.jpg', 'Andy_Warhol_105.jpg', 'Andy_Warhol_106.jpg', 'Andy_Warhol_107.jpg', 'Andy_Warhol_108.jpg', 'Andy_Warhol_109.jpg', 'Andy_Warhol_11.jpg', 'Andy_Warhol_110.jpg', 'Andy_Warhol_111.jpg', 'Andy_Warhol_112.jpg', 'Andy_Warhol_113.jpg', 'Andy_Warhol_114.jpg', 'Andy_Warhol_115.jpg', 'Andy_Warhol_116.jpg', 'Andy_Warhol_117.jpg', 'Andy_Warhol_118.jpg', 'Andy_Warhol_119.jpg', 'Andy_Warhol_12.jpg', 'Andy_Warhol_120.jpg', 'Andy_Warhol_121.jpg', 'Andy_Warhol_122.jpg', 'Andy_Warhol_123.jpg', 'Andy_Warhol_124.jpg', 'Andy_Warhol_125.jpg', 'Andy_Warhol_126.jpg', 'Andy_Warhol_127.jpg', 'Andy_Warhol_128.jpg', 'Andy_Warhol_129.jpg', 'Andy_Warhol_13.jpg', 'Andy_Warhol_130.jpg', 'Andy_Warhol_131.jpg', 'Andy_Warhol_132.jpg', 'Andy_Warhol_133.jpg', 'Andy_Warhol_134.jpg', 'Andy_Warhol_135.jpg', 'Andy_Warhol_136.jpg', 'Andy_Warhol_137.jpg', 'Andy_Warhol_138.jpg', 'Andy_Warhol_139.jpg', 'Andy_Warhol_14.jpg', 'Andy_Warhol_140.jpg', 'Andy_Warhol_141.jpg', 'Andy_Warhol_142.jpg', 'Andy_Warhol_143.jpg', 'Andy_Warhol_144.jpg', 'Andy_Warhol_145.jpg', 'Andy_Warhol_146.jpg', 'Andy_Warhol_147.jpg', 'Andy_Warhol_148.jpg', 'Andy_Warhol_149.jpg', 'Andy_Warhol_15.jpg', 'Andy_Warhol_150.jpg', 'Andy_Warhol_151.jpg', 'Andy_Warhol_152.jpg', 'Andy_Warhol_153.jpg', 'Andy_Warhol_154.jpg', 'Andy_Warhol_155.jpg', 'Andy_Warhol_156.jpg', 'Andy_Warhol_157.jpg', 'Andy_Warhol_158.jpg', 'Andy_Warhol_159.jpg', 'Andy_Warhol_16.jpg', 'Andy_Warhol_160.jpg', 'Andy_Warhol_161.jpg', 'Andy_Warhol_162.jpg', 'Andy_Warhol_163.jpg', 'Andy_Warhol_164.jpg', 'Andy_Warhol_165.jpg', 'Andy_Warhol_166.jpg', 'Andy_Warhol_167.jpg', 'Andy_Warhol_168.jpg', 'Andy_Warhol_169.jpg', 'Andy_Warhol_17.jpg', 'Andy_Warhol_170.jpg', 'Andy_Warhol_171.jpg', 'Andy_Warhol_172.jpg', 'Andy_Warhol_173.jpg', 'Andy_Warhol_174.jpg', 'Andy_Warhol_175.jpg', 'Andy_Warhol_176.jpg', 'Andy_Warhol_177.jpg', 'Andy_Warhol_178.jpg', 'Andy_Warhol_179.jpg', 'Andy_Warhol_18.jpg', 'Andy_Warhol_180.jpg', 'Andy_Warhol_181.jpg', 'Andy_Warhol_19.jpg', 'Andy_Warhol_2.jpg', 'Andy_Warhol_20.jpg', 'Andy_Warhol_21.jpg', 'Andy_Warhol_22.jpg', 'Andy_Warhol_23.jpg', 'Andy_Warhol_24.jpg', 'Andy_Warhol_25.jpg', 'Andy_Warhol_26.jpg', 'Andy_Warhol_27.jpg', 'Andy_Warhol_28.jpg', 'Andy_Warhol_29.jpg', 'Andy_Warhol_3.jpg', 'Andy_Warhol_30.jpg', 'Andy_Warhol_31.jpg', 'Andy_Warhol_32.jpg', 'Andy_Warhol_33.jpg', 'Andy_Warhol_34.jpg', 'Andy_Warhol_35.jpg', 'Andy_Warhol_36.jpg', 'Andy_Warhol_37.jpg', 'Andy_Warhol_38.jpg', 'Andy_Warhol_39.jpg', 'Andy_Warhol_4.jpg', 'Andy_Warhol_40.jpg', 'Andy_Warhol_41.jpg', 'Andy_Warhol_42.jpg', 'Andy_Warhol_43.jpg', 'Andy_Warhol_44.jpg', 'Andy_Warhol_45.jpg', 'Andy_Warhol_46.jpg', 'Andy_Warhol_47.jpg', 'Andy_Warhol_48.jpg', 'Andy_Warhol_49.jpg', 'Andy_Warhol_5.jpg', 'Andy_Warhol_50.jpg', 'Andy_Warhol_51.jpg', 'Andy_Warhol_52.jpg', 'Andy_Warhol_53.jpg', 'Andy_Warhol_54.jpg', 'Andy_Warhol_55.jpg', 'Andy_Warhol_56.jpg', 'Andy_Warhol_57.jpg', 'Andy_Warhol_58.jpg', 'Andy_Warhol_59.jpg', 'Andy_Warhol_6.jpg', 'Andy_Warhol_60.jpg', 'Andy_Warhol_61.jpg', 'Andy_Warhol_62.jpg', 'Andy_Warhol_63.jpg', 'Andy_Warhol_64.jpg', 'Andy_Warhol_65.jpg', 'Andy_Warhol_66.jpg', 'Andy_Warhol_67.jpg', 'Andy_Warhol_68.jpg', 'Andy_Warhol_69.jpg', 'Andy_Warhol_7.jpg', 'Andy_Warhol_70.jpg', 'Andy_Warhol_71.jpg', 'Andy_Warhol_72.jpg', 'Andy_Warhol_73.jpg', 'Andy_Warhol_74.jpg', 'Andy_Warhol_75.jpg', 'Andy_Warhol_76.jpg', 'Andy_Warhol_77.jpg', 'Andy_Warhol_78.jpg', 'Andy_Warhol_79.jpg', 'Andy_Warhol_8.jpg', 'Andy_Warhol_80.jpg', 'Andy_Warhol_81.jpg', 'Andy_Warhol_82.jpg', 'Andy_Warhol_83.jpg', 'Andy_Warhol_84.jpg', 'Andy_Warhol_85.jpg', 'Andy_Warhol_86.jpg', 'Andy_Warhol_87.jpg', 'Andy_Warhol_88.jpg', 'Andy_Warhol_89.jpg', 'Andy_Warhol_9.jpg', 'Andy_Warhol_90.jpg', 'Andy_Warhol_91.jpg', 'Andy_Warhol_92.jpg', 'Andy_Warhol_93.jpg', 'Andy_Warhol_94.jpg', 'Andy_Warhol_95.jpg', 'Andy_Warhol_96.jpg', 'Andy_Warhol_97.jpg', 'Andy_Warhol_98.jpg', 'Andy_Warhol_99.jpg', 'Camille_Pissarro_1.jpg', 'Camille_Pissarro_10.jpg', 'Camille_Pissarro_11.jpg', 'Camille_Pissarro_12.jpg', 'Camille_Pissarro_13.jpg', 'Camille_Pissarro_14.jpg', 'Camille_Pissarro_15.jpg', 'Camille_Pissarro_16.jpg', 'Camille_Pissarro_17.jpg', 'Camille_Pissarro_18.jpg', 'Camille_Pissarro_19.jpg', 'Camille_Pissarro_2.jpg', 'Camille_Pissarro_20.jpg', 'Camille_Pissarro_21.jpg', 'Camille_Pissarro_22.jpg', 'Camille_Pissarro_23.jpg', 'Camille_Pissarro_24.jpg', 'Camille_Pissarro_25.jpg', 'Camille_Pissarro_26.jpg', 'Camille_Pissarro_27.jpg', 'Camille_Pissarro_28.jpg', 'Camille_Pissarro_29.jpg', 'Camille_Pissarro_3.jpg', 'Camille_Pissarro_30.jpg', 'Camille_Pissarro_31.jpg', 'Camille_Pissarro_32.jpg', 'Camille_Pissarro_33.jpg', 'Camille_Pissarro_34.jpg', 'Camille_Pissarro_35.jpg', 'Camille_Pissarro_36.jpg', 'Camille_Pissarro_37.jpg', 'Camille_Pissarro_38.jpg', 'Camille_Pissarro_39.jpg', 'Camille_Pissarro_4.jpg', 'Camille_Pissarro_40.jpg', 'Camille_Pissarro_41.jpg', 'Camille_Pissarro_42.jpg', 'Camille_Pissarro_43.jpg', 'Camille_Pissarro_44.jpg', 'Camille_Pissarro_45.jpg', 'Camille_Pissarro_46.jpg', 'Camille_Pissarro_47.jpg', 'Camille_Pissarro_48.jpg', 'Camille_Pissarro_49.jpg', 'Camille_Pissarro_5.jpg', 'Camille_Pissarro_50.jpg', 'Camille_Pissarro_51.jpg', 'Camille_Pissarro_52.jpg', 'Camille_Pissarro_53.jpg', 'Camille_Pissarro_54.jpg', 'Camille_Pissarro_55.jpg', 'Camille_Pissarro_56.jpg', 'Camille_Pissarro_57.jpg', 'Camille_Pissarro_58.jpg', 'Camille_Pissarro_59.jpg', 'Camille_Pissarro_6.jpg', 'Camille_Pissarro_60.jpg', 'Camille_Pissarro_61.jpg', 'Camille_Pissarro_62.jpg', 'Camille_Pissarro_63.jpg', 'Camille_Pissarro_64.jpg', 'Camille_Pissarro_65.jpg', 'Camille_Pissarro_66.jpg', 'Camille_Pissarro_67.jpg', 'Camille_Pissarro_68.jpg', 'Camille_Pissarro_69.jpg', 'Camille_Pissarro_7.jpg', 'Camille_Pissarro_70.jpg', 'Camille_Pissarro_71.jpg', 'Camille_Pissarro_72.jpg', 'Camille_Pissarro_73.jpg', 'Camille_Pissarro_74.jpg', 'Camille_Pissarro_75.jpg', 'Camille_Pissarro_76.jpg', 'Camille_Pissarro_77.jpg', 'Camille_Pissarro_78.jpg', 'Camille_Pissarro_79.jpg', 'Camille_Pissarro_8.jpg', 'Camille_Pissarro_80.jpg', 'Camille_Pissarro_81.jpg', 'Camille_Pissarro_82.jpg', 'Camille_Pissarro_83.jpg', 'Camille_Pissarro_84.jpg', 'Camille_Pissarro_85.jpg', 'Camille_Pissarro_86.jpg', 'Camille_Pissarro_87.jpg', 'Camille_Pissarro_88.jpg', 'Camille_Pissarro_89.jpg', 'Camille_Pissarro_9.jpg', 'Camille_Pissarro_90.jpg', 'Camille_Pissarro_91.jpg', 'Caravaggio_1.jpg', 'Caravaggio_10.jpg', 'Caravaggio_11.jpg', 'Caravaggio_12.jpg', 'Caravaggio_13.jpg', 'Caravaggio_14.jpg', 'Caravaggio_15.jpg', 'Caravaggio_16.jpg', 'Caravaggio_17.jpg', 'Caravaggio_18.jpg', 'Caravaggio_19.jpg', 'Caravaggio_2.jpg', 'Caravaggio_20.jpg', 'Caravaggio_21.jpg', 'Caravaggio_22.jpg', 'Caravaggio_23.jpg', 'Caravaggio_24.jpg', 'Caravaggio_25.jpg', 'Caravaggio_26.jpg', 'Caravaggio_27.jpg', 'Caravaggio_28.jpg', 'Caravaggio_29.jpg', 'Caravaggio_3.jpg', 'Caravaggio_30.jpg', 'Caravaggio_31.jpg', 'Caravaggio_32.jpg', 'Caravaggio_33.jpg', 'Caravaggio_34.jpg', 'Caravaggio_35.jpg', 'Caravaggio_36.jpg', 'Caravaggio_37.jpg', 'Caravaggio_38.jpg', 'Caravaggio_39.jpg', 'Caravaggio_4.jpg', 'Caravaggio_40.jpg', 'Caravaggio_41.jpg', 'Caravaggio_42.jpg', 'Caravaggio_43.jpg', 'Caravaggio_44.jpg', 'Caravaggio_45.jpg', 'Caravaggio_46.jpg', 'Caravaggio_47.jpg', 'Caravaggio_48.jpg', 'Caravaggio_49.jpg', 'Caravaggio_5.jpg', 'Caravaggio_50.jpg', 'Caravaggio_51.jpg', 'Caravaggio_52.jpg', 'Caravaggio_53.jpg', 'Caravaggio_54.jpg', 'Caravaggio_55.jpg', 'Caravaggio_6.jpg', 'Caravaggio_7.jpg', 'Caravaggio_8.jpg', 'Caravaggio_9.jpg', 'Claude_Monet_1.jpg', 'Claude_Monet_10.jpg', 'Claude_Monet_11.jpg', 'Claude_Monet_12.jpg', 'Claude_Monet_13.jpg', 'Claude_Monet_14.jpg', 'Claude_Monet_15.jpg', 'Claude_Monet_16.jpg', 'Claude_Monet_17.jpg', 'Claude_Monet_18.jpg', 'Claude_Monet_19.jpg', 'Claude_Monet_2.jpg', 'Claude_Monet_20.jpg', 'Claude_Monet_21.jpg', 'Claude_Monet_22.jpg', 'Claude_Monet_23.jpg', 'Claude_Monet_24.jpg', 'Claude_Monet_25.jpg', 'Claude_Monet_26.jpg', 'Claude_Monet_27.jpg', 'Claude_Monet_28.jpg', 'Claude_Monet_29.jpg', 'Claude_Monet_3.jpg', 'Claude_Monet_30.jpg', 'Claude_Monet_31.jpg', 'Claude_Monet_32.jpg', 'Claude_Monet_33.jpg', 'Claude_Monet_34.jpg', 'Claude_Monet_35.jpg', 'Claude_Monet_36.jpg', 'Claude_Monet_37.jpg', 'Claude_Monet_38.jpg', 'Claude_Monet_39.jpg', 'Claude_Monet_4.jpg', 'Claude_Monet_40.jpg', 'Claude_Monet_41.jpg', 'Claude_Monet_42.jpg', 'Claude_Monet_43.jpg', 'Claude_Monet_44.jpg', 'Claude_Monet_45.jpg', 'Claude_Monet_46.jpg', 'Claude_Monet_47.jpg', 'Claude_Monet_48.jpg', 'Claude_Monet_49.jpg', 'Claude_Monet_5.jpg', 'Claude_Monet_50.jpg', 'Claude_Monet_51.jpg', 'Claude_Monet_52.jpg', 'Claude_Monet_53.jpg', 'Claude_Monet_54.jpg', 'Claude_Monet_55.jpg', 'Claude_Monet_56.jpg', 'Claude_Monet_57.jpg', 'Claude_Monet_58.jpg', 'Claude_Monet_59.jpg', 'Claude_Monet_6.jpg', 'Claude_Monet_60.jpg', 'Claude_Monet_61.jpg', 'Claude_Monet_62.jpg', 'Claude_Monet_63.jpg', 'Claude_Monet_64.jpg', 'Claude_Monet_65.jpg', 'Claude_Monet_66.jpg', 'Claude_Monet_67.jpg', 'Claude_Monet_68.jpg', 'Claude_Monet_69.jpg', 'Claude_Monet_7.jpg', 'Claude_Monet_70.jpg', 'Claude_Monet_71.jpg', 'Claude_Monet_72.jpg', 'Claude_Monet_73.jpg', 'Claude_Monet_8.jpg', 'Claude_Monet_9.jpg', 'Diego_Rivera_1.jpg', 'Diego_Rivera_10.jpg', 'Diego_Rivera_11.jpg', 'Diego_Rivera_12.jpg', 'Diego_Rivera_13.jpg', 'Diego_Rivera_14.jpg', 'Diego_Rivera_15.jpg', 'Diego_Rivera_16.jpg', 'Diego_Rivera_17.jpg', 'Diego_Rivera_18.jpg', 'Diego_Rivera_19.jpg', 'Diego_Rivera_2.jpg', 'Diego_Rivera_20.jpg', 'Diego_Rivera_21.jpg', 'Diego_Rivera_22.jpg', 'Diego_Rivera_23.jpg', 'Diego_Rivera_24.jpg', 'Diego_Rivera_25.jpg', 'Diego_Rivera_26.jpg', 'Diego_Rivera_27.jpg', 'Diego_Rivera_28.jpg', 'Diego_Rivera_29.jpg', 'Diego_Rivera_3.jpg', 'Diego_Rivera_30.jpg', 'Diego_Rivera_31.jpg', 'Diego_Rivera_32.jpg', 'Diego_Rivera_33.jpg', 'Diego_Rivera_34.jpg', 'Diego_Rivera_35.jpg', 'Diego_Rivera_36.jpg', 'Diego_Rivera_37.jpg', 'Diego_Rivera_38.jpg', 'Diego_Rivera_39.jpg', 'Diego_Rivera_4.jpg', 'Diego_Rivera_40.jpg', 'Diego_Rivera_41.jpg', 'Diego_Rivera_42.jpg', 'Diego_Rivera_43.jpg', 'Diego_Rivera_44.jpg', 'Diego_Rivera_45.jpg', 'Diego_Rivera_46.jpg', 'Diego_Rivera_47.jpg', 'Diego_Rivera_48.jpg', 'Diego_Rivera_49.jpg', 'Diego_Rivera_5.jpg', 'Diego_Rivera_50.jpg', 'Diego_Rivera_51.jpg', 'Diego_Rivera_52.jpg', 'Diego_Rivera_53.jpg', 'Diego_Rivera_54.jpg', 'Diego_Rivera_55.jpg', 'Diego_Rivera_56.jpg', 'Diego_Rivera_57.jpg', 'Diego_Rivera_58.jpg', 'Diego_Rivera_59.jpg', 'Diego_Rivera_6.jpg', 'Diego_Rivera_60.jpg', 'Diego_Rivera_61.jpg', 'Diego_Rivera_62.jpg', 'Diego_Rivera_63.jpg', 'Diego_Rivera_64.jpg', 'Diego_Rivera_65.jpg', 'Diego_Rivera_66.jpg', 'Diego_Rivera_67.jpg', 'Diego_Rivera_68.jpg', 'Diego_Rivera_69.jpg', 'Diego_Rivera_7.jpg', 'Diego_Rivera_70.jpg', 'Diego_Rivera_8.jpg', 'Diego_Rivera_9.jpg', 'Diego_Velazquez_1.jpg', 'Diego_Velazquez_10.jpg', 'Diego_Velazquez_100.jpg', 'Diego_Velazquez_101.jpg', 'Diego_Velazquez_102.jpg', 'Diego_Velazquez_103.jpg', 'Diego_Velazquez_104.jpg', 'Diego_Velazquez_105.jpg', 'Diego_Velazquez_106.jpg', 'Diego_Velazquez_107.jpg', 'Diego_Velazquez_108.jpg', 'Diego_Velazquez_109.jpg', 'Diego_Velazquez_11.jpg', 'Diego_Velazquez_110.jpg', 'Diego_Velazquez_111.jpg', 'Diego_Velazquez_112.jpg', 'Diego_Velazquez_113.jpg', 'Diego_Velazquez_114.jpg', 'Diego_Velazquez_115.jpg', 'Diego_Velazquez_116.jpg', 'Diego_Velazquez_117.jpg', 'Diego_Velazquez_118.jpg', 'Diego_Velazquez_119.jpg', 'Diego_Velazquez_12.jpg', 'Diego_Velazquez_120.jpg', 'Diego_Velazquez_121.jpg', 'Diego_Velazquez_122.jpg', 'Diego_Velazquez_123.jpg', 'Diego_Velazquez_124.jpg', 'Diego_Velazquez_125.jpg', 'Diego_Velazquez_126.jpg', 'Diego_Velazquez_127.jpg', 'Diego_Velazquez_128.jpg', 'Diego_Velazquez_13.jpg', 'Diego_Velazquez_14.jpg', 'Diego_Velazquez_15.jpg', 'Diego_Velazquez_16.jpg', 'Edgar_Degas_1.jpg', 'Edgar_Degas_10.jpg', 'Edgar_Degas_100.jpg', 'Edgar_Degas_101.jpg', 'Edgar_Degas_102.jpg', 'Edgar_Degas_103.jpg', 'Edgar_Degas_104.jpg', 'Edgar_Degas_105.jpg', 'Edgar_Degas_106.jpg', 'Edgar_Degas_107.jpg', 'Edgar_Degas_108.jpg', 'Edgar_Degas_109.jpg', 'Edgar_Degas_11.jpg', 'Edgar_Degas_110.jpg', 'Edgar_Degas_111.jpg', 'Edgar_Degas_112.jpg', 'Edgar_Degas_113.jpg', 'Edgar_Degas_114.jpg', 'Edgar_Degas_115.jpg', 'Edgar_Degas_116.jpg', 'Edgar_Degas_117.jpg', 'Edgar_Degas_118.jpg', 'Edgar_Degas_119.jpg', 'Edgar_Degas_12.jpg', 'Edgar_Degas_120.jpg', 'Edgar_Degas_121.jpg', 'Edgar_Degas_122.jpg', 'Edgar_Degas_123.jpg', 'Edgar_Degas_124.jpg', 'Edgar_Degas_125.jpg', 'Edgar_Degas_126.jpg', 'Edgar_Degas_127.jpg', 'Edgar_Degas_128.jpg', 'Edgar_Degas_129.jpg', 'Edgar_Degas_13.jpg', 'Edgar_Degas_130.jpg', 'Edgar_Degas_131.jpg', 'Edgar_Degas_132.jpg', 'Edgar_Degas_133.jpg', 'Edgar_Degas_134.jpg', 'Edgar_Degas_135.jpg', 'Edgar_Degas_136.jpg', 'Edgar_Degas_137.jpg', 'Edgar_Degas_138.jpg', 'Edgar_Degas_139.jpg', 'Edgar_Degas_14.jpg', 'Edgar_Degas_140.jpg', 'Edgar_Degas_141.jpg', 'Edgar_Degas_142.jpg', 'Edgar_Degas_143.jpg', 'Edgar_Degas_144.jpg', 'Edgar_Degas_145.jpg', 'Edgar_Degas_146.jpg', 'Edgar_Degas_147.jpg', 'Edgar_Degas_148.jpg', 'Edgar_Degas_149.jpg', 'Edgar_Degas_15.jpg', 'Edgar_Degas_150.jpg', 'Edgar_Degas_151.jpg', 'Edgar_Degas_152.jpg', 'Edgar_Degas_153.jpg', 'Edgar_Degas_154.jpg', 'Edgar_Degas_155.jpg', 'Edgar_Degas_156.jpg', 'Edgar_Degas_157.jpg', 'Edgar_Degas_158.jpg', 'Edgar_Degas_159.jpg', 'Edgar_Degas_16.jpg', 'Edgar_Degas_160.jpg', 'Edgar_Degas_161.jpg', 'Edgar_Degas_162.jpg', 'Edgar_Degas_163.jpg', 'Edgar_Degas_164.jpg', 'Edgar_Degas_165.jpg', 'Edgar_Degas_166.jpg', 'Edgar_Degas_167.jpg', 'Edgar_Degas_168.jpg', 'Edgar_Degas_169.jpg', 'Edgar_Degas_17.jpg', 'Edgar_Degas_170.jpg', 'Edgar_Degas_171.jpg', 'Edgar_Degas_172.jpg', 'Edgar_Degas_173.jpg', 'Edgar_Degas_174.jpg', 'Edgar_Degas_175.jpg', 'Edgar_Degas_176.jpg', 'Edgar_Degas_177.jpg', 'Edgar_Degas_178.jpg', 'Edgar_Degas_179.jpg', 'Edgar_Degas_18.jpg', 'Edgar_Degas_180.jpg', 'Edgar_Degas_181.jpg', 'Edgar_Degas_182.jpg', 'Edgar_Degas_183.jpg', 'Edgar_Degas_184.jpg', 'Edgar_Degas_185.jpg', 'Edgar_Degas_186.jpg', 'Edgar_Degas_187.jpg', 'Edgar_Degas_188.jpg', 'Edgar_Degas_189.jpg', 'Edgar_Degas_19.jpg', 'Edgar_Degas_190.jpg', 'Edgar_Degas_191.jpg', 'Edgar_Degas_192.jpg', 'Edgar_Degas_193.jpg', 'Edgar_Degas_194.jpg', 'Edgar_Degas_195.jpg', 'Edgar_Degas_196.jpg', 'Edgar_Degas_197.jpg', 'Edgar_Degas_198.jpg', 'Edgar_Degas_199.jpg', 'Edgar_Degas_2.jpg', 'Edgar_Degas_20.jpg', 'Edgar_Degas_200.jpg', 'Edgar_Degas_201.jpg', 'Edgar_Degas_202.jpg', 'Edgar_Degas_203.jpg', 'Edgar_Degas_204.jpg', 'Edgar_Degas_205.jpg', 'Edgar_Degas_206.jpg', 'Edgar_Degas_207.jpg', 'Edgar_Degas_208.jpg', 'Edgar_Degas_209.jpg', 'Edgar_Degas_21.jpg', 'Edgar_Degas_210.jpg', 'Edgar_Degas_211.jpg', 'Edgar_Degas_212.jpg', 'Edgar_Degas_213.jpg', 'Edgar_Degas_214.jpg', 'Edgar_Degas_215.jpg', 'Edgar_Degas_216.jpg', 'Edgar_Degas_217.jpg', 'Edgar_Degas_218.jpg', 'Edgar_Degas_219.jpg', 'Edgar_Degas_22.jpg', 'Edgar_Degas_220.jpg', 'Edgar_Degas_221.jpg', 'Edgar_Degas_222.jpg', 'Edgar_Degas_223.jpg', 'Edgar_Degas_224.jpg', 'Edgar_Degas_225.jpg', 'Edgar_Degas_226.jpg', 'Edgar_Degas_227.jpg', 'Edgar_Degas_228.jpg', 'Edgar_Degas_229.jpg', 'Edgar_Degas_23.jpg', 'Edgar_Degas_230.jpg', 'Edgar_Degas_231.jpg', 'Edgar_Degas_232.jpg', 'Edgar_Degas_233.jpg', 'Edgar_Degas_234.jpg', 'Edgar_Degas_235.jpg', 'Edgar_Degas_236.jpg', 'Edgar_Degas_237.jpg', 'Edgar_Degas_238.jpg', 'Edgar_Degas_239.jpg', 'Edgar_Degas_24.jpg', 'Edgar_Degas_240.jpg', 'Edgar_Degas_241.jpg', 'Edgar_Degas_242.jpg', 'Edgar_Degas_243.jpg', 'Edgar_Degas_244.jpg', 'Edgar_Degas_245.jpg', 'Edgar_Degas_246.jpg', 'Edgar_Degas_247.jpg', 'Edgar_Degas_248.jpg', 'Edgar_Degas_249.jpg', 'Edgar_Degas_25.jpg', 'Edgar_Degas_250.jpg', 'Edgar_Degas_251.jpg', 'Edgar_Degas_252.jpg', 'Edgar_Degas_253.jpg', 'Edgar_Degas_254.jpg', 'Edgar_Degas_255.jpg', 'Edgar_Degas_256.jpg', 'Edgar_Degas_257.jpg', 'Edgar_Degas_258.jpg', 'Edgar_Degas_259.jpg', 'Edgar_Degas_26.jpg', 'Edgar_Degas_260.jpg', 'Edgar_Degas_261.jpg', 'Edgar_Degas_262.jpg', 'Edgar_Degas_263.jpg', 'Edgar_Degas_264.jpg', 'Edgar_Degas_265.jpg', 'Edgar_Degas_266.jpg', 'Edgar_Degas_267.jpg', 'Edgar_Degas_268.jpg', 'Edgar_Degas_269.jpg', 'Edgar_Degas_27.jpg', 'Edgar_Degas_270.jpg', 'Edgar_Degas_271.jpg', 'Edgar_Degas_272.jpg', 'Edgar_Degas_273.jpg', 'Edgar_Degas_274.jpg', 'Edgar_Degas_275.jpg', 'Edgar_Degas_276.jpg', 'Edgar_Degas_277.jpg', 'Edgar_Degas_278.jpg', 'Edgar_Degas_279.jpg', 'Edgar_Degas_28.jpg', 'Edgar_Degas_280.jpg', 'Edgar_Degas_281.jpg', 'Edgar_Degas_282.jpg', 'Edgar_Degas_283.jpg', 'Edgar_Degas_284.jpg', 'Edgar_Degas_285.jpg', 'Edgar_Degas_286.jpg', 'Edgar_Degas_287.jpg', 'Edgar_Degas_288.jpg', 'Edgar_Degas_289.jpg', 'Edgar_Degas_29.jpg', 'Edgar_Degas_290.jpg', 'Edgar_Degas_291.jpg', 'Edgar_Degas_292.jpg', 'Edgar_Degas_293.jpg', 'Edgar_Degas_294.jpg', 'Edgar_Degas_295.jpg', 'Edgar_Degas_296.jpg', 'Edgar_Degas_297.jpg', 'Edgar_Degas_298.jpg', 'Edgar_Degas_299.jpg', 'Edgar_Degas_3.jpg', 'Edgar_Degas_30.jpg', 'Edgar_Degas_300.jpg', 'Edgar_Degas_301.jpg', 'Edgar_Degas_302.jpg', 'Edgar_Degas_303.jpg', 'Edgar_Degas_304.jpg', 'Edgar_Degas_305.jpg', 'Edgar_Degas_306.jpg', 'Edgar_Degas_307.jpg', 'Edgar_Degas_308.jpg', 'Edgar_Degas_309.jpg', 'Edgar_Degas_31.jpg', 'Edgar_Degas_310.jpg', 'Edgar_Degas_311.jpg', 'Edgar_Degas_312.jpg', 'Edgar_Degas_313.jpg', 'Edgar_Degas_314.jpg', 'Edgar_Degas_315.jpg', 'Edgar_Degas_316.jpg', 'Edgar_Degas_317.jpg', 'Edgar_Degas_318.jpg', 'Edgar_Degas_319.jpg', 'Edgar_Degas_32.jpg', 'Edgar_Degas_320.jpg', 'Edgar_Degas_321.jpg', 'Edgar_Degas_322.jpg', 'Edgar_Degas_323.jpg', 'Edgar_Degas_324.jpg', 'Edgar_Degas_325.jpg', 'Edgar_Degas_326.jpg', 'Edgar_Degas_327.jpg', 'Edgar_Degas_328.jpg', 'Edgar_Degas_329.jpg', 'Edgar_Degas_33.jpg', 'Edgar_Degas_330.jpg', 'Edgar_Degas_331.jpg', 'Edgar_Degas_332.jpg', 'Edgar_Degas_333.jpg', 'Edgar_Degas_334.jpg', 'Edgar_Degas_335.jpg', 'Edgar_Degas_336.jpg', 'Edgar_Degas_337.jpg', 'Edgar_Degas_338.jpg', 'Edgar_Degas_339.jpg', 'Edgar_Degas_34.jpg', 'Edgar_Degas_340.jpg', 'Edgar_Degas_341.jpg', 'Edgar_Degas_342.jpg', 'Edgar_Degas_343.jpg', 'Edgar_Degas_344.jpg', 'Edgar_Degas_345.jpg', 'Edgar_Degas_346.jpg', 'Edgar_Degas_347.jpg', 'Edgar_Degas_348.jpg', 'Edgar_Degas_349.jpg', 'Edgar_Degas_35.jpg', 'Edgar_Degas_350.jpg', 'Edgar_Degas_351.jpg', 'Edgar_Degas_352.jpg', 'Edgar_Degas_353.jpg', 'Edgar_Degas_354.jpg', 'Edgar_Degas_355.jpg', 'Edgar_Degas_356.jpg']
0 223 0 223
Albrecht_DuÔòá├¬rer_1 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_10 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_100 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_101 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_102 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_103 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_104 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_105 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_106 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_107 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_108 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_109 :  (224, 224, 3)
2 221 2 221
Albrecht_DuÔòá├¬rer_11 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_110 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_111 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_112 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_113 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_114 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_115 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_116 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_117 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_118 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_119 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_12 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_120 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_121 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_122 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_123 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_124 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_125 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_126 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_127 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_128 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_129 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_13 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_130 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_131 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_132 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_133 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_134 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_135 :  (224, 224, 3)
2 220 3 222
Albrecht_DuÔòá├¬rer_136 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_137 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_138 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_139 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_14 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_140 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_141 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_142 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_143 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_144 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_145 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_146 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_147 :  (224, 224, 3)
0 223 1 222
Albrecht_DuÔòá├¬rer_148 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_149 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_15 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_150 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_151 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_152 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_153 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_154 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_155 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_156 :  (224, 224, 3)
2 221 2 222
Albrecht_DuÔòá├¬rer_157 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_158 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_159 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_16 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_160 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_161 :  (224, 224, 3)
2 220 2 222
Albrecht_DuÔòá├¬rer_162 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_163 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_164 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_165 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_166 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_167 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_168 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_169 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_17 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_170 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_171 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_172 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_173 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_174 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_175 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_176 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_177 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_178 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_179 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_18 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_180 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_181 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_182 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_183 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_184 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_185 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_186 :  (224, 224, 3)
1 222 2 221
Albrecht_DuÔòá├¬rer_187 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_188 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_189 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_19 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_190 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_191 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_192 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_193 :  (224, 224, 3)
3 221 2 221
Albrecht_DuÔòá├¬rer_194 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_195 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_196 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_197 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_198 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_199 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_2 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_20 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_200 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_201 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_202 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_203 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_204 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_205 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_206 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_207 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_208 :  (224, 224, 3)
4 221 2 221
Albrecht_DuÔòá├¬rer_209 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_21 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_210 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_211 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_212 :  (224, 224, 3)
1 222 3 222
Albrecht_DuÔòá├¬rer_213 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_214 :  (224, 224, 3)
2 220 1 222
Albrecht_DuÔòá├¬rer_215 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_216 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_217 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_218 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_219 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_22 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_220 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_221 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_222 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_223 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_224 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_225 :  (224, 224, 3)
3 220 1 222
Albrecht_DuÔòá├¬rer_226 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_227 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_228 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_229 :  (224, 224, 3)
1 223 0 222
Albrecht_DuÔòá├¬rer_23 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_230 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_231 :  (224, 224, 3)
2 221 2 222
Albrecht_DuÔòá├¬rer_232 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_233 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_234 :  (224, 224, 3)
4 220 2 222
Albrecht_DuÔòá├¬rer_235 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_236 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_237 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_238 :  (224, 224, 3)
0 221 1 223
Albrecht_DuÔòá├¬rer_239 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_24 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_240 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_241 :  (224, 224, 3)
0 221 1 221
Albrecht_DuÔòá├¬rer_242 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_243 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_244 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_245 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_246 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_247 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_248 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_249 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_25 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_250 :  (224, 224, 3)
3 220 2 222
Albrecht_DuÔòá├¬rer_251 :  (224, 224, 3)
2 221 1 223
Albrecht_DuÔòá├¬rer_252 :  (224, 224, 3)
1 222 2 221
Albrecht_DuÔòá├¬rer_253 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_254 :  (224, 224, 3)
2 221 2 222
Albrecht_DuÔòá├¬rer_255 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_256 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_257 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_258 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_259 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_26 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_260 :  (224, 224, 3)
2 221 4 222
Albrecht_DuÔòá├¬rer_261 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_262 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_263 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_264 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_265 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_266 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_267 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_268 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_269 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_27 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_270 :  (224, 224, 3)
3 221 1 222
Albrecht_DuÔòá├¬rer_271 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_272 :  (224, 224, 3)
2 220 1 222
Albrecht_DuÔòá├¬rer_273 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_274 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_275 :  (224, 224, 3)
2 221 1 222
Albrecht_DuÔòá├¬rer_276 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_277 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_278 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_279 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_28 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_280 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_281 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_282 :  (224, 224, 3)
2 221 1 222
Albrecht_DuÔòá├¬rer_283 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_284 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_285 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_286 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_287 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_288 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_289 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_29 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_290 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_291 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_292 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_293 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_294 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_295 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_296 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_297 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_298 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_299 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_3 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_30 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_300 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_301 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_302 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_303 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_304 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_305 :  (224, 224, 3)
0 223 2 221
Albrecht_DuÔòá├¬rer_306 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_307 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_308 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_309 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_31 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_310 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_311 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_312 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_313 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_314 :  (224, 224, 3)
4 219 1 222
Albrecht_DuÔòá├¬rer_315 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_316 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_317 :  (224, 224, 3)
2 222 2 222
Albrecht_DuÔòá├¬rer_318 :  (224, 224, 3)
3 220 2 221
Albrecht_DuÔòá├¬rer_319 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_32 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_320 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_321 :  (224, 224, 3)
3 221 2 221
Albrecht_DuÔòá├¬rer_322 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_323 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_324 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_325 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_326 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_327 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_328 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_33 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_34 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_35 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_36 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_37 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_38 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_39 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_4 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_40 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_41 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_42 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_43 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_44 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_45 :  (224, 224, 3)
1 222 2 222
Albrecht_DuÔòá├¬rer_46 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_47 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_48 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_49 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_5 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_50 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_51 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_52 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_53 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_54 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_55 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_56 :  (224, 224, 3)
1 221 2 222
Albrecht_DuÔòá├¬rer_57 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_58 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_59 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_6 :  (224, 224, 3)
4 219 2 221
Albrecht_DuÔòá├¬rer_60 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_61 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_62 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_63 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_64 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_65 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_66 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_67 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_68 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_69 :  (224, 224, 3)
2 222 1 222
Albrecht_DuÔòá├¬rer_7 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_70 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_71 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_72 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_73 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_74 :  (224, 224, 3)
3 220 1 222
Albrecht_DuÔòá├¬rer_75 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_76 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_77 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_78 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_79 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_8 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_80 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_81 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_82 :  (224, 224, 3)
3 218 2 221
Albrecht_DuÔòá├¬rer_83 :  (224, 224, 3)
2 222 2 221
Albrecht_DuÔòá├¬rer_84 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_85 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_86 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_87 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_88 :  (224, 224, 3)
2 221 3 221
Albrecht_DuÔòá├¬rer_89 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_9 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_90 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_91 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_92 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_93 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_94 :  (224, 224, 3)
2 222 1 223
Albrecht_DuÔòá├¬rer_95 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_96 :  (224, 224, 3)
1 220 2 222
Albrecht_DuÔòá├¬rer_97 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_98 :  (224, 224, 3)
0 223 0 223
Albrecht_DuÔòá├¬rer_99 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_1 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_10 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_100 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_101 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_102 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_103 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_104 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_105 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_106 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_107 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_108 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_109 :  (224, 224, 3)
2 221 2 221
Albrecht_Du╠êrer_11 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_110 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_111 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_112 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_113 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_114 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_115 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_116 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_117 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_118 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_119 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_12 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_120 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_121 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_122 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_123 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_124 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_125 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_126 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_127 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_128 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_129 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_13 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_130 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_131 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_132 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_133 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_134 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_135 :  (224, 224, 3)
2 220 3 222
Albrecht_Du╠êrer_136 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_137 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_138 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_139 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_14 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_140 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_141 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_142 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_143 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_144 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_145 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_146 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_147 :  (224, 224, 3)
0 223 1 222
Albrecht_Du╠êrer_148 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_149 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_15 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_150 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_151 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_152 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_153 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_154 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_155 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_156 :  (224, 224, 3)
2 221 2 222
Albrecht_Du╠êrer_157 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_158 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_159 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_16 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_160 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_161 :  (224, 224, 3)
2 220 2 222
Albrecht_Du╠êrer_162 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_163 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_164 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_165 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_166 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_167 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_168 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_169 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_17 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_170 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_171 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_172 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_173 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_174 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_175 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_176 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_177 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_178 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_179 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_18 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_180 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_181 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_182 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_183 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_184 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_185 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_186 :  (224, 224, 3)
1 222 2 221
Albrecht_Du╠êrer_187 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_188 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_189 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_19 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_190 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_191 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_192 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_193 :  (224, 224, 3)
3 221 2 221
Albrecht_Du╠êrer_194 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_195 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_196 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_197 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_198 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_199 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_2 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_20 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_200 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_201 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_202 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_203 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_204 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_205 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_206 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_207 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_208 :  (224, 224, 3)
4 221 2 221
Albrecht_Du╠êrer_209 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_21 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_210 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_211 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_212 :  (224, 224, 3)
1 222 3 222
Albrecht_Du╠êrer_213 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_214 :  (224, 224, 3)
2 220 1 222
Albrecht_Du╠êrer_215 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_216 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_217 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_218 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_219 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_22 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_220 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_221 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_222 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_223 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_224 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_225 :  (224, 224, 3)
3 220 1 222
Albrecht_Du╠êrer_226 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_227 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_228 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_229 :  (224, 224, 3)
1 223 0 222
Albrecht_Du╠êrer_23 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_230 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_231 :  (224, 224, 3)
2 221 2 222
Albrecht_Du╠êrer_232 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_233 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_234 :  (224, 224, 3)
4 220 2 222
Albrecht_Du╠êrer_235 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_236 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_237 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_238 :  (224, 224, 3)
0 221 1 223
Albrecht_Du╠êrer_239 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_24 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_240 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_241 :  (224, 224, 3)
0 221 1 221
Albrecht_Du╠êrer_242 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_243 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_244 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_245 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_246 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_247 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_248 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_249 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_25 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_250 :  (224, 224, 3)
3 220 2 222
Albrecht_Du╠êrer_251 :  (224, 224, 3)
2 221 1 223
Albrecht_Du╠êrer_252 :  (224, 224, 3)
1 222 2 221
Albrecht_Du╠êrer_253 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_254 :  (224, 224, 3)
2 221 2 222
Albrecht_Du╠êrer_255 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_256 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_257 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_258 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_259 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_26 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_260 :  (224, 224, 3)
2 221 4 222
Albrecht_Du╠êrer_261 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_262 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_263 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_264 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_265 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_266 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_267 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_268 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_269 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_27 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_270 :  (224, 224, 3)
3 221 1 222
Albrecht_Du╠êrer_271 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_272 :  (224, 224, 3)
2 220 1 222
Albrecht_Du╠êrer_273 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_274 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_275 :  (224, 224, 3)
2 221 1 222
Albrecht_Du╠êrer_276 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_277 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_278 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_279 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_28 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_280 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_281 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_282 :  (224, 224, 3)
2 221 1 222
Albrecht_Du╠êrer_283 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_284 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_285 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_286 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_287 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_288 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_289 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_29 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_290 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_291 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_292 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_293 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_294 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_295 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_296 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_297 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_298 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_299 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_3 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_30 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_300 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_301 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_302 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_303 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_304 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_305 :  (224, 224, 3)
0 223 2 221
Albrecht_Du╠êrer_306 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_307 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_308 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_309 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_31 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_310 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_311 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_312 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_313 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_314 :  (224, 224, 3)
4 219 1 222
Albrecht_Du╠êrer_315 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_316 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_317 :  (224, 224, 3)
2 222 2 222
Albrecht_Du╠êrer_318 :  (224, 224, 3)
3 220 2 221
Albrecht_Du╠êrer_319 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_32 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_320 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_321 :  (224, 224, 3)
3 221 2 221
Albrecht_Du╠êrer_322 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_323 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_324 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_325 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_326 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_327 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_328 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_33 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_34 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_35 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_36 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_37 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_38 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_39 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_4 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_40 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_41 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_42 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_43 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_44 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_45 :  (224, 224, 3)
1 222 2 222
Albrecht_Du╠êrer_46 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_47 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_48 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_49 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_5 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_50 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_51 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_52 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_53 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_54 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_55 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_56 :  (224, 224, 3)
1 221 2 222
Albrecht_Du╠êrer_57 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_58 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_59 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_6 :  (224, 224, 3)
4 219 2 221
Albrecht_Du╠êrer_60 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_61 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_62 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_63 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_64 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_65 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_66 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_67 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_68 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_69 :  (224, 224, 3)
2 222 1 222
Albrecht_Du╠êrer_7 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_70 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_71 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_72 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_73 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_74 :  (224, 224, 3)
3 220 1 222
Albrecht_Du╠êrer_75 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_76 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_77 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_78 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_79 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_8 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_80 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_81 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_82 :  (224, 224, 3)
3 218 2 221
Albrecht_Du╠êrer_83 :  (224, 224, 3)
2 222 2 221
Albrecht_Du╠êrer_84 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_85 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_86 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_87 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_88 :  (224, 224, 3)
2 221 3 221
Albrecht_Du╠êrer_89 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_9 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_90 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_91 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_92 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_93 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_94 :  (224, 224, 3)
2 222 1 223
Albrecht_Du╠êrer_95 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_96 :  (224, 224, 3)
1 220 2 222
Albrecht_Du╠êrer_97 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_98 :  (224, 224, 3)
0 223 0 223
Albrecht_Du╠êrer_99 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_1 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_10 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_100 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_101 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_102 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_103 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_104 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_105 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_106 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_107 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_108 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_109 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_11 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_110 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_111 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_112 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_113 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_114 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_115 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_116 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_117 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_118 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_119 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_12 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_120 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_121 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_122 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_123 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_124 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_125 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_126 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_127 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_128 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_129 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_13 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_130 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_131 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_132 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_133 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_134 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_135 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_136 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_137 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_138 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_139 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_14 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_140 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_141 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_142 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_143 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_144 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_145 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_146 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_147 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_148 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_149 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_15 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_150 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_151 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_152 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_153 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_154 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_155 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_156 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_157 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_158 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_159 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_16 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_160 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_161 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_162 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_163 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_164 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_165 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_166 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_167 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_168 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_169 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_17 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_170 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_171 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_172 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_173 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_174 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_175 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_176 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_177 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_178 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_179 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_18 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_180 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_181 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_182 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_183 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_184 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_185 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_186 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_187 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_188 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_189 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_19 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_190 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_191 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_192 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_193 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_194 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_195 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_196 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_197 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_198 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_199 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_2 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_20 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_200 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_201 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_202 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_203 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_204 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_205 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_206 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_207 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_208 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_209 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_21 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_210 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_211 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_212 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_213 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_214 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_215 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_216 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_217 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_218 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_219 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_22 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_220 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_221 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_222 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_223 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_224 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_225 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_226 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_227 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_228 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_229 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_23 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_230 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_231 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_232 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_233 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_234 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_235 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_236 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_237 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_238 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_239 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_24 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_240 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_241 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_242 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_243 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_244 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_245 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_246 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_247 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_248 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_249 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_25 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_250 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_251 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_252 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_253 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_254 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_255 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_256 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_257 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_258 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_259 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_26 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_27 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_28 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_29 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_3 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_30 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_31 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_32 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_33 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_34 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_35 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_36 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_37 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_38 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_39 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_4 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_40 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_41 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_42 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_43 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_44 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_45 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_46 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_47 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_48 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_49 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_5 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_50 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_51 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_52 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_53 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_54 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_55 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_56 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_57 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_58 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_59 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_6 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_60 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_61 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_62 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_63 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_64 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_65 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_66 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_67 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_68 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_69 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_7 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_70 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_71 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_72 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_73 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_74 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_75 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_76 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_77 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_78 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_79 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_8 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_80 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_81 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_82 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_83 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_84 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_85 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_86 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_87 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_88 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_89 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_9 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_90 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_91 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_92 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_93 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_94 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_95 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_96 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_97 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_98 :  (224, 224, 3)
0 223 0 223
Alfred_Sisley_99 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_1 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_10 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_100 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_101 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_102 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_103 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_104 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_105 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_106 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_107 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_108 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_109 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_11 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_110 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_111 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_112 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_113 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_114 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_115 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_116 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_117 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_118 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_119 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_12 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_120 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_121 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_122 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_123 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_124 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_125 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_126 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_127 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_128 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_129 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_13 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_130 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_131 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_132 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_133 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_134 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_135 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_136 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_137 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_138 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_139 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_14 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_140 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_141 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_142 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_143 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_144 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_145 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_146 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_147 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_148 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_149 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_15 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_150 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_151 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_152 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_153 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_154 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_155 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_156 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_157 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_158 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_159 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_16 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_160 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_161 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_162 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_163 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_164 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_165 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_166 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_167 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_168 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_169 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_17 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_170 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_171 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_172 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_173 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_174 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_175 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_176 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_177 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_178 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_179 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_18 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_180 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_181 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_182 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_183 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_184 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_185 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_186 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_187 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_188 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_189 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_19 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_190 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_191 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_192 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_193 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_2 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_20 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_21 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_22 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_23 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_24 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_25 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_26 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_27 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_28 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_29 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_3 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_30 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_31 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_32 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_33 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_34 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_35 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_36 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_37 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_38 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_39 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_4 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_40 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_41 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_42 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_43 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_44 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_45 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_46 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_47 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_48 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_49 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_5 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_50 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_51 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_52 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_53 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_54 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_55 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_56 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_57 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_58 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_59 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_6 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_60 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_61 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_62 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_63 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_64 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_65 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_66 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_67 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_68 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_69 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_7 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_70 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_71 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_72 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_73 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_74 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_75 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_76 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_77 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_78 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_79 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_8 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_80 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_81 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_82 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_83 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_84 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_85 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_86 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_87 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_88 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_89 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_9 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_90 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_91 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_92 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_93 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_94 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_95 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_96 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_97 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_98 :  (224, 224, 3)
0 223 0 223
Amedeo_Modigliani_99 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_1 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_10 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_11 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_12 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_13 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_14 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_15 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_16 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_17 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_18 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_19 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_2 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_20 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_21 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_22 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_23 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_24 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_25 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_26 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_27 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_28 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_29 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_3 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_30 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_31 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_32 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_33 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_34 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_35 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_36 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_37 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_38 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_39 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_4 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_40 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_41 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_42 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_43 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_44 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_45 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_46 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_47 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_48 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_49 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_5 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_50 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_51 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_52 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_53 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_54 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_55 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_56 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_57 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_58 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_59 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_6 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_60 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_61 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_62 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_63 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_64 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_65 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_66 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_67 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_68 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_69 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_7 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_70 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_71 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_72 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_73 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_74 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_75 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_76 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_77 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_78 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_79 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_8 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_80 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_81 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_82 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_83 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_84 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_85 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_86 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_87 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_88 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_89 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_9 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_90 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_91 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_92 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_93 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_94 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_95 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_96 :  (224, 224, 3)
0 222 0 223
Andrei_Rublev_97 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_98 :  (224, 224, 3)
0 223 0 223
Andrei_Rublev_99 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_1 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_10 :  (224, 224, 3)
0 223 6 223
Andy_Warhol_100 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_101 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_102 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_103 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_104 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_105 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_106 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_107 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_108 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_109 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_11 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_110 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_111 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_112 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_113 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_114 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_115 :  (224, 224, 3)
10 214 8 220
Andy_Warhol_116 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_117 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_118 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_119 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_12 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_120 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_121 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_122 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_123 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_124 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_125 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_126 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_127 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_128 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_129 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_13 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_130 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_131 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_132 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_133 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_134 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_135 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_136 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_137 :  (224, 224, 3)
0 222 0 223
Andy_Warhol_138 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_139 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_14 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_140 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_141 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_142 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_143 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_144 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_145 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_146 :  (224, 224, 3)
5 217 4 218
Andy_Warhol_147 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_148 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_149 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_15 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_150 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_151 :  (224, 224, 3)
0 217 0 223
Andy_Warhol_152 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_153 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_154 :  (224, 224, 3)
25 202 13 205
Andy_Warhol_155 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_156 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_157 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_158 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_159 :  (224, 224, 3)
0 222 0 218
Andy_Warhol_16 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_160 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_161 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_162 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_163 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_164 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_165 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_166 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_167 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_168 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_169 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_17 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_170 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_171 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_172 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_173 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_174 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_175 :  (224, 224, 3)
48 177 48 176
Andy_Warhol_176 :  (224, 224, 3)
29 223 0 223
Andy_Warhol_177 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_178 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_179 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_18 :  (224, 224, 3)
0 223 0 216
Andy_Warhol_180 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_181 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_19 :  (224, 224, 3)
13 223 0 223
Andy_Warhol_2 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_20 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_21 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_22 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_23 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_24 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_25 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_26 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_27 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_28 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_29 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_3 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_30 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_31 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_32 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_33 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_34 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_35 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_36 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_37 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_38 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_39 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_4 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_40 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_41 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_42 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_43 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_44 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_45 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_46 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_47 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_48 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_49 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_5 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_50 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_51 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_52 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_53 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_54 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_55 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_56 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_57 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_58 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_59 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_6 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_60 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_61 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_62 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_63 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_64 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_65 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_66 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_67 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_68 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_69 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_7 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_70 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_71 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_72 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_73 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_74 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_75 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_76 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_77 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_78 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_79 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_8 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_80 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_81 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_82 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_83 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_84 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_85 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_86 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_87 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_88 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_89 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_9 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_90 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_91 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_92 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_93 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_94 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_95 :  (224, 224, 3)
2 223 0 223
Andy_Warhol_96 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_97 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_98 :  (224, 224, 3)
0 223 0 223
Andy_Warhol_99 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_1 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_10 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_11 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_12 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_13 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_14 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_15 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_16 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_17 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_18 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_19 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_2 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_20 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_21 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_22 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_23 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_24 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_25 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_26 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_27 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_28 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_29 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_3 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_30 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_31 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_32 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_33 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_34 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_35 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_36 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_37 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_38 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_39 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_4 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_40 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_41 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_42 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_43 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_44 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_45 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_46 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_47 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_48 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_49 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_5 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_50 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_51 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_52 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_53 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_54 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_55 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_56 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_57 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_58 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_59 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_6 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_60 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_61 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_62 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_63 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_64 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_65 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_66 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_67 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_68 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_69 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_7 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_70 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_71 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_72 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_73 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_74 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_75 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_76 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_77 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_78 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_79 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_8 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_80 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_81 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_82 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_83 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_84 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_85 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_86 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_87 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_88 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_89 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_9 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_90 :  (224, 224, 3)
0 223 0 223
Camille_Pissarro_91 :  (224, 224, 3)
0 223 0 223
Caravaggio_1 :  (224, 224, 3)
0 223 0 223
Caravaggio_10 :  (224, 224, 3)
0 223 0 223
Caravaggio_11 :  (224, 224, 3)
0 223 0 223
Caravaggio_12 :  (224, 224, 3)
0 223 0 223
Caravaggio_13 :  (224, 224, 3)
0 223 0 223
Caravaggio_14 :  (224, 224, 3)
0 223 0 223
Caravaggio_15 :  (224, 224, 3)
0 223 0 223
Caravaggio_16 :  (224, 224, 3)
0 223 0 223
Caravaggio_17 :  (224, 224, 3)
0 223 0 223
Caravaggio_18 :  (224, 224, 3)
0 223 0 223
Caravaggio_19 :  (224, 224, 3)
0 223 0 223
Caravaggio_2 :  (224, 224, 3)
0 223 0 223
Caravaggio_20 :  (224, 224, 3)
0 223 0 223
Caravaggio_21 :  (224, 224, 3)
0 223 0 223
Caravaggio_22 :  (224, 224, 3)
0 223 0 223
Caravaggio_23 :  (224, 224, 3)
0 223 0 223
Caravaggio_24 :  (224, 224, 3)
2 221 2 221
Caravaggio_25 :  (224, 224, 3)
0 223 0 223
Caravaggio_26 :  (224, 224, 3)
0 223 0 223
Caravaggio_27 :  (224, 224, 3)
0 223 0 223
Caravaggio_28 :  (224, 224, 3)
0 223 0 223
Caravaggio_29 :  (224, 224, 3)
0 223 0 223
Caravaggio_3 :  (224, 224, 3)
0 223 0 223
Caravaggio_30 :  (224, 224, 3)
0 223 0 223
Caravaggio_31 :  (224, 224, 3)
0 223 0 223
Caravaggio_32 :  (224, 224, 3)
0 223 0 223
Caravaggio_33 :  (224, 224, 3)
0 223 0 223
Caravaggio_34 :  (224, 224, 3)
0 223 0 223
Caravaggio_35 :  (224, 224, 3)
0 223 0 223
Caravaggio_36 :  (224, 224, 3)
0 223 0 223
Caravaggio_37 :  (224, 224, 3)
0 223 0 223
Caravaggio_38 :  (224, 224, 3)
0 223 0 223
Caravaggio_39 :  (224, 224, 3)
0 223 0 223
Caravaggio_4 :  (224, 224, 3)
0 223 0 223
Caravaggio_40 :  (224, 224, 3)
0 223 0 223
Caravaggio_41 :  (224, 224, 3)
0 223 0 223
Caravaggio_42 :  (224, 224, 3)
0 223 0 223
Caravaggio_43 :  (224, 224, 3)
0 223 0 223
Caravaggio_44 :  (224, 224, 3)
0 223 0 223
Caravaggio_45 :  (224, 224, 3)
0 223 0 223
Caravaggio_46 :  (224, 224, 3)
0 223 0 223
Caravaggio_47 :  (224, 224, 3)
0 223 0 223
Caravaggio_48 :  (224, 224, 3)
0 223 0 223
Caravaggio_49 :  (224, 224, 3)
0 223 0 223
Caravaggio_5 :  (224, 224, 3)
0 223 0 223
Caravaggio_50 :  (224, 224, 3)
0 223 0 223
Caravaggio_51 :  (224, 224, 3)
0 223 0 223
Caravaggio_52 :  (224, 224, 3)
0 223 0 223
Caravaggio_53 :  (224, 224, 3)
0 223 0 223
Caravaggio_54 :  (224, 224, 3)
0 223 0 223
Caravaggio_55 :  (224, 224, 3)
0 223 0 223
Caravaggio_6 :  (224, 224, 3)
0 223 0 223
Caravaggio_7 :  (224, 224, 3)
0 223 0 223
Caravaggio_8 :  (224, 224, 3)
0 223 0 223
Caravaggio_9 :  (224, 224, 3)
0 223 0 223
Claude_Monet_1 :  (224, 224, 3)
0 223 0 223
Claude_Monet_10 :  (224, 224, 3)
0 223 0 223
Claude_Monet_11 :  (224, 224, 3)
0 223 0 223
Claude_Monet_12 :  (224, 224, 3)
0 223 0 223
Claude_Monet_13 :  (224, 224, 3)
0 223 0 223
Claude_Monet_14 :  (224, 224, 3)
0 223 0 223
Claude_Monet_15 :  (224, 224, 3)
0 223 0 223
Claude_Monet_16 :  (224, 224, 3)
0 223 0 223
Claude_Monet_17 :  (224, 224, 3)
0 223 0 223
Claude_Monet_18 :  (224, 224, 3)
0 223 0 223
Claude_Monet_19 :  (224, 224, 3)
0 223 0 223
Claude_Monet_2 :  (224, 224, 3)
0 223 0 223
Claude_Monet_20 :  (224, 224, 3)
0 223 0 223
Claude_Monet_21 :  (224, 224, 3)
0 223 0 223
Claude_Monet_22 :  (224, 224, 3)
0 223 0 223
Claude_Monet_23 :  (224, 224, 3)
0 223 0 223
Claude_Monet_24 :  (224, 224, 3)
0 223 0 223
Claude_Monet_25 :  (224, 224, 3)
0 223 0 223
Claude_Monet_26 :  (224, 224, 3)
0 223 0 223
Claude_Monet_27 :  (224, 224, 3)
0 223 0 223
Claude_Monet_28 :  (224, 224, 3)
0 223 0 223
Claude_Monet_29 :  (224, 224, 3)
0 223 0 223
Claude_Monet_3 :  (224, 224, 3)
0 223 0 223
Claude_Monet_30 :  (224, 224, 3)
0 223 0 223
Claude_Monet_31 :  (224, 224, 3)
0 223 0 223
Claude_Monet_32 :  (224, 224, 3)
0 223 0 223
Claude_Monet_33 :  (224, 224, 3)
0 223 0 223
Claude_Monet_34 :  (224, 224, 3)
0 223 0 223
Claude_Monet_35 :  (224, 224, 3)
0 223 0 223
Claude_Monet_36 :  (224, 224, 3)
0 223 0 223
Claude_Monet_37 :  (224, 224, 3)
0 223 0 223
Claude_Monet_38 :  (224, 224, 3)
0 223 0 223
Claude_Monet_39 :  (224, 224, 3)
0 223 0 223
Claude_Monet_4 :  (224, 224, 3)
0 223 0 223
Claude_Monet_40 :  (224, 224, 3)
0 223 0 223
Claude_Monet_41 :  (224, 224, 3)
0 223 0 223
Claude_Monet_42 :  (224, 224, 3)
0 223 0 223
Claude_Monet_43 :  (224, 224, 3)
0 223 0 223
Claude_Monet_44 :  (224, 224, 3)
0 223 0 223
Claude_Monet_45 :  (224, 224, 3)
0 223 0 223
Claude_Monet_46 :  (224, 224, 3)
0 223 0 223
Claude_Monet_47 :  (224, 224, 3)
0 223 0 223
Claude_Monet_48 :  (224, 224, 3)
0 223 0 223
Claude_Monet_49 :  (224, 224, 3)
0 223 0 223
Claude_Monet_5 :  (224, 224, 3)
0 223 0 223
Claude_Monet_50 :  (224, 224, 3)
0 223 0 223
Claude_Monet_51 :  (224, 224, 3)
0 223 0 223
Claude_Monet_52 :  (224, 224, 3)
0 223 0 223
Claude_Monet_53 :  (224, 224, 3)
0 223 0 223
Claude_Monet_54 :  (224, 224, 3)
0 223 0 223
Claude_Monet_55 :  (224, 224, 3)
0 223 0 223
Claude_Monet_56 :  (224, 224, 3)
0 223 0 223
Claude_Monet_57 :  (224, 224, 3)
0 223 0 223
Claude_Monet_58 :  (224, 224, 3)
0 223 0 223
Claude_Monet_59 :  (224, 224, 3)
0 223 0 223
Claude_Monet_6 :  (224, 224, 3)
0 223 0 223
Claude_Monet_60 :  (224, 224, 3)
0 223 0 223
Claude_Monet_61 :  (224, 224, 3)
0 223 0 223
Claude_Monet_62 :  (224, 224, 3)
0 223 0 223
Claude_Monet_63 :  (224, 224, 3)
0 223 0 223
Claude_Monet_64 :  (224, 224, 3)
0 223 0 223
Claude_Monet_65 :  (224, 224, 3)
0 223 0 223
Claude_Monet_66 :  (224, 224, 3)
0 223 0 223
Claude_Monet_67 :  (224, 224, 3)
0 223 0 223
Claude_Monet_68 :  (224, 224, 3)
0 223 0 223
Claude_Monet_69 :  (224, 224, 3)
0 223 0 223
Claude_Monet_7 :  (224, 224, 3)
0 223 0 223
Claude_Monet_70 :  (224, 224, 3)
0 223 0 223
Claude_Monet_71 :  (224, 224, 3)
0 223 0 223
Claude_Monet_72 :  (224, 224, 3)
0 223 0 223
Claude_Monet_73 :  (224, 224, 3)
0 223 0 223
Claude_Monet_8 :  (224, 224, 3)
0 223 0 223
Claude_Monet_9 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_1 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_10 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_11 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_12 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_13 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_14 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_15 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_16 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_17 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_18 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_19 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_2 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_20 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_21 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_22 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_23 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_24 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_25 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_26 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_27 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_28 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_29 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_3 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_30 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_31 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_32 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_33 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_34 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_35 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_36 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_37 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_38 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_39 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_4 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_40 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_41 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_42 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_43 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_44 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_45 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_46 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_47 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_48 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_49 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_5 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_50 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_51 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_52 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_53 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_54 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_55 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_56 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_57 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_58 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_59 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_6 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_60 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_61 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_62 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_63 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_64 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_65 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_66 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_67 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_68 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_69 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_7 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_70 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_8 :  (224, 224, 3)
0 223 0 223
Diego_Rivera_9 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_1 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_10 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_100 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_101 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_102 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_103 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_104 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_105 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_106 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_107 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_108 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_109 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_11 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_110 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_111 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_112 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_113 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_114 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_115 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_116 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_117 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_118 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_119 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_12 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_120 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_121 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_122 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_123 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_124 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_125 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_126 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_127 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_128 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_13 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_14 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_15 :  (224, 224, 3)
0 223 0 223
Diego_Velazquez_16 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_1 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_10 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_100 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_101 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_102 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_103 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_104 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_105 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_106 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_107 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_108 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_109 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_11 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_110 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_111 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_112 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_113 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_114 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_115 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_116 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_117 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_118 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_119 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_12 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_120 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_121 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_122 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_123 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_124 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_125 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_126 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_127 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_128 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_129 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_13 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_130 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_131 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_132 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_133 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_134 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_135 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_136 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_137 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_138 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_139 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_14 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_140 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_141 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_142 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_143 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_144 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_145 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_146 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_147 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_148 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_149 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_15 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_150 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_151 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_152 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_153 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_154 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_155 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_156 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_157 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_158 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_159 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_16 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_160 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_161 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_162 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_163 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_164 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_165 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_166 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_167 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_168 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_169 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_17 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_170 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_171 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_172 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_173 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_174 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_175 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_176 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_177 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_178 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_179 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_18 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_180 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_181 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_182 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_183 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_184 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_185 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_186 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_187 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_188 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_189 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_19 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_190 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_191 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_192 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_193 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_194 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_195 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_196 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_197 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_198 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_199 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_2 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_20 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_200 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_201 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_202 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_203 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_204 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_205 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_206 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_207 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_208 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_209 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_21 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_210 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_211 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_212 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_213 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_214 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_215 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_216 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_217 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_218 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_219 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_22 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_220 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_221 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_222 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_223 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_224 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_225 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_226 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_227 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_228 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_229 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_23 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_230 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_231 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_232 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_233 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_234 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_235 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_236 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_237 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_238 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_239 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_24 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_240 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_241 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_242 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_243 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_244 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_245 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_246 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_247 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_248 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_249 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_25 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_250 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_251 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_252 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_253 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_254 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_255 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_256 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_257 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_258 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_259 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_26 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_260 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_261 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_262 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_263 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_264 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_265 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_266 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_267 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_268 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_269 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_27 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_270 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_271 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_272 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_273 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_274 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_275 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_276 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_277 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_278 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_279 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_28 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_280 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_281 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_282 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_283 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_284 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_285 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_286 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_287 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_288 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_289 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_29 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_290 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_291 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_292 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_293 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_294 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_295 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_296 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_297 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_298 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_299 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_3 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_30 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_300 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_301 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_302 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_303 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_304 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_305 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_306 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_307 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_308 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_309 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_31 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_310 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_311 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_312 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_313 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_314 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_315 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_316 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_317 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_318 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_319 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_32 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_320 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_321 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_322 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_323 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_324 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_325 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_326 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_327 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_328 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_329 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_33 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_330 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_331 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_332 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_333 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_334 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_335 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_336 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_337 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_338 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_339 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_34 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_340 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_341 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_342 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_343 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_344 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_345 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_346 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_347 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_348 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_349 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_35 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_350 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_351 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_352 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_353 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_354 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_355 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_356 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_366 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_37 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_438 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_44 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_540 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_584 :  (224, 224, 3)
0 223 0 223
Edgar_Degas_639 :  (224, 224, 3)
0 223 3 223
Kazimir_Malevich_100 :  (224, 224, 3)
0 223 0 223
Kazimir_Malevich_115 :  (224, 224, 3)
0 223 0 223
Kazimir_Malevich_124 :  (224, 224, 3)
0 223 0 223
Kazimir_Malevich_72 :  (224, 224, 3)
0 223 0 223
Leonardo_da_Vinci_11 :  (224, 224, 3)
0 222 0 222
Leonardo_da_Vinci_110 :  (224, 224, 3)
0 223 0 223
Leonardo_da_Vinci_121 :  (224, 224, 3)
0 223 0 223
Leonardo_da_Vinci_122 :  (224, 224, 3)
0 223 0 223
Leonardo_da_Vinci_124 :  (224, 224, 3)
0 223 0 223
Leonardo_da_Vinci_45 :  (224, 224, 3)
0 223 0 223
Leonardo_da_Vinci_89 :  (224, 224, 3)
0 223 0 223
Marc_Chagall_123 :  (224, 224, 3)
0 223 0 223
Marc_Chagall_152 :  (224, 224, 3)
0 223 0 223
Marc_Chagall_170 :  (224, 224, 3)
0 223 0 223
Marc_Chagall_178 :  (224, 224, 3)
0 223 0 223
Marc_Chagall_183 :  (224, 224, 3)
0 223 0 223
Marc_Chagall_2 :  (224, 224, 3)
0 223 0 223
Marc_Chagall_52 :  (224, 224, 3)
0 223 0 223
Marc_Chagall_56 :  (224, 224, 3)
0 223 0 223
Marc_Chagall_8 :  (224, 224, 3)
0 223 0 223
Marc_Chagall_92 :  (224, 224, 3)
0 223 0 223
Michelangelo_16 :  (224, 224, 3)
0 223 0 223
Michelangelo_21 :  (224, 224, 3)
0 223 0 223
Mikhail_Vrubel_140 :  (224, 224, 3)
0 223 0 223
Mikhail_Vrubel_16 :  (224, 224, 3)
0 223 0 223
Mikhail_Vrubel_36 :  (224, 224, 3)
0 223 0 223
Mikhail_Vrubel_5 :  (224, 224, 3)
0 223 0 223
Mikhail_Vrubel_67 :  (224, 224, 3)
0 223 0 223
Mikhail_Vrubel_96 :  (224, 224, 3)
0 223 0 223
Pablo_Picasso_249 :  (224, 224, 3)
0 223 0 223
Pablo_Picasso_305 :  (224, 224, 3)
0 223 0 223
Pablo_Picasso_322 :  (224, 224, 3)
0 223 0 223
Pablo_Picasso_381 :  (224, 224, 3)
0 223 0 223
Pablo_Picasso_438 :  (224, 224, 3)
0 223 0 223
Pablo_Picasso_46 :  (224, 224, 3)
0 223 0 223
Paul_Gauguin_107 :  (224, 224, 3)
0 223 0 223
Paul_Gauguin_116 :  (224, 224, 3)
0 223 0 223
Paul_Gauguin_290 :  (224, 224, 3)
0 223 0 223
Paul_Gauguin_49 :  (224, 224, 3)
0 223 0 223
Paul_Klee_1 :  (224, 224, 3)
0 223 0 223
Paul_Klee_14 :  (224, 224, 3)
0 223 0 223
Paul_Klee_23 :  (224, 224, 3)
0 223 0 223
Paul_Klee_38 :  (224, 224, 3)
0 223 0 223
Pierre-Auguste_Renoir_142 :  (224, 224, 3)
0 223 0 223
Pierre-Auguste_Renoir_257 :  (224, 224, 3)
0 223 0 223
Rene_Magritte_138 :  (224, 224, 3)
0 223 0 223
Salvador_Dali_35 :  (224, 224, 3)
0 223 0 223
Salvador_Dali_61 :  (224, 224, 3)
0 223 0 223
Salvador_Dali_8 :  (224, 224, 3)
0 223 0 223
Sandro_Botticelli_74 :  (224, 224, 3)
0 223 0 223
Titian_81 :  (224, 224, 3)
0 223 0 223
Vincent_van_Gogh_276 :  (224, 224, 3)
0 223 0 223
Vincent_van_Gogh_281 :  (224, 224, 3)
0 223 0 223
Vincent_van_Gogh_309 :  (224, 224, 3)
0 223 0 223
Vincent_van_Gogh_645 :  (224, 224, 3)
0 223 0 223
William_Turner_65 :  (224, 224, 3)
(224, 224, 3)
Forma del arreglo combinado: (2063, 224, 224, 3)
2063
2063
```


**[Celda 15 - Código]**
```python
print(image_paths)
```


*Salida:*
```text
None
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
        dist = (dist+dist2)/2
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
   Imagen de consulta:  Alfred_Sisley_110
   Mas cercanos:  [[20.04587173461914, 'Alfred_Sisley_110'], [39.53287124633789, 'Albrecht_DuÔòá├¬rer_176'], [39.53288269042969, 'Albrecht_Du╠êrer_176'], [40.05213165283203, 'Albrecht_DuÔòá├¬rer_308'], [40.0521354675293, 'Albrecht_Du╠êrer_308'], [40.290748596191406, 'Albrecht_DuÔòá├¬rer_154'], [40.290748596191406, 'Albrecht_Du╠êrer_154'], [40.36202621459961, 'Albrecht_DuÔòá├¬rer_123'], [40.36202621459961, 'Albrecht_Du╠êrer_123'], [40.56928253173828, 'Alfred_Sisley_233']]
--> ACIERTO
    Primer Lugar: 1
    Entre los tres primeros: 1
<Figure size 2000x400 with 6 Axes>



Consulta número:  1
   Imagen de consulta:  Alfred_Sisley_84
   Mas cercanos:  [[5.449702739715576, 'Alfred_Sisley_84'], [27.546382904052734, 'Albrecht_DuÔòá├¬rer_229'], [27.546382904052734, 'Albrecht_Du╠êrer_229'], [28.282392501831055, 'Alfred_Sisley_161'], [28.333654403686523, 'Alfred_Sisley_9'], [28.496753692626953, 'Claude_Monet_32'], [28.60931396484375, 'Alfred_Sisley_141'], [28.91187858581543, 'Albrecht_DuÔòá├¬rer_75'], [28.91187858581543, 'Albrecht_Du╠êrer_75'], [28.94503402709961, 'Camille_Pissarro_54']]
--> ACIERTO
    Primer Lugar: 2
    Entre los tres primeros: 2
<Figure size 2000x400 with 6 Axes>



Consulta número:  2
   Imagen de consulta:  Alfred_Sisley_95
   Mas cercanos:  [[14.985897064208984, 'Alfred_Sisley_95'], [25.550649642944336, 'Camille_Pissarro_70'], [25.619199752807617, 'Alfred_Sisley_114'], [25.687902450561523, 'Camille_Pissarro_31'], [25.748687744140625, 'Alfred_Sisley_175'], [25.842615127563477, 'Alfred_Sisley_184'], [25.84796905517578, 'Alfred_Sisley_80'], [25.852123260498047, 'Alfred_Sisley_53'], [25.866600036621094, 'Alfred_Sisley_193'], [25.88623046875, 'Camille_Pissarro_28']]
--> ACIERTO
    Primer Lugar: 3
    Entre los tres primeros: 3
<Figure size 2000x400 with 6 Axes>



Consulta número:  3
   Imagen de consulta:  Amedeo_Modigliani_114
   Mas cercanos:  [[20.342824935913086, 'Amedeo_Modigliani_114'], [31.593666076660156, 'Alfred_Sisley_38'], [31.927200317382812, 'Amedeo_Modigliani_54'], [31.95206069946289, 'Camille_Pissarro_50'], [31.97067642211914, 'Albrecht_DuÔòá├¬rer_57'], [31.97067642211914, 'Albrecht_Du╠êrer_57'], [31.995006561279297, 'Albrecht_DuÔòá├¬rer_73'], [31.995014190673828, 'Albrecht_Du╠êrer_73'], [32.035526275634766, 'Albrecht_DuÔòá├¬rer_214'], [32.035526275634766, 'Albrecht_Du╠êrer_214']]
--> ACIERTO
    Primer Lugar: 4
    Entre los tres primeros: 4
<Figure size 2000x400 with 6 Axes>



Consulta número:  4
   Imagen de consulta:  Amedeo_Modigliani_172
   Mas cercanos:  [[3.105778217315674, 'Amedeo_Modigliani_172'], [8.81093692779541, 'Amedeo_Modigliani_8'], [19.341867446899414, 'Amedeo_Modigliani_142'], [19.380096435546875, 'Amedeo_Modigliani_7'], [20.191225051879883, 'Albrecht_DuÔòá├¬rer_225'], [20.191234588623047, 'Albrecht_Du╠êrer_225'], [20.22389030456543, 'Albrecht_DuÔòá├¬rer_61'], [20.22389030456543, 'Albrecht_Du╠êrer_61'], [20.307565689086914, 'Albrecht_DuÔòá├¬rer_184'], [20.307565689086914, 'Albrecht_Du╠êrer_184']]
--> ACIERTO
    Primer Lugar: 5
    Entre los tres primeros: 5
<Figure size 2000x400 with 6 Axes>



Consulta número:  5
   Imagen de consulta:  Amedeo_Modigliani_26
   Mas cercanos:  [[11.338921546936035, 'Amedeo_Modigliani_26'], [37.50868225097656, 'Albrecht_Du╠êrer_198'], [37.508689880371094, 'Albrecht_DuÔòá├¬rer_198'], [37.6381721496582, 'Albrecht_DuÔòá├¬rer_305'], [37.6381721496582, 'Albrecht_Du╠êrer_305'], [37.783714294433594, 'Alfred_Sisley_198'], [37.7846794128418, 'Albrecht_DuÔòá├¬rer_313'], [37.78468322753906, 'Albrecht_Du╠êrer_313'], [37.83624267578125, 'Albrecht_Du╠êrer_193'], [37.836246490478516, 'Albrecht_DuÔòá├¬rer_193']]
--> ACIERTO
    Primer Lugar: 6
    Entre los tres primeros: 6
<Figure size 2000x400 with 6 Axes>



Consulta número:  6
   Imagen de consulta:  Amedeo_Modigliani_5
   Mas cercanos:  [[5.6151580810546875, 'Amedeo_Modigliani_5'], [30.058795928955078, 'Camille_Pissarro_64'], [30.1263427734375, 'Claude_Monet_33'], [30.143049240112305, 'Alfred_Sisley_127'], [30.14405059814453, 'Albrecht_DuÔòá├¬rer_53'], [30.14405059814453, 'Albrecht_Du╠êrer_53'], [30.14557456970215, 'Albrecht_DuÔòá├¬rer_121'], [30.14557456970215, 'Albrecht_Du╠êrer_121'], [30.209869384765625, 'Camille_Pissarro_34'], [30.224672317504883, 'Albrecht_DuÔòá├¬rer_274']]
--> ACIERTO
    Primer Lugar: 7
    Entre los tres primeros: 7
<Figure size 2000x400 with 6 Axes>



Consulta número:  7
   Imagen de consulta:  Amedeo_Modigliani_52
   Mas cercanos:  [[3.681959390640259, 'Amedeo_Modigliani_52'], [31.00519561767578, 'Camille_Pissarro_37'], [31.08290672302246, 'Alfred_Sisley_198'], [31.466773986816406, 'Alfred_Sisley_39'], [31.528675079345703, 'Camille_Pissarro_59'], [31.90573501586914, 'Camille_Pissarro_27'], [31.947952270507812, 'Alfred_Sisley_80'], [32.02888107299805, 'Camille_Pissarro_20'], [32.046119689941406, 'Edgar_Degas_187'], [32.24347686767578, 'Andrei_Rublev_85']]
--> ACIERTO
    Primer Lugar: 8
    Entre los tres primeros: 8
<Figure size 2000x400 with 6 Axes>



Consulta número:  8
   Imagen de consulta:  Amedeo_Modigliani_94
   Mas cercanos:  [[3.6397149562835693, 'Amedeo_Modigliani_94'], [45.976905822753906, 'Albrecht_DuÔòá├¬rer_140'], [45.976905822753906, 'Albrecht_Du╠êrer_140'], [46.38119125366211, 'Amedeo_Modigliani_153'], [46.39908981323242, 'Alfred_Sisley_139'], [46.44823455810547, 'Camille_Pissarro_9'], [46.4775276184082, 'Claude_Monet_65'], [46.56962203979492, 'Alfred_Sisley_235'], [46.646419525146484, 'Alfred_Sisley_76'], [46.70145797729492, 'Edgar_Degas_101']]
--> ACIERTO
    Primer Lugar: 9
    Entre los tres primeros: 9
<Figure size 2000x400 with 6 Axes>



Consulta número:  9
   Imagen de consulta:  Andrei_Rublev_36
   Mas cercanos:  [[3.052873373031616, 'Andrei_Rublev_36'], [34.230369567871094, 'Albrecht_Du╠êrer_159'], [34.23037338256836, 'Albrecht_DuÔòá├¬rer_159'], [34.775062561035156, 'Albrecht_Du╠êrer_321'], [34.77506637573242, 'Albrecht_DuÔòá├¬rer_321'], [34.89957809448242, 'Camille_Pissarro_56'], [35.106693267822266, 'Albrecht_Du╠êrer_285'], [35.10669708251953, 'Albrecht_DuÔòá├¬rer_285'], [35.17793273925781, 'Alfred_Sisley_78'], [35.2241096496582, 'Albrecht_DuÔòá├¬rer_2']]
--> ACIERTO
    Primer Lugar: 10
    Entre los tres primeros: 10
<Figure size 2000x400 with 6 Axes>



Consulta número:  10
   Imagen de consulta:  Andrei_Rublev_53
   Mas cercanos:  [[4.613982677459717, 'Andrei_Rublev_53'], [31.75282859802246, 'Albrecht_Du╠êrer_73'], [31.752830505371094, 'Albrecht_DuÔòá├¬rer_73'], [32.586856842041016, 'Andrei_Rublev_49'], [32.72325897216797, 'Albrecht_Du╠êrer_115'], [32.723262786865234, 'Albrecht_DuÔòá├¬rer_115'], [32.98215103149414, 'Albrecht_Du╠êrer_294'], [32.982154846191406, 'Albrecht_DuÔòá├¬rer_294'], [33.0130615234375, 'Camille_Pissarro_24'], [33.01319122314453, 'Edgar_Degas_356']]
--> ACIERTO
    Primer Lugar: 11
    Entre los tres primeros: 11
<Figure size 2000x400 with 6 Axes>



Consulta número:  11
   Imagen de consulta:  Andrei_Rublev_64
   Mas cercanos:  [[5.104398727416992, 'Andrei_Rublev_64'], [34.58336639404297, 'Edgar_Degas_170'], [34.857242584228516, 'Alfred_Sisley_183'], [35.095985412597656, 'Alfred_Sisley_137'], [35.248741149902344, 'Edgar_Degas_190'], [35.2673225402832, 'Alfred_Sisley_76'], [35.35185623168945, 'Albrecht_DuÔòá├¬rer_98'], [35.35185623168945, 'Albrecht_Du╠êrer_98'], [35.35340881347656, 'Camille_Pissarro_18'], [35.403411865234375, 'Claude_Monet_67']]
--> ACIERTO
    Primer Lugar: 12
    Entre los tres primeros: 12
<Figure size 2000x400 with 6 Axes>



Consulta número:  12
   Imagen de consulta:  Andrei_Rublev_81
   Mas cercanos:  [[18.839710235595703, 'Andrei_Rublev_81'], [34.29336929321289, 'Camille_Pissarro_2'], [34.394020080566406, 'Claude_Monet_6'], [34.412025451660156, 'Alfred_Sisley_248'], [34.49163818359375, 'Alfred_Sisley_29'], [34.6092643737793, 'Andrei_Rublev_48'], [34.693702697753906, 'Claude_Monet_21'], [34.69561004638672, 'Alfred_Sisley_76'], [34.77599334716797, 'Andrei_Rublev_42'], [34.821563720703125, 'Albrecht_DuÔòá├¬rer_46']]
--> ACIERTO
    Primer Lugar: 13
    Entre los tres primeros: 13
<Figure size 2000x400 with 6 Axes>



Consulta número:  13
   Imagen de consulta:  Andy_Warhol_119
   Mas cercanos:  [[3.6976122856140137, 'Andy_Warhol_119'], [32.480003356933594, 'Amedeo_Modigliani_93'], [32.64756774902344, 'Albrecht_DuÔòá├¬rer_113'], [32.6475715637207, 'Albrecht_Du╠êrer_113'], [32.67749786376953, 'Alfred_Sisley_9'], [32.69820022583008, 'Alfred_Sisley_228'], [32.74436950683594, 'Alfred_Sisley_201'], [32.8123664855957, 'Albrecht_Du╠êrer_115'], [32.812374114990234, 'Albrecht_DuÔòá├¬rer_115'], [32.83313751220703, 'Albrecht_DuÔòá├¬rer_147']]
--> ACIERTO
    Primer Lugar: 14
    Entre los tres primeros: 14
<Figure size 2000x400 with 6 Axes>



Consulta número:  14
   Imagen de consulta:  Andy_Warhol_17
   Mas cercanos:  [[17.20524024963379, 'Andy_Warhol_17'], [30.033536911010742, 'Edgar_Degas_240'], [30.337871551513672, 'Claude_Monet_58'], [31.304039001464844, 'Alfred_Sisley_64'], [31.325681686401367, 'Edgar_Degas_315'], [31.491302490234375, 'Claude_Monet_33'], [31.527767181396484, 'Amedeo_Modigliani_142'], [31.55172348022461, 'Amedeo_Modigliani_7'], [31.55774688720703, 'Albrecht_DuÔòá├¬rer_184'], [31.55774688720703, 'Albrecht_Du╠êrer_184']]
--> ACIERTO
    Primer Lugar: 15
    Entre los tres primeros: 15
<Figure size 2000x400 with 6 Axes>



Consulta número:  15
   Imagen de consulta:  Andy_Warhol_28
   Mas cercanos:  [[6.275485038757324, 'Andy_Warhol_28'], [55.47744369506836, 'Diego_Rivera_38'], [55.50832748413086, 'Albrecht_DuÔòá├¬rer_308'], [55.50834274291992, 'Albrecht_Du╠êrer_308'], [55.64373779296875, 'Albrecht_Du╠êrer_11'], [55.64374923706055, 'Albrecht_DuÔòá├¬rer_11'], [55.82777786254883, 'Albrecht_DuÔòá├¬rer_170'], [55.82777786254883, 'Albrecht_Du╠êrer_170'], [55.832542419433594, 'Alfred_Sisley_17'], [55.87208557128906, 'Albrecht_DuÔòá├¬rer_123']]
--> ACIERTO
    Primer Lugar: 16
    Entre los tres primeros: 16
<Figure size 2000x400 with 6 Axes>



Consulta número:  16
   Imagen de consulta:  Andy_Warhol_69
   Mas cercanos:  [[3.6612350940704346, 'Andy_Warhol_69'], [36.05852127075195, 'Andy_Warhol_127'], [37.31626892089844, 'Andy_Warhol_26'], [37.52870559692383, 'Edgar_Degas_286'], [37.57994079589844, 'Albrecht_DuÔòá├¬rer_69'], [37.57994079589844, 'Albrecht_Du╠êrer_69'], [37.75509262084961, 'Albrecht_DuÔòá├¬rer_180'], [37.75509262084961, 'Albrecht_Du╠êrer_180'], [37.82014465332031, 'Alfred_Sisley_248'], [37.91507339477539, 'Albrecht_DuÔòá├¬rer_73']]
--> ACIERTO
    Primer Lugar: 17
    Entre los tres primeros: 17
<Figure size 2000x400 with 6 Axes>



Consulta número:  17
   Imagen de consulta:  Andy_Warhol_77
   Mas cercanos:  [[7.524970054626465, 'Andy_Warhol_77'], [80.85575866699219, 'Albrecht_DuÔòá├¬rer_224'], [80.85577392578125, 'Albrecht_Du╠êrer_224'], [82.5589599609375, 'Albrecht_DuÔòá├¬rer_65'], [82.55896759033203, 'Albrecht_Du╠êrer_65'], [82.72544860839844, 'Albrecht_DuÔòá├¬rer_297'], [82.72544860839844, 'Albrecht_Du╠êrer_297'], [82.91806030273438, 'Albrecht_DuÔòá├¬rer_327'], [82.91806030273438, 'Albrecht_Du╠êrer_327'], [83.54497528076172, 'Andy_Warhol_181']]
--> ACIERTO
    Primer Lugar: 18
    Entre los tres primeros: 18
<Figure size 2000x400 with 6 Axes>



Consulta número:  18
   Imagen de consulta:  Camille_Pissarro_53
   Mas cercanos:  [[4.240609645843506, 'Camille_Pissarro_53'], [21.172386169433594, 'Camille_Pissarro_64'], [21.29198455810547, 'Camille_Pissarro_41'], [21.292369842529297, 'Camille_Pissarro_73'], [21.385385513305664, 'Alfred_Sisley_126'], [21.397613525390625, 'Claude_Monet_36'], [21.451644897460938, 'Albrecht_Du╠êrer_155'], [21.45166015625, 'Albrecht_DuÔòá├¬rer_155'], [21.52703094482422, 'Alfred_Sisley_175'], [21.6176815032959, 'Alfred_Sisley_92']]
--> ACIERTO
    Primer Lugar: 19
    Entre los tres primeros: 19
<Figure size 2000x400 with 6 Axes>



Consulta número:  19
   Imagen de consulta:  Camille_Pissarro_84
   Mas cercanos:  [[6.61083984375, 'Camille_Pissarro_84'], [41.42356872558594, 'Claude_Monet_46'], [41.60285949707031, 'Albrecht_DuÔòá├¬rer_278'], [41.602867126464844, 'Albrecht_Du╠êrer_278'], [41.6103515625, 'Camille_Pissarro_44'], [41.70391845703125, 'Albrecht_DuÔòá├¬rer_305'], [41.70391845703125, 'Albrecht_Du╠êrer_305'], [41.73270797729492, 'Alfred_Sisley_47'], [41.79161834716797, 'Amedeo_Modigliani_148'], [41.82624816894531, 'Camille_Pissarro_51']]
--> ACIERTO
    Primer Lugar: 20
    Entre los tres primeros: 20
<Figure size 2000x400 with 6 Axes>



Consulta número:  20
   Imagen de consulta:  Caravaggio_14
   Mas cercanos:  [[20.685890197753906, 'Caravaggio_14'], [50.07065200805664, 'Diego_Velazquez_13'], [50.353370666503906, 'Albrecht_Du╠êrer_197'], [50.35337829589844, 'Albrecht_DuÔòá├¬rer_197'], [50.437774658203125, 'Amedeo_Modigliani_98'], [51.2296257019043, 'Albrecht_Du╠êrer_181'], [51.22962951660156, 'Albrecht_DuÔòá├¬rer_181'], [51.55931854248047, 'Amedeo_Modigliani_72'], [51.58009338378906, 'Vincent_van_Gogh_309'], [51.604698181152344, 'Albrecht_Du╠êrer_24']]
--> ACIERTO
    Primer Lugar: 21
    Entre los tres primeros: 21
<Figure size 2000x400 with 6 Axes>



Consulta número:  21
   Imagen de consulta:  Claude_Monet_39
   Mas cercanos:  [[6.080742835998535, 'Claude_Monet_39'], [31.411054611206055, 'Albrecht_DuÔòá├¬rer_27'], [31.411060333251953, 'Albrecht_Du╠êrer_27'], [31.443252563476562, 'Andrei_Rublev_89'], [31.701473236083984, 'Albrecht_Du╠êrer_43'], [31.70147705078125, 'Albrecht_DuÔòá├¬rer_43'], [31.7546329498291, 'Albrecht_Du╠êrer_181'], [31.754638671875, 'Albrecht_DuÔòá├¬rer_181'], [31.783111572265625, 'Albrecht_DuÔòá├¬rer_187'], [31.78311538696289, 'Albrecht_Du╠êrer_187']]
--> ACIERTO
    Primer Lugar: 22
    Entre los tres primeros: 22
<Figure size 2000x400 with 6 Axes>



Consulta número:  22
   Imagen de consulta:  Diego_Rivera_1
   Mas cercanos:  [[16.54607582092285, 'Diego_Rivera_1'], [44.0017204284668, 'Camille_Pissarro_47'], [44.156497955322266, 'Camille_Pissarro_62'], [44.325862884521484, 'Andrei_Rublev_42'], [44.57743835449219, 'Camille_Pissarro_48'], [44.77374267578125, 'Albrecht_Du╠êrer_110'], [44.77375411987305, 'Albrecht_DuÔòá├¬rer_110'], [44.81368637084961, 'Alfred_Sisley_248'], [44.83833312988281, 'Alfred_Sisley_17'], [44.88149642944336, 'Albrecht_DuÔòá├¬rer_123']]
--> ACIERTO
    Primer Lugar: 23
    Entre los tres primeros: 23
<Figure size 2000x400 with 6 Axes>



Consulta número:  23
   Imagen de consulta:  Diego_Rivera_20
   Mas cercanos:  [[4.749136924743652, 'Diego_Rivera_20'], [23.60418701171875, 'Albrecht_DuÔòá├¬rer_191'], [23.60418701171875, 'Albrecht_Du╠êrer_191'], [23.63328742980957, 'Edgar_Degas_312'], [23.651430130004883, 'Alfred_Sisley_126'], [23.690298080444336, 'Albrecht_DuÔòá├¬rer_195'], [23.690298080444336, 'Albrecht_Du╠êrer_195'], [23.741025924682617, 'Claude_Monet_36'], [23.7703800201416, 'Alfred_Sisley_212'], [23.788646697998047, 'Alfred_Sisley_229']]
--> ACIERTO
    Primer Lugar: 24
    Entre los tres primeros: 24
<Figure size 2000x400 with 6 Axes>



Consulta número:  24
   Imagen de consulta:  Diego_Rivera_22
   Mas cercanos:  [[51.31134796142578, 'Diego_Rivera_22'], [56.76020812988281, 'Albrecht_Du╠êrer_11'], [56.76021194458008, 'Albrecht_DuÔòá├¬rer_11'], [57.95074462890625, 'Diego_Rivera_66'], [58.09998321533203, 'Albrecht_Du╠êrer_253'], [58.09999465942383, 'Albrecht_DuÔòá├¬rer_253'], [58.19160842895508, 'Albrecht_DuÔòá├¬rer_30'], [58.19160842895508, 'Albrecht_Du╠êrer_30'], [58.2750358581543, 'Albrecht_DuÔòá├¬rer_125'], [58.2750358581543, 'Albrecht_Du╠êrer_125']]
--> ACIERTO
    Primer Lugar: 25
    Entre los tres primeros: 25
<Figure size 2000x400 with 6 Axes>



Consulta número:  25
   Imagen de consulta:  Diego_Rivera_62
   Mas cercanos:  [[29.20634651184082, 'Diego_Rivera_62'], [47.397499084472656, 'Albrecht_Du╠êrer_4'], [47.39750289916992, 'Albrecht_DuÔòá├¬rer_4'], [47.71936798095703, 'Andrei_Rublev_62'], [47.91497039794922, 'Andrei_Rublev_84'], [48.197509765625, 'Albrecht_DuÔòá├¬rer_308'], [48.19752502441406, 'Albrecht_Du╠êrer_308'], [48.464542388916016, 'Albrecht_DuÔòá├¬rer_227'], [48.46454620361328, 'Albrecht_Du╠êrer_227'], [48.96270751953125, 'Alfred_Sisley_3']]
--> ACIERTO
    Primer Lugar: 26
    Entre los tres primeros: 26
<Figure size 2000x400 with 6 Axes>



Consulta número:  26
   Imagen de consulta:  Diego_Velazquez_103
   Mas cercanos:  [[14.004610061645508, 'Diego_Velazquez_103'], [34.21895980834961, 'Camille_Pissarro_51'], [34.793426513671875, 'Claude_Monet_46'], [34.81101608276367, 'Alfred_Sisley_3'], [34.8766975402832, 'Caravaggio_45'], [35.125911712646484, 'Alfred_Sisley_90'], [35.19327926635742, 'Alfred_Sisley_29'], [35.31951904296875, 'Albrecht_Du╠êrer_85'], [35.319522857666016, 'Albrecht_DuÔòá├¬rer_85'], [35.37931442260742, 'Alfred_Sisley_67']]
--> ACIERTO
    Primer Lugar: 27
    Entre los tres primeros: 27
<Figure size 2000x400 with 6 Axes>



Consulta número:  27
   Imagen de consulta:  Diego_Velazquez_14
   Mas cercanos:  [[2.7898590564727783, 'Diego_Velazquez_14'], [15.695337295532227, 'Amedeo_Modigliani_7'], [15.864692687988281, 'Edgar_Degas_196'], [15.94699478149414, 'Amedeo_Modigliani_142'], [15.95833683013916, 'Albrecht_DuÔòá├¬rer_225'], [15.958344459533691, 'Albrecht_Du╠êrer_225'], [16.088775634765625, 'Edgar_Degas_247'], [16.2216739654541, 'Albrecht_DuÔòá├¬rer_184'], [16.221677780151367, 'Albrecht_Du╠êrer_184'], [16.514480590820312, 'Alfred_Sisley_64']]
--> ACIERTO
    Primer Lugar: 28
    Entre los tres primeros: 28
<Figure size 2000x400 with 6 Axes>



Consulta número:  28
   Imagen de consulta:  Edgar_Degas_115
   Mas cercanos:  [[3.6763970851898193, 'Edgar_Degas_115'], [33.78725051879883, 'Albrecht_Du╠êrer_37'], [33.787254333496094, 'Albrecht_DuÔòá├¬rer_37'], [34.04398727416992, 'Amedeo_Modigliani_157'], [34.07040786743164, 'Albrecht_DuÔòá├¬rer_156'], [34.07040786743164, 'Albrecht_Du╠êrer_156'], [34.18474578857422, 'Albrecht_DuÔòá├¬rer_147'], [34.18474578857422, 'Albrecht_Du╠êrer_147'], [34.18846130371094, 'Albrecht_Du╠êrer_73'], [34.18846893310547, 'Albrecht_DuÔòá├¬rer_73']]
--> ACIERTO
    Primer Lugar: 29
    Entre los tres primeros: 29
<Figure size 2000x400 with 6 Axes>



Consulta número:  29
   Imagen de consulta:  Edgar_Degas_15
   Mas cercanos:  [[19.646440505981445, 'Edgar_Degas_195'], [29.066768646240234, 'Edgar_Degas_15'], [33.57297897338867, 'Edgar_Degas_181'], [34.43388748168945, 'Albrecht_DuÔòá├¬rer_227'], [34.43389129638672, 'Albrecht_Du╠êrer_227'], [34.64169692993164, 'Albrecht_DuÔòá├¬rer_187'], [34.64170455932617, 'Albrecht_Du╠êrer_187'], [34.69902038574219, 'Camille_Pissarro_51'], [34.75865936279297, 'Alfred_Sisley_47'], [34.84928512573242, 'Claude_Monet_46']]
--> ACIERTO
    Entre los tres primeros: 30
<Figure size 2000x400 with 6 Axes>



Consulta número:  30
   Imagen de consulta:  Edgar_Degas_179
   Mas cercanos:  [[7.590395450592041, 'Edgar_Degas_179'], [19.84383201599121, 'Amedeo_Modigliani_7'], [20.09674644470215, 'Edgar_Degas_142'], [20.177900314331055, 'Albrecht_DuÔòá├¬rer_70'], [20.17791175842285, 'Albrecht_Du╠êrer_70'], [20.364351272583008, 'Albrecht_DuÔòá├¬rer_225'], [20.364364624023438, 'Albrecht_Du╠êrer_225'], [20.402488708496094, 'Amedeo_Modigliani_142'], [20.473718643188477, 'Albrecht_DuÔòá├¬rer_184'], [20.473722457885742, 'Albrecht_Du╠êrer_184']]
--> ACIERTO
    Primer Lugar: 30
    Entre los tres primeros: 31
<Figure size 2000x400 with 6 Axes>



Consulta número:  31
   Imagen de consulta:  Edgar_Degas_24
   Mas cercanos:  [[4.974094390869141, 'Edgar_Degas_24'], [30.153661727905273, 'Alfred_Sisley_75'], [30.591094970703125, 'Alfred_Sisley_170'], [30.69705581665039, 'Alfred_Sisley_129'], [30.710411071777344, 'Claude_Monet_47'], [30.77659797668457, 'Albrecht_DuÔòá├¬rer_43'], [30.77660369873047, 'Albrecht_Du╠êrer_43'], [30.831905364990234, 'Claude_Monet_32'], [30.833826065063477, 'Camille_Pissarro_47'], [30.84515953063965, 'Camille_Pissarro_12']]
--> ACIERTO
    Primer Lugar: 31
    Entre los tres primeros: 32
<Figure size 2000x400 with 6 Axes>



Consulta número:  32
   Imagen de consulta:  Edgar_Degas_275
   Mas cercanos:  [[15.03319263458252, 'Edgar_Degas_1'], [15.731895446777344, 'Edgar_Degas_275'], [23.515748977661133, 'Edgar_Degas_3'], [35.32051467895508, 'Andrei_Rublev_33'], [35.72379684448242, 'Alfred_Sisley_22'], [35.998748779296875, 'Andrei_Rublev_89'], [36.221351623535156, 'Alfred_Sisley_76'], [36.24809646606445, 'Albrecht_Du╠êrer_46'], [36.248104095458984, 'Albrecht_DuÔòá├¬rer_46'], [36.259830474853516, 'Alfred_Sisley_28']]
--> ACIERTO
    Entre los tres primeros: 33
<Figure size 2000x400 with 6 Axes>



Consulta número:  33
   Imagen de consulta:  Edgar_Degas_293
   Mas cercanos:  [[5.327079772949219, 'Edgar_Degas_293'], [27.714191436767578, 'Edgar_Degas_294'], [28.25067138671875, 'Edgar_Degas_345'], [28.442569732666016, 'Albrecht_Du╠êrer_106'], [28.44257164001465, 'Albrecht_DuÔòá├¬rer_106'], [28.589160919189453, 'Claude_Monet_47'], [28.66140365600586, 'Edgar_Degas_346'], [28.795669555664062, 'Albrecht_DuÔòá├¬rer_182'], [28.79567527770996, 'Albrecht_Du╠êrer_182'], [29.003568649291992, 'Leonardo_da_Vinci_122']]
--> ACIERTO
    Primer Lugar: 32
    Entre los tres primeros: 34
<Figure size 2000x400 with 6 Axes>



Consulta número:  34
   Imagen de consulta:  Edgar_Degas_297
   Mas cercanos:  [[3.407200574874878, 'Edgar_Degas_297'], [31.320274353027344, 'Edgar_Degas_170'], [31.351579666137695, 'Edgar_Degas_138'], [31.462451934814453, 'Edgar_Degas_219'], [31.641206741333008, 'Alfred_Sisley_28'], [31.71670150756836, 'Camille_Pissarro_24'], [31.77312469482422, 'Claude_Monet_33'], [31.847074508666992, 'Alfred_Sisley_127'], [31.8494930267334, 'Edgar_Degas_171'], [31.857519149780273, 'Andrei_Rublev_66']]
--> ACIERTO
    Primer Lugar: 33
    Entre los tres primeros: 35
<Figure size 2000x400 with 6 Axes>



Consulta número:  35
   Imagen de consulta:  Edgar_Degas_328
   Mas cercanos:  [[20.060508728027344, 'Edgar_Degas_328'], [39.126338958740234, 'Alfred_Sisley_183'], [39.19642639160156, 'Alfred_Sisley_29'], [39.371360778808594, 'Camille_Pissarro_43'], [40.193973541259766, 'Edgar_Degas_103'], [40.2393684387207, 'Albrecht_DuÔòá├¬rer_79'], [40.2393684387207, 'Albrecht_Du╠êrer_79'], [40.2702751159668, 'Albrecht_DuÔòá├¬rer_71'], [40.270286560058594, 'Albrecht_Du╠êrer_71'], [40.276702880859375, 'Albrecht_DuÔòá├¬rer_176']]
--> ACIERTO
    Primer Lugar: 34
    Entre los tres primeros: 36
<Figure size 2000x400 with 6 Axes>



Consulta número:  36
   Imagen de consulta:  Edgar_Degas_35
   Mas cercanos:  [[9.634378433227539, 'Edgar_Degas_35'], [25.902929306030273, 'Albrecht_DuÔòá├¬rer_182'], [25.902929306030273, 'Albrecht_Du╠êrer_182'], [25.980478286743164, 'Camille_Pissarro_31'], [26.351547241210938, 'Alfred_Sisley_198'], [26.386978149414062, 'Claude_Monet_36'], [26.407875061035156, 'Albrecht_DuÔòá├¬rer_184'], [26.407880783081055, 'Albrecht_Du╠êrer_184'], [26.472393035888672, 'Albrecht_DuÔòá├¬rer_313'], [26.472396850585938, 'Albrecht_Du╠êrer_313']]
--> ACIERTO
    Primer Lugar: 35
    Entre los tres primeros: 37
<Figure size 2000x400 with 6 Axes>



Consulta número:  37
   Imagen de consulta:  Edgar_Degas_366
   Mas cercanos:  [[3.499825954437256, 'Edgar_Degas_366'], [21.091596603393555, 'Albrecht_DuÔòá├¬rer_13'], [21.09160041809082, 'Albrecht_Du╠êrer_13'], [21.16826820373535, 'Albrecht_Du╠êrer_82'], [21.168277740478516, 'Albrecht_DuÔòá├¬rer_82'], [21.36599349975586, 'Albrecht_DuÔòá├¬rer_182'], [21.366003036499023, 'Albrecht_Du╠êrer_182'], [21.431936264038086, 'Albrecht_Du╠êrer_184'], [21.43193817138672, 'Albrecht_DuÔòá├¬rer_184'], [21.431978225708008, 'Albrecht_DuÔòá├¬rer_149']]
--> ACIERTO
    Primer Lugar: 36
    Entre los tres primeros: 38
<Figure size 2000x400 with 6 Axes>



Consulta número:  38
   Imagen de consulta:  Edgar_Degas_37
   Mas cercanos:  [[4.690967559814453, 'Edgar_Degas_37'], [31.88218879699707, 'Edgar_Degas_324'], [32.26020812988281, 'Edgar_Degas_188'], [32.41352462768555, 'Alfred_Sisley_39'], [32.44906997680664, 'Edgar_Degas_249'], [32.45864486694336, 'Camille_Pissarro_59'], [32.6107292175293, 'Edgar_Degas_216'], [32.69438552856445, 'Edgar_Degas_273'], [32.69666290283203, 'Camille_Pissarro_88'], [32.75279998779297, 'Andrei_Rublev_48']]
--> ACIERTO
    Primer Lugar: 37
    Entre los tres primeros: 39
<Figure size 2000x400 with 6 Axes>



Consulta número:  39
   Imagen de consulta:  Edgar_Degas_438
   Mas cercanos:  [[3.712244749069214, 'Edgar_Degas_438'], [31.43564796447754, 'Camille_Pissarro_27'], [31.545602798461914, 'Alfred_Sisley_13'], [31.628799438476562, 'Edgar_Degas_322'], [31.726694107055664, 'Edgar_Degas_190'], [31.759191513061523, 'Albrecht_Du╠êrer_294'], [31.759193420410156, 'Albrecht_DuÔòá├¬rer_294'], [31.784717559814453, 'Edgar_Degas_344'], [31.889535903930664, 'Edgar_Degas_167'], [31.979806900024414, 'Alfred_Sisley_163']]
--> ACIERTO
    Primer Lugar: 38
    Entre los tres primeros: 40
<Figure size 2000x400 with 6 Axes>



Consulta número:  40
   Imagen de consulta:  Edgar_Degas_44
   Mas cercanos:  [[13.57127857208252, 'Edgar_Degas_44'], [18.586816787719727, 'Camille_Pissarro_64'], [18.760631561279297, 'Alfred_Sisley_228'], [18.936132431030273, 'Alfred_Sisley_194'], [19.158405303955078, 'Claude_Monet_58'], [19.24751853942871, 'Alfred_Sisley_92'], [19.27947425842285, 'Alfred_Sisley_15'], [19.339107513427734, 'Edgar_Degas_196'], [19.362035751342773, 'Alfred_Sisley_175'], [19.41986083984375, 'Albrecht_DuÔòá├¬rer_70']]
--> ACIERTO
    Primer Lugar: 39
    Entre los tres primeros: 41
<Figure size 2000x400 with 6 Axes>



Consulta número:  41
   Imagen de consulta:  Edgar_Degas_540
   Mas cercanos:  [[6.3714752197265625, 'Edgar_Degas_540'], [22.764780044555664, 'Albrecht_Du╠êrer_39'], [22.764793395996094, 'Albrecht_DuÔòá├¬rer_39'], [23.161880493164062, 'Alfred_Sisley_194'], [23.292795181274414, 'Alfred_Sisley_175'], [23.52155876159668, 'Camille_Pissarro_64'], [23.54102325439453, 'Claude_Monet_17'], [23.575963973999023, 'Albrecht_DuÔòá├¬rer_70'], [23.575973510742188, 'Albrecht_Du╠êrer_70'], [23.58245849609375, 'Albrecht_DuÔòá├¬rer_195']]
--> ACIERTO
    Primer Lugar: 40
    Entre los tres primeros: 42
<Figure size 2000x400 with 6 Axes>



Consulta número:  42
   Imagen de consulta:  Edgar_Degas_584
   Mas cercanos:  [[2.352262258529663, 'Edgar_Degas_584'], [15.525382041931152, 'Amedeo_Modigliani_142'], [16.027097702026367, 'Albrecht_DuÔòá├¬rer_225'], [16.027109146118164, 'Albrecht_Du╠êrer_225'], [16.081558227539062, 'Albrecht_Du╠êrer_202'], [16.08156394958496, 'Albrecht_DuÔòá├¬rer_202'], [16.19032859802246, 'Albrecht_Du╠êrer_184'], [16.190330505371094, 'Albrecht_DuÔòá├¬rer_184'], [16.19330596923828, 'Amedeo_Modigliani_7'], [16.260772705078125, 'Edgar_Degas_315']]
--> ACIERTO
    Primer Lugar: 41
    Entre los tres primeros: 43
<Figure size 2000x400 with 6 Axes>



Consulta número:  43
   Imagen de consulta:  Edgar_Degas_639
   Mas cercanos:  [[41.58933639526367, 'Edgar_Degas_639'], [43.15009689331055, 'Edgar_Degas_347'], [52.77775573730469, 'Edgar_Degas_33'], [53.296653747558594, 'Edgar_Degas_255'], [53.522056579589844, 'Alfred_Sisley_90'], [53.62406921386719, 'Albrecht_Du╠êrer_253'], [53.62407684326172, 'Albrecht_DuÔòá├¬rer_253'], [53.85067367553711, 'Diego_Rivera_45'], [54.24814987182617, 'Albrecht_DuÔòá├¬rer_51'], [54.24814987182617, 'Albrecht_Du╠êrer_51']]
--> ACIERTO
    Primer Lugar: 42
    Entre los tres primeros: 44
<Figure size 2000x400 with 6 Axes>



Consulta número:  44
   Imagen de consulta:  Kazimir_Malevich_100
   Mas cercanos:  [[17.199275970458984, 'Kazimir_Malevich_100'], [64.57350158691406, 'Caravaggio_14'], [65.8951416015625, 'Albrecht_DuÔòá├¬rer_197'], [65.8951416015625, 'Albrecht_Du╠êrer_197'], [66.48966217041016, 'Albrecht_DuÔòá├¬rer_129'], [66.48966217041016, 'Albrecht_Du╠êrer_129'], [66.6294937133789, 'Albrecht_DuÔòá├¬rer_192'], [66.6294937133789, 'Albrecht_Du╠êrer_192'], [67.03726196289062, 'Albrecht_DuÔòá├¬rer_24'], [67.03726196289062, 'Albrecht_Du╠êrer_24']]
--> ACIERTO
    Primer Lugar: 43
    Entre los tres primeros: 45
<Figure size 2000x400 with 6 Axes>



Consulta número:  45
   Imagen de consulta:  Kazimir_Malevich_115
   Mas cercanos:  [[15.06616497039795, 'Kazimir_Malevich_115'], [41.54920196533203, 'Albrecht_DuÔòá├¬rer_105'], [41.54920196533203, 'Albrecht_Du╠êrer_105'], [41.99131393432617, 'Albrecht_DuÔòá├¬rer_10'], [41.99131393432617, 'Albrecht_Du╠êrer_10'], [42.31310272216797, 'Camille_Pissarro_58'], [42.589599609375, 'Albrecht_DuÔòá├¬rer_237'], [42.589599609375, 'Albrecht_Du╠êrer_237'], [42.724639892578125, 'Albrecht_Du╠êrer_159'], [42.72464370727539, 'Albrecht_DuÔòá├¬rer_159']]
--> ACIERTO
    Primer Lugar: 44
    Entre los tres primeros: 46
<Figure size 2000x400 with 6 Axes>



Consulta número:  46
   Imagen de consulta:  Kazimir_Malevich_124
   Mas cercanos:  [[7.612442493438721, 'Kazimir_Malevich_124'], [33.7850227355957, 'Albrecht_Du╠êrer_159'], [33.785030364990234, 'Albrecht_DuÔòá├¬rer_159'], [33.80724334716797, 'Albrecht_DuÔòá├¬rer_229'], [33.807254791259766, 'Albrecht_Du╠êrer_229'], [33.96573257446289, 'Albrecht_DuÔòá├¬rer_122'], [33.96573257446289, 'Albrecht_Du╠êrer_122'], [34.050846099853516, 'Camille_Pissarro_63'], [34.073604583740234, 'Alfred_Sisley_239'], [34.128562927246094, 'Claude_Monet_46']]
--> ACIERTO
    Primer Lugar: 45
    Entre los tres primeros: 47
<Figure size 2000x400 with 6 Axes>



Consulta número:  47
   Imagen de consulta:  Kazimir_Malevich_72
   Mas cercanos:  [[5.017528533935547, 'Kazimir_Malevich_72'], [40.96649169921875, 'Albrecht_DuÔòá├¬rer_57'], [40.96649169921875, 'Albrecht_Du╠êrer_57'], [40.99137878417969, 'Amedeo_Modigliani_35'], [40.9978141784668, 'Edgar_Degas_273'], [41.02777862548828, 'Albrecht_DuÔòá├¬rer_313'], [41.02779006958008, 'Albrecht_Du╠êrer_313'], [41.22028350830078, 'Albrecht_Du╠êrer_58'], [41.22029113769531, 'Albrecht_DuÔòá├¬rer_58'], [41.25164794921875, 'Albrecht_DuÔòá├¬rer_61']]
--> ACIERTO
    Primer Lugar: 46
    Entre los tres primeros: 48
<Figure size 2000x400 with 6 Axes>



Consulta número:  48
   Imagen de consulta:  Leonardo_da_Vinci_11
   Mas cercanos:  [[4.112301826477051, 'Leonardo_da_Vinci_11'], [46.47068405151367, 'Edgar_Degas_102'], [46.72032928466797, 'Camille_Pissarro_27'], [46.87545394897461, 'Edgar_Degas_181'], [46.8997917175293, 'Andrei_Rublev_97'], [47.024925231933594, 'Edgar_Degas_169'], [47.04806900024414, 'Edgar_Degas_170'], [47.06034851074219, 'Alfred_Sisley_18'], [47.172760009765625, 'Edgar_Degas_317'], [47.18061828613281, 'Edgar_Degas_104']]
--> ACIERTO
    Primer Lugar: 47
    Entre los tres primeros: 49
<Figure size 2000x400 with 6 Axes>



Consulta número:  49
   Imagen de consulta:  Leonardo_da_Vinci_110
   Mas cercanos:  [[11.77800178527832, 'Leonardo_da_Vinci_110'], [20.681711196899414, 'Albrecht_Du╠êrer_80'], [20.681713104248047, 'Albrecht_DuÔòá├¬rer_80'], [21.433582305908203, 'Albrecht_DuÔòá├¬rer_225'], [21.4335880279541, 'Albrecht_Du╠êrer_225'], [21.63388442993164, 'Albrecht_Du╠êrer_149'], [21.633892059326172, 'Albrecht_DuÔòá├¬rer_149'], [21.84775733947754, 'Amedeo_Modigliani_7'], [21.906661987304688, 'Claude_Monet_19'], [21.924150466918945, 'Alfred_Sisley_15']]
--> ACIERTO
    Primer Lugar: 48
    Entre los tres primeros: 50
<Figure size 2000x400 with 6 Axes>



Consulta número:  50
   Imagen de consulta:  Leonardo_da_Vinci_121
   Mas cercanos:  [[3.096465826034546, 'Leonardo_da_Vinci_121'], [18.131235122680664, 'Amedeo_Modigliani_142'], [18.892343521118164, 'Camille_Pissarro_4'], [19.037752151489258, 'Amedeo_Modigliani_7'], [19.17371368408203, 'Albrecht_DuÔòá├¬rer_225'], [19.173725128173828, 'Albrecht_Du╠êrer_225'], [19.298234939575195, 'Edgar_Degas_315'], [19.304887771606445, 'Edgar_Degas_226'], [19.366567611694336, 'Albrecht_DuÔòá├¬rer_184'], [19.3665714263916, 'Albrecht_Du╠êrer_184']]
--> ACIERTO
    Primer Lugar: 49
    Entre los tres primeros: 51
<Figure size 2000x400 with 6 Axes>



Consulta número:  51
   Imagen de consulta:  Leonardo_da_Vinci_122
   Mas cercanos:  [[5.445281505584717, 'Leonardo_da_Vinci_122'], [14.985285758972168, 'Amedeo_Modigliani_142'], [15.28600788116455, 'Albrecht_DuÔòá├¬rer_225'], [15.286011695861816, 'Albrecht_Du╠êrer_225'], [15.410386085510254, 'Albrecht_DuÔòá├¬rer_184'], [15.410386085510254, 'Albrecht_Du╠êrer_184'], [15.467100143432617, 'Edgar_Degas_196'], [15.557950019836426, 'Edgar_Degas_226'], [15.677949905395508, 'Edgar_Degas_240'], [15.777693748474121, 'Claude_Monet_73']]
--> ACIERTO
    Primer Lugar: 50
    Entre los tres primeros: 52
<Figure size 2000x400 with 6 Axes>



Consulta número:  52
   Imagen de consulta:  Leonardo_da_Vinci_124
   Mas cercanos:  [[4.0271430015563965, 'Leonardo_da_Vinci_124'], [30.707332611083984, 'Albrecht_DuÔòá├¬rer_169'], [30.707332611083984, 'Albrecht_Du╠êrer_169'], [31.228343963623047, 'Albrecht_DuÔòá├¬rer_43'], [31.228343963623047, 'Albrecht_Du╠êrer_43'], [31.235380172729492, 'Andy_Warhol_46'], [31.317129135131836, 'Albrecht_DuÔòá├¬rer_181'], [31.3171329498291, 'Albrecht_Du╠êrer_181'], [31.335906982421875, 'Camille_Pissarro_37'], [31.463516235351562, 'Alfred_Sisley_82']]
--> ACIERTO
    Primer Lugar: 51
    Entre los tres primeros: 53
<Figure size 2000x400 with 6 Axes>



Consulta número:  53
   Imagen de consulta:  Leonardo_da_Vinci_45
   Mas cercanos:  [[29.192245483398438, 'Leonardo_da_Vinci_45'], [43.0881233215332, 'Albrecht_DuÔòá├¬rer_305'], [43.0881233215332, 'Albrecht_Du╠êrer_305'], [43.430355072021484, 'Albrecht_Du╠êrer_151'], [43.43035888671875, 'Albrecht_DuÔòá├¬rer_151'], [43.57550048828125, 'Albrecht_DuÔòá├¬rer_197'], [43.57550048828125, 'Albrecht_Du╠êrer_197'], [43.64468002319336, 'Albrecht_DuÔòá├¬rer_52'], [43.64468002319336, 'Albrecht_Du╠êrer_52'], [43.66958236694336, 'Albrecht_DuÔòá├¬rer_119']]
--> ACIERTO
    Primer Lugar: 52
    Entre los tres primeros: 54
<Figure size 2000x400 with 6 Axes>



Consulta número:  54
   Imagen de consulta:  Leonardo_da_Vinci_89
   Mas cercanos:  [[5.007612705230713, 'Leonardo_da_Vinci_89'], [23.678770065307617, 'Albrecht_DuÔòá├¬rer_43'], [23.678770065307617, 'Albrecht_Du╠êrer_43'], [24.42428970336914, 'Albrecht_Du╠êrer_320'], [24.424291610717773, 'Albrecht_DuÔòá├¬rer_320'], [24.6668701171875, 'Albrecht_DuÔòá├¬rer_274'], [24.666873931884766, 'Albrecht_Du╠êrer_274'], [25.02437973022461, 'Albrecht_DuÔòá├¬rer_27'], [25.024381637573242, 'Albrecht_Du╠êrer_27'], [25.039817810058594, 'Albrecht_DuÔòá├¬rer_209']]
--> ACIERTO
    Primer Lugar: 53
    Entre los tres primeros: 55
<Figure size 2000x400 with 6 Axes>



Consulta número:  55
   Imagen de consulta:  Marc_Chagall_123
   Mas cercanos:  [[5.135194301605225, 'Marc_Chagall_123'], [33.37395477294922, 'Camille_Pissarro_15'], [33.596824645996094, 'Claude_Monet_35'], [33.67698287963867, 'Alfred_Sisley_253'], [33.78874588012695, 'Albrecht_DuÔòá├¬rer_229'], [33.788753509521484, 'Albrecht_Du╠êrer_229'], [33.82498550415039, 'Alfred_Sisley_238'], [33.94969177246094, 'Claude_Monet_46'], [34.02058792114258, 'Andrei_Rublev_55'], [34.051048278808594, 'Edgar_Degas_286']]
--> ACIERTO
    Primer Lugar: 54
    Entre los tres primeros: 56
<Figure size 2000x400 with 6 Axes>



Consulta número:  56
   Imagen de consulta:  Marc_Chagall_152
   Mas cercanos:  [[21.065052032470703, 'Marc_Chagall_152'], [30.861812591552734, 'Albrecht_Du╠êrer_40'], [30.86182403564453, 'Albrecht_DuÔòá├¬rer_40'], [31.104185104370117, 'Claude_Monet_47'], [31.216232299804688, 'Alfred_Sisley_9'], [31.327735900878906, 'Camille_Pissarro_24'], [31.347484588623047, 'Claude_Monet_65'], [31.381881713867188, 'Alfred_Sisley_94'], [31.38689613342285, 'Andy_Warhol_110'], [31.399412155151367, 'Claude_Monet_36']]
--> ACIERTO
    Primer Lugar: 55
    Entre los tres primeros: 57
<Figure size 2000x400 with 6 Axes>



Consulta número:  57
   Imagen de consulta:  Marc_Chagall_170
   Mas cercanos:  [[5.212220191955566, 'Marc_Chagall_170'], [41.56109619140625, 'Albrecht_DuÔòá├¬rer_94'], [41.56109619140625, 'Albrecht_Du╠êrer_94'], [41.855186462402344, 'Albrecht_Du╠êrer_116'], [41.855194091796875, 'Albrecht_DuÔòá├¬rer_116'], [41.91557693481445, 'Albrecht_DuÔòá├¬rer_287'], [41.91557693481445, 'Albrecht_Du╠êrer_287'], [41.958492279052734, 'Albrecht_DuÔòá├¬rer_180'], [41.95849609375, 'Albrecht_Du╠êrer_180'], [42.009761810302734, 'Albrecht_DuÔòá├¬rer_176']]
--> ACIERTO
    Primer Lugar: 56
    Entre los tres primeros: 58
<Figure size 2000x400 with 6 Axes>



Consulta número:  58
   Imagen de consulta:  Marc_Chagall_178
   Mas cercanos:  [[10.5349702835083, 'Marc_Chagall_178'], [50.338443756103516, 'Albrecht_DuÔòá├¬rer_77'], [50.338443756103516, 'Albrecht_Du╠êrer_77'], [50.548274993896484, 'Albrecht_DuÔòá├¬rer_308'], [50.548282623291016, 'Albrecht_Du╠êrer_308'], [50.64262390136719, 'Amedeo_Modigliani_184'], [50.65766525268555, 'Albrecht_Du╠êrer_183'], [50.65766906738281, 'Albrecht_DuÔòá├¬rer_183'], [51.166114807128906, 'Albrecht_DuÔòá├¬rer_319'], [51.16612243652344, 'Albrecht_Du╠êrer_319']]
--> ACIERTO
    Primer Lugar: 57
    Entre los tres primeros: 59
<Figure size 2000x400 with 6 Axes>



Consulta número:  59
   Imagen de consulta:  Marc_Chagall_183
   Mas cercanos:  [[15.312824249267578, 'Marc_Chagall_183'], [20.31566619873047, 'Alfred_Sisley_64'], [20.383237838745117, 'Albrecht_DuÔòá├¬rer_70'], [20.383251190185547, 'Albrecht_Du╠êrer_70'], [20.536285400390625, 'Albrecht_DuÔòá├¬rer_275'], [20.536291122436523, 'Albrecht_Du╠êrer_275'], [20.64740753173828, 'Edgar_Degas_315'], [20.68971824645996, 'Albrecht_DuÔòá├¬rer_182'], [20.68972396850586, 'Albrecht_Du╠êrer_182'], [20.709789276123047, 'Albrecht_DuÔòá├¬rer_184']]
--> ACIERTO
    Primer Lugar: 58
    Entre los tres primeros: 60
<Figure size 2000x400 with 6 Axes>



Consulta número:  60
   Imagen de consulta:  Marc_Chagall_2
   Mas cercanos:  [[6.241950035095215, 'Marc_Chagall_2'], [38.436119079589844, 'Alfred_Sisley_238'], [38.569679260253906, 'Vincent_van_Gogh_309'], [38.58207321166992, 'Albrecht_DuÔòá├¬rer_308'], [38.58207702636719, 'Albrecht_Du╠êrer_308'], [38.78117752075195, 'Albrecht_DuÔòá├¬rer_229'], [38.781185150146484, 'Albrecht_Du╠êrer_229'], [38.9243049621582, 'Alfred_Sisley_157'], [39.03280258178711, 'Edgar_Degas_160'], [39.05530548095703, 'Albrecht_Du╠êrer_293']]
--> ACIERTO
    Primer Lugar: 59
    Entre los tres primeros: 61
<Figure size 2000x400 with 6 Axes>



Consulta número:  61
   Imagen de consulta:  Marc_Chagall_52
   Mas cercanos:  [[31.9542293548584, 'Marc_Chagall_52'], [39.04536056518555, 'Albrecht_DuÔòá├¬rer_77'], [39.04536056518555, 'Albrecht_Du╠êrer_77'], [39.61823654174805, 'Albrecht_DuÔòá├¬rer_247'], [39.61824035644531, 'Albrecht_Du╠êrer_247'], [40.086143493652344, 'Edgar_Degas_189'], [40.668212890625, 'Albrecht_Du╠êrer_151'], [40.66822052001953, 'Albrecht_DuÔòá├¬rer_151'], [40.6829833984375, 'Andy_Warhol_80'], [40.73062515258789, 'Albrecht_DuÔòá├¬rer_119']]
--> ACIERTO
    Primer Lugar: 60
    Entre los tres primeros: 62
<Figure size 2000x400 with 6 Axes>



Consulta número:  62
   Imagen de consulta:  Marc_Chagall_56
   Mas cercanos:  [[4.19731330871582, 'Marc_Chagall_56'], [43.106056213378906, 'Albrecht_DuÔòá├¬rer_52'], [43.106056213378906, 'Albrecht_Du╠êrer_52'], [44.562957763671875, 'Alfred_Sisley_90'], [44.762569427490234, 'Edgar_Degas_33'], [44.766380310058594, 'Camille_Pissarro_2'], [44.9950065612793, 'Albrecht_DuÔòá├¬rer_223'], [44.9950065612793, 'Albrecht_Du╠êrer_223'], [45.11970138549805, 'Edgar_Degas_157'], [45.34579086303711, 'Albrecht_DuÔòá├¬rer_324']]
--> ACIERTO
    Primer Lugar: 61
    Entre los tres primeros: 63
<Figure size 2000x400 with 6 Axes>



Consulta número:  63
   Imagen de consulta:  Marc_Chagall_8
   Mas cercanos:  [[29.41973304748535, 'Marc_Chagall_8'], [57.206565856933594, 'Diego_Rivera_12'], [57.349449157714844, 'Mikhail_Vrubel_16'], [57.35369110107422, 'Albrecht_Du╠êrer_116'], [57.353694915771484, 'Albrecht_DuÔòá├¬rer_116'], [57.397953033447266, 'Claude_Monet_37'], [57.480125427246094, 'Claude_Monet_18'], [57.69942855834961, 'Alfred_Sisley_186'], [57.85149002075195, 'Albrecht_DuÔòá├¬rer_52'], [57.85149002075195, 'Albrecht_Du╠êrer_52']]
--> ACIERTO
    Primer Lugar: 62
    Entre los tres primeros: 64
<Figure size 2000x400 with 6 Axes>



Consulta número:  64
   Imagen de consulta:  Marc_Chagall_92
   Mas cercanos:  [[29.537906646728516, 'Marc_Chagall_92'], [41.69628143310547, 'Camille_Pissarro_27'], [42.267608642578125, 'Albrecht_DuÔòá├¬rer_98'], [42.267608642578125, 'Albrecht_Du╠êrer_98'], [42.38047409057617, 'Alfred_Sisley_29'], [42.44303894042969, 'Camille_Pissarro_43'], [42.447547912597656, 'Albrecht_DuÔòá├¬rer_322'], [42.447547912597656, 'Albrecht_Du╠êrer_322'], [42.66067123413086, 'Claude_Monet_21'], [42.689422607421875, 'Camille_Pissarro_26']]
--> ACIERTO
    Primer Lugar: 63
    Entre los tres primeros: 65
<Figure size 2000x400 with 6 Axes>



Consulta número:  65
   Imagen de consulta:  Michelangelo_16
   Mas cercanos:  [[19.95408058166504, 'Michelangelo_16'], [43.89795684814453, 'Albrecht_DuÔòá├¬rer_52'], [43.89795684814453, 'Albrecht_Du╠êrer_52'], [44.863887786865234, 'Albrecht_Du╠êrer_253'], [44.8638916015625, 'Albrecht_DuÔòá├¬rer_253'], [45.077850341796875, 'Alfred_Sisley_90'], [45.12664794921875, 'Camille_Pissarro_2'], [45.233802795410156, 'Albrecht_DuÔòá├¬rer_71'], [45.23381042480469, 'Albrecht_Du╠êrer_71'], [45.29188537597656, 'Albrecht_DuÔòá├¬rer_223']]
--> ACIERTO
    Primer Lugar: 64
    Entre los tres primeros: 66
<Figure size 2000x400 with 6 Axes>



Consulta número:  66
   Imagen de consulta:  Michelangelo_21
   Mas cercanos:  [[44.88964080810547, 'Michelangelo_21'], [46.00300216674805, 'Alfred_Sisley_90'], [46.58380126953125, 'Albrecht_Du╠êrer_85'], [46.583805084228516, 'Albrecht_DuÔòá├¬rer_85'], [46.82756423950195, 'Albrecht_DuÔòá├¬rer_308'], [46.82756805419922, 'Albrecht_Du╠êrer_308'], [46.96785354614258, 'Albrecht_DuÔòá├¬rer_52'], [46.96785354614258, 'Albrecht_Du╠êrer_52'], [47.013675689697266, 'Albrecht_DuÔòá├¬rer_305'], [47.013675689697266, 'Albrecht_Du╠êrer_305']]
--> ACIERTO
    Primer Lugar: 65
    Entre los tres primeros: 67
<Figure size 2000x400 with 6 Axes>



Consulta número:  67
   Imagen de consulta:  Mikhail_Vrubel_140
   Mas cercanos:  [[31.088260650634766, 'Mikhail_Vrubel_140'], [33.835086822509766, 'Andrei_Rublev_52'], [33.981292724609375, 'Albrecht_DuÔòá├¬rer_227'], [33.981292724609375, 'Albrecht_Du╠êrer_227'], [34.114906311035156, 'Andrei_Rublev_82'], [34.15849304199219, 'Albrecht_DuÔòá├¬rer_95'], [34.15849304199219, 'Albrecht_Du╠êrer_95'], [34.40596389770508, 'Albrecht_Du╠êrer_199'], [34.40597152709961, 'Albrecht_DuÔòá├¬rer_199'], [34.89051818847656, 'Andrei_Rublev_48']]
--> ACIERTO
    Primer Lugar: 66
    Entre los tres primeros: 68
<Figure size 2000x400 with 6 Axes>



Consulta número:  68
   Imagen de consulta:  Mikhail_Vrubel_16
   Mas cercanos:  [[21.33559799194336, 'Mikhail_Vrubel_16'], [25.042293548583984, 'Alfred_Sisley_193'], [25.372339248657227, 'Alfred_Sisley_184'], [25.735305786132812, 'Claude_Monet_1'], [25.756027221679688, 'Alfred_Sisley_114'], [25.996129989624023, 'Alfred_Sisley_248'], [26.291282653808594, 'Alfred_Sisley_203'], [26.2972469329834, 'Alfred_Sisley_225'], [26.38193130493164, 'Alfred_Sisley_39'], [26.3829288482666, 'Camille_Pissarro_50']]
--> ACIERTO
    Primer Lugar: 67
    Entre los tres primeros: 69
<Figure size 2000x400 with 6 Axes>



Consulta número:  69
   Imagen de consulta:  Mikhail_Vrubel_36
   Mas cercanos:  [[2.9120850563049316, 'Mikhail_Vrubel_36'], [14.986164093017578, 'Alfred_Sisley_126'], [15.190265655517578, 'Claude_Monet_73'], [15.389487266540527, 'Albrecht_DuÔòá├¬rer_225'], [15.389494895935059, 'Albrecht_Du╠êrer_225'], [15.482259750366211, 'Albrecht_Du╠êrer_184'], [15.482260704040527, 'Albrecht_DuÔòá├¬rer_184'], [15.501387596130371, 'Alfred_Sisley_226'], [15.589670181274414, 'Albrecht_DuÔòá├¬rer_13'], [15.589675903320312, 'Albrecht_Du╠êrer_13']]
--> ACIERTO
    Primer Lugar: 68
    Entre los tres primeros: 70
<Figure size 2000x400 with 6 Axes>



Consulta número:  70
   Imagen de consulta:  Mikhail_Vrubel_5
   Mas cercanos:  [[3.065040111541748, 'Mikhail_Vrubel_5'], [27.248821258544922, 'Alfred_Sisley_228'], [27.59538459777832, 'Alfred_Sisley_231'], [28.203824996948242, 'Albrecht_Du╠êrer_155'], [28.203834533691406, 'Albrecht_DuÔòá├¬rer_155'], [28.206899642944336, 'Alfred_Sisley_201'], [28.345518112182617, 'Alfred_Sisley_241'], [28.357547760009766, 'Alfred_Sisley_63'], [28.370058059692383, 'Albrecht_DuÔòá├¬rer_140'], [28.370059967041016, 'Albrecht_Du╠êrer_140']]
--> ACIERTO
    Primer Lugar: 69
    Entre los tres primeros: 71
<Figure size 2000x400 with 6 Axes>



Consulta número:  71
   Imagen de consulta:  Mikhail_Vrubel_67
   Mas cercanos:  [[5.317039966583252, 'Mikhail_Vrubel_67'], [28.294984817504883, 'Albrecht_Du╠êrer_285'], [28.294992446899414, 'Albrecht_DuÔòá├¬rer_285'], [28.709579467773438, 'Alfred_Sisley_13'], [28.932723999023438, 'Albrecht_DuÔòá├¬rer_302'], [28.932723999023438, 'Albrecht_Du╠êrer_302'], [28.959426879882812, 'Albrecht_Du╠êrer_156'], [28.959428787231445, 'Albrecht_DuÔòá├¬rer_156'], [29.057270050048828, 'Albrecht_Du╠êrer_294'], [29.057275772094727, 'Albrecht_DuÔòá├¬rer_294']]
--> ACIERTO
    Primer Lugar: 70
    Entre los tres primeros: 72
<Figure size 2000x400 with 6 Axes>



Consulta número:  72
   Imagen de consulta:  Mikhail_Vrubel_96
   Mas cercanos:  [[6.222582817077637, 'Mikhail_Vrubel_96'], [37.64654541015625, 'Camille_Pissarro_27'], [38.0, 'Albrecht_DuÔòá├¬rer_189'], [38.00000762939453, 'Albrecht_Du╠êrer_189'], [38.23859405517578, 'Albrecht_Du╠êrer_248'], [38.23860168457031, 'Albrecht_DuÔòá├¬rer_248'], [38.3360710144043, 'Andrei_Rublev_89'], [38.55366897583008, 'Albrecht_DuÔòá├¬rer_278'], [38.553672790527344, 'Albrecht_Du╠êrer_278'], [38.737979888916016, 'Edgar_Degas_322']]
--> ACIERTO
    Primer Lugar: 71
    Entre los tres primeros: 73
<Figure size 2000x400 with 6 Axes>



Consulta número:  73
   Imagen de consulta:  Pablo_Picasso_249
   Mas cercanos:  [[4.360044956207275, 'Pablo_Picasso_249'], [32.48509979248047, 'Albrecht_DuÔòá├¬rer_304'], [32.48509979248047, 'Albrecht_Du╠êrer_304'], [32.567771911621094, 'Claude_Monet_32'], [32.679229736328125, 'Albrecht_DuÔòá├¬rer_238'], [32.679229736328125, 'Albrecht_Du╠êrer_238'], [32.717742919921875, 'Albrecht_Du╠êrer_285'], [32.71774673461914, 'Albrecht_DuÔòá├¬rer_285'], [32.9124870300293, 'Albrecht_DuÔòá├¬rer_302'], [32.9124870300293, 'Albrecht_Du╠êrer_302']]
--> ACIERTO
    Primer Lugar: 72
    Entre los tres primeros: 74
<Figure size 2000x400 with 6 Axes>



Consulta número:  74
   Imagen de consulta:  Pablo_Picasso_305
   Mas cercanos:  [[5.462154388427734, 'Pablo_Picasso_305'], [57.912940979003906, 'Albrecht_Du╠êrer_16'], [57.91295623779297, 'Albrecht_DuÔòá├¬rer_16'], [58.267608642578125, 'Diego_Velazquez_118'], [58.43986511230469, 'Albrecht_Du╠êrer_297'], [58.43986892700195, 'Albrecht_DuÔòá├¬rer_297'], [58.458499908447266, 'Albrecht_DuÔòá├¬rer_28'], [58.458499908447266, 'Albrecht_Du╠êrer_28'], [58.52901840209961, 'Albrecht_DuÔòá├¬rer_224'], [58.52902603149414, 'Albrecht_Du╠êrer_224']]
--> ACIERTO
    Primer Lugar: 73
    Entre los tres primeros: 75
<Figure size 2000x400 with 6 Axes>



Consulta número:  75
   Imagen de consulta:  Pablo_Picasso_322
   Mas cercanos:  [[18.738147735595703, 'Pablo_Picasso_322'], [58.28359603881836, 'Diego_Rivera_34'], [59.38250732421875, 'Albrecht_DuÔòá├¬rer_36'], [59.38250732421875, 'Albrecht_Du╠êrer_36'], [61.32539367675781, 'Diego_Rivera_53'], [61.353511810302734, 'Albrecht_DuÔòá├¬rer_223'], [61.353511810302734, 'Albrecht_Du╠êrer_223'], [61.36211395263672, 'Albrecht_DuÔòá├¬rer_188'], [61.36212921142578, 'Albrecht_Du╠êrer_188'], [61.52865982055664, 'Albrecht_DuÔòá├¬rer_282']]
--> ACIERTO
    Primer Lugar: 74
    Entre los tres primeros: 76
<Figure size 2000x400 with 6 Axes>



Consulta número:  76
   Imagen de consulta:  Pablo_Picasso_381
   Mas cercanos:  [[30.764402389526367, 'Pablo_Picasso_381'], [58.02173614501953, 'Albrecht_DuÔòá├¬rer_188'], [58.02174377441406, 'Albrecht_Du╠êrer_188'], [58.52959442138672, 'Andy_Warhol_94'], [59.13517379760742, 'Albrecht_DuÔòá├¬rer_99'], [59.13517379760742, 'Albrecht_Du╠êrer_99'], [60.54175567626953, 'Andrei_Rublev_77'], [60.81676483154297, 'Albrecht_Du╠êrer_246'], [60.8167724609375, 'Albrecht_DuÔòá├¬rer_246'], [61.987281799316406, 'Albrecht_Du╠êrer_297']]
--> ACIERTO
    Primer Lugar: 75
    Entre los tres primeros: 77
<Figure size 2000x400 with 6 Axes>



Consulta número:  77
   Imagen de consulta:  Pablo_Picasso_438
   Mas cercanos:  [[44.83351135253906, 'Albrecht_DuÔòá├¬rer_151'], [44.83351135253906, 'Albrecht_Du╠êrer_151'], [45.156612396240234, 'Albrecht_DuÔòá├¬rer_206'], [45.156612396240234, 'Albrecht_Du╠êrer_206'], [45.29219055175781, 'Camille_Pissarro_74'], [45.367706298828125, 'Claude_Monet_39'], [45.37216567993164, 'Andrei_Rublev_79'], [45.497215270996094, 'Albrecht_DuÔòá├¬rer_273'], [45.49721908569336, 'Albrecht_Du╠êrer_273'], [45.631126403808594, 'Albrecht_DuÔòá├¬rer_52']]
<Figure size 2000x400 with 6 Axes>



Consulta número:  78
   Imagen de consulta:  Pablo_Picasso_46
   Mas cercanos:  [[38.0004997253418, 'Albrecht_DuÔòá├¬rer_52'], [38.0004997253418, 'Albrecht_Du╠êrer_52'], [38.47174072265625, 'Andrei_Rublev_33'], [38.9425048828125, 'Alfred_Sisley_90'], [39.000091552734375, 'Camille_Pissarro_27'], [39.14590072631836, 'Albrecht_DuÔòá├¬rer_278'], [39.145904541015625, 'Albrecht_Du╠êrer_278'], [39.277103424072266, 'Albrecht_DuÔòá├¬rer_189'], [39.27711486816406, 'Albrecht_Du╠êrer_189'], [39.43619155883789, 'Alfred_Sisley_29']]
<Figure size 2000x400 with 6 Axes>



Consulta número:  79
   Imagen de consulta:  Paul_Gauguin_107
   Mas cercanos:  [[27.348106384277344, 'Paul_Gauguin_107'], [31.079166412353516, 'Andy_Warhol_110'], [31.338939666748047, 'Albrecht_DuÔòá├¬rer_95'], [31.338939666748047, 'Albrecht_Du╠êrer_95'], [31.666629791259766, 'Albrecht_DuÔòá├¬rer_227'], [31.66663360595703, 'Albrecht_Du╠êrer_227'], [32.24867248535156, 'Albrecht_DuÔòá├¬rer_209'], [32.24867248535156, 'Albrecht_Du╠êrer_209'], [32.287906646728516, 'Albrecht_DuÔòá├¬rer_271'], [32.287906646728516, 'Albrecht_Du╠êrer_271']]
--> ACIERTO
    Primer Lugar: 76
    Entre los tres primeros: 78
<Figure size 2000x400 with 6 Axes>



Consulta número:  80
   Imagen de consulta:  Paul_Gauguin_116
   Mas cercanos:  [[15.65234661102295, 'Paul_Gauguin_116'], [22.937536239624023, 'Mikhail_Vrubel_36'], [23.03680992126465, 'Albrecht_DuÔòá├¬rer_13'], [23.03680992126465, 'Albrecht_Du╠êrer_13'], [23.04427719116211, 'Camille_Pissarro_57'], [23.188907623291016, 'Albrecht_Du╠êrer_252'], [23.188913345336914, 'Albrecht_DuÔòá├¬rer_252'], [23.190044403076172, 'Albrecht_DuÔòá├¬rer_195'], [23.190052032470703, 'Albrecht_Du╠êrer_195'], [23.2034854888916, 'Albrecht_Du╠êrer_39']]
--> ACIERTO
    Primer Lugar: 77
    Entre los tres primeros: 79
<Figure size 2000x400 with 6 Axes>



Consulta número:  81
   Imagen de consulta:  Paul_Gauguin_290
   Mas cercanos:  [[4.119620323181152, 'Paul_Gauguin_290'], [18.704517364501953, 'Albrecht_DuÔòá├¬rer_13'], [18.704524993896484, 'Albrecht_Du╠êrer_13'], [18.770858764648438, 'Amedeo_Modigliani_142'], [18.786588668823242, 'Edgar_Degas_315'], [18.903831481933594, 'Albrecht_Du╠êrer_202'], [18.90383529663086, 'Albrecht_DuÔòá├¬rer_202'], [18.91900634765625, 'Claude_Monet_73'], [18.996850967407227, 'Albrecht_DuÔòá├¬rer_316'], [18.996858596801758, 'Albrecht_Du╠êrer_316']]
--> ACIERTO
    Primer Lugar: 78
    Entre los tres primeros: 80
<Figure size 2000x400 with 6 Axes>



Consulta número:  82
   Imagen de consulta:  Paul_Gauguin_49
   Mas cercanos:  [[5.6872639656066895, 'Paul_Gauguin_49'], [43.66930389404297, 'Albrecht_DuÔòá├¬rer_176'], [43.669315338134766, 'Albrecht_Du╠êrer_176'], [44.64677047729492, 'Albrecht_DuÔòá├¬rer_174'], [44.64678192138672, 'Albrecht_Du╠êrer_174'], [44.70083999633789, 'Albrecht_Du╠êrer_253'], [44.70084762573242, 'Albrecht_DuÔòá├¬rer_253'], [44.921451568603516, 'Edgar_Degas_195'], [44.999351501464844, 'Camille_Pissarro_47'], [45.025535583496094, 'Alfred_Sisley_146']]
--> ACIERTO
    Primer Lugar: 79
    Entre los tres primeros: 81
<Figure size 2000x400 with 6 Axes>



Consulta número:  83
   Imagen de consulta:  Paul_Klee_1
   Mas cercanos:  [[9.097290992736816, 'Paul_Klee_1'], [37.9144287109375, 'Alfred_Sisley_100'], [38.138999938964844, 'Camille_Pissarro_47'], [38.15339660644531, 'Camille_Pissarro_17'], [38.17589569091797, 'Alfred_Sisley_76'], [38.27949523925781, 'Camille_Pissarro_15'], [38.42685317993164, 'Camille_Pissarro_37'], [38.6910400390625, 'Claude_Monet_49'], [38.70603561401367, 'Amedeo_Modigliani_188'], [38.74822998046875, 'Alfred_Sisley_253']]
--> ACIERTO
    Primer Lugar: 80
    Entre los tres primeros: 82
<Figure size 2000x400 with 6 Axes>



Consulta número:  84
   Imagen de consulta:  Paul_Klee_14
   Mas cercanos:  [[3.864387035369873, 'Paul_Klee_14'], [35.98316192626953, 'Alfred_Sisley_140'], [37.54800033569336, 'Alfred_Sisley_167'], [37.605464935302734, 'Albrecht_Du╠êrer_105'], [37.605472564697266, 'Albrecht_DuÔòá├¬rer_105'], [37.630435943603516, 'Camille_Pissarro_15'], [37.68490982055664, 'Amedeo_Modigliani_144'], [37.69672393798828, 'Andrei_Rublev_55'], [37.73469543457031, 'Alfred_Sisley_95'], [37.746803283691406, 'Alfred_Sisley_253']]
--> ACIERTO
    Primer Lugar: 81
    Entre los tres primeros: 83
<Figure size 2000x400 with 6 Axes>



Consulta número:  85
   Imagen de consulta:  Paul_Klee_23
   Mas cercanos:  [[4.527896404266357, 'Paul_Klee_23'], [31.779653549194336, 'Andrei_Rublev_74'], [31.808151245117188, 'Edgar_Degas_188'], [32.0966911315918, 'Albrecht_DuÔòá├¬rer_270'], [32.09669494628906, 'Albrecht_Du╠êrer_270'], [32.13425064086914, 'Alfred_Sisley_174'], [32.17020034790039, 'Edgar_Degas_187'], [32.185951232910156, 'Camille_Pissarro_57'], [32.18992233276367, 'Albrecht_DuÔòá├¬rer_195'], [32.18992614746094, 'Albrecht_Du╠êrer_195']]
--> ACIERTO
    Primer Lugar: 82
    Entre los tres primeros: 84
<Figure size 2000x400 with 6 Axes>



Consulta número:  86
   Imagen de consulta:  Paul_Klee_38
   Mas cercanos:  [[13.436422348022461, 'Paul_Klee_38'], [37.2968635559082, 'Amedeo_Modigliani_17'], [37.602989196777344, 'Amedeo_Modigliani_9'], [37.89509201049805, 'Camille_Pissarro_90'], [38.095985412597656, 'Andy_Warhol_46'], [38.17751693725586, 'Edgar_Degas_304'], [38.301666259765625, 'Edgar_Degas_147'], [38.31985855102539, 'Albrecht_DuÔòá├¬rer_95'], [38.31985855102539, 'Albrecht_Du╠êrer_95'], [38.39912033081055, 'Amedeo_Modigliani_8']]
--> ACIERTO
    Primer Lugar: 83
    Entre los tres primeros: 85
<Figure size 2000x400 with 6 Axes>



Consulta número:  87
   Imagen de consulta:  Pierre-Auguste_Renoir_142
   Mas cercanos:  [[15.186469078063965, 'Pierre-Auguste_Renoir_142'], [36.68653106689453, 'Edgar_Degas_103'], [36.863121032714844, 'Claude_Monet_65'], [36.930633544921875, 'Alfred_Sisley_28'], [37.0114860534668, 'Camille_Pissarro_51'], [37.016685485839844, 'Albrecht_DuÔòá├¬rer_140'], [37.016685485839844, 'Albrecht_Du╠êrer_140'], [37.12067794799805, 'Alfred_Sisley_94'], [37.130733489990234, 'Edgar_Degas_101'], [37.186092376708984, 'Albrecht_DuÔòá├¬rer_229']]
--> ACIERTO
    Primer Lugar: 84
    Entre los tres primeros: 86
<Figure size 2000x400 with 6 Axes>



Consulta número:  88
   Imagen de consulta:  Pierre-Auguste_Renoir_257
   Mas cercanos:  [[22.84090805053711, 'Pierre-Auguste_Renoir_257'], [35.19008255004883, 'Albrecht_Du╠êrer_46'], [35.190093994140625, 'Albrecht_DuÔòá├¬rer_46'], [35.288978576660156, 'Albrecht_DuÔòá├¬rer_227'], [35.28898620605469, 'Albrecht_Du╠êrer_227'], [35.34926223754883, 'Andrei_Rublev_42'], [35.42507553100586, 'Alfred_Sisley_248'], [35.68806076049805, 'Andrei_Rublev_55'], [35.72446060180664, 'Albrecht_DuÔòá├¬rer_123'], [35.72446060180664, 'Albrecht_Du╠êrer_123']]
--> ACIERTO
    Primer Lugar: 85
    Entre los tres primeros: 87
<Figure size 2000x400 with 6 Axes>



Consulta número:  89
   Imagen de consulta:  Rene_Magritte_138
   Mas cercanos:  [[6.886043548583984, 'Rene_Magritte_138'], [26.53824234008789, 'Camille_Pissarro_40'], [26.574142456054688, 'Albrecht_DuÔòá├¬rer_316'], [26.574146270751953, 'Albrecht_Du╠êrer_316'], [26.7692813873291, 'Amedeo_Modigliani_117'], [26.86524200439453, 'Albrecht_DuÔòá├¬rer_155'], [26.86524200439453, 'Albrecht_Du╠êrer_155'], [26.88744354248047, 'Albrecht_DuÔòá├¬rer_301'], [26.88744354248047, 'Albrecht_Du╠êrer_301'], [27.013830184936523, 'Albrecht_DuÔòá├¬rer_208']]
--> ACIERTO
    Primer Lugar: 86
    Entre los tres primeros: 88
<Figure size 2000x400 with 6 Axes>



Consulta número:  90
   Imagen de consulta:  Salvador_Dali_35
   Mas cercanos:  [[32.05120086669922, 'Salvador_Dali_35'], [36.79553985595703, 'Alfred_Sisley_13'], [37.11256408691406, 'Albrecht_DuÔòá├¬rer_95'], [37.11256408691406, 'Albrecht_Du╠êrer_95'], [37.11574172973633, 'Claude_Monet_1'], [37.27175521850586, 'Andrei_Rublev_38'], [37.27267837524414, 'Alfred_Sisley_38'], [37.339149475097656, 'Andrei_Rublev_52'], [37.376251220703125, 'Diego_Rivera_61'], [37.41574478149414, 'Andy_Warhol_85']]
--> ACIERTO
    Primer Lugar: 87
    Entre los tres primeros: 89
<Figure size 2000x400 with 6 Axes>



Consulta número:  91
   Imagen de consulta:  Salvador_Dali_61
   Mas cercanos:  [[10.669782638549805, 'Salvador_Dali_61'], [34.59703063964844, 'Alfred_Sisley_5'], [34.61711502075195, 'Alfred_Sisley_241'], [34.65556716918945, 'Alfred_Sisley_174'], [34.67802810668945, 'Camille_Pissarro_51'], [34.7547607421875, 'Edgar_Degas_345'], [34.76125717163086, 'Alfred_Sisley_99'], [34.774566650390625, 'Camille_Pissarro_37'], [34.87422180175781, 'Alfred_Sisley_80'], [34.88370132446289, 'Albrecht_DuÔòá├¬rer_156']]
--> ACIERTO
    Primer Lugar: 88
    Entre los tres primeros: 90
<Figure size 2000x400 with 6 Axes>



Consulta número:  92
   Imagen de consulta:  Salvador_Dali_8
   Mas cercanos:  [[5.586972236633301, 'Salvador_Dali_8'], [36.66075897216797, 'Claude_Monet_46'], [36.85325622558594, 'Albrecht_Du╠êrer_156'], [36.8532600402832, 'Albrecht_DuÔòá├¬rer_156'], [36.906795501708984, 'Alfred_Sisley_42'], [36.945411682128906, 'Albrecht_DuÔòá├¬rer_294'], [36.945411682128906, 'Albrecht_Du╠êrer_294'], [37.197410583496094, 'Alfred_Sisley_238'], [37.40216827392578, 'Claude_Monet_67'], [37.494285583496094, 'Albrecht_DuÔòá├¬rer_59']]
--> ACIERTO
    Primer Lugar: 89
    Entre los tres primeros: 91
<Figure size 2000x400 with 6 Axes>



Consulta número:  93
   Imagen de consulta:  Sandro_Botticelli_74
   Mas cercanos:  [[30.111438751220703, 'Sandro_Botticelli_74'], [43.66364669799805, 'Edgar_Degas_224'], [43.884796142578125, 'Albrecht_DuÔòá├¬rer_60'], [43.88480758666992, 'Albrecht_Du╠êrer_60'], [43.998565673828125, 'Albrecht_DuÔòá├¬rer_57'], [43.998565673828125, 'Albrecht_Du╠êrer_57'], [44.03807067871094, 'Camille_Pissarro_27'], [44.0783576965332, 'Amedeo_Modigliani_20'], [44.13215255737305, 'Edgar_Degas_292'], [44.24827194213867, 'Albrecht_DuÔòá├¬rer_6']]
--> ACIERTO
    Primer Lugar: 90
    Entre los tres primeros: 92
<Figure size 2000x400 with 6 Axes>



Consulta número:  94
   Imagen de consulta:  Titian_81
   Mas cercanos:  [[19.894039154052734, 'Titian_81'], [23.496320724487305, 'Amedeo_Modigliani_142'], [24.332767486572266, 'Amedeo_Modigliani_7'], [24.37163734436035, 'Albrecht_DuÔòá├¬rer_184'], [24.371641159057617, 'Albrecht_Du╠êrer_184'], [24.38078498840332, 'Edgar_Degas_315'], [24.4143009185791, 'Albrecht_DuÔòá├¬rer_61'], [24.4143009185791, 'Albrecht_Du╠êrer_61'], [24.41790008544922, 'Alfred_Sisley_64'], [24.438358306884766, 'Albrecht_DuÔòá├¬rer_225']]
--> ACIERTO
    Primer Lugar: 91
    Entre los tres primeros: 93
<Figure size 2000x400 with 6 Axes>



Consulta número:  95
   Imagen de consulta:  Vincent_van_Gogh_276
   Mas cercanos:  [[26.589101791381836, 'Vincent_van_Gogh_276'], [40.183597564697266, 'Albrecht_DuÔòá├¬rer_197'], [40.18360137939453, 'Albrecht_Du╠êrer_197'], [40.309898376464844, 'Albrecht_DuÔòá├¬rer_42'], [40.309906005859375, 'Albrecht_Du╠êrer_42'], [40.94388961791992, 'Albrecht_DuÔòá├¬rer_129'], [40.94389343261719, 'Albrecht_Du╠êrer_129'], [40.9804801940918, 'Alfred_Sisley_29'], [41.10280227661133, 'Claude_Monet_46'], [41.15005111694336, 'Albrecht_DuÔòá├¬rer_176']]
--> ACIERTO
    Primer Lugar: 92
    Entre los tres primeros: 94
<Figure size 2000x400 with 6 Axes>



Consulta número:  96
   Imagen de consulta:  Vincent_van_Gogh_281
   Mas cercanos:  [[6.023087978363037, 'Vincent_van_Gogh_281'], [21.973682403564453, 'Albrecht_DuÔòá├¬rer_49'], [21.973682403564453, 'Albrecht_Du╠êrer_49'], [22.476886749267578, 'Alfred_Sisley_92'], [22.588581085205078, 'Albrecht_DuÔòá├¬rer_43'], [22.588581085205078, 'Albrecht_Du╠êrer_43'], [22.64059829711914, 'Camille_Pissarro_64'], [22.643970489501953, 'Albrecht_Du╠êrer_131'], [22.643972396850586, 'Albrecht_DuÔòá├¬rer_131'], [22.66032600402832, 'Albrecht_DuÔòá├¬rer_274']]
--> ACIERTO
    Primer Lugar: 93
    Entre los tres primeros: 95
<Figure size 2000x400 with 6 Axes>



Consulta número:  97
   Imagen de consulta:  Vincent_van_Gogh_309
   Mas cercanos:  [[14.17460823059082, 'Vincent_van_Gogh_309'], [14.852246284484863, 'Albrecht_DuÔòá├¬rer_225'], [14.852252006530762, 'Albrecht_Du╠êrer_225'], [14.910338401794434, 'Claude_Monet_73'], [14.975250244140625, 'Alfred_Sisley_126'], [15.122296333312988, 'Amedeo_Modigliani_142'], [15.198454856872559, 'Amedeo_Modigliani_7'], [15.203206062316895, 'Alfred_Sisley_15'], [15.20926570892334, 'Alfred_Sisley_64'], [15.34941291809082, 'Albrecht_DuÔòá├¬rer_184']]
--> ACIERTO
    Primer Lugar: 94
    Entre los tres primeros: 96
<Figure size 2000x400 with 6 Axes>



Consulta número:  98
   Imagen de consulta:  Vincent_van_Gogh_645
   Mas cercanos:  [[28.54009437561035, 'Amedeo_Modigliani_72'], [29.730037689208984, 'Vincent_van_Gogh_645'], [30.079071044921875, 'Amedeo_Modigliani_20'], [30.090496063232422, 'Amedeo_Modigliani_54'], [30.096900939941406, 'Amedeo_Modigliani_35'], [30.542531967163086, 'Camille_Pissarro_22'], [30.776704788208008, 'Albrecht_DuÔòá├¬rer_187'], [30.77670669555664, 'Albrecht_Du╠êrer_187'], [30.801881790161133, 'Albrecht_DuÔòá├¬rer_195'], [30.801883697509766, 'Albrecht_Du╠êrer_195']]
--> ACIERTO
    Entre los tres primeros: 97
<Figure size 2000x400 with 6 Axes>



Consulta número:  99
   Imagen de consulta:  William_Turner_65
   Mas cercanos:  [[16.863439559936523, 'William_Turner_65'], [17.302392959594727, 'Albrecht_DuÔòá├¬rer_184'], [17.30239486694336, 'Albrecht_Du╠êrer_184'], [17.439863204956055, 'Albrecht_Du╠êrer_252'], [17.439868927001953, 'Albrecht_DuÔòá├¬rer_252'], [17.899372100830078, 'Albrecht_Du╠êrer_202'], [17.89937400817871, 'Albrecht_DuÔòá├¬rer_202'], [17.95025634765625, 'Albrecht_DuÔòá├¬rer_225'], [17.95026397705078, 'Albrecht_Du╠êrer_225'], [17.972396850585938, 'Albrecht_Du╠êrer_112']]
--> ACIERTO
    Primer Lugar: 95
    Entre los tres primeros: 98
<Figure size 2000x400 with 6 Axes>



Porcentaje de aciertos NN1:  0.95
Porcentaje de aciertos NN3:  0.98
Porcentaje de aciertos NN5:  0.98
  Cantidad consultas:  100
```


**[Celda 21 - Código]**
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
['Pablo_Picasso_438', 'Pablo_Picasso_46']
[2040, 2041]
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>
```


**[Celda 22 - Código]**
```python
print(y_imagenesBD[133])
np.where(y_imagenesBD=='102462')
```


*Salida:*
```text
102462
(array([133], dtype=int64),)
```


**[Celda 23 - Código]**
```python
conv2_block1_out
['Diego_Rivera_22', 'Edgar_Degas_15', 'Marc_Chagall_52', 'Michelangelo_21', 'Mikhail_Vrubel_16', 'Pablo_Picasso_438', 'Pablo_Picasso_46', 'Vincent_van_Gogh_645']
[1621, 1770, 2024, 2029, 2031, 2040, 2041, 2061]

conv2_block3_out

conv3_block2_out
['Michelangelo_21', 'Pablo_Picasso_438', 'Pablo_Picasso_46', 'Vincent_van_Gogh_645']
[2029, 2040, 2041, 2061]

conv3_block3_out
['Pablo_Picasso_438', 'Pablo_Picasso_46']
[2040, 2041]

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


**[Celda 28 - Código]**
```python
# Asignar la capa de pooling global a una variable
# salida = resnet.get_layer('conv4_block5_add')
# salida = resnet.get_layer('conv4_block2_add')
salida = resnet.get_layer('pool1_pool')
salida2 = resnet.get_layer('conv3_block4_out')

# Promedio de:
# salida = resnet.get_layer('pool1_pool')
# salida2 = resnet.get_layer('conv3_block4_out')
# Porcentaje de aciertos NN1:  0.95
# Porcentaje de aciertos NN3:  0.98
# Porcentaje de aciertos NN5:  0.98



# pool1_pool (56, 56, 64)
# Porcentaje de aciertos NN1:  0.88
# Porcentaje de aciertos NN3:  0.89
# Porcentaje de aciertos NN5:  0.92

### RESULTADO conv2_block1_out (56, 56, 256)
# Porcentaje de aciertos NN1:  0.89
# Porcentaje de aciertos NN3:  0.91
# Porcentaje de aciertos NN5:  0.93


### RESULTADO conv2_block3_out (56, 56, 256)
# Porcentaje de aciertos NN1:  0.88
# Porcentaje de aciertos NN3:  0.89
# Porcentaje de aciertos NN5:  0.9

### RESULTADO conv3_block2_out (28, 28, 512)
# Porcentaje de aciertos NN1:  0.94
# Porcentaje de aciertos NN3:  0.96
# Porcentaje de aciertos NN5:  0.96

### RESULTADO conv3_block3_out (28, 28, 512)
# Porcentaje de aciertos NN1:  0.94
# Porcentaje de aciertos NN3:  0.97
# Porcentaje de aciertos NN5:  0.98

### RESULTADO conv3_block4_out (28, 28, 512)
# Porcentaje de aciertos NN1:  0.95
# Porcentaje de aciertos NN3:  0.98
# Porcentaje de aciertos NN5:  0.98

### RESULTADO conv4_block1_out (14, 14, 1024)
# Porcentaje de aciertos NN1:  0.93
# Porcentaje de aciertos NN3:  0.97
# Porcentaje de aciertos NN5:  0.98

### RESULTADO conv4_block4_out (14, 14, 1024)
# Porcentaje de aciertos NN1:  0.91
# Porcentaje de aciertos NN3:  0.95
# Porcentaje de aciertos NN5:  0.96

### RESULTADO conv5_block3_out (7, 7, 2048)
# Porcentaje de aciertos NN1:  0.87
# Porcentaje de aciertos NN3:  0.88
# Porcentaje de aciertos NN5:  0.9

### RESULTADO global_average_pooling2d (1, 1, 2048)
# Porcentaje de aciertos NN1:  0.73
# Porcentaje de aciertos NN3:  0.78
# Porcentaje de aciertos NN5:  0.81

# Crear un nuevo modelo que tenga la capa de pooling global como salida
new_model_resnet = Model(inputs=resnet.input, outputs=salida.output)
new_model_resnet2 = Model(inputs=resnet.input, outputs=salida2.output)
```


**[Celda 29 - Código]**
```python

```

## Fine Tuning


**[Celda 31 - Código]**
```python
imagenesBD[0].shape
print(len(imagenesBD))
```


*Salida:*
```text
10388
```


**[Celda 32 - Código]**
```python
 np.where(y_imagenesBD == '131152')
```


*Salida:*
```text
(array([2285], dtype=int64),)
```


**[Celda 33 - Código]**
```python
np.where(y_consultas == '1708')
```


*Salida:*
```text
(array([1], dtype=int64),)
```


**[Celda 34 - Código]**
```python
dist = np.sqrt(np.sum(np.square(vectores_consulta[0]-vectores_elemento[133])))
print(dist)
```


*Salida:*
```text
118.8077
```


**[Celda 35 - Código]**
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


**[Celda 36 - Código]**
```python
type(imagenesBD[0])
```


*Salida:*
```text
numpy.ndarray
```


**[Celda 37 - Código]**
```python
y_consultas[77]
```


*Salida:*
```text
'boca'
```


**[Celda 38 - Código]**
```python
type(consultas[0])
```


*Salida:*
```text
numpy.ndarray
```


**[Celda 39 - Código]**
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


**[Celda 40 - Código]**
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


**[Celda 41 - Código]**
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


**[Celda 42 - Código]**
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


**[Celda 43 - Código]**
```python

```
