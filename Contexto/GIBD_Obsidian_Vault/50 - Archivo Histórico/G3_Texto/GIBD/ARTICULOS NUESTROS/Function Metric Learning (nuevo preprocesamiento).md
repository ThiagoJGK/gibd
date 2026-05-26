---
aliases: [Function Metric Learning (nuevo preprocesamiento)]
tags:
  - grupo/g3_texto
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2023-08-07
origen_zip: GIBD-20260521T205218Z-3-004.zip
ruta_interna: "GIBD/ARTICULOS NUESTROS/2023 - CONAIISI/Caracteres Chinos/Function Metric Learning (nuevo preprocesamiento).ipynb"
tamanio_bytes: 3049913
---

# Notebook: Function Metric Learning (nuevo preprocesamiento).ipynb

Ruta interna: `GIBD/ARTICULOS NUESTROS/2023 - CONAIISI/Caracteres Chinos/Function Metric Learning (nuevo preprocesamiento).ipynb`

---


**[Celda 1 - Código]**
```python
#Declaraciones 
from tensorflow.keras.layers import Input, Conv2D, Lambda, Dense, GlobalAveragePooling2D,Flatten,MaxPooling2D,Activation, Dropout, BatchNormalization, Concatenate # merge,
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
from tqdm.notebook import tqdm_notebook

BATCH_SIZE = 128
ANCHO = 28
ALTO = 28
RADIO = 5
N_CLASES = 203
DIRECTORIO = r'D:\Escritorio\GIBD'
```

# Preprocesar Imágenes
###### Código necesario para realizar el procesamiento de las imágenes


**[Celda 3 - Código]**
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


**[Celda 4 - Código]**
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

# Carga de imágenes
###### Código necesario para la carga de las imágenes y armado de los pares


**[Celda 6 - Código]**
```python
#Cargar imágenes entrenamiento

inicio= time.time()
# Selección del directorio que contiene la base de datos de marcas (con aumentación y relieve) 
directorio = r'D:\Escritorio\GIBD\Bases de datos\BDNormalizada_R_Aumentada(28x28)_91350_Relieve'

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
        if len(nombre)>2:
            n = nombre[:nombre.find('_')]
        else:
            n=nombre
        dic[int(n)].append(image_gris) 
        
fin = time.time()
sum = 0
for key in dic.keys():
    sum += len(dic[key])
print(str(sum)+ " imágenes cargadas, tiempo de ejecucion: " + str(fin-inicio))    
```


*Salida:*
```text
91350 imágenes cargadas, tiempo de ejecucion: 44.82025957107544
```


**[Celda 7 - Código]**
```python
print("cantidad de imágenes: " + str(sum))
for key in dic.keys():
    print("Clase: " + str(key) + " Cantidad de imágenes: " + str(len(dic[key]))) 
```


**[Celda 8 - Código]**
```python
def obtener_pares(porc_pares_similares = 0.5, duplicar_pares = False) :
    left_input = []
    right_input = []
    targets = []    
    pares_similares = 0
    #Numero de pares por clase
    pairs = 2000
    if(duplicar_pares):        
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
    else:
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
                    targets.append(0.)
                    pares_similares += 1 
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


**[Celda 9 - Código]**
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


**[Celda 10 - Código]**
```python
#Carga Queries dibujadas
directorio = r'D:\Escritorio\GIBD\Bases de datos\QueriesDibujadas'
#directorio = r'D:\Escritorio\GIBD\Bases de datos\NuevasMarcasQueries'
contenido = os.listdir(directorio)
consultas_dibujadas = []
y_consultas_dibujadas = []
for fichero in contenido:
    if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.jpg'):
        image = directorio + '/' + fichero
        image_gris = cv2.imread(image, cv2.IMREAD_GRAYSCALE)       
        consultas_dibujadas.append(preprocesarImagen(image_gris,ANCHO,ALTO,RADIO))
        nombre = fichero[:fichero.find('.')]
        #Código útil si se quiere cargar las imágenes de la db aumentada
        #if len(nombre)>2:
        #    n = nombre[:nombre.find('_')]
        #else:
        #    n=nombre       
        n=nombre
        y_consultas_dibujadas.append(n)
print(y_consultas_dibujadas)
```


*Salida:*
```text
['q_100', 'q_107', 'q_115', 'q_118', 'q_125', 'q_126', 'q_127', 'q_128', 'q_129', 'q_130', 'q_131', 'q_132', 'q_133', 'q_134', 'q_134', 'q_143', 'q_144', 'q_145', 'q_146', 'q_166', 'q_170', 'q_174', 'q_175', 'q_192', 'q_195', 'q_197', 'q_2', 'q_24', 'q_31', 'q_32', 'q_37', 'q_47', 'q_55', 'q_58', 'q_59', 'q_7', 'q_79', 'q_8', 'q_88', 'q_94']
```


**[Celda 11 - Código]**
```python
#Carga Queries nuevas marcas
#directorio = r'D:\Escritorio\GIBD\Bases de datos\QueriesDibujadas'
directorio = r'D:\Escritorio\GIBD\Bases de datos\NuevasMarcasQueries'
contenido = os.listdir(directorio)
consultas_nuevas_marcas = []
y_consultas_nuevas_marcas = []
for fichero in contenido:
    if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.jpg'):
        image = directorio + '/' + fichero
        image_gris = cv2.imread(image, cv2.IMREAD_GRAYSCALE)       
        consultas_nuevas_marcas.append(preprocesarImagen(image_gris,ANCHO,ALTO,RADIO))
        nombre = fichero[:fichero.find('.')]
        #Código útil si se quiere cargar las imágenes de la db aumentada
        #if len(nombre)>2:
        #    n = nombre[:nombre.find('_')]
        #else:
        #    n=nombre       
        n=nombre
        y_consultas_nuevas_marcas.append(n)
print(y_consultas_nuevas_marcas)
```


*Salida:*
```text
['10_310', '1_301', '2_302', '3_303', '4_304', '5_305', '6_306', '7_307', '8_308', '9_309']
```

# Modelo
###### Código necesario para el entrenamiento, creación, carga, y guardado de modelos


**[Celda 13 - Código]**
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

    model.add(Dense(VECTOR_SIZE, activation='softmax'))
    return model

def crear_modelo_nuevo2():
    VECTOR_SIZE = 128

    model = Sequential()

    model.add(Conv2D(32,kernel_size=3,padding='same',activation='relu',input_shape=(ANCHO,ALTO,1)))
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
    
    return model
```


**[Celda 14 - Código]**
```python
#DEFINIMOS EL MODELO BASE, LE AGREGAMOS LOS INPUTS Y COMPILAMOS
#Creamos que modelo usar
#model = crear_modelo_viejo()
#model = crear_modelo_nuevo()
model = crear_modelo_nuevo2()


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
#final_rep = merged_representation
output = Dense(1, activation='sigmoid')(final_rep)

# Crea el modelo siamese
model_siamese = tf.keras.Model(inputs=[input_a, input_b], outputs=output)

#Compilamos el modelo  - Se puede probar con 1e-4 o 3e-4 o 5e-4
optimizer=keras.optimizers.Adam(1e-3)
model_siamese.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])

```


**[Celda 15 - Código]**
```python
model.summary()
```


*Salida:*
```text
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_12 (Conv2D)          (None, 28, 28, 32)        320       
                                                                 
 batch_normalization_12 (Bat  (None, 28, 28, 32)       128       
 chNormalization)                                                
                                                                 
 conv2d_13 (Conv2D)          (None, 28, 28, 64)        18496     
                                                                 
 batch_normalization_13 (Bat  (None, 28, 28, 64)       256       
 chNormalization)                                                
                                                                 
 activation_3 (Activation)   (None, 28, 28, 64)        0         
                                                                 
 dropout_4 (Dropout)         (None, 28, 28, 64)        0         
                                                                 
 conv2d_14 (Conv2D)          (None, 28, 28, 32)        18464     
                                                                 
 batch_normalization_14 (Bat  (None, 28, 28, 32)       128       
 chNormalization)                                                
                                                                 
 activation_4 (Activation)   (None, 28, 28, 32)        0         
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 14, 14, 32)       0         
 2D)                                                             
                                                                 
 conv2d_15 (Conv2D)          (None, 14, 14, 64)        18496     
                                                                 
 batch_normalization_15 (Bat  (None, 14, 14, 64)       256       
 chNormalization)                                                
                                                                 
 dropout_5 (Dropout)         (None, 14, 14, 64)        0         
                                                                 
 conv2d_16 (Conv2D)          (None, 14, 14, 256)       147712    
                                                                 
 batch_normalization_16 (Bat  (None, 14, 14, 256)      1024      
 chNormalization)                                                
                                                                 
 activation_5 (Activation)   (None, 14, 14, 256)       0         
                                                                 
 global_average_pooling2d (G  (None, 256)              0         
 lobalAveragePooling2D)                                          
                                                                 
 flatten_2 (Flatten)         (None, 256)               0         
                                                                 
 dense_7 (Dense)             (None, 128)               32896     
                                                                 
=================================================================
Total params: 238,176
Trainable params: 237,280
Non-trainable params: 896
_________________________________________________________________
```


**[Celda 16 - Código]**
```python
# SIAMESA

#Creamos que modelo usar
#model = crear_modelo_viejo()
model = crear_modelo_nuevo2()
# We have 2 inputs, 1 for each picture
input_a = Input((28,28,1))
input_b = Input((28,28,1))


# We will use 2 instances of 1 network for this task

# Connect each 'leg' of the network to each input
# Remember, they have the same weights
encoded_l = model(input_a)
encoded_r = model(input_b)

# Getting the L1 Distance between the 2 encodings
L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))

# Add the distance function to the network
L1_distance = L1_layer([encoded_l, encoded_r])

prediction = Dense(1,activation='sigmoid')(L1_distance)
model_siamese = Model(inputs=[input_a,input_b],outputs=prediction)

optimizer = Adam(1e-3)

#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
model_siamese.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])
```


**[Celda 17 - Código]**
```python
# SIAMESA
directorio_entrenamiento = r'D:\Escritorio\GIBD\Entrenamientos\NM_OS_BD91350R_5E_400kParesSD_adam0.001'
if not os.path.exists(directorio_entrenamiento):    
    os.makedirs(directorio_entrenamiento)
    
for i in range(27,61):
    left_input, right_input, targets = obtener_pares(porc_pares_similares = 0.5)   
    #mean_squared_error    
    history = model_siamese.fit([left_input,right_input], targets,
              batch_size=64,
              #steps_per_epoch=625,
              epochs=5,
              verbose=1,
              validation_split=0.2)    
    # Gráfica de presición
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Precisión del modelo')
    plt.ylabel('Precisión')
    plt.xlabel('epocas')
    plt.legend(['Entrenamiento', 'Validación'], loc='upper right')   
    plt.savefig(directorio_entrenamiento+"\\"+str(i)+'_precision.png')
    plt.show()
    # Gráfica de pérdida
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Pérdidas del modelo')
    plt.ylabel('perdidas')
    plt.xlabel('epocas')
    plt.legend(['Entrenamiento', 'Validación'], loc='upper right')   
    plt.savefig(directorio_entrenamiento+"\\"+str(i)+'_perdida.png')
    plt.show()
    primera_d,tercera_d,quinta_d = obtener_resultados(model_siamese,"Dibujadas")
    primera_n,tercera_n,quinta_n = obtener_resultados(model_siamese,"Nuevas")
    nombre_modelo = str(i)+"_Entreno_"+str(quinta_d)+"_"+str(quinta_n)+".h5"
    model_siamese.save(directorio_entrenamiento+"\\"+nombre_modelo)
```


*Salida:*
```text
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 0. 0. ... 0. 0. 0.]
Epoch 1/5
5048/5048 [==============================] - 66s 13ms/step - loss: 0.0402 - accuracy: 0.9871 - val_loss: 0.0285 - val_accuracy: 0.9910
Epoch 2/5
5048/5048 [==============================] - 66s 13ms/step - loss: 0.0335 - accuracy: 0.9892 - val_loss: 0.0303 - val_accuracy: 0.9908
Epoch 3/5
5048/5048 [==============================] - 67s 13ms/step - loss: 0.0291 - accuracy: 0.9906 - val_loss: 0.0346 - val_accuracy: 0.9898
Epoch 4/5
5048/5048 [==============================] - 67s 13ms/step - loss: 0.0271 - accuracy: 0.9912 - val_loss: 0.0307 - val_accuracy: 0.9909
Epoch 5/5
5048/5048 [==============================] - 66s 13ms/step - loss: 0.0254 - accuracy: 0.9919 - val_loss: 0.0328 - val_accuracy: 0.9909
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 0. 1. ... 1. 0. 1.]
Epoch 1/5
5048/5048 [==============================] - 66s 13ms/step - loss: 0.0412 - accuracy: 0.9866 - val_loss: 0.0295 - val_accuracy: 0.9906
Epoch 2/5
5048/5048 [==============================] - 67s 13ms/step - loss: 0.0335 - accuracy: 0.9892 - val_loss: 0.0273 - val_accuracy: 0.9918
Epoch 3/5
5048/5048 [==============================] - 67s 13ms/step - loss: 0.0297 - accuracy: 0.9905 - val_loss: 0.0304 - val_accuracy: 0.9908
Epoch 4/5
5048/5048 [==============================] - 67s 13ms/step - loss: 0.0265 - accuracy: 0.9915 - val_loss: 0.0297 - val_accuracy: 0.9912
Epoch 5/5
5048/5048 [==============================] - 67s 13ms/step - loss: 0.0255 - accuracy: 0.9917 - val_loss: 0.0329 - val_accuracy: 0.9902
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 2ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[1. 0. 1. ... 0. 0. 0.]
Epoch 1/5
5048/5048 [==============================] - 67s 13ms/step - loss: 0.0409 - accuracy: 0.9869 - val_loss: 0.0285 - val_accuracy: 0.9910
Epoch 2/5
5048/5048 [==============================] - 67s 13ms/step - loss: 0.0326 - accuracy: 0.9897 - val_loss: 0.0323 - val_accuracy: 0.9901
Epoch 3/5
5048/5048 [==============================] - 67s 13ms/step - loss: 0.0287 - accuracy: 0.9909 - val_loss: 0.0316 - val_accuracy: 0.9905
Epoch 4/5
5048/5048 [==============================] - 67s 13ms/step - loss: 0.0258 - accuracy: 0.9918 - val_loss: 0.0311 - val_accuracy: 0.9907
Epoch 5/5
5048/5048 [==============================] - 67s 13ms/step - loss: 0.0245 - accuracy: 0.9923 - val_loss: 0.0338 - val_accuracy: 0.9900
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 1. 1. ... 1. 0. 1.]
Epoch 1/5
5048/5048 [==============================] - 69s 14ms/step - loss: 0.0395 - accuracy: 0.9875 - val_loss: 0.0247 - val_accuracy: 0.9925
Epoch 2/5
5048/5048 [==============================] - 67s 13ms/step - loss: 0.0321 - accuracy: 0.9897 - val_loss: 0.0276 - val_accuracy: 0.9913
Epoch 3/5
5048/5048 [==============================] - 67s 13ms/step - loss: 0.0283 - accuracy: 0.9908 - val_loss: 0.0285 - val_accuracy: 0.9914
Epoch 4/5
5048/5048 [==============================] - 67s 13ms/step - loss: 0.0256 - accuracy: 0.9918 - val_loss: 0.0284 - val_accuracy: 0.9917
Epoch 5/5
5048/5048 [==============================] - 67s 13ms/step - loss: 0.0236 - accuracy: 0.9925 - val_loss: 0.0292 - val_accuracy: 0.9917
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[1. 1. 1. ... 0. 0. 1.]
Epoch 1/5
5048/5048 [==============================] - 67s 13ms/step - loss: 0.0388 - accuracy: 0.9875 - val_loss: 0.0267 - val_accuracy: 0.9920
Epoch 2/5
5048/5048 [==============================] - 67s 13ms/step - loss: 0.0304 - accuracy: 0.9903 - val_loss: 0.0271 - val_accuracy: 0.9921
Epoch 3/5
5048/5048 [==============================] - 67s 13ms/step - loss: 0.0277 - accuracy: 0.9911 - val_loss: 0.0279 - val_accuracy: 0.9916
Epoch 4/5
5048/5048 [==============================] - 67s 13ms/step - loss: 0.0258 - accuracy: 0.9916 - val_loss: 0.0280 - val_accuracy: 0.9920
Epoch 5/5
5048/5048 [==============================] - 67s 13ms/step - loss: 0.0230 - accuracy: 0.9926 - val_loss: 0.0318 - val_accuracy: 0.9907
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 2ms/step
67/67 [==============================] - 0s 2ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 1. 0. ... 0. 1. 0.]
Epoch 1/5
5048/5048 [==============================] - 68s 13ms/step - loss: 0.0391 - accuracy: 0.9877 - val_loss: 0.0267 - val_accuracy: 0.9919
Epoch 2/5
5048/5048 [==============================] - 67s 13ms/step - loss: 0.0310 - accuracy: 0.9902 - val_loss: 0.0260 - val_accuracy: 0.9920
Epoch 3/5
5048/5048 [==============================] - 67s 13ms/step - loss: 0.0283 - accuracy: 0.9910 - val_loss: 0.0260 - val_accuracy: 0.9924
Epoch 4/5
5048/5048 [==============================] - 68s 13ms/step - loss: 0.0250 - accuracy: 0.9920 - val_loss: 0.0288 - val_accuracy: 0.9915
Epoch 5/5
5048/5048 [==============================] - 66s 13ms/step - loss: 0.0229 - accuracy: 0.9927 - val_loss: 0.0287 - val_accuracy: 0.9917
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 1. 1. ... 0. 0. 1.]
Epoch 1/5
5048/5048 [==============================] - 66s 13ms/step - loss: 0.0384 - accuracy: 0.9876 - val_loss: 0.0259 - val_accuracy: 0.9923
Epoch 2/5
5048/5048 [==============================] - 66s 13ms/step - loss: 0.0310 - accuracy: 0.9899 - val_loss: 0.0267 - val_accuracy: 0.9920
Epoch 3/5
5048/5048 [==============================] - 66s 13ms/step - loss: 0.0268 - accuracy: 0.9912 - val_loss: 0.0322 - val_accuracy: 0.9901
Epoch 4/5
5048/5048 [==============================] - 66s 13ms/step - loss: 0.0252 - accuracy: 0.9921 - val_loss: 0.0316 - val_accuracy: 0.9909
Epoch 5/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0229 - accuracy: 0.9926 - val_loss: 0.0316 - val_accuracy: 0.9909
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 1. 1. ... 0. 1. 1.]
Epoch 1/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0375 - accuracy: 0.9882 - val_loss: 0.0247 - val_accuracy: 0.9927
Epoch 2/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0304 - accuracy: 0.9903 - val_loss: 0.0252 - val_accuracy: 0.9925
Epoch 3/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0271 - accuracy: 0.9914 - val_loss: 0.0265 - val_accuracy: 0.9920
Epoch 4/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0243 - accuracy: 0.9922 - val_loss: 0.0280 - val_accuracy: 0.9917
Epoch 5/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0227 - accuracy: 0.9926 - val_loss: 0.0275 - val_accuracy: 0.9918
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 2ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 2ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 0. 0. ... 1. 1. 1.]
Epoch 1/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0373 - accuracy: 0.9881 - val_loss: 0.0244 - val_accuracy: 0.9924
Epoch 2/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0298 - accuracy: 0.9905 - val_loss: 0.0266 - val_accuracy: 0.9919
Epoch 3/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0269 - accuracy: 0.9912 - val_loss: 0.0259 - val_accuracy: 0.9924
Epoch 4/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0241 - accuracy: 0.9925 - val_loss: 0.0289 - val_accuracy: 0.9918
Epoch 5/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0225 - accuracy: 0.9928 - val_loss: 0.0289 - val_accuracy: 0.9912
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 2ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 1. 1. ... 0. 0. 0.]
Epoch 1/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0363 - accuracy: 0.9885 - val_loss: 0.0272 - val_accuracy: 0.9915
Epoch 2/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0292 - accuracy: 0.9907 - val_loss: 0.0256 - val_accuracy: 0.9923
Epoch 3/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0258 - accuracy: 0.9918 - val_loss: 0.0286 - val_accuracy: 0.9913
Epoch 4/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0234 - accuracy: 0.9923 - val_loss: 0.0291 - val_accuracy: 0.9913
Epoch 5/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0220 - accuracy: 0.9930 - val_loss: 0.0292 - val_accuracy: 0.9914
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[1. 0. 1. ... 0. 1. 0.]
Epoch 1/5
5048/5048 [==============================] - 66s 13ms/step - loss: 0.0366 - accuracy: 0.9883 - val_loss: 0.0248 - val_accuracy: 0.9929
Epoch 2/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0293 - accuracy: 0.9907 - val_loss: 0.0280 - val_accuracy: 0.9914
Epoch 3/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0261 - accuracy: 0.9915 - val_loss: 0.0254 - val_accuracy: 0.9926
Epoch 4/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0234 - accuracy: 0.9926 - val_loss: 0.0269 - val_accuracy: 0.9923
Epoch 5/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0221 - accuracy: 0.9930 - val_loss: 0.0268 - val_accuracy: 0.9923
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 0. 0. ... 0. 1. 0.]
Epoch 1/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0359 - accuracy: 0.9888 - val_loss: 0.0281 - val_accuracy: 0.9912
Epoch 2/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0286 - accuracy: 0.9911 - val_loss: 0.0310 - val_accuracy: 0.9904
Epoch 3/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0253 - accuracy: 0.9920 - val_loss: 0.0291 - val_accuracy: 0.9914
Epoch 4/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0235 - accuracy: 0.9925 - val_loss: 0.0281 - val_accuracy: 0.9921
Epoch 5/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0215 - accuracy: 0.9932 - val_loss: 0.0288 - val_accuracy: 0.9914
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 1. 1. ... 0. 0. 1.]
Epoch 1/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0351 - accuracy: 0.9888 - val_loss: 0.0245 - val_accuracy: 0.9925
Epoch 2/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0282 - accuracy: 0.9912 - val_loss: 0.0241 - val_accuracy: 0.9928
Epoch 3/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0253 - accuracy: 0.9921 - val_loss: 0.0245 - val_accuracy: 0.9924
Epoch 4/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0235 - accuracy: 0.9925 - val_loss: 0.0255 - val_accuracy: 0.9926
Epoch 5/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0216 - accuracy: 0.9931 - val_loss: 0.0251 - val_accuracy: 0.9928
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 2ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[1. 0. 0. ... 1. 0. 0.]
Epoch 1/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0352 - accuracy: 0.9891 - val_loss: 0.0241 - val_accuracy: 0.9928
Epoch 2/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0285 - accuracy: 0.9910 - val_loss: 0.0255 - val_accuracy: 0.9926
Epoch 3/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0253 - accuracy: 0.9920 - val_loss: 0.0260 - val_accuracy: 0.9921
Epoch 4/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0229 - accuracy: 0.9927 - val_loss: 0.0264 - val_accuracy: 0.9924
Epoch 5/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0211 - accuracy: 0.9932 - val_loss: 0.0277 - val_accuracy: 0.9926
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[1. 1. 1. ... 0. 0. 0.]
Epoch 1/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0346 - accuracy: 0.9892 - val_loss: 0.0210 - val_accuracy: 0.9941
Epoch 2/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0284 - accuracy: 0.9913 - val_loss: 0.0249 - val_accuracy: 0.9925
Epoch 3/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0245 - accuracy: 0.9923 - val_loss: 0.0254 - val_accuracy: 0.9924
Epoch 4/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0225 - accuracy: 0.9930 - val_loss: 0.0271 - val_accuracy: 0.9921
Epoch 5/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0213 - accuracy: 0.9934 - val_loss: 0.0270 - val_accuracy: 0.9922
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 0. 1. ... 1. 1. 1.]
Epoch 1/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0337 - accuracy: 0.9895 - val_loss: 0.0225 - val_accuracy: 0.9932
Epoch 2/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0274 - accuracy: 0.9916 - val_loss: 0.0243 - val_accuracy: 0.9927
Epoch 3/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0245 - accuracy: 0.9922 - val_loss: 0.0220 - val_accuracy: 0.9937
Epoch 4/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0228 - accuracy: 0.9927 - val_loss: 0.0241 - val_accuracy: 0.9929
Epoch 5/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0208 - accuracy: 0.9935 - val_loss: 0.0254 - val_accuracy: 0.9924
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 2ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 2ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 1. 1. ... 0. 1. 1.]
Epoch 1/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0342 - accuracy: 0.9892 - val_loss: 0.0219 - val_accuracy: 0.9937
Epoch 2/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0278 - accuracy: 0.9913 - val_loss: 0.0220 - val_accuracy: 0.9936
Epoch 3/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0246 - accuracy: 0.9924 - val_loss: 0.0244 - val_accuracy: 0.9932
Epoch 4/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0224 - accuracy: 0.9929 - val_loss: 0.0234 - val_accuracy: 0.9936
Epoch 5/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0209 - accuracy: 0.9934 - val_loss: 0.0238 - val_accuracy: 0.9934
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[1. 1. 1. ... 1. 0. 0.]
Epoch 1/5
5048/5048 [==============================] - 63s 12ms/step - loss: 0.0328 - accuracy: 0.9896 - val_loss: 0.0208 - val_accuracy: 0.9937
Epoch 2/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0263 - accuracy: 0.9916 - val_loss: 0.0211 - val_accuracy: 0.9937
Epoch 3/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0244 - accuracy: 0.9924 - val_loss: 0.0221 - val_accuracy: 0.9935
Epoch 4/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0211 - accuracy: 0.9933 - val_loss: 0.0243 - val_accuracy: 0.9932
Epoch 5/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0198 - accuracy: 0.9936 - val_loss: 0.0230 - val_accuracy: 0.9942
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 2ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 2ms/step
67/67 [==============================] - 0s 2ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 0. 0. ... 1. 0. 1.]
Epoch 1/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0333 - accuracy: 0.9896 - val_loss: 0.0217 - val_accuracy: 0.9936
Epoch 2/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0267 - accuracy: 0.9917 - val_loss: 0.0221 - val_accuracy: 0.9935
Epoch 3/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0236 - accuracy: 0.9925 - val_loss: 0.0243 - val_accuracy: 0.9929
Epoch 4/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0221 - accuracy: 0.9931 - val_loss: 0.0235 - val_accuracy: 0.9934
Epoch 5/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0197 - accuracy: 0.9937 - val_loss: 0.0255 - val_accuracy: 0.9928
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 2ms/step
267/267 [==============================] - 1s 2ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 2ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 1. 1. ... 1. 0. 0.]
Epoch 1/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0329 - accuracy: 0.9897 - val_loss: 0.0230 - val_accuracy: 0.9929
Epoch 2/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0264 - accuracy: 0.9917 - val_loss: 0.0206 - val_accuracy: 0.9941
Epoch 3/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0241 - accuracy: 0.9926 - val_loss: 0.0225 - val_accuracy: 0.9933
Epoch 4/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0215 - accuracy: 0.9932 - val_loss: 0.0238 - val_accuracy: 0.9928
Epoch 5/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0203 - accuracy: 0.9939 - val_loss: 0.0253 - val_accuracy: 0.9927
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 2ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 2ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 1. 1. ... 0. 0. 0.]
Epoch 1/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0324 - accuracy: 0.9900 - val_loss: 0.0201 - val_accuracy: 0.9942
Epoch 2/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0257 - accuracy: 0.9920 - val_loss: 0.0226 - val_accuracy: 0.9934
Epoch 3/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0235 - accuracy: 0.9926 - val_loss: 0.0219 - val_accuracy: 0.9936
Epoch 4/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0216 - accuracy: 0.9931 - val_loss: 0.0225 - val_accuracy: 0.9934
Epoch 5/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0196 - accuracy: 0.9940 - val_loss: 0.0280 - val_accuracy: 0.9914
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 0. 0. ... 0. 1. 1.]
Epoch 1/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0319 - accuracy: 0.9901 - val_loss: 0.0193 - val_accuracy: 0.9943
Epoch 2/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0260 - accuracy: 0.9921 - val_loss: 0.0214 - val_accuracy: 0.9940
Epoch 3/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0233 - accuracy: 0.9928 - val_loss: 0.0212 - val_accuracy: 0.9942
Epoch 4/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0214 - accuracy: 0.9936 - val_loss: 0.0261 - val_accuracy: 0.9924
Epoch 5/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0197 - accuracy: 0.9937 - val_loss: 0.0249 - val_accuracy: 0.9925
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 0. 1. ... 0. 1. 0.]
Epoch 1/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0315 - accuracy: 0.9902 - val_loss: 0.0203 - val_accuracy: 0.9940
Epoch 2/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0259 - accuracy: 0.9920 - val_loss: 0.0187 - val_accuracy: 0.9944
Epoch 3/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0235 - accuracy: 0.9926 - val_loss: 0.0223 - val_accuracy: 0.9929
Epoch 4/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0208 - accuracy: 0.9936 - val_loss: 0.0215 - val_accuracy: 0.9939
Epoch 5/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0191 - accuracy: 0.9941 - val_loss: 0.0230 - val_accuracy: 0.9932
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 0. 0. ... 0. 1. 1.]
Epoch 1/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0327 - accuracy: 0.9898 - val_loss: 0.0186 - val_accuracy: 0.9945
Epoch 2/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0258 - accuracy: 0.9920 - val_loss: 0.0202 - val_accuracy: 0.9941
Epoch 3/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0234 - accuracy: 0.9927 - val_loss: 0.0203 - val_accuracy: 0.9939
Epoch 4/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0215 - accuracy: 0.9933 - val_loss: 0.0203 - val_accuracy: 0.9942
Epoch 5/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0198 - accuracy: 0.9938 - val_loss: 0.0235 - val_accuracy: 0.9933
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 0. 1. ... 0. 0. 1.]
Epoch 1/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0313 - accuracy: 0.9903 - val_loss: 0.0222 - val_accuracy: 0.9939
Epoch 2/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0258 - accuracy: 0.9919 - val_loss: 0.0196 - val_accuracy: 0.9944
Epoch 3/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0225 - accuracy: 0.9930 - val_loss: 0.0216 - val_accuracy: 0.9938
Epoch 4/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0211 - accuracy: 0.9935 - val_loss: 0.0218 - val_accuracy: 0.9941
Epoch 5/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0196 - accuracy: 0.9939 - val_loss: 0.0243 - val_accuracy: 0.9932
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 2ms/step
267/267 [==============================] - 1s 2ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 2ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 0. 0. ... 1. 0. 0.]
Epoch 1/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0307 - accuracy: 0.9906 - val_loss: 0.0194 - val_accuracy: 0.9944
Epoch 2/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0254 - accuracy: 0.9921 - val_loss: 0.0201 - val_accuracy: 0.9939
Epoch 3/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0225 - accuracy: 0.9933 - val_loss: 0.0189 - val_accuracy: 0.9949
Epoch 4/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0210 - accuracy: 0.9934 - val_loss: 0.0205 - val_accuracy: 0.9941
Epoch 5/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0191 - accuracy: 0.9940 - val_loss: 0.0240 - val_accuracy: 0.9936
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 2ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 1. 1. ... 1. 1. 1.]
Epoch 1/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0313 - accuracy: 0.9904 - val_loss: 0.0213 - val_accuracy: 0.9937
Epoch 2/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0255 - accuracy: 0.9920 - val_loss: 0.0207 - val_accuracy: 0.9940
Epoch 3/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0223 - accuracy: 0.9930 - val_loss: 0.0217 - val_accuracy: 0.9935
Epoch 4/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0206 - accuracy: 0.9936 - val_loss: 0.0209 - val_accuracy: 0.9941
Epoch 5/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0198 - accuracy: 0.9939 - val_loss: 0.0235 - val_accuracy: 0.9935
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 2ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 2ms/step
67/67 [==============================] - 0s 2ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 1. 1. ... 1. 1. 1.]
Epoch 1/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0306 - accuracy: 0.9906 - val_loss: 0.0208 - val_accuracy: 0.9938
Epoch 2/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0250 - accuracy: 0.9923 - val_loss: 0.0193 - val_accuracy: 0.9943
Epoch 3/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0220 - accuracy: 0.9930 - val_loss: 0.0213 - val_accuracy: 0.9937
Epoch 4/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0209 - accuracy: 0.9935 - val_loss: 0.0214 - val_accuracy: 0.9938
Epoch 5/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0190 - accuracy: 0.9939 - val_loss: 0.0258 - val_accuracy: 0.9926
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 2ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 0. 0. ... 1. 1. 1.]
Epoch 1/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0298 - accuracy: 0.9908 - val_loss: 0.0218 - val_accuracy: 0.9936
Epoch 2/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0251 - accuracy: 0.9924 - val_loss: 0.0212 - val_accuracy: 0.9937
Epoch 3/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0217 - accuracy: 0.9933 - val_loss: 0.0200 - val_accuracy: 0.9944
Epoch 4/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0206 - accuracy: 0.9936 - val_loss: 0.0218 - val_accuracy: 0.9937
Epoch 5/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0186 - accuracy: 0.9942 - val_loss: 0.0244 - val_accuracy: 0.9932
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 1. 0. ... 1. 1. 0.]
Epoch 1/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0298 - accuracy: 0.9907 - val_loss: 0.0206 - val_accuracy: 0.9941
Epoch 2/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0243 - accuracy: 0.9925 - val_loss: 0.0228 - val_accuracy: 0.9933
Epoch 3/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0220 - accuracy: 0.9931 - val_loss: 0.0226 - val_accuracy: 0.9939
Epoch 4/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0196 - accuracy: 0.9939 - val_loss: 0.0240 - val_accuracy: 0.9936
Epoch 5/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0185 - accuracy: 0.9943 - val_loss: 0.0237 - val_accuracy: 0.9935
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 0. 1. ... 1. 1. 1.]
Epoch 1/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0301 - accuracy: 0.9907 - val_loss: 0.0172 - val_accuracy: 0.9947
Epoch 2/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0243 - accuracy: 0.9924 - val_loss: 0.0175 - val_accuracy: 0.9949
Epoch 3/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0217 - accuracy: 0.9933 - val_loss: 0.0195 - val_accuracy: 0.9945
Epoch 4/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0200 - accuracy: 0.9939 - val_loss: 0.0221 - val_accuracy: 0.9937
Epoch 5/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0188 - accuracy: 0.9941 - val_loss: 0.0228 - val_accuracy: 0.9936
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 1. 0. ... 1. 0. 0.]
Epoch 1/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0294 - accuracy: 0.9910 - val_loss: 0.0175 - val_accuracy: 0.9949
Epoch 2/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0248 - accuracy: 0.9924 - val_loss: 0.0184 - val_accuracy: 0.9948
Epoch 3/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0216 - accuracy: 0.9932 - val_loss: 0.0186 - val_accuracy: 0.9947
Epoch 4/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0202 - accuracy: 0.9937 - val_loss: 0.0189 - val_accuracy: 0.9945
Epoch 5/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0184 - accuracy: 0.9943 - val_loss: 0.0185 - val_accuracy: 0.9949
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 2ms/step
267/267 [==============================] - 1s 2ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 2ms/step
67/67 [==============================] - 0s 2ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[1. 0. 1. ... 0. 0. 1.]
Epoch 1/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0290 - accuracy: 0.9912 - val_loss: 0.0193 - val_accuracy: 0.9945
Epoch 2/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0239 - accuracy: 0.9927 - val_loss: 0.0230 - val_accuracy: 0.9933
Epoch 3/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0212 - accuracy: 0.9936 - val_loss: 0.0196 - val_accuracy: 0.9946
Epoch 4/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0203 - accuracy: 0.9938 - val_loss: 0.0206 - val_accuracy: 0.9941
Epoch 5/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0181 - accuracy: 0.9944 - val_loss: 0.0226 - val_accuracy: 0.9936
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 3ms/step
Cantidad de pares similares:  201798
Cantidad total de pares:  403798
[0. 1. 1. ... 0. 0. 0.]
Epoch 1/5
5048/5048 [==============================] - 65s 13ms/step - loss: 0.0289 - accuracy: 0.9912 - val_loss: 0.0192 - val_accuracy: 0.9942
Epoch 2/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0244 - accuracy: 0.9924 - val_loss: 0.0210 - val_accuracy: 0.9937
Epoch 3/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0218 - accuracy: 0.9933 - val_loss: 0.0217 - val_accuracy: 0.9935
Epoch 4/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0193 - accuracy: 0.9940 - val_loss: 0.0187 - val_accuracy: 0.9947
Epoch 5/5
5048/5048 [==============================] - 64s 13ms/step - loss: 0.0185 - accuracy: 0.9942 - val_loss: 0.0213 - val_accuracy: 0.9942
<Figure size 640x480 with 1 Axes><Figure size 640x480 with 1 Axes>Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
Obteniendo resultados para las consultas de nuevas marcas ...
Cantidad de consultas: 10
67/67 [==============================] - 0s 3ms/step
67/67 [==============================] - 0s 3ms/step
```


**[Celda 18 - Código]**
```python
model_siamese = tf.keras.models.load_model(DIRECTORIO+r'\Entrenamientos\NM_OS_BD91350R_5E_400kParesSD_adam0.001\16_Entreno_0.54_0.71_0.73.h5')
```


**[Celda 19 - Código]**
```python
import pandas as pd
hist_df = pd.DataFrame(history.history) 
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
    
print(hist_csv_file)
```


*Salida:*
```text
history.csv
```

# Evaluar resultados
###### Código necesario para cargar evaluar resultados


**[Celda 21 - Código]**
```python
#Crear arrays para las consultas
def crear_arrays_consultas(consultas,imagenesBD):
    array_consultas = []
    array_imagenes_db = []   
    for consulta in consultas:      
        for imagen_db in imagenesBD:
            array_consultas.append(consulta.reshape(28,28,1)) 
            array_imagenes_db.append(imagen_db.reshape(28,28,1))
    return array_consultas, array_imagenes_db
```


**[Celda 22 - Código]**
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


**[Celda 23 - Código]**
```python
# CONSULTAS POR SIMILITUD UTILIZANDO LA BASE DE CONSULTAS DIBUJADAS - OPTIMIZADO
def obtener_resultados(model_siamese,consultar_por,imprimir_imagenes_resultado = False, imprimir_resultados = False):    
    
    if(consultar_por == "Dibujadas"):
        consultas = consultas_dibujadas
        y_consultas = y_consultas_dibujadas
        print("Obteniendo resultados para las consultas dibujadas ...")
    else:
        consultas = consultas_nuevas_marcas
        y_consultas = y_consultas_nuevas_marcas
        print("Obteniendo resultados para las consultas de nuevas marcas ...")
    array_consultas,array_imagenes_db = crear_arrays_consultas(consultas,imagenesBD)
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
        if(imprimir_resultados):
            print("Imagen:" , y_consultas[i][y_consultas[i].find('_')+1:])
            print("Resultados:", y_imagenes_resultado)
            print("Distancias:", (['{:0.5e}'.format(x[0]) for x in distancias]))       
            if(imprimir_imagenes_resultado):
                imprimir_imagenes([consultas[i]]+imagenes,[y_consultas[i]]+y_imagenes_resultado)
    
    if(imprimir_resultados):
        print()
        print("Porcentaje de aciertos primera posicion:", contador_aciertos_1/cantidad_consultas , "  Cantidad aciertos: " + str(contador_aciertos_1)+"/"+str(cantidad_consultas))
        print("Porcentaje de aciertos tercera posicion:", contador_aciertos_3/cantidad_consultas , "  Cantidad aciertos: " + str(contador_aciertos_3)+"/"+str(cantidad_consultas) )
        print("Porcentaje de aciertos quinta  posicion:", contador_aciertos_5/cantidad_consultas , "  Cantidad aciertos: " + str(contador_aciertos_5)+"/"+str(cantidad_consultas) )
    return(round(contador_aciertos_1/cantidad_consultas,2),round(contador_aciertos_3/cantidad_consultas,2),round(contador_aciertos_5/cantidad_consultas,2))
```


**[Celda 24 - Código]**
```python

primera,tercera,quinta = obtener_resultados(model_siamese,"Dibujadas",imprimir_imagenes_resultado = False,imprimir_resultados = False)
print(primera,tercera,quinta)
```


*Salida:*
```text
Obteniendo resultados para las consultas dibujadas ...
Cantidad de consultas: 40
267/267 [==============================] - 1s 3ms/step
267/267 [==============================] - 1s 3ms/step
0.8 0.9 0.97
```


**[Celda 25 - Código]**
```python
#Testeo preprocesamiento (testeamos si el procesamiento sobre las imagenes de entrenamiento es el mismo con el que vamos a cargar las consultas)
imagenes_a_testear = [dic[10][0],imagenesBD[1]]
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


**[Celda 26 - Código]**
```python
from itertools import combinations
def imprimir_estadisticas(resultados, completas):
    print("Resultado máximo:", max(resultados))
    print("Resultado mínimo:", min(resultados))
    print("Promedio de resultados:", sum(resultados)/len(resultados))
    if completas:
        print("Varianza:", np.var(resultados))
        print("Desviacion estandar:", np.std(resultados))
        print("Promedio al cuadrado sobre varianza",((sum(resultados)/len(resultados))**2)/np.var(resultados))

def obtener_tuplas_sin_repetir(lista,cant):
    tuplas = list(combinations(lista,cant))
    return tuplas

    
def predecir_por_partes(array_x,array_y,partes):
    resultados = []
    tqdm_notebook.pandas() 
    for i in tqdm_notebook(range(partes), desc="Realizando predicciones..."):
        K.clear_session()
        tf.compat.v1.reset_default_graph()       
        tam_bloque = math.floor(len(array_x)/partes)
        if (i != partes-1):
            parcial_1 = model_siamese.predict([array_x[i*tam_bloque:i*tam_bloque+tam_bloque],array_y[i*tam_bloque:i*tam_bloque+tam_bloque]],verbose = False)
            parcial_2 = model_siamese.predict([array_y[i*tam_bloque:i*tam_bloque+tam_bloque],array_x[i*tam_bloque:i*tam_bloque+tam_bloque]],verbose = False)
          
        else: 
            parcial_1 = model_siamese.predict([array_x[i*tam_bloque:len(array_x)],array_y[i*tam_bloque:len(array_x)]],verbose = False)
            parcial_2 = model_siamese.predict([array_y[i*tam_bloque:len(array_x)],array_x[i*tam_bloque:len(array_x)]],verbose = False)          
        resultados.extend(parcial_1+parcial_2)
    return resultados   
         
```


**[Celda 27 - Código]**
```python
#Reflexividad
np_array_imagenesBD = np.array(imagenesBD)
resultados = model_siamese.predict([np_array_imagenesBD,np_array_imagenesBD])
imprimir_estadisticas(resultados,False)
```


*Salida:*
```text
7/7 [==============================] - 1s 20ms/step
Resultado máximo: [0.02997103]
Resultado mínimo: [3.0513687e-09]
Promedio de resultados: [0.00131601]
```


**[Celda 28 - Código]**
```python
print(resultados.shape)
```


*Salida:*
```text
(213, 1)
```


**[Celda 29 - Código]**
```python
#Desigualdad triangular
triplas = obtener_tuplas_sin_repetir(imagenesBD,3)
array_x = np.array([t[0] for t in triplas])
array_y = np.array([t[1] for t in triplas])
array_z = np.array([t[2] for t in triplas])
resultados_x_y = predecir_por_partes(array_x,array_y,300)
resultados_y_z = predecir_por_partes(array_y,array_x,300)
resultados_x_z = predecir_por_partes(array_x,array_z,300)
contador = 0
cant_triplas = len(resultados_x_y)
contador_fallos = 0
for i in range(cant_triplas):
    if ((resultados_x_y[i] + resultados_y_z[i]) >= resultados_x_z[i]):
        contador += 1
    else:
        contador_fallos += 1
print("% de triplas que cumple la desigualdad triangular: ", contador/cant_triplas)
print("Cantidad de fallos", contador_fallos)
```


*Salida:*
```text
Realizando predicciones...:   0%|          | 0/300 [00:00<?, ?it/s]Realizando predicciones...:   0%|          | 0/300 [00:00<?, ?it/s]Realizando predicciones...:   0%|          | 0/300 [00:00<?, ?it/s]% de triplas que cumple la desigualdad triangular:  0.9219155584495078
Cantidad de fallos 123997
```


**[Celda 30 - Código]**
```python
#Promedio distancias
pares = obtener_tuplas_sin_repetir(imagenesBD,2)
array_1 = np.array([t[0] for t in pares])
array_2 = np.array([t[1] for t in pares])
resultados = predecir_por_partes(array_1,array_2,3)
imprimir_estadisticas(resultados,True)
```


*Salida:*
```text
Realizando predicciones...:   0%|          | 0/3 [00:00<?, ?it/s]Resultado máximo: [2.]
Resultado mínimo: [2.3059965e-06]
Promedio de resultados: [1.8787409]
Varianza: 0.14999475
Desviacion estandar: 0.38729155
Promedio al cuadrado sobre varianza [23.53194]
```


**[Celda 31 - Código]**
```python
print(len(pares))
```


*Salida:*
```text
22578
```


**[Celda 32 - Código]**
```python
imprimir_imagenes([imagenesBD[0],imagenesBD[123]],[y_imagenesBD[0],y_imagenesBD[123]])
print(y_imagenesBD[0],y_imagenesBD[123])
```


*Salida:*
```text
<Figure size 640x480 with 2 Axes>1 27
```


**[Celda 33 - Código]**
```python

```
