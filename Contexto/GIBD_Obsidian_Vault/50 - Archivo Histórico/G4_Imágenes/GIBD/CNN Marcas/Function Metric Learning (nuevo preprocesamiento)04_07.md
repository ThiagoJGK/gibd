---
aliases: [Function Metric Learning (nuevo preprocesamiento)04_07]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2023-07-04
origen_zip: GIBD-20260521T205218Z-3-006.zip
ruta_interna: "GIBD/CNN Marcas/Notebooks/Metric Learning/Function Metric Learning (nuevo preprocesamiento)04_07.ipynb"
tamanio_bytes: 31452
---

# Notebook: Function Metric Learning (nuevo preprocesamiento)04_07.ipynb

Ruta interna: `GIBD/CNN Marcas/Notebooks/Metric Learning/Function Metric Learning (nuevo preprocesamiento)04_07.ipynb`

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


**[Celda 4 - Código]**
```python
print("cantidad de imágenes: " + str(sum))
for key in dic.keys():
    print("Clase: " + str(key) + " Cantidad de imágenes: " + str(len(dic[key]))) 
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

    model.add(Dense(VECTOR_SIZE, activation='softmax'))
    return model
```


**[Celda 10 - Código]**
```python
model.summary()
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
#final_rep = Dense(256, activation='relu')(merged_representation)
#final_rep = Dense(128, activation='relu')(final_rep)
final_rep = merged_representation
output = Dense(1, activation='sigmoid')(final_rep)

# Crea el modelo siamese
model_siamese = tf.keras.Model(inputs=[input_a, input_b], outputs=output)

#Compilamos el modelo
#optimizer=keras.optimizers.Adam(1e-3) - Se puede probar con 1e-4 o 3e-4 o 5e-4
model_siamese.compile(loss='BinaryCrossentropy', optimizer=keras.optimizers.Adam(1e-3), metrics=['accuracy'])
```


**[Celda 12 - Código]**
```python
# SIAMESA

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

optimizer = Adam(0.001, decay=2.5e-4)

#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
model_siamese.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])
```


**[Celda 13 - Código]**
```python
model_siamese = tf.keras.models.load_model(DIRECTORIO+r'\Entrenamientos\10ep_adam3-4_modviejo\2doEntrenamiento78.h5')
```


**[Celda 14 - Código]**
```python
model_siamese.summary()
```


**[Celda 15 - Código]**
```python
# SIAMESA
#for i in range(0,2):
left_input, right_input, targets = obtener_pares(porc_pares_similares = 0.5)   
#mean_squared_error    
model_siamese.fit([left_input,right_input], targets,
          batch_size=64,
          #steps_per_epoch=625,
          epochs=5,
          verbose=1,
          validation_split=0.2)
```


**[Celda 16 - Código]**
```python
#Cuando se encuentre un modelo con un porcentaje mayor al anterior guardar indicando ese porcentaje
model_siamese.save(DIRECTORIO+r'\Entrenamientos\ModeloViejo_BDNueva_5Epocas_ParesDuplicados\7_Entreno_70.h5')
```

# Evaluar resultados
###### Código necesario para cargar evaluar resultados


**[Celda 18 - Código]**
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


**[Celda 19 - Código]**
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


**[Celda 20 - Código]**
```python
# CONSULTAS POR SIMILITUD UTILIZANDO LA BASE DE CONSULTAS DIBUJADAS - OPTIMIZADO

imprimir_imagenes_resultado = False
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

# Preprocesar Imágenes
###### Código necesario para realizar el procesamiento de las imágenes


**[Celda 22 - Código]**
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


**[Celda 23 - Código]**
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


**[Celda 24 - Código]**
```python
#Testeo preprocesamiento (testeamos si el procesamiento sobre las imagenes de entrenamiento es el mismo con el que vamos a cargar las consultas)
imagenes_a_testear = [dic[101][0],imagenesBD[3]]
imprimir_imagenes(imagenes_a_testear,["a","b"])
print(dic[10][0]==imagenesBD[1])
```


**[Celda 25 - Código]**
```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#if len(left_input) < 1:
#left_input, right_input, targets = obtener_pares(porc_pares_similares = 0.5) 
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


**[Celda 26 - Código]**
```python
print(left_input[k] == right_input[k] )
```


**[Celda 27 - Código]**
```python

```


**[Celda 28 - Código]**
```python
#Testeo preprocesamiento (testeamos si el procesamiento sobre las imagenes de entrenamiento es el mismo con el que vamos a cargar las consultas)
imagenes_a_testear = [dic[101][0],imagenesBD[3],consultas[1]]
imprimir_imagenes(imagenes_a_testear,["a","b","c"])
print(dic[100][0]==imagenesBD[2])

```


**[Celda 29 - Código]**
```python

```
