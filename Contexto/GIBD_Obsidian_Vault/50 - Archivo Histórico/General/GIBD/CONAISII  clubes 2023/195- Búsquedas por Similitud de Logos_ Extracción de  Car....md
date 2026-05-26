---
aliases: [195- Búsquedas por Similitud de Logos_ Extracción de  Car...]
tags:
  - grupo/general
  - estado/archivo
  - tipo/documento
formato_original: pdf
fecha_modificacion: 2023-11-02
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/CONAISII  clubes 2023/195- Búsquedas por Similitud de Logos_ Extracción de  Características usando IA en Escenarios de  Datos Escasos.pdf"
tamanio_bytes: 2086395
---

# 195- Búsquedas por Similitud de Logos_ Extracción de  Características usando IA en Escenarios de  Datos Escasos

Ruta interna: `GIBD/CONAISII  clubes 2023/195- Búsquedas por Similitud de Logos_ Extracción de  Características usando IA en Escenarios de  Datos Escasos.pdf`

---

195 - Búsquedas por Similitud de Logos: 
Extracción de Características usando IA en 
Escenarios de Datos Escasos
Andres J. Pascal, Adrián N. Planas, Zoe Florencia Vidal, 
Iván Bonti, Lucas T onelotto, Agustina Bonti
bontii@frcu.utn.edu.ar
vidalz@frcu.utn.edu.ar San Miguel de Tucumán, Tucumán, Argentina – 2 y 3 de Noviembre 2023

01
CONTENIDO
02 03
04 05 06
INTRODUCCIÓN
BÚSQUEDAS 
POR SIMILITUD
EXTRACCIÓN DE 
CARACTERÍSTICAS
BASE DE DATOS
MODELOS Y 
EXPERIMENTOS
CONCLUSIONES
Y TRABAJO
FUTURO
Logos de Clubes 
INTRODUCCIÓN
01

02
DB 
tradicional
Números
Fechas
Cadenas
Objetos no 
Estructurados
% de 
Similitud
BÚSQUEDAS POR 
SIMILITUD
Preprocesamiento
Características
Color
Forma T extura
Función de 
Distancia 
Similitud
02
BÚSQUEDAS POR 
SIMILITUD
Recuperación de imágenes basada en contenido (CBIR)
Función de 
distancia
Consulta (q)
Índice
02
BÚSQUEDAS POR 
SIMILITUD
Función de distancia
Espacios Métricos
El modelo (U, d), donde:
U : universo de objetos
d : función de distancia 
Objetivo de la consulta:
Dado q ∈ U, recuperar los elementos de U similares a q.
Base de datos X será un subconjunto finito de U:  
X   U, |X| = n⊆
La función d es una métrica, es decir: d: U x U → R+  
1. Positividad  
2. Simetría 
3. Reflexividad
4. Desigualdad Triangular
02
BÚSQUEDAS POR 
SIMILITUD

CNN – Redes Neuronales Profundas Convolucionales
EXTRACCIÓN DE 
CARACTERÍSTICAS
03

Origen y evolución de 
las CNN
Arquitecturas 
actuales
Desafíos
Soluciones
DenseNet
ResNet
EXTRACCIÓN DE 
CARACTERÍSTICAS
03
CNN – Redes Neuronales Profundas Convolucionales
Inception
Datos Escasos
Generalización
Origen y evolución de 
las CNN
Arquitecturas 
actuales
Desafíos
Soluciones
03
CNN – Redes Neuronales Profundas Convolucionales
EXTRACCIÓN DE 
CARACTERÍSTICAS
Origen y evolución de las CNN
Arquitecturas 
actuales
Desafíos
Soluciones
EXTRACCIÓN DE 
CARACTERÍSTICAS
03
CNN – Redes Neuronales Profundas Convolucionales
Transfer Learning/Fine Tuning
Augmentation
Siamese Networks
Logos de Clubes
04
BASE DE DATOS
La base de datos se encuentra compuesta por 
un total de 150 imágenes de distintos clubes, 
diferentes ligas y países. 
Características de las imágenes:
● Resolución de 100 x 100 píxeles.
● Formato de imagen PNG.
● Profundidad de color de 32 bits.
Solución: Aumentación de Datos (estándar en este caso)
Desafío: One-Shot Learning
04
BASE DE DATOS
Rotación Escalado  Corte 
 Ruido
Además, modificación de Brillo, Contraste y otros
Redes 
Siamesas
05
MODELOS Y 
EXPERIMENTOS
Arquitecturas Utilizadas
Modelo interno utilizado
● 5 capas Conv2D
● 1 capa MaxPooling
● 1 capa GlobalAvgPool
● 32, 64 y 256 kernels, 
todos de 3x3
● padding=same
● 237.856 parámetros
05
Arquitecturas Utilizadas
MODELOS Y 
EXPERIMENTOS
Densenet 121
● 121 Capas totales
● Imágenes de entrada deben estar en 224x224. 
● Arquitectura: cada capa recibe entradas de todas las capas previas y a su vez, 
envía sus características a todas las capas siguientes.
● Alrededor de 8 millones de parámetros entrenables.
 
Resnet-50
● 50 capas
● Imágenes de entrada deben estar en 224x224. 
● Arquitectura: bloques residuales, que introducen conexiones de "skip" o 
"shortcut" (conexiones directas) para saltar una o más capas.
● Alrededor de 25 millones de parámetros entrenables.
Inception V3
● 48 capas 
● Imágenes de entrada deben estar en 299x299. 
● Arquitectura: bloques que incluyen múltiples operaciones de convolución con 
diferentes tamaños de filtros y luego combinan sus salidas.
● Alrededor de 24 millones de parámetros entrenables.
MODELOS Y 
EXPERIMENTOS
05
● Consultas: 60 imágenes de logos de clubes extraídas de 
internet.
● Imágenes en formato PNG en 32 bits de profundidad, pero con 
distintas resoluciones, todas mayores o iguales a 100x100. 
● Las consultas se escalaron al tamaño requerido por cada 
modelo utilizado.
● Búsqueda de los NNk (k vecinos más cercanos) para k=1, 3 y 5.
● Función de Distancia: Euclidiana.
● Métrica empleada: Porcentaje de aciertos.
Experimentos
MODELOS Y 
EXPERIMENTOS
05
Modelos de Extracción de Características: 
● Siamesa20: red siamesa entrenada sobre la BD usando aumentación
● DenseNetU: transfer learning de DenseNet entrenada sobre  ImageNet, 
tomando la última capa antes de las fully connected.
● DenseNetUFT: igual que la anterior pero agregando fine tuning.
● DenseNetM: similar a DenseNetU, pero tomando una capa del medio. 
● InceptionM: transfer learning de Inception entrenada sobre  ImageNet, 
tomando una capa del medio.
● ResNetM: similar a la anterior, pero con ResNet.
Experimentos: modelos
Resultados Obtenidos
MODELOS Y 
EXPERIMENTOS
05
Comparación de 
Porcentajes de Acierto
Búsquedas por 
Similitud

06
CONCLUSIONES
Y TRABAJO
FUTURO
Conclusiones
Se investigaron y compararon diversos métodos, incluyendo el 
uso de Redes Neuronales Siamesas y Transfer Learning  con 
arquitecturas preentrenadas como DenseNet, Inception y ResNet.
Los resultados revelaron que la Red Siamesa, entrenada con 
técnicas de preprocesamiento previo y aumentación de datos, 
alcanza un rendimiento razonable en casos de One-Shot Learning.
El Transfer Learning superó notablemente su rendimiento cuando 
se extrajeron características de las capas centrales de las redes 
preentrenadas. 
Estos hallazgos ofrecen valiosas perspectivas para la 
implementación práctica de sistemas de búsqueda por similitud de 
imágenes en aplicaciones que involucran logos de clubes y objetos 
similares.
06
CONCLUSIONES
Y TRABAJO
FUTURO
Trabajo Futuro
● Realizar nuevamente los experimentos pero con un 
conjunto de datos mayor. Actualmente contamos con más 
de 10.000 logos  sobre los cuales realizaron nuevos 
experimentos.
● Extender estos estudios a otros tipos de imágenes a color.
●  Modificar la Red Siamesa para que utilice Triplet Loss como 
función de pérdida.
●  Analizar estrategias para que la Red generalice mejor, de 
tal manera de que sea robusta ante la incorporación de 
nuevas imágenes.
MUCHAS
GRACIAS!
Preguntas?