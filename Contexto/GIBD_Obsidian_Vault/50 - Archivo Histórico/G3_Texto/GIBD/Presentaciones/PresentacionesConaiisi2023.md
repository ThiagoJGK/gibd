---
aliases: [PresentacionesConaiisi2023]
tags:
  - grupo/g3_texto
  - estado/archivo
  - tipo/documento
formato_original: pdf
fecha_modificacion: 2023-10-31
origen_zip: GIBD-20260521T205218Z-3-007.zip
ruta_interna: "GIBD/Presentaciones/2023/Expo NOV Tucumán/PresentacionesConaiisi2023.pdf"
tamanio_bytes: 1360287
---

# PresentacionesConaiisi2023

Ruta interna: `GIBD/Presentaciones/2023/Expo NOV Tucumán/PresentacionesConaiisi2023.pdf`

---

Construcción de una Función de 
Distancia para Consultar por Similitud 
Caracteres de Hueso de Oráculo
Andres J. Pascal, Adrián N. Planas, Federico J. 
Stauber, Leon Castiglioni, Martin R. Lopez
martin1597lopez@gmail.com San Miguel de Tucumán, Tucumán, Argentina – 2 y 3 de Noviembre 2023

01
CONTENIDO
02 03
04 05 06
INTRODUCCIÓN BÚSQUEDAS 
POR SIMILITUD
APRENDIZAJE DE 
FUNCIÓN DE 
DISTANCIA
BASE DE 
DATOS
MODELO Y 
EXPERIMENTOS
CONCLUSIONES
Y TRABAJO
FUTURO
INTRODUCCIÓN
01
Expansión de 
datos
Imágenes
Audio
Video
Texto Similitud
Técnicas de 
aprendizaje
Espacios 
métricos
Enfoque 
formalizado

OBC - Caracteres Óseos de oráculo
Edad de 
bronce Dificultades
Análisis 
exhaustivo
Glifos
INTRODUCCIÓN
01

02
DB tradicional
Números
Fechas
Cadenas
Objetos no 
Estructurados
% de Similitud
BÚSQUEDAS POR 
SIMILITUD
DB no estructurada
Preprocesamiento
Características
Color
Forma Textura
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
X ⊆ U, |X| = n
La función d es una métrica, es decir: d: U x U → R+  
1. Positividad  
2. Simetría 
3. Reflexividad
4. Desigualdad Triangular
02
BÚSQUEDAS POR 
SIMILITUD

CNN – Redes Neuronales Profundas Convolucionales
APRENDIZAJE DE 
FUNCIÓN DE 
DISTANCIA
03
Origen y evolución de las CNN
Desafíos
Soluciones
APRENDIZAJE DE 
FUNCIÓN DE 
DISTANCIA
03
CNN – Redes Neuronales Profundas Convolucionales
Arquitecturas 
actuales
Fukushima - 1988
LeCun – 90’s
Avances notables en el 
reconocimiento de 
patrones

Origen y evolución de las CNN
Arquitecturas 
actuales
Desafíos
Soluciones
AlexNet
GoogLeNet
APRENDIZAJE DE 
FUNCIÓN DE 
DISTANCIA
03
CNN – Redes Neuronales Profundas Convolucionales
NiN
VGG Net
DenseNet
Residual Networks

One-Shot Learning o Few-Shots 
Learning
Clases que nunca se entrenaron
Origen y evolución de las CNN
Arquitecturas 
actuales
Desafíos
Soluciones
APRENDIZAJE DE 
FUNCIÓN DE 
DISTANCIA
03
CNN – Redes Neuronales Profundas Convolucionales

Origen y evolución de las CNN
Desafíos
Soluciones
APRENDIZAJE DE 
FUNCIÓN DE 
DISTANCIA
03
CNN – Redes Neuronales Profundas Convolucionales
Metric Learning
Augmentation
Siamese Networks
Arquitecturas 
actuales
APRENDIZAJE DE 
FUNCIÓN DE 
DISTANCIA
03
Redes Siamesas
Desafío: One-Shot Learning
Aumentación Estándar
Rotación - Traslación - Escalado - Corte - Ruido 
Gaussiano
Relieve
Aprendizaje de Funciones de Distancia: CNN y Redes 
Siamesas04
BASE DE DATOS
OBC
MODELO Y 
EXPERIMENTOS
05
Modelo de CNN (ModCar1)
Tres capas convolucionales de 32 
kernels y tres de 64.
Kernels de (3, 3) en cuatro capas y de 
(5, 5) en las dos restantes. 
Dos capas de Dropout. 
Una capa "flatten"
128 neuronas
193.664 parámetros 

MODELO Y 
EXPERIMENTOS
05
QD: 40 imágenes de consulta dibujadas. 
● 20 consultas se dibujaron en papel.
● 20 se dibujaron utilizando una aplicación de diseño gráfico.
QN: Nuevas consultas
● 10 nuevas imágenes incorporadas a la base de datos.
● 10 consultas dibujadas a mano, una por cada nueva marca.
● Sin reentrenamiento.
Experimentos
MODELO Y 
EXPERIMENTOS
05
Resultados
60A1A_q38
60257_1
60A1A 60A1D 600CB 601E4 603CA
602C7 60257 602A4 60269 60253

Resultados
MODELO Y 
EXPERIMENTOS
05
Comparación de Porcentajes de 
Acierto
Comparación de la Precisión

06
CONCLUSIONES
Y TRABAJO
FUTURO
Conclusiones
Se propuso un modelo de Red Siamesa que incluye una CNN, diseñado 
específicamente para abordar estos desafíos, y un mecanismo de 
preprocesamiento y aumento de datos que resultan eficaces en el entrenamiento 
de una función de distancia que determina el grado de similitud entre imágenes 
OBC.
 Los resultados experimentales verificaron que el modelo ModCar1 logró 
tasas de acierto importantes en las búsquedas de los k vecinos más cercanos (NNk), 
alcanzando valores entre el 90% y el 100% para distintos k.
 Estos resultados tienen potenciales aplicaciones en la identificación y 
búsqueda de glifos OBC en contextos de investigación y preservación del 
patrimonio 
06
CONCLUSIONES
Y TRABAJO
FUTURO
Trabajo Futuro
● Realizar experimentos sobre la BD completa, que está compuesta por 
alrededor de 15.000 imágenes de OBCs.
● Explorar estrategias destinadas a mejorar la capacidad de generalización 
de la red, para que su rendimiento sea aún mayor frente a nuevas 
imágenes sin necesidad de un proceso de reentrenamiento.
● Evaluar el rendimiento de índices métricos aproximados para mejorar los 
tiempos de búsqueda.
● Aplicar el modelo propuesto con algunas variaciones para bases de datos 
de imágenes a color
¿Preguntas?