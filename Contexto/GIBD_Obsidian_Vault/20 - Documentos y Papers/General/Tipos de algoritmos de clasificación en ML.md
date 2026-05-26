---
aliases: [Tipos de algoritmos de clasificación en ML]
tags:
  - grupo/general
  - estado/util
  - tipo/documento
formato_original: docx
fecha_modificacion: 2022-12-15
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Tareas Iván/Tipos de algoritmos de clasificación en ML.docx"
tamanio_bytes: 167547
---

# Tipos de algoritmos de clasificación en ML

Ruta interna: `GIBD/Tareas Iván/Tipos de algoritmos de clasificación en ML.docx`

---

Tipos de algoritmos de clasificación en ML
Introducción:
Cada tipo de machine Learning es usado para resolver un problema específico, si tenemos muchos recursos computacionales podemos como ir probando diferentes algoritmos para resolver el mismo problema y entonces así sabremos empíricamente cual es el mas adecuado.
¿Qué es la clasificación?
La clasificación es un proceso de reconocimiento, entendimiento y agrupamiento de ideas y objetos en subpoblaciones o subconjuntos. Usando la pre-categorizado, entrenamiento de conjuntos de datos, el machine learning usa una variedad de algoritmos, para clasificar futuros conjuntos de datos en categorías.
Un ejemplo de esto sería clasificar mails en spam o no-spam.
Además puede usarse por ejemplo para reconocer opiniones de las personas en positiva, negativa o intermedias.
 
tipos De algoritmos de clasificación
Logistic regression (Regresión logística)
La regresión logística sirve básicamente para poder predecir resultados binarios, ósea, aquellos que son si/no, pasa/falla, etc.
La variable independiente puede caer en una o dos categorías, la variable dependiente debe estar siempre categorizada, esta tendrá una probabilidad del 0 o 1, es decir 0 0 100%, será un resultado binario. 
Por ejemplo: Tenemos una foto, y tenemos que decir que objetos están dentro de ella, ósea, árbol, flor, casa, nube, lo que sea, cada objeto tiene una probabilidad de 0 o 100% de estar o no en la foto.
 
Naive bayes
Naive Bayes calcula la posibilidad de si un punto de datos pertenece a una determinada categoría o no. En el análisis de texto, se puede utilizar para categorizar palabras o frases como pertenecientes a una "etiqueta" preestablecida (clasificación) o no. Por ejemplo:.


Decision Tree
Es un algoritmo que necesita supervisión es decir, es supervised learning.
Esto va Armando etiquetas, pudiendo identificar, tanto tronco, ramas como hojas del árbol, entonces esto va tomando decisión para que categoría pertenece, continuando con el ejemplo de los deportes sería algo como esto:

Random forest es una variante del árbol de decisión, o más que variante es una expansión, primero se construyen una multitud de decisiones con el árbol de decisiones para luego armar esto.
Es un modelo que sirve para poder resolver los problemas que tiene el árbol de decisión para por ahí crear categorías innecesariamente.
Support Vector Machines  
Usa algoritmos para entrenar y clasificar data en los grados de polarización, esto es algo que va mucho más allá de la predicción con X e Y.
Para una simple explicación, usamos dos etiquetas: rojo y azul, con dos funciones X e Y, entonces entrenamos nuestra clasificación con la salida en x e y, arreglando como rojo o azul.
Lo que hace es trazar un hiperplano en el plano de eje x e y y ahora una vez que tenemos el mejor hiperplano trazado, desde un lado de la línea la cosas que caigan dentro de ese lado son rojas y las que queden afuera o del otro lado de la línea van a ser azules.
A veces las ubicaciones que tienen los datos son muy complejas y no es tan fácil poder trazar una simple línea como por ejemplo:

En este caso el mejor hiperplano va a ser el siguiente:


Aplicaciones de los algoritmos de clasificación
Análisis de sentimientos:
Lo que hacemos en esta técnica es analizar una frase y poder detectar tanto sentimientos, emociones, etc. Y poder ahora clasificar el texto escrito por una persona, como positivo, negativo o neutral.
Esto les permite a las compañías seguir en tiempo real el lanzamiento de un producto, por las reacciones de un cliente, además puede por ejemplo ver cómo impacta dicho producto en el mismo, además estas redes pueden identificar el sarcasmo, palabras mal escritas, etc.
Clasificación de los emails en spam o no spam
Es uno de los usos más comunes de todos, que estamos usando todo el tiempo, con técnicas de reconocimiento de texto, etc. Podemos identificar cuando el email es no deseado, o es un malware.
Clasificación de documentos
Esto sirve tanto como para clasificar documentos enteros en diferentes categorías, como por ejemplo, legales, de salud, etc. Usando técnicas de clasificación de texto, etc.
Clasificación de imágenes
Es previamente entrenado para diferentes categorías de imágenes, esto podría ser el asunto de una imagen, un valor numérico, un tema, etc. Por ejemplo una imagen para ver si es un animal, agua, etc.
Podemos con varias etiquetas de imágenes, ver si cae dentro de cada una o no.

 
 
 
 
 
 
 
 
 
 
