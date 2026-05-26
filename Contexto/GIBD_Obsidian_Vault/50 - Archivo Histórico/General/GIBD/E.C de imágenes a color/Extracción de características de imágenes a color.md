---
aliases: [Extracción de características de imágenes a color]
tags:
  - grupo/general
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2023-08-07
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/E.C de imágenes a color/Extracción de características de imágenes a color.docx"
tamanio_bytes: 920591
---

# Extracción de características de imágenes a color

Ruta interna: `GIBD/E.C de imágenes a color/Extracción de características de imágenes a color.docx`

---

Extracción de características de imágenes a color

Extracción de características sobre otra BD de imágenes a color (tipo escudos de clubes, es decir, imágenes geométricas), utilizando Redes Siamesas y aumentación. Ver si el relieve se puede adaptar a imágenes a color. Lucas Tonelotto - Agustina Bonti 

Para desarrollar nuestra tarea, en principio, vamos a definir algunos conceptos importantes, como la aumentación, las redes siamesas, y en qué consiste la extracción de características. 
La extracción de características es un proceso importante en el aprendizaje automático que consiste en identificar y representar patrones en los datos de entrada. En el caso de imágenes, esto implica identificar las características visuales de la imagen, como bordes, formas, texturas y colores, que son relevantes para una tarea específica, como la clasificación o la detección de objetos.
Las Redes Siamesas son una arquitectura de red neuronal que se utiliza para comparar dos imágenes. La idea es alimentar dos imágenes idénticas a dos ramas de la red y hacer que ambas ramas compartan los mismos pesos. Esto permite que la red aprenda a medir la similitud entre dos imágenes. En el caso de la extracción de características, la idea es utilizar una red siamesa para comparar cada imagen de la base de datos con una imagen de referencia y extraer las características comunes.
La aumentación de datos es un proceso que implica generar nuevas instancias de datos a partir de las instancias existentes. En el caso de imágenes, esto puede implicar rotar, escalar, recortar o cambiar el brillo y el contraste de las imágenes existentes. La aumentación de datos es útil porque aumenta la cantidad de datos disponibles para el entrenamiento, lo que puede mejorar el rendimiento del modelo.
En el contexto de la extracción de características de imágenes de escudos de clubes, la idea sería utilizar una red siamesa para comparar cada imagen de la base de datos con una imagen de referencia de un escudo de clubes y extraer las características comunes. La aumentación de datos se podría utilizar para generar nuevas instancias de cada imagen en la base de datos mediante la rotación, el cambio de escala y otros cambios menores para aumentar la cantidad de datos disponibles para el entrenamiento.
https://drive.google.com/drive/folders/12cVA9aUXiq20uRXWRTzC8gXJBHvVHliY?usp=sharing


¿Podrías darme el código para generar esta red siamesa que compare las imágenes de la base de datos a color con alguna imagen de referencia y extraiga sus características? Si puede ser incluí comentarios en el código que expliquen el funcionamiento

https://www.kaggle.com/code/taranmarley/football-logo-classification-100

https://colab.research.google.com/drive/19cHJowFfI6RUpmsSSD29ybrhItye2WDG?authuser=1

Por cada imagen de la base de datos original debemos formar dos pares de imágenes similares, es decir que a partir de la imagen original creamos dos similar con colores diferentes para formar los pares, y dos pares de diferentes, simplemente por cada escudo lo pareamos con otro para ser diferente. Todo esto lo ubicamos en una lista  de 20000 pares similares y 20000 diferentes. Utilizaremos esto para entrenar la red siamesa, donde 0 son los pares similares y 1 los diferentes. (Código utilizado )

Cambiamos BRILLO y CONTRASTE con el imageDataGenerator

“pueden servir, si, aunque es raro que la deje así, porque la original tiene transparencia en esa parte, es decir, en los alrededores, mientras que la generada tiene ese "ruido" de fondo. De todos modos, como se ve claramente que son imágenes similares, pueden servir”

Algunos apuntes:


Las redes siamesas son un tipo de red neuronal que procesa dos o más entradas idénticas o similares a través de redes neuronales simétricas que comparten los mismos pesos y arquitectura. Los inputs en las redes siamesas se utilizan para comparar y contrastar los patrones en los datos de entrada.

Por ejemplo, en tareas de reconocimiento de imágenes, las redes siamesas pueden recibir dos imágenes como entrada, procesar cada imagen por separado a través de la red neuronal simétrica compartida, y luego comparar las representaciones finales de cada imagen para determinar si son similares o diferentes. En otras palabras, las redes siamesas son útiles para comparar y evaluar la similitud entre dos entradas.

Las redes siamesas se utilizan en una variedad de aplicaciones, incluyendo el reconocimiento de voz, el reconocimiento de escritura a mano, la detección de fraude en transacciones financieras y la detección de plagio en documentos. También son utilizadas en tareas de procesamiento de lenguaje natural, como la identificación de sinónimos y la evaluación de la similitud entre oraciones.

Tenemos este código de red siamesa:

# SIAMESA

siamese_net.summary()
siamese_net.fit([left_input,right_input], targets,
          batch_size=16,
          epochs=5,
          verbose=1,
          validation_data=([test_left,test_right],test_targets))

Primero, se llama al método summary() en el objeto siamese_net para imprimir un resumen de la arquitectura de la red neuronal siamesa.

Luego, se entrena la red neuronal siamesa llamando al método fit(). Esta red toma como entrada dos conjuntos de datos: left_input y right_input, que probablemente sean matrices de numpy con vectores de características de alguna tarea específica, y un conjunto de etiquetas objetivo targets que representan la clase o valor esperado para cada par de entradas.

La red neuronal siamesa intentará aprender una representación de características compartidas para cada entrada y luego comparar las dos representaciones para determinar si son similares o diferentes. El método fit() se usa para entrenar la red neuronal siamesa en los datos de entrada y ajustar sus pesos para minimizar la función de pérdida de la red.

El entrenamiento se realiza en mini lotes de tamaño 16 y se realiza durante 5 épocas. La opción verbose=1 indica que la salida del entrenamiento se mostrará en la consola. Finalmente, el método validation_data se utiliza para proporcionar un conjunto de datos de validación adicional para evaluar el rendimiento del modelo durante el entrenamiento.





https://colab.research.google.com/drive/1WbPhdylnqsjvt8xoc9xuBe92oJ0BQlE-?usp=sharing
Si deseas utilizar tus propias imágenes para entrenar un modelo de aprendizaje profundo, necesitarás seguir algunos pasos adicionales:

Cargar tus imágenes: Primero, debes cargar tus imágenes en tu entorno de programación. Puedes hacerlo utilizando una biblioteca como OpenCV o PIL.

Preprocesar tus imágenes: Antes de pasar tus imágenes al modelo de aprendizaje profundo, debes preprocesarlas para que sean compatibles con el modelo. Esto podría incluir redimensionar las imágenes, normalizar las intensidades de los píxeles, cambiar la escala de color, etc.

Guardar tus imágenes: Una vez que hayas cargado y preprocesado tus imágenes, debes guardarlas en una estructura de datos que el modelo pueda utilizar para el entrenamiento. Esto podría ser una matriz NumPy o un objeto de TensorFlow Dataset.

Entrenar el modelo: Finalmente, puedes utilizar tus imágenes para entrenar el modelo de aprendizaje profundo utilizando las funciones de entrenamiento de TensorFlow, como fit() o train_on_batch().

https://colab.research.google.com/drive/1WbPhdylnqsjvt8xoc9xuBe92oJ0BQlE-?usp=sharing#scrollTo=h6hYvai7_bjX

1 - Etiquetas de clases, como guardarlas en un array?
2 - Generador de datos para entrenar directamente sin guardar en memoria. 
 
100 imagenes  → 70 para entrenamiento, hay que pensar cuantos pares hay que crear 


Anotaciones

Dividir el conjunto de 100 imágenes, dejar 70 en una carpeta que llamamos ‘val’ y le vamos a poner ‘entrenamiento’ de nombre. Las otras 30 en una llamada ‘validación’.
Después tenemos que hacer que estas 70, pasen por un for del 1 al 10 en la que, con un random(i) e if que vayan de 10 en 10 aplicar distintas transformaciones (brillo, contraste, rotación,traslación, etc). La cantidad de veces que van a pasar por el for (es decir del 1 al 10) está dada por la cantidad de imágenes que queremos generar a partir de una misma, es decir los pares similares que vamos a tener.



Dejamos otra imagen de otra posible solución a continuación:




30/05
Tenemos que probarlo, antes tenemos q hacer q el random ese ande, el codigo está a lo ultimo

13/06
Crear carpeta con todas imagenes de consulta (70) (imagenes png de clubes que aparezcan en la bd, de nombre le pongo el mismo que el de la consulta pero ‘_nro’ siendo nro desde 1 ejemplo, milan_1,milan_2,..

Agregar parte de busqueda por similitud, 
+agregar el np.array en el código del final


07/08
Rehacer la carpeta de consultas y dejarlas 100x100, sin fondo
Ver cuál falta en imágenesbd


