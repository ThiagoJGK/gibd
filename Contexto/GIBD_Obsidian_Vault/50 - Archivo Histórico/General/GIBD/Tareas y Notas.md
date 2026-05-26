---
aliases: [Tareas y Notas]
tags:
  - grupo/general
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2023-04-18
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Tareas y Notas.docx"
tamanio_bytes: 15075
---

# Tareas y Notas

Ruta interna: `GIBD/Tareas y Notas.docx`

---

TAREAS 2023 - Primer Cuatrimestre

La meta es obtener resultados para principios de Junio, de tal manera de poder presentar trabajos en congresos/revistas, de los cuales la mayoría cierra el envío a fines de junio o en julio.


Extracción de características sobre BD marcas, utilizando transfer learning y transfer learning+fine tuning, con modelos entrenados sobre ImageNet y alguna otra BD de elementos parecidos a las marcas. Zoe Vidal - Iván Bonti.

Extracción de características sobre BD marcas utilizando Triplet loss y Categorical loss.
Federico Lederhos - Andres Pascal
Extracción de características sobre otra BD de imágenes en blanco y negro (simbolos), utilizando Redes Siamesas y aumentación con relieve (tratando de reducir la cantidad de imágenes de entrenamiento necesarias). Lopez Martin, León Castiglioni 
Extracción de características sobre otra BD de imágenes a color (tipo escudos de clubes, es decir, imágenes geométricas), utilizando Redes Siamesas y aumentación. Ver si el relieve se puede adaptar a imágenes a color. Lucas Tonelotto - Agustina Bonti 
Extracción de características sobre BD de imágenes a color (tipo escudos), utilizando transfer learning y transfer learning+fine tuning, con modelos entrenados sobre ImageNet
Función de distancia utilizando Triplet loss, entrenado para marcas. Federico Stauber - Adrián Planas

Texto: investigar mecanismos de atención y transformers

NOTA: Los juegos de llaves del box los tienen León, Fede Lederhos y Zoe

NUEVAS ASIGNACIONES
30/08/2022

Image Segmentation (IS)
Agustina Bonti
	División de una imagen en sus objetos componentes
	ver https://neptune.ai/blog/image-segmentation

Object Detection (OD)
Zoe Vidal
Detección y localización de un objeto dentro de una imagen
ver https://machinelearningmastery.com/object-recognition-with-deep-learning/
https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/
https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/
	

Clasificación de Imágenes con otras técnicas que no sean CNNs: (CI)
Iván Bonti
Logistic Regression.
Naive Bayes.
K-Nearest Neighbors.
Decision Tree.
Random Forest
Support Vector Machines.
	ver 
		https://monkeylearn.com/machine-learning/
  		https://monkeylearn.com/blog/classification-algorithms/
Notas: Si se quiere clasificar imágenes de animales, hay 4 clases de los mismos y tengo que ver si pertenece a cada clase, hay muchos métodos que vienen desde antes de las redes neuronales. Sirve de comparación para las técnicas que estamos utilizando. 

Clustering (Agrupamiento) - Aprendizaje No Supervisado - (CL)
León castiglioni
Los objetos no poseen etiquetas, todos estos son tomados y se intentan dividir en clases  según su similitud. 
ver	
https://bdtechtalks.com/2020/12/28/machine-learning-customer-segmentation/
https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/

GMM
https://vitalflux.com/gaussian-mixture-models-what-are-they-when-to-use/#:~:text=Gaussian%20mixture%20models%20(GMMs)%20are,marketing%20and%20so%20much%20more!

Mean-shift
https://www.geeksforgeeks.org/ml-mean-shift-clustering/
https://towardsdatascience.com/understanding-mean-shift-clustering-and-implementation-with-python-6d5809a2ac40


videos encontrados:
https://www.youtube.com/watch?v=EZOab1vkFmI&t=467s
https://www.youtube.com/watch?v=T76paW6fJBI&t=387s

Extracción de Características para Búsquedas por Similitud (Siamesas, Triplet Loss) (EC)
Federico Lederhos/Andrés

Cómo crear generadores de datos
https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3


 
Metric Learning: Funciones de Distancia para Búsquedas por Similitud (ML)
Adrián/Federico Stauber

https://scikit-learn.org/stable/
http://scikit-learn.org/metric-learn/user_guide.html


Redes Generativas Adversarias (GAN)
Lucas Tonelotto
https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/
https://notebook.community/jonbruner/generative-adversarial-networks/gan-notebook


Autoencoders (AE)
Lopez Martin
https://www.v7labs.com/blog/autoencoders-guide
https://atcold.github.io/pytorch-Deep-Learning/es/week07/07-3/
https://colab.research.google.com/github/rickwierenga/notebooks/blob/master/autoencoders.ipynb 


Nombre de los archivos: Iniciale temática + Año + Nombre.
Ejemplo: CL - 2017 - Agrupamiento por características de pétalos de flores


Entorno de desarrollo:
https://www.anaconda.com/


Tareas

1) Utilizar una CNN sobre un MNIST (la última capa) como extractor de características para buscar marcas por similitud.
2) Utilizar una CNN sobre un MNIST (la capa flatten) como extractor de características para buscar marcas por similitud.
3) Agrandar la CNN sobre un MNIST (la última capa) como extractor de características para buscar marcas por similitud.
4) Agrandar la CNN sobre un MNIST (la capa flatten) como extractor de características para buscar marcas por similitud.
5) Utilizar una red tipo ResNet50, VGG16, DenseNet, etc. ya entrenada sobre MNIST (o entrenarla), como extractor de características para buscar marcas por similitud.
Verificar que x_train y x_test estén normalizados.
ResNet50

RedNet50


VGG16



DenseNet
https://www.kaggle.com/code/zhouchen1998/mnist-keras-densenet/notebook

https://gitcode.net/qq_39898814/deeplearning-models/-/blob/master/pytorch_ipynb/cnn/cnn-densenet121-mnist.ipynb


No van:
https://www.kaggle.com/code/jcy1996/mnist-with-vgg-16/notebook

6) Entrenar una red Siamesa sobre MNIST y utilizarla como extractor de características para buscar marcas por similitud.
https://www.kaggle.com/code/scratchpad/notebookb1d92d7cb8/edit
Verificar la cantidad de elementos de entrenamiento (toma 1000 con una base de 60000).
Verificar en la creación de los pares de entrenamiento, si el valor correspondiente a imágenes similares tiene que ser 0 e imágenes distintas 1.
Verificar el balance entre los pares de entrenamiento que son similares y que no lo son.
El entrenamiento de la red siamesa sobre Mnist dio muy buenos resultados ya que las búsquedas por similitud sobre los dígitos arrojaron un 99% de aciertos. Utilizando esta misma red para buscar por similitud Marcas el porcentaje de aciertos se redujo al 26.5%.
La tarea siguiente es:
Realizar un fine-tuning re-entrenando con Marcas utilizando aumentación para generar marcas similares.

Sitios:
https://machinelearningmastery.com/best-practices-for-preparing-and-augmenting-image-data-for-convolutional-neural-networks/

https://nanonets.com/blog/data-augmentation-how-to-use-deep-learning-when-you-have-limited-data-part-2/

https://towardsdatascience.com/improves-cnn-performance-by-applying-data-transformation-bf86b3f4cef4




7) Entrenar una red Triplet Loss sobre MNIST y utilizarla como extractor de características para buscar marcas por similitud.
https://www.kaggle.com/code/guichristmann/training-a-triplet-loss-model-on-mnist/notebook

https://omoindrot.github.io/triplet-loss

https://towardsdatascience.com/image-similarity-using-triplet-loss-3744c0f67973#:~:text=For%20Triplet%20Loss%2C%20the%20objective,define%20similar%20and%20dissimilar%20images

Usando CIFAR10:
https://colab.research.google.com/drive/1VgOTzr_VZNHkXh2z9IiTAcEgg5qr19y0

Video acerca de Triplet Loss:
https://www.coursera.org/lecture/convolutional-neural-networks/triplet-loss-HuUtN


8) Utilizar una CNN sobre la BD del Alfabeto (la capa flatten) como extractor de características para buscar marcas por similitud.


Info extra:
https://keras.io/api/applications/#resnet50



Buscar bibliografía sobre:
Redes siamesas
triplet loss
transfer learning
one shot learning
few shot learning
extracción de características de imágenes
Augmentation


Adrián y Federico:
Leer el siguiente artículo sobre medidas a tener en cuenta en los experimentos:
https://learncuriously.wordpress.com/2018/10/21/confused-by-the-confusion-matrix/




AUGMENTACIÓN

https://claudiovz.github.io/scipy-lecture-notes-ES/advanced/image_processing/index.html
https://www.tensorflow.org/tutorials/images/data_augmentation
https://omes-va.com/transformacion-de-perspectiva-opencv-con-python/
https://pythoneyes.wordpress.com/2017/06/23/anadir-ruido-a-imagenes-con-python/
https://neptune.ai/blog/data-augmentation-in-python
https://machinelearningmastery.com/best-practices-for-preparing-and-augmenting-image-data-for-convolutional-neural-networks/

