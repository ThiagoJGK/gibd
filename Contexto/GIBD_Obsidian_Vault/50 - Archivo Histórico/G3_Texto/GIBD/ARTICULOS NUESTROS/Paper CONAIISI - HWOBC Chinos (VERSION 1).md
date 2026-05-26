---
aliases: [Paper CONAIISI - HWOBC Chinos (VERSION 1)]
tags:
  - grupo/g3_texto
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2023-09-08
origen_zip: GIBD-20260521T205218Z-3-004.zip
ruta_interna: "GIBD/ARTICULOS NUESTROS/2023 - CONAIISI/Caracteres Chinos/Paper CONAIISI - HWOBC Chinos (VERSION 1).docx"
tamanio_bytes: 877548
---

# Paper CONAIISI - HWOBC Chinos (VERSION 1)

Ruta interna: `GIBD/ARTICULOS NUESTROS/2023 - CONAIISI/Caracteres Chinos/Paper CONAIISI - HWOBC Chinos (VERSION 1).docx`

---


Paper CONAIISI - HWOBC chinos


Resumen
El carácter de hueso de oráculo (OBC) de la antigua China representa el sistema de escritura antiguo más renombrado a nivel mundial. El estudio e identificación de los OBC y su desciframiento se erigen como uno de los aspectos más cruciales dentro de la esfera de investigación de estos artefactos históricos. Entre los desafíos que enfrenta esta investigación, destaca el hecho de que la revisión de la literatura al respecto suele demandar considerables recursos temporales y de mano de obra. En consecuencia, la digitalización de la literatura OBC, empleando técnicas de reconocimiento automático, surge como una dirección inevitable para el desarrollo futuro de este campo. No obstante, se observa una carencia significativa en la disponibilidad de bases de datos que abarquen caracteres óseos de Oracle escritos a mano. En este contexto, el presente artículo introduce una base de datos denominada HWOBC, que recopila caracteres óseos de Oracle escritos a mano, constituyendo así un aporte valioso para la investigación en este ámbito.

Durante la última década, las Redes Neuronales Profundas Convolucionales (CNN) han demostrado un alto rendimiento en el procesamiento de imágenes. Este artículo presenta un enfoque que combina preprocesamiento, aumento de datos y modelos CNN para aprender una función de distancia en One-Shot learning, utilizando una arquitectura de Redes Siamesas en su proceso de entrenamiento. Este método busca abordar la escasez de datos en tareas de reconocimiento y clasificación de objetos, contribuyendo al avance de la visión por computadora y el reconocimiento de patrones.

Palabras Clave: Búsquedas por Similitud, Marcas de Ganado, Metric Learning, CNNs, Redes Siamesas, One-Shot Learning, Aumentación.
1. Introducción
La expansión de tipos de datos que abarcan imágenes, audio, video y texto ha evidenciado la limitación de los modelos de búsqueda convencionales. Como respuesta, la búsqueda por similitud emerge como una solución eficaz para localizar objetos análogos a una consulta en bases de datos no convencionales. Los Espacios Métricos [1] representan un enfoque formalizado que facilita búsquedas por similitud y la implementación de métodos de acceso más eficaces. 
Este artículo se centra en el desarrollo de técnicas de aprendizaje de funciones de distancia con el fin de facilitar la búsqueda de similitud en imágenes que representan caracteres óseos de oráculo (OBC). Estos caracteres se remontan a la Edad del Bronce en China y representan el sistema de escritura más antiguo documentado en el país [2]. Una de las principales dificultades abordadas en esta investigación se relaciona con la identificación de los glifos dentro de los OBC. Desde la perspectiva de la temática estudiada, se reconoce que los glifos de los OBC son intrincados y diversos, y un solo OBC puede presentar múltiples variantes, lo que resulta en un significativo desafío en términos de inversión de tiempo y recursos humanos. Desde la perspectiva metodológica, la investigación de los glifos de OBC implica un exhaustivo análisis de la literatura existente; sin embargo, los libros impresos sobre este tema suelen ser densos en contenido y difíciles de recuperar, lo que agrega complejidad al proceso investigativo.
Recientemente, el campo de la visión por computadora ha experimentado un avance sustancial gracias a las técnicas de Redes Neuronales Convolucionales Profundas (DCNN). Los notables logros alcanzados por las DCNN en la comunidad de visión por computadora nos motivaron a considerar su aplicación en la identificación de Caracteres Óseos de Oráculo (OBC). Sin embargo, estos enfoques basados en DCNN dependen en gran medida de conjuntos de datos de entrenamiento con etiquetas, recursos que a menudo resultan inaccesibles, particularmente en este contexto específico.
Este problema plantea tres desafíos sustanciales en el ámbito del aprendizaje automático: a) se dispone solamente de una única muestra por categoría (aprendizaje One-Shot), b) la cantidad de categorías aumenta constantemente, lo que resulta en un costo prohibitivo al reentrenar el modelo por cada adición de nueva categoría, y c) la definición de una función de distancia precisa para medir la similitud entre dos imágenes es una tarea no trivial. En este artículo, se exponen modelos de Redes Neuronales Convolucionales (CNN) junto con técnicas de preprocesamiento, aumento de datos y entrenamiento, con el propósito de crear una función de distancia efectiva que evalúe el nivel de similitud entre imágenes de caracteres óseos de oráculo (OBC).
El resto de este documento está organizado de la siguiente manera: la Sección 2 presenta el trabajo relacionado; el proceso de aprendizaje de la función de distancia y los modelos utilizados se explican en detalle en la Sección 3. La Sección 4 describe los experimentos realizados y la Sección 5 los resultados obtenidos. Finalmente, en la Sección 6 se presentan las conclusiones y el trabajo futuro.
2. Trabajo Relacionado
En esta sección, se expone el contexto del estudio actual, destacando las técnicas fundamentales empleadas para emplear la Convolutional Neural Network (CNN) como métrica de similitud en la comparación de marcas de ganado.

2.1 Recuperación de Imágenes Basada en Contenido
La Recuperación de Imágenes Basada en Contenido (CBIR) es un procedimiento que busca recuperar imágenes de una base de datos considerando características visuales como similitud en colores o formas. Los atributos predominantes son color, textura y forma [3, 4]. El color prevalece debido a su facilidad para ser extraído [5]. En contraste, adquirir datos sobre forma y textura es complejo y costoso [6]. Este enfoque es crucial para aplicaciones que requieren la búsqueda eficiente de imágenes relevantes, como la medicina y la gestión de activos visuales. Mejorar la extracción de atributos complejos es un desafío continuo en el CBIR.
Los histogramas [7] son soluciones ampliamente empleadas, caracterizadas por su eficiencia computacional. Sin embargo, presentan insensibilidad a la variación de la posición de la cámara, carencia de información espacial y limitaciones en la adaptación a cambios en la luminosidad [8].
La forma es una característica visual esencial para describir imágenes, aunque puede ser influenciada por defectos, ruido, oclusión y distorsiones, lo que dificulta su análisis. Aspectos como centro de gravedad, masa, dispersión, varianza, eje de menor inercia, rectangularidad y convexidad permiten su descripción [9]. Además, enfoques avanzados como Momentos de Hu, Legendre o Zernike [11, 12, 13] ofrecen resultados más precisos y confiables.

2.2 Búsquedas en Espacios Métricos
Los CBIR pueden ser generalizados y modelados mediante Espacios Métricos con el fin de lograr búsquedas eficientes. En [1], se muestra que el problema de búsqueda de similitud puede ser formulado de la siguiente manera: dado un conjunto U de objetos y una función de distancia d definida entre ellos, el objetivo es recuperar todos los elementos cercanos a un objeto dado, utilizando d como criterio de similitud. 
Esta función d satisface las propiedades requeridas para ser una métrica:
(a) ∀x∈U, d(x, x)=0 (reflexividad)
(b) ∀x, y∈U, d(x, y)≥ 0 (positividad)
(c) ∀x, y∈U, d(x, y)=d(y, x) (simetría)
(d) ∀x, y, z∈U, d(x, z)≤ d(x, y)+d(y, z) (desigualdad triangular)
Un subconjunto finito X de U, denominado base de datos, se utiliza para realizar la búsqueda. Uno de los tipos de consulta relevante en las búsquedas por similitud es la consulta de los k Vecinos Más Cercanos o NNk(q)d, que recupera los k elementos de X  más cercanos a  q: 
NNk(q)d = A ={x∈X /∀y∈(X-A), d(q, x)≤ d(q, y) }  y  |A|=k
Dada una base de datos con n objetos, responder a estas consultas de forma trivial implica realizar n evaluaciones de distancia, lo cual puede resultar altamente costoso en aplicaciones prácticas. La relevancia de modelar estas consultas mediante espacios métricos se basa en la posibilidad de utilizar índices que aprovechan la propiedad de la desigualdad triangular para descartar elementos sin necesidad de compararlos directamente con la consulta. Esto mejora significativamente el proceso de búsqueda [1, 14, 15], convirtiéndo en una solución eficiente en diversas aplicaciones.

2.3 Redes Neuronales Convolucionales (DCNN o CNN) y Redes Siamesas
La arquitectura de las Redes Neuronales Convolucionales (CNNs) fue concebida por Fukushima en 1988 [16], mientras que en la década de los años '90, LeCun y otros investigadores [17] introdujeron algoritmos de aprendizaje basados en el gradiente que demostraron resultados alentadores en tareas de clasificación. Sin embargo, fue en la última década cuando las CNNs experimentaron avances notables y destacaron en una variedad de problemas de reconocimiento de patrones. 
Ejemplos de arquitecturas actuales de CNN incluyen AlexNet [18], VGG Net [19], y NiN [20]. Además, se han propuesto alternativas avanzadas y eficientes como DenseNet [21], GoogLeNet [22], y Residual Networks [23]. Aunque los componentes básicos son similares en todas las arquitecturas, las variaciones topológicas conducen a diferencias significativas en la eficiencia del entrenamiento y la precisión de los resultados.
El algoritmo de aprendizaje de las Redes Neuronales Convolucionales (CNNs) no intrínsecamente incorpora el concepto de similitud entre imágenes. No obstante, arquitecturas recientes, como las Redes Neuronales Siamesas [24, 25] y Triplet Loss [26], han surgido para abordar este concepto y han demostrado eficacia en el reconocimiento facial. Las Redes Siamesas constan de dos CNNs idénticas, utilizadas para extraer vectores de características o entrenar funciones de distancia. Durante el entrenamiento, ajustan sus parámetros para minimizar la distancia entre imágenes similares y maximizarla entre imágenes diferentes, lo que facilita la representación y comparación de similitudes en el espacio de características. 
La utilización de Redes Siamesas en conjunto con Redes Neuronales Convolucionales (CNNs) con el objetivo de extraer características destinadas a la Búsqueda por Similitud presenta dos desafíos críticos:
El primero radica en la limitada disponibilidad de instancias para el entrenamiento, un fenómeno conocido como One-Shot Learning o Few-Shots Learning [27, 28]. Esta limitación dificulta considerablemente el entrenamiento directo de los modelos. Hasta la fecha, la estrategia más efectiva para abordar esta dificultad ha sido el Transfer Learning o Fine Tuning [29, 30]. Sin embargo, esta técnica se ve restringida por la disponibilidad de bases de datos que contengan características similares.
El segundo desafío se relaciona con la generalización. Cuando una CNN se entrena para extraer vectores característicos, su comportamiento tiende a estar fuertemente ligado a las clases con las que se entrenó inicialmente. Esto conlleva dificultades para adaptarse eficazmente a nuevas imágenes incorporadas a la base de datos sin un proceso de reentrenamiento. Una solución recientemente investigada es el Aprendizaje Métrico (Metric Learning) [31], que propone la directa obtención de una función de distancia correspondiente a los datos existentes en lugar de centrarse en la extracción de características. Este enfoque se emplea en el contexto del presente estudio.
2.4 Aprendizaje de Funciones de Distancia (Metric Learning)
El aprendizaje métrico [32] se centra en la adquisición de funciones de distancia que se ajustan a una definición de similitud específica. En el contexto del reconocimiento de patrones, esta definición de similitud es intrínseca a la tarea en cuestión, y el éxito del aprendizaje radica en la adecuada alineación con dicha tarea.
En un estudio exhaustivo [33], se realiza un análisis detallado de los métodos de aprendizaje métrico, proporcionando una visión panorámica de las diversas técnicas y enfoques empleados en este campo. En [34], se propone un enfoque de aprendizaje métrico basado en márgenes. Además, se describe una técnica de aprendizaje métrico discriminatorio aplicada a la verificación de rostros en otro trabajo relacionado [35]. Por último, [36] ofrece una panorámica completa del aprendizaje métrico, abarcando múltiples enfoques y aplicaciones, y subrayando los desafíos y tendencias prominentes en este dominio de investigación.

2.5 Búsquedas por Similitud de OBC
Investigaciones previas han abordado este tema mediante diversos enfoques. En [37], se emplearon técnicas de visión por computadora para analizar caracteres oráculo, generando el conjunto de datos Oracle-20K. Se estudiaron y analizaron representaciones visuales relacionadas con formas y bocetos, proponiendo una nueva representación jerárquica. Además, se integró el aprendizaje autosupervisado en el aumento de datos y se presentó un enfoque innovador, denominado Orc-Bert Augmentor, preentrenado mediante aprendizaje autosupervisado, para el reconocimiento de personajes oráculo en escenarios de datos escasos (few-shot) [38].
En [39], se propone un método basado en una red neuronal convolucional profunda para el reconocimiento de OBC incompletos. Se extraen características del conjunto de datos de OBC incompletos con pocas muestras, permitiendo la identificación de OBC incompletos en diferentes imágenes. En otro trabajo [40], se utiliza una red neuronal convolucional (CNN) para mapear las imágenes de OBC a un espacio euclidiano donde la distancia entre las muestras mide sus similitudes, posibilitando la clasificación mediante la regla del vecino más cercano (NN).
3. Aprendizaje de la Función de Distancia
En esta sección se describe la base de datos y los modelos utilizados para el aprendizaje de funciones de distancia a partir de una base de datos compuesta por sólo una instancia de cada marca, utilizando Redes Siamesas, técnicas clásicas de Aumentación y el método de relieve presentado en [41].
3.1 Base de Datos de OBCs
En el presente estudio, se trabajó con una base de datos que consta de 1000 OBCs donde, se tienen entre 19 y 24 dibujos a mano de cada uno, lo que plantea una mejora significativa en cuanto a las 450 imágenes por imágen utilizadas en [41]. Cabe remarcar que estas imágenes fueron obtenidas de la base de datos pública llamada HWOBC, la cual fue compuesta por caracteres chinos dibujados a mano por diferentes científicos, provenientes de la época de bronce, y recopilados por investigadores en ella, para su uso de forma libre.
La Fig. 1 exhibe algunos ejemplos de estos caracteres óseos de oráculo.

Fig. 1. OBCs pertenecientes a la base de datos.

3.2 Modelos utilizados
En trabajos de investigación relacionados a esta problemática se probaron doce modelos de los cuales se obtuvieron los tres mejores (Mod1, Mod2 y Mod3). El mejor de ellos, acorde a estas pruebas previas, Mod1 es el que se utiliza en este trabajo. Mod1 consiste en una CNN de seis capas convolucionales, todas con activación ReLU. Cada capa convolucional está intercalada con una capa de Batch-Normalization. También este modelo utiliza un tamaño de vector de salida de 128 elementos. Las tres primeras capas convolucionales emplean 32 kernels, mientras que las restantes utilizan 64. Se emplearon kernels de (3, 3) en cuatro capas y de (5, 5) en las dos restantes. Adicionalmente, se incorporan dos capas de Dropout para mitigar el overfitting. Finalmente, el modelo incluye una capa "flatten", que representa la salida que es utilizada por la red siamesa. Este modelo posee 193.664 parámetros entrenables. Éste, fué utilizado para obtener representaciones de las dos imágenes a comparar, que luego se concatenan y pasan por dos capas densas con función de activación ReLU. Por último, se agrega una capa extra con la función Sigmoide que devuelve el valor final de similitud entre ambas imágenes. Se emplea Binary Cross Entropy como función de pérdida y el optimizador Adam para el entrenamiento del modelo.
Los modelos Mod2 y Mod3 se diferenciaban del modelo Mod1 en las siguientes características:
Mod2: Este modelo utiliza un tamaño de vector de salida de 256 neuronas. Las cuatro primeras capas convolucionales emplean 32 kernels, mientras que las restantes utilizan 64. Se han empleado kernels de (3, 3) en todas las capas convolucionales. Adicionalmente, se incorporan una capa de MaxPooling2D y  dos capas de Dropout para mitigar el overfitting. Finalmente, el modelo incluye una capa "flatten" antes de pasar a una capa densa de activación softmax, que representa la salida que será utilizada por la red siamesa.
Mod3: En este modelo, se utiliza la propiedad padding=’same’ en las capas de convolución para mantener las dimensiones de la imagen. A su vez, el modelo utiliza un tamaño de vector de 128 neuronas como salida. Además posee 5 capas convolucionales en lugar de 6, como los modelos anteriores. Las cuatro primeras capas convolucionales se intercalan entre 32 y 64 kernels, mientras que la restante es de 256. Se han empleado kernels de (3, 3) en todas las capas convolucionales. Adicionalmente, se incorporan, una capa de MaxPooling2D, una de GlobalAveragePooling2D y dos capas de Dropout para mitigar el overfitting. Finalmente, el modelo incluye una capa "flatten" y una capa densa que representa la salida que será utilizada por la red siamesa.

Las imágenes utilizadas en la base de datos y en las consultas fueron de 50x50 píxeles.
Una característica importante de estas CNNs es que a pesar de ser profundas son relativamente pequeñas (Mod1 posee solo 193.664 parámetros), lo que permite que su uso como función de distancia no sea excesivamente costoso.
A pesar de ello, hay que notar que estos modelos no cumplen con las propiedades de las distancias métricas. De hecho, la función obtenida no es reflexiva, ni simétrica y tampoco cumple la desigualdad triangular. La simetría se resuelve fácilmente haciendo d(x,y) = CNN(x,y) + CNN(y,x). Sin embargo, desde el punto de vista de los índices métricos, la propiedad que más importa es la desigualdad triangular, que es la que permite descartar elementos sin necesidad de compararlos con la consulta. Por esta razón, se realizó un experimento adicional para determinar en qué porcentaje la función de distancia satisface la desigualdad triangular para los elementos de la BD. Para ello se tomaron todas las ternas posibles y se verificó el cumplimiento de esta propiedad, alcanzando un 97,61% de éxito para Mod1. Esto significa que sería posible utilizar índices métricos para acelerar la búsqueda, aun cuando en un porcentaje reducido de casos se podría estar descartando elementos erróneamente (índices métricos aproximados).

3.3 Preprocesamiento de Datos
Para salvar posibles diferencias entre los trazos de cada artista, así también como posibles desplazamientos dentro del lienzo contenedor de la imágen, se propuso aplicar a las imágenes una serie de transformaciones. Dichas transformaciones utilizadas en los experimentos y para poder dar comienzo práctico a este trabajo consistieron en el reescalado a 50x50 píxeles de las imágenes. Posteriormente, se expandieron con deformación, para ocupar todo el ancho y alto del lienzo contenedor. Y posteriormente se les agregó el procesamiento “Relieve”, propuesto en [41], que consiste en aplicar una expansión linealmente decreciente hacia los lados de las líneas que componen la imagen. Esta técnica produce mejoras significativas en la eficacia del entrenamiento, según nuestros experimentos. La Fig. 2 exhibe dos marcas (izquierda) después de su preprocesamiento y sus correspondientes relieves (derecha).







Fig. 2. OBCs y sus Relieves.


Los resultados experimentales demuestran que este mecanismo contribuye significativamente a que la Red Siamesa realice predicciones más robustas y estables en comparación con aquellas basadas exclusivamente en la aumentación estándar.
4. Experimentos Realizados
Para el entrenamiento de la Red Siamesa se seleccionaron al azar 200.000 pares de imágenes dibujadas, asegurando un equilibrio entre pares similares y no similares. Para esta tarea se procedió a utilizar 200 pares de cada imagen, 100 positivos y 100 negativos. Donde los negativos no se repitieron.
Se llevaron a cabo pruebas para evaluar la eficacia de los modelos mediante la función de distancia obtenida a partir de la Red Siamesa y las imágenes dibujadas. Éstas pruebas involucraron un lote de consultas (Q). Donde se utilizaron 50 imágenes de consulta dibujadas manualmente para este estudio, las cuales se generaron replicando marcas preexistentes en la base de datos. Estas se realizaron íntegramente utilizando una aplicación de diseño gráfico.
Los modelos fueron entrenados en bloques de 5 épocas con 200.000 pares distintos de imágenes. En total se realizaron 50 épocas de entrenamiento alcanzando un  un 0,996 de precisión. Se utilizó una PC con procesador Ryzen 7 5800x, 32 GB RAM y GPU GeForce GTX 1080 con 2560 núcleos CUDA.
Como métrica principal para evaluar la función de distancia, se empleó la tasa de aciertos (Hit Rate, expresada en porcentaje), dado que es el indicador más representativo para determinar si la solución propuesta es adecuada en el contexto del problema real planteado.
A continuación, en la sección 5, se comparan los resultados de los experimentos de este trabajo con otros previamente realizados y se agregan pruebas de búsquedas de nuevas imágenes no pertenecientes a la base de datos entrenada originalmente. Donde el problema reside en que los vectores resultantes de la extracción de características basadas en CNNs (y muchos otros métodos de aprendizaje automático actuales) están altamente ligados a las clases con las que se entrenan los modelos, lo que limita su capacidad de generalización. Estos vectores no logran adaptarse de manera efectiva a nuevas imágenes incorporadas a la base de datos, correspondientes a posibles "clases" adicionales. Una alternativa estudiada recientemente es el Aprendizaje Métrico (Metric Learning) [32-36], que propone aprender directamente una función de distancia correspondiente a los datos existentes en lugar de simplemente extraer características. Este es el enfoque utilizado en este estudio.
5. Resultados y Discusión
5.1 Comparación de Resultados
En [41] se utilizó una base de datos de marcas de ganado, la cual posee características similares a la presentada en este trabajo, por lo tanto, también se utilizó cómo punto de comparación de resultados el modelo de mejores resultados de dicho trabajo (SinRelQA), dada la semejanza de las características de la base de datos y dichos modelos utilizados para las búsquedas. 

Tabla 1 - Comparación de Resultados con [41]

En la Tabla 1 podemos observar que para ambos resultados expuestos en [41], para el modelo comparado, para los k vecinos más cercanos (NNk), para k=3 y 5 los resultados fueron mejores con Mod1, independientemente de las diferencias ya expresadas de ambos modelos.
En resumen del trabajo realizado, sin comparación con otros, los porcentajes de acierto en las búsquedas por similitud para los k vecinos más cercanos (NNk), para k=1, 3 y 5, para el modelo Mod1 propuesto fueron: 90% para k = 1; 98% para k = 3 y 100% para k = 5.
5.2 Inserciones y búsquedas de nuevas imágenes
En [2] se evaluaron cinco modelos convolucionales (DCNN), AlexNet, ResNet, Vgg11, Cascada y Melnyk-Net, para estos se realizaron modificaciones en las capas conectadas para equilibrar la precisión y el almacenamiento requerido mediante el uso de técnicas como global weighted average pooling (GAP), global weighted average pooling (GWAP) y global weighted output average pooling (GWOAP)

Tabla 2 - Comparación de Resultados de [2]

Los resultados mostrados en la Tabla 2 reflejan una tasa de precisión promedio de estos métodos, donde se destaca que Melnyk-Net alcanza una precisión del 97,67%, superando a los demás.	
Se puede observar, a simple vista, que los resultados de este trabajo son mejores que los de Mod1, de esta presentación. Pero, debemos remarcar que en [2] se utiliza un algoritmo que genera clases de agrupamiento para la base de datos inicial, por lo tanto, al momento de nuevas inserciones y búsquedas de imágenes no pertenecientes a esa base de datos, los algoritmos comenzarán a bajar la performance de aciertos. 
Por lo tanto, otro problema abarcado por los experimentos fué el de la búsqueda de imágenes no pertenecientes a la base de datos. Donde se expone al modelo a búsquedas de imágenes que no pertenecen al conjunto de entrenamiento, lo cual representa un desafío extra al de las búsquedas de imágenes preexistentes en la base de datos entrenada. Relacionados con la sección 2.4. 
Para este caso, se tomaron 30 imágenes nuevas y se las agregó al conjunto de datos inicial. Cabe resaltar que para estas 30 imágenes no hubo entrenamiento. Por lo tanto, las búsquedas de las mismas se realizaron sobre lo ya entrenado al modelo. Otra aclaración pertinente es que, las búsquedas (también 30 en total) se realizaron con dibujos hechos a mano de las 30 imágenes insertadas en la base de datos inicial.
En la Tabla 3, evidenciamos que Mod1 soporta de manera exitosa dichas inserciones y búsquedas.




Tabla 3 - Resultados de búsquedas de imágenes no pertenecientes a la base de datos inicial.

Los resultados obtenidos son satisfactorios, ya que, teniendo en cuenta que las 30 imágenes insertadas a la base de datos, no correspondientes al entrenamiento inicial, pero sí fueron encontradas en su gran mayoría. Por lo tanto, para k=1 se alcanzaron 21 aciertos, para k=3 fueron 27 aciertos y para k=5 se lograron 30 aciertos.
6. Conclusión y Trabajo Futuro
En este artículo se presentaron modelos de CNN para el de Redes Siamesas, con Relieve [41] para obtener una función distancia a utilizar en la búsqueda de imágenes de OBCs. Se realizaron los experimentos con el modelo (Mod1) alcanzando buenos resultados. Comparamos los mismos con los presentados en [41] y en [2], este segundo en relación a la inserción de nuevas imágenes a la base de datos y demostramos que Mod1 es un modelo que presenta una configuración más eficiente para búsquedas de este tipo.
Las tareas actuales y las actividades previstas para el futuro próximo son las siguientes:
Realizar experimentos sobre la BD completa, que está compuesta por alrededor de 15.000 imágenes de OBCs, cada una de ellas con entre 19 y 24 dibujos a mano. 
Experimentar con otras bases de datos de imágenes de líneas en escala de grises/blanco y negro.


Referencias
Chávez, Edgar, et al. Searching in metric spaces. ACM computing surveys (CSUR) 33.3: 273-321 (2001).
LI, Bang, et al. HWOBC-a handwriting oracle bone character recognition database. En Journal of Physics: Conference Series. IOP Publishing, (2020). p. 012050.
Aslandogan, Y. Alp, and Clement T. Yu. Techniques and systems for image and video retrieval. IEEE transactions on Knowledge and Data Engineering 11.1: 56-63 (1999).
Smeulders, Arnold WM, et al. Content-based image retrieval at the end of the early years. IEEE Transactions on pattern analysis and machine intelligence 22.12: 1349-1380 (2000).
Irena, V., Rachev, B., Vassilakopoulos, M. Optimization of the algorithm for image retrieval by color features. CompSysTech. (2006).
Sarfraz, M., Ahmad, R. Content-based image retrieval using multiple shape descriptors.  IEEE/ACS International Conference on Computer Systems and Applications (2007).
Pass, G., Ramin Z. Histogram refinement for content-based image retrieval. Proceedings Third IEEE Workshop on Applications of Computer Vision. WACV'96. IEEE, (1996).
Zhang, HongJiang, et al. Image retrieval based on color features: An evaluation study. Digital Image Storage and Archiving Systems. Vol. 2606. SPIE, (1995).
Zhang, Dengsheng, and Guojun Lu. Review of shape representation and description techniques. Pattern recognition 37.1: 1-19 (2004).
Traina, Agma JM, et al. Content-based image retrieval using approximate shape of objects. Proceedings. 17th IEEE Symposium on Computer-Based Medical Systems. IEEE, (2004).
Celebi, M. Emre, and Y. Alp Aslandogan. A comparative study of three moment-based shape descriptors. ITCC'05-Volume II. Vol. 1. IEEE, (2005).
Zhang, Dengsheng, and Guojun Lu. Content-based shape retrieval using different shape descriptors: A comparative study. ICME 2001. IEEE Computer Society, (2001).
Li, Shan, Moon-Chuen Lee, and Chi-Man Pun. Complex Zernike moments features for shape-based image retrieval. IEEE Transactions on Systems, Man, and Cybernetics-Part A: Systems and Humans 39.1: 227-237 (2008).
Novak, David, Michal Batko, and Pavel Zezula. Metric index: An efficient and scalable solution for precise and approximate similarity search. Information Systems 36.4 (2011).
Britos, L., Printista, M., Reyes, N. DSACL+-tree: A dynamic data structure for similarity search in secondary memory. ICSSA. Springer, Berlin, Heidelberg (2012).
Fukushima, Kunihiko. Neocognitron: A hierarchical neural network capable of visual pattern recognition. Neural networks 1.2: 119-130 (1988).
LeCun, Yann, et al. Gradient-based learning applied to document recognition. Proceedings of the IEEE 86.11: 2278-2324 (1998).
Krizhevsky, Alex, Sutskever, Ilya and Geoffrey E. Hinton. ImageNet classification with deep convolutional neural networks. In Proceedings of the NIPS. (2012).
Simonyan, Karen, and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556 (2014).
Lin, Min, Qiang Chen, and Shuicheng Yan. Network in network. arXiv preprint arXiv:1312.4400 (2013).
Huang, Gao, et al. Densely connected convolutional networks. Proceedings of the IEEE conference on computer vision and pattern recognition. (2017).
Szegedy, Christian, et al. Inception-v4, Inception-Resnet and the Impact of Residual Connections on Learning. Thirty-first AAAI conference on artificial intelligence. (2017).
He, Kaiming, et al. Deep Residual Learning for Image Recognition. Proceedings of the IEEE conference on computer vision and pattern recognition. (2016).
Fierro, Atoany N., et al. Redes Convolucionales Siamesas y Tripletas para la Recuperación de Imágenes Similares en Contenido. Información tecnológica 30.6: 243-254 (2019).
Melekhov, Iaroslav, Juho Kannala, and Esa Rahtu. Siamese network features for image matching. 23rd international conference on pattern recognition (ICPR). IEEE, (2016).
Hoffer, Elad, and Nir Ailon. Deep metric learning using triplet network. International workshop on similarity-based pattern recognition. Springer, Cham, (2015).
Wang, Yaqing, et al. Generalizing from a few examples: A survey on few-shot learning. ACM computing surveys (csur) 53.3: 1-34 (2020).
Lake, Brenden, et al. One shot learning of simple visual concepts. Proceedings of the annual meeting of the cognitive science society. Vol. 33. No. 33. (2011).
Pan, Sinno Jialin, and Qiang Yang. A survey on transfer learning. IEEE Transactions on knowledge and data engineering 22.10: 1345-1359 (2009).
Storkey, Amos. When training and test sets are different: characterizing learning transfer. Dataset shift in machine learning 30: 3-28 (2009).
Schultz, Matthew, and Thorsten Joachims. Learning a distance metric from relative comparisons. Advances in neural information processing systems 16 (2003).
Brian Kulis. Metric Learning: A Survey. Foundations and Trends in Machine Learning (FTML), 5(4):287–364, 2013.
Yang, Liu, and Rong Jin. Distance metric learning: A comprehensive survey. Michigan State Universiy 2.2 (2006): 4.
Kilian Q. Weinberger and Lawrence K. Saul. Distance Metric Learning for Large Margin Nearest Neighbor Classification. J. Mach. Learn. Res. 10, 207–244. 2012.
Chopra, S., Hadsell, R., & LeCun, Y. (2005). Learning a similarity metric discriminatively, with application to face verification. In Proceedings, CVPR, IEEE. (2005). 
Kulis, B. Metric Learning: A Survey. Now Foundations and Trends. (2013).
Guo, Jun, et al. "Building hierarchical representations for oracle character and sketch recognition." IEEE Transactions on Image Processing 25.1 (2015): 104-118.
Han, W.; Ren, X.; Lin, H.; Fu, Y.; Xue, X. Self-supervised learning of orc-bert augmentator for recognizing few-shot oracle characters. In Proceedings of the Asian Conference on Computer Vision, Kyoto, Japan, 30 November–4 December (2020).
Liu, M.; Liu, G.; Liu, Y.; Jiao, Q. Oracle-bone inscription recognition based on deep convolutional neural network. J. Image Graph. (2020), 8, 114–119. 
Zhang, Y.K.; Zhang, H.; Liu, Y.G.; Yang, Q.; Liu, C.L. Oracle character recognition by nearest neighbor classification with deep metric learning. In Proceedings of the 2019 International Conference on Document Analysis and Recognition (ICDAR), Sydney, Australia, 20–25 September (2019); pp. 309–314.
Pascal, A., Planas, A., Lederhos, F., Herrera, N. Extracción de Características utilizando Redes Siamesas y Aumentación aplicada a las Búsquedas por Similitud de Marcas de Ganado. CoNaIISI. Entre Ríos, Argentina (2022).






