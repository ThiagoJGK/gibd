---
aliases: [Cattle brand recognition using convolutional neural netwo...]
tags:
  - grupo/g4_imágenes
  - estado/util
  - tipo/documento
formato_original: docx
fecha_modificacion: 2022-11-15
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Trabajos parecidos/Brasileros/Cattle brand recognition using convolutional neural network and support vector machines.docx"
tamanio_bytes: 1830003
---

# Cattle brand recognition using convolutional neural network and support vector machines

Ruta interna: `GIBD/Trabajos parecidos/Brasileros/Cattle brand recognition using convolutional neural network and support vector machines.docx`

---

I. INTRODUCCIÓN
El desarrollo de herramientas computacionales para ayudar al análisis y reconocimiento de imágenes es tema de interés de los centros de investigación más renombrados del mundo. El uso de la computación para el análisis y reconocimiento de imágenes está en constante desarrollo, generando múltiples beneficios a la sociedad en las más diversas áreas del conocimiento. Particularmente en lo que se refiere al reconocimiento de marcas ganaderas, actividad tan tradicional y de gran relevancia socioeconómica para los países latinoamericanos, incluido Brasil, no existe un método específico y debidamente consolidado para tal fin. Para tener una dimensión de la importancia de esta actividad, según la Organización de las Naciones Unidas para la Agricultura y la Alimentación - FAO, entre los países productores, Brasil e India tienen los mayores rebaños, con Brasil en el 1° lugar, con un promedio de 209.215.666 cabezas [1].
La producción pecuaria tiene un papel importante en la formación social, y que aún hoy sigue siendo una actividad de gran importancia en las expresiones culturales asociadas a ella, ya que está asociada a la cultura y modo de vida del campo, además del papel en la afirmación o construcción de identidades individuales o grupales [2].
El uso de marcas o signos en el ganado presupone el reconocimiento público de su propiedad por parte de un individuo o grupo. Utilizadas desde el inicio de la colonización ibérica en América, el inicio de su institucionalización se produjo a partir de la inscripción en organismos, siendo oficiales, ciertamente portadores de legitimidad pública [2]. A estos registros le siguen normas que buscan, además de oficializar el marcaje mismo, orientar la forma y plazo a realizar, discriminar la forma de registro, establecer valores para el mismo, la construcción de los hierros y la tributación gubernamental. En general, los registros de marcas de ganado se componen de libros con los dibujos de las marcas y la identificación de su propietario.
En Brasil, los intentos e inversiones para mejorar el sistema de registro de marcas de ganado siempre han sido objeto de controversia, debido a la resistencia de los ganaderos. Gran parte de este miedo está asociado al miedo a perder las marcas familiares y el significado que han adquirido con el tiempo.
Actualmente, los registros de marcas en Brasil se realizan en los municipios, en general, sin una sistematización más efectiva y sin necesidad de renovación.
Dado el contexto presentado, este trabajo tiene como objetivo presentar y evaluar una herramienta que realice el reconocimiento automático de marcas de ganado, con el propósito de reemplazar el control manual de marcas de ganado que se realiza actualmente, con el fin de reducir potencialmente la posibilidad de registros duplicados, reducir el tiempo de espera para el registro de nuevas marcas, mejorar la gestión gubernamental del cobro de las marcas bajo su responsabilidad y asistir a las autoridades de seguridad en la lucha contra los delitos de abigeato.
El trabajo se centró en el desarrollo de algoritmos responsables del reconocimiento de marcas de ganado del Municipio de São Francisco de Assis, Rio Grande do Sul, Brasil. De esta forma, los empleados del Sector de Registro de Marcas Pecuarias y del Centro de Procesamiento de Datos del municipio validaron la herramienta propuesta.
El resto del trabajo está organizado de la siguiente manera: la sección 2 describe trabajos relacionados.
La sección 3 presenta los materiales y métodos utilizados para este trabajo. La sección 4 describirá los resultados y discusiones obtenidos a través de la aplicación de los métodos propuestos. Finalmente, en la sección 5 se describirán las consideraciones finales.
II. OBRAS RELACIONADAS
En general, no se encontraron obras en la revisión de literatura que reporta el uso de redes neuronales herramientas convolucionales para el reconocimiento de imágenes de marcas de ganado.
Sanchez et al [3] presenta una herramienta para el reconocimiento de marcas de ganado que utiliza momentos de Hu y Legendre para extraer características de imágenes en escala de grises y un clasificador de k-vecinos más cercanos (k-NN). Los autores utilizaron los momentos de Hu y Legendre para extraer características que no fueran susceptibles de rotación, traslación y transformaciones de escala. El porcentaje máximo de clasificación correcta presentado por los autores fue del 99,3%, pero con una disminución significativa en la precisión a medida que aumentaba el número de imágenes clasificadas. Otro resultado presentado fue el tiempo de procesamiento de la clasificación. Debido a que se usó un clasificador k-NN, para cada nuevo objeto a clasificar, los datos de entrenamiento se usan para verificar qué objetos en esta base de datos son más similares al nuevo objeto a clasificar. El objeto se clasifica en la clase más común a la que pertenecen los objetos más similares a él.
Así, la clasificación se hace por analogía. No se crea ningún modelo de clasificación. En cambio, con cada nuevo objeto a clasificar, los datos de entrenamiento se escanean, por lo que el clasificador propuesto se vuelve computacionalmente costoso.
A diferencia del trabajo presentado por Sanchez et al [3], el trabajo que aquí se propone pretende presentar resultados que puedan ser generalizados o replicados, utilizando técnicas de punta en cuanto a extracción de características y clasificación estadística de imágenes digitales, tales como Neural Convolucional (CNN) y Máquinas de Vectores Soporte (SVM), buscando un rendimiento superior en el reconocimiento de imágenes en bases de datos con mayor número de registros, pero con menor costo computacional y tiempo de procesamiento.
Los trabajos encontrados en la literatura para el reconocimiento y clasificación de imágenes que utilizan descriptores o filtros para la extracción de características, seguido de un paso de cuantización y agrupamiento y, finalmente, un paso de clasificación, se dividen en dos categorías: características únicas y de dos o más etapas. algoritmos [4].
La posibilidad de entrenar redes neuronales con múltiples capas intermedias plantea la aparición de varios algoritmos agrupados en un área conocida como aprendizaje profundo.
El objetivo principal de los algoritmos que utilizan dos o más etapas es aprender no solo a distinguir clases en función de descriptores artificiales, sino también aprender los descriptores en sí mismos en función de los datos sin procesar, en el caso de las imágenes, los valores de píxeles en sí mismos. 5].
Las redes neuronales convolucionales se han utilizado durante varios años en el reconocimiento de imágenes, habiendo obtenido un gran éxito en el reconocimiento de caracteres en el trabajo de Cun et al [6].
Estudios más recientes que utilizan redes neuronales convolucionales también conocidas como Deep Convolutional Neural Networks (CNN) ha alcanzado el nuevo estado del arte en reconocimiento de objetos en bases CIFAR-10 y NORB [7].
En general, las CNN se entrenan de forma supervisada, pero los estudios sugieren que el preentrenamiento de las CNN con filtros obtenidos de forma no supervisada presenta un mejor resultado [8].
En el trabajo realizado por Sermanet et al [9] se presenta un framework que utiliza CNN para realizar reconocimiento, localización y detección de imágenes, siendo ganador del campeonato ImageNet Large Scale Visual Recognition Challenge 2013 (ILSVRC2013). La base de datos ILSVRC2013 consta de 1,2 millones de imágenes divididas en 1000 categorías.
Una característica importante de CNN es la capacidad de ser reutilizado y refinado para diferentes bases de imágenes. En el trabajo realizado por Razavian et al [10] se utilizó una CNN preentrenada llamada Overfeat [9] para extraer un descriptor de diferentes bases de imágenes en las que la CNN no fue entrenada originalmente. En este caso, los descriptores se clasifican utilizando un clasificador lineal SVM. Los resultados demuestran un desempeño compatible con el estado del arte, incluso comparándolo con algoritmos que utilizan imágenes segmentadas manualmente, procedimiento que no es necesario al utilizar CNN, y entrenado específicamente sobre la base analizada.
El uso del aprendizaje profundo también se describe en el trabajo de Constante et al [11], quienes utilizaron una red neuronal de tres capas con entrada mediante retropropagación. En este trabajo, se utilizó el método para clasificar las fresas, y alcanzó resultados de reconocimiento de 92,5% en la categoría “Extra”; 90% en la categoría “Consumo”; el 90% en la categoría “Materia prima”; y 100% en la categoría “Objetos extraños”.

III. MATERIALES Y MÉTODOS
Las imágenes de las marcas de ganado utilizadas en este trabajo fueron proporcionadas por el Municipio de São Francisco de Assis - RS. Se utilizaron doce imágenes de marcaje de ganado, cada una compuesta por 45 subimágenes (muestras), totalizando 540 muestras de las imágenes originales, pero con variaciones en tamaño y orientación, con el objetivo de identificar patrones con la mayor independencia posible de estos factores. Las imágenes se pusieron a disposición en alta resolución en formato Portable Network Graphics con un tamaño de 600 x 600 píxeles.
Para la implementación de la herramienta propuesta, además del almacenamiento del banco de imágenes, procesamiento de los algoritmos y visualización de los resultados, se utilizó una computadora personal con tarjeta de video que soporta la plataforma de cómputo paralelo CUDA con capacidad de cómputo versión 5.0. Además, el software MatLab se utilizó con las bibliotecas Neural Network, Parallel Computing and Statistics y Machine Learning y el modelo de red convolucional preentrenado obtenido de la biblioteca de código abierto de VLFeat.org [12].
El método propuesto consta de seis pasos, que son: selección del banco de imágenes; selección del modelo CNN pre-entrenado; preprocesamiento de imágenes y aplicación de CNN; extracción de características de las imágenes; entrenamiento y clasificación de imágenes a través de Máquinas de Vectores de Soporte; y, finalmente, evaluación de los resultados de la clasificación. La Fig. 1 ilustra el diagrama de flujo resumido del método propuesto.

Figura 1. Diagrama de flujo resumido del método propuesto.

A. Selección del banco de imágenes
Las marcas utilizadas en el trabajo se ilustran en la Fig. 2. El código de las marcas, los propietarios y el número de muestras de cada marca se pueden obtener de la Tabla I.

Figura 2. Imágenes de marcas de ganado utilizadas en la obra.

Tabla I
Marcas, Propietarios y total de muestras por marca


Las redes neuronales convolucionales (CNN) son arquitecturas inspiradas biológicamente capaces de ser entrenadas y aprender representaciones invariantes a escala, traslación, rotación y transformaciones relacionadas [13]. Las CNN comprenden uno de los tipos de algoritmos en el área conocida como inclinación profunda y están diseñadas para usarse con datos en dos dimensiones, lo que las convierte en un buen candidato para resolver problemas relacionados con el reconocimiento de imágenes [14]. Por definición, una arquitectura profunda es una estructura jerárquica de múltiples pasos, donde cada paso está formado por una red neuronal de al menos 3 capas, y cada paso es entrenado por backpropagation. La Fig. 3 ilustra el modelo general de una arquitectura CNN.


Figura 3. Modelo general de una arquitectura CNN.

B. Selección del modelo de CNN preentrenado
Las redes neuronales convolucionales tienen una propiedad aún poco explorada conocida como transferencia de conocimiento. Esta propiedad se refiere al hecho de que una CNN se puede entrenar sobre una base de imágenes A (por lo tanto, sus pesos se ajustan para la clasificación de la base A) y los pesos de aprendizaje (y los filtros en las CNN) se consideran lo suficientemente genéricos para ser utilizados en el entrenamiento una nueva base B. Utilizando este concepto, se seleccionó el modelo CNN de la biblioteca de código abierto VLFeat, cuyos datos utilizados para el pre-entrenamiento de la red neuronal convolucional utilizada en el trabajo presentado fueron obtenidos del archivo “imagenet-caffe-alex.mat”. El uso del modelo CNN previamente entrenado mencionado anteriormente no influyó directamente en la tasa de reconocimiento de las imágenes de marca de ganado.

C. Preprocesamiento de imágenes y aplicación de CNN
El método adoptado consiste en una red neuronal con cinco capas convolucionales. Si la imagen está en escala de grises, se realiza un preprocesamiento, en el que la imagen se replica 3 veces para crear una imagen RGB. La primera capa convolucional tiene los 3 canales de color de imagen (RGB) como entrada. Cada convolución realiza la aplicación de la función de activación no lineal ReLu y la reducción a través de Maxpooling. Las últimas capas están compuestas por neuronas totalmente conectado. Para el entrenamiento de CNN se utilizó la función Softmax y el algoritmo backpropagation. La Fig. 4 presenta la arquitectura de red neuronal convolucional propuesta.


Figura 4. Arquitectura de la red neuronal convolucional propuesta.

D. Extracción de características de la imagen
El conjunto de filtros aprendido por CNN durante el entrenamiento es responsable de detectar las características en la nueva imagen en el momento de una consulta. En el primer nivel de filtros es posible observar algunas líneas y orientaciones utilizadas para esta detección. Fig. 5 muestra los filtros aprendidos en la primera capa convolucional utilizando el espacio de color RGB. Hay 96 conjuntos individuales que representan los 96 filtros utilizados en esta capa. Puede ver cómo se resaltan las áreas con protuberancias horizontales, verticales y diagonales después de realizar la primera convolución.

Figura 5. Filtros de la 1ª capa convolucional del experimento realizado.

Figura 6. Ilustración del algoritmo de extracción de características desarrollado.

E. Entrenamiento y clasificación de imágenes usando Máquinas de Vectores de Soporte
El modelo de aprendizaje automático adoptado en el trabajo presentado fue el clasificador supervisado Support Vector Machine (SVM). Support Vector Machine es un algoritmo de clasificación conocido por tener éxito en una amplia variedad de aplicaciones. Las SVM son uno de los enfoques más populares para el modelado y la clasificación de datos. Sus ventajas incluyen una excelente capacidad de generalización, que se refiere a la capacidad de clasificar correctamente las muestras que no están dentro del espacio de características utilizado para el entrenamiento [15]. Dadas dos clases y un conjunto de puntos a estas clases, la SVM determina el hiperplano que separa los puntos para colocar la mayor cantidad de puntos de la misma clase en el mismo lado, maximizando la distancia de cada clase a este hiperplano, por lo que se denomina clasificador de margen máximo [16]. De hecho, un gran margen entre los valores correspondientes a los puntos de los dos subconjuntos de datos implica un riesgo minimizado de generalización del clasificador.
Las SVM se utilizan para clasificar y reconocer patrones en diferentes tipos de datos, y se utilizan en varias aplicaciones, como reconocimiento facial, diagnóstico clínico, supervisión de procesos industriales, procesamiento y análisis de imágenes [17].
En la herramienta propuesta en este trabajo se utilizó el clasificador luego de extraer características de las marcas pertenecientes a diferentes conjuntos de muestras. En el aprendizaje supervisado, dado un conjunto de ejemplos (X1, X2) en los que X1 representa un ejemplo y X2 su clasificación, se debe reproducir un clasificador capaz de predecir a qué clase pertenecen los nuevos datos, realizando así el proceso de entrenamiento. En este contexto, utilizamos el método de validación cruzada [15], en la que las imágenes se dividieron aleatoriamente en 2 partes, donde una de estas partes se utilizó para el entrenamiento y la otra para la validación, de modo que no hubo polarización de los resultados. El resultado final es la media del resultado obtenido en la validación. La división porcentual utilizada fue 30% para entrenamiento y 70% para validación.

F. Evaluación de los resultados de clasificación - Matriz de confusión
La matriz de confusión contiene información relacionada con las clasificaciones realizadas mediante la aplicación de un clasificador. El desempeño de los clasificadores se evalúa frecuentemente a través de los datos tomados de esta matriz [18]. 
La matriz de confusión de un clasificador indica el número de clasificaciones correctas frente a las predicciones realizadas para cada caso, sobre un conjunto de ejemplos. En esta matriz, las líneas representan los casos reales y las columnas las predicciones realizadas por el modelo. A través de la matriz de confusión es posible obtener información sobre el número de imágenes correctamente clasificadas e incorrectamente clasificadas para cada conjunto de muestras. 
Esta matriz es de tipo AxA, donde A es el número de categorías a las que se les aplica el clasificador, en el caso del experimento realizado son 12 marcas, por lo tanto, la matriz de confusión es de tamaño 12x12.

IV. RESULTADOS Y DISCUSIONES
A través de los resultados obtenidos, fue posible evaluar el método propuesto. La evaluación de los resultados de los experimentos se realizó en base a la tasa de reconocimiento obtenida a partir de la matriz de confusión generada a partir de la clasificación realizada en la etapa de validación. Además, también se verificó el tiempo total de procesamiento del método propuesto.
La Fig. 7 presenta la matriz de confusión para el mejor resultado obtenido en los experimentos, cuya tasa de reconocimiento alcanzó el 93,28%. Se puede observar a través del análisis de la diagonal principal que el índice de aciertos se destaca en cuatro marcas, “802”, “812”, “815” y “821”, en las que el porcentaje de aciertos alcanza el 100%, correspondiente a 31 aciertos. También es posible observar que las marcas que presentaron los índices más bajos de clasificación correcta fueron “813” y “805”, con porcentajes de 77,42% y 80,64%, en ese orden. Las otras marcas, “803”; "804"; "811"; "814"; “822” y “1093” lograron una tasa de clasificación correcta del 96,77%; 93,55%; 90,32%; 90,3%; 96,77% y 93,55%, respectivamente.
La hipótesis de clasificación errónea de marcas de ganado como se muestra en la matriz de confusión también puede estar asociada con la complejidad de las muestras, ya que algunas imágenes tienen características similares entre sí.
En general, las muestras de imágenes de marcas con mayor poder descriptivo y mejor calidad clasificaron correctamente más imágenes, ya que abarcan más características en comparación con las marcas con menor calidad de muestra y, en consecuencia, se extraen menos características. La capacidad de reconocer patrones de una imagen sobre un conjunto de imágenes depende de la cantidad de información que se conoce a priori sobre el objeto en cuestión.

Figura 7. Matriz de confusión obtenida en el paso de validación.
Otro dato importante a presentar, como resultado del análisis de la Fig. 7, es el bajo número de falsos positivos y falsos negativos, observado fuera de la diagonal principal de la matriz de confusión, lo que se debe principalmente a la capacidad del método propuesto para extraer características de las imágenes, incluso en situaciones adversas, en las que las imágenes presentan diferentes características: tamaños, formas, escalas, orientaciones, distorsiones, ruido, diferentes colores y diferentes contextos de fondo. En este tipo de experimento, el ruido en las imágenes puede afectar la precisión de la clasificación.
Teniendo en cuenta la matriz de confusión presentada en la Fig. 7, se observa que a través de la suma de la diagonal principal se obtiene el número total de marcas de ganado correctamente clasificadas, más precisamente 347, mientras que la suma de los demás valores es equivalente a las marcas de ganado mal clasificadas, es decir, 25.
La Fig. 8 muestra un cuadro comparativo entre marcas correctamente clasificadas en función del número de muestras utilizadas en la validación.

Figura 8. Proporción de marcas correctamente clasificadas en función del número de muestras.

Las barras azules muestran las marcas de ganado clasificadas correctamente (exactitud), y las barras rojas muestran el total de muestras de cada marca. De las 12 marcas analizadas, 4 presentaron clasificaciones 100% correctas, en este caso las marcas “802”; "812"; “815” y “821”. Las marcas de ganado que tuvieron menor tasa de acierto fueron “813” y “805”, con 24 y 25 imágenes correctamente clasificadas, respectivamente. También se observa el alto índice de clasificaciones correctas, incluso en las marcas que no alcanzan el 100% de precisión, como es el caso de las marcas “803”; "804"; "811"; "814"; “822” y “1093”.
El tiempo de procesamiento del algoritmo en función del número de muestras de marcas de ganado se muestra en la Fig. 9.
Los tiempos de procesamiento del método propuesto para la clasificación se midieron en cinco grupos de muestras. Cada grupo contiene 108, 216, 324, 432 y 540 imágenes. Los tiempos de procesamiento para clasificar las imágenes en cada grupo fueron 6.206s; 8,01 s; 9.488s; 11.295s y 12.716s, en ese orden.
Al analizar el gráfico ilustrado en la Fig. 9 se puede observar que el tiempo de procesamiento observado en el eje y varía directamente proporcional al aumento en el número de muestras de marcas de ganado clasificadas en el eje x, es decir, se observa un patrón de crecimiento lineal de la función, aunque la tasa de crecimiento no es exactamente un valor constante.

Figura 9. Tiempo de procesamiento del algoritmo en función del número de muestras.

Los resultados obtenidos con los experimentos realizados con 12 marcas de ganado y 540 muestras de imágenes, utilizadas tanto para entrenamiento como para validación, alcanzaron una tasa de reconocimiento del 93,28%, una tasa de error del 6,72% y un tiempo total de procesamiento de 12.716 segundos.
La tasa de reconocimiento se obtuvo calculando la media aritmética de las marcas clasificadas correctamente en la matriz de confusión, y el tiempo total de procesamiento se obtuvo mediante el software MatLab, que al final del procesamiento del código detalla la velocidad de procesamiento del algoritmo.

V. CONCLUSIÓN
En este trabajo se presentó un método automatizado para el reconocimiento de marcas de ganado. El proyecto se desarrolló entre dos instituciones; la Municipalidad de São Francisco de Assis y UNIPAMPA.
Los experimentos realizados en este trabajo utilizaron una Red Neural Convolucional (CNN) para la extracción de características y un clasificador SVM supervisado. En CNN, se creó una red convolucional completa utilizando imágenes transformadas en formato de color RGB como entrada. Todos los experimentos se realizaron sobre la base de marcas de ganado proporcionadas por la Municipalidad de São Francisco de Assis.
El uso del método propuesto presentó una precisión promedio de 93,28% y un tiempo de procesamiento del algoritmo de 12,716 segundos para 12 marcas evaluadas, en un total de 540 muestras utilizadas para entrenamiento y validación.
El método utilizado realizó de manera efectiva y eficiente el reconocimiento de diferentes marcas de ganado, incluso utilizando una CNN pre-entrenada, pero su principal limitación fue la necesidad de una gran cantidad de imágenes de muestra para entrenar al clasificador, ya que la CNN necesita estas imágenes para la función. extracción en capas convolucionales.
Como trabajo futuro, el objetivo es ampliar la base de imágenes tomadas para entrenamiento y validación, con el fin de obtener nuevas medidas de precisión y desempeño del método propuesto, buscando la disponibilidad de una herramienta debidamente validada y consolidada para las agencias gubernamentales responsables del registro y control de marcas ganaderas.

REFERENCIAS
[1] SEPLAN. Secretaria do Planejamento e Desenvolvimento Regional – Governo do Estado do Rio Grande do Sul, Brasil. 2015. URL:
http://www.scp.rs.gov.br/atlas/conteudo.asp?cod_menu_filho=819&cod_menu=817&tipo_menu=ECONOMIA&cod_conteudo=1580
[2] ARNONI, R. “Os Registros e Catálogos de Marcas de Gado da Região Platina.” Pelotas: Revista Memória em Rede da UFPEL, 2013.
[3] SANCHEZ, G.; RODRIGUEZ, M. “Cattle Marks Recognition by Hu and Legendre Invariant Moments”. ARPN Journal of Engineering and Applied Sciences, Vol. 11, Nº 1, 2016.
[4] JARRETT, K.; KAVUKCUOGLU, K.; LECUN, Y. “What Is The Best Multi-Stage Architecture for Object Recognition?” IEEE 12th International
Conference on Computer Vision, 2009.
[5] JURASZEK, G. “Reconhecimento de Produtos por Imagem Utilizando Palavras Visuais e Redes Neurais Convolucionais”. Joinville: UDESC, 2014.
[6] CUN, L.; BOSER, B.; DENKER, J. S.; HENDERSON, D.; HOWARD, R. E.; HUBBARD, W.; JACKEL, L. D. “Handwritten Digit Recognition with a Back-Propagation Network”. In: Advances in Neural Information Processing Systems. [S.l.]: Morgan Kaufmann, 1990. p. 396-404.
[7] CIRESAN, D.; MEIER, U.; SCHMIDHUBER, J. “Multi-Columm Deep Neural Networks for Image Classification”. In: Proceedings of the 25th IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2012). [S.l.:s.n.], 2012. p. 3642-3649.
[8] KAVUKCUOGLU, K.; SERMANET, P.; BOUREAU, Y. lan; GREGOR, K.; MATHIEU, M.; LECUN, Y. Learning Convolutional Feature Hierarchies for Visual Recognition. 2012.
[9] SERMANET, P.; EIGEN, D.; ZHANG, X.; MATHIEU, M.; FERGUS, R.; LECUN, Y. Overfeat: Integrated Recognition, Localization and Detection
Using Convolutional Networks. CoRR, abs/1312.6229, 2013. 
[10] RAZAVIAN, A. S.; AZIZPOUR, H.; SULLIVAN, J.; CARLSSON, S. CNN Features Off-the-Shelf: An Astounding Baseline for Recognition. CoRR,
abs/1403.6382, 2014.
[11] CONSTANTE, P.; GORDÓN, A.; CHANG, O.; PRUNA, E.; ESCOBAR, I.; ACUÑA, F. “Artificial Vision Techniques for Strawberry's
Industrial Classification”. IEEE Latin America Transactions, Vol. 14, Nº 6, 2016.
[12] VLFEAT. Biblioteca Open Source VLFeat. 2016. URL: http://www.vlfeat.org/matconvnet/models/beta16/imagenet-caffe-alex.mat
[13] LECUN, Y.; KAVUKCUOGLU, K.; FARABET, C. “Convolutional Networks and Applications in Vision”. In: Circuits and Systems (ISCAS),
Proceedings of 2010 IEEE International Symposium on. [S.l.: s.n.], 2010. p. 253-256.
[14] AREL, I.; ROSE, D.; KARNOWSKI, T. “Deep Machine Learning – A New Frontier in Artificial Intellingence Research [research frontier]”.
Computational Intelligence Magazine, IEEE, v. 5, n. 4, p. 13-18, 2010. ISSN 1556-603X.
[15] TEIXEIRA, A. “Desenvolvimento de uma Interface Gráfica para Classificadores de Imagem”. 2016. URL:
https://repositorio.ipcb.pt/bitstream/10400.11/1155/1/disserta%C3%A7ao.pdf
[16] LU, H.; HUANG, Y. CHEN, Y.; YANG, D. “Real-Time Facial Expression Recognition Based on Pixel Pattern-Based Texture Feature”. In:
Proc. Electronic Letters, pp. 916-918, 2007.
[17] TCHANGANI, A. “Support Vector Machines: A Tool for Pattern Recognition and Classification”. Studies in Informatics & Control Journal 14: 2. 99-110, 2005.
[18] KOHAVI, R.; PROVOST, F. “Glossary of Terms”. Machine Learning, 30(2-3), 271-274, 1998.
[19] VAPINIK, N. “The Nature of Statistical Learning Theory”. New York: Springer, 1995.
[20] KIM, K. I.; JUNG, K.; PARK, S. H.; KIM, H. J. “Support Vector Machines for Texture Classification”. IEEE Trans, PAMI, 2002.
[21] LU, H.; HUANG, Y. CHEN, Y.; YANG, D. “Real-Time Facial Expression Recognition Based on Pixel Pattern-Based Texture Feature”. In:
Proc. Electronic Letters, pp. 916-918, 2007.
[22] LOVELL, B.; WALDER, C. “Support Vector Machines for Business Applications”. Business Applications and Computational Intelligence, Idea
Group Publishers, 2006.

CITACIÓN
APA
Silva, C., Welfer, D., Gioda, F. P., & Dornelles, C. (2017). Cattle brand recognition using convolutional neural network and support vector machines. IEEE Latin America Transactions, 15(2), 310-316.
ISO 690
SILVA, Carlos, et al. Cattle brand recognition using convolutional neural network and support vector machines. IEEE Latin America Transactions, 2017, vol. 15, no 2, p. 310-316.
MLA
Silva, Carlos, et al. "Cattle brand recognition using convolutional neural network and support vector machines." IEEE Latin America Transactions 15.2 (2017): 310-316.

BibTeX
@article{silva2017cattle,
  title={Cattle brand recognition using convolutional neural network and support vector machines},
  author={Silva, Carlos and Welfer, Daniel and Gioda, Francisco Paulo and Dornelles, Claudia},
  journal={IEEE Latin America Transactions},
  volume={15},
  number={2},
  pages={310--316},
  year={2017},
  publisher={IEEE}
}