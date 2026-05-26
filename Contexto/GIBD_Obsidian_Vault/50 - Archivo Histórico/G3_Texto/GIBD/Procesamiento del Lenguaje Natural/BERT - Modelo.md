---
aliases: [BERT - Modelo]
tags:
  - grupo/g3_texto
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2023-04-21
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Procesamiento del Lenguaje Natural/BERT - Modelo.docx"
tamanio_bytes: 320468
---

# BERT - Modelo

Ruta interna: `GIBD/Procesamiento del Lenguaje Natural/BERT - Modelo.docx`

---

El modelo BERT está constituido por varias capas de redes neuronales que se apilan una encima de la otra para formar una arquitectura profunda. La arquitectura de BERT se basa en un tipo de red neuronal llamada "Transformers", que es una arquitectura de atención basada en modelos de lenguaje autoregresivos.
La arquitectura de BERT consta de dos partes principales:
Codificador: Esta parte del modelo se encarga de procesar las secuencias de códigos de tokens de entrada. Consiste en una serie de capas de codificación de tokens, donde cada capa tiene dos subcapas: una capa de atención multi-cabeza y una capa de alimentación hacia adelante. La capa de atención multi-cabeza permite al modelo capturar las relaciones de dependencia a largo plazo entre las palabras en las secuencias de tokens, mientras que la capa de alimentación hacia adelante aplica transformaciones no lineales a los vectores de representación de los tokens.
Clasificador: Después del codificador, el modelo BERT puede tener una o más capas de clasificación adicionales que se utilizan para tareas específicas, como clasificación de texto, extracción de entidades o tareas de generación de texto. Estas capas de clasificación adicionales toman las representaciones de los tokens obtenidas del codificador y las utilizan para producir salidas específicas de la tarea.
En términos de tamaño, BERT tiene una cantidad significativa de parámetros, ya que es un modelo profundo con muchas capas. El tamaño exacto del modelo puede variar dependiendo de la implementación específica de BERT utilizada, pero puede llegar a tener cientos de millones de parámetros. Estos parámetros se ajustan durante el proceso de entrenamiento de BERT utilizando grandes conjuntos de datos de entrenamiento para aprender las representaciones de palabras y las capacidades de modelado de lenguaje necesarias para tareas de procesamiento del lenguaje natural.


Un modelo autoregresivo es un tipo de modelo de lenguaje en el que se genera una secuencia de texto de manera secuencial, una palabra o token a la vez, basándose en las palabras o tokens previamente generados en la secuencia. En otras palabras, el modelo genera cada palabra o token condicionado a las palabras o tokens que lo preceden en la secuencia.
En el contexto de BERT, aunque no es un modelo autoregresivo, se utiliza una técnica similar para el preentrenamiento. Durante el preentrenamiento de BERT, se oculta una parte de la secuencia de entrada y se entrena al modelo para que prediga las palabras o tokens ocultos en función del contexto de las palabras o tokens visibles circundantes. Este proceso de predicción autoregresiva permite a BERT aprender las representaciones de palabras y las relaciones de dependencia a largo plazo en el contexto de las secuencias de texto durante el preentrenamiento, lo que luego puede ser utilizado en tareas específicas de procesamiento del lenguaje natural durante el ajuste fino del modelo. Sin embargo, durante la inferencia o uso en tareas específicas, BERT no es autoregresivo, ya que no genera texto secuencialmente, sino que procesa las secuencias de entrada completas en paralelo para producir representaciones de palabras o tokens y realizar tareas específicas como clasificación o extracción de información.


Aquí tienes un ejemplo simplificado del proceso de codificación de palabras o tokens en BERT:
Supongamos que tenemos la siguiente oración de entrada: "El perro está jugando en el parque."
Tokenización: La oración se divide en palabras o tokens individuales. Por ejemplo, "El", "perro", "está", "jugando", "en", "el", "parque", "." serían tokens individuales.
Codificación de Tokens: Cada token se convierte en un vector de números enteros que representa su identificador único en el vocabulario de BERT. Estos identificadores se utilizan como entrada para el modelo. Por ejemplo, "El" podría ser codificado como 101, "perro" como 102, "está" como 103, y así sucesivamente.
Adición de Tokens Especiales: Se añaden tokens especiales al principio y al final de la secuencia para indicar el inicio y el fin de la oración. Por ejemplo, se puede añadir el token [CLS] al principio de la secuencia y el token [SEP] al final de la secuencia.
La secuencia de tokens codificados y con los tokens especiales agregados quedaría así:
[CLS], 101, 102, 103, 104, 105, 101, 106, 102, [SEP]
Donde [CLS] representa el inicio de la secuencia, [SEP] indica el final de la secuencia, y los números enteros representan los identificadores de los tokens codificados correspondientes a las palabras en la oración.
Esta secuencia de tokens codificados es la entrada que se alimenta al codificador de BERT para obtener las representaciones de palabras o tokens a través de las capas de atención y alimentación hacia adelante. Estas representaciones de palabras o tokens aprendidas por BERT durante el entrenamiento se utilizan posteriormente para tareas específicas como clasificación de texto o extracción de entidades en el proceso de ajuste fino del modelo.


Claro, te puedo explicar brevemente cómo se procesa la secuencia de tokens en el codificador de BERT, incluyendo la capa de atención multi-cabeza y la capa de feed-forward.
Capa de Atención Multi-cabeza: La capa de atención multi-cabeza en BERT procesa la secuencia de tokens para capturar las relaciones de dependencia y la información contextual. La atención es un mecanismo que permite que el modelo se enfoque en diferentes partes de la secuencia de entrada durante la codificación. La capa de atención multi-cabeza divide la representación de entrada en múltiples cabezas o subespacios, y luego aplica la atención en paralelo en cada cabeza. Cada cabeza tiene su propia matriz de pesos de atención aprendidos durante el entrenamiento, lo que permite al modelo capturar relaciones de dependencia a diferentes escalas y niveles de abstracción.
Capa de Feed-Forward: Después de la capa de atención multi-cabeza, la salida se pasa a través de una capa de feed-forward, que es una red neuronal de dos capas completamente conectadas con una función de activación intermedia. Esta capa de feed-forward procesa la representación de cada token de manera independiente y no tiene en cuenta las relaciones de dependencia entre tokens. Ayuda a capturar patrones locales en la secuencia y a transformar la representación de cada token en un espacio de características de mayor dimensión.
Residuales y Normalización: Para mejorar el flujo de información y facilitar el entrenamiento, se utilizan conexiones residuales y normalización en las capas de atención multi-cabeza y de feed-forward. Las conexiones residuales permiten que la información fluya directamente desde la entrada a la salida de cada capa, lo que ayuda a mitigar el problema de desvanecimiento de gradientes y permite un entrenamiento más profundo. La normalización, como la normalización por lotes, se utiliza para normalizar las activaciones en cada capa y estabilizar el proceso de entrenamiento.
El proceso de atención multi-cabeza y de feed-forward se repite en varias capas del codificador de BERT, lo que permite capturar información contextual y aprender representaciones de palabras o tokens en niveles de abstracción cada vez más profundos. Las representaciones de palabras o tokens resultantes se utilizan posteriormente en tareas específicas durante el ajuste fino del modelo.


La arquitectura de BERT base, que es la versión original de BERT propuesta por Devlin et al. en su artículo "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", consta de las siguientes características principales:
Capa de entrada: La entrada de BERT base consta de secuencias de tokens de longitud fija, que se obtienen mediante la tokenización de texto. Cada token se codifica como un vector de características llamado "embedding de token" que es aprendido durante el entrenamiento. Además, se agregan dos vectores especiales llamados "embedding de segmento" para indicar qué tokens pertenecen a la misma oración o segmento, y "embedding de posición" para indicar la posición relativa de cada token en la secuencia.
Codificador: BERT base utiliza un codificador basado en la arquitectura del Transformer, que consta de múltiples capas de atención multi-cabeza y capas de feed-forward. El número total de capas en BERT base es de 12 capas en el codificador.
Atención multi-cabeza: Cada capa de atención multi-cabeza en BERT base tiene 12 cabezas de atención, lo que significa que se aplican 12 matrices de pesos de atención aprendidos en paralelo para capturar relaciones de dependencia a diferentes escalas y niveles de abstracción.
Capa de feed-forward: Después de la capa de atención multi-cabeza, la salida se pasa a través de una capa de feed-forward, que consta de dos capas completamente conectadas con una función de activación intermedia.
Conexiones residuales y normalización: BERT base utiliza conexiones residuales para permitir que la información fluya directamente desde la entrada a la salida de cada capa, lo que ayuda a mitigar el problema de desvanecimiento de gradientes. Además, se utiliza normalización, como la normalización por lotes, para normalizar las activaciones en cada capa y estabilizar el proceso de entrenamiento.
Tamaño de las capas ocultas: El tamaño de las capas ocultas en BERT base es de 768 neuronas en todas las capas del codificador, incluyendo las capas de atención multi-cabeza y las capas de feed-forward.
Es importante tener en cuenta que BERT base es solo una de las variantes de BERT, y existen otras variantes con tamaños de modelo más grandes, como BERT large, que tiene 24 capas en el codificador y un tamaño de capa oculta de 1024 neuronas, lo que resulta en un modelo más potente pero también más costoso en términos de recursos computacionales.
