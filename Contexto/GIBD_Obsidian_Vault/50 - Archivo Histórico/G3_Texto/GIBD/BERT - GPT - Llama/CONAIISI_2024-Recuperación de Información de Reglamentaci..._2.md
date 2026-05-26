---
aliases: [CONAIISI_2024-Recuperación de Información de Reglamentaci...]
tags:
  - grupo/g3_texto
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2024-09-16
origen_zip: GIBD-20260521T205218Z-3-008.zip
ruta_interna: "GIBD/BERT - GPT - Llama/Artículo CONAISI 2024/CONAIISI_2024-Recuperación de Información de Reglamentación Academica en Español utilizando Modelos del Lenguaje Natural.docx"
tamanio_bytes: 54515
---

# CONAIISI_2024-Recuperación de Información de Reglamentación Academica en Español utilizando Modelos del Lenguaje Natural

Ruta interna: `GIBD/BERT - GPT - Llama/Artículo CONAISI 2024/CONAIISI_2024-Recuperación de Información de Reglamentación Academica en Español utilizando Modelos del Lenguaje Natural.docx`

---


Recuperación de Información de Reglamentación Académica en Español utilizando Modelos del Lenguaje Natural


Andrés J. Pascal 					Maximiliano Olivera
FRCU - Universidad Tecnológica Nacional 	FRCU - Universidad Tecnológica Nacional
pascala@frcu.utn.edu.ar 				maxi97olivera@gmail.com

		Pablo Suárez Lapalma                                               Iara Martinelli
FRCU - Universidad Tecnológica Nacional 	FRCU - Universidad Tecnológica Nacional
	   pablosuarezlapalma@gmail.com	  		    iaramarti789@gmail.com 

María Emilia Fernández 				Luciano Emmanuel Davezac
FRCU - Universidad Tecnológica Nacional	FRCU - Universidad Tecnológica Nacional 11emiliafernandez@gmail.com	   		lucianodavezac@gmail.com

Thiago Gomez Kehler
FRCU - Universidad Tecnológica Nacional
 thiagojgk@gmail.com



Resumen
En el ámbito universitario, la interpretación y aplicación de normativas es esencial, pero la complejidad y extensión de estas pueden dificultar su consulta efectiva. Este artículo describe un sistema de consultas en español, enfocado en la recuperación de información (Information Retrieval) de reglamentación académica. El sistema utiliza consultas en el lenguaje natural, en español, que se procesan para encontrar los artículos más relevantes  que responden a la consulta, utilizando modelos de representación del lenguaje y modelos generativos. En particular, se emplearon los modelos BETO, NOMIC y BGE-M3 para los embeddings tanto de las consultas como de los artículos de la reglamentación, y los modelos generativos Llama y GPT-4 para la expansión de las preguntas. Los experimentos realizados y el análisis de los resultados ofrecen valiosos criterios que pueden ser aplicados a problemas similares de recuperación de información en otros dominios y aplicaciones prácticas.
Introducción
En el ámbito universitario, la correcta interpretación y aplicación de las normativas es fundamental para el funcionamiento adecuado de las instituciones académicas. Estas normativas incluyen leyes, estatutos, ordenanzas, resoluciones, disposiciones y jurisprudencia que regulan diversas áreas, desde los derechos y responsabilidades de los estudiantes, hasta las atribuciones de los consejos universitarios y el personal administrativo. No obstante, la vasta cantidad de reglamentación y la complejidad del lenguaje jurídico-académico dificultan su consulta efectiva. Estudiantes, docentes, personal administrativo, miembros de los distintos consejos y autoridades universitarias suelen enfrentarse a dificultades a la hora de encontrar los artículos específicos que responden a sus consultas.
Para resolver este problema, surge la necesidad de desarrollar sistemas de recuperación de información que puedan responder a preguntas formuladas en lenguaje natural, simplificando el acceso a la normativa universitaria. Este tipo de solución no solo facilitaría el acceso a la reglamentación para los diversos claustros, sino que también mejoraría la eficiencia administrativa y la toma de decisiones.
En este contexto, las técnicas avanzadas de procesamiento del lenguaje natural (PLN) y los modelos de embeddings han demostrado ser herramientas clave para la búsqueda y recuperación de información basada en similitud semántica. Modelos como BGE-M3, Nomic, y BERT-BETO ofrecen soluciones prometedoras para abordar el desafío de interpretar consultas en lenguaje natural y mapearlas a los artículos correspondientes de la reglamentación universitaria.
En este trabajo, presentamos una evaluación exhaustiva del desempeño de diferentes modelos de embeddings aplicados a este dominio. Los experimentos incluyen tanto consultas originales como variantes generadas mediante técnicas de ampliación, utilizando modelos de lenguaje de última generación como GPT-4 y Llama 3.1. El objetivo es medir la precisión y la robustez de los modelos al enfrentar consultas con diversas formulaciones, asegurando que el sistema pueda manejar con éxito las diferencias en la redacción de preguntas sin perder efectividad en la recuperación de los artículos pertinentes.
Este estudio no solo evalúa las tecnologías de punta en éste ámbito, sino que también abre el camino hacia una solución práctica para mejorar el acceso a la normativa académica en el entorno universitario, beneficiando a todos los actores que forman parte de la comunidad educativa.
Este artículo está organizado de la siguiente manera: en la Sección 2 se presentan los trabajos relacionados, junto con una breve explicación de los modelos de lenguaje utilizados. La Sección 3 detalla el corpus empleado en el estudio. En la Sección 4 se expone la metodología utilizada. La Sección 5 describe los experimentos realizados, y en la Sección 6 se analizan los resultados obtenidos. Finalmente, en la Sección 7 se presentan las conclusiones y se discute el trabajo futuro.
Trabajo Relacionado
La Recuperación de Información (IR) [1] se centra en ayudar a las personas a encontrar la información más relevante en el formato más adecuado en el momento preciso. El desafío principal es estimar la relevancia entre una consulta q y un documento d, y el objetivo del motor de búsqueda es devolver los resultados más relevantes para la consulta q y presentarlos en una lista ordenada para el usuario. La recuperación de texto, específicamente, tiene como objetivo identificar recursos informativos relevantes en respuesta a consultas en lenguaje natural. Este enfoque es crucial para manejar la sobrecarga de información y se aplica en diversas áreas, como la respuesta a preguntas [2, 3], sistemas de diálogo [4, 5], enlace de entidades [6, 7] y búsqueda en la web [8].
Inicialmente los sistemas de recuperación de texto se centraron en la selección de términos representativos para la indexación de textos. Un avance crucial fue el modelo de espacio vectorial, que utiliza una "bolsa de palabras" (BoW) para representar documentos y consultas como vectores dispersos. Métodos de ponderación de términos como TF-IDF [11] estiman la relevancia según la similitud léxica entre los vectores de consulta y texto, apoyados en estructuras como el índice invertido [12] y el modelo BM25 [13]. Los enfoques de modelado de lenguaje estadístico [14] también han sido ampliamente explorados.
Con el avance del aprendizaje automático, el aprendizaje de rankings [15, 16] introdujo funciones  basadas en características diseñadas manualmente y entrenadas con juicios de relevancia. Sin embargo, estos métodos aún dependían de la ingeniería de características.
El resurgimiento de las redes neuronales ha permitido desarrollar sistemas de recuperación de texto más avanzados que no requieren características textuales diseñadas manualmente. Los enfoques de aprendizaje profundo [17] aprenden representaciones automáticas de consultas y documentos, mapeándolos en vectores densos en un espacio de representación latente. Este paradigma, conocido como Recuperación de Información Neuronal (Neural IR) [18, 19, 20], representa una exploración inicial de técnicas de recuperación densa. Los modelos de Neural IR anteriores al uso de modelos de lenguaje preentrenados se denominan modelos pre-BERT.
En tiempos recientes, los modelos de lenguaje preentrenados (PLM) [21], basados en la arquitectura Transformer [22], han revolucionado el campo. Estos modelos, entrenados en grandes volúmenes de datos textuales, han demostrado una capacidad mejorada para entender y representar la semántica del texto. Desde 2019, se han desarrollado modelos de recuperación de texto basados en PLM que han mejorado significativamente el rendimiento en conjuntos de datos de referencia [23, 24, 25] y han dominado en tareas de recuperación de documentos y pasajes [26].
Los Transformers [22] se han convertido en la base para el preentrenamiento de modelos de lenguaje, diseñado para modelar datos secuenciales de manera eficiente. A diferencia de las redes neuronales secuenciales tradicionales, como RNN y sus variantes [27], los Transformers introducen un mecanismo de autoatención que permite a un token atender todas las posiciones del input simultáneamente, facilitando la paralelización con GPU o TPU. Este diseño permite entrenar redes neuronales muy grandes de manera más flexible y efectiva.
Basados en los Transformers, los modelos de lenguaje preentrenados (PLM) [21, 160, 161] han transformado el procesamiento del lenguaje natural. Los PLMs se entrenan previamente en grandes corpus textuales generales utilizando funciones de pérdida auto-supervisadas y pueden ajustarse para tareas específicas mediante el preentrenamiento y ajuste fino [28, 29]. BERT [21], uno de los PLMs más representativos, utiliza arquitecturas bidireccionales profundas y enmascara palabras para aprender representaciones generales del texto, elevando significativamente el rendimiento en diversas tareas de procesamiento de lenguaje natural. El éxito de BERT ha dado lugar a estudios que incluyen enfoques de preentrenamiento mejorados [28], representaciones bidireccionales refinadas [30], y compresión de modelos [31], todos centrados en modelar la interacción semántica entre consultas y textos basándose en representaciones aprendidas en un espacio semántico denso latente.
BETO [32], es una versión de BERT entrenada en el lenguaje español.
El modelo Nomic (Nomic-embed-text-v1) [33] es una avanzada implementación de embebding de texto que sigue un enfoque de entrenamiento en múltiples etapas para mejorar la calidad de las representaciones textuales. Inicialmente, se entrena un Transformer preentrenado con un gran corpus de datos mediante un objetivo de pérdida contrastiva no supervisada. Luego, se realiza un ajuste fino adicional sobre conjuntos de datos etiquetados de alta calidad para refinar el modelo. Este enfoque de dos etapas permite al modelo beneficiarse de la gran cantidad de datos débiles disponibles y, a su vez, mejorar su precisión mediante el ajuste fino en datos más específicos y de alta calidad. El resultado es un modelo que supera a sus predecesores en varias métricas de evaluación, incluyendo benchmarks especializados en contextos largos.
El (BGE) M3-Embedding [34, 35] es un modelo de embedding innovador que destaca por su multi-lingualidad, multi-funcionalidad y multi-granularidad. Capaz de manejar más de 100 idiomas, este modelo realiza recuperación semántica en múltiples lenguas y soporta tres tipos de recuperación: densa, multi-vectorial y dispersa. Además, procesa desde frases cortas hasta documentos extensos de 8,192 tokens. Su entrenamiento incorpora técnicas avanzadas como la auto-distilación de conocimientos y una optimización en la estrategia de batching, lo que mejora la calidad y discriminatividad de los embeddings. M3-Embedding ha logrado resultados de vanguardia en benchmarks multilingües, cruzados entre idiomas y de documentos largos.
GPT-4 (Generative Pre-trained Transformer 4) [36] es un modelo generativo de lenguaje avanzado desarrollado por OpenAI, basado en la arquitectura de Transformer. A diferencia de sus predecesores, GPT-4 ofrece mejoras significativas en la generación de texto y la comprensión del lenguaje gracias a su mayor tamaño, capacidad de entrenamiento en un corpus más amplio y técnicas de ajuste fino avanzadas. 
LLaMA 3.1 (Large Language Model Meta AI) [37] es una versión avanzada de la serie LLaMA desarrollada por Meta AI. Este modelo de lenguaje se basa en la arquitectura de Transformer y está diseñado para mejorar la generación y comprensión del lenguaje natural. LLaMA 3.1 se distingue por su capacidad para manejar tareas complejas con una mayor precisión en la generación de texto y la comprensión del contexto en comparación con versiones anteriores.
En este trabajo, se compara la eficacia de estos tres modelos avanzados de representación del lenguaje—BETO, Nomic y BGE-M3— en la generación de embeddings para captar la semántica y recuperar artículos relevantes en respuesta a consultas en español. Además, se incorpora Llama y GPT-4 para expandir y refinar las consultas, verificando la robustez de estos modelos. Esta evaluación ofrece una visión integral sobre cómo resolver la recuperación efectiva de información textual en español, en dominios especializados.
Corpus (Base de Datos)
El corpus sobre el cual se realizó el estudio se encuentra semi-estructurado en una base de datos. La misma abarca las reglamentaciones asociadas con las leyes, estatutos, ordenanzas, resoluciones, disposiciones y jurisprudencia de la Universidad. Está organizada en una tabla de documentos, la cual incluye la identificación, fecha y descripción de cada reglamentación. Además, hay una tabla que contiene los artículos correspondientes de estos documentos, con el texto completo, información de capítulos y diversos metadatos. 
En total, contiene 674 artículos correspondientes a 27 documentos, los cuales abarcan diversas áreas de la gestión universitaria. Entre los temas que se abordan se incluyen la organización de concursos para cargos docentes, las funciones de los consejos de carrera, y los derechos y obligaciones de los estudiantes, entre otros.
Por ejemplo, una entrada en la tabla de artículos es la siguiente:
Documento: 10
Artículo: 5
Capítulo: "De los Consejos de Carrera. Disposiciones transitorias"
Contenido: "Los representantes del Consejo de carrera durarán 2 (dos) años en sus funciones y desempeñarán las mismas ad-honorem."
Metodología
En este estudio se desarrolló un sistema de búsqueda de información académica basado en consultas en lenguaje natural, orientado a la reglamentación universitaria. El enfoque metodológico se centró en el uso de modelos avanzados de procesamiento de lenguaje natural (PLN) para extraer respuestas relevantes a partir de una base de datos de normativas académicas. Para evaluar la eficacia del sistema, se utilizó un conjunto de consultas formuladas por usuarios reales, lo que permitió probar el rendimiento del modelo en situaciones cercanas a un entorno de uso cotidiano.
La base de consultas utilizada contiene un total de 300 preguntas relacionadas con aspectos específicos de la reglamentación mencionada anteriormente, acompañadas de una respuesta esperada que remite al número de artículo correspondiente en la normativa. Estas consultas fueron formuladas por 25 personas diferentes, de las cuales 20 eran estudiantes y las otras 5 docentes. Las preguntas se realizaron en función de la reglamentación, y se registró el artículo correcto que respondía a cada una de ellas. Algunas preguntas presentan errores ortográficos o problemas de redacción, pero se dejaron sin modificar para mantener la naturalidad del escenario y asegurar que la evaluación reflejara un entorno realista.
La estructura de esta base de consultas es la siguiente:
idConsulta: identificador único para cada pregunta.
Consulta: pregunta planteada, vinculada a alguna normativa específica.
idRespuestaEsperada (idArticulo): El número del artículo en la reglamentación que proporciona la respuesta esperada.
En la Tabla 1 se muestran tres de estas entradas. En la  pregunta 19, por ejemplo, la palabra “recusaciones” tiene un error ortográfico. Y en la 45 la letra “e” posee una ubicación incorrecta.
Tabla 1.  Ejemplos de Consultas

Para cada consulta, se generaron 10 preguntas similares utilizando Llama 3.1 y otras 10 con GPT-4o. Estas nuevas preguntas fueron examinadas manualmente por los miembros del equipo, para determinar si el  significado era consistente con la consulta original. Este proceso permitió ampliar el conjunto de consultas y evaluar la robustez del sistema frente a distintas formas de plantear una misma pregunta.
Tanto para las consultas como para los artículos correspondientes, se generaron embeddings utilizando los modelos preentrenados Sentence-embebdings-Beto, Nomic-embed-text-v1 y BGE-m3. La distancia coseno se empleó como medida de similitud semántica y para cada consulta se devolvieron los k artículos con mayor similitud, ordenados por este criterio. 
En los experimentos con preguntas similares, se calculó el promedio de las distancias entre las consultas reformuladas y cada artículo, utilizando esta medida como la distancia final. Esta estrategia proporcionó resultados más consistentes y robustos, al evaluar cómo el sistema respondía ante diferentes formas de expresar la misma pregunta.
Para evaluar la efectividad del método, se emplearon las métricas Top-k success rate y nDCG (Normalized Discounted Cumulative Gain), lo que permitió analizar el desempeño del sistema en términos de precisión y relevancia de las respuestas obtenidas:

y

donde:

reli es 1 si el ítem en la posición i es relevante, y 0 de lo contrario (ya que solo hay una respuesta correcta por pregunta).
k es la cantidad de respuestas devueltas por cada consulta.
e IDCGk (DCGk Ideal) en este caso es igual a 1, ya que éste es el valor correspondiente al caso en que la respuesta correcta se obtenga en la primer posición.
Experimentos Realizados
En esta sección, se detallan los experimentos realizados para evaluar el desempeño de diferentes modelos de embeddings en un conjunto de consultas. Los experimentos se han diseñado para comparar cómo cada modelo maneja consultas originales frente a consultas similares generadas mediante técnicas de ampliación. 
Cada modelo ha sido sometido a pruebas en distintas configuraciones para analizar su eficacia y precisión en la recuperación y comparación de información. A continuación, se describen los experimentos realizados, clasificados según el modelo de embedding utilizado.
Experimentos con el modelo BGE-M3
Consultas Originales: Evaluación del modelo BGE-M3 utilizando únicamente las consultas originales para determinar su desempeño en la recuperación de información sin modificaciones (Bge-m3Ori).
Consultas Similares Generadas por Llama 3.1: Evaluación del modelo bge-m3 con consultas similares generadas mediante Llama 3.1 con Ollama. Se consideran tres configuraciones (en todos los casos de preguntas similares las cantidades incluyen la pregunta original): 3, 5 y 10 consultas similares por pregunta (Bge-m3Lla3, Bge-m3Lla5 y Bge-m3Lla10).
Consultas Similares Generadas por GPT-4o: Evaluación del modelo bge-m3 con consultas similares generadas por GPT-4o en modo supervisado. Se consideran tres configuraciones: 3, 5 y 10 consultas similares por pregunta (Bge-m3GPT3, Bge-m3GPT5 y Bge-m3GPT10).
Experimentos con el modelo Nomic-embed-text-v1
Consultas Originales: Evaluación del modelo Nomic-embed-text-v1 utilizando únicamente las consultas originales para establecer su desempeño básico en la recuperación de información (NomicOri).
Consultas Similares Generadas por GPT-4o: Evaluación del modelo nomic-embed-text-v1 con consultas similares generadas por GPT-4o en modo supervisado, utilizando una única configuración: 10 consultas similares por pregunta (NomicGPT10).
Experimentos con el modelo Sentence-embeddings-BETO
Consultas Originales: Evaluación del modelo Sentence-embeddings-BETO con consultas originales para analizar su rendimiento en la recuperación de información sin consultas adicionales (Beto) .
Consultas Similares Generadas con GPT-4o: Evaluación del modelo Sentence-embeddings-BETO con consultas similares generadas por GPT-4o, utilizando una única configuración: 10 consultas similares por pregunta (BetoGPT10).
Cada uno de estos experimentos se diseñó para evaluar la capacidad de cada modelo de embedding en diferentes escenarios, permitiendo una comparación de su rendimiento en la recuperación de información basada en consultas originales y ampliadas. Los resultados obtenidos se presentan en las Tabla 2, y en la siguiente sección se discuten las implicaciones de cada configuración en el desempeño de los modelos.
Tabla 2.  Rendimiento de los distintos modelos
La mayor cantidad de experimentos fueron llevadas a cabo con el modelo BGE-M3 ya que desde un inicio mostró un rendimiento superior. La Figura 1 resume gráficamente los resultados obtenidos.
Análisis de los Resultados
Los experimentos detallados muestran un análisis exhaustivo del desempeño de tres modelos de embeddings: BGE-M3, Nomic-embed-text-v1, y Sentence-embeddings-BETO para resolver el problema de la recuperación de reglamentación académica como respuesta a consultas en el lenguaje natural. Se evaluaron tanto consultas originales como la original mas consultas similares generadas por modelos avanzados (Llama 3.1 y GPT-4o). A continuación, se analizan los principales hallazgos:
Desempeño del Modelo BGE-M3
El modelo BGE-M3 demostró ser el mejor de los tres en términos de precisión, y también es el más robusto frente a variaciones en las consultas:
Bge-m3Ori (solo consultas originales) alcanzó un Top-1 del 81.67%, Top-3 del 94.67%, y un nDCG3 de 0.89, lo que refleja un rendimiento excepcional con consultas no modificadas.
Al introducir consultas similares generadas por GPT-4o, los resultados se mantuvieron prácticamente iguales, con ligeras variaciones. Por ejemplo, Bge-m3GPT5 logró un Top-1 del 80.33% y un nDCG3 de 0.89, lo que indica que el modelo sigue respondiendo bien incluso cuando las consultas presentan diferencias en la redacción. Esto sugiere una alta robustez del modelo, capaz de adaptarse a diferentes formas de expresar la misma consulta sin pérdida de precisión significativa.
Por otro lado, con las consultas generadas por Llama 3.1, el rendimiento disminuye levemente, como en Bge-m3Lla10 (Top-1 del 65.66% y nDCG3 de 0.76). Aunque la precisión es menor que en el caso de GPT-4o, el modelo sigue respondiendo de manera efectiva ante una mayor variabilidad, lo que confirma su solidez. 
Desempeño del Modelo Nomic-embed-text-v1
El modelo Nomic-embed-text-v1 mostró un rendimiento razonable, aunque inferior al BGE-M3, tanto en precisión como en robustez:
Con consultas originales (NomicOri), el modelo alcanzó un Top-1 del 49.33% y un nDCG3 de 0.62, lo que representa un desempeño intermedio en la recuperación de información.
Sin embargo, al introducir consultas similares generadas por GPT-4o (NomicGPT10), se observó una disminución significativa en la precisión: Top-1 del 39.66% y nDCG3 de 0.53. Este resultado indica que el modelo es menos robusto frente a variaciones en la consulta. La caída en el rendimiento sugiere que Nomic tiene dificultades para manejar la flexibilidad en la redacción de las preguntas, lo que lo hace menos adaptable ante cambios en la formulación de las consultas.
Desempeño del Modelo Sentence-embeddings-BETO
El modelo BETO, aunque entrenado para textos en español, mostró el rendimiento más bajo de los tres en términos de precisión, alcanzando solo un nDCG3 de 0.46, lo que indica una capacidad limitada para recuperar información de manera precisa para el caso de estudio.
Generación de Preguntas Similares
Como resultado de la verificación manual de las consultas generadas por GPT-4o y Llama, se observó que, en general, GPT-4o produce preguntas con mayor similitud semántica en comparación con Llama. Esto explica por qué, en todos los casos, los resultados de las búsquedas utilizando consultas similares fueron mejores con GPT-4o. Por otro lado, Llama tiende a introducir "ruido" en algunas ocasiones, generando preguntas que no mantienen el mismo significado que la consulta original, lo que afecta la precisión de los resultados.
Aplicación Real al Problema de la Recuperación de Artículos Académicos
Los resultados obtenidos con el modelo BGE-M3 demuestran ser suficientemente robustos y adecuados para su aplicación práctica en la búsqueda de reglamentación académica. Y tiene la ventaja de no requerir entrenamiento adicional o ajuste fino. Esta característica reduce significativamente los costos operativos y de desarrollo. Además, el rendimiento del BGE-M3 en el dominio de la búsqueda de reglamentación académica supera el rendimiento obtenido en dominios abiertos con distintas variantes de BERT, por ejemplo sobre los conjuntos de datos TREC-DL19 y TREC-DL20, que presentan un nDCG@10 de entre el 70% y el 76% [38] . Este desempeño superior subraya la eficacia de BGE-M3 en contextos especializados, ofreciendo resultados de alta calidad con menor inversión en entrenamiento y ajuste fino.
Conclusiones y Trabajo Futuro
Este estudio muestra que la aplicación de modelos de embeddings avanzados en la recuperación de información jurídica-académica mejora significativamente la precisión y relevancia de las respuestas obtenidas. Entre los tres modelos evaluados —BGE-M3, Nomic-embed-text-v1 y Sentence-embeddings-BETO—, el modelo BGE-M3 mostró un rendimiento claramente superior, con una tasa de éxito notable en las consultas originales y similares.
Los experimentos con BGE-M3, que incluyeron variantes generadas tanto por Llama 3.1 como GPT-4, indicaron que el modelo puede manejar de manera robusta diferentes formulaciones de una misma consulta. Aunque la incorporación de consultas similares tiende a disminuir levemente la precisión, BGE-M3 sigue ofreciendo resultados altamente competitivos, especialmente en configuraciones con un menor número de consultas adicionales. Esto destaca su capacidad para adaptarse a diversas expresiones sin perder eficacia.
El rendimiento de Nomic-embed-text-v1 fue inferior, pero aceptable, en términos de precisión semántica. Por su parte, Sentence-embeddings-BETO, aunque menos eficaz que BGE-M3 y Nomic, podría tener aplicaciones en escenarios donde se priorice una mayor flexibilidad lingüística.
En cuanto a las métricas utilizadas (Top-k success rate y nDCG), BGE-M3 alcanzó un desempeño superior en casi todas las configuraciones, especialmente en el Top-1 y Top-3, lo que lo convierte en una opción recomendada para la implementación de sistemas de búsqueda de reglamentación como el expuesto en este artículo.
Respecto al trabajo futuro, es necesario seguir explorando el uso de modelos generativos para la mejora de los sistemas de recuperación de información. Además, se sugiere investigar cómo optimizar la incorporación de consultas similares sin que esto afecte la precisión de los resultados. La expansión de este enfoque a otros dominios normativos puede permitir validar la generalización del sistema y abrir nuevas oportunidades para la automatización en contextos académicos y administrativos.
Referencias
[1] Ceri, S., Bozzon, A., Brambilla, M., Della Valle, E., Fraternali, P., & Quarteroni, S. . An Introduction to Information Retrieval (2013).
[2]	V. Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, andW.-t. Yih, “Dense passage retrieval for open-domain question answering,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), (2020).
[3] 	E. M. Voorhees et al., “The trec-8 question answering track report.” in Trec, vol. 99, (1999).
[4] 	Z. Ji, Z. Lu, and H. Li, “An information retrieval approach to short text conversation,” CoRR, (2014).
[5] 	H. Chen, X. Liu, D. Yin, and J. Tang, “A survey on dialogue systems: Recent advances and new frontiers,” SIGKDD Explor., vol. 19, no. 2, pp. 25–35, (2017).
[6] 	D. Gillick, S. Kulkarni, L. Lansing, A. Presta, J. Baldridge, E. Ie, and D. Garcia-Olano, “Learning dense representations for entity retrieval,” in Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL), (2019).
[7] L. Wu, F. Petroni, M. Josifoski, S. Riedel, and L. Zettlemoyer, “Scalable zero-shot entity linking with dense entity retrieval,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), (2020).
[8] 	B. Mitra, F. Diaz, and N. Craswell, “Learning to match using local and distributed representations of text for web search,” in Proceedings of the 26th International Conference on World Wide Web, WWW 2017, Perth, Australia, April 3-7, (2017).
[9] 	G. Salton and C. Buckley, “Term-weighting approaches in automatic text retrieval,” Inf. Process. Manag., vol. 24, pp. 513–523, (1988).
[10] A. Aizawa, “An information-theoretic perspective of tf–idf measures,” Information Processing & Management, vol. 39, no. 1, pp. 45–65, (2003).
[11] S. Robertson, “Understanding inverse document frequency: on theoretical arguments for IDF” Journal of documentation, (2004).
[12] J. Zobel and A. Moffat, “Inverted files for text search engines” ACM Comput. Surv., vol. 38, p. 6, 2006.
[13] S. Robertson and H. Zaragoza, The probabilistic relevance framework: BM25 and beyond, 2009.
[14] C. Zhai, Statistical Language Models for Information Retrieval, ser. Synthesis Lectures on Human Language Technologies. Morgan & Claypool Publishers, 2008.
[15] T. Liu, “Learning to rank for information retrieval,” in Proceeding of the 33rd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2010, Geneva, Switzerland, July 19-23, 2010.
[16] H. Li, “Learning to rank for information retrieval and natural language processing,” Synthesis Lectures on Human Language Technologies, vol. 4, pp. 1–113, 2011.
[17] P. Huang, X. He, J. Gao, L. Deng, A. Acero, and L. P. Heck, “Learning deep structured semantic models for web search using clickthrough data,” in 22nd ACM International Conference on Information and Knowledge Management, CIKM’13, San Francisco, CA, USA, October 27 - November 1, (2013).
[18] J. Guo, Y. Fan, Q. Ai, andW. B. Croft, “A deep relevance matching model for ad-hoc retrieval,” in Proceedings of the 25th ACM International Conference on Information and Knowledge Management, CIKM 2016, Indianapolis, IN, USA, October 24-28, (2016).
[19] B. Mitra and N. Craswell, “Neural models for information retrieval,” arXiv preprint arXiv:1705.01509, (2017).
[20] J. Guo, Y. Fan, L. Pang, L. Yang, Q. Ai, H. Zamani, C. Wu, W. B. Croft, and X. Cheng, “A deep look into neural ranking models for information retrieval,” Information Processing & Management, vol. 57, no. 6, p. 102067, (2020).
[21] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pretraining of deep bidirectional transformers for language understanding” in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1, (2019).
[22]	A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, (2017).
[23]	J. Lin, R. Nogueira, and A. Yates, “Pretrained transformers for text ranking: BERT and beyond” Synthesis Lectures on Human Language Technologies, vol. 14, no. 4, pp. 1–325, (2021).
[24] 	Y. Fan, X. Xie, Y. Cai, J. Chen, X. Ma, X. Li, R. Zhang, J. Guo, and Y. Liu, “Pre-training methods in information retrieval,” arXiv:2111.13853, (2021).
[25]	Y. Cai, Y. Fan, J. Guo, F. Sun, R. Zhang, and X. Cheng, “Semantic models for the first-stage retrieval: A comprehensive review,” arXiv:2103.04831, (2021).
[26] 	N. Craswell, B. Mitra, E. Yilmaz, D. Campos, and J. Lin, “Overview of the trec 2021 deep learning track,” in Text Retrieval Conference (TREC), (2022).
[27 ]	K. Cho, B. van Merri¨enboer, D. Bahdanau, and Y. Bengio, “On the properties of neural machine translation: Encoder–decoder approaches,” in Proceedings of SSST-8, Eighth Workshop on Syntax, Semantics and Structure in Statistical Translation, (2014).
[28]	Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, “Roberta: A robustly optimized bert pretraining approach,” arXiv:1907.11692, (2019).
[29] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, et al.  “Language models are few-shot learners,” in ANIPS 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS (2020). 
[30] X. Ma, Z. Wang, P. Ng, R. Nallapati, and B. Xiang, “Universal text representation from bert: An empirical study,” arXiv:1910.07973, (2019).
[31] 	V. Sanh, L. Debut, J. Chaumond, and T. Wolf, “Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter,” arXiv:1910.01108, (2019).
[32] 	Cañete, J., Chaperon, G., Fuentes, R., Ho, J.H., Kang, H., Pérez, J.: Spanish pretrained bert model and evaluation data. In: PML4DC at ICLR 2020 (2020).
[33] 	Nussbaum, Z., Morris, J.X., Duderstadt, B., & Mulyar, A. Nomic Embed: Training a Reproducible Long Context Text Embedder. ArXiv, abs/2402.01613 (2024).
[34] 	Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. “C-Pack: Packaged Resources To Advance General Chinese Embedding”. ArXiv, abs/2309.07597 (2023).
[35] 	Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., & Liu, Z. BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. Annual Meeting of the Association for Computational Linguistics (2024).
[36] 	Achiam, O.J., Adler, S., Agarwal, et al. GPT-4 Technical Report (2023).
[37] Touvron, H., Martin, L., Stone, et al. Llama 2: Open Foundation and Fine-Tuned Chat Models. ArXiv, abs/2307.09288 (2023).
[38]	Zhu, Y., Yuan, H., Wang, S., Liu, J., Liu, W., Deng, C., Dou, Z., & Wen, J. . Large Language Models for Information Retrieval: A Survey. ArXiv, abs/2308.07107 (2023).