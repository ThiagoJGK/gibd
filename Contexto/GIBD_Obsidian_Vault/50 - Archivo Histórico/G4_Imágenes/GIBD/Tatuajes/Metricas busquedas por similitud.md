---
aliases: [Metricas busquedas por similitud]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2024-09-12
origen_zip: GIBD-20260521T205218Z-3-005.zip
ruta_interna: "GIBD/Tatuajes/CoNaIISI 2024/Metricas busquedas por similitud.docx"
tamanio_bytes: 101509
---

# Metricas busquedas por similitud

Ruta interna: `GIBD/Tatuajes/CoNaIISI 2024/Metricas busquedas por similitud.docx`

---

1. Título
Ejemplo: "Extracción Automática de Artículos Reglamentarios Usando Modelos de Lenguaje y Técnicas de Similaridad Semántica"
2. Resumen
Contenido: Presenta un resumen conciso del objetivo del estudio, la metodología utilizada, los resultados principales y las conclusiones. Por ejemplo:
Objetivo: Este estudio explora un método basado en modelos de lenguaje para extraer artículos específicos de un reglamento académico mediante la comparación de embeddings semánticos.
Metodología: Se utiliza BERT y otros modelos para generar embeddings de preguntas y artículos. La pregunta se expande a múltiples formas mediante Llama, y se comparan los embeddings usando distancia coseno.
Resultados: Los resultados muestran una alta precisión en la extracción de artículos relevantes en comparación con métodos tradicionales.
Conclusiones: La técnica propuesta mejora la precisión y relevancia en la recuperación de información reglamentaria.
3. Introducción
Contenido: Introduce el problema de la búsqueda y extracción de información en documentos reglamentarios, la importancia de la precisión en estos sistemas, y la motivación para usar modelos de lenguaje.
Problema: La dificultad en encontrar y extraer artículos relevantes de grandes documentos reglamentarios.
Motivación: La necesidad de métodos automáticos precisos para el cumplimiento normativo y consultas académicas.
Contribución: Describir cómo el uso de modelos de lenguaje avanzados puede mejorar la precisión en la extracción de información.
4. Trabajo Relacionado
4. Trabajo Relacionado
En esta sección, se revisan los enfoques y técnicas relevantes para la extracción de información y los sistemas de pregunta y respuesta (QA), con un enfoque especial en el uso de modelos de lenguaje avanzados como BERT y técnicas de similitud semántica.
4.1. Sistemas de Pregunta y Respuesta
Los sistemas de pregunta y respuesta han sido un área activa de investigación en procesamiento del lenguaje natural (PLN). Tradicionalmente, estos sistemas se basaban en técnicas de recuperación de información, como búsqueda por palabras clave y coincidencia de patrones. Sin embargo, los avances en modelos de lenguaje han revolucionado este campo.
Deep Learning for Question Answering: Los primeros sistemas de QA basados en aprendizaje profundo, como el modelo BiDAF (Bidirectional Attention Flow), han mostrado una notable mejora en la comprensión de texto al utilizar mecanismos de atención bidireccional para alinear preguntas y respuestas en contextos específicos (Seo et al., 2016).
Transformers y Modelos Preentrenados: La introducción de modelos de transformers, como BERT (Bidirectional Encoder Representations from Transformers), ha establecido nuevos estándares en la calidad de los sistemas de QA. BERT permite una comprensión profunda del contexto bidireccional, mejorando significativamente la precisión de las respuestas (Devlin et al., 2018).
4.2. Modelos de Lenguaje para Extracción de Información
Los modelos de lenguaje preentrenados, como BERT y sus variantes, se utilizan para generar embeddings semánticos de texto, que se pueden utilizar para la extracción de información y comparación de similitud.
BERT para Extracción de Información: BERT se ha utilizado eficazmente para la extracción de información mediante la generación de embeddings que capturan el significado contextual del texto. Esta técnica es esencial para encontrar pasajes relevantes dentro de grandes documentos (Liu et al., 2019).
Transformers para Preguntas y Respuestas: Modelos como RoBERTa y ALBERT, variantes de BERT, han sido desarrollados para mejorar el rendimiento en tareas de QA mediante el ajuste de la arquitectura de transformers y la preentrenamiento en grandes corpus de datos (Liu et al., 2019; Lan et al., 2019).
4.3. Técnicas de Similaridad Semántica
La comparación de embeddings generados por modelos de lenguaje se realiza mediante técnicas de similitud semántica, como la distancia coseno. Esta técnica mide la similitud entre dos vectores en el espacio de embeddings.
Distancia Coseno y Similitud: La distancia coseno es una medida popular para comparar vectores de embeddings y evaluar la similitud semántica. Se ha utilizado en diversos contextos, desde la recuperación de información hasta la clasificación de texto (Manning et al., 2008).
Comparación de Embeddings para QA: La comparación de embeddings de preguntas y respuestas mediante distancia coseno ha demostrado ser efectiva en la recuperación precisa de información en sistemas de QA (Reimers and Gurevych, 2019).
4.4. Expansión de Preguntas
La expansión de preguntas para mejorar la recuperación de información se ha abordado mediante técnicas de generación de lenguaje natural y modelos generativos.
Llama y Generación de Preguntas: Llama y otros modelos generativos pueden expandir una pregunta a varias formas sinónimas, mejorando la cobertura de la consulta. Estos enfoques son útiles para cubrir variaciones en la formulación de preguntas y aumentar la precisión de la recuperación (Brown et al., 2020).
Técnicas de Parafraseo: La generación de parafraseos y variaciones de preguntas también contribuye a la mejora en sistemas de QA, proporcionando múltiples formas de hacer la misma pregunta para asegurar una recuperación más robusta (Wang et al., 2018).

5. Metodología
Contenido: Describe en detalle la metodología utilizada en tu estudio, incluyendo la generación de embeddings, la expansión de preguntas y la comparación de similitud.
Modelo de Lenguaje: Explicar el uso de BERT para generar embeddings de artículos y preguntas.
Expansión de Preguntas: Describir cómo Llama se utiliza para generar múltiples formas de la misma pregunta.
Comparación de Embeddings: Detallar el proceso de comparación utilizando distancia coseno.
Base de Datos: Describir el reglamento académico utilizado, su estructura y formato.
6. Resultados
Contenido: Presenta los resultados de tus experimentos, incluyendo métricas de evaluación, comparaciones entre diferentes configuraciones y modelos.
Métricas de Evaluación: Precisión, recuperación, F1-score, etc.
Comparación de Configuraciones: Comparar resultados entre diferentes modelos de embeddings y técnicas de expansión de preguntas.
Casos de Estudio: Ejemplos de preguntas y artículos relevantes extraídos.
7. Discusión
Contenido: Analiza los resultados obtenidos, discute las implicaciones, y compara con el trabajo relacionado. Examina la efectividad de la técnica propuesta y sus limitaciones.
Interpretación de Resultados: Reflexiona sobre la efectividad de la técnica en comparación con enfoques tradicionales.
Limitaciones: Discute las limitaciones encontradas, como posibles ambigüedades en las preguntas o variaciones en la estructura del reglamento.
Implicaciones: Analiza cómo los resultados pueden impactar en el uso de sistemas de QA en contextos reglamentarios y académicos.
8. Conclusiones y Trabajo Futuro
Contenido: Resume las conclusiones principales del estudio y sugiere posibles direcciones para investigaciones futuras.
Conclusiones: Resalta los principales hallazgos y la efectividad de la técnica en la extracción de información.
Trabajo Futuro: Propón mejoras en la técnica, exploración de otros modelos de lenguaje, y aplicación en diferentes tipos de documentos o contextos.
9. Referencias
Contenido: Incluye todas las referencias bibliográficas y fuentes citadas en el artículo.
10. Apéndices (si es necesario)
Contenido: Proporciona información adicional como ejemplos de datos, configuraciones experimentales detalladas, o código relevante.
Esta estructura debería ayudarte a organizar tu artículo de manera clara y efectiva, permitiendo una presentación lógica y coherente de tus resultados y metodologías.


Referencias
Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805.
Liu, Y., Ott, M., Goyal, N., Du, J., & Wang, C. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692.
Lan, Z., Chen, J., Goodman, S., Gimpel, K., & Sharma, P. (2019). ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. arXiv:1909.11942.
Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Stanford NLP.
Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. arXiv:1908.10084.
Brown, T. B., Mann, B., Ryder, N., & Subbiah, M. (2020). Language Models are Few-Shot Learners. arXiv:2005.14165.
Wang, S., & Jiang, J. (2018). A Compare of the Most Common Models for Paraphrase Identification. ArXiv:1804.00735.


–
Para evaluar la eficacia de las técnicas de búsqueda por similitud de imágenes, se utilizan varias métricas que permiten medir qué tan precisas, completas y eficientes son las técnicas en cuestión. Las más comunes son las siguientes:
1. Precisión (Precision)
Definición: La proporción de imágenes relevantes entre las que se han recuperado.
Fórmula: Precisioˊn=Imaˊgenes relevantes recuperadasTotal de imaˊgenes recuperadas\text{Precisión} = \frac{\text{Imágenes relevantes recuperadas}}{\text{Total de imágenes recuperadas}}Precisioˊn=Total de imaˊgenes recuperadasImaˊgenes relevantes recuperadas​
Uso: Mide la exactitud del sistema al devolver solo las imágenes relevantes. Es útil en sistemas donde es importante reducir los falsos positivos.
2. Cobertura o Exhaustividad (Recall)
Definición: La proporción de imágenes relevantes que fueron recuperadas frente a todas las imágenes relevantes en la base de datos.
Fórmula: Recall=Imaˊgenes relevantes recuperadasTotal de imaˊgenes relevantes en la base de datos\text{Recall} = \frac{\text{Imágenes relevantes recuperadas}}{\text{Total de imágenes relevantes en la base de datos}}Recall=Total de imaˊgenes relevantes en la base de datosImaˊgenes relevantes recuperadas​
Uso: Mide cuántas imágenes relevantes están siendo omitidas por el sistema, es decir, si el sistema encuentra todas las imágenes que debería.
3. F1-Score
Definición: La media armónica entre precisión y recall, que proporciona una métrica única que balancea ambos aspectos.
Fórmula: F1-Score=2×Precisioˊn×RecallPrecisioˊn+Recall\text{F1-Score} = 2 \times \frac{\text{Precisión} \times \text{Recall}}{\text{Precisión} + \text{Recall}}F1-Score=2×Precisioˊn+RecallPrecisioˊn×Recall​
Uso: Es útil cuando se busca un equilibrio entre precisión y recall.
4. Precisión en el Top-N (Precision@N)
Definición: La precisión calculada sobre los primeros N resultados devueltos por el sistema.
Uso: Se utiliza para evaluar qué tan buenos son los resultados en las primeras posiciones, que generalmente son las más importantes en una búsqueda.
5. mAP (Mean Average Precision)
Definición: El promedio de las precisiones calculadas en diferentes puntos de recuperación en una consulta.
Fórmula: mAP=1Q∑q=1Q1Rq∑r=1RqPrecisioˊn(r)\text{mAP} = \frac{1}{Q} \sum_{q=1}^{Q} \frac{1}{R_q} \sum_{r=1}^{R_q} \text{Precisión}(r)mAP=Q1​q=1∑Q​Rq​1​r=1∑Rq​​Precisioˊn(r) Donde QQQ es el número de consultas, y RqR_qRq​ es el número de resultados relevantes para la consulta qqq.
Uso: Es una métrica robusta que considera tanto la precisión como el orden de los resultados.
6. nDCG (Normalized Discounted Cumulative Gain)
Definición: Mide la calidad de la clasificación de los resultados, tomando en cuenta no solo si los resultados son relevantes, sino también su posición.
Fórmula: nDCG=DCGIDCG\text{nDCG} = \frac{\text{DCG}}{\text{IDCG}}nDCG=IDCGDCG​ Donde: DCG=∑i=1N2Relevancia(i)−1log⁡2(i+1)\text{DCG} = \sum_{i=1}^{N} \frac{2^{\text{Relevancia}(i)} - 1}{\log_2(i+1)}DCG=i=1∑N​log2​(i+1)2Relevancia(i)−1​ IDCG es el DCG ideal, que se obtiene cuando todos los resultados relevantes están en el orden perfecto.
Uso: Mide la utilidad de los resultados recuperados con énfasis en el orden de los resultados.
7. ROC y AUC (Receiver Operating Characteristic y Area Under the Curve)
Definición: ROC muestra la tasa de verdaderos positivos frente a la tasa de falsos positivos. AUC mide el área bajo la curva ROC.
Uso: Estas métricas son útiles para medir el desempeño general de los sistemas de recuperación de imágenes.
8. MRR (Mean Reciprocal Rank)
Definición: El promedio del recíproco de la posición del primer resultado relevante.
Fórmula: MRR=1∣Q∣∑i=1∣Q∣1Ranki\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{Rank}_i}MRR=∣Q∣1​i=1∑∣Q∣​Ranki​1​
Uso: Evalúa qué tan temprano aparece el primer resultado relevante en la lista de recuperados.
9. Recall@K
Definición: Es el porcentaje de consultas que tienen al menos un resultado relevante en los primeros KKK resultados devueltos.
Uso: Ayuda a medir qué tan bueno es el sistema en asegurar que al menos un resultado relevante esté entre las primeras posiciones.
10. Coverage Rate
Definición: Es la proporción de imágenes recuperadas que cubren diferentes categorías o clases.
Uso: Evalúa la diversidad de los resultados recuperados, lo cual es importante en sistemas que deben recuperar imágenes de diversas categorías.
11. Latency (Latencia)
Definición: El tiempo que tarda el sistema en devolver resultados.
Uso: Importante para medir la eficiencia del sistema en términos de tiempo de respuesta.
12. Precision-Recall Curve
Definición: Es una curva que muestra la relación entre precisión y recall a diferentes umbrales de clasificación.
Uso: Ofrece una visualización completa del trade-off entre precisión y recall en diferentes niveles de confianza.
Estas métricas proporcionan diferentes perspectivas sobre la eficacia de los sistemas de búsqueda por similitud de imágenes y suelen ser utilizadas en combinación para obtener una evaluación completa del rendimiento del sistema.
–
Tienes razón, cuando trabajas con la búsqueda de los kkk vecinos más cercanos (k-NN), la precisión y el recall se ajustan a la situación específica. Aquí hay algunas métricas más adecuadas para este contexto:
Precisión en el Top-k (Top-k Precision): Mide la proporción de consultas en las que la imagen relevante está entre los kkk vecinos más cercanos recuperados.
Precisioˊn en el Top-k=Nuˊmero de consultas donde la imagen relevante estaˊ en los k vecinos maˊs cercanosNuˊmero total de consultas\text{Precisión en el Top-k} = \frac{\text{Número de consultas donde la imagen relevante está en los k vecinos más cercanos}}{\text{Número total de consultas}}Precisioˊn en el Top-k=Nuˊmero total de consultasNuˊmero de consultas donde la imagen relevante estaˊ en los k vecinos maˊs cercanos​
En tu caso, si la imagen correcta está entre los kkk vecinos más cercanos, la precisión en el Top-k sería 1, y 0 si no está entre los kkk vecinos.
Recuperación en el Top-k (Top-k Recall): Aunque en tu caso siempre hay una sola imagen relevante, esta métrica se puede adaptar para indicar si la imagen correcta está entre los kkk vecinos más cercanos, similar a la precisión en el Top-k.
Recuperacioˊn en el Top-k=Nuˊmero de consultas donde la imagen relevante estaˊ en los k vecinos maˊs cercanosNuˊmero total de consultas\text{Recuperación en el Top-k} = \frac{\text{Número de consultas donde la imagen relevante está en los k vecinos más cercanos}}{\text{Número total de consultas}}Recuperacioˊn en el Top-k=Nuˊmero total de consultasNuˊmero de consultas donde la imagen relevante estaˊ en los k vecinos maˊs cercanos​
Dado que hay solo una imagen relevante, esta métrica será equivalente a la precisión en el Top-k.
Ranking de la Imagen Correcta (Rank of the Correct Image): Mide el rango o la posición de la imagen correcta entre los kkk vecinos más cercanos. Si la imagen correcta está dentro del rango kkk, esta métrica te dice en qué posición está. Para una única imagen relevante, puedes calcular el promedio del rango de las imágenes correctas entre todas las consultas.
Tasa de Éxito en el Top-k (Top-k Success Rate): Calcula la proporción de consultas para las cuales la imagen relevante está en alguno de los kkk vecinos más cercanos.
Tasa de Eˊxito en el Top-k=Nuˊmero de consultas donde la imagen relevante estaˊ en los k vecinos maˊs cercanosNuˊmero total de consultas\text{Tasa de Éxito en el Top-k} = \frac{\text{Número de consultas donde la imagen relevante está en los k vecinos más cercanos}}{\text{Número total de consultas}}Tasa de Eˊxito en el Top-k=Nuˊmero total de consultasNuˊmero de consultas donde la imagen relevante estaˊ en los k vecinos maˊs cercanos​
Tiempo de Consulta en el Top-k (Top-k Query Time): Mide el tiempo que toma para recuperar los kkk vecinos más cercanos y puede ser útil para evaluar la eficiencia de la búsqueda además de la precisión.
Estas métricas te ayudarán a evaluar la efectividad de tu técnica de búsqueda en el contexto específico de kkk vecinos más cercanos, proporcionando una visión clara de cómo de bien encuentra la imagen relevante en los primeros kkk resultados.

METRICAS A UTILIZAR
Tasa de Éxito en el Top-k (Top-k Success Rate): Calcula la proporción de consultas para las cuales la imagen relevante está en alguno de los kkk vecinos más cercanos.
Tasa de Eˊxito en el Top-k=Nuˊmero de consultas donde la imagen relevante estaˊ en los k vecinos maˊs cercanosNuˊmero total de consultas\text{Tasa de Éxito en el Top-k} = \frac{\text{Número de consultas donde la imagen relevante está en los k vecinos más cercanos}}{\text{Número total de consultas}}Tasa de Eˊxito en el Top-k=Nuˊmero total de consultasNuˊmero de consultas donde la imagen relevante estaˊ en los k vecinos maˊs cercanos​
Latency (Latencia)
Definición: El tiempo que tarda el sistema en devolver resultados.
Uso: Importante para medir la eficiencia del sistema en términos de tiempo de respuesta.

TIEMPO PROMEDIO DE BÚSQUEDA:  0.7849000011171613  secs
TIEMPO PROMEDIO DE CÁLCULO DE LOS VECTORES CARACTERÍSTICOS DE UNA CONSULTA:  0.013414287567138672  secs
TIEMPO PROMEDIO DE CÁLCULO DE LOS VECTORES CARACTERÍSTICOS DE UNA CONSULTA:  0.03367647353340598  secs

Porcentaje de aciertos NN1:  0.9558823529411765
	Porcentaje de aciertos NN3:  0.9852941176470589
	Porcentaje de aciertos NN5:  1.0
	nDGC (Normalized Discounted Cumulative Gain):  0.9769217140893147
68 consultas

i5, 16GB RAM, SSD 960GB y GPU NVIDIA GTX 960

Para evaluar la calidad de la búsqueda en un escenario donde la respuesta es única, pero quieres que la posición de la respuesta relevante influya en la métrica, puedes usar Discounted Cumulative Gain (DCG) o Normalized Discounted Cumulative Gain (nDCG), adaptándola a tu contexto. Estos enfoques penalizan las posiciones más bajas en el ranking, dándole mayor importancia a las respuestas correctas que aparecen en las primeras posiciones.
Cómo Aplicar DCG/nDCG
Discounted Cumulative Gain (DCG):
DCG penaliza las respuestas correctas que aparecen en posiciones más bajas. La fórmula para DCG para una única consulta es:
DCG=Relevancia1+∑i=2nRelevanciailog⁡2(i+1)\text{DCG} = \text{Relevancia}_1 + \sum_{i=2}^{n} \frac{\text{Relevancia}_i}{\log_2(i+1)}DCG=Relevancia1​+i=2∑n​log2​(i+1)Relevanciai​​
Para una única respuesta relevante, la fórmula se simplifica a:
DCG=Relevancialog⁡2(posicioˊn+1)\text{DCG} = \frac{\text{Relevancia}}{\log_2(\text{posición} + 1)}DCG=log2​(posicioˊn+1)Relevancia​
Si la respuesta relevante está en la primera posición, la fórmula será:
DCG=Relevancia\text{DCG} = \text{Relevancia}DCG=Relevancia
Si está en la segunda posición:
DCG=Relevancialog⁡2(3)\text{DCG} = \frac{\text{Relevancia}}{\log_2(3)}DCG=log2​(3)Relevancia​
Ideal Discounted Cumulative Gain (IDCG):
IDCG es el DCG máximo posible cuando la respuesta relevante está en la mejor posición posible. Para una sola respuesta relevante, IDCG es simplemente la relevancia en la primera posición.
IDCG=Relevancia\text{IDCG} = \text{Relevancia}IDCG=Relevancia
Normalized Discounted Cumulative Gain (nDCG):
Para normalizar DCG y comparar entre diferentes consultas, divides DCG entre IDCG:
nDCG=DCGIDCG\text{nDCG} = \frac{\text{DCG}}{\text{IDCG}}nDCG=IDCGDCG​
Para tu caso específico, dado que IDCG es igual a la relevancia (siempre 1 en el caso de relevancia binaria), nDCG simplificará a simplemente DCG en función de la posición de la respuesta correcta.
Ejemplo
Supongamos que la respuesta relevante es "1" y aparece en la siguiente posición:
Primera posición:
DCG=1log⁡2(1+1)=1.0\text{DCG} = \frac{1}{\log_2(1+1)} = 1.0DCG=log2​(1+1)1​=1.0
Segunda posición:
DCG=1log⁡2(2+1)≈0.63\text{DCG} = \frac{1}{\log_2(2+1)} \approx 0.63DCG=log2​(2+1)1​≈0.63
Tercera posición:
DCG=1log⁡2(3+1)≈0.50\text{DCG} = \frac{1}{\log_2(3+1)} \approx 0.50DCG=log2​(3+1)1​≈0.50
En este caso, la importancia de la respuesta disminuye conforme la posición aumenta, y obtendrás una evaluación más precisa de la calidad del ranking.















Consultas recortando los bordes
Consultas con rotación
Consultas con ruido?
Consultas con traslación
Consultas vista de lado (perspectiva)
Consultas con efectos
				Sin DesenfoqueGauss		Con DesenfGaussiano
	cubismo 			no				si
	deformación			si				si
	desenfoque gaussiano		no				si
	fotocopia			si				si				
	lente				si				si
	lienzo				si				si
	luz-flash			no				si
	monitor baja resolución	si				si
	ondas				si				si
	pintura al óleo			si				si			
	pixelado			no				si
	


Los tatuajes han demostrado ser una herramienta eficaz en la aplicación de la ley para identificar a delincuentes y víctimas, ya que contienen características únicas y reconocibles. Son particularmente útiles para identificar a individuos asociados con pandillas o organizaciones criminales. Sin embargo, las preocupaciones de privacidad en la recolección de imágenes de tatuajes han resultado en la escasez de bases de datos, lo que ha retrasado el desarrollo de métodos avanzados de recuperación de imágenes de tatuajes. Para abordar esta limitación, proponemos un enfoque generativo no supervisado que crea un conjunto de datos balanceado con 28,550 imágenes semisintéticas de 571 categorías de tatuajes. Además, presentamos la Red de Reconstrucción de Plantillas de Tatuajes (TattTRN), que aprende a mapear el tatuaje de entrada a su plantilla correspondiente, mejorando las características distintivas del embebido final. Nuestros experimentos con bases de datos del mundo real, como WebTattoo y BIVTatt, muestran que este enfoque alcanza hasta un 99% de precisión al revisar las primeras 20 entradas de la lista de candidatos. 

El tatuaje es una forma de biometría suave que refleja características discriminativas de una persona, como sus creencias y personalidad. La detección y reconocimiento automáticos de imágenes de tatuajes es un problema complejo. En este trabajo, presentamos métodos basados en redes neuronales convolucionales profundas para la correspondencia automática de imágenes de tatuajes, utilizando las arquitecturas de AlexNet y redes Siamese. Además, demostramos que el uso de la función de pérdida triplet en lugar de la función de pérdida contrastiva puede mejorar significativamente el rendimiento de un sistema de correspondencia de tatuajes. A través de extensos experimentos en el recientemente introducido conjunto de datos Tatt-C, mostramos que nuestro método captura la estructura significativa de los tatuajes y supera de manera notable a muchos algoritmos competitivos de reconocimiento de tatuajes. 

El uso de tatuajes como biometría suave para asistir a las fuerzas del orden en la identificación de sospechosos ha crecido constantemente junto con su popularidad en la sociedad. En 2015, el Instituto Nacional de Estándares y Tecnología (NIST, por sus siglas en inglés) informó que una quinta parte de los adultos en los EE. UU. tenía al menos un tatuaje, lo que coloca a los EE. UU. como la tercera población más tatuada del mundo, después de Italia y Suecia. Esta tendencia seguía expandiéndose, como lo demostró una encuesta realizada en 2021, que reveló que el 26% de los estadounidenses tienen al menos un tatuaje. 

A diferencia de las características biométricas tradicionales, como las huellas dactilares y los rostros, los tatuajes no se pueden utilizar para establecer directamente la identidad de una persona. Sin embargo, a diferencia de otras biometrías suaves como el género, la edad o la raza, los tatuajes contienen información más discriminativa que puede ser útil para identificar a individuos y rastrear a miembros de pandillas u organizaciones criminales. Por lo tanto, el reconocimiento de tatuajes es un área de interés para los investigadores forenses, lo que motiva el desarrollo de técnicas automatizadas de recuperación de imágenes basadas en tatuajes.

La tabla 1 muestra un resumen de las bases de datos de tatuajes disponibles, destacando que la mayoría son limitadas en tamaño o no son de acceso público, lo que ha ralentizado el desarrollo de nuevos métodos de recuperación de tatuajes.

Para superar estos desafíos, introducimos una gran base de datos de tatuajes semi-sintéticos y proponemos un novedoso enfoque de recuperación de tatuajes que transforma la piel tatuada en una plantilla de tatuaje correspondiente para mejorar la descripción final. Los resultados experimentales con imágenes reales muestran una mejora convincente en el rendimiento del sistema propuesto frente a las líneas base.

El resto del artículo está organizado de la siguiente manera: en la sección 2 se presenta una revisión de los sistemas de recuperación de tatuajes existentes; la sección 3 describe los fundamentos de nuestro enfoque; en la sección 4 se detalla el montaje experimental, seguido de la discusión de resultados en la sección 5; finalmente, se presentan las conclusiones y futuras líneas de trabajo en la sección 6.

**Trabajo relacionado**  
Ver pie de foto  
Figura 2: Descripción conceptual del sistema propuesto: la plantilla transformada de la imagen de entrada ayuda a mitigar los desafíos relacionados con el proceso de captura. El embedding final de características, que representa las propiedades sobresalientes del tatuaje de entrada, es la concatenación de los dos embeddings de características calculados y se puede usar para recuperar muestras de tatuajes similares en una base de datos.

Entre los diversos rasgos biométricos suaves, los tatuajes han recibido considerable atención por parte de los investigadores forenses en los últimos años, debido a su prevalencia en el sector criminal de la población y su prominencia en la atención visual. Durante más de 5,000 años, los humanos han marcado sus cuerpos con tatuajes para expresar creencias personales o para asociarse con un grupo. La evidencia más antigua de tatuajes fue encontrada en el cuerpo de Ötzi, el Hombre de Hielo. Ötzi, la momia más famosa de Europa, fue descubierta por excursionistas alemanes en los Alpes en 1991. La siguiente evidencia más antigua de tatuajes proviene de momias que se cree que murieron entre el 3351 y el 3017 a.C. en el Antiguo Egipto.  
  
En el ámbito forense, los tatuajes también han sido útiles para ayudar a identificar víctimas de ataques terroristas como el 11 de septiembre y desastres naturales como el tsunami del Océano Índico en 2004.

Para ayudar en la identificación de sujetos, los primeros sistemas de recuperación de tatuajes se basaban en la coincidencia de palabras clave o metadatos. Para este fin, las agencias de aplicación de la ley generalmente seguían el estándar ANSI/NIST-ITL 1-2000 y asignaban una única palabra clave a cada imagen de tatuaje en la base de datos. Dado que estas soluciones presentaban inconvenientes obvios, como un vocabulario limitado para describir numerosos patrones de tatuajes, la posibilidad de que se necesitaran varias palabras clave para describir adecuadamente un tatuaje, y la inconsistencia en el etiquetado de los tatuajes, el siguiente conjunto de enfoques de recuperación se centró en el uso de descriptores diseñados a mano para representar el color, la textura, la forma y la apariencia de los tatuajes. Una revisión exhaustiva de estos métodos hasta 2019 se puede encontrar en [11].

En este trabajo, nos limitamos a resumir brevemente las soluciones de recuperación de tatuajes basadas en aprendizaje profundo debido a su éxito reportado y su excelente rendimiento en diversas tareas de reconocimiento de patrones y visión por computadora. En 2019, Han et al. [11] introdujeron un sistema capaz de aprender la detección de tatuajes y una representación compacta de tatuajes de manera conjunta en una única red neuronal profunda mediante aprendizaje multitarea. Siguiendo esta idea, Zhang et al. [47] propusieron un marco basado en redes neuronales profundas para la detección de tatuajes y la re-identificación de individuos. Se informó que sus tasas de recuperación mejoraron en la base de datos WebTattoo [11] cuando se activaba el módulo de detección. Nicolás-Díaz et al. [33] evaluaron modelos de redes neuronales profundas disponibles públicamente, preentrenados con grandes bases de datos de imágenes genéricas, para la identificación de tatuajes y mostraron que estas redes neuronales pueden lograr altas tasas de recuperación sin siquiera afinarlas para la tarea específica de recuperación de tatuajes. Los mismos autores también hicieron pública la base de datos BIVTatt en [33]. Además, Nicolás-Díaz et al. [34] presentaron un mecanismo de agrupamiento de atención para adaptar las capas convolucionales intermedias de las redes neuronales profundas preentrenadas a la tarea de identificación de tatuajes.

Los sistemas de recuperación de tatuajes mencionados anteriormente se basan en técnicas de aprendizaje profundo, que requieren una cantidad considerable de datos de entrenamiento para lograr un buen rendimiento. En la Tabla 1, resumimos las principales características de las bases de datos utilizadas para tales fines. Cabe destacar que, por un lado, Tatt-C y PinTatt no están disponibles para la comunidad investigadora. Por otro lado, las bases de datos existentes solo consisten en unas pocas categorías o muestras de tatuajes que no cubren la mayoría de los escenarios de la vida real. Esto podría hacer que los algoritmos entrenados en esas bases de datos sean propensos al sobreajuste.

La síntesis de imágenes ha surgido recientemente como una solución confiable tanto para las preocupaciones de privacidad como para la falta de datos de entrenamiento. Hasta donde sabemos, pocos estudios han abordado la generación (semi-)sintética de imágenes de tatuajes para fines de recuperación. Los métodos existentes para crear imágenes faciales con tatuajes están diseñados principalmente para la visualización de tatuajes en imágenes de piel controladas o para la segmentación de tatuajes.

–
Identificación de Tatuajes: Para este caso de uso, se investiga la coincidencia de diferentes fotos de la misma imagen de tatuaje del mismo sujeto a lo largo del tiempo. Al buscar 11,921 imágenes de tatuajes no recortadas contra una galería de 100,000 tatuajes inscritos tal como están (es decir, no recortados), el algoritmo con mejor rendimiento (B31I) alcanzó una tasa de aciertos en el ranking 10 de 72.1% (tasa de errores de 27.9%). Sección 4.1.1. Cuando el mismo conjunto de imágenes de prueba fue recortado manualmente alrededor del contenido principal del tatuaje antes del procesamiento del algoritmo (las imágenes de la galería permanecieron sin recortar), se observaron mejoras notables en la precisión. El algoritmo con mejor rendimiento (B11I) logró una tasa de aciertos en el ranking 10 de 84.8% (tasa de errores de 15.2%). Esto cuantifica que un ˜13% adicional de las búsquedas se pueden coincidir al usar imágenes de prueba recortadas en lugar de imágenes no recortadas, lo que constituye una disminución de casi el 50% en la tasa de errores. Sección 4.1.2. Los factores que influyen en la precisión incluyen:
– Algoritmos: La precisión del reconocimiento de tatuajes depende de la implementación de la tecnología principal. Aunque el rendimiento del algoritmo varió entre los dos participantes, se observaron mejoras notables en la precisión de los algoritmos presentados por ambos proveedores en fases sucesivas de la prueba.
– Relación Tatuaje-Imagen: El tamaño del tatuaje en relación con la imagen completa afecta las tasas de recuperación de tatuajes. Para todos los algoritmos, más del 60% de los tatuajes que se capturan como un porcentaje muy pequeño (menos o igual al 1%) de la imagen completa no se coincidieron dentro de las 300 imágenes recuperadas. Las tasas de errores disminuyen a medida que aumenta la relación tatuaje-imagen, lo que indica que los tatuajes que ocupan un mayor porcentaje de la imagen tienen una mayor probabilidad de ser coincididos. Sección 4.1.3.
– Brillo de la Piel y Contraste del Tatuaje: El impacto del brillo de la piel y el contraste del tatuaje en la precisión de coincidencia depende del algoritmo. Todos los algoritmos parecen ser sensibles a contrastes muy bajos. Algunos algoritmos no se ven muy afectados por el brillo de la piel, mientras que otros tienen dificultades para coincidir imágenes con un brillo de piel muy bajo. Sección 4.1.4.
– Tamaño de la Galería: El tamaño de la galería tiene un impacto en la precisión. Aunque se observa una disminución en la precisión en todos los algoritmos cuando se aumenta el tamaño de la galería, la precisión disminuye a tasas bastante benignas. Para los algoritmos más precisos, la disminución en la tasa de aciertos varía entre 2% y 3% cuando la galería se incrementa en un factor de 5. Estas tendencias no lineales demuestran la viabilidad del reconocimiento de tatuajes en bases de datos de tamaño operativo. Sección 4.1.5.
Bocetos: Para este caso de uso, se investiga la coincidencia de bocetos con imágenes de tatuajes inscritas en una base de datos. En una galería de 100,000 tatuajes inscritos, el algoritmo con mejor rendimiento (A30I) alcanza una tasa de aciertos en el ranking 10 de 40%. Aumentar el número de candidatos revisados ofrece beneficios significativos en precisión, con una tasa de aciertos en el ranking 300 que llega hasta el 71%. Sección 4.2.
Multiespectral: Para este caso de uso, se investiga la coincidencia de imágenes de tatuajes recolectadas en el espectro SWIR contra imágenes de tatuajes visibles inscritas en una base de datos. En SWIR, todos los algoritmos mostraron el mejor rendimiento en coincidencia en imágenes de búsqueda recolectadas entre los longitudes de onda de 1100 nm a 1300 nm. El algoritmo con mejor rendimiento (B11I) logró una tasa de aciertos en el ranking 10 de 95% en imágenes recolectadas en las longitudes de onda de 1100 nm y 1200 nm. En el mismo conjunto de tatuajes de prueba, el mejor rendimiento en coincidencia sigue siendo en imágenes recolectadas en el espectro visible, lo cual se observa en todos los algoritmos. Sección 4.3.
Detección de Tatuajes: Para este caso de uso, se investiga la capacidad para detectar si una imagen contiene un tatuaje o no. En un conjunto de datos mixto de 131,662 imágenes de tatuajes y 125,253 imágenes sin tatuajes, los algoritmos con mejor rendimiento (B21D, B30D, y B31D) logran una tasa de detección de verdaderos positivos de 99.5% cuando la tasa de detección de falsos positivos se establece en 1%. Sección 4.4.
Localización de Tatuajes: Para este caso de uso, se investiga la capacidad de localizar y segmentar uno o más tatuajes contenidos en una imagen. En un conjunto de 12,613 tatuajes contenidos en 10,926 imágenes, los algoritmos con mejor rendimiento (A20D, A21D, A30D, y A31D) logran una precisión de localización espacial de 89.5%. Sección 4.5.
Compensación entre Velocidad y Precisión: Existe una compensación observable entre la velocidad de búsqueda y la precisión para las dos organizaciones participantes. Los algoritmos más rápidos tienden a ser menos precisos, y los algoritmos más precisos tienden a ser más lentos. La precisión en el ranking 10 varía entre 10.3% (A10I) y 72.1% (B31I), mientras que las velocidades de búsqueda varían entre 2.0 segundos (A10I, A11I) y 255.4 segundos (B11I) para buscar una imagen de tatuaje contra una galería de 100,000. Ambos participantes aprovecharon el marco de aprendizaje profundo de código abierto, Caffe [2]. Para el tiempo, algunos algoritmos se ejecutaron en una sola GPU NVIDIA Tesla K40, y algunas presentaciones se ejecutaron en una sola CPU Intel Xeon E5-2695 v3. Nota que las diferencias en hardware dificultan las comparaciones directas de tiempos, pero medir el tiempo absoluto sigue siendo útil ya que diferentes aplicaciones tendrán diferentes requisitos de velocidad. Sección 4.1.6. (Tattoo Recognition Technology - Evaluation (Tatt-E) Performance of Tattoo Identification Algorithms)
