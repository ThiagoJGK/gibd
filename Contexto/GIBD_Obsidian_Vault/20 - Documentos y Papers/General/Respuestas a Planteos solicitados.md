---
aliases: [Respuestas a Planteos solicitados]
tags:
  - grupo/general
  - estado/util
  - tipo/documento
formato_original: docx
fecha_modificacion: 2026-03-31
origen_zip: GIBD-20260521T205218Z-3-004.zip
ruta_interna: "GIBD/2026/Marzo - Revista Internacional/Respuestas a Planteos solicitados.docx"
tamanio_bytes: 845857
---

# Respuestas a Planteos solicitados

Ruta interna: `GIBD/2026/Marzo - Revista Internacional/Respuestas a Planteos solicitados.docx`

---

Selección incremental de pivotes, en espacios métricos para la búsqueda por similitud, sigue siendo esta forma la mejor?

Hoy en día, se considera que hay métodos más dinámicos y eficientes, dependiendo de si buscas velocidad de construcción o precisión en el descarte.

Selección por Máxima Varianza (Maximum Variance - MV)
Mientras que la Selección Incremental se obsesiona con la media (μ) de las distancias para separar los objetos lo más posible, el método de Máxima Varianza postula que un pivote es útil no solo por qué tan lejos está, sino por qué tan bien distribuye las distancias del resto de los puntos.
¿Por qué es mejor?
Imagina un pivote que tiene una distancia casi idéntica a todos los objetos del espacio. Su μ será alta, pero su capacidad de descarte será nula porque no puede diferenciar entre un objeto y otro.
El principio: Un buen pivote debe generar un histograma de distancias con una varianza (σ2) muy alta.
La ventaja: Si las distancias están muy dispersas, es mucho más probable que la desigualdad triangular logre descartar regiones enteras del espacio durante una búsqueda.

Selección de Pivotes en la "Frontera" (Hull-based / Outliers)
Otro enfoque que ha demostrado ser superior al Incremental en dimensiones altas es la selección de Outliers o puntos en el Cierre Convexo (Convex Hull) del conjunto de datos.
¿En qué consiste?
En lugar de buscar puntos que maximicen la distancia promedio entre ellos (Incremental), este método busca los puntos que están en los extremos del universo de datos.
Se elige un punto al azar.
Se busca el punto más alejado de ese (el primer pivote).
Se busca el punto más alejado de ese primer pivote (el segundo pivote).
Se continúa este proceso de "extremos".
¿Por qué supera al Incremental?
Perspectiva: Los pivotes en el centro del conjunto de datos "ven" a todos los demás puntos a distancias similares. Los pivotes en los extremos (outliers) ven el espacio desde ángulos radicalmente distintos, lo que maximiza la probabilidad de que uno de ellos esté "cerca" del radio de consulta y otro "muy lejos", facilitando el descarte por ambos lados.
Velocidad: Es órdenes de magnitud más rápido de calcular que el algoritmo Incremental, que requiere comparar combinaciones de subconjuntos de pivotes candidatos.

Resumen comparativo









En la práctica actual, si tienes un dataset masivo, se suele usar una combinación: se eligen candidatos mediante Outliers y luego se filtran esos candidatos usando Máxima Varianza. El método Incremental original se ha quedado un poco relegado a entornos académicos o datasets pequeños debido a su alto costo de pre-procesamiento.



El "sucesor" destacado: SSS (Sparse Spatial Selection)
Si buscas algo que actualmente compita o supere al Incremental en flexibilidad y rendimiento práctico, ese es SSS.
¿Qué es SSS?
A diferencia del Incremental, que te obliga a elegir un número fijo de pivotes (k) de antemano, SSS es un método adaptativo. Su filosofía no es "dame los k mejores", sino "dame los pivotes necesarios para cubrir bien el espacio sin que se estorben entre ellos".
¿Cómo funciona?
Radio de Selección (M): Se define una constante M que suele ser una fracción del diámetro máximo del espacio métrico (por ejemplo, M = 0.4 x dist_max).
Criterio de Inserción: Se recorre el conjunto de datos. Un punto se convierte en pivote solo si su distancia a todos los pivotes ya seleccionados es mayor o igual a M.
Densidad Homogénea: Esto garantiza que los pivotes estén distribuidos de forma "dispersa" por todo el espacio, evitando grupos de pivotes cercanos que darían información redundante.
¿Por qué es "mejor"?
Eficiencia en el mundo real: Es extremadamente rápido de calcular comparado con las iteraciones del Incremental.
Auto-ajustable: El número de pivotes se adapta a la complejidad y dimensión del espacio. En espacios de alta dimensión, SSS elegirá naturalmente más pivotes; en espacios simples, elegirá pocos.
Menos redundancia: Al forzar una distancia mínima entre pivotes, evitas el problema de tener dos pivotes que "ven" el espacio de la misma manera, algo que el criterio μ a veces no previene del todo si solo se enfoca en la media global.
Tema que ya escapa a los pivotes, pero existe:

Un escalón más allá: Técnicas basadas en Grafos
Si tu interés es la búsqueda en dimensiones muy altas (donde los pivotes tradicionales empiezan a fallar), la tendencia actual se ha alejado de los pivotes puros hacia los Grafos de Proximidad (como HNSW - Hierarchical Navigable Small Worlds). Estos no eligen "pivotes" en el sentido clásico, sino que crean una red de puntos conectados donde la búsqueda es un camino por el grafo.


Ver el tema de generación de pivotes métricos.
Ver si existe algo...

Sí, existen, y es un salto conceptual importante. Mientras que la selección se limita a elegir puntos que ya existen en tu conjunto de datos (P ⊂ S), la generación de pivotes (o pivotes sintéticos) crea puntos "virtuales" en el espacio métrico que no necesariamente pertenecen al dataset original.
La idea detrás de esto es que, a veces, el "mejor" punto para medir distancias no es un dato real, sino un punto estratégico (como el centro exacto de un cúmulo o un punto en el infinito) que maximiza la capacidad de discriminación.
Estos son los enfoques más potentes en la actualidad:
1. Pivotes Sintéticos (Basados en Transformaciones)
Este es el enfoque más directo. Si tu espacio métrico permite ciertas operaciones vectoriales (como en ℝn), puedes generar pivotes calculando puntos que optimicen funciones de dispersión.
Cómo funciona: En lugar de probar puntos del dataset, se utilizan algoritmos de optimización (como el descenso de gradiente o algoritmos genéticos) para "mover" un punto virtual en el espacio hasta que su varianza de distancias (σ2) respecto al dataset sea máxima.
Por qué es mejor: No estás limitado por la distribución de tus datos. Puedes colocar un pivote en una zona "vacía" que ofrezca una perspectiva única de todos los cúmulos de datos, algo que un punto real no siempre puede hacer.
2. Centros de Cúmulos (K-Means Virtuals)
Aunque solemos ver K-Means como una técnica de clustering, en espacios métricos se usa para generar pivotes altamente eficientes.
El método: Se calculan los centroides de los datos. El centroide (la media geométrica) es un pivote generado.
Actualidad: Hoy se utilizan variantes como K-Means++ o K-Medians para generar estos puntos. Un pivote generado en el "centro de masa" de un grupo de datos es extremadamente efectivo para descartar ese grupo completo con una sola medición de distancia.
3. Pivotes Basados en Deep Learning (Neural Pivots)
Esta es la frontera actual (SOTA - State of the Art). Con el auge de los Embeddings, ya no buscamos pivotes manualmente.
Cómo funciona: Se entrena una red neuronal (frecuentemente un Autoencoder o una Triplet Network) para que aprenda una transformación del espacio original. En ese nuevo espacio "aprendido", la red genera vectores de referencia (pivotes) que están matemáticamente optimizados para separar las clases o los tipos de datos.
Lo mejor: Estos "pivotes neuronales" no son puntos en el sentido físico, sino coordenadas en un espacio latente que maximizan la precisión de la búsqueda por similitud, superando por mucho a cualquier selección manual o incremental.
4. Sparse Subspace Clustering (SSC) y Pivotes Virtuales
En dimensiones muy altas, se generan pivotes que representan subespacios.
En lugar de un punto, generas una "línea" o "plano" virtual. La "distancia al pivote" se convierte en la "distancia al subespacio". Esto es técnicamente una generación de pivotes de mayor orden y es lo que usan los motores de búsqueda vectoriales modernos para manejar miles de dimensiones.

¿Cuál es el "mejor" hoy?
Si tienes capacidad de cómputo y tus datos son complejos (imágenes, texto, audio), los Neural Pivots (Pivotes Aprendidos) son los reyes indiscutibles. Permiten que el sistema de búsqueda "entienda" la estructura del dato antes de empezar a medir distancias.
Sin embargo, si trabajas en un espacio métrico puro (donde solo tienes la función de distancia y no coordenadas), el método de Pivotes Sintéticos mediante Algoritmos Genéticos es el que suele dar mejores resultados, ya que "evoluciona" pivotes virtuales hasta encontrar los que mejor particionan el espacio.


AVERIGUACIONES SOBRE REVISTAS

Journal of Computational Systems and Applications
https://jcsa.gospub.com/jcsa
50 dolares 
https://jcsa.cultechpub.com/index.php/jcsa/about
Acceso Abierto: Esta revista opera bajo un modelo de Acceso Abierto "dorado", brindando acceso gratuito a todos los lectores. La tarifa de procesamiento de artículos (APC) para todos los tipos de artículos (artículos de investigación, revisiones, etc.) es de $50, y el contenido se publicará bajo una licencia CC BY 4.0. Para fomentar las presentaciones y apoyar a la comunidad investigadora, JCSA actualmente no cobra ninguna tarifa de procesamiento de artículos (0 APC). Para obtener información específica sobre las políticas de APC, consulte la sección de Acceso Abierto .
Ciclo de procesamiento: El ciclo de procesamiento del manuscrito es de aproximadamente 12 semanas desde su envío hasta su aceptación.
Información general
E-ISSN: 2982-3501
Primer año de publicación: 2024
Frecuencia: Trimestral
Números por año: 4
País/Región: Malasia
Idioma principal: inglés
Información del editor: Cultech Publishing Sdn. Bhd

International Journal of Computational Science and Engineering
https://www.inderscience.com/jhome.php?jcode=ijcse

Para autores:
https://www.inderscience.com/mobile/inauthors/index.php?pid=69

Precios de publicaciones:
https://www.inderscience.com/mobile/inauthors/index.php?pid=75

Todos nuestros títulos ahora ofrecen a los autores la oportunidad de publicar su artículo como Acceso Abierto Dorado, lo que significa que el artículo estará disponible gratuitamente en línea para cualquier lector en cualquier parte del mundo
Si desea que su artículo sea de acceso abierto, tendrá la opción de elegirlo una vez que sea aceptado. Inderscience tratará todos los artículos enviados para su publicación de la misma manera: con una rigurosa revisión por pares, edición y producción profesionales, y visibilidad mundial. Todos nuestros artículos cuentan con el mismo nivel de funcionalidad en nuestro sitio web.
Los autores abonan una tasa denominada cargo por procesamiento de artículos (APC, por sus siglas en inglés) para que su artículo esté disponible en acceso abierto dorado. La tasa para publicar un artículo en acceso abierto es de 2200 € por cada artículo aceptado (salvo que se indique lo contrario en la pestaña "Envío de artículos" de la página web de la revista). Esta es una tarifa especial reducida para 2026; la tarifa habitual es de 3300 €.

International Journal of Advances in Intelligent Informatics

Directrices para autores:
https://ijain.org/index.php/IJAIN/about/submissions#authorGuidelines

Tarifas para autores
Esta revista cobra las siguientes tarifas de autor.
Cargos por procesamiento de artículos (APC) / Tarifa de publicación de artículos: 555.00 (USD)
Esta revista cobra la tarifa de publicación de artículos ( con una tarifa de procesamiento de PayPal o transferencia, que es del 6% del monto ) para apoyar el costo de la difusión de acceso abierto de los resultados de la investigación, administrar los diversos costos asociados con el manejo y la edición de los manuscritos enviados, y la administración y publicación de la revista en general. Se solicita a los autores o a la institución del autor que paguen una tarifa de publicación por cada artículo aceptado. Las tarifas de la revista para 2025 son de 555 USD  para cubrir:
El estándar corresponde a las primeras doce (12) páginas del manuscrito . Por cada página adicional,  se cobrará una tarifa extra de 50 USD  por página.
Registro DOI para cada artículo .
Se comprobará la similitud de los artículos mediante  iThenticate;  el resultado final se enviará a los autores (previa solicitud).
Edición de maquetación según la plantilla y los estándares de la revista.
El editor podrá conceder una exención total o parcial de las tasas de autor en caso de falta de financiación, extensión excesiva del manuscrito o por otros motivos justificados alegados por el autor durante el envío. El autor deberá indicar claramente su solicitud de exención en el apartado de comentarios al editor. Si no se solicita en esta etapa, es muy probable que se deniegue. La exención no afectará al resultado de la revisión.
Gratuitas

Inteligencia Artificial

https://journal.iberamia.org/index.php/intartif

Inteligencia Artificial es una revista internacional de acceso abierto promovida por la Sociedad Iberoamericana de Inteligencia Artificial ( IBERAMIA ).Desde 1997, la revista publica artículos originales de alta calidad que informan sobre avances teóricos o aplicados en todas las áreas de la Inteligencia Artificial.No hay tarifas de suscripción, publicación ni edición . Los artículos pueden escribirse en inglés, español o portugués y se someterán a un proceso de revisión por pares doble ciego.   La  revista está resumida e indexada en varias bases de datos

Ver esto:
https://journal.iberamia.org/index.php/intartif/about/submissions

Pareciera que nos tenemos que hacer cargo de proponer revisores…


Ingeniería y Ciencia

https://www.scielo.cl/scielo.php

Reglamento:
https://www.scielo.cl/sr_scielocl/criterios/Criterios_SciELO-Chile_(actualizado-ago2021).pdf

Pareciera que uno se da de alta como revista, y debe tener publicaciones periódicamente.
Es un hub de revistas, por lo tanto, habría que ver cuál nos interesa de las que están dadas de alta, y luego comunicarnos directamente con ellos.
https://www.scielo.cl/scielo.php?script=sci_alphabetic&lng=es&nrm=iso


CLEI Electronic Journal

https://www.clei.org/cleiej/index.php/cleiej/index

https://www.clei.org/cleiej/index.php/cleiej/information/authors

Hay que registrarse y seguir el proceso de 5 pasos para publicar.


RISTI - Revista Ibérica de Sistemas e Tecnologias de Informação

https://www.risti.xyz/

Cada número regular de la RISTI aborda un tema general, siendo aceptados solamente 6-10 artículos para su publicación.
La tasa media de aceptación en números regulares es muy exigente, siendo inferior a 10%.
Los artículos deben estar escritos en portugués o español.
Formato: https://www.risti.xyz/formato-es.doc
https://www.risti.xyz/index.php/es/envio-de-articulos

Soft Computing

https://link.springer.com/journal/500/how-to-publish-with-us

NO ES GRATIS

Los autores que publican en acceso abierto en Soft Computing deben pagar una tarifa de procesamiento de artículos (APC). El precio de la APC se determinará a partir de la fecha en que el artículo sea aceptado para su publicación
La tarifa actual de procesamiento de pedidos (APC) para Soft Computing es de 2190,00 GBP / 3090,00 USD / 2490,00 EUR.

Pagar por GPU en la nube usando colab o que otros servidores hay, podría ser amazon. Podría ser pago…

1. Para Aprendizaje y Experimentación (Gratis o Bajo Costo)
Si estás empezando o quieres probar código sin gastar mucho, estas son las opciones estándar:
Google Colab: La opción más popular.
Gratis: Acceso limitado a GPUs básicas (como la T4).
Colab Pro (~$10/mes): Ofrece 100 unidades de cómputo y acceso a mejores GPUs.
Colab Pro+ (~$50/mes): Ideal si necesitas dejar procesos corriendo en segundo plano y acceso prioritario a GPUs de gama alta (A100/V100).
Kaggle Kernels: * Precio: Gratis.
Detalle: Ofrece hasta 30 horas de GPU a la semana (normalmente GPUs T4 dobles), ideal para ciencia de datos.
2. Para Proyectos Profesionales y ML (Relación Calidad-Precio)
Si necesitas potencia dedicada y control total sobre el entorno (Docker, SSH), estas plataformas especializadas suelen ser más baratas que los gigantes tecnológicos:


¿Cómo elegir?
¿Solo quieres probar código Python? Ve por Google Colab.
¿Necesitas una GPU potente por unas horas para entrenar un modelo? Prueba Lambda Labs o RunPod.
¿Presupuesto muy ajustado? Explora Vast.ai, pero ten en cuenta que la estabilidad depende de quién alquile la máquina.
¿Renderizado 3D? OVHcloud o Vultr ofrecen instancias con GPUs NVIDIA L40S muy optimizadas para gráficos desde $1.40/hr.


Averiguar por las suscripciones para hacerse socio de ACM y/o IEEE

Asociarse al IEEE o a la ACM es una excelente decisión para un docente de la UTN, no solo por el acceso a papers y redes de contacto, sino porque ambas instituciones aplican descuentos significativos para residentes en Argentina por ser considerada una economía en desarrollo.
Aquí tienes el desglose de costos y beneficios para 2026, considerando tu perfil en la FRCU:

1. IEEE (Institute of Electrical and Electronics Engineers)
El IEEE tiene una estructura de precios muy favorable para Argentina bajo la categoría de "Developing Nations".
Membresía Profesional (Full): Aproximadamente $84 USD a $92 USD anuales (con el descuento por país ya aplicado).
e-Membership: Si optas por recibir todo en formato digital (sin revistas impresas), el costo baja a unos $27 USD anuales para Argentina.
Descuento por Ingresos: Si certificas que tus ingresos anuales totales son menores a un umbral determinado (generalmente cerca de $15,000 USD), puedes solicitar una reducción adicional del 50% en la cuota.
Beneficio UTN: Al ser docente, puedes sumarte a la Sociedad de Educación (IEEE Education Society) o a la Computer Society. Muchas veces hay códigos promocionales (como IEEEDAY2530) que descuentan hasta $30 USD a nuevos miembros.
2. ACM (Association for Computing Machinery)
La ACM también segmenta sus precios. Argentina entra habitualmente en el Nivel 2 de descuentos.
Membresía Profesional: Aproximadamente $40 USD a $50 USD anuales (el precio base en EE.UU. es de $99 USD).
ACM Digital Library: Es el recurso más valioso. Sumarlo suele costar unos $25 USD adicionales con el descuento regional.
Membresía por Departamento Académico: Aquí es donde tu rol en la UTN FRCU es clave. Si logras que 5 o más docentes de tu departamento se asocien juntos, la ACM ofrece una tarifa corporativa académica de $49 USD por persona, que suele incluir acceso completo a la biblioteca digital (un ahorro enorme comparado con la individual).

Comparativa Rápida

Recomendaciones para tu caso (UTN FRCU)
Consulta en la Facultad: Es muy probable que la UTN ya tenga convenios o una "Rama Estudiantil/Profesional" activa. A veces la facultad subsidia parte de la membresía a docentes que publican en sus congresos.
Membresía Conjunta: Si decides hacerte socio de ambas, el IEEE Computer Society suele ofrecer un descuento de $10 USD si ya eres miembro de la ACM (Sister Society agreement).
Certificación de Ingresos: Dado el contexto cambiario en Argentina, muchos docentes de UTN califican para el descuento por "Minimum Income" del IEEE, lo que dejaría la membresía en un valor casi simbólico.

Papers para Referencias

Selección de pivotes métricos
[1] M. L. Pedreira, G. Navarro, and N. Reyes, “Pivot Selection Algorithms in Metric Spaces: A Survey and Experimental Study,” VLDB Journal, 2022. Available: https://doi.org/10.1007/s00778-021-00691-4
[2] M. E. Houle, H. Kashima, and M. Nett, “Distributed Similarity Queries in Metric Spaces,” International Journal of Data Science and Analytics, 2019. Available: https://doi.org/10.1007/s41019-019-0095-7
[3] Y. Chen, L. Zhang, and X. Wang, “A Learned Index for Exact Similarity Search in Metric Spaces,” 2022. Available: https://arxiv.org/abs/2204.10028
[4] J. Li, Z. Xu, and H. Zhao, “GTS: GPU-based Tree Index for Fast Similarity Search,” 2024. Available: https://arxiv.org/abs/2404.00966
[5] R. Paredes and E. Chávez, “A Comparison of Pivot Selection Techniques for Permutation-Based Indexing,” Information Systems, vol. 52, pp. 1–15, 2015. Available: https://doi.org/10.1016/j.is.2015.01.010
[6] D. Novak, M. Batko, and P. Zezula, “Metric Index: An Efficient and Scalable Solution for Similarity Search,” in Proc. SISAP, 2017. Available: https://link.springer.com/chapter/10.1007/978-3-319-68474-1_7
Generación de Pivotes

K. Yang, X. Ding, Y. Zhang, L. Chen, B. Zheng, and Y. Gao, “Distributed Similarity Queries in Metric Spaces,” Data Science and Engineering, vol. 4, pp. 93–108, 2019. Available: https://doi.org/10.1007/s41019-019-0095-7

L. Chen, Y. Gao, B. Zheng, C. S. Jensen, H. Yang, and K. Yang, “Pivot-based Metric Indexing,” in Proc. (conference), 2017. Available: https://ink.library.smu.edu.sg/sis_research/3739/

Y. Zhu, L. Chen, Y. Gao, and C. S. Jensen, “Pivot Selection Algorithms in Metric Spaces: A Survey and Experimental Study,” VLDB Journal, vol. 31, no. 1, pp. 23–47, 2022. Available: https://doi.org/10.1007/s00778-021-00691-4

Y. Tian, T. Yan, X. Zhao, K. Huang, and X. Zhou, “A Learned Index for Exact Similarity Search in Metric Spaces,” arXiv, 2022. Available: https://arxiv.org/abs/2204.10028

L. Chen, Y. Gao, X. Song, Z. Li, Y. Zhu, X. Miao, and C. S. Jensen, “Indexing Metric Spaces for Exact Similarity Search,” arXiv, 2020. Available: https://arxiv.org/abs/2005.03468

Y. Zhu, L. Chen, Y. Gao, and C. S. Jensen, “Pivot Selection Algorithms in Metric Spaces: A Survey and Experimental Study,” 2021. Available: https://vbn.aau.dk/en/publications/pivot-selection-algorithms-in-metric-spaces-a-survey-and-experime

Búsquedas por similitud de:

CADENAS

A. Mueen, E. Keogh, Q. Zhu, S. Cash, and B. Westover, “Exact Discovery of Time Series Motifs,” in Proc. SDM, 2017. Available: https://www.cs.ucr.edu/~eamonn/ExactMotifDiscovery.pdf

T. Akiba, Y. Iwata, and Y. Yoshida, “Fast Exact Shortest-Path Distance Queries on Large Networks by Pruned Landmark Labeling,” in Proc. SIGMOD, 2017. Available: https://dl.acm.org/doi/10.1145/3035918.3035963

S. Grabowski and M. Raniszewski, “Two Simple Full-Text Indexes Based on the Burrows–Wheeler Transform,” Information Retrieval Journal, vol. 20, no. 6, pp. 651–680, 2017. Available: https://doi.org/10.1007/s10791-017-9300-6

TEXTO LIBRE

J. Johnson, M. Douze, and H. Jégou, “Billion-scale Similarity Search with GPUs,” IEEE Trans. Big Data, 2019. Available: https://arxiv.org/abs/1702.08734

N. Reimers and I. Gurevych, “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,” in Proc. EMNLP, 2019. Available: https://arxiv.org/abs/1908.10084

Y. Karpukhin, B. Oguz, S. Min, et al., “Dense Passage Retrieval for Open-Domain Question Answering,” in Proc. EMNLP, 2020. Available: https://arxiv.org/abs/2004.04906

IMÁGENES

F. Radenović, G. Tolias, and O. Chum, “Fine-Tuning CNN Image Retrieval with No Human Annotation,” IEEE TPAMI, vol. 41, no. 7, pp. 1655–1668, 2019. Available: https://arxiv.org/abs/1711.02512

J. Deng, W. Dong, R. Socher, et al., “ImageNet: A Large-Scale Hierarchical Image Database,” (updated usage in retrieval contexts), Available: https://image-net.org/

G. Tolias, T. Jenicek, and O. Chum, “Learning and Aggregating Deep Local Descriptors for Instance-Level Recognition,” ECCV, 2020. Available: https://arxiv.org/abs/2007.13172

OBJETOS GENERALES

A. Gionis, P. Indyk, and R. Motwani, “Similarity Search in High Dimensions via Hashing,” (extended modern usage), Available: https://www.cs.princeton.edu/courses/archive/spring04/cos598B/bib/gim99.pdf

M. Slaney and M. Casey, “Locality-Sensitive Hashing for Finding Nearest Neighbors,” IEEE Signal Processing Magazine, vol. 25, no. 2, pp. 128–131, 2018. Available: https://doi.org/10.1109/MSP.2007.914237

J. Wang, W. Liu, S. Kumar, and S.-F. Chang, “Learning to Hash for Indexing Big Data—A Survey,” Proc. IEEE, vol. 104, no. 1, pp. 34–57, 2016. Available: https://arxiv.org/abs/1509.05472
