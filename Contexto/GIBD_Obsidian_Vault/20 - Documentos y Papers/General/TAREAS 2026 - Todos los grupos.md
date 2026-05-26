---
aliases: [TAREAS 2026 - Todos los grupos]
tags:
  - grupo/general
  - estado/util
  - tipo/documento
formato_original: docx
fecha_modificacion: 2026-05-21
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/2026/TAREAS 2026 - Todos los grupos.docx"
tamanio_bytes: 17247
---

# TAREAS 2026 - Todos los grupos

Ruta interna: `GIBD/2026/TAREAS 2026 - Todos los grupos.docx`

---

Grupo 1 — Sonido
Rodrigo Rodriguez 
Benay Franco
Objetivo General
Investigar, desarrollar y evaluar técnicas de búsqueda por similitud de sonido utilizando modelos de representación profunda, aprendizaje métrico y búsqueda semántica sobre distintos tipos de audio.
Líneas de trabajo
Búsqueda por similitud de sonidos cotidianos/eventos acústicos.
Búsqueda por similitud musical.
Evaluación comparativa con técnicas tipo Shazam y sistemas modernos de embeddings de audio.
Tipos de sonidos a trabajar
Sonidos/eventos acústicos
Gotas de agua
Lluvia
Vidrio roto
Golpes
Explosiones
Pasos
Sirenas
Perros/ladridos
Motores
Alarmas
Sonidos urbanos
Música
Fragmentos de canciones
Covers/versiones nuevas
Distintas grabaciones de una misma canción
Distorsiones, ruido y cambios de calidad
Variaciones de tempo o tono
Tareas
Realizar revisión bibliográfica y análisis del estado del arte.
Investigar técnicas utilizadas en:
Audio retrieval
Music retrieval
Audio fingerprinting
Contrastive learning para audio
Investigar modelos modernos:
AudioCLIP
CLAP
VGGish
YAMNet
AST (Audio Spectrogram Transformer)
OpenL3
Analizar funcionamiento de:
Shazam
Google
Obtener, limpiar y preparar datasets.
Diseñar consultas etiquetadas y ground truth.
Generar pipelines experimentales reproducibles.
Implementar extracción de características:
Espectrogramas
Mel Spectrograms
MFCC
Embeddings profundos
Implementar búsqueda por similitud utilizando:
Cosine similarity
FAISS
ANN
Índices métricos
Evaluar robustez:
Ruido
Compresión
Cambios de volumen
Cambios de velocidad
Diseñar e implementar experimentos con:
Siamese Networks
Triplet Loss
Contrastive Learning
Analizar resultados y generar métricas:
Precision@K
Recall@K
mAP
Identificar resultados publicables y redactar papers.
Posibles aportes novedosos
Comparar retrieval clásico vs embeddings modernos multimodales.
Evaluar robustez extrema en retrieval de sonido.
Investigar reducción dimensional sobre embeddings de audio.
Explorar retrieval semántico de eventos acústicos.

Grupo 2 — Video
León Castiglioni
Valentino Chiozza
Franco Gutierrez 
Mateo Mista
Franco Rodríguez
Objetivo General
Desarrollar técnicas de búsqueda por similitud y clasificación automática de violencia/eventos significativos utilizando representaciones profundas de video.
Líneas de trabajo
Línea A — Video Retrieval
Búsqueda por similitud de videos o fragmentos de video.
Línea B — Detección de violencia/eventos significativos
Clasificación automática de:
Peleas
Robos
Accidentes
Explosiones
Caídas
Eventos abruptos
Tareas
Revisión bibliográfica del estado del arte.
Investigar:
Video retrieval
Action recognition
Violence detection
Event detection
Preparar datasets y ground truth.
Investigar datasets públicos:
UCF-Crime
Hockey Fight
RWF-2000
XD-Violence
Diseñar datasets propios y consultas etiquetadas.
Implementar extracción de características:
CNN 3D
TimeSformer
VideoMAE
ViViT
I3D
Diseñar pipelines temporales.
Implementar búsqueda por similitud de video.
Evaluar:
Robustez
Escalabilidad
Generalización
Implementar clasificación de violencia.
Comparar:
Modelos espaciales
Modelos temporales
Transformers de video
Diseñar experimentos en tiempo real.
Analizar falsos positivos:
Juegos
Deportes
Movimientos bruscos no violentos
Evaluar métricas:
Accuracy
F1
Recall
Precision
ROC-AUC
Redactar papers y reportes.
Posibles aportes novedosos
Retrieval y detección unificados en un mismo embedding.
Detección basada en “cambio abrupto de dinámica”.
Few-shot violence detection.
Representaciones compactas para video retrieval masivo.

Grupo 3 — Texto
Objetivo General
Investigar técnicas modernas de representación semántica y clasificación de texto utilizando embeddings profundos y búsqueda semántica.
Línea principal
Comparación entre técnicas previas de análisis de sentimientos y embeddings modernos utilizando BGE-M3.
Tareas
Releer y documentar el trabajo previo sobre análisis de sentimientos.
Preparar nuevamente la BD de comentarios de películas.
Investigar:
Sentence embeddings
Semantic retrieval
Cross-lingual embeddings
Implementar y evaluar:
BGE-M3
SBERT
BETO
E5
Comparar resultados con modelos anteriores.
Evaluar:
Accuracy
F1
Robustez
Diseñar nuevos experimentos.
Nuevas líneas sugeridas
A — Búsqueda semántica
Construir un sistema capaz de recuperar textos semánticamente similares aunque no compartan palabras.
B — Clasificación few-shot
Evaluar clasificación con pocos ejemplos utilizando embeddings modernos.
C — Clustering semántico
Agrupar automáticamente comentarios similares.
D — Detección de similitud textual
Aplicar embeddings para:
detección de plagio
recuperación documental
recomendación de contenido
E — Retrieval jurídico
Explorar aplicaciones futuras para PTAH-Jurídico.
Tareas adicionales
Diseñar datasets de consultas etiquetadas.
Implementar pipelines de retrieval.
Evaluar reducción dimensional sobre embeddings textuales.
Integrar experimentos con FAISS o índices ANN.
Redactar publicaciones.
Posibles aportes novedosos
Comparación profunda entre embeddings clásicos y modernos.
Evaluación semántica en español.
Retrieval semántico jurídico.
Few-shot semantic classification.

Grupo 4 — Imágenes
Objetivo General
Evaluar la capacidad de generalización de técnicas de búsqueda por similitud visual sobre bases masivas y heterogéneas de imágenes.
Contexto
Actualmente existen experimentos sobre bases pequeñas y especializadas (~10k imágenes). El objetivo es escalar a una base genérica de aproximadamente 1 millón de imágenes.
Tareas
Revisar el estado del arte en:
Image retrieval
Visual embeddings
Large-scale retrieval
Preparar consultas etiquetadas y ground truth.
Diseñar protocolos experimentales.
Implementar extracción de características utilizando:
ResNet50
Vision Transformers (ViT)
CLIP
DINOv2
Comparar:
Modelos CNN
Transformers visuales
Implementar búsquedas utilizando:
FAISS
ANN
Índices métricos
Evaluar:
Escalabilidad
Robustez
Tiempo de respuesta
Uso de memoria
Analizar comportamiento en:
imágenes heterogéneas
múltiples categorías
ruido visual
Comparar con resultados previos en:
logos
marcas de ganado
Evaluar impacto de reducción dimensional.
Generar resultados publicables.
Posibles aportes novedosos
Generalización de embeddings sobre dominios heterogéneos.
Retrieval masivo multimodal.
Comparación ResNet vs ViT a gran escala.
Evaluación de degradación al aumentar tamaño y diversidad de la BD.

Grupo 5 — Reducción de la Dimensionalidad
Objetivo General
Investigar técnicas de reducción dimensional aplicadas a embeddings de alta dimensionalidad para mejorar almacenamiento, eficiencia y compatibilidad con índices métricos.
Contexto
Se trabajará con vectores entre ~700 y ~200.000 dimensiones, buscando reducirlos a menos de 100 dimensiones preservando capacidad discriminativa.
Tareas
Revisar estado del arte en:
reducción dimensional
compact embeddings
metric indexing
Investigar e implementar:
PCA
UMAP
Autoencoders
VAEs
Random Projection
Product Quantization
Evaluar preservación de similitud.
Integrar reducción dimensional en:
imágenes
texto
audio
video
Repetir experimentos previos sobre:
logos
marcas de ganado
retrieval multimodal
Analizar impacto en:
Precision@K
Recall
mAP
Evaluar:
velocidad
memoria
escalabilidad
Investigar compatibilidad con:
índices métricos
ANN
FAISS
Generar visualizaciones de embeddings.
Analizar separabilidad semántica.
Redactar publicaciones científicas.
Posibles aportes novedosos
Embeddings ultracompactos para retrieval multimodal.
Comparación reducción lineal vs no lineal.
Evaluación sobre múltiples modalidades.
Compatibilidad con índices métricos reales.

Grupo 6 — Desarrollo de Aplicaciones de Búsqueda por Similitud (Web y Móvil)
Equipo:
Emanuel Davezac
Renato Bonín
Thiago J. Gomez Kehler

Objetivo General
Diseñar e implementar aplicaciones funcionales web y móviles para búsqueda por similitud utilizando los modelos y bases desarrollados por el proyecto.
Objetivos específicos
Integrar frontend y backend funcional.
Implementar búsqueda multimodal.
Aplicar metodologías modernas asistidas por IA y agentes.
Tareas
Revisar prototipos existentes.
Diseñar arquitectura completa:
frontend
backend
APIs
almacenamiento
Implementar aplicación web funcional.
Implementar aplicación móvil funcional.
Integrar:
búsqueda por imágenes
búsqueda textual
futuras búsquedas multimodales
Diseñar sistema de consultas y resultados.
Implementar autenticación y administración básica.
Investigar y aplicar:
agentes de desarrollo
programación asistida por IA
Spec Driven Development
Generar especificaciones formales antes de programar.
Diseñar flujos de trabajo reproducibles.
Documentar arquitectura y decisiones técnicas.
Integrar motores de búsqueda:
FAISS
APIs internas
Diseñar pruebas de usabilidad y rendimiento.
Implementar logging y monitoreo.
Generar demos funcionales para transferencia tecnológica.
Tecnologías sugeridas (se pueden cambiar!)
Backend
Python
FastAPI
Docker
Frontend Web
React / Next.js
Mobile
Flutter o React Native
Bases vectoriales
FAISS
Qdrant
Milvus
Posibles aportes novedosos
Uso de agentes IA en desarrollo real.
Desarrollo guiado por especificaciones.
Plataforma multimodal unificada.
Retrieval multimodal integrado en tiempo real.
