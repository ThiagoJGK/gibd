---
aliases: [VIT BASE Y LARGE]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2025-09-09
origen_zip: GIBD-20260521T205218Z-3-002.zip
ruta_interna: "GIBD/ViT/VIT BASE Y LARGE.docx"
tamanio_bytes: 8210
---

# VIT BASE Y LARGE

Ruta interna: `GIBD/ViT/VIT BASE Y LARGE.docx`

---

BASE


El Vision Transformer (ViT) es un modelo de codificación basado en la arquitectura Transformer, análogo a BERT, adaptado al dominio de la visión por computadora. Su entrenamiento inicial se realiza de manera supervisada en ImageNet-21k a resolución 224×224 píxeles, seguido de un ajuste fino (fine-tuning) en ImageNet/ILSVRC2012, un conjunto de aproximadamente 1 millón de imágenes distribuidas en 1.000 clases.

A diferencia de las arquitecturas convolucionales tradicionales, el ViT procesa las imágenes dividiéndolas en parches de tamaño fijo (16×16 píxeles). Cada parche se proyecta linealmente a un espacio de 768 dimensiones mediante una convolución 2D con kernel_size = stride = 16. La secuencia resultante se complementa con un token especial [CLS], destinado a la clasificación, y con embeddings posicionales absolutos que preservan la información espacial antes de ingresar a la pila de capas Transformer.

El núcleo del modelo está compuesto por 12 capas de codificación Transformer. Cada capa combina dos bloques principales:

Atención propia multi-cabeza (Self-Attention), que aprende relaciones globales entre los parches mediante proyecciones lineales de consultas, claves y valores (Linear(768→768)).

Red neuronal feed-forward (FFN), que expande la dimensionalidad a 3072, aplica la función de activación GELU, y posteriormente reduce nuevamente a 768 dimensiones (Linear(768→3072→768)).

Cada bloque incluye normalización por capas (LayerNorm) y conexiones residuales, lo que favorece la estabilidad y eficiencia del entrenamiento. Tras atravesar las 12 capas, la representación final del token [CLS] se considera un resumen global de la imagen.

Finalmente, esta representación se introduce en una capa lineal de clasificación:

(classifier): Linear(in_features=768, out_features=2, bias=True)


la cual transforma el vector de 768 dimensiones en un espacio de salida de 2 clases, correspondiente al problema de clasificación planteado.

En términos funcionales, el flujo de datos en el modelo es el siguiente:

Imagen de entrada (224×224×3).

División en parches de 16×16 píxeles → proyección lineal (dim=768).

Inserción del token [CLS] + embeddings posicionales.

Procesamiento mediante 12 capas Transformer (atención + FFN).

Obtención del embedding final del token [CLS].

Proyección lineal hacia las clases de salida.

Este diseño permite que el ViT capture dependencias globales entre distintas regiones de la imagen, lo cual lo diferencia de las CNNs tradicionales que dependen de filtros locales y jerarquías espaciales.

