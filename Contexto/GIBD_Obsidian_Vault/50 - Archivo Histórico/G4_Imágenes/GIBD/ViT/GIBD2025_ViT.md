---
aliases: [GIBD2025_ViT]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2025-09-03
origen_zip: GIBD-20260521T205218Z-3-002.zip
ruta_interna: "GIBD/ViT/GIBD2025_ViT.docx"
tamanio_bytes: 576046
---

# GIBD2025_ViT

Ruta interna: `GIBD/ViT/GIBD2025_ViT.docx`

---

ViT - Vision Transformer
Teoría ViT
Arquitectura de redes neuronales basada en Transformers → arquitectura de red neuronal basada en mecanismos de atención → al procesar un elemento, se puede decidir cuánto peso asignarle al resto de los elementos.

¿Cómo se asigna el peso? → mediante la operación self-attention.

Cálculo de la atención
Por cada elemento se generan 3 vectores: Q (Query), K (Key), V (Value). Luego se calcula la atención:
Attention(Q, K, V) = softmax(QKᵀ / √d) * V → El resultado es una nueva representación para la entrada (combinación ponderada de las demás).

El vector Q representa lo que el token está buscando (¿a qué otros patches debería prestar atención?).
El vector K representa lo que el token ofrece o describe (¿cuán relevante soy yo para las preguntas de los otros?).
El vector V representa la información que se pasa a otros tokens si son considerados relevantes (esto es lo que tengo para ofrecer si te interesa).

Los vectores Q, K y V no contienen información predeterminada sobre qué buscar ni qué ofrecer. Lo aprenden durante el entrenamiento. Es decir, al principio tienen valores aleatorios, y se van ajustando para que el modelo funcione bien.

Aplicación de self-attention en imágenes
Imagen de entrada.
	Imagen RGB de 224 x 224 píxeles.

División en patches.
Si se divide en 16 x 16, entonces 224/16 x 224/16 = 14 x 14 = 196 patches.
Cada patch queda de 16 x 16 x 3 (tres capas RGB) y se aplanan (flatten). De esta manera queda un vector D de dimensión 768.
Se crea además el vector CLS (classification vector), contiene información resumida de toda la imágen (mezcla los Values [V] según la atención).
Por lo tanto el resultado es una secuencia de 197 elementos.

Generación de Q, K, V.
Se generan 3 matrices: 



Cálculo de self-attention.
Se produce una matriz de forma , donde cada vector representa un patch con información contextual de la imagen.

Se requiere de un positional encoding para que el modelo sepa la posición de cada parche (los Transformers no tienen conocimiento espacial).

Papers importantes
An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (https://arxiv.org/abs/2010.11929)

How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers (https://arxiv.org/abs/2106.10270)

When Vision Transformers Outperform ResNets without Pre-training or Strong Data Augmentations (https://arxiv.org/abs/2106.01548)
- Demuestra que cuando se entrena un ViT desde cero (con un optimizador como SAM), puede superar a ResNet de tamaño similar, evitando pre entrenamientos y augmentations.
- La comparación se hace entre ViT con patches de 16x16 y ResNet-152 (y otras variantes).
- Establece que los ViTs entrenados desde cero presentan curvaturas altas que implican mala generalización.
- Halla que los ViTs tienen neuronas activas altamente dispersas (gran potencial de pruning), y que las primeras capas de los ViTs (embedding layers) son las que más sufren de curvaturas elevadas.
- Se propone SAM (Sharpness-Aware Minimization) como solución que optimiza una versión modificada de la función de pérdida que penaliza regiones de alta curvatura.
- SAM además mejora la interpretabilidad de los mapas de atención, aumenta la norm de los pesos, da buenos resultados combinando con optimizadores como Adam.

Attention is All You Need (https://arxiv.org/abs/1706.03762) 
Google Colab
Desarrollo de modelo y entrenamiento desde cero
https://colab.research.google.com/drive/14ghEeoVMa-H8wHD3jz83VI1Eg_Jap5Uh?usp=sharing 



Desarrollo con modelo pre-entrenado
https://colab.research.google.com/drive/1JNOERt2DWrf-Q08DfmCfkhQMYGA360KB#scrollTo=-HgSIa1h9STj 






