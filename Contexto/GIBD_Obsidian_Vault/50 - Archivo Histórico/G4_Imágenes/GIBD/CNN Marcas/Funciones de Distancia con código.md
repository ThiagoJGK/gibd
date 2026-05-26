---
aliases: [Funciones de Distancia con código]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2023-04-25
origen_zip: GIBD-20260521T205218Z-3-009.zip
ruta_interna: "GIBD/CNN Marcas/Notebooks/Metric Learning/Funciones de Distancia con código.docx"
tamanio_bytes: 674581
---

# Funciones de Distancia con código

Ruta interna: `GIBD/CNN Marcas/Notebooks/Metric Learning/Funciones de Distancia con código.docx`

---

https://sitiobigdata.com/2019/12/24/funciones-comunes-de-perdida-en-el-aprendizaje-automatico/#


La función de pérdida "n-pair" es una función de aprendizaje de métricas que se utiliza comúnmente para entrenar redes neuronales profundas en tareas como reconocimiento facial, recuperación de imágenes y reidentificación de personas. Fue introducida por Sohn et al. en el artículo "Improved Deep Metric Learning with Multi-class N-pair Loss Objective" en 2016.
La función de pérdida "n-pair" está diseñada para aprender representaciones discriminativas en un modelo de red siamesa, donde pares de muestras son alimentados a la red y su similitud se aprende en un espacio de representación. El objetivo de la función de pérdida "n-pair" es maximizar la similitud entre muestras de la misma clase y minimizar la similitud entre muestras de diferentes clases.
La función de pérdida "n-pair" se define de la siguiente manera:
L_npair = ∑[i=1 hasta N] log(1 + ∑[j=1 hasta K] exp[-s_pos(i,j) + s_neg(i,j)])
donde N es el número de muestras, K es el número de muestras negativas por cada muestra ancla, s_pos(i,j) es la puntuación de similitud entre la muestra ancla i y su muestra positiva j, y s_neg(i,j) es la puntuación de similitud entre la muestra ancla i y su muestra negativa j.
La función de pérdida "n-pair" tiene dos componentes: una componente positiva que fomenta que las muestras de la misma clase tengan puntuaciones de similitud altas, y una componente negativa que desalienta que las muestras de diferentes clases tengan puntuaciones de similitud altas. La pérdida se calcula para cada muestra ancla y se suma sobre todas las muestras en el lote de entrenamiento.
La función de pérdida "n-pair" es efectiva para aprender representaciones discriminativas al modelar explícitamente la relación entre muestras de la misma clase y muestras de diferentes clases. Se ha demostrado que alcanza resultados de vanguardia en varias tareas de aprendizaje de métricas, y se utiliza ampliamente en investigaciones y aplicaciones que requieren el aprendizaje de métricas de similitud o distancia entre muestras.


import numpy as np def n_pair_loss(s_pos, s_neg, K): 
""" Compute the n-pair loss given similarity scores for positive and negative pairs. Args: s_pos (numpy array): Similarity scores for positive pairs, shape (N, K), where N is the number of samples and K is the number of positive samples per anchor. s_neg (numpy array): Similarity scores for negative pairs, shape (N, K), where N is the number of samples and K is the number of negative samples per anchor. K (int): Number of negative samples per anchor. Returns: float: Computed n-pair loss. """ 
N = s_pos.shape[0] pos_exp = np.exp(-s_pos) # Compute exp(-s_pos) element-wise 
neg_exp_sum = np.sum(np.exp(-s_neg), axis=1) # Compute sum of exp(-s_neg) along axis 1 
loss = np.sum(np.log(1 + neg_exp_sum)) # Compute log(1 + ∑ exp(-s_neg)) and sum over all samples 
loss += np.sum(np.log(1 + pos_exp)) # Compute log(1 + exp(-s_pos)) and sum over all samples 
loss /= N * K # Normalize by the number of samples and negative samples per anchor 
return loss
You can use this function by passing in the similarity scores for positive and negative pairs, as well as the number of negative samples per anchor (K). Note that the similarity scores should be negative values, where smaller values indicate higher similarity between samples. The function computes the n-pair loss as described in the previous answer, and returns the computed loss value.




VER de Chaof Wang

https://github.com/ChaofWang/Npair_loss_pytorch/blob/master/Npair_loss.py


ANGULAR LOSS 

https://github.com/geonm/tf_angular_loss

https://github.com/geonm/tf_angular_loss/blob/master/tf_angular_loss.py




https://github.com/shubhangb97/Metric_Learning_CS_543
