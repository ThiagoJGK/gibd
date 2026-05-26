---
aliases: [Triplet_loss]
tags:
  - grupo/g4_imágenes
  - estado/util
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2022-07-26
origen_zip: GIBD-20260521T205218Z-3-003.zip
ruta_interna: "GIBD/CNN Marcas/Notebooks/Triplet_loss.ipynb"
tamanio_bytes: 83368
---

# Notebook: Triplet_loss.ipynb

Ruta interna: `GIBD/CNN Marcas/Notebooks/Triplet_loss.ipynb`

---

Test the idea of a triplet loss as described in https://towardsdatascience.com/siamese-network-triplet-loss-b4ca82c1aec8 using the CIFAR10 dataset and keras.


**[Celda 2 - Código]**
```python
import numpy as np
import tensorflow as tf

VECTOR_SIZE = 32

```


**[Celda 3 - Código]**
```python
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```


*Salida:*
```text
Using TensorFlow backend.
```


**[Celda 4 - Código]**
```python
import random

def data_generator(x, y, batch_size=32):
  n_classes = 10

  class_ids = [None] * n_classes
  for n in range(n_classes):
    class_ids[n] = np.where(y == n)[0]

  while True:
    X_batch_0 = np.empty((batch_size, 32, 32, 3), dtype=np.float32)
    X_batch_1 = np.empty((batch_size, 32, 32, 3), dtype=np.float32)
    X_batch_2 = np.empty((batch_size, 32, 32, 3), dtype=np.float32)

    for i in range(batch_size):
      tgt_class = random.randint(0, n_classes - 1)
      neg_class = random.randint(0, n_classes - 2)
      if neg_class == tgt_class:
        neg_class = n_classes - 1
      a = random.randint(0, class_ids[tgt_class].size - 1)
      p = random.randint(0, class_ids[tgt_class].size - 2)
      if p == a:
        p = class_ids[tgt_class].size - 1
      n = random.randint(0, class_ids[neg_class].size - 1)
      idx_a = class_ids[tgt_class][a]
      idx_p = class_ids[tgt_class][p]
      idx_n = class_ids[neg_class][n]
      X_batch_0[i] = x[idx_a] / 256
      X_batch_1[i] = x[idx_p] / 256
      X_batch_2[i] = x[idx_n] / 256

    yield [X_batch_0, X_batch_1, X_batch_2], np.empty((batch_size, VECTOR_SIZE * 3))

```


**[Celda 5 - Código]**
```python
gen = data_generator(x_train, y_train)
X_list, _ = next(gen)
```


**[Celda 6 - Código]**
```python
%matplotlib inline
import matplotlib.pyplot as plt

INDEX = 16
fig, axes = plt.subplots(1, 3)
for i, ax in enumerate(axes):
  ax.imshow(X_list[i][INDEX])
plt.show()
```


*Salida:*
```text
<Figure size 432x288 with 3 Axes>
```


**[Celda 7 - Código]**
```python
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K

img_width = 32
img_height = 32
img_colors = 3

margin = 1.0

def triplet_loss(y_true, y_pred):
    """ ignore y_true
    """
    anchor_vec = y_pred[:, :VECTOR_SIZE]
    positive_vec = y_pred[:, VECTOR_SIZE:2*VECTOR_SIZE]
    negative_vec = y_pred[:, 2*VECTOR_SIZE:]
    d1 = keras.losses.cosine_proximity(anchor_vec, positive_vec)
#     d1 = K.print_tensor(d1, message='d1=')
    d2 = keras.losses.cosine_proximity(anchor_vec, negative_vec)
#     d2 = K.print_tensor(d2, message='d2=')
    return K.clip(d2 - d1 + margin, 0, None)

def cos_sim_pos(y_true, y_pred):
    """ Cosine similarity between anchor and positive sample
        Higher value is better.
    """
    anchor_vec = y_pred[:, :VECTOR_SIZE]
    positive_vec = y_pred[:, VECTOR_SIZE:2*VECTOR_SIZE]
    d1 = keras.losses.cosine_proximity(anchor_vec, positive_vec)
    return d1

def cos_sim_neg(y_true, y_pred):
    """ Cosine similarity between anchor and negative sample
        Lower value is better.
    """
    anchor_vec = y_pred[:, :VECTOR_SIZE]
    negative_vec = y_pred[:, 2*VECTOR_SIZE:]
    d2 = keras.losses.cosine_proximity(anchor_vec, negative_vec)
    return d2


def make_image_model():
  model = Sequential()
  model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(img_width, img_height, img_colors)))
  model.add(Activation('relu'))
  model.add(Conv2D(32, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(64, (3, 3), padding='same'))
  model.add(Activation('relu'))
  model.add(Conv2D(64, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(512))
  model.add(Activation('relu'))
#   model.add(Dropout(0.5))
  model.add(Dense(VECTOR_SIZE))
  return model

def make_image_model_small():
  inp = Input(shape=(img_width, img_height, img_colors))
  l1 = Conv2D(16, (3, 3))(inp)
  l1 = MaxPooling2D()(l1)
  l2 = Conv2D(32, (2, 2))(l1)
  l2 = MaxPooling2D()(l2)
  l3 = Conv2D(64, (2, 2))(l2)
  l3 = MaxPooling2D()(l3)
  conv_out = Flatten()(l3)
  out = Dense(VECTOR_SIZE)(conv_out)
  model = Model(inp, out)
  return model

def make_siamese_model(img_model):
  anchor = Input(shape=(img_width, img_height, img_colors))
  positive = Input(shape=(img_width, img_height, img_colors))
  negative = Input(shape=(img_width, img_height, img_colors))
  anchor_vec = img_model(anchor)
  positive_vec = img_model(positive)
  negative_vec = img_model(negative)
  vecs = Concatenate(axis=1, name='vectors')([anchor_vec, positive_vec, negative_vec])
  model = Model([anchor, positive, negative], vecs)
  model.compile('adam', triplet_loss, metrics=[cos_sim_pos, cos_sim_neg])
  return model

tf.set_random_seed(42)
img_model = make_image_model()
train_model = make_siamese_model(img_model)
img_model.summary()
train_model.summary()
```


*Salida:*
```text
WARNING: Logging before flag parsing goes to stderr.
W0624 07:59:28.298809 139782001362816 deprecation.py:506] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/init_ops.py:1251: calling VarianceScaling.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 32, 32, 32)        896       
_________________________________________________________________
activation (Activation)      (None, 32, 32, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 30, 30, 32)        9248      
_________________________________________________________________
activation_1 (Activation)    (None, 30, 30, 32)        0         
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0         
_________________________________________________________________
dropout (Dropout)            (None, 15, 15, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 15, 15, 64)        18496     
_________________________________________________________________
activation_2 (Activation)    (None, 15, 15, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 13, 13, 64)        36928     
_________________________________________________________________
activation_3 (Activation)    (None, 13, 13, 64)        0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 6, 6, 64)          0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 6, 6, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 2304)              0         
_________________________________________________________________
dense (Dense)                (None, 512)               1180160   
_________________________________________________________________
activation_4 (Activation)    (None, 512)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 32)                16416     
=================================================================
Total params: 1,262,144
Trainable params: 1,262,144
Non-trainable params: 0
_________________________________________________________________
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 32, 32, 3)]  0                                            
__________________________________________________________________________________________________
input_2 (InputLayer)            [(None, 32, 32, 3)]  0                                            
__________________________________________________________________________________________________
input_3 (InputLayer)            [(None, 32, 32, 3)]  0                                            
__________________________________________________________________________________________________
sequential (Sequential)         (None, 32)           1262144     input_1[0][0]                    
                                                                 input_2[0][0]                    
                                                                 input_3[0][0]                    
__________________________________________________________________________________________________
vectors (Concatenate)           (None, 96)           0           sequential[1][0]                 
                                                                 sequential[2][0]                 
                                                                 sequential[3][0]                 
==================================================================================================
Total params: 1,262,144
Trainable params: 1,262,144
Non-trainable params: 0
__________________________________________________________________________________________________
```


**[Celda 8 - Código]**
```python
# generate a single batch
random.seed(42)
gen = data_generator(x_train, y_train, batch_size=1)
x_b0, y_b0 = next(gen)

```


**[Celda 9 - Código]**
```python
# debug code to make sure that cosine similarity matches model metrics / loss
from sklearn.metrics.pairwise import cosine_similarity
output = train_model.predict(x_b0)
v1 = output[:, :VECTOR_SIZE]
v2 = output[:, VECTOR_SIZE:2*VECTOR_SIZE]
v3 = output[:, 2*VECTOR_SIZE:]
print('pos', cosine_similarity(v1, v2))
print('neg', cosine_similarity(v1, v3))
```


*Salida:*
```text
pos [[0.9334991]]
neg [[0.94116354]]
```


**[Celda 10 - Código]**
```python
train_model.evaluate(x_b0)
```


*Salida:*
```text
1/1 [==============================] - 0s 112ms/sample - loss: 1.0077 - cos_sim_pos: 0.9335 - cos_sim_neg: 0.9412
[1.007664442062378, 0.93349916, 0.94116354]
```


**[Celda 11 - Código]**
```python
history = train_model.fit_generator(
    data_generator(x_train, y_train, batch_size=64), validation_data=data_generator(x_test, y_test, batch_size=3),
    steps_per_epoch=1000, validation_steps=200,
    epochs=30)
```


*Salida:*
```text
Epoch 1/30
W0624 07:59:31.270154 139782001362816 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
1000/1000 [==============================] - 42s 42ms/step - loss: 0.8735 - cos_sim_pos: 0.6894 - cos_sim_neg: 0.5191 - val_loss: 0.6840 - val_cos_sim_pos: 0.5385 - val_cos_sim_neg: 0.0629
Epoch 2/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.6487 - cos_sim_pos: 0.5370 - cos_sim_neg: 0.0404 - val_loss: 0.6393 - val_cos_sim_pos: 0.5484 - val_cos_sim_neg: 0.0318
Epoch 3/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.5934 - cos_sim_pos: 0.5569 - cos_sim_neg: 0.0074 - val_loss: 0.5878 - val_cos_sim_pos: 0.5538 - val_cos_sim_neg: 0.0054
Epoch 4/30
1000/1000 [==============================] - 39s 39ms/step - loss: 0.5413 - cos_sim_pos: 0.5785 - cos_sim_neg: -0.0163 - val_loss: 0.4945 - val_cos_sim_pos: 0.6086 - val_cos_sim_neg: -0.0286
Epoch 5/30
1000/1000 [==============================] - 39s 39ms/step - loss: 0.4874 - cos_sim_pos: 0.6062 - cos_sim_neg: -0.0348 - val_loss: 0.4686 - val_cos_sim_pos: 0.5917 - val_cos_sim_neg: -0.0642
Epoch 6/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.4499 - cos_sim_pos: 0.6371 - cos_sim_neg: -0.0419 - val_loss: 0.4346 - val_cos_sim_pos: 0.6200 - val_cos_sim_neg: -0.0717
Epoch 7/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.4151 - cos_sim_pos: 0.6582 - cos_sim_neg: -0.0535 - val_loss: 0.4822 - val_cos_sim_pos: 0.6365 - val_cos_sim_neg: 0.0034
Epoch 8/30
1000/1000 [==============================] - 39s 39ms/step - loss: 0.3877 - cos_sim_pos: 0.6854 - cos_sim_neg: -0.0584 - val_loss: 0.4080 - val_cos_sim_pos: 0.6647 - val_cos_sim_neg: -0.0438
Epoch 9/30
1000/1000 [==============================] - 39s 39ms/step - loss: 0.3625 - cos_sim_pos: 0.7021 - cos_sim_neg: -0.0639 - val_loss: 0.3561 - val_cos_sim_pos: 0.6874 - val_cos_sim_neg: -0.0877
Epoch 10/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.3411 - cos_sim_pos: 0.7188 - cos_sim_neg: -0.0714 - val_loss: 0.3942 - val_cos_sim_pos: 0.6766 - val_cos_sim_neg: -0.0466
Epoch 11/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.3235 - cos_sim_pos: 0.7345 - cos_sim_neg: -0.0732 - val_loss: 0.3892 - val_cos_sim_pos: 0.6303 - val_cos_sim_neg: -0.0890
Epoch 12/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.3021 - cos_sim_pos: 0.7537 - cos_sim_neg: -0.0779 - val_loss: 0.4039 - val_cos_sim_pos: 0.6618 - val_cos_sim_neg: -0.0404
Epoch 13/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.2868 - cos_sim_pos: 0.7654 - cos_sim_neg: -0.0819 - val_loss: 0.4243 - val_cos_sim_pos: 0.6565 - val_cos_sim_neg: -0.0272
Epoch 14/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.2726 - cos_sim_pos: 0.7796 - cos_sim_neg: -0.0824 - val_loss: 0.3832 - val_cos_sim_pos: 0.6510 - val_cos_sim_neg: -0.0857
Epoch 15/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.2610 - cos_sim_pos: 0.7933 - cos_sim_neg: -0.0811 - val_loss: 0.4257 - val_cos_sim_pos: 0.6426 - val_cos_sim_neg: -0.0428
Epoch 16/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.2487 - cos_sim_pos: 0.8044 - cos_sim_neg: -0.0834 - val_loss: 0.3614 - val_cos_sim_pos: 0.6796 - val_cos_sim_neg: -0.0806
Epoch 17/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.2352 - cos_sim_pos: 0.8170 - cos_sim_neg: -0.0875 - val_loss: 0.3570 - val_cos_sim_pos: 0.6608 - val_cos_sim_neg: -0.0997
Epoch 18/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.2267 - cos_sim_pos: 0.8251 - cos_sim_neg: -0.0871 - val_loss: 0.4178 - val_cos_sim_pos: 0.6824 - val_cos_sim_neg: -0.0103
Epoch 19/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.2171 - cos_sim_pos: 0.8348 - cos_sim_neg: -0.0885 - val_loss: 0.3726 - val_cos_sim_pos: 0.6881 - val_cos_sim_neg: -0.0552
Epoch 20/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.2085 - cos_sim_pos: 0.8424 - cos_sim_neg: -0.0898 - val_loss: 0.3838 - val_cos_sim_pos: 0.6632 - val_cos_sim_neg: -0.0614
Epoch 21/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.1976 - cos_sim_pos: 0.8527 - cos_sim_neg: -0.0930 - val_loss: 0.3793 - val_cos_sim_pos: 0.6712 - val_cos_sim_neg: -0.0687
Epoch 22/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.1947 - cos_sim_pos: 0.8574 - cos_sim_neg: -0.0910 - val_loss: 0.4055 - val_cos_sim_pos: 0.6323 - val_cos_sim_neg: -0.0686
Epoch 23/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.1893 - cos_sim_pos: 0.8634 - cos_sim_neg: -0.0911 - val_loss: 0.3709 - val_cos_sim_pos: 0.6683 - val_cos_sim_neg: -0.0733
Epoch 24/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.1804 - cos_sim_pos: 0.8709 - cos_sim_neg: -0.0937 - val_loss: 0.4094 - val_cos_sim_pos: 0.6593 - val_cos_sim_neg: -0.0401
Epoch 25/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.1766 - cos_sim_pos: 0.8732 - cos_sim_neg: -0.0939 - val_loss: 0.3856 - val_cos_sim_pos: 0.6683 - val_cos_sim_neg: -0.0626
Epoch 26/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.1730 - cos_sim_pos: 0.8764 - cos_sim_neg: -0.0950 - val_loss: 0.3954 - val_cos_sim_pos: 0.6475 - val_cos_sim_neg: -0.0610
Epoch 27/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.1683 - cos_sim_pos: 0.8813 - cos_sim_neg: -0.0932 - val_loss: 0.4155 - val_cos_sim_pos: 0.6422 - val_cos_sim_neg: -0.0431
Epoch 28/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.1608 - cos_sim_pos: 0.8887 - cos_sim_neg: -0.0944 - val_loss: 0.3677 - val_cos_sim_pos: 0.6774 - val_cos_sim_neg: -0.0679
Epoch 29/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.1608 - cos_sim_pos: 0.8867 - cos_sim_neg: -0.0949 - val_loss: 0.4145 - val_cos_sim_pos: 0.6430 - val_cos_sim_neg: -0.0498
Epoch 30/30
1000/1000 [==============================] - 40s 40ms/step - loss: 0.1543 - cos_sim_pos: 0.8925 - cos_sim_neg: -0.0973 - val_loss: 0.3447 - val_cos_sim_pos: 0.7084 - val_cos_sim_neg: -0.0616
```


**[Celda 12 - Código]**
```python
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()

plt.plot(range(len(loss)), loss, 'bo', label='Training Loss')
plt.plot(range(len(val_loss)), val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```


*Salida:*
```text
<Figure size 432x288 with 1 Axes>
```


**[Celda 13 - Código]**
```python
# use image model to generate image vectors
train_vecs = img_model.predict(x_train)
test_vecs = img_model.predict(x_test)

from sklearn.metrics.pairwise import cosine_similarity
y_sim = cosine_similarity(test_vecs, train_vecs)
y_max = np.argmax(y_sim, axis=1)

y_pred = y_train[y_max]
```


**[Celda 14 - Código]**
```python
from sklearn.metrics import classification_report
class_names = [str(i) for i in range(10)]
print(classification_report(y_test, y_pred, target_names=class_names))
```


*Salida:*
```text
precision    recall  f1-score   support

           0       0.60      0.62      0.61      1000
           1       0.68      0.68      0.68      1000
           2       0.46      0.42      0.44      1000
           3       0.31      0.33      0.32      1000
           4       0.44      0.43      0.43      1000
           5       0.38      0.40      0.39      1000
           6       0.72      0.69      0.70      1000
           7       0.70      0.66      0.68      1000
           8       0.68      0.70      0.69      1000
           9       0.61      0.62      0.62      1000

    accuracy                           0.56     10000
   macro avg       0.56      0.56      0.56     10000
weighted avg       0.56      0.56      0.56     10000
```


**[Celda 15 - Código]**
```python
# Examine prediction for a specific value.
INDEX = 16
fig, axes = plt.subplots(1, 2)
axes[0].imshow(x_test[INDEX])
axes[1].imshow(x_train[y_max[INDEX]])
plt.show()
```


*Salida:*
```text
<Figure size 432x288 with 2 Axes>
```


**[Celda 16 - Código]**
```python
# classes of the top K vectors most similar to test
print('y_true:', y_test[INDEX])
k = 10
top_k = np.argpartition(y_sim[INDEX], -k)[-k:]
y_train[top_k]

```


*Salida:*
```text
y_true: [5]
array([[3],
       [3],
       [3],
       [5],
       [5],
       [5],
       [3],
       [5],
       [4],
       [5]], dtype=uint8)
```
