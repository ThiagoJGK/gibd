---
aliases: [Replica]
tags:
  - grupo/g4_imágenes
  - estado/util
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2022-11-15
origen_zip: GIBD-20260521T205218Z-3-006.zip
ruta_interna: "GIBD/CNN Marcas/Notebooks/Metric Learning/Replica.ipynb"
tamanio_bytes: 69858
---

# Notebook: Replica.ipynb

Ruta interna: `GIBD/CNN Marcas/Notebooks/Metric Learning/Replica.ipynb`

---

**Utils**


**[Celda 2 - Código]**
```python
 pip install -U scikit-image
```


*Salida:*
```text
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Requirement already satisfied: scikit-image in /usr/local/lib/python3.7/dist-packages (0.18.3)
Collecting scikit-image
  Downloading scikit_image-0.19.3-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (13.5 MB)
[K     |████████████████████████████████| 13.5 MB 10.8 MB/s 
[?25hRequirement already satisfied: scipy>=1.4.1 in /usr/local/lib/python3.7/dist-packages (from scikit-image) (1.7.3)
Requirement already satisfied: numpy>=1.17.0 in /usr/local/lib/python3.7/dist-packages (from scikit-image) (1.21.6)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.7/dist-packages (from scikit-image) (21.3)
Requirement already satisfied: imageio>=2.4.1 in /usr/local/lib/python3.7/dist-packages (from scikit-image) (2.9.0)
Requirement already satisfied: networkx>=2.2 in /usr/local/lib/python3.7/dist-packages (from scikit-image) (2.6.3)
Requirement already satisfied: tifffile>=2019.7.26 in /usr/local/lib/python3.7/dist-packages (from scikit-image) (2021.11.2)
Requirement already satisfied: PyWavelets>=1.1.1 in /usr/local/lib/python3.7/dist-packages (from scikit-image) (1.3.0)
Requirement already satisfied: pillow!=7.1.0,!=7.1.1,!=8.3.0,>=6.1.0 in /usr/local/lib/python3.7/dist-packages (from scikit-image) (7.1.2)
Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging>=20.0->scikit-image) (3.0.9)
Installing collected packages: scikit-image
  Attempting uninstall: scikit-image
    Found existing installation: scikit-image 0.18.3
    Uninstalling scikit-image-0.18.3:
      Successfully uninstalled scikit-image-0.18.3
Successfully installed scikit-image-0.19.3
```


**[Celda 3 - Código]**
```python
import skimage
print(skimage.__version__)
#si no devuelve 19.3 o superior ejecutar la celda de arriba
```


*Salida:*
```text
0.19.3
```


**[Celda 4 - Código]**
```python
from google.colab import drive
drive.mount('/content/drive')
```


*Salida:*
```text
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
```


**[Celda 5 - Código]**
```python
# UTILS 

import keras
import matplotlib as plt
import glob
import numpy as np
from sklearn.utils import shuffle
import skimage
import random
import skimage.io as io

def load_images(x):

    x_input = []
    for i in range(len(x)):
        image = io.imread(x[i])
        image_augmented = data_augmentation(image)
        x_input.append(image_augmented)

    return np.asarray(x_input)


def load_data(dataset_path):
    '''
        You can obtain the dataset here: https://www.nature.com/articles/srep27988
    '''

    # Nature images here
    folders = glob.glob(dataset_path + '/*')
    N_CAT_TOT = len(folders)

    x = []
    y = []
    n_cat = 0
    for folder in folders:
        image_paths = glob.glob(folder + '/*.tif')
        for img in image_paths:
            x.append(img)
            y.append(n_cat)
        n_cat += 1

    x, y = np.asarray(x), np.asarray(y)
    x, y = shuffle(x, y, random_state=666)

    return x, y


def data_augmentation(img):

    a =  random.randint(0,1)
    b = random.randint(0,3)

    image_a = [img, img[::-1,:,:]]
    angle_a = [0,90,180,270]

    image_transformed = skimage.transform.rotate(image_a[a],angle=angle_a[b])
    return preprocess_input(image_transformed)

def preprocess_input(x):
    x -= 0.5
    x *= 2.
    return x

```


*Salida:*
```text
/usr/local/lib/python3.7/dist-packages/skimage/io/manage_plugins.py:23: UserWarning: Your installed pillow version is < 8.1.2. Several security issues (CVE-2021-27921, CVE-2021-25290, CVE-2021-25291, CVE-2021-25293, and more) have been fixed in pillow 8.1.2 or higher. We recommend to upgrade this library.
  from .collection import imread_collection_wrapper
```

**Network**


**[Celda 7 - Código]**
```python
import tensorflow as tf
import keras
from keras.layers import Lambda
import keras.backend as K
from sklearn.utils import shuffle
import skimage.io as io
import numpy as np

# ANTES DE ESTA CELDA HAY QUE EJECUTAR UTILS


def inception(EMB_VECTOR, IMG_SIZE, use_imagenet=False):
    # load pre-trained model graph, don't add final layer
    model = tf.keras.applications.InceptionV3(include_top=False, input_shape=(28, 28, 1),
                                          weights='imagenet' if use_imagenet else None)
    # add global pooling just like in InceptionV3
    new_output = tf.keras.layers.GlobalAveragePooling2D()(model.output)

    # # add new dense layer for our labels
    new_output = tf.keras.layers.Dense(EMB_VECTOR, activation='sigmoid')(new_output)

    model = keras.engine.training.Model(model.inputs, new_output)
    return model

def generator(x, y):

    x_class = [[] for i in range(len(np.unique(y)))]
    for i in range(len(x)):
        x_class[y[i]].append(x[i])

    ind = [0 for i in range(len(np.unique(y)))]
    y_class = [i for i in range(len(np.unique(y)))]

    while True:
        x_in = []
        y_in= []

        x_class, ind, y_class = shuffle(x_class, ind, y_class)

        for n in range(2):

            for i in range(len(ind)):
                if ind[i] >= len(x_class[i]):
                    x_class[i] = shuffle(x_class[i])
                    ind[i] = 0

            for i in range(len(ind)):
                image = io.imread(x_class[i][ind[i]])
                image_augmented = data_augmentation(image)

                x_in.append(image_augmented)
                y_in.append(y_class[i])

                ind[i] += 1

        x_in, y_in = np.asarray(x_in), np.asarray(y_in)

        yield x_in, y_in

```

**Multiclass_loss**


**[Celda 9 - Código]**
```python
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.summary import summary

'Inplementation in tensorflow by: https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/losses/python/metric_learning/metric_loss_ops.py'

def npairs_loss(labels, embeddings, BATCH_SIZE = 16, reg_lambda=0.002, print_losses=False):

    labels = labels[:BATCH_SIZE//2, 0]
    embeddings_anchor = embeddings[:BATCH_SIZE // 2, :]
    embeddings_positive = embeddings[BATCH_SIZE // 2:, :]

    # pylint: enable=line-too-long
    # Add the regularizer on the embedding.
    reg_anchor = math_ops.reduce_mean(
        math_ops.reduce_sum(math_ops.square(embeddings_anchor), 1))
    reg_positive = math_ops.reduce_mean(
        math_ops.reduce_sum(math_ops.square(embeddings_positive), 1))
    l2loss = math_ops.multiply(
        0.25 * reg_lambda, reg_anchor + reg_positive, name='l2loss')

    # Get per pair similarities.
    similarity_matrix = math_ops.matmul(
        embeddings_anchor, embeddings_positive, transpose_a=False,
        transpose_b=True)

    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    lshape = array_ops.shape(labels)
    assert lshape.shape == 1
    labels = array_ops.reshape(labels, [lshape[0], 1])

    labels_remapped = math_ops.to_float(
        math_ops.equal(labels, array_ops.transpose(labels)))
    labels_remapped /= math_ops.reduce_sum(labels_remapped, 1, keepdims=True)

    # Add the softmax loss.
    xent_loss = nn.softmax_cross_entropy_with_logits(
        logits=similarity_matrix, labels=labels_remapped)
    xent_loss = math_ops.reduce_mean(xent_loss, name='xentropy')

    if print_losses:
        xent_loss = logging_ops.Print(
            xent_loss, ['cross entropy:', xent_loss, 'l2loss:', l2loss])

    return l2loss + xent_loss

```

**Main_tnse**


**[Celda 11 - Código]**
```python
# Alfonso Medela & Artzai Picon, "Constellation Loss: Improving the efficiency of deep metric learning loss functions for optimal embedding.", submitted to NeurIPS 2019.

import os
import keras
import numpy as np
import tensorflow as tf
import pandas as pd
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import davies_bouldin_score, silhouette_score, balanced_accuracy_score
from sklearn.manifold import TSNE

# ANTES DE ESTA CELDA HAY QUE EJECUTAR: multiclass_loss, utils, network


if __name__ == '__main__':

    DATASET_PATH = '/content/drive/MyDrive/GIBD/CNN Marcas/Notebooks/Metric Learning/dataset UMCM'

    

    # PARAMETERS
    IMG_SIZE = 150 
    EMB_VECTOR = 128

    # LOAD THE DATA
    x, y = load_data(DATASET_PATH)

    fold = 0  # Choose fold
    random_seeds = [666, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    SEED = random_seeds[fold]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)
    x_test_input = load_images(x_test)    

    # DEFINE THE MODEL
    model = inception(EMB_VECTOR, IMG_SIZE)

    model.compile(loss=npairs_loss, optimizer=keras.optimizers.Adam(1e-3))
    model.load_weights('/content/drive/MyDrive/GIBD/CNN Marcas/Notebooks/Metric Learning/weights/inception_multiclass_fold_' + str(fold) + '.h5')

    embeddings_test = model.predict([x_test_input])
    x_tsne = TSNE(n_components=2).fit_transform(embeddings_test)

    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', (1., 170. / 255., 58. / 255.)]
    labels = ['Empty', 'Lympho', 'Mucosa', 'Stroma', 'Tumor', 'Complex', 'Debris', 'Adipose']
    ind = [0 for i in range(len(np.unique(y_test)))]

    for i in range(len(embeddings_test)):
        if ind[y_test[i]] == 0:
            plt.scatter(x_tsne[i, 0], x_tsne[i, 1], color=color[y_test[i]], label=labels[y_test[i]])
            ind[y_test[i]] = 1
        else:
            plt.scatter(x_tsne[i, 0], x_tsne[i, 1], color=color[y_test[i]])

    plt.ylabel(r'$z_2$')
    plt.xlabel(r'$z_1$')
    # plt.legend()
    plt.savefig('tsne.png')

```


*Salida:*
```text
32/32 [==============================] - 2s 36ms/step
[[1.8006800e-03 9.9998939e-01 2.5562868e-03 ... 4.0102475e-03
  1.3259377e-01 6.3752045e-04]
 [7.0450164e-04 1.3459361e-09 1.4085504e-03 ... 1.4027335e-03
  1.1870856e-11 1.7266160e-04]
 [1.8903427e-02 1.9056656e-03 2.9375651e-03 ... 3.2584623e-03
  5.7567763e-03 2.5152208e-03]
 ...
 [3.2325584e-02 5.0415255e-02 8.0945216e-02 ... 2.5840652e-01
  3.9696887e-02 1.6884318e-02]
 [2.7669496e-03 1.3418024e-08 2.8049375e-03 ... 2.3213529e-03
  1.2898413e-10 2.6815155e-04]
 [3.2791018e-02 2.9459244e-03 8.5003329e-03 ... 2.0168623e-02
  2.8800415e-03 7.0730154e-03]]
/usr/local/lib/python3.7/dist-packages/sklearn/manifold/_t_sne.py:783: FutureWarning: The default initialization in TSNE will change from 'random' to 'pca' in 1.2.
  FutureWarning,
/usr/local/lib/python3.7/dist-packages/sklearn/manifold/_t_sne.py:793: FutureWarning: The default learning rate in TSNE will change from 200.0 to 'auto' in 1.2.
  FutureWarning,
<Figure size 432x288 with 1 Axes>
```

**main_train**


**[Celda 13 - Código]**
```python
# Alfonso Medela & Artzai Picon, "Constellation Loss: Improving the efficiency of deep metric learning loss functions for optimal embedding.", submitted to NeurIPS 2019.
import os
import keras
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ANTES DE ESTA CELDA HAY QUE EJECUTAR: multiclass_loss, utils, network


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.savefig('/content/drive/MyDrive/GIBD/CNN Marcas/Notebooks/Metric Learning/plot_losses/inception_multiclass_fold_' + str(fold) + '.png');


if __name__ == '__main__':

    loss_plot = PlotLosses()

    DATASET_PATH = '/content/drive/MyDrive/GIBD/CNN Marcas/Notebooks/Metric Learning/dataset UMCM'

    # PARAMETERS
    BATCH_SIZE = 16
    IMG_SIZE = 150
    EMB_VECTOR = 128
    EPOCHS = 10

    # LOAD THE DATA
    x, y = load_data(DATASET_PATH)

    random_seeds = [666, 100, 200, 300, 400, 500, 600, 700, 800, 900]

    fold = 0
    for seed in random_seeds:

        # 80% - 20%
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

        steps_train = int(len(x_train) * 1. / BATCH_SIZE ) + 1
        steps_test = int(len(x_test) * 1. / BATCH_SIZE ) + 1

        print('Fold: ' + str(fold))

        # TRAIN
        #gpu_device = "/gpu:0"  # 0,1,2,3
        #if keras.backend.backend() == 'tensorflow':
        #    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device.rsplit(':', 1)[-1]
        #    session_config = K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        #    session_config.gpu_options.allow_growth = True
        #    session = K.tf.Session(config=session_config)
        #    with K.tf.device(gpu_device):

        # DEFINE THE MODEL
        model = inception(EMB_VECTOR, IMG_SIZE)
        for layer in model.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.momentum = 0.9

        for layer in model.layers[:-50]:
            if not isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False

        # TRAIN THE MODEL

        model.compile(loss=npairs_loss, optimizer=keras.optimizers.Adam(1e-3))

        model.fit(generator(x_train, y_train), steps_per_epoch=steps_train,
                            validation_data=generator(x_test, y_test), validation_steps=steps_test,
                            epochs=EPOCHS, callbacks=[loss_plot])

        model.save_weights('/content/drive/MyDrive/GIBD/CNN Marcas/Notebooks/Metric Learning/weights/inception_multiclass_fold_' + str(fold) + '.h5')

        # close session and add a fold
        session.close()
        fold += 1

        #para guardar entrenamientos a disco local 
        #from google.colab import files
        #files.download('example.txt')
```


*Salida:*
```text
Fold: 0
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
87910968/87910968 [==============================] - 1s 0us/step
Epoch 1/10
WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/tensorflow/python/util/dispatch.py:1082: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.cast` instead.
WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/tensorflow/python/util/dispatch.py:1082: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
Instructions for updating:

Future major versions of TensorFlow will allow gradients to flow
into the labels input on backprop by default.

See `tf.nn.softmax_cross_entropy_with_logits_v2`.

252/252 [==============================] - 1019s 4s/step - loss: 0.9145 - val_loss: 0.6270
Epoch 2/10
252/252 [==============================] - 69s 276ms/step - loss: 0.4575 - val_loss: 0.5478
Epoch 3/10
252/252 [==============================] - 50s 198ms/step - loss: 0.3810 - val_loss: 0.4932
Epoch 4/10
252/252 [==============================] - 52s 208ms/step - loss: 0.3353 - val_loss: 0.4745
Epoch 5/10
252/252 [==============================] - 50s 198ms/step - loss: 0.2880 - val_loss: 0.4626
Epoch 6/10
252/252 [==============================] - 51s 201ms/step - loss: 0.2712 - val_loss: 0.4491
Epoch 7/10
252/252 [==============================] - 50s 197ms/step - loss: 0.2125 - val_loss: 0.4259
Epoch 8/10
252/252 [==============================] - 51s 201ms/step - loss: 0.2421 - val_loss: 0.3643
Epoch 9/10
252/252 [==============================] - 49s 193ms/step - loss: 0.1871 - val_loss: 0.3695
Epoch 10/10
252/252 [==============================] - 49s 195ms/step - loss: 0.1896 - val_loss: 0.4468
<Figure size 432x288 with 1 Axes>
```


**main_test**


**[Celda 15 - Código]**
```python
# Alfonso Medela & Artzai Picon, "Constellation Loss: Improving the efficiency of deep metric learning loss functions for optimal embedding.", submitted to NeurIPS 2019.

import os
import keras
import numpy as np
import tensorflow as tf
import pandas as pd
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import davies_bouldin_score, silhouette_score, balanced_accuracy_score

# ANTES DE ESTA CELDA HAY QUE EJECUTAR: multiclass_loss, utils, network

if __name__ == '__main__':

    # PARAMETERS
    IMG_SIZE = 150
    EMB_VECTOR = 128

    DATASET_PATH = '/content/drive/MyDrive/GIBD/CNN Marcas/Notebooks/Metric Learning/dataset UMCM'

    # LOAD THE DATA
    x, y = load_data(DATASET_PATH)

    #random_seeds = [666, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    random_seeds = [666, 100, 200]
    accuracy = []
    silhouette = []
    davis = []
    bac_list = []

    fold = 0
    for seed in random_seeds:

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

        x_train_input = load_images(x_train)
        x_test_input = load_images(x_test)

        print('Train data:', x_train.shape, y_train.shape)
        print('Test data:', x_test.shape, y_test.shape)

        print('')
        print('Training...')
        
        # DEFINE THE MODEL
        model = inception(EMB_VECTOR, IMG_SIZE)

        model.compile(loss=npairs_loss, optimizer=keras.optimizers.Adam(1e-3))
        model.load_weights('/content/drive/MyDrive/GIBD/CNN Marcas/Notebooks/Metric Learning/weights/inception_multiclass_fold_' + str(fold) + '.h5')

        embeddings_train = model.predict([x_train_input])
        embeddings_test = model.predict([x_test_input])

        'ML model to train with extracted feature vectors'

        KNN = KNeighborsClassifier()
        KNN.fit(embeddings_train, y_train)
        score = KNN.score(embeddings_test, y_test)

        accuracy.append(score * 100.)

        y_pred = KNN.predict(embeddings_test)
        BAC = balanced_accuracy_score(y_test, y_pred) * 100.
        bac_list.append(BAC)

        'Homogeneity test'
        d_score = davies_bouldin_score(embeddings_test, y_test)
        s_score = silhouette_score(embeddings_test, y_test)

        silhouette.append(s_score)
        davis.append(d_score)

        #fold += 1

    accuracy = np.asarray(accuracy)
    davis = np.asarray(davis)
    silhouette = np.asarray(silhouette)
    bac_list = np.asarray(bac_list)

    # Get results from list of crossvalidations
    mean_accuracy, std_accuracy = np.mean(accuracy), np.std(accuracy)
    mean_s, std_s = np.mean(silhouette), np.std(silhouette)
    mean_d, std_d = np.mean(davis), np.std(davis)
    mean_bac, std_bac = np.mean(bac_list), np.std(bac_list)

    mean_accuracy, std_accuracy = np.reshape(mean_accuracy, (1)), np.reshape(std_accuracy, (1))
    mean_s, std_s = np.reshape(mean_s, (1)), np.reshape(std_s, (1))
    mean_d, std_d = np.reshape(mean_d, (1)), np.reshape(std_d, (1))
    mean_bac, std_bac = np.reshape(mean_bac, (1)), np.reshape(std_bac, (1))

    # Save the data
    accuracy_row = np.concatenate((accuracy, mean_accuracy, std_accuracy), axis=0)
    accuracy_row = np.reshape(accuracy_row, (len(accuracy_row), 1))
    accuracy_row = np.around(accuracy_row, decimals=2)

    silhouette_row = np.concatenate((silhouette, mean_s, std_s), axis=0)
    silhouette_row = np.reshape(silhouette_row, (len(silhouette_row), 1))
    silhouette_row = np.around(silhouette_row, decimals=4)

    davis_row = np.concatenate((davis, mean_d, std_d), axis=0)
    davis_row = np.reshape(davis_row, (len(davis_row), 1))
    davis_row = np.around(davis_row, decimals=4)

    bac_row = np.concatenate((bac_list, mean_bac, std_bac), axis=0)
    bac_row = np.reshape(bac_row, (len(bac_row), 1))
    bac_row = np.around(bac_row, decimals=2)

    csv_array = np.concatenate((accuracy_row, bac_row, davis_row, silhouette_row), axis=-1)

    df = pd.DataFrame(csv_array, columns=['accuracy', 'BAC', 'Davis', 'Silhouette'])

    df.to_csv('/content/drive/MyDrive/GIBD/CNN Marcas/Notebooks/Metric Learning/resultados/final_exp_8_pair_mc.csv')
```


*Salida:*
```text
Train data: (4016,) (4016,)
Test data: (1004,) (1004,)

Training...
126/126 [==============================] - 14s 38ms/step
32/32 [==============================] - 2s 59ms/step
Train data: (4016,) (4016,)
Test data: (1004,) (1004,)

Training...
126/126 [==============================] - 6s 32ms/step
32/32 [==============================] - 1s 32ms/step
Train data: (4016,) (4016,)
Test data: (1004,) (1004,)

Training...
126/126 [==============================] - 5s 34ms/step
32/32 [==============================] - 1s 33ms/step
```


**[Celda 16 - Código]**
```python
# pylint: disable=line-too-long
def npairs_loss(labels, embeddings_anchor, embeddings_positive,
                reg_lambda=0.002, print_losses=False):
  """Computes the npairs loss.
  Npairs loss expects paired data where a pair is composed of samples from the
  same labels and each pairs in the minibatch have different labels. The loss
  has two components. The first component is the L2 regularizer on the
  embedding vectors. The second component is the sum of cross entropy loss
  which takes each row of the pair-wise similarity matrix as logits and
  the remapped one-hot labels as labels.
  See: http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf
  Args:
    labels: 1-D tf.int32 `Tensor` of shape [batch_size/2].
    embeddings_anchor: 2-D Tensor of shape [batch_size/2, embedding_dim] for the
      embedding vectors for the anchor images. Embeddings should not be
      l2 normalized.
    embeddings_positive: 2-D Tensor of shape [batch_size/2, embedding_dim] for the
      embedding vectors for the positive images. Embeddings should not be
      l2 normalized.
    reg_lambda: Float. L2 regularization term on the embedding vectors.
    print_losses: Boolean. Option to print the xent and l2loss.
  Returns:
    npairs_loss: tf.float32 scalar.
  """
  # pylint: enable=line-too-long
  # Add the regularizer on the embedding.
  reg_anchor = math_ops.reduce_mean(
      math_ops.reduce_sum(math_ops.square(embeddings_anchor), 1))
  reg_positive = math_ops.reduce_mean(
      math_ops.reduce_sum(math_ops.square(embeddings_positive), 1))
  l2loss = math_ops.multiply(
      0.25 * reg_lambda, reg_anchor + reg_positive, name='l2loss')

  # Get per pair similarities.
  similarity_matrix = math_ops.matmul(
      embeddings_anchor, embeddings_positive, transpose_a=False,
      transpose_b=True)

  # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
  lshape = array_ops.shape(labels)
  assert lshape.shape == 1
  labels = array_ops.reshape(labels, [lshape[0], 1])

  labels_remapped = math_ops.to_float(
      math_ops.equal(labels, array_ops.transpose(labels)))
  labels_remapped /= math_ops.reduce_sum(labels_remapped, 1, keepdims=True)

  # Add the softmax loss.
  xent_loss = nn.softmax_cross_entropy_with_logits(
      logits=similarity_matrix, labels=labels_remapped)
  xent_loss = math_ops.reduce_mean(xent_loss, name='xentropy')

  if print_losses:
    xent_loss = logging_ops.Print(
        xent_loss, ['cross entropy:', xent_loss, 'l2loss:', l2loss])

  return l2loss + xent_loss

```
