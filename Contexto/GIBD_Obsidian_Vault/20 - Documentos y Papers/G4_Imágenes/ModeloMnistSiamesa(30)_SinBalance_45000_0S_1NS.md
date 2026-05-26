---
aliases: [ModeloMnistSiamesa(30)_SinBalance_45000_0S_1NS]
tags:
  - grupo/g4_imágenes
  - estado/util
  - tipo/documento
formato_original: txt
fecha_modificacion: 2022-06-10
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/CNN Marcas/Modelos 2022/ModeloMnistSiamesa(30)_SinBalance_45000_0S_1NS.txt"
tamanio_bytes: 6455
---

# ModeloMnistSiamesa(30)_SinBalance_45000_0S_1NS

Ruta interna: `GIBD/CNN Marcas/Modelos 2022/ModeloMnistSiamesa(30)_SinBalance_45000_0S_1NS.txt`

---

Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 28, 28, 1)]  0           []                               
                                                                                                  
 input_2 (InputLayer)           [(None, 28, 28, 1)]  0           []                               
                                                                                                  
 sequential (Sequential)        (None, 128)          325440      ['input_1[0][0]',                
                                                                  'input_2[0][0]']                
                                                                                                  
 lambda (Lambda)                (None, 128)          0           ['sequential[0][0]',             
                                                                  'sequential[1][0]']             
                                                                                                  
 dense_1 (Dense)                (None, 1)            129         ['lambda[0][0]']                 
                                                                                                  
==================================================================================================
Total params: 325,569
Trainable params: 324,993
Non-trainable params: 576
__________________________________________________________________________________________________
Epoch 1/30
14063/14063 [==============================] - 1535s 108ms/step - loss: 0.0460 - accuracy: 0.9853 - val_loss: -26.1495 - val_accuracy: 0.1073
Epoch 2/30
14063/14063 [==============================] - 1497s 106ms/step - loss: 0.0150 - accuracy: 0.9955 - val_loss: -29.4619 - val_accuracy: 0.1075
Epoch 3/30
14063/14063 [==============================] - 1456s 104ms/step - loss: 0.0098 - accuracy: 0.9972 - val_loss: -31.4255 - val_accuracy: 0.1074
Epoch 4/30
14063/14063 [==============================] - 1391s 99ms/step - loss: 0.0071 - accuracy: 0.9979 - val_loss: -34.0501 - val_accuracy: 0.1075
Epoch 5/30
14063/14063 [==============================] - 1292s 92ms/step - loss: 0.0056 - accuracy: 0.9983 - val_loss: -34.7144 - val_accuracy: 0.1075
Epoch 6/30
14063/14063 [==============================] - 1225s 87ms/step - loss: 0.0050 - accuracy: 0.9985 - val_loss: -35.6330 - val_accuracy: 0.1075
Epoch 7/30
14063/14063 [==============================] - 1228s 87ms/step - loss: 0.0038 - accuracy: 0.9990 - val_loss: -36.0429 - val_accuracy: 0.1075
Epoch 8/30
14063/14063 [==============================] - 1291s 92ms/step - loss: 0.0033 - accuracy: 0.9991 - val_loss: -36.4438 - val_accuracy: 0.1075
Epoch 9/30
14063/14063 [==============================] - 1314s 93ms/step - loss: 0.0029 - accuracy: 0.9991 - val_loss: -38.1785 - val_accuracy: 0.1075
Epoch 10/30
14063/14063 [==============================] - 1312s 93ms/step - loss: 0.0027 - accuracy: 0.9992 - val_loss: -38.2144 - val_accuracy: 0.1075
Epoch 11/30
14063/14063 [==============================] - 1323s 94ms/step - loss: 0.0024 - accuracy: 0.9994 - val_loss: -38.2437 - val_accuracy: 0.1075
Epoch 12/30
14063/14063 [==============================] - 1327s 94ms/step - loss: 0.0022 - accuracy: 0.9994 - val_loss: -38.6034 - val_accuracy: 0.1075
Epoch 13/30
14063/14063 [==============================] - 1330s 95ms/step - loss: 0.0020 - accuracy: 0.9995 - val_loss: -38.8031 - val_accuracy: 0.1075
Epoch 14/30
14063/14063 [==============================] - 1283s 91ms/step - loss: 0.0019 - accuracy: 0.9994 - val_loss: -39.0991 - val_accuracy: 0.1075
Epoch 15/30
14063/14063 [==============================] - 1278s 91ms/step - loss: 0.0016 - accuracy: 0.9995 - val_loss: -39.7530 - val_accuracy: 0.1075
Epoch 16/30
14063/14063 [==============================] - 1282s 91ms/step - loss: 0.0017 - accuracy: 0.9995 - val_loss: -40.2225 - val_accuracy: 0.1075
Epoch 17/30
14063/14063 [==============================] - 1287s 92ms/step - loss: 0.0015 - accuracy: 0.9996 - val_loss: -41.2955 - val_accuracy: 0.1075
Epoch 18/30
14063/14063 [==============================] - 1285s 91ms/step - loss: 0.0013 - accuracy: 0.9997 - val_loss: -41.1860 - val_accuracy: 0.1075
Epoch 19/30
14063/14063 [==============================] - 1290s 92ms/step - loss: 0.0014 - accuracy: 0.9996 - val_loss: -41.0650 - val_accuracy: 0.1075
Epoch 20/30
14063/14063 [==============================] - 1291s 92ms/step - loss: 0.0011 - accuracy: 0.9997 - val_loss: -41.9664 - val_accuracy: 0.1075
Epoch 21/30
14063/14063 [==============================] - 1296s 92ms/step - loss: 0.0012 - accuracy: 0.9997 - val_loss: -42.3390 - val_accuracy: 0.1075
Epoch 22/30
14063/14063 [==============================] - 1298s 92ms/step - loss: 0.0011 - accuracy: 0.9997 - val_loss: -41.8276 - val_accuracy: 0.1075
Epoch 23/30
14063/14063 [==============================] - 1300s 92ms/step - loss: 0.0012 - accuracy: 0.9996 - val_loss: -42.2172 - val_accuracy: 0.1075
Epoch 24/30
14063/14063 [==============================] - 1307s 93ms/step - loss: 9.9267e-04 - accuracy: 0.9998 - val_loss: -42.5770 - val_accuracy: 0.1075
Epoch 25/30
14063/14063 [==============================] - 1323s 94ms/step - loss: 0.0010 - accuracy: 0.9997 - val_loss: -42.9879 - val_accuracy: 0.1075
Epoch 26/30
14063/14063 [==============================] - 1312s 93ms/step - loss: 0.0011 - accuracy: 0.9997 - val_loss: -42.8805 - val_accuracy: 0.1075
Epoch 27/30
14063/14063 [==============================] - 1307s 93ms/step - loss: 9.1317e-04 - accuracy: 0.9998 - val_loss: -43.1587 - val_accuracy: 0.1075
Epoch 28/30
14063/14063 [==============================] - 1312s 93ms/step - loss: 8.8539e-04 - accuracy: 0.9998 - val_loss: -43.5615 - val_accuracy: 0.1075
Epoch 29/30
14063/14063 [==============================] - 1317s 94ms/step - loss: 9.1946e-04 - accuracy: 0.9997 - val_loss: -43.9017 - val_accuracy: 0.1075
Epoch 30/30
14063/14063 [==============================] - 1321s 94ms/step - loss: 8.3545e-04 - accuracy: 0.9998 - val_loss: -44.2969 - val_accuracy: 0.1075