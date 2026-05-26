---
aliases: [log_resultado_primer_entrenamiento(cpu)]
tags:
  - grupo/g4_imágenes
  - estado/util
  - tipo/documento
formato_original: txt
fecha_modificacion: 2022-07-07
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/CNN Marcas/Función de Distancia/log_resultado_primer_entrenamiento(cpu).txt"
tamanio_bytes: 6580
---

# log_resultado_primer_entrenamiento(cpu)

Ruta interna: `GIBD/CNN Marcas/Función de Distancia/log_resultado_primer_entrenamiento(cpu).txt`

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
46875/46875 [==============================] - 1347s 28ms/step - loss: 0.0207 - accuracy: 0.9933 - val_loss: -31.5905 - val_accuracy: 0.1111
Epoch 2/30
46875/46875 [==============================] - 1324s 28ms/step - loss: 0.0037 - accuracy: 0.9989 - val_loss: -37.3892 - val_accuracy: 0.1110
Epoch 3/30
46875/46875 [==============================] - 1333s 28ms/step - loss: 0.0020 - accuracy: 0.9995 - val_loss: -40.9874 - val_accuracy: 0.1111
Epoch 4/30
46875/46875 [==============================] - 1339s 29ms/step - loss: 0.0012 - accuracy: 0.9997 - val_loss: -43.3192 - val_accuracy: 0.1110
Epoch 5/30
46875/46875 [==============================] - 1346s 29ms/step - loss: 9.2754e-04 - accuracy: 0.9998 - val_loss: -44.5169 - val_accuracy: 0.1110
Epoch 6/30
46875/46875 [==============================] - 1328s 28ms/step - loss: 7.0128e-04 - accuracy: 0.9998 - val_loss: -44.6248 - val_accuracy: 0.1110
Epoch 7/30
46875/46875 [==============================] - 1346s 29ms/step - loss: 6.3606e-04 - accuracy: 0.9999 - val_loss: -45.2539 - val_accuracy: 0.1110
Epoch 8/30
46875/46875 [==============================] - 1364s 29ms/step - loss: 5.1525e-04 - accuracy: 0.9999 - val_loss: -46.2130 - val_accuracy: 0.1110
Epoch 9/30
46875/46875 [==============================] - 1379s 29ms/step - loss: 4.3457e-04 - accuracy: 0.9999 - val_loss: -47.2473 - val_accuracy: 0.1110
Epoch 10/30
46875/46875 [==============================] - 1416s 30ms/step - loss: 4.2961e-04 - accuracy: 0.9999 - val_loss: -47.4014 - val_accuracy: 0.1111
Epoch 11/30
46875/46875 [==============================] - 1430s 31ms/step - loss: 3.8539e-04 - accuracy: 0.9999 - val_loss: -47.8968 - val_accuracy: 0.1110
Epoch 12/30
46875/46875 [==============================] - 1455s 31ms/step - loss: 3.2034e-04 - accuracy: 0.9999 - val_loss: -48.8279 - val_accuracy: 0.1110
Epoch 13/30
46875/46875 [==============================] - 1476s 31ms/step - loss: 2.8413e-04 - accuracy: 0.9999 - val_loss: -49.3450 - val_accuracy: 0.1110
Epoch 14/30
46875/46875 [==============================] - 1524s 33ms/step - loss: 3.0307e-04 - accuracy: 0.9999 - val_loss: -49.4853 - val_accuracy: 0.1110
Epoch 15/30
46875/46875 [==============================] - 1552s 33ms/step - loss: 2.6876e-04 - accuracy: 0.9999 - val_loss: -49.7889 - val_accuracy: 0.1110
Epoch 16/30
46875/46875 [==============================] - 1582s 34ms/step - loss: 2.8398e-04 - accuracy: 0.9999 - val_loss: -50.3135 - val_accuracy: 0.1110
Epoch 17/30
46875/46875 [==============================] - 1623s 35ms/step - loss: 2.3037e-04 - accuracy: 0.9999 - val_loss: -50.1998 - val_accuracy: 0.1111
Epoch 18/30
46875/46875 [==============================] - 1957s 42ms/step - loss: 2.4183e-04 - accuracy: 0.9999 - val_loss: -50.4978 - val_accuracy: 0.1111
Epoch 19/30
46875/46875 [==============================] - 2103s 45ms/step - loss: 2.4229e-04 - accuracy: 0.9999 - val_loss: -50.7400 - val_accuracy: 0.1110
Epoch 20/30
46875/46875 [==============================] - 2198s 47ms/step - loss: 2.4036e-04 - accuracy: 0.9999 - val_loss: -50.7962 - val_accuracy: 0.1110
Epoch 21/30
46875/46875 [==============================] - 2274s 49ms/step - loss: 1.8649e-04 - accuracy: 1.0000 - val_loss: -51.2857 - val_accuracy: 0.1111
Epoch 22/30
46875/46875 [==============================] - 2378s 51ms/step - loss: 2.2054e-04 - accuracy: 1.0000 - val_loss: -51.6905 - val_accuracy: 0.1110
Epoch 23/30
46875/46875 [==============================] - 2491s 53ms/step - loss: 1.7376e-04 - accuracy: 1.0000 - val_loss: -52.1926 - val_accuracy: 0.1110
Epoch 24/30
46875/46875 [==============================] - 2542s 54ms/step - loss: 1.8681e-04 - accuracy: 0.9999 - val_loss: -52.1705 - val_accuracy: 0.1110
Epoch 25/30
46875/46875 [==============================] - 2637s 56ms/step - loss: 1.5575e-04 - accuracy: 1.0000 - val_loss: -52.1494 - val_accuracy: 0.1110
Epoch 26/30
46875/46875 [==============================] - 2661s 57ms/step - loss: 1.6335e-04 - accuracy: 1.0000 - val_loss: -52.4995 - val_accuracy: 0.1110
Epoch 27/30
46875/46875 [==============================] - 2785s 59ms/step - loss: 1.7662e-04 - accuracy: 1.0000 - val_loss: -52.7272 - val_accuracy: 0.1110
Epoch 28/30
46875/46875 [==============================] - 2929s 62ms/step - loss: 1.6680e-04 - accuracy: 1.0000 - val_loss: -52.7396 - val_accuracy: 0.1110
Epoch 29/30
46875/46875 [==============================] - 2976s 63ms/step - loss: 1.6957e-04 - accuracy: 1.0000 - val_loss: -52.6381 - val_accuracy: 0.1111
Epoch 30/30
46875/46875 [==============================] - 3074s 66ms/step - loss: 1.6288e-04 - accuracy: 1.0000 - val_loss: -52.6700 - val_accuracy: 0.1110
<keras.callbacks.History at 0x275de602280>