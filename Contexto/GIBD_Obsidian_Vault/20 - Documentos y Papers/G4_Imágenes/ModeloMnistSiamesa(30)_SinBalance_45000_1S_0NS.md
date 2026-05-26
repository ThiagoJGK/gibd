---
aliases: [ModeloMnistSiamesa(30)_SinBalance_45000_1S_0NS]
tags:
  - grupo/g4_imágenes
  - estado/util
  - tipo/documento
formato_original: txt
fecha_modificacion: 2022-06-11
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/CNN Marcas/Modelos 2022/ModeloMnistSiamesa(30)_SinBalance_45000_1S_0NS.txt"
tamanio_bytes: 6431
---

# ModeloMnistSiamesa(30)_SinBalance_45000_1S_0NS

Ruta interna: `GIBD/CNN Marcas/Modelos 2022/ModeloMnistSiamesa(30)_SinBalance_45000_1S_0NS.txt`

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
14063/14063 [==============================] - 1505s 107ms/step - loss: 0.0475 - accuracy: 0.9844 - val_loss: 35.2948 - val_accuracy: 0.0997
Epoch 2/30
14063/14063 [==============================] - 1430s 102ms/step - loss: 0.0150 - accuracy: 0.9953 - val_loss: 37.8679 - val_accuracy: 0.0998
Epoch 3/30
14063/14063 [==============================] - 1380s 98ms/step - loss: 0.0095 - accuracy: 0.9972 - val_loss: 39.6982 - val_accuracy: 0.0998
Epoch 4/30
14063/14063 [==============================] - 1553s 110ms/step - loss: 0.0067 - accuracy: 0.9981 - val_loss: 43.5946 - val_accuracy: 0.0998
Epoch 5/30
14063/14063 [==============================] - 1709s 122ms/step - loss: 0.0053 - accuracy: 0.9984 - val_loss: 46.4606 - val_accuracy: 0.0998
Epoch 6/30
14063/14063 [==============================] - 1580s 112ms/step - loss: 0.0045 - accuracy: 0.9987 - val_loss: 46.9558 - val_accuracy: 0.0998
Epoch 7/30
14063/14063 [==============================] - 1484s 106ms/step - loss: 0.0037 - accuracy: 0.9989 - val_loss: 47.8816 - val_accuracy: 0.0998
Epoch 8/30
14063/14063 [==============================] - 1511s 107ms/step - loss: 0.0030 - accuracy: 0.9991 - val_loss: 48.9153 - val_accuracy: 0.0998
Epoch 9/30
14063/14063 [==============================] - 1447s 103ms/step - loss: 0.0028 - accuracy: 0.9992 - val_loss: 50.9701 - val_accuracy: 0.0998
Epoch 10/30
14063/14063 [==============================] - 1415s 101ms/step - loss: 0.0023 - accuracy: 0.9993 - val_loss: 51.8561 - val_accuracy: 0.0998
Epoch 11/30
14063/14063 [==============================] - 1363s 97ms/step - loss: 0.0023 - accuracy: 0.9993 - val_loss: 51.5484 - val_accuracy: 0.0998
Epoch 12/30
14063/14063 [==============================] - 1218s 87ms/step - loss: 0.0019 - accuracy: 0.9995 - val_loss: 52.0941 - val_accuracy: 0.0998
Epoch 13/30
14063/14063 [==============================] - 1222s 87ms/step - loss: 0.0019 - accuracy: 0.9994 - val_loss: 53.1629 - val_accuracy: 0.0998
Epoch 14/30
14063/14063 [==============================] - 1223s 87ms/step - loss: 0.0017 - accuracy: 0.9996 - val_loss: 53.8102 - val_accuracy: 0.0998
Epoch 15/30
14063/14063 [==============================] - 1227s 87ms/step - loss: 0.0016 - accuracy: 0.9995 - val_loss: 54.7324 - val_accuracy: 0.0998
Epoch 16/30
14063/14063 [==============================] - 1228s 87ms/step - loss: 0.0015 - accuracy: 0.9996 - val_loss: 55.8462 - val_accuracy: 0.0998
Epoch 17/30
14063/14063 [==============================] - 1229s 87ms/step - loss: 0.0014 - accuracy: 0.9996 - val_loss: 56.4759 - val_accuracy: 0.0998
Epoch 18/30
14063/14063 [==============================] - 1234s 88ms/step - loss: 0.0013 - accuracy: 0.9997 - val_loss: 56.4125 - val_accuracy: 0.0998
Epoch 19/30
14063/14063 [==============================] - 1231s 88ms/step - loss: 0.0011 - accuracy: 0.9997 - val_loss: 57.2799 - val_accuracy: 0.0998
Epoch 20/30
14063/14063 [==============================] - 1231s 88ms/step - loss: 0.0012 - accuracy: 0.9997 - val_loss: 58.2166 - val_accuracy: 0.0998
Epoch 21/30
14063/14063 [==============================] - 1232s 88ms/step - loss: 0.0012 - accuracy: 0.9997 - val_loss: 57.8689 - val_accuracy: 0.0998
Epoch 22/30
14063/14063 [==============================] - 1235s 88ms/step - loss: 0.0010 - accuracy: 0.9997 - val_loss: 58.2639 - val_accuracy: 0.0998
Epoch 23/30
14063/14063 [==============================] - 1235s 88ms/step - loss: 0.0010 - accuracy: 0.9998 - val_loss: 58.5289 - val_accuracy: 0.0998
Epoch 24/30
14063/14063 [==============================] - 1236s 88ms/step - loss: 0.0010 - accuracy: 0.9997 - val_loss: 58.3466 - val_accuracy: 0.0998
Epoch 25/30
14063/14063 [==============================] - 1242s 88ms/step - loss: 0.0010 - accuracy: 0.9997 - val_loss: 59.0156 - val_accuracy: 0.0998
Epoch 26/30
14063/14063 [==============================] - 1240s 88ms/step - loss: 9.2979e-04 - accuracy: 0.9998 - val_loss: 59.2866 - val_accuracy: 0.0998
Epoch 27/30
14063/14063 [==============================] - 1252s 89ms/step - loss: 8.0866e-04 - accuracy: 0.9998 - val_loss: 59.7781 - val_accuracy: 0.0998
Epoch 28/30
14063/14063 [==============================] - 1257s 89ms/step - loss: 8.7220e-04 - accuracy: 0.9998 - val_loss: 59.7925 - val_accuracy: 0.0998
Epoch 29/30
14063/14063 [==============================] - 1245s 89ms/step - loss: 9.2001e-04 - accuracy: 0.9997 - val_loss: 60.2214 - val_accuracy: 0.0998
Epoch 30/30
14063/14063 [==============================] - 1244s 88ms/step - loss: 8.1520e-04 - accuracy: 0.9998 - val_loss: 60.1355 - val_accuracy: 0.0998