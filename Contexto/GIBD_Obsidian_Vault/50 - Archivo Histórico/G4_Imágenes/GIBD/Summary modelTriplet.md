---
aliases: [Summary modelTriplet]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/documento
formato_original: txt
fecha_modificacion: 2023-04-25
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Summary modelTriplet.txt"
tamanio_bytes: 1891
---

# Summary modelTriplet

Ruta interna: `GIBD/Summary modelTriplet.txt`

---

Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 28, 28, 1)]  0           []                               
                                                                                                  
 input_2 (InputLayer)           [(None, 28, 28, 1)]  0           []                               
                                                                                                  
 input_3 (InputLayer)           [(None, 28, 28, 1)]  0           []                               
                                                                                                  
 sequential_1 (Sequential)      (None, 128)          280960      ['input_1[0][0]',                
                                                                  'input_2[0][0]',                
                                                                  'input_3[0][0]']                
                                                                                                  
 vectors (Concatenate)          (None, 384)          0           ['sequential_1[0][0]',           
                                                                  'sequential_1[1][0]',           
                                                                  'sequential_1[2][0]']           
                                                                                                  
==================================================================================================
Total params: 280,960
Trainable params: 280,256
Non-trainable params: 704