---
aliases: [Analisis de resultados de los intentos fallidos]
tags:
  - grupo/general
  - estado/util
  - tipo/documento
formato_original: docx
fecha_modificacion: 2022-08-16
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Analisis de resultados de los intentos fallidos.docx"
tamanio_bytes: 831593
---

# Analisis de resultados de los intentos fallidos

Ruta interna: `GIBD/Analisis de resultados de los intentos fallidos.docx`

---

Distancia:  31.245703  al vector en la posición:  187  desde el vector que se encuentra en:  9
Imagen de BD:  85  Imagen de consulta:  130
Más cercanos:  [[31.245703, 187], [32.232826, 108], [35.82196, 65], [38.065567, 67], [38.41814, 92]]

DB								QUERY
                          

Si bien la imagen es similar, vemos que la dibujada posee una mayor altura en relación a la imagen de la DB. Vemos como la primera ocupa sólo ⅓ del espacio, a diferencia de la segunda que ocupa todo.
Además podemos observar como la línea recta debajo de la curva es más “corta” en comparación a la imagen de la DB.


Distancia:  31.896038  al vector en la posición:  108  desde el vector que se encuentra en:  11
Imagen de BD:  197  Imagen de consulta:  132
Mas cercanos:  [[31.896038, 108], [35.03962, 48], [36.916687, 71], [37.7245, 187], [37.858948, 44]]

DB								QUERY
                              

Nuevamente vemos que la imagen es muy similar, sin embargo podemos observar como las siguientes secciones:
           2) 

Se encuentran en una posición mucho más alejada entre sí y por debajo en comparación a la imagen dibujada. 


Distancia:  24.68142  al vector en la posición:  20  desde el vector que se encuentra en:  13
Imagen de BD:  117  Imagen de consulta:  134 (2)
Mas cercanos:  [[24.68142, 20], [30.149788, 23], [30.786058, 3], [32.86965, 117], [33.4562, 81]]

DB								QUERY
                            
Las imágenes muestran varias similitudes, sin embargo al analizarlas a detalle podemos ver que la imagen DB está más estirada, mientras que la QUERY está comprimida.	Viendo más a detalle logramos ver una sección donde la imagen QUERY deja de ser continua,además de notar ausencia de píxeles en otros sectores.


Distancia:  35.005817  al vector en la posición:  81  desde el vector que se encuentra en:  14
Imagen de BD:  172  Imagen de consulta:  134
Mas cercanos:  [[35.005817, 81], [38.33793, 179], [38.444504, 30], [38.592644, 197], [38.688187, 83]]

DB								QUERY
                           
Las imágenes muestran una similitud muy grande. El error pudo haberse causado por una diferencia de expansión y la diferencia de posicion de las partes inferiores de las imágenes.

Distancia:  22.337666  al vector en la posición:  113  desde el vector que se encuentra en:  22
Imagen de BD:  200  Imagen de consulta:  175
Más cercanos:  [[22.337666, 113], [29.172201, 128], [33.0814, 37], [33.91303, 62], [35.237846, 44]]

DB								QUERY

la imagen es bastante similar pero la dibujada es menos ancha en la parte del medio lo que cambia la proporción de distancia de las líneas verticales de los laterales


Distancia:  31.828312  al vector en la posición:  45  desde el vector que se encuentra en:  23
Imagen de BD:  14  Imagen de consulta:  192
Mas cercanos:  [[31.828312, 45], [33.561768, 54], [33.928177, 140], [35.640728, 174], [36.129982, 89]]

DB								QUERY

La imagen es bastante similar pero en la parte superior, la imagen dibujada ocupa menos espacio y tiene menos inclinación que en la original y en la parte inferior tiene menos inclinación y es más grande en proporción.

Distancia:  23.212982  al vector en la posición:  38  desde el vector que se encuentra en:  36
Imagen de BD:  133  Imagen de consulta:  79
Mas cercanos:  [[23.212982, 38], [28.470385, 82], [31.323673, 62], [34.821823, 39], [35.56939, 28]]
DB								QUERY


En este caso, podemos observar que la imagen dibujada, en su parte superior es más angosta en comparación a la imagen de la DB. 
Además, vemos que en la sección que se muestra a continuación hay una diferencia de posicionamiento y longitud con respecto a la DB ya que esta se encuentra más
 a la izquierda y posee un tamaño mayor.
  




