---
aliases: [GIBD2025_ViT_Experimentos]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2025-09-11
origen_zip: GIBD-20260521T205218Z-3-002.zip
ruta_interna: "GIBD/ViT/GIBD2025_ViT_Experimentos.docx"
tamanio_bytes: 442772
---

# GIBD2025_ViT_Experimentos

Ruta interna: `GIBD/ViT/GIBD2025_ViT_Experimentos.docx`

---

Para evaluar el desempeño de distintos métodos de extracción de embeddings con modelos Vision Transformer (ViT), se realizaron una serie de experimentos empleando dos arquitecturas: ViT-Base y ViT-Large. En todos los casos, las consultas consistieron en 80 imágenes, comparadas contra una base de datos de 10k imágenes.
Se implementaron cuatro variantes principales de extracción de características (denominadas Casos):
Caso CLS: se utilizó directamente el token de clasificación (CLS) de la última capa como representación de la imagen. Dimensión del vector característico de salida: 1024.
Caso Max Pooling: se aplicó un pooling sobre los embeddings de los parches de la última capa, considerando tanto la variante mean como max. Dimensión del vector característico de salida: 1024.
Caso Multi-Layer: se combinaron salidas intermedias de distintos bloques del encoder. Se aplicaron tres estrategias:
CLS multi-capa: tomando el token CLS de varias capas y promediándolos. Dimensión del vector característico de salida: 1024.
Max multi-capa: aplicando max sobre los parches (sin incluir el CLS) en varias capas y promediando los resultados. Dimensión del vector característico de salida: 1024.
Flatten multi-capa: aplanando directamente las salidas completas de cada capa (con o sin CLS), normalizando cada vector y luego calculando un promedio entre capas. Dimensión del vector característico de salida: 200704 (sin CLS), 201728 (con CLS).
Caso Flatten: se utilizó directamente la salida completa de la última capa aplanada. En algunas variantes se excluyó el token CLS. Dimensión del vector característico de salida: 200704 (sin CLS), 201728 (con CLS).
Los experimentos realizados se pueden dividir en tres bloques que se corresponden con la Tabla 1, Tabla 2 y Tabla 3 respectivamente.
En el primer bloque se hicieron pruebas combinando los diferentes modelos y casos explicados anteriormente, a su vez se aplicaron otras variaciones relacionadas al pre-procesamiento de las imágenes y a la obtención del vector característico final utilizado para medir la similitud.
En el segundo bloque se realizaron 24 pruebas secuenciales acumulando progresivamente bloques correspondientes a la arquitectura del modelo ViT-Large desde el bloque 0 unitariamente hasta la combinación de los bloques de 0 a 23. Cada prueba fue una aplicación del Caso Multi-Layer basado en la estrategia Flatten, normalizado, con gaussian blur y excluyendo el CLS. La tabla 2 muestra un resumen de las 24 pruebas, suficiente para reflejar el comportamiento.
En el tercer bloque se realizaron 24 pruebas secuenciales de cada uno de los bloques individualmente correspondientes a la arquitectura del modelo ViT-Large desde el bloque 0 unitariamente hasta el bloque 23 unitariamente. Cada prueba fue una aplicación del Caso Multi-Layer basado en la estrategia Flatten, normalizado, con gaussian blur y excluyendo el CLS. La tabla 3 muestra un resumen de las 24 pruebas, suficiente para reflejar el comportamiento.
Tabla 1.  Resultados correspondientes al primer bloque de experimentos.

Tabla 2.  Resultados correspondientes al segundo bloque de experimentos.

Tabla 3.  Resultados correspondientes al tercer bloque de experimentos.
Glosario de las tablas de resultados:
vit-b: modelo ViT-Base.
vit-l: modelo ViT-Large.
cls: uso del caso 1 descrito previamente (Caso CLS).
max: uso del caso 2 descrito previamente (Caso Max-Pooling).
ml-1: uso del caso 3 descrito previamente basado en la estrategia CLS (Caso Multi-Layer).
ml-2: uso del caso 3 descrito previamente basado en la estrategia MAX (Caso Multi-Layer).
ml-4: uso del caso 3 descrito previamente basado en la estrategia FLATTEN (Caso Multi-Layer).
flat: uso del caso 4 descrito previamente (Caso Flatten).
[x,y]: indica que se usaron los bloques x e y de la arquitectura para combinar.
[x-y]: indica que se usó desde el bloque x hasta el bloque y de la arquitectura para combinar.
N: indica que el vector de características usado para comparar fue normalizado.
B: indica que a las imágenes se les aplicó un gaussian blur como pre-procesamiento.
R: indica que a las imágenes se las transformó en el rectángulo mínimo como pre-procesamiento.
s/c: indica que se excluyó el CLS antes de aplicar cualquier operación para obtener el vector característico para comparar.
Tabla 4.  Tiempos de ejecución respecto del modelo y caso.

Gráficos




