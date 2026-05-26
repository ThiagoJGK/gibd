---
aliases: [Instrucciones]
tags:
  - grupo/general
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2025-07-07
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/DataSet/Instrucciones.docx"
tamanio_bytes: 8155
---

# Instrucciones

Ruta interna: `GIBD/DataSet/Instrucciones.docx`

---

Comprobando la descarga
También ponemos a su disposición archivos md5sum para comprobar la integridad de los archivos descargados. Cada archivo md5sum corresponde a uno de los archivos TAR mencionados anteriormente; se encuentran en los directorios md5sum/index/, md5sum/test/y md5sum/train/ , con nombres de archivo md5.images_000.txt, md5.images_001.txt, etc. Por ejemplo, el archivo md5sum correspondiente al images_000.tararchivo del indexconjunto se puede encontrar en el siguiente enlace:
https://s3.amazonaws.com/google-landmark/md5sum/index/md5.images_000.txt
Y lo mismo para los demás archivos.
Si utiliza el download-dataset.shscript proporcionado, la integridad de los archivos se comprobará inmediatamente después de la descarga.
Extrayendo los datos
Recomendamos extraer el conjunto de archivos TAR correspondiente a cada división del conjunto de datos a un directorio por cada división; es decir, los indexarchivos TAR se extraen a un indexdirectorio; trainlos archivos TAR se extraen a un traindirectorio; testlos archivos TAR se extraen a un testdirectorio. Esto se realiza automáticamente si se utilizan las instrucciones/script de descarga anteriores.
La estructura de directorios de los datos de imagen es la siguiente: cada imagen se almacena en el directorio ${a}//.jpg , donde , y son las tres primeras letras del ID de la imagen, y es el ID ${b}de la imagen que se encuentra en los archivos CSV. Por ejemplo, una imagen con el ID se almacenaría en .${c}${id}${a}${b}${c}${id}0123456789abcdef0/1/2/0123456789abcdef.jpg
Código de cálculo métrico
Los scripts de cálculo métrico están disponibles a través del repositorio de DELF en GitHub ; consulte los scripts de Python compute_recognition_metrics.py. compute_retrieval_metrics.pyEstos scripts aceptan como entrada los archivos de datos de referencia, junto con las predicciones en el formato enviado a Kaggle.
