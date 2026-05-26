---
aliases: [Script_]
tags:
  - grupo/general
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2025-07-14
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/DataSet/Datos de prueba (test)/Script_.docx"
tamanio_bytes: 7407
---

# Script_

Ruta interna: `GIBD/DataSet/Datos de prueba (test)/Script_.docx`

---

El test conjunto se divide en 20 archivos TAR (cada uno de aproximadamente 500 MB) que contienen imágenes codificadas en JPG. Los archivos se encuentran en el test/directorio y se llaman images_000.tar, images_001.tar, ..., images_019.tar. Para descargarlos, acceda al siguiente enlace:
https://s3.amazonaws.com/google-landmark/test/images_000.tar
Y lo mismo para los demás archivos.
Usando el script proporcionado
mkdir test && cd test
bash ../download-dataset.sh test 19

Esto descargará, verificará y extraerá automáticamente las imágenes al test directorio.
Nota: Este script descarga archivos en paralelo. Para ajustar el número de descargas paralelas, modifique NUM_PROCel script.

