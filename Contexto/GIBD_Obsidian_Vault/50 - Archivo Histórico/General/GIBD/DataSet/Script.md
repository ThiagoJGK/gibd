---
aliases: [Script]
tags:
  - grupo/general
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2025-07-14
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/DataSet/Datos del índice (index) /Script.docx"
tamanio_bytes: 7454
---

# Script

Ruta interna: `GIBD/DataSet/Datos del índice (index) /Script.docx`

---

El index conjunto se divide en 100 archivos TAR (cada uno con un tamaño aproximado de 850 MB) que contienen imágenes codificadas en JPG. Los archivos se encuentran en el index/directorio y se llaman images_000.tar, images_001.tar, ..., images_099.tar. Para descargarlos, acceda al siguiente enlace:

https://s3.amazonaws.com/google-landmark/index/images_000.tar

Y lo mismo para los demás archivos.
Usando el script proporcionado
mkdir index && cd index
bash ../download-dataset.sh index 99

Esto descargará, verificará y extraerá automáticamente las imágenes al index directorio.
Nota: Este script descarga archivos en paralelo. Para ajustar el número de descargas paralelas, modifique NUM_PROCel script.
