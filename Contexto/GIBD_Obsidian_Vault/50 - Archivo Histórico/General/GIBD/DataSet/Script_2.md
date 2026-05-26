---
aliases: [Script]
tags:
  - grupo/general
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2025-07-14
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/DataSet/Train/Script.docx"
tamanio_bytes: 7411
---

# Script

Ruta interna: `GIBD/DataSet/Train/Script.docx`

---

El train conjunto se divide en 500 archivos TAR (cada uno de aproximadamente 1 GB) que contienen imágenes codificadas en JPG. Los archivos se encuentran en el train/directorio y se llaman images_000.tar, images_001.tar, ..., images_499.tar. Para descargarlos, acceda al siguiente enlace:
https://s3.amazonaws.com/google-landmark/train/images_000.tar
Y lo mismo para los demás archivos.
Usando el script proporcionado:
mkdir train && cd train
bash ../download-dataset.sh train 499

Esto descargará, verificará y extraerá automáticamente las imágenes al train directorio.
Nota: Este script descarga archivos en paralelo. Para ajustar el número de descargas paralelas, modifique NUM_PROCel script.
