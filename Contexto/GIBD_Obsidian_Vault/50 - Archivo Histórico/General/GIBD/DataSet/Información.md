---
aliases: [Información]
tags:
  - grupo/general
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2025-07-07
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/DataSet/Datos del índice (index) /lista de imágenes y metadatos/Información.docx"
tamanio_bytes: 7393
---

# Información

Ruta interna: `GIBD/DataSet/Datos del índice (index) /lista de imágenes y metadatos/Información.docx`

---

Index.csv CSV de una sola columna con campo de identificación. Id es una cadena de 16 caracteres.
Index_image_to_landmark.csv CSV con los campos id y landmark_id: id es una cadena de 16 caracteres y landmark_id es un entero.
Index_label_to_category.csv CSV con landmark_id y campos de categoría: landmark_id es un entero y category es una URL de Wikimedia que hace referencia a la definición de la clase. 
Index_label_to_hierarchical.csv CSV con los campos "histórico_id", "categoría", "supercategoría", "etiqueta_jerárquica" y "natural_o_hecho_por_humanos": landmark_id es un entero, category es una URL de Wikimedia que hace referencia a la definición de la clase, es supercategory una cadena que hace referencia al tipo de hito extraído de Wikimedia, hierarchical_label es una cadena que corresponde a la etiqueta jerárquica, natural_or_human_made es una cadena que indica si el hito es natural o hecho por el hombre.