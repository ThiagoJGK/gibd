---
aliases: [Preprocesamiento del Texto para entrenar BERT]
tags:
  - grupo/g3_texto
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2023-04-21
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Procesamiento del Lenguaje Natural/Preprocesamiento del Texto para entrenar BERT.docx"
tamanio_bytes: 318315
---

# Preprocesamiento del Texto para entrenar BERT

Ruta interna: `GIBD/Procesamiento del Lenguaje Natural/Preprocesamiento del Texto para entrenar BERT.docx`

---

Las palabras en los fragmentos de entrada para el entrenamiento de BERT se codifican mediante un proceso llamado "tokenización". La tokenización es el proceso de dividir el texto en unidades más pequeñas llamadas "tokens", que pueden ser palabras, subpalabras, caracteres u otras unidades, dependiendo de la implementación específica de BERT.
En BERT, la tokenización se realiza utilizando una técnica llamada "tokenización de subpalabras" (subword tokenization), donde las palabras se dividen en subpalabras basadas en la frecuencia de aparición en el corpus de entrenamiento. Esta técnica permite manejar palabras raras o poco frecuentes, así como palabras compuestas o con prefijos/sufijos de manera más eficiente, ya que estas palabras pueden ser divididas en subpalabras más comunes y compartidas.
El proceso de tokenización en BERT generalmente sigue los siguientes pasos:
Se convierte el texto en minúsculas: Todas las letras del texto se convierten a minúsculas para normalizar el texto y reducir la variabilidad.
Se divide el texto en palabras: El texto se divide en palabras utilizando un espacio en blanco como delimitador. Las puntuaciones y caracteres especiales pueden tratarse como tokens separados o pueden ser eliminados, dependiendo de la implementación específica.
Se aplica la tokenización de subpalabras: Cada palabra se tokeniza en subpalabras basadas en la frecuencia de aparición en el corpus de entrenamiento. Esto implica dividir las palabras en subpalabras más comunes y compartidas, y agregar un prefijo especial "##" a las subpalabras que no son la primera subpalabra de una palabra original.
Se agregan tokens especiales: Se añaden tokens especiales de inicio ([CLS]) y separación ([SEP]) al principio y al final de cada oración o fragmento de texto, y posiblemente otros tokens especiales como [MASK] para la estrategia de Máscara de palabras (MLM) durante el pre-entrenamiento.
Se asignan identificadores numéricos: Cada token se asigna a un identificador numérico (índice) según un vocabulario predefinido, que mapea los tokens a números enteros. Estos identificadores numéricos son los que finalmente se utilizan como entrada para el modelo BERT durante el entrenamiento y la inferencia.
Cabe destacar que el proceso de tokenización puede variar según la implementación específica de BERT y el idioma utilizado, ya que diferentes implementaciones pueden tener vocabularios y reglas de tokenización específicas para cada idioma.


En BERT, la entrada para el entrenamiento es un conjunto de vectores de códigos de tokens de longitud fija. Después de pasar por el proceso de tokenización, donde se convierten las palabras en códigos de tokens, los fragmentos de texto se representan como secuencias de códigos de tokens. Estas secuencias de códigos de tokens pueden tener longitudes variables, pero se ajustan a una longitud máxima permitida mediante el uso de relleno (padding) con tokens especiales ([PAD]).
Durante el entrenamiento de BERT, los fragmentos de texto se agrupan en lotes (batches) y se representan como matrices de códigos de tokens, donde cada fila de la matriz corresponde a una secuencia de códigos de tokens de una oración o fragmento de texto. Estas matrices de códigos de tokens tienen dimensiones fijas, donde el número de filas representa la cantidad de oraciones o fragmentos de texto en el lote y el número de columnas representa la longitud máxima permitida de las secuencias de códigos de tokens después del relleno.
Por lo tanto, la entrada para el entrenamiento de BERT es una matriz de códigos de tokens de longitud fija, donde cada elemento de la matriz es un número entero que representa un código de token correspondiente a una palabra, subpalabra o caracter en el fragmento de texto. Esta matriz de códigos de tokens de longitud fija se utiliza como entrada para el modelo BERT durante el entrenamiento y la inferencia.
