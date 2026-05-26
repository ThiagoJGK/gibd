---
aliases: [Llama-3-8b-Finetune-Reglamentación]
tags:
  - grupo/g3_texto
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2024-08-17
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/BERT - GPT - Llama/Llama-3-8b-Finetune-Reglamentación.docx"
tamanio_bytes: 2136422
---

# Llama-3-8b-Finetune-Reglamentación

Ruta interna: `GIBD/BERT - GPT - Llama/Llama-3-8b-Finetune-Reglamentación.docx`

---

Recursos Utilizados
Modelos Originales
Llama 3 8B de Kaggle: Meta | Llama 3 | Kaggle
Llama 3.1 8B Instruct: Meta | Llama 3.1 | Kaggle
Gemma2-2b: google/gemma-2-2b · Hugging Face 
Gemma2-2b-it: Google | Gemma 2 | Kaggle
Phi3 Mini 4k instruct: microsoft/Phi-3-mini-4k-instruct · Hugging Face

        Modelos Finetuned y Adapters subidos a Hugginface
Llama3-8b-Finetuned-Versions 
Llama3.1-8b-Finetuned-Versions
GGUF Finetuned Models
Gemma2-2b-Finetuned -Versions
Adapters-Models
Gemma-2b-finetuned
Phi3-Finetuned-Models 


Notebooks
Notebook de Expansión del Dataset Original: Data Augmentation Reglamentacion (kaggle.com)
Notebook de Entrenamientos: Llama3-8B-Finetune (kaggle.com)
Notebook Merge con el modelo base: Llama-3-8b-Merge (kaggle.com)
Notebook Entrenamiento con Phi3 4k Mini: HF-Phi3-Mini-4k-Gemma2-2b-Finetune (kaggle.com)
Test de modelos GGUF Finetuned: Test-Modelo-GGUF-FineTuned (kaggle.com)
Test-Gemma2-2b-it-reglamentacion-v1: Test-Gemma2-2b-it-reglamentacion-v1 (kaggle.com)

Datasets
Dataset Original: HuggMaxi/dataset_reglamentoUTN · Datasets at Hugging Face
Dataset Expandido: PabloSuaLap/Reglamentacion-Extended · Datasets at Hugging Face

Reportes
Reporte Comparativo de las distintas ejecuciones para el modelo Llama 3 (Weigths and Biases API): https://api.wandb.ai/links/5jr49qafz-utn-frcu/w76fdj1s 
Reporte Comparativo de las distintas ejecuciones para el modelo Llama 3.1 (Weigths and Biases API): https://api.wandb.ai/links/5jr49qafz-utn-frcu/gv1jn0ha 
Reporte Comparativo de las distintas ejecuciones para el modelo Gemma2-2b (Weigths and Biases API): https://api.wandb.ai/links/5jr49qafz-utn-frcu/vqov2z9s 
Reporte Comparativo de las distintas ejecuciones para el modelo Gemma2-2b-it (Weigths and Biases API):https://api.wandb.ai/links/5jr49qafz-utn-frcu/6wkx206y 
Reporte Comparativo de las distintas ejecuciones para el modelo Phi3-Mini-4k (Weigths and Biases API): https://api.wandb.ai/links/5jr49qafz-utn-frcu/2kv4a344 


      Recursos y Enlaces de Referencia
Proceso basado en el siguiente artículo: Fine-Tuning Llama 3 and Using It Locally: A Step-by-Step Guide | DataCamp
Conversor de modelos Safetensors a GGUF y Cuantizar: GGUF My Repo - a Hugging Face Space by ggml-org

Parámetros y Resultados por Ejecución
Ejecución por Defecto
Parámetros 0


Resultados 0

https://api.wandb.ai/links/5jr49qafz-utn-frcu/w4uqzyyw 



Primera Ejecución
Parámetros 1



Resultados 1 




Segunda Ejecución
Parámetros 2


Resultados 2




Tercera Ejecución
Parámetros 3

Resultados 3


Cuarta Ejecución (Sin Early Stopping Callback)
Parámetros 4


Resultados 4



Quinta Ejecución (Dataset Extendido)
Parámetros 5

Resultados 5
No se pudo completar la ejecución por exceder el tiempo de Kaggle (12hs)
Los resultados parciales se encuentran en el reporte.
Sexta Ejecución (Dataset Extendido)
Parámetros 6


Resultados 6



Séptima Ejecución (Modelo Llama 3.1 8B Instruct)
Parámetros 7


Resultados 7





Test del Modelo GGUF
Llama 3 8B Fine Tuned V1


Como se puede observar el modelo por alguna razón está repitiendo la última oración en forma de loop indefinido. Sin embargo, la primera oración es la respuesta correcta, el sucesivo contenido de la respuesta es información adicional que ocasiona el loop.


Llama 3.1 8B Instruct Fine Tuned V1

El modelo presenta problemas de compatibilidad con la versión 0.3.0 de Ollama debido a una actualización reciente del proyecto llama.cpp por lo que la única forma de cargar el modelo fue por medio del software LM Studio, el cual al intentar realizar una consulta no devolvió un resultado útil.

Issue donde se discute el problema: Add llama 3.1 rope scaling factors to llama conversion and inference by jmorganca · Pull Request #8676 

Por otro lado, usando Llamacpp directamente desde código, el resultado no mejora:

Notas y Consideraciones
Las ejecuciones Defecto, 1, 2, 3, 4, 5, 6 fueron realizadas sobre una Nvidia Tesla P100 16Gb


Carga del modelo en Ollama
Para poder importar el modelo GGUF a Ollama es necesario crear un archivo Modelfile dentro del cual se debe escribir 
FROM {ruta al modelo gguf en la computadora}
Luego se deberá ejecutar ollama create nombreModelo -f Modelfile
Tener en cuenta que se deberá ejecutar el comando posicionado en el mismo directorio que el Modelfile

