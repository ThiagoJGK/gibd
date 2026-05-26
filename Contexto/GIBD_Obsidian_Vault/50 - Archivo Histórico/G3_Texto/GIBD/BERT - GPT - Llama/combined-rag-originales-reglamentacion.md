---
aliases: [combined-rag-originales-reglamentacion]
tags:
  - grupo/g3_texto
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2024-09-05
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/BERT - GPT - Llama/Notebooks/Deprecated/combined-rag-originales-reglamentacion.ipynb"
tamanio_bytes: 786144
---

# Notebook: combined-rag-originales-reglamentacion.ipynb

Ruta interna: `GIBD/BERT - GPT - Llama/Notebooks/Deprecated/combined-rag-originales-reglamentacion.ipynb`

---

# Dependencias y modelo de embedding 


**[Celda 2 - Código]**
```python
!pip install langchain -q
!pip install langchain_community -q
!pip install pandas -q
!pip install datasets huggingface_hub -q
!pip install PyPDF2 -q
!pip install -U sentence-transformers -q
!pip install -U langchain-ollama -q
!pip install einops -q
!pip install matplotlib -q
```


*Salida:*
```text
[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
cudf 24.8.2 requires cubinlinker, which is not installed.
cudf 24.8.2 requires cupy-cuda11x>=12.0.0, which is not installed.
cudf 24.8.2 requires ptxcompiler, which is not installed.
cuml 24.8.0 requires cupy-cuda11x>=12.0.0, which is not installed.
dask-cudf 24.8.2 requires cupy-cuda11x>=12.0.0, which is not installed.
cudf 24.8.2 requires cuda-python<12.0a0,>=11.7.1, but you have cuda-python 12.6.0 which is incompatible.
distributed 2024.7.1 requires dask==2024.7.1, but you have dask 2024.8.1 which is incompatible.
google-cloud-bigquery 2.34.4 requires packaging<22.0dev,>=14.3, but you have packaging 24.1 which is incompatible.
jupyterlab 4.2.4 requires jupyter-lsp>=2.0.0, but you have jupyter-lsp 1.5.1 which is incompatible.
jupyterlab-lsp 5.1.0 requires jupyter-lsp>=2.0.0, but you have jupyter-lsp 1.5.1 which is incompatible.
libpysal 4.9.2 requires shapely>=2.0.1, but you have shapely 1.8.5.post1 which is incompatible.
momepy 0.7.2 requires shapely>=2, but you have shapely 1.8.5.post1 which is incompatible.
osmnx 1.9.4 requires shapely<2.1,>=2.0, but you have shapely 1.8.5.post1 which is incompatible.
pointpats 2.5.0 requires shapely>=2, but you have shapely 1.8.5.post1 which is incompatible.
rapids-dask-dependency 24.8.0a0 requires dask==2024.7.1, but you have dask 2024.8.1 which is incompatible.
spaghetti 1.7.6 requires shapely>=2.0.1, but you have shapely 1.8.5.post1 which is incompatible.
spopt 0.6.1 requires shapely>=2.0.1, but you have shapely 1.8.5.post1 which is incompatible.
ydata-profiling 4.9.0 requires scipy<1.14,>=1.4.1, but you have scipy 1.14.0 which is incompatible.[0m[31m
[0m
```


**[Celda 3 - Código]**
```python
import ollama
import pandas as pd
from datasets import Dataset
import PyPDF2
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import matplotlib.pyplot as plt


```

### Seleccionar el modelo de embedding 


**[Celda 5 - Código]**
```python
#model = SentenceTransformer('espejelomar/sentece-embeddings-BETO')
#model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
```


**[Celda 6 - Código]**
```python
primary_model = SentenceTransformer('BAAI/bge-m3')
```


*Salida:*
```text
config_sentence_transformers.json:   0%|          | 0.00/123 [00:00<?, ?B/s]config.json:   0%|          | 0.00/687 [00:00<?, ?B/s]pytorch_model.bin:   0%|          | 0.00/2.27G [00:00<?, ?B/s]tokenizer_config.json:   0%|          | 0.00/444 [00:00<?, ?B/s]sentencepiece.bpe.model:   0%|          | 0.00/5.07M [00:00<?, ?B/s]tokenizer.json:   0%|          | 0.00/17.1M [00:00<?, ?B/s]special_tokens_map.json:   0%|          | 0.00/964 [00:00<?, ?B/s]1_Pooling/config.json:   0%|          | 0.00/191 [00:00<?, ?B/s]
```


**[Celda 7 - Código]**
```python
support_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
```


*Salida:*
```text
modules.json:   0%|          | 0.00/349 [00:00<?, ?B/s]config_sentence_transformers.json:   0%|          | 0.00/128 [00:00<?, ?B/s]README.md:   0%|          | 0.00/70.9k [00:00<?, ?B/s]sentence_bert_config.json:   0%|          | 0.00/54.0 [00:00<?, ?B/s]config.json:   0%|          | 0.00/2.03k [00:00<?, ?B/s]configuration_hf_nomic_bert.py:   0%|          | 0.00/1.96k [00:00<?, ?B/s]A new version of the following files was downloaded from https://huggingface.co/nomic-ai/nomic-bert-2048:
- configuration_hf_nomic_bert.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
modeling_hf_nomic_bert.py:   0%|          | 0.00/85.3k [00:00<?, ?B/s]A new version of the following files was downloaded from https://huggingface.co/nomic-ai/nomic-bert-2048:
- modeling_hf_nomic_bert.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
pytorch_model.bin:   0%|          | 0.00/547M [00:00<?, ?B/s]/root/.cache/huggingface/modules/transformers_modules/nomic-ai/nomic-bert-2048/4bb68f63016e88e53e48df904c6ab4e6f718e198/modeling_hf_nomic_bert.py:98: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  state_dict = loader(resolved_archive_file)
tokenizer_config.json:   0%|          | 0.00/1.19k [00:00<?, ?B/s]vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]tokenizer.json:   0%|          | 0.00/711k [00:00<?, ?B/s]special_tokens_map.json:   0%|          | 0.00/125 [00:00<?, ?B/s]1_Pooling/config.json:   0%|          | 0.00/270 [00:00<?, ?B/s]
```


**[Celda 8 - Código]**
```python
import torch 

# Define el dispositivo para mover el modelo a la GPU en caso de que no se realice por defecto
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```


**[Celda 9 - Código]**
```python
primary_model = primary_model.to(device)
```


**[Celda 10 - Código]**
```python
support_model = support_model.to(device)
```

# Creación de embeddings del reglamento


**[Celda 12 - Código]**
```python
#Cambiar por el path de la reglamentación en la computadora a ejecutar
with open("/kaggle/input/reglamentacion-rag/_Reglamento.txt", "r", encoding="utf-8") as file:
    texto  = file.read()

articulos = texto.split("Documento:")
articulos = ["Documento:" + articulo.strip() for articulo in articulos if articulo.strip()]
print(f"Existen {len(articulos)} artículos.")
```


*Salida:*
```text
Existen 674 artículos.
```


**[Celda 13 - Código]**
```python
articles_embedding_primary = []
for i,article in enumerate(articulos):
    article_embedding= primary_model.encode(article)
    articles_embedding_primary.append((i, article_embedding))
```


*Salida:*
```text
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]
```


**[Celda 14 - Código]**
```python
articles_embedding_support = []
for i,article in enumerate(articulos):
    article_embedding = support_model.encode(article)
    articles_embedding_support.append((i, article_embedding))
```


*Salida:*
```text
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]
```

# Recuperación de articulos más similares


**[Celda 16 - Código]**
```python
def devolver_N_articulos_similares(pregunta, model, articles_embedding, articulos,N):
    question_embedding = model.encode(pregunta)
    results = []
    
    for i, article_embedding in articles_embedding:  
        distance = cosine_similarity(article_embedding.reshape((1, -1)), question_embedding.reshape((1, -1)))[0][0]
        results.append((articulos[i], distance))
    
    results = sorted(results, key=lambda x: x[1], reverse=True)[:N]
    
    return results

```


**[Celda 17 - Código]**
```python
def funcionArticuloEncontrado(lista_articulos_proximos, articulo_correcto):
    for i, (article, _) in enumerate(lista_articulos_proximos):
        if articulo_correcto.strip() == article.strip():
            return i + 1 
    return None  
```

# Sección de pruebas y evaluación 

## Importanción de DataSet de preguntas etiquetadas



**[Celda 20 - Código]**
```python
dataset = load_dataset("HuggMaxi/preguntasEtiquetadas")
df_preguntasEtiquetadas = pd.DataFrame(dataset['train'])  
df_preguntasEtiquetadas
```


*Salida:*
```text
Downloading readme:   0%|          | 0.00/382 [00:00<?, ?B/s]Downloading data:   0%|          | 0.00/98.2k [00:00<?, ?B/s]Generating train split:   0%|          | 0/300 [00:00<?, ? examples/s]     idConsulta                                           Consulta Articulo  \
0             1   Cuales son las condiciones para presentarse a...        9   
1             2   Quién debe regir la convocatoria para la desi...        3   
2             3           Cómo se publicita un llamado a concurso?        6   
3             4                 Qué contenido tienen los anuncios?        8   
4             5   Cuales son los requisitos para los aspirantes...        9   
..          ...                                                ...      ...   
295         296   Cual es la responsabilidad principal del Esta...      315   
296         297   Que sucede cuando una carrera no obtiene su a...      299   
297         298   Quienes seran los organos de coordinacion y c...      294   
298         299   Que atribuciones tiene la autonomia academica...      250   
299         300   Cuales son los deberes de los docentes de las...      233   

                                                 Texto  
0    Documento:8\nArticulo: 9\nCapitulo: III - INSC...  
1    Documento:8\nArticulo: 3\nCapitulo: I - CONVOC...  
2    Documento:8\nArticulo: 6\nCapitulo: II - PUBLI...  
3    Documento:8\nArticulo: 8\nCapitulo: II - PUBLI...  
4    Documento:8\nArticulo: 9\nCapitulo: III - INSC...  
..                                                 ...  
295  Documento:14\nArticulo: 315\nCapitulo: TÍTULO ...  
296  Documento:13\nArticulo: 299\nCapitulo: TÍTULO ...  
297  Documento:13\nArticulo: 294\nCapitulo: TÍTULO ...  
298  Documento:13\nArticulo: 250\nCapitulo: TÍTULO ...  
299  Documento:13\nArticulo: 233\nCapitulo: TÍTULO ...  

[300 rows x 4 columns]
```

## Evaluación por consulta individual 


**[Celda 22 - Código]**
```python
idConsulta = 1
texto_objetivo = df_preguntasEtiquetadas[df_preguntasEtiquetadas['idConsulta'] == idConsulta]['Texto'].values[0]
pregunta_consulta = df_preguntasEtiquetadas[df_preguntasEtiquetadas['idConsulta'] == idConsulta]['Consulta'].values[0]

resultadoArticulosEncontrados = devolver_N_articulos_similares(pregunta_consulta, primary_model, articles_embedding_primary, articulos,10)
resultadoPosicion = funcionArticuloEncontrado(resultadoArticulosEncontrados, texto_objetivo)

print(f"CONSULTA: {pregunta_consulta}")
print(f"RESULTADO ESPERADO:")
print("")
print(texto_objetivo)
print("")
print("*********************************************")
print("")
print(f"RESULTADO ENCONTRADO EN LA POSICION: {resultadoPosicion}")
print(f"SIMILARIDAD COSENO: {resultadoArticulosEncontrados[0][1]}")
print("")
print("CONTENIDO ENCONTRADO")
print("*********************************************")
print(resultadoArticulosEncontrados[0][0])
print("*********************************************")


```


*Salida:*
```text
Batches:   0%|          | 0/1 [00:00<?, ?it/s]CONSULTA:  Cuales son las condiciones para presentarse a un concurso?
RESULTADO ESPERADO:

Documento:8
Articulo: 9
Capitulo: III - INSCRIPCIÓN

Condiciones de los aspirantes: Para presentarse a Concurso el aspirante debe reunir los siguientes requisitos: a) Tener título universitario de grado o en su defecto acreditar antecedentes excepcionales que lo suplan, y b) no estar comprendido en las causales de inhabilitación para el desempeño de cargos públicos y de faltas a la ética universitaria que se mencionan en el artículo 18 del presente Reglamento.

*********************************************

RESULTADO ENCONTRADO EN LA POSICION: 1
SIMILARIDAD COSENO: 0.7201571464538574

CONTENIDO ENCONTRADO
*********************************************
Documento:8
Articulo: 9
Capitulo: III - INSCRIPCIÓN

Condiciones de los aspirantes: Para presentarse a Concurso el aspirante debe reunir los siguientes requisitos: a) Tener título universitario de grado o en su defecto acreditar antecedentes excepcionales que lo suplan, y b) no estar comprendido en las causales de inhabilitación para el desempeño de cargos públicos y de faltas a la ética universitaria que se mencionan en el artículo 18 del presente Reglamento.
*********************************************
```


**[Celda 23 - Código]**
```python
for i in range(len(resultadoArticulosEncontrados)):
    print(f"-------------------------------------- {i + 1}")
    print(resultadoArticulosEncontrados[i][0])
    print(resultadoArticulosEncontrados[i][1])
    print()  
```


*Salida:*
```text
-------------------------------------- 1
Documento:8
Articulo: 9
Capitulo: III - INSCRIPCIÓN

Condiciones de los aspirantes: Para presentarse a Concurso el aspirante debe reunir los siguientes requisitos: a) Tener título universitario de grado o en su defecto acreditar antecedentes excepcionales que lo suplan, y b) no estar comprendido en las causales de inhabilitación para el desempeño de cargos públicos y de faltas a la ética universitaria que se mencionan en el artículo 18 del presente Reglamento.
0.72015715

-------------------------------------- 2
Documento:8
Articulo: 18
Capitulo: III - INSCRIPCIÓN

Inscripciones múltiples: El aspirante que se presente a más de un Concurso deberá cumplir en cada uno de ellos con todos los requisitos establecidos en esta ordenanza, sin poder remitirse a los escritos o documentos presentados en los otros. Si los Concursos a los que se presenta integran el mismo llamado de una Unidad Académica, basta que el aspirante cumplimente en uno de ellos la exigencia del articulo 10°, refiriéndose en las solicitudes de inscripción de los otros Concursos, en cual agregó la documentación que menciona el citado articulo y siempre que no la haya retirado de la Unidad Académica.
0.62762904

-------------------------------------- 3
Documento:8
Articulo: 12
Capitulo: III - INSCRIPCIÓN

Solicitud de Inscripción: Las solicitudes de inscripción tendrán el carácter de Declaración Jurada y serán presentadas, bajo recibo en el que constará la fecha y hora de recepción, por los aspirantes o personas autorizadas, en la dependencia habilitada en la respectiva Unidad Académica en un (1) original y cinco (5) copias y, además, en soporte digital, con la información básica siguiente: 1) Apellido y nombres, nacionalidad, estado civil, fecha y lugar de nacimiento, clase y número de documento de identidad, domicilio real y constituir el especial dentro de la ciudad asiento de la Unidad Académica o en un radio de hasta 30 km de la misma. 2) Título universitario, con indicación de la Facultad y Universidad que lo otorgó. Los títulos universitarios deberán presentarse en original y fotocopia simple la que será autenticada por personal autorizado por la Facultad. En su defecto podrán presentarse fotocopias legalizadas ante escribano público o juez de paz. 3) Títulos no universitarios, si los tuviere, con indicación de la Entidad que los otorgó, los que deberán presentarse bajo las mismas formalidades indicadas en el inciso anterior. RESOLUCIÓN Nº289/08. 4) La totalidad de los antecedentes docentes y no docentes e índole de las tareas desarrolladas, indicando la institución, el período de ejercicio y la naturaleza de su designación. 5) Nómina de obras y publicaciones, consignando la editorial o revista y lugar y fecha de publicación. Los miembros del jurado podrán exigir que se presenten copias u originales de las publicaciones y trabajos realizados, las que serán devueltas una vez sustanciado el Concurso. 6) Otros antecedentes relacionados con la especialidad, tales como cursos realizados, conferencias dictadas, etc. 7) La totalidad de los antecedentes que hagan a su actuación en Universidades e Instituciones nacionales, provinciales y privadas, registradas en el país o en el extranjero; cargo que desempeñó o desempeña en la Administración Pública o en la actividad privada, en el país o en el extranjero. Los antecedentes de Instituciones extranjeras deberán presentarse acompañados de su traducción al español. 8) Participación en Congresos o acontecimientos similares, nacionales o internacionales. 9) Distinciones, premios y becas obtenidas. 10) Una síntesis de los aportes originales efectuados en el ejercicio de la especialidad respectiva. 11) Una síntesis de la actuación profesional. 12) Otros cargos y antecedentes que a juicio del aspirante pueden contribuir a una mejor ilustración sobre su competencia en la materia en Concurso. Indicación de la dedicación o dedicaciones a que aspira, o bien que el aspirante proponga una distinta a la consignada en el respectivo llamado.
0.5955132

-------------------------------------- 4
Documento:8
Articulo: 58
Capitulo: X - NORMAS GENERALES

La inscripción al Concurso importará para el aspirante su conformidad con las normas de este Reglamento y las específicas dictadas por cada Unidad Académica.
0.5892316

-------------------------------------- 5
Documento:8
Articulo: 61
Capitulo: X - NORMAS GENERALES

Los llamados a Concurso, se harán dentro de lo posible para una cátedra, área, núcleo, asignaturas que integran los núcleos, disciplina, laboratorio, pero no para los cursos en que estos estuvieran divididos, pudiendo cada Facultad adaptar lo aquí dispuesto a su estructura académica.
0.58815616

-------------------------------------- 6
Documento:8
Articulo: 50
Capitulo: VIII - RESOLUCIÓN DEL CONCURSO

Dentro de los treinta (30) días de haberse expedido el Jurado y sobre la base de su dictamen, de las observaciones formuladas por los participantes mencionados en el Articulo 44 y, de las impugnaciones que hubieran formulado los aspirantes, las cuales quedarán resueltas con asesoramiento legal si correspondiere, el Consejo Directivo, expresando los fundamentos podrá: A) Solicitar al Jurado la ampliación o aclaración del dictamen, en cuyo caso aquel deberá expedirse dentro de los ocho (8) días de tomar conocimiento de la solicitud. B) Proponer al Consejo Superior declarar desierto el Concurso. C) Proponer al Consejo Superior dejar sin efecto el Concurso y llamar uno nuevo. D) Aprobar el Concurso y elevarlo al Consejo Superior con las siguientes posibilidades: 1. Proponer al aspirante ubicado en primer término del dictamen único o alguno de los dictámenes del Jurado. 2. Proponer en forma alternativa para cubrir el/los cargo/s a dos (2) aspirantes ubicados en el primer y segundo término del orden de mérito del dictamen o algunos de los dictámenes del Jurado. Para los llamados a Concurso para cubrir más de un (1) cargo realizado de acuerdo al Artículo 58° de este Reglamento, podrá proponer a los aspirantes respetando el orden de mérito elaborado del Jurado conforme a las pautas del párrafo anterior.
0.5855793

-------------------------------------- 7
Documento:8
Articulo: 6
Capitulo: II - PUBLICIDAD

Difusión: La difusión del llamado a concurso estará a cargo del Rectorado quién publicará dentro de los diez (10) días de aprobado el llamado a concurso, haciendo como mínimo un aviso por un (1) día en un diario de circulación nacional y debiéndose publicar por tres (3) días, en un diario de la localidad sede y subsede de la Unidad Académica respectiva. En aquellas localidades en las que la tirada de los diarios sea de frecuencia semanal bastará con una sola publicación. En todos los casos, la publicidad deberá efectuarse entre los diarios de mayor tirada. El rectorado se hará cargo además, de la difusión del llamado a través de los medios de comunicación con que contara y las Unidades Académicas comunicaran asimismo a sus similares afines dentro del ámbito universitario nacional y a los medios de difusión que consideren convenientes. Las respectivas publicaciones se deberán agregar para constancia al expediente de Concurso.
0.5851388

-------------------------------------- 8
Documento:13
Articulo: 273
Capitulo: TÍTULO IV: DE LA EDUCACIÓN SUPERIOR UNIVERSITARIA. CAPÍTULO 4: De las instituciones universitarias nacionales. Sección I: Creación y bases organizativas

El ingreso a la carrera académica universitaria se hará mediante concurso publico y abierto de antecedentes y oposición, debiéndose asegurar la constitución de jurados integrados por profesores por concurso, o excepcionalmente por personas de idoneidad indiscutible aunque no reúnan esa condición, que garanticen la mayor imparcialidad y el máximo rigor académico. Con carácter excepcional, las universidades e institutos universitarios nacionales podrán contratar, al margen del régimen de concursos y solo por tiempo determinado, a personalidades de reconocido prestigio y méritos académicos sobresalientes para que desarrollen cursos, seminarios o actividades similares. Podrán igualmente prever la designación temporaria de docentes interinos, cuando ello sea imprescindible y mientras se sustancia el correspondiente concurso. Los docentes designados por concurso deberán representar un porcentaje no inferior al setenta por ciento (70%) de las respectivas plantas de cada institución universitaria.
0.58408225

-------------------------------------- 9
Documento:8
Articulo: 52
Capitulo: VIII - RESOLUCIÓN DEL CONCURSO

El Consejo Superior podrá solicitar aclaraciones sobre la o las propuestas del Consejo Directivo, en cuyo caso este deberá expedirse dentro de los quince (15) días de tomar conocimiento de la solicitud. En un plazo no mayor de veinticuatro (24) días de recibida la propuesta y/o las aclaraciones de dicho órgano, podrá aceptar la propuesta del Consejo Directivo, o rechazarla. Si la propuesta fuera rechazada el Concurso quedará sin efecto. Dicha resolución se notificará a los interesados y será difundida a través de las carteleras murales y de la página electrónica de la Unidad Académica correspondiente. En los Concursos en los que no se formulen propuestas para la totalidad de los cargos concursados, los que no se provean serán declarados desiertos.
0.5789505

-------------------------------------- 10
Documento:9
Articulo: 73
Capitulo: Reglamento de Concurso Docentes de Antecedentes

Las solicitudes de inscripción se formalizarán por nota dirigida al Decano, tendrán carácter de declaración jurada y serán presentadas por los aspirantes o personas autorizadas por éstos, en sede o subsede de la Facultad que corresponda, en tres (3) ejemplares, bajo recibo en el que constará la fecha de recepción, con la información básica siguiente: 4.1 Apellido y nombres, nacionalidad, estado civil, fecha y lugar de nacimiento, domicilio real, clase y número de documento. 4.2 Títulos con la indicación de la institución que los otorgó. 4.3 La totalidad de los antecedentes docentes (actuación en universidades y Facultades nacionales, provinciales y/o privados registrados en el país o en el extranjero) y los no docentes (cargos desempeñados en la administración pública o en la actividad privada) e índole de las tareas desarrolladas, indicando la institución, el período de ejercicio y la naturaleza de la designación. 4.4 Nómina de obras y publicaciones realizadas por el aspirante, consignando la editorial o revista y el lugar y fecha de publicación. Se podrá exigir que se presenten copias de las publicaciones y los trabajos mencionados las que serán devueltas una vez efectuada la evaluación de los antecedentes. 4.5 Otros antecedentes relacionados con la especialidad tales como: cursos realizados, cursos y conferencias dictados, etc. 4.6 Participación en Congresos y/o Jornadas nacionales y/o internacionales. 4.7 Distinciones, premios y becas obtenidos.
0.57846826
```

## Creación de Dataset de resultados 


**[Celda 25 - Código]**
```python
batch_size = 10  # Tamaño del lote
total_consultas = len(df_preguntasEtiquetadas)
resultados = []  # Lista para almacenar los resultados antes de crear el DataFrame final

for start in range(0, total_consultas, batch_size):
    # Definir el rango de consultas en el lote
    end = min(start + batch_size, total_consultas)
    batch_df = df_preguntasEtiquetadas.iloc[start:end]
    
    for _, row in batch_df.iterrows():
        idCon = row['idConsulta']
        pregunta_consulta = row['Consulta']
        texto_consulta = row['Texto']
        
        # Obtener los N artículos más similares (N=10 en este caso)
        resultadoPrimary = devolver_N_articulos_similares(pregunta_consulta, primary_model, articles_embedding_primary, articulos, 10)
        resultadoSupport = devolver_N_articulos_similares(pregunta_consulta, support_model, articles_embedding_support, articulos, 10)
        
        # Encontrar la posición del artículo correcto
        posPrimary = funcionArticuloEncontrado(resultadoPrimary, texto_consulta)
        posSupport = funcionArticuloEncontrado(resultadoSupport, texto_consulta)
        
        # Caso 1: Ambos resultados tienen posiciones válidas
        if posPrimary is not None and posSupport is not None:
            if 0 <= posPrimary < len(resultadoPrimary) and 0 <= posSupport < len(resultadoSupport):
                if len(resultadoPrimary[posPrimary][0]) > 1 and resultadoPrimary[posPrimary][0][1] is not None and \
                   len(resultadoSupport[posSupport][0]) > 1 and resultadoSupport[posSupport][0][1] is not None:
                    # if resultadoPrimary[posPrimary][1] > resultadoSupport[posSupport][1]:
                    if posPrimary < posSupport:
                        similaridad = resultadoPrimary[posPrimary][1]
                        modelo = 'BAAI/bge-m3'
                        posicion = posPrimary
                    else:
                        similaridad = resultadoSupport[posSupport][1]
                        modelo = 'nomic-ai/nomic-embed-text-v1'
                        posicion = posSupport
                else:
                    similaridad = -1
                    modelo = 'N/A'
                    posicion = -1

        # Caso 2: Solo uno es None, usa el otro
        elif posPrimary is not None and 0 <= posPrimary < len(resultadoPrimary):
            similaridad = resultadoPrimary[posPrimary][1] if len(resultadoPrimary[posPrimary][0]) > 1 and resultadoPrimary[posPrimary][0][1] is not None else -1
            modelo = 'BAAI/bge-m3'
            posicion = posPrimary

        elif posSupport is not None and 0 <= posSupport < len(resultadoSupport):
            similaridad = resultadoSupport[posSupport][1] if len(resultadoSupport[posSupport][0]) > 1 and resultadoSupport[posSupport][0][1] is not None else -1
            modelo = 'nomic-ai/nomic-embed-text-v1'
            posicion = posSupport

        # Caso 3: Ambos son None
        else:
            similaridad = 0
            modelo = 'N/A'
            posicion = -1

        # Agregar el resultado a la lista de resultados
        resultados.append({
            'idConsulta': idCon,
            'Consulta': pregunta_consulta,
            'Posicion': posicion,
            'Similaridad': similaridad,
            'Modelo': modelo
        })

# Convertir la lista de resultados en un DataFrame
resultados_df = pd.DataFrame(resultados)

# Mostrar el DataFrame
print(resultados_df)

```


*Salida:*
```text
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]     idConsulta                                           Consulta  Posicion  \
0             1   Cuales son las condiciones para presentarse a...         1   
1             2   Quién debe regir la convocatoria para la desi...         1   
2             3           Cómo se publicita un llamado a concurso?         1   
3             4                 Qué contenido tienen los anuncios?         1   
4             5   Cuales son los requisitos para los aspirantes...         1   
..          ...                                                ...       ...   
295         296   Cual es la responsabilidad principal del Esta...         1   
296         297   Que sucede cuando una carrera no obtiene su a...         1   
297         298   Quienes seran los organos de coordinacion y c...         1   
298         299   Que atribuciones tiene la autonomia academica...         1   
299         300   Cuales son los deberes de los docentes de las...         1   

     Similaridad                        Modelo  
0       0.627629                   BAAI/bge-m3  
1       0.707805  nomic-ai/nomic-embed-text-v1  
2       0.660449  nomic-ai/nomic-embed-text-v1  
3       0.443331                   BAAI/bge-m3  
4       0.627449  nomic-ai/nomic-embed-text-v1  
..           ...                           ...  
295     0.607184                   BAAI/bge-m3  
296     0.649355  nomic-ai/nomic-embed-text-v1  
297     0.591162                   BAAI/bge-m3  
298     0.666521                   BAAI/bge-m3  
299     0.677320  nomic-ai/nomic-embed-text-v1  

[300 rows x 5 columns]
```


**[Celda 26 - Código]**
```python
#Opcional guardar Dataset - Cambiar para guardar los resultados según preferencias
csv_path = r'/kaggle/working/resultados_combined.csv'
resultados_df.to_csv(csv_path, index=False)
```


**[Celda 27 - Código]**
```python
print(resultados_df)
```


*Salida:*
```text
idConsulta                                           Consulta  Posicion  \
0             1   Cuales son las condiciones para presentarse a...         1   
1             2   Quién debe regir la convocatoria para la desi...         1   
2             3           Cómo se publicita un llamado a concurso?         1   
3             4                 Qué contenido tienen los anuncios?         1   
4             5   Cuales son los requisitos para los aspirantes...         1   
..          ...                                                ...       ...   
295         296   Cual es la responsabilidad principal del Esta...         1   
296         297   Que sucede cuando una carrera no obtiene su a...         1   
297         298   Quienes seran los organos de coordinacion y c...         1   
298         299   Que atribuciones tiene la autonomia academica...         1   
299         300   Cuales son los deberes de los docentes de las...         1   

     Similaridad                        Modelo  
0       0.627629                   BAAI/bge-m3  
1       0.707805  nomic-ai/nomic-embed-text-v1  
2       0.660449  nomic-ai/nomic-embed-text-v1  
3       0.443331                   BAAI/bge-m3  
4       0.627449  nomic-ai/nomic-embed-text-v1  
..           ...                           ...  
295     0.607184                   BAAI/bge-m3  
296     0.649355  nomic-ai/nomic-embed-text-v1  
297     0.591162                   BAAI/bge-m3  
298     0.666521                   BAAI/bge-m3  
299     0.677320  nomic-ai/nomic-embed-text-v1  

[300 rows x 5 columns]
```

## Vizualización de resultados


**[Celda 29 - Código]**
```python
# Función para categorizar las posiciones
def categorizar_posicion(pos):
    if pos == -1:
        return 'No aparece'
    elif 1 <= pos <= 3:
        return '1-3'
    elif 4 <= pos <= 6:
        return '4-6'
    elif 7 <= pos <= 10:
        return '7-10'
    else:
        return 'No aparece'

# Aplicar la categorización
resultados_df['Categoria'] = resultados_df['Posicion'].apply(categorizar_posicion)

# Calcular los porcentajes y conteos
conteo = resultados_df['Categoria'].value_counts()
porcentajes = (conteo / len(resultados_df)) * 100

# Ordenar las categorías
orden_categorias = ['1-3', '4-6', '7-10', 'No aparece']
conteo = conteo.reindex(orden_categorias)
porcentajes = porcentajes.reindex(orden_categorias)

# Crear el gráfico
plt.figure(figsize=(12, 6))
colores = ['#98FB98', '#FFFF99', '#FF6347', '#D3D3D3']  # Verde, Amarillo pastel, Rojo oscuro, Gris (pasteles)
barras = plt.bar(porcentajes.index, porcentajes.values, color=colores)

plt.title('Distribución de Posiciones para bge-m3 + nomic-embed-text-v1 ', fontsize=16)
plt.xlabel('Categorías de Posición en lista de articulos similares', fontsize=12)
plt.ylabel('Porcentaje', fontsize=12)
plt.ylim(0, 100)

# Añadir etiquetas de porcentaje encima de cada barra
for barra in barras:
    altura = barra.get_height()
    plt.text(barra.get_x() + barra.get_width()/2, altura,
             f'{altura:.1f}%\n({int(altura*len(resultados_df)/100)})',
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# Imprimir resultados detallados
print("Resultados detallados:")
for categoria, porcentaje in porcentajes.items():
    num_preguntas = int(porcentaje * len(resultados_df) / 100)
    print(f"{categoria}: {porcentaje:.1f}% ({num_preguntas} preguntas)")

```


*Salida:*
```text
<Figure size 1200x600 with 1 Axes>Resultados detallados:
1-3: 97.0% (291 preguntas)
4-6: 1.7% (5 preguntas)
7-10: 0.3% (1 preguntas)
No aparece: 1.0% (3 preguntas)
```


**[Celda 30 - Código]**
```python
def mostrar_preguntas_por_posicion(df):
    # Resumen de cantidad por posición
    print("Resumen de cantidad por posición:")
    for posicion in range(1, 11):
        cantidad = df[df['Posicion'] == posicion].shape[0]
        print(f"Posición {posicion}: {cantidad}")
    no_encontradas = df[df['Posicion'] == -1].shape[0]
    print(f"No encontradas: {no_encontradas}")
    
    print("\nPreguntas detalladas por posición:")
    for posicion in range(1, 11):
        preguntas = df[df['Posicion'] == posicion]
        if not preguntas.empty:
            print(f"\nPosición {posicion}:")
            for _, row in preguntas.iterrows():
                print(f"  idConsulta {row['idConsulta']}: {row['Consulta']}")
    
    # Preguntas no encontradas (Posicion == -1)
    no_encontradas = df[df['Posicion'] == -1]
    if not no_encontradas.empty:
        print("\nPreguntas no encontradas:")
        for _, row in no_encontradas.iterrows():
            print(f"  idConsulta {row['idConsulta']}: {row['Consulta']}")

# Ejecutar la función
mostrar_preguntas_por_posicion(resultados_df)
```


**[Celda 31 - Código]**
```python

```
