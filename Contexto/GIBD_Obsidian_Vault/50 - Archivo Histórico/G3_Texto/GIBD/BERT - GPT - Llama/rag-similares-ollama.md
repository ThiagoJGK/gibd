---
aliases: [rag-similares-ollama]
tags:
  - grupo/g3_texto
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2024-09-14
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/BERT - GPT - Llama/Notebooks/rag-similares-ollama.ipynb"
tamanio_bytes: 429039
---

# Notebook: rag-similares-ollama.ipynb

Ruta interna: `GIBD/BERT - GPT - Llama/Notebooks/rag-similares-ollama.ipynb`

---

# Dependencias y modelo de embedding


**[Celda 2 - Código]**
```python
!pip install Ollama -q
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
from langchain_community.llms import Ollama
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
model = SentenceTransformer('BAAI/bge-m3')
```


*Salida:*
```text
modules.json:   0%|          | 0.00/349 [00:00<?, ?B/s]config_sentence_transformers.json:   0%|          | 0.00/123 [00:00<?, ?B/s]README.md:   0%|          | 0.00/15.8k [00:00<?, ?B/s]sentence_bert_config.json:   0%|          | 0.00/54.0 [00:00<?, ?B/s]config.json:   0%|          | 0.00/687 [00:00<?, ?B/s]pytorch_model.bin:   0%|          | 0.00/2.27G [00:00<?, ?B/s]tokenizer_config.json:   0%|          | 0.00/444 [00:00<?, ?B/s]sentencepiece.bpe.model:   0%|          | 0.00/5.07M [00:00<?, ?B/s]tokenizer.json:   0%|          | 0.00/17.1M [00:00<?, ?B/s]special_tokens_map.json:   0%|          | 0.00/964 [00:00<?, ?B/s]1_Pooling/config.json:   0%|          | 0.00/191 [00:00<?, ?B/s]
```

# Creación de embeddings del reglamento


**[Celda 7 - Código]**
```python
#Cambiar la ruta para el archivo original de la reglamentación
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


**[Celda 8 - Código]**
```python
import torch

# Para mover el modelo a la GPU en el caso de que no lo haga por default
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```


**[Celda 9 - Código]**
```python
articles_embedding = []
for i,article in enumerate(articulos):
    article_embedding = model.encode(article)
    articles_embedding.append((i, article_embedding))
```


*Salida:*
```text
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]
```

# Creación de preguntas similares


**[Celda 11 - Código]**
```python
def generate_questions_3(pregunta_original):
    formatted_prompt = f"""
    Eres un experto en generar preguntas similares en español, específicamente relacionadas con el reglamento de una facultad universitaria. Tu tarea es:

    1. Tomar esta pregunta original extraída del reglamento de una universidad: "{pregunta_original}"
    2. Generar EXACTAMENTE 2 preguntas similares enfatizando sinónimos para los sustantivos relevantes, manteniendo el contexto académico y administrativo de una universidad.
    3. Presentar el resultado en el siguiente formato estricto, sin espacios adicionales entre líneas:

    {pregunta_original}
    [Primera pregunta similar]
    [Segunda pregunta similar]

    IMPORTANTE:
    - No incluyas NINGÚN texto adicional, comentario o aclaración.
    - Asegúrate de que haya EXACTAMENTE 3 preguntas en total (la original más 2 similares).
    - Mantén el significado y la intención de la pregunta original en todas las variaciones.
    - Todas las preguntas deben estar relacionadas con el contexto de reglamentos, procedimientos o políticas universitarias.
    - No dejes espacios adicionales entre las preguntas.

    Comienza ahora:
    """

    response = ollama.chat(model='llama3.1',
                           messages=[{'role': 'user', 'content': formatted_prompt}],
                           options={'temperature': 0})

    return response['message']['content']

```


**[Celda 12 - Código]**
```python
def generate_questions_5(pregunta_original):
    formatted_prompt = f"""
    Eres un experto en generar preguntas similares en español, específicamente relacionadas con el reglamento de una facultad universitaria. Tu tarea es:

    1. Tomar esta pregunta original extraída del reglamento de una universidad: "{pregunta_original}"
    2. Generar EXACTAMENTE 4 preguntas similares enfatizando sinónimos para los sustantivos relevantes, manteniendo el contexto académico y administrativo de una universidad.
    3. Presentar el resultado en el siguiente formato estricto, sin espacios adicionales entre líneas:

    {pregunta_original}
    [Primera pregunta similar]
    [Segunda pregunta similar]
    [Tercera pregunta similar]
    [Cuarta pregunta similar]

    IMPORTANTE:
    - No incluyas NINGÚN texto adicional, comentario o aclaración.
    - Asegúrate de que haya EXACTAMENTE 5 preguntas en total (la original más 4 similares).
    - Mantén el significado y la intención de la pregunta original en todas las variaciones.
    - Todas las preguntas deben estar relacionadas con el contexto de reglamentos, procedimientos o políticas universitarias.
    - No dejes espacios adicionales entre las preguntas.

    Comienza ahora:
    """

    response = ollama.chat(model='llama3.1',
                           messages=[{'role': 'user', 'content': formatted_prompt}],
                           options={'temperature': 0})

    return response['message']['content']

```


**[Celda 13 - Código]**
```python
def generate_questions(pregunta_original):
    formatted_prompt = f"""
    Eres un experto en generar preguntas similares en español, específicamente relacionadas con el reglamento de una facultad universitaria. Tu tarea es:

    1. Tomar esta pregunta original extraída del reglamento de una universidad: "{pregunta_original}"
    2. Generar EXACTAMENTE 9 preguntas similares enfatizando sinónimos para los sustantivos relevantes, manteniendo el contexto académico y administrativo de una universidad.
    3. Presentar el resultado en el siguiente formato estricto, sin espacios adicionales entre líneas:

    {pregunta_original}
    [Primera pregunta similar]
    [Segunda pregunta similar]
    [Tercera pregunta similar]
    [Cuarta pregunta similar]
    [Quinta pregunta similar]
    [Sexta pregunta similar]
    [Séptima pregunta similar]
    [Octava pregunta similar]
    [Novena pregunta similar]

    IMPORTANTE:
    - No incluyas NINGÚN texto adicional, comentario o aclaración.
    - Asegúrate de que haya EXACTAMENTE 10 preguntas en total (la original más 9 similares).
    - Mantén el significado y la intención de la pregunta original en todas las variaciones.
    - Todas las preguntas deben estar relacionadas con el contexto de reglamentos, procedimientos o políticas universitarias.
    - No dejes espacios adicionales entre las preguntas.

    Comienza ahora:
    """

    response = ollama.chat(model='llama3.1',
                           messages=[{'role': 'user', 'content': formatted_prompt}],
                           options={'temperature': 0})

    return response['message']['content']
```


**[Celda 14 - Código]**
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


**[Celda 15 - Código]**
```python
idConsulta = 1
prueba_pregunta_consulta = df_preguntasEtiquetadas[df_preguntasEtiquetadas['idConsulta'] == idConsulta]['Consulta'].values[0]
prueba_preguntas_similares = generate_questions_5(prueba_pregunta_consulta)
print(prueba_preguntas_similares)
```


*Salida:*
```text
time=2024-08-30T04:40:18.460Z level=INFO source=sched.go:715 msg="new model will fit in available VRAM in single GPU, loading" model=/root/.ollama/models/blobs/sha256-8eeb52dfb3bb9aefdf9d1ef24b3bdbcfbe82238798c4b918278320b6fcef18fe gpu=GPU-fb2b6441-1e8f-89c7-59c1-e5c5833101f9 parallel=4 available=15720382464 required="6.2 GiB"
time=2024-08-30T04:40:18.461Z level=INFO source=memory.go:309 msg="offload to cuda" layers.requested=-1 layers.model=33 layers.offload=33 layers.split="" memory.available="[14.6 GiB]" memory.required.full="6.2 GiB" memory.required.partial="6.2 GiB" memory.required.kv="1.0 GiB" memory.required.allocations="[6.2 GiB]" memory.weights.total="4.7 GiB" memory.weights.repeating="4.3 GiB" memory.weights.nonrepeating="411.0 MiB" memory.graph.full="560.0 MiB" memory.graph.partial="677.5 MiB"
time=2024-08-30T04:40:18.463Z level=INFO source=server.go:391 msg="starting llama server" cmd="/tmp/ollama3583115271/runners/cuda_v12/ollama_llama_server --model /root/.ollama/models/blobs/sha256-8eeb52dfb3bb9aefdf9d1ef24b3bdbcfbe82238798c4b918278320b6fcef18fe --ctx-size 8192 --batch-size 512 --embedding --log-disable --n-gpu-layers 33 --parallel 4 --port 32989"
time=2024-08-30T04:40:18.463Z level=INFO source=sched.go:450 msg="loaded runners" count=1
time=2024-08-30T04:40:18.463Z level=INFO source=server.go:591 msg="waiting for llama runner to start responding"
time=2024-08-30T04:40:18.464Z level=INFO source=server.go:625 msg="waiting for server to become available" status="llm server error"
llama_model_loader: loaded meta data with 29 key-value pairs and 292 tensors from /root/.ollama/models/blobs/sha256-8eeb52dfb3bb9aefdf9d1ef24b3bdbcfbe82238798c4b918278320b6fcef18fe (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Meta Llama 3.1 8B Instruct
llama_model_loader: - kv   3:                           general.finetune str              = Instruct
llama_model_loader: - kv   4:                           general.basename str              = Meta-Llama-3.1
llama_model_loader: - kv   5:                         general.size_label str              = 8B
llama_model_loader: - kv   6:                            general.license str              = llama3.1
llama_model_loader: - kv   7:                               general.tags arr[str,6]       = ["facebook", "meta", "pytorch", "llam...
llama_model_loader: - kv   8:                          general.languages arr[str,8]       = ["en", "de", "fr", "it", "pt", "hi", ...
llama_model_loader: - kv   9:                          llama.block_count u32              = 32
llama_model_loader: - kv  10:                       llama.context_length u32              = 131072
llama_model_loader: - kv  11:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv  12:                  llama.feed_forward_length u32              = 14336
llama_model_loader: - kv  13:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv  14:              llama.attention.head_count_kv u32              = 8
llama_model_loader: - kv  15:                       llama.rope.freq_base f32              = 500000.000000
llama_model_loader: - kv  16:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  17:                          general.file_type u32              = 2
llama_model_loader: - kv  18:                           llama.vocab_size u32              = 128256
llama_model_loader: - kv  19:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv  20:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  21:                         tokenizer.ggml.pre str              = llama-bpe
llama_model_loader: - kv  22:                      tokenizer.ggml.tokens arr[str,128256]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  23:                  tokenizer.ggml.token_type arr[i32,128256]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
INFO [main] build info | build=1 commit="1e6f655" tid="137702258139136" timestamp=1724992818
INFO [main] system info | n_threads=2 n_threads_batch=-1 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 0 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 0 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 0 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | " tid="137702258139136" timestamp=1724992818 total_threads=4
INFO [main] HTTP server listening | hostname="127.0.0.1" n_threads_http="6" port="32989" tid="137702258139136" timestamp=1724992818
time=2024-08-30T04:40:18.715Z level=INFO source=server.go:625 msg="waiting for server to become available" status="llm server loading model"
llama_model_loader: - kv  24:                      tokenizer.ggml.merges arr[str,280147]  = ["Ġ Ġ", "Ġ ĠĠĠ", "ĠĠ ĠĠ", "...
llama_model_loader: - kv  25:                tokenizer.ggml.bos_token_id u32              = 128000
llama_model_loader: - kv  26:                tokenizer.ggml.eos_token_id u32              = 128009
llama_model_loader: - kv  27:                    tokenizer.chat_template str              = {{- bos_token }}\n{%- if custom_tools ...
llama_model_loader: - kv  28:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:   66 tensors
llama_model_loader: - type q4_0:  225 tensors
llama_model_loader: - type q6_K:    1 tensors
llm_load_vocab: special tokens cache size = 256
llm_load_vocab: token to piece cache size = 0.7999 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 128256
llm_load_print_meta: n_merges         = 280147
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 131072
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 4
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 14336
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 500000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 131072
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 8B
llm_load_print_meta: model ftype      = Q4_0
llm_load_print_meta: model params     = 8.03 B
llm_load_print_meta: model size       = 4.33 GiB (4.64 BPW) 
llm_load_print_meta: general.name     = Meta Llama 3.1 8B Instruct
llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
llm_load_print_meta: EOS token        = 128009 '<|eot_id|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'
llm_load_print_meta: max token length = 256
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: Tesla T4, compute capability 7.5, VMM: yes
llm_load_tensors: ggml ctx size =    0.27 MiB
llm_load_tensors: offloading 32 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 33/33 layers to GPU
llm_load_tensors:        CPU buffer size =   281.81 MiB
llm_load_tensors:      CUDA0 buffer size =  4156.00 MiB
llama_new_context_with_model: n_ctx      = 8192
llama_new_context_with_model: n_batch    = 512
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: freq_base  = 500000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  1024.00 MiB
llama_new_context_with_model: KV self size  = 1024.00 MiB, K (f16):  512.00 MiB, V (f16):  512.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     2.02 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   560.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    24.01 MiB
llama_new_context_with_model: graph nodes  = 1030
llama_new_context_with_model: graph splits = 2
INFO [main] model loaded | tid="137702258139136" timestamp=1724992821
time=2024-08-30T04:40:21.225Z level=INFO source=server.go:630 msg="llama runner started in 2.76 seconds"
[GIN] 2024/08/30 - 04:40:23 | 200 |  5.716191583s |       127.0.0.1 | POST     "/api/chat"
Cuales son las condiciones para presentarse a un concurso?
¿Qué requisitos hay que cumplir para participar en una competencia académica?
¿Cuáles son los criterios para acceder a un proceso de selección universitario?
¿Qué normas deben seguirse para postularse a un certamen estudiantil?
¿Cómo se determinan las condiciones para inscribirse en un concurso escolar?
```


**[Celda 16 - Código]**
```python
from tqdm import tqdm


def generar_preguntas_similares_por_lotes(df, lote_tamaño=10):
    resultados = []

    # Proceso las consultas en lotes usando tqdm para ver el progreso
    for i in tqdm(range(0, len(df), lote_tamaño)):
        # Seleccionar un lote de consultas
        lote = df.iloc[i:i + lote_tamaño]

        # Iterar sobre cada consulta en el lote
        for idx, fila in lote.iterrows():
            idConsulta = fila['idConsulta']
            pregunta_consulta = fila['Consulta']

            # Generar las preguntas similares para cada consulta
            preguntas_similares = generate_questions_5(pregunta_consulta)

            # Guardar el resultado
            resultados.append({
                'idConsulta': idConsulta,
                'preguntas_similares': preguntas_similares
            })

    # Crear un nuevo DataFrame con los resultados
    df_resultados = pd.DataFrame(resultados)
    return df_resultados

# Aplicar la función al DataFrame original
df_resultados = generar_preguntas_similares_por_lotes(df_preguntasEtiquetadas, lote_tamaño=10)

# Ruta para guardar el archivo CSV. Cambiar para el caso de una ruta local
csv_path = r'/kaggle/working/ds_preguntasSimilares_CINCO.csv'

# Guardar el DataFrame resultante en un archivo CSV
df_resultados.to_csv(csv_path, index=False)

print(f"El archivo se ha guardado en {csv_path}")

```


*Salida:*
```text
0%|          | 0/30 [00:00<?, ?it/s][GIN] 2024/08/30 - 04:40:26 | 200 |  2.273839974s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:40:29 | 200 |  3.310477126s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:40:31 | 200 |  1.653656529s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:40:33 | 200 |  1.948976462s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:40:36 | 200 |  2.825626974s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:40:38 | 200 |  2.144491903s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:40:40 | 200 |  2.170744769s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:40:42 | 200 |  2.267149495s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:40:45 | 200 |  2.706927148s |       127.0.0.1 | POST     "/api/chat"
  3%|▎         | 1/30 [00:24<11:41, 24.17s/it][GIN] 2024/08/30 - 04:40:48 | 200 |  2.848703484s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:40:50 | 200 |  2.278697554s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:40:53 | 200 |  3.025475121s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:40:56 | 200 |  3.111593623s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:40:59 | 200 |  2.997142156s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:41:02 | 200 |  2.912846577s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:41:05 | 200 |   2.33897316s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:41:07 | 200 |  2.574907525s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:41:11 | 200 |   3.82531591s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:41:14 | 200 |  3.055849722s |       127.0.0.1 | POST     "/api/chat"
  7%|▋         | 2/30 [00:53<12:47, 27.43s/it][GIN] 2024/08/30 - 04:41:18 | 200 |  3.561638727s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:41:21 | 200 |  2.867664451s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:41:24 | 200 |  3.706485517s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:41:27 | 200 |  2.554921897s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:41:30 | 200 |   3.48496622s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:41:34 | 200 |  3.240295968s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:41:36 | 200 |   2.75806281s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:41:39 | 200 |  3.120394729s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:41:42 | 200 |  3.018895242s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:41:45 | 200 |  2.480262764s |       127.0.0.1 | POST     "/api/chat"
 10%|█         | 3/30 [01:24<12:54, 28.67s/it][GIN] 2024/08/30 - 04:41:48 | 200 |  2.894894659s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:41:51 | 200 |  2.749518315s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:41:53 | 200 |  2.848090699s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:41:56 | 200 |  2.697135231s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:41:58 | 200 |  1.880122871s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:42:00 | 200 |  1.883425847s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:42:02 | 200 |  2.015580492s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:42:04 | 200 |  2.348654504s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:42:07 | 200 |   2.58516372s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:42:09 | 200 |  2.265368553s |       127.0.0.1 | POST     "/api/chat"
 13%|█▎        | 4/30 [01:48<11:37, 26.84s/it][GIN] 2024/08/30 - 04:42:12 | 200 |  2.743931972s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:42:13 | 200 |  1.472194564s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:42:16 | 200 |  3.034800039s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:42:19 | 200 |  2.985343834s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:42:22 | 200 |  2.326224837s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:42:24 | 200 |  1.868474924s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:42:26 | 200 |  2.286467307s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:42:28 | 200 |   2.35566644s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:42:30 | 200 |  1.977728607s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:42:32 | 200 |  1.888486884s |       127.0.0.1 | POST     "/api/chat"
 17%|█▋        | 5/30 [02:11<10:41, 25.64s/it][GIN] 2024/08/30 - 04:42:35 | 200 |   3.29286932s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:42:37 | 200 |  2.057375252s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:42:39 | 200 |  1.807574815s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:42:42 | 200 |  2.538577336s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:42:44 | 200 |  2.104050263s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:42:46 | 200 |  1.806931923s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:42:49 | 200 |  3.294891112s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:42:52 | 200 |   2.64885099s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:42:55 | 200 |  2.918209358s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:42:57 | 200 |  2.253328461s |       127.0.0.1 | POST     "/api/chat"
 20%|██        | 6/30 [02:36<10:06, 25.27s/it][GIN] 2024/08/30 - 04:43:00 | 200 |  3.105294453s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:43:02 | 200 |  1.809422075s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:43:04 | 200 |   2.01211029s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:43:06 | 200 |  2.567645472s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:43:09 | 200 |  2.387617296s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:43:11 | 200 |  2.307211173s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:43:14 | 200 |  2.689213595s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:43:16 | 200 |  2.455409079s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:43:19 | 200 |  2.606042995s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:43:22 | 200 |  3.492939397s |       127.0.0.1 | POST     "/api/chat"
 23%|██▎       | 7/30 [03:01<09:39, 25.19s/it][GIN] 2024/08/30 - 04:43:25 | 200 |  2.664132184s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:43:27 | 200 |   2.37568811s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:43:29 | 200 |  2.064691043s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:43:32 | 200 |  2.920593562s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:43:34 | 200 |  2.049267302s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:43:36 | 200 |  1.670054628s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:43:38 | 200 |  1.941560453s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:43:42 | 200 |  3.763953417s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:43:44 | 200 |  2.258837375s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:43:46 | 200 |  2.238267048s |       127.0.0.1 | POST     "/api/chat"
 27%|██▋       | 8/30 [03:25<09:06, 24.83s/it][GIN] 2024/08/30 - 04:43:49 | 200 |  2.756115288s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:43:51 | 200 |   1.89255609s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:43:53 | 200 |  1.857465349s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:43:55 | 200 |  2.529860185s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:43:58 | 200 |  2.498913488s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:44:00 | 200 |  2.381133448s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:44:03 | 200 |  2.606276481s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:44:06 | 200 |  2.751868309s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:44:07 | 200 |  1.955393657s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:44:10 | 200 |  2.372059982s |       127.0.0.1 | POST     "/api/chat"
 30%|███       | 9/30 [03:48<08:32, 24.42s/it][GIN] 2024/08/30 - 04:44:13 | 200 |  2.656473124s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:44:15 | 200 |  2.857178932s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:44:18 | 200 |  2.795536161s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:44:21 | 200 |  2.437793008s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:44:23 | 200 |   1.99546477s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:44:25 | 200 |   2.41675597s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:44:27 | 200 |  2.113001567s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:44:30 | 200 |  2.400611502s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:44:33 | 200 |  3.063978686s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:44:35 | 200 |  1.992921627s |       127.0.0.1 | POST     "/api/chat"
 33%|███▎      | 10/30 [04:13<08:10, 24.50s/it][GIN] 2024/08/30 - 04:44:37 | 200 |  2.582010023s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:44:41 | 200 |  3.637610548s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:44:44 | 200 |  3.005370348s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:44:47 | 200 |  3.531993299s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:44:50 | 200 |  2.834304015s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:44:53 | 200 |  3.000233385s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:44:57 | 200 |  3.600032641s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:45:01 | 200 |  4.056533541s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:45:05 | 200 |  3.989346267s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:45:12 | 200 |  7.447870282s |       127.0.0.1 | POST     "/api/chat"
 37%|███▋      | 11/30 [04:51<09:01, 28.51s/it][GIN] 2024/08/30 - 04:45:15 | 200 |  2.464374859s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:45:20 | 200 |  5.000665718s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:45:23 | 200 |  3.649001589s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:45:27 | 200 |   3.19199281s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:45:31 | 200 |   4.66223098s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:45:34 | 200 |  3.071278265s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:45:40 | 200 |  5.443723029s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:45:42 | 200 |  1.733507656s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:45:46 | 200 |    4.5689404s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:45:48 | 200 |  2.036455208s |       127.0.0.1 | POST     "/api/chat"
 40%|████      | 12/30 [05:27<09:14, 30.78s/it][GIN] 2024/08/30 - 04:45:51 | 200 |  2.602889117s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:45:54 | 200 |  3.081999882s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:45:56 | 200 |  2.561254827s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:46:00 | 200 |  3.390659435s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:46:02 | 200 |  2.540982753s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:46:05 | 200 |  2.676155574s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:46:08 | 200 |  2.801532273s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:46:13 | 200 |  4.698826721s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:46:15 | 200 |  2.388450035s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:46:17 | 200 |   2.33629781s |       127.0.0.1 | POST     "/api/chat"
 43%|████▎     | 13/30 [05:57<08:40, 30.64s/it][GIN] 2024/08/30 - 04:46:21 | 200 |  3.803037839s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:46:24 | 200 |  2.840424153s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:46:28 | 200 |  3.792218124s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:46:31 | 200 |  3.476564831s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:46:34 | 200 |   2.86849185s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:46:40 | 200 |  6.000494044s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:46:42 | 200 |  2.048644981s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:46:44 | 200 |  2.082069454s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:46:47 | 200 |  2.696468985s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:46:49 | 200 |   2.19470967s |       127.0.0.1 | POST     "/api/chat"
 47%|████▋     | 14/30 [06:27<08:08, 30.50s/it][GIN] 2024/08/30 - 04:46:51 | 200 |  2.174270989s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:46:55 | 200 |  3.683263198s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:46:57 | 200 |  2.318783212s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:47:00 | 200 |  2.451831058s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:47:03 | 200 |  2.989929342s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:47:06 | 200 |  3.174401809s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:47:08 | 200 |  2.360547983s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:47:11 | 200 |  2.250021031s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:47:13 | 200 |  2.887492094s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:47:16 | 200 |  2.948315063s |       127.0.0.1 | POST     "/api/chat"
 50%|█████     | 15/30 [06:56<07:30, 30.06s/it][GIN] 2024/08/30 - 04:47:20 | 200 |  3.937717066s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:47:22 | 200 |  1.386478392s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:47:26 | 200 |   4.66131243s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:47:29 | 200 |  2.897371311s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:47:34 | 200 |  4.506410423s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:47:35 | 200 |  1.685792622s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:47:37 | 200 |  1.834372954s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:47:40 | 200 |   2.55283504s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:47:42 | 200 |  2.467530266s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:47:44 | 200 |  2.041667353s |       127.0.0.1 | POST     "/api/chat"
 53%|█████▎    | 16/30 [07:22<06:45, 28.98s/it][GIN] 2024/08/30 - 04:47:47 | 200 |  2.408729896s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:47:49 | 200 |   2.55151369s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:47:52 | 200 |  3.102177138s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:47:55 | 200 |  3.051747185s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:47:58 | 200 |  2.108766017s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:48:00 | 200 |   2.12312421s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:48:02 | 200 |  2.173557141s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:48:04 | 200 |  1.602796878s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:48:06 | 200 |  2.349326201s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:48:08 | 200 |  1.750761796s |       127.0.0.1 | POST     "/api/chat"
 57%|█████▋    | 17/30 [07:45<05:52, 27.11s/it][GIN] 2024/08/30 - 04:48:10 | 200 |  1.942555269s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:48:12 | 200 |  2.574179059s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:48:15 | 200 |  2.511373658s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:48:18 | 200 |  3.397738371s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:48:21 | 200 |  2.766109941s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:48:23 | 200 |  2.092229478s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:48:25 | 200 |  1.619729432s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:48:27 | 200 |  2.854537571s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:48:30 | 200 |  2.306574914s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:48:33 | 200 |  3.177324871s |       127.0.0.1 | POST     "/api/chat"
 60%|██████    | 18/30 [08:11<05:19, 26.61s/it][GIN] 2024/08/30 - 04:48:35 | 200 |  2.114814207s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:48:38 | 200 |  2.658517051s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:48:39 | 200 |  1.832844328s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:48:42 | 200 |  2.121777584s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:48:46 | 200 |  4.419762792s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:48:48 | 200 |  2.253535106s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:48:50 | 200 |  1.413684876s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:48:52 | 200 |  1.847350236s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:48:54 | 200 |  2.146573229s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:48:56 | 200 |  2.305255295s |       127.0.0.1 | POST     "/api/chat"
 63%|██████▎   | 19/30 [08:34<04:43, 25.74s/it][GIN] 2024/08/30 - 04:48:59 | 200 |  2.674189449s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:49:02 | 200 |  3.485004263s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:49:06 | 200 |  3.462789049s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:49:08 | 200 |  2.086997223s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:49:10 | 200 |  2.046893897s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:49:13 | 200 |  2.716042541s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:49:15 | 200 |  2.524806519s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:49:18 | 200 |  2.981624093s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:49:21 | 200 |  3.153144485s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:49:25 | 200 |  3.377546632s |       127.0.0.1 | POST     "/api/chat"
 67%|██████▋   | 20/30 [09:04<04:29, 26.92s/it][GIN] 2024/08/30 - 04:49:28 | 200 |  3.829511617s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:49:31 | 200 |  3.049575753s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:49:33 | 200 |  1.832759403s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:49:38 | 200 |  4.412256551s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:49:41 | 200 |  3.470579889s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:49:43 | 200 |  2.309375648s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:49:46 | 200 |  2.394811244s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:49:48 | 200 |  2.339353601s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:49:51 | 200 |  2.378975534s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:49:53 | 200 |  2.123723128s |       127.0.0.1 | POST     "/api/chat"
 70%|███████   | 21/30 [09:30<03:59, 26.66s/it][GIN] 2024/08/30 - 04:49:54 | 200 |  1.718944882s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:49:58 | 200 |  3.100085783s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:50:00 | 200 |  2.312282706s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:50:02 | 200 |  2.199996424s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:50:04 | 200 |  1.736915197s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:50:06 | 200 |  2.113982092s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:50:08 | 200 |  2.002398078s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:50:10 | 200 |  1.818663593s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:50:12 | 200 |  2.504086946s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:50:14 | 200 |  1.647790163s |       127.0.0.1 | POST     "/api/chat"
 73%|███████▎  | 22/30 [09:51<03:19, 24.95s/it][GIN] 2024/08/30 - 04:50:15 | 200 |   1.49353163s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:50:17 | 200 |   2.05639287s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:50:19 | 200 |  1.714817776s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:50:21 | 200 |  2.201832075s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:50:24 | 200 |  2.669108813s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:50:28 | 200 |  3.613860646s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:50:30 | 200 |  2.170585943s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:50:32 | 200 |  2.281448484s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:50:35 | 200 |    2.8369912s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:50:37 | 200 |  2.112293556s |       127.0.0.1 | POST     "/api/chat"
 77%|███████▋  | 23/30 [10:15<02:52, 24.70s/it][GIN] 2024/08/30 - 04:50:40 | 200 |  2.436504744s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:50:42 | 200 |  2.323040956s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:50:44 | 200 |  2.341299488s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:50:46 | 200 |  2.273607882s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:50:50 | 200 |  3.662882341s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:50:52 | 200 |   2.31100754s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:50:56 | 200 |  3.130212483s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:50:58 | 200 |  2.086634431s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:51:00 | 200 |  2.161859091s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:51:03 | 200 |  2.698584534s |       127.0.0.1 | POST     "/api/chat"
 80%|████████  | 24/30 [10:42<02:31, 25.24s/it][GIN] 2024/08/30 - 04:51:06 | 200 |  3.481635827s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:51:09 | 200 |  2.884372097s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:51:11 | 200 |  2.020608488s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:51:14 | 200 |  3.329344972s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:51:17 | 200 |  3.172647139s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:51:20 | 200 |   2.32741369s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:51:22 | 200 |  2.135936473s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:51:25 | 200 |  2.643380265s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:51:26 | 200 |  1.681214036s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:51:29 | 200 |  2.729367458s |       127.0.0.1 | POST     "/api/chat"
 83%|████████▎ | 25/30 [11:07<02:06, 25.34s/it][GIN] 2024/08/30 - 04:51:32 | 200 |  2.622000976s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:51:34 | 200 |  2.598050748s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:51:37 | 200 |  2.899135612s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:51:39 | 200 |  2.162757458s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:51:43 | 200 |  3.985724019s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:51:45 | 200 |  2.038539192s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:51:48 | 200 |  2.259091747s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:51:50 | 200 |  2.790916635s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:51:53 | 200 |  2.249874632s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:51:55 | 200 |  2.503171618s |       127.0.0.1 | POST     "/api/chat"
 87%|████████▋ | 26/30 [11:34<01:42, 25.72s/it][GIN] 2024/08/30 - 04:51:58 | 200 |  3.091636895s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:52:00 | 200 |  2.100926198s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:52:03 | 200 |  2.276423187s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:52:04 | 200 |  1.853357553s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:52:07 | 200 |  2.946147103s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:52:10 | 200 |  2.443112784s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:52:13 | 200 |  2.873159603s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:52:16 | 200 |  3.508446495s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:52:19 | 200 |  2.524005148s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:52:21 | 200 |  1.938876787s |       127.0.0.1 | POST     "/api/chat"
 90%|█████████ | 27/30 [11:59<01:16, 25.47s/it][GIN] 2024/08/30 - 04:52:23 | 200 |  2.420576001s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:52:27 | 200 |  3.853549434s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:52:30 | 200 |  3.036521576s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:52:33 | 200 |  3.435255965s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:52:36 | 200 |  2.198535006s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:52:38 | 200 |  2.547911825s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:52:41 | 200 |  2.368227275s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:52:42 | 200 |  1.572854174s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:52:44 | 200 |  1.854939043s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:52:47 | 200 |  3.115806595s |       127.0.0.1 | POST     "/api/chat"
 93%|█████████▎| 28/30 [12:25<00:51, 25.60s/it][GIN] 2024/08/30 - 04:52:49 | 200 |  1.876112855s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:52:52 | 200 |  3.255881493s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:52:55 | 200 |  3.090600983s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:52:58 | 200 |   2.76576912s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:53:01 | 200 |   3.22381137s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:53:04 | 200 |  2.836469625s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:53:06 | 200 |  2.080653717s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:53:10 | 200 |  3.424425419s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:53:11 | 200 |  1.685230177s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:53:14 | 200 |  2.358890715s |       127.0.0.1 | POST     "/api/chat"
 97%|█████████▋| 29/30 [12:53<00:26, 26.43s/it][GIN] 2024/08/30 - 04:53:17 | 200 |  3.624484188s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:53:20 | 200 |  2.222250185s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:53:22 | 200 |  2.175325842s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:53:24 | 200 |  2.648119533s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:53:26 | 200 |  1.942938915s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:53:28 | 200 |  1.901483866s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:53:32 | 200 |  3.356280382s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:53:35 | 200 |  3.483493597s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:53:39 | 200 |  3.493498253s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2024/08/30 - 04:53:42 | 200 |  3.365031249s |       127.0.0.1 | POST     "/api/chat"
100%|██████████| 30/30 [13:21<00:00, 26.71s/it][GIN] 2024/08/30 - 04:53:45 | 200 |  3.165600528s |       127.0.0.1 | POST     "/api/chat"
El archivo se ha guardado en /kaggle/working/ds_preguntasSimilares_CINCO.csv
```

# Retrival: recuperación de articulos más similares


**[Celda 18 - Código]**
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


**[Celda 19 - Código]**
```python
def funcionArticuloEncontrado(lista_articulos_proximos, articulo_correcto):
    for i, (article, _) in enumerate(lista_articulos_proximos):
        if articulo_correcto.strip() == article.strip():
            return i + 1
    return None
```

# Sección de pruebas y evaluación

## DataSet de preguntas etiquetadas



**[Celda 22 - Código]**
```python
# Cargar el archivo  preguntas similares CSV en un DataFrame

# RUTAS  EN LA PC DEL LABORATORIO
#csv_path = r'C:\Users\Usuario\Documents\GIBD\RAG\dataSets_Consultas\ds_preguntasSimilares_llama31_TRES.csv'
#csv_path = r'C:\Users\Usuario\Documents\GIBD\RAG\dataSets_Consultas\ds_preguntasSimilares_llama31_DIEZ.csv'

#Cambiar el path para obtener el csv generado anteriormente en el caso de querer cargar distintos csv para visualizar
csv_path = r'/kaggle/working/ds_preguntasSimilares_CINCO.csv'
similaresCSV_df = pd.read_csv(csv_path)
print(similaresCSV_df .head())
```


*Salida:*
```text
idConsulta                                preguntas_similares
0           1  Cuales son las condiciones para presentarse a ...
1           2  Quién debe regir la convocatoria para la desig...
2           3  Cómo se publicita un llamado a concurso?\nCómo...
3           4  Qué contenido tienen los anuncios?\n¿Cuál es l...
4           5  Cuales son los requisitos para los aspirantes ...
```


**[Celda 23 - Código]**
```python
#Cargar preguntas etiquetadas

dataset = load_dataset("HuggMaxi/preguntasEtiquetadas")
df_preguntasEtiquetadas = pd.DataFrame(dataset['train'])
df_preguntasEtiquetadas
```


*Salida:*
```text
idConsulta                                           Consulta Articulo  \
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


**[Celda 24 - Código]**
```python
# Unir los DataFrames en base a la columna 'idConsulta'
df_similares = pd.merge(similaresCSV_df, df_preguntasEtiquetadas, left_on='idConsulta', right_on='idConsulta')
df_similares
```


*Salida:*
```text
idConsulta                                preguntas_similares  \
0             1  Cuales son las condiciones para presentarse a ...   
1             2  Quién debe regir la convocatoria para la desig...   
2             3  Cómo se publicita un llamado a concurso?\nCómo...   
3             4  Qué contenido tienen los anuncios?\n¿Cuál es l...   
4             5  Cuales son los requisitos para los aspirantes ...   
..          ...                                                ...   
295         296  Cual es la responsabilidad principal del Estad...   
296         297  Que sucede cuando una carrera no obtiene su ac...   
297         298  Quienes seran los organos de coordinacion y co...   
298         299  Que atribuciones tiene la autonomia academica ...   
299         300  Cuales son los deberes de los docentes de las ...   

                                              Consulta Articulo  \
0     Cuales son las condiciones para presentarse a...        9   
1     Quién debe regir la convocatoria para la desi...        3   
2             Cómo se publicita un llamado a concurso?        6   
3                   Qué contenido tienen los anuncios?        8   
4     Cuales son los requisitos para los aspirantes...        9   
..                                                 ...      ...   
295   Cual es la responsabilidad principal del Esta...      315   
296   Que sucede cuando una carrera no obtiene su a...      299   
297   Quienes seran los organos de coordinacion y c...      294   
298   Que atribuciones tiene la autonomia academica...      250   
299   Cuales son los deberes de los docentes de las...      233   

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

[300 rows x 5 columns]
```

## Evaluación por consulta individual


**[Celda 26 - Código]**
```python
idConsulta = 1
texto_objetivo = df_similares[df_similares['idConsulta'] == idConsulta]['Texto'].values[0]
pregunta_consulta = df_similares[df_similares['idConsulta'] == idConsulta]['preguntas_similares'].values[0]

resultadoArticulosEncontrados = devolver_N_articulos_similares(pregunta_consulta, model, articles_embedding, articulos,10)
resultadoPosicion = funcionArticuloEncontrado(resultadoArticulosEncontrados, texto_objetivo)

print(f"CONSULTA: {pregunta_consulta} \n")
print(f"POSICION: {resultadoPosicion} \n")
print("*********************************************")
print(texto_objetivo)
print("*********************************************")


```


*Salida:*
```text
Batches:   0%|          | 0/1 [00:00<?, ?it/s]CONSULTA: Cuales son las condiciones para presentarse a un concurso?
¿Qué requisitos hay que cumplir para participar en una competencia académica?
¿Cuáles son los criterios para acceder a un proceso de selección universitario?
¿Qué normas deben seguirse para postularse a un certamen estudiantil?
¿Cómo se determinan las condiciones para inscribirse en un concurso escolar? 

POSICION: 1 

*********************************************
Documento:8
Articulo: 9
Capitulo: III - INSCRIPCIÓN

Condiciones de los aspirantes: Para presentarse a Concurso el aspirante debe reunir los siguientes requisitos: a) Tener título universitario de grado o en su defecto acreditar antecedentes excepcionales que lo suplan, y b) no estar comprendido en las causales de inhabilitación para el desempeño de cargos públicos y de faltas a la ética universitaria que se mencionan en el artículo 18 del presente Reglamento.
*********************************************
```


**[Celda 27 - Código]**
```python
for i in range(len(resultadoArticulosEncontrados)):
    print(f"-------------------------------------- {i + 1}")
    print(resultadoArticulosEncontrados[i][0])
    print()
```


*Salida:*
```text
-------------------------------------- 1
Documento:8
Articulo: 9
Capitulo: III - INSCRIPCIÓN

Condiciones de los aspirantes: Para presentarse a Concurso el aspirante debe reunir los siguientes requisitos: a) Tener título universitario de grado o en su defecto acreditar antecedentes excepcionales que lo suplan, y b) no estar comprendido en las causales de inhabilitación para el desempeño de cargos públicos y de faltas a la ética universitaria que se mencionan en el artículo 18 del presente Reglamento.

-------------------------------------- 2
Documento:13
Articulo: 273
Capitulo: TÍTULO IV: DE LA EDUCACIÓN SUPERIOR UNIVERSITARIA. CAPÍTULO 4: De las instituciones universitarias nacionales. Sección I: Creación y bases organizativas

El ingreso a la carrera académica universitaria se hará mediante concurso publico y abierto de antecedentes y oposición, debiéndose asegurar la constitución de jurados integrados por profesores por concurso, o excepcionalmente por personas de idoneidad indiscutible aunque no reúnan esa condición, que garanticen la mayor imparcialidad y el máximo rigor académico. Con carácter excepcional, las universidades e institutos universitarios nacionales podrán contratar, al margen del régimen de concursos y solo por tiempo determinado, a personalidades de reconocido prestigio y méritos académicos sobresalientes para que desarrollen cursos, seminarios o actividades similares. Podrán igualmente prever la designación temporaria de docentes interinos, cuando ello sea imprescindible y mientras se sustancia el correspondiente concurso. Los docentes designados por concurso deberán representar un porcentaje no inferior al setenta por ciento (70%) de las respectivas plantas de cada institución universitaria.

-------------------------------------- 3
Documento:13
Articulo: 256
Capitulo: TÍTULO IV: DE LA EDUCACIÓN SUPERIOR UNIVERSITARIA. CAPÍTULO 3: De las condiciones para su funcionamiento. Sección I: Requisitos generales

Para ingresar como alumno a las instituciones universitarias, sean estatales o privadas, deberá reunirse como mínimo la condición prevista en el artículo 7º y cumplir con los demás requisitos del sistema de admisión que cada institución establezca.

-------------------------------------- 4
Documento:8
Articulo: 18
Capitulo: III - INSCRIPCIÓN

Inscripciones múltiples: El aspirante que se presente a más de un Concurso deberá cumplir en cada uno de ellos con todos los requisitos establecidos en esta ordenanza, sin poder remitirse a los escritos o documentos presentados en los otros. Si los Concursos a los que se presenta integran el mismo llamado de una Unidad Académica, basta que el aspirante cumplimente en uno de ellos la exigencia del articulo 10°, refiriéndose en las solicitudes de inscripción de los otros Concursos, en cual agregó la documentación que menciona el citado articulo y siempre que no la haya retirado de la Unidad Académica.

-------------------------------------- 5
Documento:8
Articulo: 58
Capitulo: X - NORMAS GENERALES

La inscripción al Concurso importará para el aspirante su conformidad con las normas de este Reglamento y las específicas dictadas por cada Unidad Académica.

-------------------------------------- 6
Documento:10006
Articulo: 10041
Capitulo: De la Tesis. Capítulo II. Aspirantes a Tesis y Cuestiones Relativas a los Estudiantes.

La Secretaría Académica de las respectivas Facultades, acorde a los Planes de Estudios, fijará los requisitos que los estudiantes de su(s) carrera(s) deberán cumplir para poder inscribirse en la asignatura de TESIS o equivalente, pudiendo basarse en: I. Número de unidades aprobadas de la carrera. II. Número de materias que faltan cursar de la carrera. III. Además de los requisitos anteriores, el cursar ciertos materias que sean de importancia específica para el desarrollo del proyecto en que el estudiante desea involucrarse.

-------------------------------------- 7
Documento:13
Articulo: 277
Capitulo: TÍTULO IV: DE LA EDUCACIÓN SUPERIOR UNIVERSITARIA. CAPÍTULO 4: De las instituciones universitarias nacionales. Sección II: Órganos de gobierno

Los representantes de los docentes, que deberán haber accedido a sus cargos por concurso, serán elegidos por docentes que reúnan igual calidad. Los representantes estudiantiles serán elegidos por sus pares, siempre que estos tengan el rendimiento académico mínimo que establece el artículo 50.

-------------------------------------- 8
Documento:8
Articulo: 61
Capitulo: X - NORMAS GENERALES

Los llamados a Concurso, se harán dentro de lo posible para una cátedra, área, núcleo, asignaturas que integran los núcleos, disciplina, laboratorio, pero no para los cursos en que estos estuvieran divididos, pudiendo cada Facultad adaptar lo aquí dispuesto a su estructura académica.

-------------------------------------- 9
Documento:5
Articulo: 104
Capitulo: IV. DE LOS ALUMNOS. CONDICIÓN DE LOS ALUMNOS Y REGULARIDAD

La inscripción en una asignatura  siempre que se ajuste al régimen de correlatividades establecido en el Plan de Estudios correspondiente - otorga al alumno la posibilidad de acceder a la condición de regular, que le permite concurrir a clase y participar de todas las actividades inherentes al proceso de enseñanza y de aprendizaje planificadas por la cátedra. Tal condición será adquirida cuando el alumno cumplimente los requisitos fijados en la planificación de cada cátedra.

-------------------------------------- 10
Documento:13
Articulo: 254
Capitulo: TÍTULO IV: DE LA EDUCACIÓN SUPERIOR UNIVERSITARIA. CAPÍTULO 3: De las condiciones para su funcionamiento. Sección I: Requisitos generales

Las instituciones universitarias deben promover la excelencia y asegurar la libertad académica, la igualdad de oportunidades y posibilidades, la jerarquización docente, la corresponsabilidad de todos los miembros de la comunidad universitaria, así como la convivencia pluralista de corrientes, teorías y líneas de investigación. Cuando se trate de instituciones universitarias privadas, dicho pluralismo se entenderá en un contexto de respeto a las cosmovisiones y valores expresamente declarados en sus estatutos.
```

## Creación de Dataset de resultados


**[Celda 29 - Código]**
```python
batch_size = 10  # Tamaño del lote
total_consultas = len(df_similares)
resultados = []  # Lista para almacenar los resultados antes de crear el DataFrame final

for start in range(0, total_consultas, batch_size):
    # Definir el rango de consultas en el lote
    end = min(start + batch_size, total_consultas)
    batch_df = df_similares.iloc[start:end]

    for _, row in batch_df.iterrows():
        idCon = row['idConsulta']
        pregunta_consultaConSimilares = row['preguntas_similares']
        texto_consulta = row['Texto']

        # Obtener los N artículos más similares (N=10 en este caso)
        resultadoArticulosEncontrados = devolver_N_articulos_similares(pregunta_consultaConSimilares, model, articles_embedding, articulos, total_consultas)

        # Encontrar la posición del artículo correcto
        resultadoPosicion = funcionArticuloEncontrado(resultadoArticulosEncontrados, texto_consulta)

        # Agregar el resultado a la lista de resultados
        resultados.append({
            'idConsulta': idCon,
            'preguntas_similares': pregunta_consultaConSimilares,
            'Posicion': resultadoPosicion
        })

# Convertir la lista de resultados en un DataFrame
resultados_df = pd.DataFrame(resultados)

# Mostrar el DataFrame
print(resultados_df)
```


*Salida:*
```text
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]     idConsulta                                preguntas_similares  Posicion
0             1  Cuales son las condiciones para presentarse a ...       1.0
1             2  Quién debe regir la convocatoria para la desig...       1.0
2             3  Cómo se publicita un llamado a concurso?\nCómo...       1.0
3             4  Qué contenido tienen los anuncios?\n¿Cuál es l...       1.0
4             5  Cuales son los requisitos para los aspirantes ...       1.0
..          ...                                                ...       ...
295         296  Cual es la responsabilidad principal del Estad...       1.0
296         297  Que sucede cuando una carrera no obtiene su ac...       1.0
297         298  Quienes seran los organos de coordinacion y co...       1.0
298         299  Que atribuciones tiene la autonomia academica ...       1.0
299         300  Cuales son los deberes de los docentes de las ...       1.0

[300 rows x 3 columns]
```


**[Celda 30 - Código]**
```python
#Opcional guardar Dataset

# 1. Convertir 'Posicion' a enteros, manejando los valores NaN como -1
resultados_df['Posicion'] = resultados_df['Posicion'].fillna(-1).astype(int)
csv_path = r'/kaggle/working/resultados_bge-m3_llama3_1_similares_CINCO.csv' #Cambiar según necesario para almacenar los resultados en un csv
resultados_df.to_csv(csv_path, index=False)

```

## Visualización de resultados


**[Celda 32 - Código]**
```python
# Cargar el archivo CSV en un DataFrame, si es un DataSet importado
#csv_path = r'C:\Users\Usuario\Documents\GIBD\RAG\dataSets_Resultados\resultados_beto_llama31_individual.csv'
#resultadosCSV_df = pd.read_csv(csv_path)
#resultados_df = resultadosCSV_df
#print(resultadosCSV_df .head())
```


**[Celda 33 - Código]**
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

plt.title('Distribución de Posiciones para bge-m3', fontsize=16)
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
1-3: 85.0% (255 preguntas)
4-6: 6.0% (18 preguntas)
7-10: 1.3% (4 preguntas)
No aparece: 7.7% (23 preguntas)
```


**[Celda 34 - Código]**
```python

```
