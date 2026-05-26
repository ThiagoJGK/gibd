---
aliases: [combined-rag-similares-ollama]
tags:
  - grupo/g3_texto
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2024-09-05
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/BERT - GPT - Llama/Notebooks/Deprecated/combined-rag-similares-ollama.ipynb"
tamanio_bytes: 905401
---

# Notebook: combined-rag-similares-ollama.ipynb

Ruta interna: `GIBD/BERT - GPT - Llama/Notebooks/Deprecated/combined-rag-similares-ollama.ipynb`

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
primary_model = SentenceTransformer('BAAI/bge-m3')
```


*Salida:*
```text
modules.json:   0%|          | 0.00/349 [00:00<?, ?B/s]config_sentence_transformers.json:   0%|          | 0.00/123 [00:00<?, ?B/s]README.md:   0%|          | 0.00/15.8k [00:00<?, ?B/s]sentence_bert_config.json:   0%|          | 0.00/54.0 [00:00<?, ?B/s]config.json:   0%|          | 0.00/687 [00:00<?, ?B/s]pytorch_model.bin:   0%|          | 0.00/2.27G [00:00<?, ?B/s]tokenizer_config.json:   0%|          | 0.00/444 [00:00<?, ?B/s]sentencepiece.bpe.model:   0%|          | 0.00/5.07M [00:00<?, ?B/s]tokenizer.json:   0%|          | 0.00/17.1M [00:00<?, ?B/s]special_tokens_map.json:   0%|          | 0.00/964 [00:00<?, ?B/s]1_Pooling/config.json:   0%|          | 0.00/191 [00:00<?, ?B/s]
```


**[Celda 6 - Código]**
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

# Creación de embeddings del reglamento


**[Celda 8 - Código]**
```python
import torch 

# Define el dispositivo
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


**[Celda 11 - Código]**
```python
#Indicar el path al reglamento original
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


**[Celda 12 - Código]**
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


**[Celda 13 - Código]**
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

# Creación de preguntas similares


**[Celda 15 - Código]**
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


**[Celda 16 - Código]**
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


**[Celda 17 - Código]**
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


**[Celda 18 - Código]**
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


**[Celda 19 - Código]**
```python
#idConsulta = 1
#prueba_pregunta_consulta = df_preguntasEtiquetadas[df_preguntasEtiquetadas['idConsulta'] == idConsulta]['Consulta'].values[0]
#prueba_preguntas_similares = generate_questions_3(prueba_pregunta_consulta)
#print(prueba_preguntas_similares)
```


**[Celda 20 - Código]**
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
            preguntas_similares = generate_questions_3(pregunta_consulta)
            
            # Guardar el resultado
            resultados.append({
                'idConsulta': idConsulta,
                'preguntas_similares': preguntas_similares
            })
    
    # Crear un nuevo DataFrame con los resultados
    df_resultados = pd.DataFrame(resultados)
    return df_resultados

# Aplicar la función al DataFrame original
#df_resultados = generar_preguntas_similares_por_lotes(df_preguntasEtiquetadas, lote_tamaño=10)

# Ruta para guardar el archivo CSV
#csv_path = r'/kaggle/working/ds_preguntasSimilares_TRES_nomic-bge.csv'

# Guardar el DataFrame resultante en un archivo CSV
#df_resultados.to_csv(csv_path, index=False)

#print(f"El archivo se ha guardado en {csv_path}")

```

# Retrieval: recuperación de articulos más similares


**[Celda 22 - Código]**
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


**[Celda 23 - Código]**
```python
def funcionArticuloEncontrado(lista_articulos_proximos, articulo_correcto):
    for i, (article, _) in enumerate(lista_articulos_proximos):
        if articulo_correcto.strip() == article.strip():
            return i + 1 
    return None  
```

# Sección de pruebas y evaluación 

## DataSet de preguntas etiquetadas



**[Celda 26 - Código]**
```python
# Cargar el archivo  preguntas similares CSV en un DataFrame

# RUTAS  EN LA PC DEL LABORATORIO
#csv_path = r'C:\Users\Usuario\Documents\GIBD\RAG\dataSets_Consultas\ds_preguntasSimilares_llama31_TRES.csv'
#csv_path = r'C:\Users\Usuario\Documents\GIBD\RAG\dataSets_Consultas\ds_preguntasSimilares_llama31_DIEZ.csv'
import pandas as pd

#Cambiar el dataset según la ubicación en la máquina donde se va a ejecutar
csv_path = r'/kaggle/input/gpt-4o-supervised/DataSet_preguntasSimilares_survervisado.csv'
similaresCSV_df = pd.read_csv(csv_path)
print(similaresCSV_df .head())
```


*Salida:*
```text
idConsulta                                           Consulta  \
0            1   Cuales son las condiciones para presentarse a...   
1            2   Quién debe regir la convocatoria para la desi...   
2            3           Cómo se publicita un llamado a concurso?   
3            4                 Qué contenido tienen los anuncios?   
4            5   Cuales son los requisitos para los aspirantes...   

                              10_preguntas_similares  \
0  Cuales son las condiciones para presentarse a ...   
1  Quién debe regir la convocatoria para la desig...   
2  Cómo se publicita un llamado a concurso?\n¿Cóm...   
3  Qué contenido tienen los anuncios?\n¿Qué infor...   
4  Cuales son los requisitos para los aspirantes ...   

                               5_preguntas_similares  \
0  Cuales son las condiciones para presentarse a ...   
1  Quién debe regir la convocatoria para la desig...   
2  Cómo se publicita un llamado a concurso?\n ¿Có...   
3  Qué contenido tienen los anuncios?\n ¿Qué info...   
4  Cuales son los requisitos para los aspirantes ...   

                               3_preguntas_similares  
0  Cuales son las condiciones para presentarse a ...  
1  Quién debe regir la convocatoria para la desig...  
2  Cómo se publicita un llamado a concurso?\n¿Cóm...  
3  Qué contenido tienen los anuncios?\n¿Qué infor...  
4  Cuales son los requisitos para los aspirantes ...
```


**[Celda 27 - Código]**
```python
# Eliminar espacios al inicio y al final de los nombres de columnas
similaresCSV_df.columns = similaresCSV_df.columns.str.strip()
df_preguntasEtiquetadas.columns = df_preguntasEtiquetadas.columns.str.strip()

```


**[Celda 28 - Código]**
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


**[Celda 29 - Código]**
```python
# Unir los DataFrames en base a la columna 'idConsulta'
df_similares = pd.merge(similaresCSV_df, df_preguntasEtiquetadas, left_on='idConsulta', right_on='idConsulta')
df_similares
```


*Salida:*
```text
idConsulta                                         Consulta_x  \
0             1   Cuales son las condiciones para presentarse a...   
1             2   Quién debe regir la convocatoria para la desi...   
2             3           Cómo se publicita un llamado a concurso?   
3             4                 Qué contenido tienen los anuncios?   
4             5   Cuales son los requisitos para los aspirantes...   
..          ...                                                ...   
295         296  Cual es la responsabilidad principal del Estad...   
296         297  Que sucede cuando una carrera no obtiene su ac...   
297         298  Quienes seran los organos de coordinacion y co...   
298         299  Que atribuciones tiene la autonomia academica ...   
299         300  Cuales son los deberes de los docentes de las ...   

                                10_preguntas_similares  \
0    Cuales son las condiciones para presentarse a ...   
1    Quién debe regir la convocatoria para la desig...   
2    Cómo se publicita un llamado a concurso?\n¿Cóm...   
3    Qué contenido tienen los anuncios?\n¿Qué infor...   
4    Cuales son los requisitos para los aspirantes ...   
..                                                 ...   
295  Cual es la responsabilidad principal del Estad...   
296  Que sucede cuando una carrera no obtiene su ac...   
297  Quienes seran los organos de coordinacion y co...   
298  Que atribuciones tiene la autonomia academica ...   
299  Cuales son los deberes de los docentes de las ...   

                                 5_preguntas_similares  \
0    Cuales son las condiciones para presentarse a ...   
1    Quién debe regir la convocatoria para la desig...   
2    Cómo se publicita un llamado a concurso?\n ¿Có...   
3    Qué contenido tienen los anuncios?\n ¿Qué info...   
4    Cuales son los requisitos para los aspirantes ...   
..                                                 ...   
295  Cual es la responsabilidad principal del Estad...   
296  Que sucede cuando una carrera no obtiene su ac...   
297  Quienes seran los organos de coordinacion y co...   
298  Que atribuciones tiene la autonomia academica ...   
299  Cuales son los deberes de los docentes de las ...   

                                 3_preguntas_similares  \
0    Cuales son las condiciones para presentarse a ...   
1    Quién debe regir la convocatoria para la desig...   
2    Cómo se publicita un llamado a concurso?\n¿Cóm...   
3    Qué contenido tienen los anuncios?\n¿Qué infor...   
4    Cuales son los requisitos para los aspirantes ...   
..                                                 ...   
295  Cual es la responsabilidad principal del Estad...   
296  Que sucede cuando una carrera no obtiene su ac...   
297  Quienes seran los organos de coordinacion y co...   
298  Que atribuciones tiene la autonomia academica ...   
299  Cuales son los deberes de los docentes de las ...   

                                            Consulta_y Articulo  \
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

[300 rows x 8 columns]
```

## Evaluación por consulta individual 


**[Celda 31 - Código]**
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


**[Celda 32 - Código]**
```python
for i in range(len(resultadoArticulosEncontrados)):
    print(f"-------------------------------------- {i + 1}")
    print(resultadoArticulosEncontrados[i][0])
    print() 
```

## Creación de Dataset de resultados 


**[Celda 34 - Código]**
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
        pregunta_consulta = row['3_preguntas_similares']
        print(pregunta_consulta)
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
Cuales son las condiciones para presentarse a un concurso?
¿Cuáles son los requisitos para participar en un concurso?
¿Qué condiciones se deben cumplir para presentarse a un concurso?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Quién debe regir la convocatoria para la designación por concurso de profesores ordinarios?
¿Quién tiene la autoridad para manejar la convocatoria para la designación por concurso de profesores ordinarios?
¿Quién es responsable de llevar adelante la convocatoria para la designación por concurso de profesores ordinarios?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cómo se publicita un llamado a concurso?
¿Cómo se anuncia un llamado a concurso?
¿Cómo se hace pública una convocatoria a concurso?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Qué contenido tienen los anuncios?
¿Qué información contienen los anuncios?
¿Qué datos se detallan en los anuncios?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cuales son los requisitos para los aspirantes a un concurso?
¿Cuáles son las condiciones que deben cumplir los aspirantes a un concurso?
¿Qué se necesita para anotarse en un concurso?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cuales son los plazos de inscripción?
¿Cuáles son los tiempos para anotarse?
¿Qué plazos están estipulados para la inscripción?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]En dónde se realizan las inscripciones?
¿Dónde se lleva a cabo la inscripción?
¿En qué lugar hay que anortase para las inscripciones?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que documentación deben presentar los aspirantes a un concurso?
¿Qué papeles tienen que presentar los aspirantes a un concurso?
¿Qué documentos deben entregar los aspirantes a un concurso?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cuales son los requisitos ante inscripciones múltiples?
¿Cuáles son las condiciones ante inscripciones múltiples?
¿Qué requisitos se piden en casos de inscripciones múltiples?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que se hace en el cierre de inscripción?
¿Qué acciones se realizan al finalizar la inscripción?
¿Qué procedimientos se siguen en el cierre del periodo de inscripción?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que se realiza en la exhibición de nómina de aspirantes
¿Qué se hace durante la exhibición de la lista de aspirantes?
¿Qué se lleva a cabo en la exhibición de la nómina de aspirantes?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Quienes pueden ejercer el derecho de impugnar la participación en el concurso de los aspirantes?
¿Quiénes están habilitados para impugnar la participación de los candidatos en un concurso?
¿Quiénes tienen la potestad de objetar la participación de aspirantes en el concurso?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que consecuencias tiene la interposición de la impugnación?
¿Qué efectos produce la presentación de una impugnación?
¿Qué sucede al interponer una impugnación?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que tiene que pasar para que el consejo directivo excluya a un aspirante?
¿Qué condiciones se deben cumplir para que el consejo directivo decida excluir a un aspirante?
¿Qué motivos debe haber para que el consejo directivo expulse a un aspirante?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que organismo designa a los integrantes del jurado?
¿Qué entidad es responsable de nombrar a los miembros del jurado?
¿Quién designa a los integrantes del jurado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que condiciones deben reunir los integrantes del jurado?
¿Qué requisitos deben cumplir los miembros del jurado?
¿Qué características se exigen a los integrantes del jurado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cómo se realiza la recusación de los miembros del jurado?
¿Cómo se lleva a cabo la recusación de un miembro del jurado?
¿Qué pasos se siguen para recusar a un miembro del jurado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cuales son los causales de recusación para algún aspirante?
¿Qué motivos pueden llevar a la recusación de un aspirante?
¿Qué razones justifican la recusación de un candidato?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Qué organismo resuelve las recusaciones y excusaciones del jurado?
¿Qué entidad es la encargada de resolver las recusaciones y excusaciones del jurado?
¿Quién toma decisiones sobre las recusaciones y excusaciones del jurado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Quien reemplaza a un jurado recusado o excusado?
¿Quién toma el lugar de un jurado que ha sido recusado o excusado?
¿Quién sustituye a un miembro del jurado que es recusado o excusado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que presencia hace falta para que el jurado funcione válidamente?
¿Qué asistencia es necesaria para que el jurado pueda operar válidamente?
¿Qué presencia se requiere para que el jurado sea válido?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que antelación tiene que haber para realizar un sorteo del tema y orden de exposición?
¿Cuánto tiempo previo es necesario para realizar el sorteo del tema y orden de exposición?
¿Qué anticipación se requiere para efectuar el sorteo de tema y orden de exposición?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cuales son las pautas para la evaluación de un aspirante?
¿Qué criterios se utilizan para evaluar a un aspirante?
¿Cuáles son las normas para la evaluación de un candidato?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cuanto tiempo debe pasar para notificar las conclusiones del dictamen del jurado?
¿Cuál es el plazo para informar las conclusiones del dictamen del jurado?
¿Cuánto tiempo se necesita para notificar el resultado del dictamen del jurado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]A cargo de que organismo está la designación de profesores por concurso?
¿Qué entidad es responsable de la designación de profesores por concurso?
¿A qué organismo le compete la designación de profesores mediante concurso?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Qué sucede cuando existen reales dificultades para obtener miembros del jurado?
¿Qué ocurre si hay dificultades reales para conseguir miembros del jurado?
¿Qué se hace cuando es difícil conseguir miembros para el jurado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que se realiza ante la aparición de una nueva vacante o puesto curricular?
¿Qué acciones se toman al surgir una nueva vacante o puesto curricular?
¿Qué se hace cuando aparece una nueva vacante o puesto curricular?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Qué carácter tienen las solicitudes de inscripción a un concurso docente?
¿Qué tipo de carácter poseen las solicitudes de inscripción a un concurso docente?
¿Cómo se caracteriza una solicitud de inscripción en un concurso docente?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Quienes pueden recusar a los miembros del jurado?
¿Quiénes están habilitados para recusar a los miembros del jurado?
¿Quiénes tienen el derecho de recusar a un miembro del jurado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Quien resuelve las excusaciones y recusaciones
¿Quién se encarga de resolver las excusaciones y recusaciones?
¿Qué autoridad decide sobre las excusaciones y recusaciones?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que actividades no puede realizar un alumno en categoría pasivo?
¿Qué actividades tiene prohibidas un alumno en categoría pasivo?
¿Qué actividades le están vedadas a un alumno en categoría pasivo?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Hasta cuando se mantiene la regularidad de la asignatura?
¿Cuánto tiempo se conserva la regularidad de una asignatura?
¿Hasta qué momento se mantiene la regularidad en una asignatura?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que implica la pérdida de regularidad?
¿Qué consecuencias tiene perder la regularidad?
¿Qué sucede cuando se pierde la regularidad en una asignatura?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que es un alumno vocacional?
¿Qué se entiende por alumno vocacional?
¿Qué características definen a un alumno vocacional?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que derechos tiene un alumno vocacional?
¿Qué derechos le corresponden a un alumno vocacional?
¿Qué privilegios posee un alumno vocacional?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como puede aprobar una asignatura un alumno regular?
¿Cómo logra un alumno regular aprobar una materia?
¿Qué tiene que hacer un alumno regular para aprobar una asignatura?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como puede aprobar una materia un alumno libre?
¿Qué debe hacer un alumno libre para aprobar una materia?
¿Cómo logra un alumno libre aprobar una asignatura?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como se expresa la calificación final de una asignatura?
¿Cómo se indica la calificación final de una asignatura?
¿De qué manera se muestra la nota final de una asignatura?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Quienes pueden solicitar equivalencias de algunas asignaturas?
¿Quiénes están habilitados para pedir equivalencias de asignaturas?
¿Quiénes pueden gestionar la equivalencia de algunas asignaturas?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que tipo de ámbitos son los consejos de carrera?
¿Qué clase de entidades son los consejos de carrera?
¿Qué son los consejos de carrera?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que datos contienen los anuncios?
¿Qué información aparece en los anuncios?
¿Qué detalles se incluyen en los anuncios?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Quien esta a cargo de la difusión del llamado a concurso?
¿Quién es responsable de la difusión del llamado a concurso?
¿A quién le corresponde la difusión del llamado a concurso?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cual es el periodo de inscripción a la convocatoria?
¿Cuál es el plazo de inscripción a la convocatoria?
¿Cuánto tiempo dura la inscripción a la convocatoria?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como realizo mi solicitud de inscripción?
¿Cómo puedo hacer mi solicitud de inscripción?
¿Qué pasos debo seguir para realizar mi solicitud de inscripción?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cuando presento el planteamiento de la cátedra?
¿Cuándo debo entregar el planteamiento de la cátedra?
¿En qué momento se presenta el planteamiento de la cátedra?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Puedo presentarme a más de un concurso?
¿Es posible presentarse a más de un concurso?
¿Tengo la opción de inscribirme en más de un concurso?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Puede un apoderado intervenir en mis trámites?
¿Es posible que un apoderado intervenga en mis trámites?
¿Puede un representante legal gestionar mis trámites?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como se designa al jurado?
¿Cómo se elige a los miembros del jurado?
¿Qué procedimiento se sigue para designar al jurado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como se compone el jurado?
¿Cómo está conformado el jurado?
¿Qué integrantes conforman el jurado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son las causales de recusación de un miembro del jurado?
¿Qué motivos existen para recusar a un miembro del jurado?
¿Bajo qué razones se puede recusar a un integrante del jurado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cómo evalúa el jurado al aspirante?
¿Qué criterios utiliza el jurado para evaluar al aspirante?
¿De qué manera realiza el jurado la evaluación del aspirante?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Quiénes participan en la deliberación del jurado?
¿Quiénes forman parte de la deliberación del jurado?
¿Quiénes intervienen en la discusión del jurado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Puede haber más de un aspirante con el mismo mérito?
¿Es posible que varios aspirantes tengan el mismo mérito?
¿Pueden existir aspirantes con igual mérito?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué restricciones tiene un alumno pasivo?
¿Qué limitaciones enfrenta un alumno pasivo?
¿Qué cosas no puede hacer un alumno pasivo?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué podrá realizar un alumno activo?
¿Qué está permitido para un alumno activo?
¿Qué actividades puede realizar un alumno activo?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuánto tiempo se mantiene la regularidad en una asignatura?
¿Durante cuánto tiempo conserva un alumno la regularidad en una asignatura?
¿Cuál es el periodo de vigencia de la regularidad en una asignatura?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿De qué consta la pérdida de la regularidad?
¿Qué implica la pérdida de la regularidad?
¿Qué consecuencias trae la pérdida de la regularidad?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué está habilitado a realizar un alumno libre?
¿Qué puede hacer un alumno libre?
¿Qué actividades puede llevar a cabo un alumno libre?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿En qué momento se da a conocer la condición de cada alumno?
¿Cuándo se informa la condición de cada alumno?
¿En qué instancia se comunica la condición de cada alumno?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuántos turnos de exámenes finales se dictan por año lectivo?
¿Cuántas oportunidades de exámenes finales hay por año lectivo?
¿Cuántos periodos de exámenes finales se realizan anualmente?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué contiene un acta de evaluación?
¿Qué cosas se escriben en un acta de evaluación?
¿Qué elementos se describen en un acta de calificaciones?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cómo se lleva control de asignaturas correlativas?
¿Cómo se maneja el control de materias correlativas?
¿De qué forma se controla la correlación de materias?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Con qué categorías de docentes universitarios cuenta la Facultad?
¿Qué tipos de profesores hay en la Facultad?
¿Qué rangos docentes existen en la Facultad?

Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son las normas para el dictado de clases?
¿Qué regulaciones existen para el proceso de enseñanza?
¿Qué reglas hay para dar clases?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cómo se determina la condición de los alumnos?
¿Qué factores influyen en la condición de los estudiantes?
¿Cómo se decide la condición de los alumnos?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Quiénes están habilitados a ingresar a la enseñanza de grado?
 ¿Quiénes pueden acceder a la enseñanza universitaria?
 ¿Qué personas son elegibles para la educación de grado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son los derechos del estudiante?
 ¿Qué derechos tienen los alumnos?
 ¿Cuáles son las protecciones del estudiante en la universidad?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué requisitos se especifican para ser docente?
 ¿Qué condiciones se deben cumplir para ejercer como profesor?
 ¿Cuáles son las exigencias para ser profesor en la universidad?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son las condiciones para acceder a la formación de posgrado?
 ¿Qué exigencias hay para estudiar un posgrado?
 ¿Cuáles son los criterios para acceder a la formación de posgrado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿De qué se encarga el estado provincial?
 ¿Qué hace el estado provincial?
 ¿Cuál es la responsabilidad del estado provincial?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son los objetivos de la educación superior?
 ¿Qué metas tiene la educación superior?
 ¿Cuáles son las finalidades de la enseñanza superior?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son las modalidades del sistema educativo?
 ¿Qué categorías de modalidades están presentes en el sistema educativo?
 ¿Qué modelos de enseñanza se aplican en el sistema educativo?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son los órganos del gobierno universitario?
 ¿Qué órganos conforman el gobierno universitario?
 ¿Qué entidades forman parte del gobierno de la universidad?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Quiénes integran el consejo superior?
 ¿Quiénes forman parte del consejo superior?
 ¿Quiénes están en el consejo superior universitario?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Quiénes integran el consejo directivo?
 ¿Quiénes están en el consejo directivo?
 ¿Qué miembros conforman el consejo directivo?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son las funciones del consejo directivo?
 ¿Qué hace el consejo directivo?
 ¿Qué responsabilidades tiene el consejo directivo?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son los requisitos para ser decano de la Facultad?
 ¿Qué se necesita para ser decano?
 ¿Cuáles son los requerimientos para ser decano de la Facultad?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son las funciones del decano?
 ¿Qué responsabilidades tiene el decano?
 ¿Cuáles son las tareas asignadas al decano?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué categorías de estudiantes reconoce la Facultad?
 ¿Qué tipos de estudiantes identifica la Facultad?
 ¿Qué clases de estudiantes existen en la Universidad?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son las formas de realizar la docencia?
 ¿Qué métodos existen para ejercer la docencia?
 ¿Qué maneras hay de impartir la docencia?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿De qué consta la docencia regular?
 ¿Qué implica la docencia regular?
 ¿Qué cosas están en la docencia regular?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿De qué consta la docencia libre?
 ¿Qué elementos forman parte de la docencia libre?
 ¿Qué implica la docencia libre?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿De qué manera se fomenta el desarrollo de la investigación?
 ¿Cómo se promueve el avance de la investigación?
 ¿Qué medidas se toman para impulsar la investigación?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuál es el fondo universitario?
 ¿Qué es el fondo universitario?
 ¿En qué consiste el fondo universitario?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué cargos son incompatibles?
 ¿Qué funciones son incompatibles?
 ¿Qué cargos no se pueden combinar?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son las medidas ante la ausencia de un docente?
 ¿Qué acciones se toman cuando falta un docente?
 ¿Qué se hace ante la ausencia de un profesor?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuál es el objetivo de la tesis de cada alumno?
 ¿Qué propósito tiene la tesis de cada estudiante?
 ¿Para qué sirve la tesis?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué es la tesis del estudiante?
 ¿En qué consiste la tesis del estudiante?
 ¿Qué comprende la tesis?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuál es el plazo para la presentación de la tesis?
 ¿Hasta cuándo se puede presentar la tesis?
 ¿Cuál es la fecha límite para entregar la tesis?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué obligaciones tienen los padres o tutores de los alumnos?
 ¿Qué responsabilidades tienen los padres o tutores de los alumnos?
 ¿Qué deberes deben cumplir los padres o tutores de los alumnos?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué derechos tienen los padres o tutores de los alumnos?
 ¿Qué privilegios tienen los padres o tutores de los alumnos?
 ¿Qué derechos tienen reconocidos los padres o tutores?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué deberes tienen los trabajadores de la educación?
 ¿Qué tienen que hacer los que trabajan en la educación?
 ¿Qué responsabilidades tienen los profesionales de la educación?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son los objetivos de la educación especial?
 ¿Qué metas tiene la educación especial?
 ¿Qué metas persigue la enseñanza especial?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son las funciones de las universidades?
 ¿Qué hacen las universidades?
 ¿Qué responsabilidades tienen las instituciones universitarias?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son los objetivos de la educación de adultos?
 ¿Qué metas tiene la educación para adultos?
 ¿Qué fines persigue la educación para personas mayores?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué tiene la formación docente como objetivos?
 ¿Para qué sirve la formación docente?
 ¿Qué metas busca la formación de los profes?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son los derechos de los educandos?
 ¿Qué derechos tienen los estudiantes?
 ¿Qué derechos les corresponden a los que estudian?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué se verifica en la evaluación de la calidad en el sistema educativo?
 ¿Qué se analiza cuando se evalúa la calidad en el sistema educativo?
 ¿Qué se examina en la evaluación de la calidad en las instituciones educativas?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué tiene como objetivos la educación de posgrado?
 ¿Qué fin tiene el posgrado?
 ¿Qué metas persigue la formación de posgrado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son los deberes de los trabajadores de la educación?
 ¿Qué deben hacer los educadores?
 ¿Qué obligaciones tienen los trabajadores del sector educativo?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué verificará la evaluación de la calidad en el sistema educativo?
 ¿Qué va a revisar la evaluación de la calidad educativa?
 ¿Qué revisará la evaluación de la calidad en el ámbito educativo?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué deberá hacer el Poder Ejecutivo nacional, a través del ministerio específico?
 ¿Qué va a hacer el Poder Ejecutivo nacional por medio del ministerio adecuado?
 ¿Qué acciones deberá realizar el Poder Ejecutivo nacional mediante el ministerio correspondiente?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuál es la misión del Consejo Federal de Cultura y Educación?
 ¿Cuál es el propósito del Consejo Federal de Cultura y Educación?
 ¿Para qué está el Consejo Federal de Cultura y Educación?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son las funciones del Consejo Federal de Cultura y Educación?
 ¿Qué tareas tiene el Consejo Federal de Cultura y Educación?
 ¿Qué responsabilidades tiene el Consejo Federal de Cultura y Educación?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son los órganos que componen el Consejo Federal de Cultura y Educación?
 ¿Qué partes forman el Consejo Federal de Cultura y Educación?
 ¿Qué organismos están dentro del Consejo Federal de Cultura y Educación?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿De qué dos Consejos Consultivos recibirá apoyo el Consejo Federal de Cultura y Educación?
 ¿Qué dos Consejos Consultivos van a apoyar al Consejo Federal de Cultura y Educación?
 ¿Qué dos Consejos Consultivos proporcionarán respaldo al Consejo Federal de Cultura y Educación?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué atribuciones tienen las autoridades competentes de las provincias y de la municipalidad de la Ciudad de Buenos Aires?
 ¿Qué poderes tienen las autoridades provinciales y de Buenos Aires?
 ¿Qué facultades poseen las autoridades de las provincias y de la municipalidad de Buenos Aires?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Con qué finalidad financia el Poder Ejecutivo nacional los programas especiales de desarrollo educativo?
 ¿Para qué financia el Poder Ejecutivo nacional los programas especiales de desarrollo educativo?
 ¿Qué busca el Poder Ejecutivo nacional al financiar los programas educativos especiales?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué acordaron el Ministerio de Cultura y Educación y las autoridades educativas de las provincias y de la Municipalidad de la Ciudad de Buenos Aires en el seno del Consejo Federal de Cultura y Educación?
 ¿Qué decidieron el Ministerio de Cultura y Educación y las autoridades provinciales y municipales con el Consejo Federal de Cultura y Educación?
 ¿Qué pactaron el Ministerio de Cultura y Educación y las autoridades provinciales y de la Ciudad de Buenos Aires dentro del Consejo Federal de Cultura y Educación?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿El presupuesto de la administración pública nacional 1993 será inferior a cuál presupuesto?
 ¿El presupuesto nacional de 1993 será menor que qué otro presupuesto?
 ¿A qué presupuesto será inferior el presupuesto de la administración pública nacional 1993?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Las disposiciones de esta ley son aplicables a todos los niveles y regímenes especiales educativos con excepción de cuáles artículos?
 ¿Qué artículos no se aplican de esta ley en todos los niveles y regímenes educativos especiales?
 ¿Qué artículos están excluidos de la aplicación de esta ley en todos los niveles y regímenes especiales educativos?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué derechos tienen los padres o tutores de los alumnos/as?
 ¿Cuáles son los derechos de los padres o encargados de los estudiantes?
 ¿Qué prerrogativas poseen los padres o tutores de los alumnos?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué obligaciones tienen los padres o tutores de los alumnos/as?
 ¿Qué responsabilidades tienen los padres o tutores de los estudiantes?
 ¿Cuáles son las obligaciones de los padres o tutores de los estudiantes?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles derechos de todos los trabajadores/as de la educación del ámbito estatal y privado se resguardarán?
 ¿Qué derechos se protegerán para todos los trabajadores/as de la educación estatal y privada?
 ¿Qué derechos se protegerán para todos los empleados de la educación en el sector público y privado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son los deberes de los trabajadores de la educación?
 ¿Qué deberes deben cumplir los profesionales de la educación?
 ¿Qué obligaciones tienen los empleados del sector educativo?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué deberán garantizar el Ministerio de Cultura y Educación de la Nación, las provincias y la Municipalidad de la Ciudad de Buenos Aires?
 ¿Qué deben asegurar el Ministerio de Cultura y Educación de la Nación, las provincias y la municipalidad?
 ¿Qué compromisos deben cumplir el Ministerio de Cultura y Educación de la Nación, las provincias y la municipalidad?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son las funciones del responsable?
 ¿Cuáles son las obligaciones del responsable?
 ¿Qué tareas realiza el encargado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué confeccionará para cada carrera de cada sede y/o subsede el Departamento Personal de la Facultad?
 ¿Qué elaborará el Departamento Personal de la Facultad para cada carrera en cada sede y subsede?
 ¿Qué creará el Departamento Personal de la Facultad para cada carrera en cada sede o subsede?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cómo será la emisión del voto?
 ¿Cómo se hará el voto?
 ¿De qué forma se llevará a cabo la votación?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Durante cuántos días será la emisión de votos?
 ¿Cuántos días durará el periodo de votación?
 ¿Durante cuántos días se podrá votar?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Que deberá presentar para la emisión del voto el sufragante?
¿Qué debe presentar el votante para emitir su voto?
¿Qué documentación es necesaria para que el elector pueda votar?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles serán las funciones de los consejos de carrera?
 ¿Cuáles serán las responsabilidades de los consejos de carrera?
 ¿Qué atribuciones tendrán los consejos de carrera?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuántos años durarán en sus funciones los representantes del Consejo de carrera?
 ¿Cuántos años permanecerán en su cargo los representantes del Consejo de carrera?
 ¿Cuál será la duración del mandato de los representantes del Consejo de carrera?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son los objetivos de la Educación Especial?
 ¿Cuáles son las metas de la Educación Especial?
 ¿Qué fines persigue la Educación Especial?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son los objetivos de la Educación de Adultos?
 ¿Cuáles son las metas de la Educación de Adultos?
 ¿Qué propósitos tiene la Educación de Adultos?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son las funciones de las autoridades educativas oficiales?
¿Cuáles son las responsabilidades de las autoridades educativas oficiales?
¿Qué tareas desempeñan las autoridades educativas oficiales?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son las obligaciones del Estado nacional, las provincias y la Municipalidad de la Ciudad de Buenos Aires?
¿Cuáles son las responsabilidades del Estado nacional, las provincias y la Municipalidad de la Ciudad de Buenos Aires?
¿Qué deberes tienen el Estado nacional, las provincias y la Municipalidad de la Ciudad de Buenos Aires?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son los derechos de los educandos?
¿Cuáles son los derechos de los estudiantes?
¿Qué derechos tienen los alumnos según el reglamento?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son los objetivos de la formación docente?
¿Cuáles son las metas de la capacitación docente?
¿Qué propósitos tiene la formación docente?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuál es el plazo sugerido para la presentación del trabajo a partir de la presentación de la propuesta?
¿Cuál es el tiempo recomendado para entregar el trabajo desde la entrega de la propuesta?
¿Cuál es el periodo sugerido para la entrega del trabajo luego de presentar la propuesta?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Quiénes son los integrantes del jurado de la tesis?
¿Quiénes conforman el comité evaluador de la tesis?
¿Quiénes forman parte del panel de evaluación de la tesis?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Por dónde debe ser avisada la fecha de presentación de la tesis de licenciatura?
¿A través de qué medio se debe notificar la fecha de defensa de la tesis de licenciatura?
¿Dónde debe ser comunicada la fecha para la presentación de la tesis de licenciatura?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Por quién será establecido el grado académico o profesional mínimo de los asesores de tesis?
¿Quién determinará el nivel académico o profesional mínimo de los asesores de tesis?
¿Quién fijará el título académico o profesional mínimo requerido para los asesores de tesis?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Por quién será establecido el nombre y contenidos específicos de cada capítulo?
¿Quién definirá el título y el contenido detallado de cada sección?
¿Quién determinará el nombre y el contenido particular de cada apartado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Quién es la responsable de administrar el proceso de tesis para los alumnos inscritos en las asignaturas pertinentes y/o candidatos al grado en proceso de titulación?
¿Quién se encarga de gestionar el procedimiento de tesis para los estudiantes matriculados en las materias correspondientes y/o aspirantes al título en proceso de graduación?
¿Quién es responsable de coordinar el trámite de tesis para los alumnos inscritos en las asignaturas aplicables y/o postulantes al grado en trámite de titulación?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Quiénes pueden ejercer la docencia libre?
¿Quiénes están habilitados para impartir clases libres?
¿Quiénes tienen la facultad de enseñar en modalidad libre?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son las funciones del consejo de investigadores?
¿Cuáles son las responsabilidades del comité de investigadores?
¿Qué tareas tiene asignadas el consejo de investigación?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué se requiere para ser electo Responsable de Carrera?
¿Qué requisitos son necesarios para ser elegido Coordinador de Carrera?
¿Qué se necesita para ser seleccionado como Director de Carrera?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son los bienes de la universidad?
¿Cuáles son los activos de la institución?
¿Qué propiedades tiene la universidad?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué constituye el fondo universitario?
¿Qué compone el patrimonio universitario?
¿Qué elementos forman el fondo de la institución?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué se requiere para ser electo como representante docente en el consejo de carrera?
¿Qué requisitos se necesitan para ser elegido como representante de los profesores en el consejo de carrera?
¿Qué condiciones deben cumplirse para ser designado como representante docente en el comité de carrera?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Para qué será utilizado el fondo universitario?
¿Para qué se empleará el capital universitario?
¿Con qué fines se usará el patrimonio universitario?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Para qué no puede ser utilizado el fondo universitario? 
¿Para qué fines está prohibido usar el capital universitario? 
¿En qué no se puede emplear el patrimonio universitario? 
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿En qué consiste la docencia libre? 
¿En qué se basa la enseñanza libre? 
¿En qué consiste el modelo de docencia no reglada? 
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Quienes podrán elegir a los docentes integrantes del consejo de carrera? 
¿Quiénes tienen la facultad de seleccionar a los profesores del comité de carrera? 
¿Quiénes podrán designar a los docentes miembros del consejo de carrera? 
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿A qué le presta preferente atención la universidad? 
¿A qué aspectos da mayor prioridad la universidad? 
¿En qué áreas se enfoca principalmente la universidad? 
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué se requiere para ser consejero estudiante? 
¿Qué se necesita para ser representante estudiantil? 
¿Qué condiciones deben cumplirse para ser miembro del consejo estudiantil? 
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué fomenta la universidad como centro de creación de conocimientos? 
¿Qué promueve la universidad como institución generadora de saberes? 
¿Qué impulsa la universidad en su rol de generadora de conocimiento? 
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué recibirá el Responsable de carrera electo como titular? 
¿Qué obtendrá el Responsable de carrera designado como titular? 
¿Qué le será otorgado al Responsable de carrera elegido como titular? 
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué sucede con un docente que falta a la cuarta parte de sus clases programadas por el horario oficial sin justificación? 
¿Qué ocurre con un profesor que se ausenta en la cuarta parte de sus clases establecidas por el horario oficial sin causa? 
¿Qué medidas se toman con un catedrático que no asiste a la cuarta parte de sus clases fijadas por el horario oficial sin razón? 
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuál es la norma General? 
¿Cuál es el reglamento General? 
¿Cuál es la directriz General? 
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué propondrá cada Unidad Académica mediante Resolución del Consejo Directivo al Consejo Superior? 
¿Qué sugerirá cada Unidad Académica mediante Acuerdo del Consejo Directivo al Consejo Superior? 
¿Qué recomendará cada Unidad Académica mediante Dictamen del Consejo Directivo al Consejo Superior? 
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿De quién estará a cargo la difusión del llamado concurso? 
¿Quién se encargará de la divulgación del mencionado concurso? 
¿Quién tendrá la responsabilidad de la promoción del referido concurso? 
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Dentro de cuántos días podrá publicar el Rectorado una vez aprobado el llamado a concurso? 
¿En cuánto tiempo podrá el Rectorado hacer pública la convocatoria tras su aprobación? 
¿En qué plazo podrá el Rectorado anunciar el concurso una vez que haya sido aprobado? 
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué contendrán sintéticamente los anuncios? 
¿Qué incluirán de manera resumida los comunicados? 
¿Qué detalles serán presentados de forma concisa en los avisos? 
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son las condiciones de los aspirantes? 
¿Cuáles son los requisitos de los candidatos? 
¿Qué condiciones deben cumplir los postulantes? 
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuál es el plazo de inscripción?
¿Cuánto tiempo tengo para inscribirme?
¿Cuándo finaliza el periodo de inscripción?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Dónde es la oficina de inscripción?
¿En qué lugar se encuentra la oficina de inscripción?
¿Dónde está ubicada la oficina de inscripción?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son las solicitudes de inscripción? 
¿Qué tipos de solicitudes de inscripción existen? 
¿Cuáles son las diferentes solicitudes de inscripción? 
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuál es el planeamiento de la cátedra?
¿En qué consiste el planeamiento de la cátedra?
¿Qué es el planeamiento de la cátedra?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿En qué consiste la presentación irregular y tardía?
¿Qué significa una presentación irregular y tardía?
¿Cómo se define una presentación irregular y tardía?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué sucede transcurridos 3 meses del cierre de la inscripción?
¿Qué pasa después de tres meses del cierre de la inscripción?
¿Cuáles son las consecuencias tres meses después del cierre de la inscripción?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿De qué se trata el apoderado?
¿Qué es un apoderado?
¿Cuál es el rol de un apoderado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿De qué se tratan las inscripciones múltiples?
¿Qué significa una inscripción múltiple?
¿Cómo se realizan las inscripciones múltiples?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuándo es el cierre de inscripción?
¿En qué fecha se cierra la inscripción?
¿Cuándo finaliza el proceso de inscripción?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿De qué se trata la exhibición de la nómina de aspirantes? 
¿Qué significa la exhibición de la nómina de aspirantes? 
¿En qué consiste la exhibición de la nómina de aspirantes? 
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿De qué se trata la impugnación de los aspirantes? 
¿Qué significa impugnar a un aspirante? 
¿Cómo se realiza la impugnación de un aspirante? 
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué suspenderá la interposición de la impugnación?
¿Qué detiene la interposición de la impugnación?
¿Cuáles son los efectos de la interposición de una impugnación?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cómo será la designación del jurado?
¿Qué proceso se sigue para la designación del jurado?
¿Cómo se lleva a cabo la designación del jurado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuál es la composición del jurado?
¿De quiénes está compuesto el jurado?
¿Cómo se integra el jurado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué condiciones deberán reunir los integrantes del jurado?
¿Qué requisitos deben cumplir los miembros del jurado?
¿Bajo qué condiciones se seleccionan los integrantes del jurado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué se deberá notificar luego de ser designados los jurados?
¿Qué información se debe comunicar tras la designación del jurado?
¿Qué se debe anunciar después de que se designen los jurados?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son los derechos de los estudiantes de las instituciones estatales de educación superior?
¿Qué derechos tienen los estudiantes en las instituciones públicas de educación superior?
¿Qué beneficios poseen los estudiantes en las instituciones estatales de educación superior?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuál es el requisito principal para acceder a la educación de grado?
¿Qué condición es necesaria para acceder a la educación de grado?
¿Qué se necesita para ingresar a la educación de grado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son los objetivos de la educación superior?
¿Qué metas tiene la educación superior?
¿Para qué fines se orienta la educación superior?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué proporciona la educación superior?
¿Qué beneficios otorga la educación superior?
¿Qué ofrece la educación superior?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué es responsable de proveer el Estado Nacional?
¿Qué debe proporcionar el Estado Nacional?
¿Qué tiene la obligación de proveer el Estado Nacional?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Qué se debe acreditar para ser alumno auxiliar?
¿Qué es necesario demostrar para ser alumno auxiliar?
¿Qué condiciones se deben acreditar para ser alumno auxiliar?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Se contarán días no hábiles en los términos del Estatuto?
¿Los días no hábiles se computarán en los plazos del Estatuto?
¿Se incluirán días no hábiles en los términos del Estatuto?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Quiénes pueden suspender a los docentes?
¿Quién tiene la autoridad para suspender a los docentes?
¿Quién está facultado para suspender a los docentes?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que forma parte del fondo Universitario?
¿Qué elementos constituyen el fondo Universitario?
¿Qué integra el patrimonio de la Universidad?



Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como se lleva a cabo la docencia libre?
¿Cómo se implementa la enseñanza libre?
¿De qué manera se realiza la docencia sin restricciones?

Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Quienes podran ejercer la docencia libre?
¿A quiénes se les permite desempeñar la docencia libre?
¿Quiénes están autorizados para ejercer docencia libre?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cuáles son las categorías de adscriptos para cátedras, proyectos de extensión y proyectos de investigación?
¿Cuáles son las clasificaciones de adscriptos para asignaturas, programas de extensión y proyectos de investigación?
¿Qué tipos de adscripciones están previstas para cátedras, programas de extensión y proyectos de investigación?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Quien podra publicar el trabajo final de un alumno?
¿Quién está habilitado para hacer público el informe final de un alumno?
¿Quién puede realizar la publicación del trabajo final de un estudiante?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como se evalua el trabajo final?  
¿Cómo se califica el proyecto final?  
¿Cómo se determina la nota del trabajo final?   
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que cualidades debe tener la tesis?
¿Qué características debe presentar la tesis?
¿Cuáles son los requisitos que debe cumplir la tesis?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Puede realizarse un cambio de director de tesis?
¿Está permitido realizar un cambio de director de tesis?
¿Se autoriza un cambio del director de tesis?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]En que momento puede realizarse correcciones al contenido del proyecto?
¿Cuándo se pueden realizar cambios en el contenido del proyecto?
¿En qué momento es posible hacer correcciones al contenido del trabajo?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]En que se basa la evaluacion de la propuesta de tesis?
¿En qué se fundamenta el análisis del plan de tesis?
¿Qué aspectos se consideran en la revisión del esquema de tesis?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]En que consiste la aprobación de la propuesta de tesis?
¿Cómo se lleva a cabo la validación de la propuesta de tesis?
¿Cómo se obtiene la aprobación de la propuesta de tesis?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que sucede cuando el alumno presenta en su tesis tecnologías totalmente nuevas?
¿Qué ocurre si el estudiante introduce en su tesis innovaciones tecnológicas completamente inéditas?
¿Qué pasa si el estudiante propone en su tesis tecnologías nunca antes vistas?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cómo se elige al director de la tesis?
¿Cómo se determina al supervisor del proyecto de tesis?
¿Cómo se decide quién será el director del trabajo final?


Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Quien propone la codireccion de la tesis?
¿Quién es el encargado de proponer la codirección de la tesis?
¿Quién tiene la responsabilidad de proponer la codirección de la tesis?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cuales son las normas para la presentación de la tesis?
¿Qué directrices deben seguirse para la presentación de una tesis?
¿Qué reglamentos se aplican para la presentación de tesis?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]¿Cual es el plazo de presentacion de la tesis?
¿Cuál es el período límite para presentar la tesis?
¿Cuándo vence el plazo para la presentación de la tesis?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cual es el plazo de entrega de la tesis antes de la presentacion?
¿Cuál es el tiempo límite para entregar el trabajo final antes de la defensa?
¿Qué plazo se tiene para la entrega de la tesis final antes de la presentación oficial?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como es la emisión de cualquier tipo de voto dentro de la Universidad?
¿Cómo se lleva a cabo la votación de cualquier clase dentro de la Universidad?
¿Cuál es el procedimiento para la emisión de votos en la Universidad?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que sucede si hay empate en las elecciones de responsables de carrera?
¿Qué ocurre en caso de empate durante la votación para cargos de dirección de carrera?
¿Cómo se resuelve un empate en las elecciones para responsables de carrera?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Quien responde ante un aspecto no previsto en el reglamento?
¿Quién se encarga de los casos no contemplados en el reglamento?
¿Quién debe intervenir en los casos no previstos por el reglamento?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]De que consejos tiene apoyo el consejo federal de cultura y educación?
¿De qué organismos recibe apoyo el consejo federal de cultura y educación?
¿Qué tipos de consejos proveen respaldo al consejo federal de cultura y educación?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Quienes integran la comunidad educativa?
¿Quiénes conforman la comunidad educativa? 
¿Quiénes son los integrantes de la comunidad académica?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]A que se obligan el estado nacional, las provincias y la municipalidad de Buenos Aires?
¿Cuáles son las responsabilidades que asumen el gobierno nacional, las provincias y la municipalidad de Buenos Aires?
¿Qué es lo que el Estado, las provincias y Buenos Aires tienen que cumplir?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]A que tienen derechos los docentes de las instituciones de gestión privada?
¿Cuáles son los derechos que poseen los profesores en las entidades educativas de gestión privada?
¿Qué derechos tienen los profesores en las instituciones privadas?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que promueven las autoridades educativas oficiales?
¿Qué fomentan las autoridades educativas oficiales?
¿A qué incentivan las autoridades universitarias?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Bajo que responsabilidad está la educación de posgrado?
¿Quién se encarga de la supervisión de la formación de posgrado?
¿A quién le corresponde la gestión de los cursos de posgrado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que tiene como objetivos la formación docente?
¿Cuáles son los fines de la capacitación de profesores?
¿Qué se pretende lograr con la preparación de los profesores?


Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cuales son los objetivos del ciclo polimodal?
¿Cuáles son las metas del ciclo polimodal?
¿Qué propósitos tiene el ciclo polimodal?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que tiene como objetivos la educación inicial basica?
¿Qué pretende lograr la enseñanza elemental?
¿A qué objetivos apunta la instrucción inicial básica?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como ha de ser el sistema educativo?
¿Cuál es el diseño que debe tener el sistema académico?
¿Cómo se debería estructurar el sistema formativo?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que debe asegurar el sistema educativo a todos los habitantes del pais?
¿Qué debe garantizar el sistema académico a todos los ciudadanos del país?
¿Qué debe proporcionar el sistema educativo a todos los residentes del país?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Qué derechos deberá respetar el estado nacional?
¿Qué derechos tiene que proteger la administración nacional?
¿Qué garantías debe cumplir el estado nacional?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]De quien son responsabilidad las acciones educativas?
¿Quién tiene que hacerse cargo de las actividades académicas?
¿Quién asume la responsabilidad de las actividades educativas?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Qué responsabilidad tiene el estado nacional?
¿Cuál es la obligación del estado nacional?
¿De qué se encarga el gobierno nacional?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que funciones tiene el consejo de carreras?
¿Qué tareas realiza el comité de carreras?
¿Qué se encarga de hacer el consejo de carreras?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Qué pasa en caso de empate?  
¿Qué ocurre si hay un empate?
¿Qué se hace si se produce un empate?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cuando se realiza el escrutinio?
¿Cuándo se lleva a cabo el escrutinio?
¿Cuándo se realiza el conteo de votos?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Qué debe hacerse para ser oficializada una postulación?
¿Cómo se oficializa una postulación?
¿De qué forma se hace oficial una postulación?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como es la emisión de voto?  
¿Cuál es el procedimiento para emitir un voto? 
¿Cómo se lleva a cabo la emisión de votos? 
¿Cómo se emite un voto?  
¿Qué tenés que hacer para votar?  
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como se constituye la junta electoral?  
¿Cómo se conforma la junta electoral?  
¿Quiénes integran la junta electoral?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como es la elección de responsables?
¿Cómo se realiza la designación de encargados?
¿Cómo se eligen a los responsables?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cuales son las funciones del responsable?
¿Cuáles son las responsabilidades del encargado?
¿Qué tareas debe cumplir el responsable?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como es la actividad del responsable suplente?
¿Qué acciones debe llevar a cabo el responsable suplente?
¿Qué tiene que hacer el suplente?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cuanto tiempo duran los responsables en sus funciones?
¿Cuánto tiempo se mantiene un responsable en su cargo?
¿Durante cuánto tiempo los responsables desempeñan sus funciones?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que se requiere para ser electo como responsable?
¿Cuáles son las exigencias para ser elegido como responsable? 
¿Que tenés que cumplir para ser nombrado como encargado?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cuantos responsables hay por cada carrera? 
¿Cuál es la cantidad de responsables asignados a cada carrera?  
¿Cuántos responsables están a cargo de cada carrera?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Con cuanta antelación se presenta la tesis?
¿Con cuánta antelación es necesario presentar la tesis?
¿Cuánto tiempo de anticipación se necesita para la presentación de la tesis?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Qué normas se deben tener en cuenta para la presentación de tesis?
¿Qué requisitos se deben cumplir para la presentación de una tesis?
¿Qué cosas debes tener en cuenta para entregar tu tesis?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que funciones tiene la secretaría académica?
¿Qué tareas realiza la secretaría académica?
¿Qué hace la secretaría de académica?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Quienes pueden fungir como asesores de tesis?
¿Quiénes están autorizados para actuar como asesores de tesis?
¿Quiénes pueden ser los asesores de tesis?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Qué tiene como responsabilidad el director de tesis?
¿Qué funciones debe cumplir el director del proyecto final?
¿Cuál es el rol del director de tesis?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cuales son las funciones del director de tesis?
¿Qué tareas realiza el director del proyecto final?
¿Qué hace el profesor que dirige la tesis?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que pasa cuando el alumno decide cambiar de tema?
¿Qué sucede cuando el estudiante opta por modificar el tema?
¿Qué pasa si decides cambiar de tema?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Qué produce un estudiante que elabora una tesis en la universidad Autónoma de Entre Ríos?
¿Qué consecuencia tiene para un estudiante la realización de una tesis en la Universidad Autónoma de Entre Ríos?
¿Qué produce un alumno que prepara una tesis en la Universidad Autónoma de Entre Ríos?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Qué se observa para la aprobación de la propuesta de tesis?
¿Qué aspectos se evalúan para la aprobación de la propuesta del proyecto de tesis?
¿Qué detalles revisan para que tu propuesta de tesis sea aprobada?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Quien establece el número máximo de integrantes para la presentación de la tesis?  
Quién determina la cantidad límite de personas para la defensa de la tesis?
Quién estipula cuántos alumnos pueden presentar juntos una tesis?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cuales son los tipos principales de tesis?  
¿Cuáles son las categorías principales de tesis?
¿Qué tipos de tesis existen?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que tiene por finalidad la tesis?
¿Cuál es el objetivo de la tesis?
¿Para qué se hace la tesis?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Con quien se acuerda el plazo de entrega del trabajo final?  
¿Con quién se coordina la fecha límite para la entrega del proyecto final?
¿Quién decide el plazo de entrega del trabajo final?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Quien resuelve la incorporación de adscriptos a una cátedra o proyecto?
¿Quién determina la aceptación de ayudantes en una materia o proyecto?
¿Quién decide si alguien puede ser adscripto en una materia o proyecto?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Qué constituye la condición de adscripto?
¿Qué define la condición de adscripto?
¿Cuáles son las características que determinan la condición de adscripto?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que puede hacer el consejo directivo?
¿Qué funciones tiene el consejo directivo?
¿Cuáles son las responsabilidades del consejo directivo?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que tienen a cargo el rector, decano y los directores?
¿Qué responsabilidades tienen el rector, el decano y los directores?
¿Cuáles son las funciones del rector, el decano y los directores?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que se entiende por bienes de la universidad?
¿Qué se considera como bienes de la universidad?
¿Cuál es la definición de bienes de la universidad?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que es lo que fomenta la universidad?
¿Qué promueve la universidad?
¿Cuáles son los objetivos que impulsa la universidad?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que tiene como funciones el consejo de investigadores?
¿Qué responsabilidades tiene el consejo de investigadores?
¿Cuáles son las funciones asignadas al consejo de investigadores?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que condiciones de realización de docencia reconoce la universidad?
¿Qué requisitos establece la universidad para la realización de docencia?
¿Cuáles son las condiciones que la universidad reconoce para la docencia?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Quien puede ser alumno vocacional?
¿Quién está habilitado para ser alumno vocacional?
¿Qué personas pueden acceder a la condición de alumno vocacional?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cual es el objetivo de instituir becas?
¿Cuál es el propósito de otorgar becas?
¿Para qué se instituyen las becas?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que tienen por funciones los profesores titulares y asociados?
¿Qué responsabilidades tienen los profesores titulares y asociados?
¿Cuáles son las funciones de los profesores titulares y asociados?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cual es la definicion del monto de inversion publica en educacion?
¿Cuál es la descripción del monto de inversión pública en educación?
¿Cómo se define el monto de inversión pública en educación?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]A que se dedica El Consejo Federal de Cultura y Educación?
¿A qué se enfoca el Consejo Federal de Cultura y Educación?
¿Cuál es la misión del Consejo Federal de Cultura y Educación?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que debera realizar el Porder Ejecutivo Nacional?
¿Qué acciones debe llevar a cabo el Poder Ejecutivo Nacional?
¿Cuáles son las tareas que deberá ejecutar el Poder Ejecutivo Nacional?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]De que se deben encargar los padres o tutores de los alumnos/as?
¿De qué son responsables los padres o tutores de los alumnos/as?
¿Qué obligaciones tienen los padres o tutores de los alumnos/as?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que proposito tiene la universidad?
¿Cuál es la finalidad de la universidad?
¿Qué objetivo persigue la universidad?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que fin tiene la Educacion Inicial?
¿Cuál es el propósito de la Educación Inicial?
¿Qué objetivo tiene la Educación Inicial?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que nos asegura el sistema educativo a todos los habitantes del pais?
¿Qué garantiza el sistema educativo a todos los habitantes del país?
¿Cuáles son las garantías que ofrece el sistema educativo a los habitantes del país?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como se conforma la estructura del sistema educativo?
¿Cómo está organizada la estructura del sistema educativo?
¿Cuál es la composición de la estructura del sistema educativo?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Quien controla el cumplimienot de la politica educativa?
¿Quién supervisa el cumplimiento de la política educativa?
¿Quién está a cargo de controlar el cumplimiento de la política educativa?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como se elige a los docentes integrantes del consejo de carreras?
¿Cómo se selecciona a los docentes que integran el consejo de carreras?
¿Cuál es el proceso de elección de los docentes del consejo de carreras?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]De que manera se oficializa una postulacion?
¿Cómo se formaliza una postulación?
¿Cuál es el proceso para oficializar una postulación?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Quienes seran electores?
¿Quiénes tendrán derecho a voto?
¿Quiénes podrán ser electores?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como se elige el Responsable de carrera?
¿Cómo se selecciona al Responsable de carrera?
¿Cuál es el proceso para elegir al Responsable de carrera?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cuales son los requisitos para ser electo Responsable de Carrera?
¿Qué condiciones se requieren para ser electo Responsable de Carrera?
¿Cuáles son las exigencias para ser elegido Responsable de Carrera?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Quien administra el proceso de la tesis?
¿Quién gestiona el proceso de la tesis?
¿Quién está a cargo del proceso de la tesis?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que funcion cumplen los asesores de tesis?
¿Qué rol desempeñan los asesores de tesis?
¿Cuál es la función de los asesores de tesis?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como se realizan avances y modificaciones en el proyecto original de investigacion y tesis?
¿Cómo se efectúan avances y modificaciones en el proyecto original de investigación y tesis?
¿Qué pasos se siguen para hacer avances y modificaciones en el proyecto de investigación y tesis?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Puede el alumno cambiar el tema de su tesis?
¿Puede el estudiante modificar el tema de su tesis?
¿Es posible que el alumno cambie el tema de su tesis?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que tipo de tesis pueden desarrollar los alumnos?
¿Qué clase de tesis pueden elaborar los alumnos?
¿Cuáles son los tipos de tesis que los alumnos pueden desarrollar?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cual es el objetivo de la tesis?
¿Cuál es la finalidad de la tesis?
¿Qué propósito tiene la tesis?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que sanciones pueden sufrir los Estudiantes de la Universidad por causas de inconducta?
¿Qué castigos pueden recibir los estudiantes de la universidad por mal comportamiento?
¿Qué consecuencias enfrentan los estudiantes de la universidad por mala conducta?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que cargos tienen incompatibilidades en su ejercicio?
¿Qué cargos presentan incompatibilidades en su ejercicio?
¿Qué funciones tienen restricciones en su ejercicio?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Quien nombra y remueve al personal de la Facultas?
¿Quién designa y destituye al personal de la Facultad?
¿Quién tiene la autoridad para nombrar y remover al personal de la Facultad?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Para que se permite utilizar el fondo Universitario?
¿Para qué está permitido usar el fondo universitario?
¿En qué casos se permite la utilización del fondo universitario?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Para que no se permite utilizar el fondo Universitario?
¿Para qué no está permitido utilizar el fondo universitario?
¿En qué casos no se puede utilizar el fondo universitario?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como se constituyen los bienes de la facultad?
¿Cómo se forman los bienes de la facultad?
¿De qué manera se constituyen los bienes de la facultad?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como se debe orientar la enseñanza?
¿Cómo debe ser dirigida la enseñanza?
¿De qué manera se debe orientar la enseñanza?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]A que esta habilitado un alumno vocacional?
¿Para qué está autorizado un alumno vocacional?
¿En qué actividades está habilitado un alumno vocacional?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Aque se debe la entrega becas la Universidad?
¿Por qué la universidad otorga becas?
¿A qué se debe la concesión de becas por parte de la universidad?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que elige el Cuerpo de Estudiantes?
¿Qué selecciona el Cuerpo de Estudiantes?
¿Qué decide el Cuerpo de Estudiantes?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cuales son los requisitos para ser Consejero Estudiante?
¿Cuáles son las condiciones para ser Consejero Estudiante?
¿Qué se necesita para ser Consejero Estudiante?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Quien esta a cargo de propender los fines del Estatuto?
¿Quién es responsable de cumplir con los fines del Estatuto?
¿Quién se encarga de propender los fines del Estatuto?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que funciones tendran los profesores titulares y asociados?
¿Qué roles desempeñarán los profesores titulares y asociados?
¿Qué responsabilidades tendrán los profesores titulares y asociados?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Quien puede gozar de un año de licencia para realizar actividades academicas?
¿Quién puede disfrutar de un año de licencia para realizar actividades académicas?
¿Quién tiene derecho a un año de licencia para actividades académicas?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como se garantiza la actualizacion y perfeccionamiento del cuerpo docente?
¿Cómo se asegura la actualización y perfeccionamiento del cuerpo docente?
¿Qué mecanismos garantizan la actualización y perfeccionamiento del cuerpo docente?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como se elige el Decano y Vicedecano?
¿Cómo se selecciona al Decano y Vicedecano?
¿Qué procedimiento se sigue para elegir al Decano y Vicedecano?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como se procede en caso que ningún candidato alcanzare tal mayoría absoluta de votos?
¿Cómo se actúa si ningún candidato obtiene la mayoría absoluta de votos?
¿Qué se hace si ningún candidato logra la mayoría absoluta de votos?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cuales son obligaciones de la facultad?
¿Cuáles son las responsabilidades de la facultad?
¿Qué deberes tiene la facultad?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cuales son las diferentes modalidades del sistema educativo?
¿Cuáles son los distintos tipos de modalidades del sistema educativo?
¿Qué clases de modalidades existen en el sistema educativo?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como se debe proceder para la creación de nuevas carreras o cambios en las ya existentes?
¿Cómo se debe actuar para la creación de nuevas carreras o modificaciones en las ya existentes?
¿Qué pasos se deben seguir para crear nuevas carreras o modificar las existentes?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cuales son los niveles del Sistema Educativo Provincial?
¿Cuáles son las etapas del Sistema Educativo Provincial?
¿Qué grados componen el Sistema Educativo Provincial?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que carateristicas tiene el Sistema Educativo Provincial?
¿Qué particularidades presenta el Sistema Educativo Provincial?
¿Cuáles son las propiedades del Sistema Educativo Provincial?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Mediante que propuestas el Estado Provincial asegura la obligatoriedad?
¿A través de qué propuestas el Estado Provincial garantiza la obligatoriedad?
¿Qué iniciativas utiliza el Estado Provincial para asegurar la obligatoriedad?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que abarca la obligatoriedad escolar?
¿Qué comprende la obligatoriedad escolar?
¿Hasta dónde llega la obligatoriedad escolar?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Como se finanza el sistema educativo?
¿Cómo se financia el sistema educativo?
¿De qué manera se financia el sistema educativo?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cual es la responsabilidad principal del Estado Provincial?
¿Cuál es el deber principal del Estado Provincial?
¿Cuál es la obligación fundamental del Estado Provincial?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que sucede cuando una carrera no obtiene su acreditacion necesaria?
¿Qué ocurre cuando una carrera no recibe la acreditación necesaria?
¿Cuál es el resultado si una carrera no consigue la acreditación necesaria?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Quienes seran los organos de coordinacion y consulta del Sistema Universitario?
¿Quiénes serán los órganos de coordinación y consulta del Sistema Universitario?
¿Qué entidades actuarán como órganos de coordinación y consulta del Sistema Universitario?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Que atribuciones tiene la autonomia academica e institucional?
¿Qué poderes otorga la autonomía académica e institucional?
¿Cuáles son las competencias de la autonomía académica e institucional?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]Cuales son los deberes de los docentes de las instituciones estatales de educación superior?
¿Cuáles son las obligaciones de los docentes en las instituciones estatales de educación superior?
¿Qué responsabilidades tienen los docentes de las instituciones estatales de educación superior?
Batches:   0%|          | 0/1 [00:00<?, ?it/s]Batches:   0%|          | 0/1 [00:00<?, ?it/s]     idConsulta                                           Consulta  Posicion  \
0             1  Cuales son las condiciones para presentarse a ...         1   
1             2  Quién debe regir la convocatoria para la desig...         1   
2             3  Cómo se publicita un llamado a concurso?\n¿Cóm...         1   
3             4  Qué contenido tienen los anuncios?\n¿Qué infor...         1   
4             5  Cuales son los requisitos para los aspirantes ...         1   
..          ...                                                ...       ...   
295         296  Cual es la responsabilidad principal del Estad...         1   
296         297  Que sucede cuando una carrera no obtiene su ac...         1   
297         298  Quienes seran los organos de coordinacion y co...         1   
298         299  Que atribuciones tiene la autonomia academica ...         1   
299         300  Cuales son los deberes de los docentes de las ...         1   

     Similaridad                        Modelo  
0       0.659720  nomic-ai/nomic-embed-text-v1  
1       0.731598  nomic-ai/nomic-embed-text-v1  
2       0.665036  nomic-ai/nomic-embed-text-v1  
3       0.506481                   BAAI/bge-m3  
4       0.672051  nomic-ai/nomic-embed-text-v1  
..           ...                           ...  
295     0.640375                   BAAI/bge-m3  
296     0.674144  nomic-ai/nomic-embed-text-v1  
297     0.651335                   BAAI/bge-m3  
298     0.711267                   BAAI/bge-m3  
299     0.734754  nomic-ai/nomic-embed-text-v1  

[300 rows x 5 columns]
```


**[Celda 35 - Código]**
```python
#Opcional guardar Dataset - Definir la ruta según la computadora a utilizar
csv_path = r'/kaggle/working/resultados_bge-m3_nomic-embed-text-v1_gpt4-similares_TRES.csv'
resultados_df.to_csv(csv_path, index=False)

```

## Visualización de resultados


**[Celda 37 - Código]**
```python
# Cargar el archivo CSV en un DataFrame, si es un DataSet importado 
csv_path = r'/kaggle/working/resultados_bge-m3_nomic-embed-text-v1_gpt4-similares_TRES.csv'
resultadosCSV_df = pd.read_csv(csv_path)
resultados_df = resultadosCSV_df
print(resultadosCSV_df .head())
```


*Salida:*
```text
idConsulta                                           Consulta  Posicion  \
0           1  Cuales son las condiciones para presentarse a ...         1   
1           2  Quién debe regir la convocatoria para la desig...         1   
2           3  Cómo se publicita un llamado a concurso?\n¿Cóm...         1   
3           4  Qué contenido tienen los anuncios?\n¿Qué infor...         1   
4           5  Cuales son los requisitos para los aspirantes ...         1   

   Similaridad                        Modelo  
0     0.659720  nomic-ai/nomic-embed-text-v1  
1     0.731598  nomic-ai/nomic-embed-text-v1  
2     0.665036  nomic-ai/nomic-embed-text-v1  
3     0.506481                   BAAI/bge-m3  
4     0.672051  nomic-ai/nomic-embed-text-v1
```


**[Celda 38 - Código]**
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

plt.title('Distribución de Posiciones para bge-m3 + nomic-embed-text-v1 3 PREGUNTAS SIMILARES GPT 4o', fontsize=16)
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
1-3: 95.0% (285 preguntas)
4-6: 2.3% (7 preguntas)
7-10: 1.3% (4 preguntas)
No aparece: 1.3% (4 preguntas)
```


**[Celda 39 - Código]**
```python
def mostrar_preguntas_por_posicion(df):
    # Resumen de cantidad por posición
    print("Resumen de cantidad por posición:")
    for posicion in range(1, 11):
        cantidad = df[df['Posicion'] == posicion].shape[0]
        print(f"Posición {posicion}: {cantidad}")
    no_encontradas = df[df['Posicion'] == -1].shape[0]
    print(f"No encontradas: {no_encontradas}")
    
    '''
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
            '''

# Ejecutar la función
mostrar_preguntas_por_posicion(resultados_df)
```


*Salida:*
```text
Resumen de cantidad por posición:
Posición 1: 242
Posición 2: 35
Posición 3: 8
Posición 4: 1
Posición 5: 2
Posición 6: 4
Posición 7: 2
Posición 8: 1
Posición 9: 1
Posición 10: 0
No encontradas: 4
```
