---
aliases: [GIBD2025_ViT-Base_Busqueda-por-Similitud (ejec.cls.10k)]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2025-08-27
origen_zip: GIBD-20260521T205218Z-3-005.zip
ruta_interna: "GIBD/ViT/GIBD2025_ViT-Base_Busqueda-por-Similitud (ejec.cls.10k).ipynb"
tamanio_bytes: 72141235
---

# Notebook: GIBD2025_ViT-Base_Busqueda-por-Similitud (ejec.cls.10k).ipynb

Ruta interna: `GIBD/ViT/GIBD2025_ViT-Base_Busqueda-por-Similitud (ejec.cls.10k).ipynb`

---

**Imports necesarios**


**[Celda 2 - Código]**
```python
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import time
import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
from transformers import ViTImageProcessor, ViTConfig, ViTForImageClassification, ViTModel
from safetensors.torch import load_file
```


**[Celda 3 - Código]**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando:", device)
```


*Salida:*
```text
Usando: cuda
```

**Cargar modelo y feature extractor**


**[Celda 5 - Código]**
```python
"""# ✨ Modelos ViT recomendados (ImageNet-1k; algunos preentrenados en 21k y fine-tuned en 1k)
RECOMMENDED_MODELS = [
    # Muy fuertes (grandes)
    "vit_large_patch16_224.augreg_in21k_ft_in1k",
    "vit_large_patch16_384.augreg_in21k_ft_in1k",
    # Fuertes (base)
    "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "vit_base_patch16_384.augreg_in21k_ft_in1k",
    # Data-efficient / variantes DeiT3 fuertes
    "deit3_large_patch16_224.fb_in1k",
    "deit3_base_patch16_224.fb_in1k",
    # Alternativas populares
    "vit_base_patch16_224.mae",   # pretrain MAE + finetune; a veces rinde muy bien como descriptor
]"""

local_model_path = r"C:\Users\Usuario\Documents\ViT 2025\vit-base-local"

feature_extractor = ViTImageProcessor.from_pretrained(local_model_path)
config = ViTConfig.from_pretrained(local_model_path)

model = ViTModel(config)

state_dict = load_file(os.path.join(local_model_path, "model.safetensors"))
model.load_state_dict(state_dict, strict=False) 

model = model.to(device)
model.eval()
```


*Salida:*
```text
ViTModel(
  (embeddings): ViTEmbeddings(
    (patch_embeddings): ViTPatchEmbeddings(
      (projection): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
    )
    (dropout): Dropout(p=0.0, inplace=False)
  )
  (encoder): ViTEncoder(
    (layer): ModuleList(
      (0-11): 12 x ViTLayer(
        (attention): ViTSdpaAttention(
          (attention): ViTSdpaSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (output): ViTSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
        )
        (intermediate): ViTIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): ViTOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (dropout): Dropout(p=0.0, inplace=False)
        )
        (layernorm_before): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        (layernorm_after): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      )
    )
  )
  (layernorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
  (pooler): ViTPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
  )
)
```

**Función para extraer embedding CLS**


**[Celda 7 - Código]**
```python
def extract_embedding(image_path, mode, pooling, layers, device):
    """
    Extrae un embedding de la imagen según el modo elegido.
    - mode="cls": usa el token CLS (CASO1).
    - mode="pool": pooling sobre los patches (CASO2).
        pooling = "max" o "mean"
    - mode="layer": promedio de CLS de varias capas (CASO3).
        layers = lista de índices de capas (ej: [5, 10])
    """
    # Preparar imagen
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        # Pedimos todos los hidden states
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tupla: [layer0, layer1, ..., layerL]
        last_hidden_state = hidden_states[-1]  # última capa

    if mode == "cls":  # CASO1
        emb = last_hidden_state[:, 0, :]  # token CLS final

    elif mode == "pool":  # CASO2
        token_embeddings = last_hidden_state[:, 1:, :]  # sin CLS
        if pooling == "mean":
            emb = token_embeddings.mean(dim=1)
        elif pooling == "max":
            emb = token_embeddings.max(dim=1).values
        else:
            raise ValueError("pooling debe ser 'mean' o 'max'")

    elif mode == "layer":  # CASO3
        if layers is None or len(layers) == 0:
            raise ValueError("En CASO3 (mode='layer') debes especificar una lista en 'layers'")
        # Tomamos CLS de las capas elegidas
        embs = [hidden_states[l][:, 0, :] for l in layers]
        emb = torch.stack(embs).mean(dim=0)  # promedio

    else:
        raise ValueError("mode debe ser 'cls', 'pool' o 'layer'")

    return emb.squeeze().cpu().numpy()
```

**Construir índice de embeddings de la base**


**[Celda 9 - Código]**
```python
def build_embeddings_index(folder_path, mode, pooling, layers, device):
    """Genera una matriz de embeddings y lista de nombres para la base."""
    embeddings, nombres = [], []
    for nombre in tqdm(os.listdir(folder_path)):
        ruta = os.path.join(folder_path, nombre)
        if ruta.lower().endswith(('.jpg', '.jpeg', '.png')):
            emb = extract_embedding(ruta, mode=mode, pooling=pooling, layers=layers, device=device)
            embeddings.append(emb)
            nombres.append(nombre)
    return np.array(embeddings), nombres
```

**Buscar top-k similares**


**[Celda 11 - Código]**
```python
def buscar_similares(imagen_consulta, embeddings_base, nombres_base, mode, pooling, layers, device, top_k=5):
    """Devuelve top_k imágenes más similares usando coseno."""
    emb_consulta = extract_embedding(imagen_consulta, mode=mode, pooling=pooling, layers=layers, device=device)
    similitudes = cosine_similarity([emb_consulta], embeddings_base)[0]
    indices = np.argsort(similitudes)[::-1][:top_k]
    return [(nombres_base[i], similitudes[i]) for i in indices]
```

**Mostrar resultados**


**[Celda 13 - Código]**
```python
def mostrar_resultados(imagen_consulta, resultados, carpeta_base):
    """Muestra imagen de consulta y resultados."""
    plt.figure(figsize=(15, 4))
    plt.subplot(1, len(resultados) + 1, 1)
    plt.imshow(Image.open(imagen_consulta))
    plt.title("Consulta")
    plt.axis("off")

    for i, (nombre, sim) in enumerate(resultados):
        img_path = os.path.join(carpeta_base, nombre)
        plt.subplot(1, len(resultados) + 1, i + 2)
        plt.imshow(Image.open(img_path))
        plt.title(f"{nombre}\nSim: {sim:.2f}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
```

**Ejecución**


**[Celda 15 - Código]**
```python
# ================== CONFIGURACIÓN CASO ================== #
# modo = "cls"  → CASO1: usar token CLS
# modo = "pool" → CASO2: usar pooling sobre los patches
# pooling = "max" o "mean" (solo aplica si modo="pool")
# modo = "layer" → CASO3: usar capas específicas
modo = "cls"
pooling = "max"
layers = None # None si no se usa el modo layer, sino elegir numeros de capa (0-23)

# ================== RUTAS ================== #
carpeta_base = r"C:\Users\Usuario\Documents\ViT 2025\DB Logos y Consultas 2025\imagenesBD-full"
carpeta_consultas = r"C:\Users\Usuario\Documents\ViT 2025\DB Logos y Consultas 2025\consultas80"

# Lista para guardar resultados en Excel
resultados_excel = []

# ================== MÉTRICAS ================== #
total_consultas = 0
aciertos_top1 = 0
aciertos_top3 = 0
aciertos_top5 = 0
reciprocal_ranks = []
tiempos_busqueda = []

# --- Medir tiempo de embeddings (base) --- #
start_time = time.time()
embeddings_base, nombres_base = build_embeddings_index(carpeta_base, mode=modo, pooling=pooling, layers=layers, device=device)
tiempo_embeddings = time.time() - start_time
print(f"⏱️ Tiempo construcción embeddings base: {tiempo_embeddings:.2f} s")

# --- Procesar consultas --- #
tiempos_busqueda = []
for i, nombre in enumerate(os.listdir(carpeta_consultas), start=1):
    if nombre.lower().endswith(('.jpg', '.jpeg', '.png')):
        total_consultas += 1
        ruta_consulta = os.path.join(carpeta_consultas, nombre)
        etiqueta_consulta = nombre.split("_")[0]

        print(f"🔍 Consulta {i}: {nombre}")

        # Medir tiempo de búsqueda
        t0 = time.time()
        resultados = buscar_similares(ruta_consulta, embeddings_base, nombres_base, mode=modo, pooling=pooling, layers=layers, device=device, top_k=5)
        t1 = time.time()
        tiempos_busqueda.append(t1 - t0)

        # Evaluar Accuracy
        # Top-1
        if resultados[0][0].startswith(etiqueta_consulta):
            aciertos_top1 += 1
        # Top-3
        if any(similar.startswith(etiqueta_consulta) for similar, _ in resultados[:3]):
            aciertos_top3 += 1
        # Top-5
        if any(similar.startswith(etiqueta_consulta) for similar, _ in resultados[:5]):
            aciertos_top5 += 1

        # Calcular MMR: Mide en qué posición aparece la primera coincidencia correcta, promediado entre consultas.
        rank_correcto = -1
        for rank, (similar, _) in enumerate(resultados, start=1):
            if similar.startswith(etiqueta_consulta):
                rank_correcto = rank
                break
        if rank_correcto != -1:
            reciprocal_ranks.append(1.0 / rank_correcto)
        else:
            reciprocal_ranks.append(0.0)

        # Guardar en lista
        for rank, (nombre_sim, sim) in enumerate(resultados, start=1):
            resultados_excel.append({
                "Consulta": nombre,
                "Rank": rank,
                "Similar": nombre_sim,
                "Similitud_Coseno": sim
            })
        resultados_excel.append({"Consulta": "", "Rank": "", "Similar": "", "Similitud_Coseno": "", "Etiqueta_consulta": ""})

        mostrar_resultados(ruta_consulta, resultados, carpeta_base)

# ================== RESULTADOS ================== #
acc1 = aciertos_top1 / total_consultas if total_consultas > 0 else 0
acc3 = aciertos_top3 / total_consultas if total_consultas > 0 else 0
acc5 = aciertos_top5 / total_consultas if total_consultas > 0 else 0
mrr = np.mean(reciprocal_ranks) if total_consultas > 0 else 0
tiempo_prom_busqueda = np.mean(tiempos_busqueda) if tiempos_busqueda else 0

print("\n📊 Resultados de Evaluación:")
print(f"Top-1 Accuracy: {acc1:.2%} ({aciertos_top1}/{total_consultas})")
print(f"Top-3 Accuracy: {acc3:.2%} ({aciertos_top3}/{total_consultas})")
print(f"Top-5 Accuracy: {acc5:.2%} ({aciertos_top5}/{total_consultas})")
print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
print(f"⏱️ Tiempo embeddings base: {tiempo_embeddings:.2f} s")
print(f"⏱️ Tiempo promedio búsqueda: {tiempo_prom_busqueda:.4f} s por consulta")

# ==== Guardar Excel con dos hojas ==== #
with pd.ExcelWriter("resultados_busqueda_similitud.xlsx") as writer:
    pd.DataFrame(resultados_excel).to_excel(writer, sheet_name="Resultados", index=False)

    resumen = pd.DataFrame([{
        "Total_consultas": total_consultas,
        "Top1_Accuracy": acc1,
        "Top3_Accuracy": acc3,
        "Top5_Accuracy": acc5,
        "MRR": mrr,
        "Tiempo_embeddings_base_s": tiempo_embeddings,
        "Tiempo_prom_busqueda_s": tiempo_prom_busqueda
    }])
    resumen.to_excel(writer, sheet_name="Resumen_métricas", index=False)

print("✅ Excel generado con hoja de resultados y hoja de métricas")

# ================== GRÁFICO ================== #
plt.bar(["Top-1", "Top-3", "Top-5"], [acc1, acc3, acc5], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Accuracy de Recuperación por Top-k")
plt.show()
```


*Salida:*
```text
100%|██████████| 10391/10391 [02:19<00:00, 74.25it/s]
⏱️ Tiempo construcción embeddings base: 140.01 s
🔍 Consulta 1: 1.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 2: 102462.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 3: 102474.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 4: 102476.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 5: 102852.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 6: 103285.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 7: 103287.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 8: 108510.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 9: 1099.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 10: 1100.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 11: 1111.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 12: 1114.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 13: 1126.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 14: 1129.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 15: 1132.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 16: 1135.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 17: 1139.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 18: 1150.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 19: 1167.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 20: 1173.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 21: 1174.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 22: 1178.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 23: 1179.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 24: 121206.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 25: 1253.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 26: 1664.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 27: 1668.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 28: 1687.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 29: 1707.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 30: 1710.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 31: 1714.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 32: 1717.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 33: 1724.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 34: 1733.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 35: 1737.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 36: 1742.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 37: 1749.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 38: 1775.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 39: 1777.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 40: 1885.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 41: 1916.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 42: 1919.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 43: 1922.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 44: 1923.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 45: 1924.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 46: 1926.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 47: 1992.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 48: 2097.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 49: 2238.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 50: 602.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 51: 603.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 52: 617.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 53: 622.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 54: 630.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 55: 642.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 56: 676.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 57: 679.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 58: 680.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 59: 691.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 60: 713.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 61: 728.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 62: 732.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 63: 735.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 64: 78.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 65: 82.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 66: 858.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 67: 862.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 68: 865.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 69: 868.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 70: 87.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 71: 89.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 72: 907.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 73: 908.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 74: 918.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 75: 920.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 76: 921.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 77: 93.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 78: 944.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 79: 96.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta 80: 961.png
<Figure size 1500x400 with 6 Axes>
📊 Resultados de Evaluación:
Top-1 Accuracy: 27.50% (22/80)
Top-3 Accuracy: 35.00% (28/80)
Top-5 Accuracy: 41.25% (33/80)
Mean Reciprocal Rank (MRR): 0.3215
⏱️ Tiempo embeddings base: 140.01 s
⏱️ Tiempo promedio búsqueda: 0.0866 s por consulta
✅ Excel generado con hoja de resultados y hoja de métricas
<Figure size 640x480 with 1 Axes>
```

# PAPELERA

Generar CSV desde nombre de archivos


**[Celda 18 - Código]**
```python
ruta_imagenes = "/content/drive/MyDrive/Colab Notebooks/Extraccion de Caracteristicas archivos/imagenesBD"

# Extraer etiquetas desde el nombre del archivo (antes del "_")
datos = []
for nombre in os.listdir(ruta_imagenes):
    if nombre.lower().endswith(('.jpg', '.jpeg', '.png')):
        etiqueta = nombre.split("_")[0]  # <-- Cambiá esto si tus nombres son distintos
        datos.append({"filename": nombre, "label": etiqueta})

df = pd.DataFrame(datos)
df.to_csv("dataset.csv", index=False)

print(df.head())

```


*Salida:*
```text
filename                label
0         freiburg.png         freiburg.png
1  atletico-madrid.png  atletico-madrid.png
2         athletic.png         athletic.png
3     estac-troyes.png     estac-troyes.png
4          hamburg.png          hamburg.png
```

Leer CSV y preparar clases


**[Celda 20 - Código]**
```python
# Cargar CSV y clases
df = pd.read_csv("dataset.csv")
clases = sorted(df['label'].unique())
label2id = {label: idx for idx, label in enumerate(clases)}
id2label = {idx: label for label, idx in label2id.items()}

# Feature extractor de ViT
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Augmentations
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

# Dataset personalizado
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")
        label = label2id[row['label']]
        if self.transform:
            image = self.transform(image)
        return {"pixel_values": image, "labels": label}

```


*Salida:*
```text
/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
preprocessor_config.json:   0%|          | 0.00/160 [00:00<?, ?B/s]/usr/local/lib/python3.11/dist-packages/transformers/models/vit/feature_extraction_vit.py:30: FutureWarning: The class ViTFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use ViTImageProcessor instead.
  warnings.warn(
```

Dividir dataset


**[Celda 22 - Código]**
```python
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = CustomImageDataset(train_df, "/content/drive/MyDrive/Colab Notebooks/Extraccion de Caracteristicas archivos/imagenesBD", transform=train_transform)
val_dataset = CustomImageDataset(val_df, "/content/drive/MyDrive/Colab Notebooks/Extraccion de Caracteristicas archivos/imagenesBD", transform=val_transform)

```


**[Celda 23 - Código]**
```python

model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")

print(model)
```


*Salida:*
```text
/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
config.json:   0%|          | 0.00/502 [00:00<?, ?B/s]model.safetensors:   0%|          | 0.00/346M [00:00<?, ?B/s]Some weights of ViTForImageClassification were not initialized from the model checkpoint at google/vit-base-patch16-224-in21k and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
ViTForImageClassification(
  (vit): ViTModel(
    (embeddings): ViTEmbeddings(
      (patch_embeddings): ViTPatchEmbeddings(
        (projection): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
      )
      (dropout): Dropout(p=0.0, inplace=False)
    )
    (encoder): ViTEncoder(
      (layer): ModuleList(
        (0-11): 12 x ViTLayer(
          (attention): ViTAttention(
            (attention): ViTSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
            )
            (output): ViTSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (intermediate): ViTIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): ViTOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (layernorm_before): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (layernorm_after): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        )
      )
    )
    (layernorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
  )
  (classifier): Linear(in_features=768, out_features=2, bias=True)
)
```

Cargar modelo ViT y configurar entrenamiento


**[Celda 25 - Código]**
```python
num_labels = len(label2id)

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)
accuracy = evaluate.load("accuracy")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return accuracy.compute(predictions=preds, references=p.label_ids)

#training_args = TrainingArguments(
#    output_dir="./vit-finetuned",
#    evaluation_strategy="epoch",
#    save_strategy="epoch",
#    learning_rate=2e-5,
#    per_device_train_batch_size=8,
#    per_device_eval_batch_size=8,
#    num_train_epochs=10,
#    weight_decay=0.01,
#    logging_steps=10,
#    save_total_limit=2,
#    load_best_model_at_end=True,
#    metric_for_best_model="accuracy",
#)
# ❌ NO recomendado, sólo si no podés actualizar
training_args = TrainingArguments(
    output_dir="./vit-finetuned",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_steps=10,
)
```


*Salida:*
```text
config.json:   0%|          | 0.00/502 [00:00<?, ?B/s]model.safetensors:   0%|          | 0.00/346M [00:00<?, ?B/s]Some weights of ViTForImageClassification were not initialized from the model checkpoint at google/vit-base-patch16-224-in21k and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Downloading builder script: 0.00B [00:00, ?B/s]
```

Entrenar modelo #No entrenamos actualmente


**[Celda 27 - Código]**
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("./vit-finetuned-model")

```


*Salida:*
```text
[34m[1mwandb[0m: [33mWARNING[0m The `run_name` is currently set to the same value as `TrainingArguments.output_dir`. If this was not intended, please specify a different run name by setting the `TrainingArguments.run_name` parameter.
<IPython.core.display.Javascript object>[34m[1mwandb[0m: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)
[34m[1mwandb[0m: You can find your API key in your browser here: https://wandb.ai/authorize?ref=models
wandb: Paste an API key from your profile and hit enter: ··········
[34m[1mwandb[0m: [33mWARNING[0m If you're specifying your api key in code, ensure this code is not shared publicly.
[34m[1mwandb[0m: [33mWARNING[0m Consider setting the WANDB_API_KEY environment variable, or running `wandb login` from the command line.
[34m[1mwandb[0m: No netrc file found, creating one.
[34m[1mwandb[0m: Appending key for api.wandb.ai to your netrc file: /root/.netrc
[34m[1mwandb[0m: Currently logged in as: [33mlucasfrantone[0m ([33mlucasfrantone-utn-gibd[0m) to [32mhttps://api.wandb.ai[0m. Use [1m`wandb login --relogin`[0m to force relogin
<IPython.core.display.HTML object><IPython.core.display.HTML object><IPython.core.display.HTML object><IPython.core.display.HTML object><IPython.core.display.HTML object><IPython.core.display.HTML object>
```

**Extraer embedding [CLS]**


> Imagen (.jpg, .png) → Feature Extractor (resize, normalize, convertir a tensor) → **Input**: tensor [1, 3, 224, 224] → Modelo ViTForImageClassification → **Output**: logits [1, num_clases] → Predicción: clase con mayor probabilidad.

> **last_hidden_state**: contiene todos los embeddings finales (con la forma [1, 197, 768]), donde 1 → batch size; 197 → número de tokens (196 patches + 1 token [CLS]); 768 → dimensión del espacio oculto (hidden size). El [:, 0, :] es el embedding CLS que se usa para clasificación.

> Diferencia *clave* entre **1) outputs = model(inputs)** y **2) outputs = model.vit(inputs)**. En el primer caso devuelve logits, es decir las predicciones por clase, por el contrario, el segundo caso accede al encoder base del modelo. Por esto es que puede acceder al last_hidden_state.


**[Celda 29 - Código]**
```python
def extract_cls_embedding(image_path, model, feature_extractor):
    image = Image.open(image_path).convert('RGB')
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.vit(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze().numpy()
```

Construir índice de embeddings


**[Celda 31 - Código]**
```python
def build_embeddings_index(folder_path, model, feature_extractor):
    embeddings = []
    nombres = []

    for nombre in tqdm(os.listdir(folder_path)):
        ruta = os.path.join(folder_path, nombre)
        if not ruta.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            continue
        emb = extract_cls_embedding(ruta, model, feature_extractor)
        print("",emb)
        embeddings.append(emb)
        nombres.append(nombre)

    return np.array(embeddings), nombres
```

Búsqueda por similitud


**[Celda 33 - Código]**
```python
def buscar_similares(imagen_consulta, embeddings_base, nombres_base, model, feature_extractor, top_k=5):
    emb_consulta = extract_cls_embedding(imagen_consulta, model, feature_extractor)
    similitudes = cosine_similarity([emb_consulta], embeddings_base)[0]
    indices_ordenados = np.argsort(similitudes)[::-1]
    return [(nombres_base[i], similitudes[i]) for i in indices_ordenados[:top_k]]
```

Mostrar resultados


**[Celda 35 - Código]**
```python
def mostrar_resultados(imagen_consulta, resultados, carpeta_base):
    k = len(resultados)
    plt.figure(figsize=(15, 4))
    # Imagen de consulta
    plt.subplot(1, k + 1, 1)
    img = Image.open(imagen_consulta)
    plt.imshow(img)
    plt.title("Consulta")
    plt.axis("off")

    for i, (nombre, sim) in enumerate(resultados):
        img_path = os.path.join(carpeta_base, nombre)
        plt.subplot(1, k + 1, i + 2)
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(f"{nombre}\nSim: {sim:.2f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

```

Ejecución de búsqueda y visualización


**[Celda 37 - Código]**
```python
# Carpeta donde están las imágenes a comparar (puede ser toda tu base)
carpeta_base = "/content/drive/MyDrive/Colab Notebooks/Extraccion de Caracteristicas archivos/imagenesBD"

# Construir índice de embeddings
embeddings_base, nombres_base = build_embeddings_index(carpeta_base, model, feature_extractor)
```


*Salida:*
```text
100%|██████████| 150/150 [02:29<00:00,  1.00it/s]
```


**[Celda 38 - Código]**
```python
import ipywidgets as widgets
from IPython.display import display
import os

# 📁 Carpeta con imágenes de consulta
carpeta_consultas = "/content/Db"

# Obtener archivos válidos
imagenes = [f for f in os.listdir(carpeta_consultas) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Widget de selección
selector = widgets.Dropdown(options=imagenes, description="Consulta:")
display(selector)

# Botón para lanzar la búsqueda
def on_button_click(b):
    imagen_seleccionada = selector.value
    ruta_imagen = os.path.join(carpeta_consultas, imagen_seleccionada)

    # Buscar similares y mostrar
    resultados = buscar_similares(ruta_imagen, embeddings_base, nombres_base, model, feature_extractor)
    mostrar_resultados(ruta_imagen, resultados, carpeta_base)

boton = widgets.Button(description="Buscar similares")
boton.on_click(on_button_click)
display(boton)

```


*Salida:*
```text
Dropdown(description='Consulta:', options=('Gemini_Generated_Image_db72ygdb72ygdb72.png', 'Gemini_Generated_Im…Button(description='Buscar similares', style=ButtonStyle())<Figure size 1500x400 with 6 Axes><Figure size 1500x400 with 6 Axes>
```


**[Celda 39 - Código]**
```python
# 📁 Carpeta con imágenes de consulta
carpeta_consultas = "/content/drive/MyDrive/Colab Notebooks/Extraccion de Caracteristicas archivos/consultas_normalizadas"
# Buscar similares para todas las imágenes en esa carpeta
for nombre in os.listdir(carpeta_consultas):
    if not nombre.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    ruta = os.path.join(carpeta_consultas, nombre)
    print(f"🔍 Consulta: {nombre}")
    resultados = buscar_similares(ruta, embeddings_base, nombres_base, model, feature_extractor)
    mostrar_resultados(ruta, resultados, carpeta_base)

```


*Salida:*
```text
🔍 Consulta: aston-villa.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: augsburg.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: cadiz.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: deportivo-alavez.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: celta.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: clermont-foot-63.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: bremen.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: udinese.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: empoli.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: paris-saint-germain.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: dusseldorf.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: elche.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: manchester-united.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: levante.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: ogc-nice.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: spezia.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: wolfsburg.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: barcelona.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: inter.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: stuttgart.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: roma.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: villarreal.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: crystal-palace.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: fiorentina.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: moenchengladbach.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: napoli.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: chelsea.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: sampdoria.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: cagliari.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: real-betis.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: olympique-lyonnais.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: juventus.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: getafe.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: brentford.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: watford.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: bologna.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: norwich-city.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: granada.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: valencia.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: mainz.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: bayern.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: fc-lorient.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: west-ham-united.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: hellas-verona.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: genoa.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: liverpool.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: southampton.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: burnley.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: hoffenheim.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: dortmund.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: freiburg.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: espanyol.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: schalke.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: redbull-leipzig.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: ac-milan.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: arsenal.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: real-madrid.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: real-sociedad.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: manchester-city.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: nacional.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: atlanta.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: peñarol.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: gimnasia.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: atletico-tucuman.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: cerro-largo.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: argentinos-juniors.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: wanders.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: colon.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: danubio.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: river-plate.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: rampla-juniors.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: bella-vista.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: central-español.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: club-atletico-progreso.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: defensor-sporting.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: aldosivi.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: arsenal-sarandi.png
<Figure size 1500x400 with 6 Axes>🔍 Consulta: boca.png
```


**[Celda 40 - Código]**
```python
from google.colab import drive
drive.mount('/content/drive')
```


*Salida:*
```text
Mounted at /content/drive
```
