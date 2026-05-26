---
aliases: [GIBD2025_ViT_Modelo+Entrenamiento0]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2025-08-05
origen_zip: GIBD-20260521T205218Z-3-003.zip
ruta_interna: "GIBD/ViT/GIBD2025_ViT_Modelo+Entrenamiento0.ipynb"
tamanio_bytes: 8015
---

# Notebook: GIBD2025_ViT_Modelo+Entrenamiento0.ipynb

Ruta interna: `GIBD/ViT/GIBD2025_ViT_Modelo+Entrenamiento0.ipynb`

---

**Importaciones**


**[Celda 2 - Código]**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sam import SAM
import os
```

**Parámetros**


**[Celda 4 - Código]**
```python
# Es necesario modificar los parámetros en base a las imágenes que contiene la BD

img_size = 224
patch_size = 16
embedding_dim = 768
d_k = d_v = 64
num_classes = 10
```

**Transformaciones**


**[Celda 6 - Código]**
```python
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])
```

**Parches**


*   La clase *PatchEmbedding* se encarga de: (1) Dividir la imagen en patches. (2) Aplanar el patch y proyectarlo a un embedding vectorial. (3) Agregar el token [CLS]. (4) Sumar los embeddings de posición.
*   *Return seq* devuelve: Primer token: [CLS]. Tokens restantes: parches de la imagen. Todos enriquecidos con información de posición. Es decir, secuencia lista para ingresar al Transformer con la forma: [batch_size, num_tokens + 1, embedding_dim].


**[Celda 8 - Código]**
```python
class PatchEmbedding(nn.Module): #nn.Module: módulo de PyTorch que se puede integrar a redes neuronales
    def __init__(self, patch_size, embedding_dim):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size) # Crea un Unfold que divide la imagen en patches no superpuestos.
        self.proj = nn.Linear(3 * patch_size * patch_size, embedding_dim) # Proyección lineal que convierte cada parche a un vector de dimensión embedding_dim.
        self.cls_token = nn.Parameter(torch.zeros(1, embedding_dim)) # Crea un parámetro aprendible para el token [CLS], inicializado en ceros.
        self.pos_embedding = nn.Parameter(torch.zeros(1 + (img_size // patch_size) ** 2, embedding_dim)) # Crea los embeddings de posición, un vector para cada token (1 para [CLS]).

    def forward(self, x):
        patches = self.unfold(x)  # Extrae los parches [B(batch), 3*16*16, num_patches].
        patches = patches.transpose(1, 2)  # Rearma el tensor para que cada parche sea un vector [B, num_patches(img_size // patch_size) ** 2), patch_dim(3 * patch_size * patch_size)].
        patches = self.proj(patches)  # Proyecta cada parche a un espacio de embedding de dimensión embedding_dim. Así todos los tokens (parches) tienen la misma dimensión que usará el Transformer [B, num_patches, embedding_dim].

        cls_tokens = self.cls_token.expand(x.size(0), -1).unsqueeze(1)  # Replica el token [CLS] para cada imagen en el batch [B, 1, embedding_dim].
        seq = torch.cat([cls_tokens, patches], dim=1)  # Concatena el token [CLS] [B, num_patches+1, embedding_dim].
        seq = seq + self.pos_embedding1
        return seq
```

**Modelo ViT simple (una sola cabeza de atención)**


**[Celda 10 - Código]**
```python
class SimpleViT(nn.Module):
    def __init__(self, embedding_dim, d_k, num_classes):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, embedding_dim) # Instancia del módulo que transforma una imagen en una secuencia de embeddings.
        self.W_Q = nn.Linear(embedding_dim, d_k) # Transformación de cada token en Query.
        self.W_K = nn.Linear(embedding_dim, d_k) # Transformación de cada token en Key.
        self.W_V = nn.Linear(embedding_dim, d_k) # Transformación de cada token en Value.
        self.classifier = nn.Linear(d_k, num_classes) # Capa lineal final para clasificar la salida del token [CLS]. Convierte un vector de tamaño d_k en un vector de logits (uno por clase).

    def forward(self, x):
        seq = self.patch_embed(x)  # x es un tensor de imágenes de tamaño [B, 3, H, W] [B, num_patches+1, embedding_dim].
        Q = self.W_Q(seq) # (queries): qué información se busca. [B, num_tokens, d_k]
        K = self.W_K(seq) # (keys): qué información ofrece cada token. [B, num_tokens, d_k]
        V = self.W_V(seq) # (values): qué información será propagada. [B, num_tokens, d_k]

        attn_scores = (Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32, device=x.device)) # Cálculo de la antención escalar. Q @ Kᵗ: producto escalar entre queries y keys. Da como resultado [B, num_tokens, num_tokens], es decir, cuánto se atiende a cada token desde cada otro token. Se divide por √d_k para estabilizar los gradientes (paper).
        attn_weights = torch.softmax(attn_scores, dim=-1) # Se convierte la matriz de scores en probabilidades de atención, usando softmax.
        output = attn_weights @ V  # Cada token "mezcla" los valores V de los demás tokens, ponderados por cuánto los atiende. Sale una nueva secuencia de tokens procesados. [B, num_patches+1, d_k]

        cls_output = output[:, 0]  # Se extrae la salida del token [CLS], que está en la primera posición. Este token resume la información de toda la imagen para clasificación. [B, d_k]
        logits = self.classifier(cls_output)  # Se pasa el vector [CLS] por una capa lineal para obtener un logit por clase. [B, num_classes]
        return logits
```

**Inicialización**


**[Celda 12 - Código]**
```python
model = SimpleViT(embedding_dim, d_k, num_classes).to(device) # Creación de la instancia del modelo SimpleViT previamente definido.
base_optimizer = torch.optim.Adam
optimizer = SAM(model.parameters(), base_optimizer, lr=1e-4)
criterion = nn.CrossEntropyLoss()
```

**Entrenamiento**


**[Celda 14 - Código]**
```python
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Forward + backward adversario
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.first_step(zero_grad=True)  # Mueve los pesos en la dirección que aumenta la pérdida, para simular una región aguda (zero_grad: Limpia los gradientes automáticamente antes de cada backward).

        # Forward + backward real desde esa región
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.second_step(zero_grad=True) # Entrena desde esa nueva posición, buscando un mínimo más plano y estable (zero_grad: Limpia los gradientes automáticamente antes de cada backward).

        total_loss += loss.item()

    print(f"Época {epoch+1}, Pérdida promedio: {total_loss / len(train_loader):.4f}")
```
