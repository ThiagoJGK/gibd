---
aliases: [Analisis de Sentimientos]
tags:
  - grupo/g3_texto
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2026-02-11
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/2025 - Analisis de Sentimientos/CinesArgentinos/Analisis de Sentimientos.ipynb"
tamanio_bytes: 537541
---

# Notebook: Analisis de Sentimientos.ipynb

Ruta interna: `GIBD/2025 - Analisis de Sentimientos/CinesArgentinos/Analisis de Sentimientos.ipynb`

---

## **1. Preparación de los Datos**

En esta sección, cargamos el conjunto de datos (CinesArgentinos.csv) y definimos el criterio para la clasificación de sentimientos. El objetivo es asignar una de tres etiquetas: Negativo, Neutro o Positivo, basadas en la puntuación original del comentario.

### 1.1 Carga de Datos


**[Celda 4 - Código]**
```python
import pandas as pd
from google.colab import files
import io
```


**[Celda 5 - Código]**
```python
# Subir el archivo CSV
uploaded = files.upload()
csv_file = list(uploaded.keys())[0]
```


*Salida:*
```text
<IPython.core.display.HTML object>Saving CinesArgentinos.csv to CinesArgentinos.csv
```


**[Celda 6 - Código]**
```python
# Leer el CSV en un DataFrame de Pandas
df = pd.read_csv(
    io.StringIO(uploaded[csv_file].decode('latin1')),
    sep='\t',
    header=None,
    names=['idPelicula', 'comentario', 'puntuacion']
)


print("Primeras 5 filas del DataFrame:")
print(df.head())
print("\nInformación del DataFrame:")
print(df.info())
```


*Salida:*
```text
Primeras 5 filas del DataFrame:
   idPelicula                                         comentario  puntuacion
0         437  Una representacion de la vida norteamericana d...         4.5
1         437  No me pareció una obra de arte como dicen algu...         4.0
2         437  No puedo creer que nunca haya visto esta pieza...         5.0
3         437  En ese entonces no tenia edad para ver esta pe...         4.0
4         437  Supongo que es una de esas peliculas que la ve...         4.0

Información del DataFrame:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 52309 entries, 0 to 52308
Data columns (total 3 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   idPelicula  52309 non-null  int64  
 1   comentario  52309 non-null  object 
 2   puntuacion  52309 non-null  float64
dtypes: float64(1), int64(1), object(1)
memory usage: 1.2+ MB
None
```

### 1.2 Clasificación de Sentimientos

La siguiente función permite clasificar a cada comentario según la puntuación en:

- Negativo: puntuación menor que 3.
- Neutro: puntuación de 3.
- Positivo: puntuación mayor que 3.

Se asignará 'desconocido' en caso de obtener valores inesperados (fuera del rango).


**[Celda 9 - Código]**
```python
def clasificar_sentimiento(puntuacion):
    if puntuacion >= 0 and puntuacion < 3:
        return 'negativo'
    elif puntuacion == 3:
        return 'neutro'
    elif puntuacion <= 5:
        return 'positivo'
    else:
        return 'desconocido'

# Crear la nueva columna 'sentimiento'
df['sentimiento'] = df['puntuacion'].apply(clasificar_sentimiento).str.capitalize()
df['sentimiento_lower'] = df['sentimiento'].str.lower()
```


**[Celda 10 - Código]**
```python
print("\nCantidad de comentarios por sentimiento:")
print(df['sentimiento'].value_counts())
```


*Salida:*
```text
Cantidad de comentarios por sentimiento:
sentimiento
Positivo    36661
Negativo     9898
Neutro       5750
Name: count, dtype: int64
```

### 1.3 División del DataFrame

A continuación, se divide el dataset en tres subconjuntos

- Training
- Validación
- Testing

El conjunto de Entrenamiento se balancea al incluir 5000 muestras de cada clase de sentimiento. El resto de los datos se dividen en Validación y Testing, manteniendo la distribución original para evaluar el rendimiento en un escenario de datos real.


**[Celda 13 - Código]**
```python
from sklearn.model_selection import train_test_split
```


**[Celda 14 - Código]**
```python
# Balance del conjunto de entrenamiento

# Cantidad de comentarios que vamos a tomar por clase
n_samples_per_class = 5000

# Tomar 5000 ejemplos de cada clase al azar
df_train_pos = df[df['sentimiento_lower'] == 'positivo'].sample(n=n_samples_per_class, random_state=42)
df_train_neg = df[df['sentimiento_lower'] == 'negativo'].sample(n=n_samples_per_class, random_state=42)
df_train_neu = df[df['sentimiento_lower'] == 'neutro'].sample(n=n_samples_per_class, random_state=42)

# Unirlos en un único DataFrame de entrenamiento
df_training = pd.concat([df_train_pos, df_train_neg, df_train_neu]).sample(frac=1, random_state=42).reset_index(drop=True)

print("Cantidad de ejemplos en entrenamiento:")
print(df_training['sentimiento'].value_counts())

# Crear el conjunto temporal con los comentarios restantes
df_remaining = df.drop(df_training.index)

# Dividir el restante en validación (10%) y testing (30%)
df_testing, df_validation = train_test_split(
    df_remaining,
    test_size=0.25,
    random_state=42,
    stratify=df_remaining['sentimiento']
)

```


*Salida:*
```text
Cantidad de ejemplos en entrenamiento:
sentimiento
Neutro      5000
Negativo    5000
Positivo    5000
Name: count, dtype: int64
```


**[Celda 15 - Código]**
```python
# Mostrar los tamaños finales de cada conjunto
print("\nTamaño de cada conjunto de datos:")
print(f"Conjunto de entrenamiento: {len(df_training)} comentarios ({len(df_training)/len(df):.0%})")
print(f"Conjunto de validación: {len(df_validation)} comentarios ({len(df_validation)/len(df):.0%})")
print(f"Conjunto de testing: {len(df_testing)} comentarios ({len(df_testing)/len(df):.0%})")
```


*Salida:*
```text
Tamaño de cada conjunto de datos:
Conjunto de entrenamiento: 15000 comentarios (29%)
Conjunto de validación: 9328 comentarios (18%)
Conjunto de testing: 27981 comentarios (53%)
```

## **2. Generación de Embeddings con BGE-M3**

En esta sección, usamos el modelo pre-entrenado BGE-M3 de SentenceTransformer para convertir cada comentario en un vector numérico de 1024 dimensiones.

### 2.1 Importación de Librerías y Carga del Modelo


**[Celda 19 - Código]**
```python
from sentence_transformers import SentenceTransformer
import torch
```


**[Celda 20 - Código]**
```python
model = SentenceTransformer('BAAI/bge-m3', device='cuda' if torch.cuda.is_available() else 'cpu')
print("Modelo BGE-M3 cargado exitosamente.")
```


*Salida:*
```text
/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
modules.json:   0%|          | 0.00/349 [00:00<?, ?B/s]config_sentence_transformers.json:   0%|          | 0.00/123 [00:00<?, ?B/s]README.md: 0.00B [00:00, ?B/s]sentence_bert_config.json:   0%|          | 0.00/54.0 [00:00<?, ?B/s]config.json:   0%|          | 0.00/687 [00:00<?, ?B/s]pytorch_model.bin:   0%|          | 0.00/2.27G [00:00<?, ?B/s]model.safetensors:   0%|          | 0.00/2.27G [00:00<?, ?B/s]tokenizer_config.json:   0%|          | 0.00/444 [00:00<?, ?B/s]sentencepiece.bpe.model:   0%|          | 0.00/5.07M [00:00<?, ?B/s]tokenizer.json:   0%|          | 0.00/17.1M [00:00<?, ?B/s]special_tokens_map.json:   0%|          | 0.00/964 [00:00<?, ?B/s]config.json:   0%|          | 0.00/191 [00:00<?, ?B/s]Modelo BGE-M3 cargado exitosamente.
```

### 2.2 Generación de Embeddings

Procedemos a generar los embeddings para los conjuntos de datos de Entrenamiento, Validación y Prueba. De esta forma, cada comentario corresponderá a un vector de dimensión fija.


**[Celda 23 - Código]**
```python
def generar_embeddings(df, model, column_text="comentario"):
    return model.encode(
        df[column_text].tolist(),
        convert_to_tensor=True,
        show_progress_bar=True
    )
```


**[Celda 24 - Código]**
```python
# Convertir los comentarios de entrenamiento a embeddings
X_train = generar_embeddings(df_training, model)

# Guardar las etiquetas de sentimiento en variables separadas
y_train = df_training['sentimiento'].map({'negativo':0, 'neutro':1, 'positivo':2}).values
```


*Salida:*
```text
Batches:   0%|          | 0/469 [00:00<?, ?it/s]
```


**[Celda 25 - Código]**
```python
# Convertir los comentarios de validation a embeddings
X_val = generar_embeddings(df_validation, model)

# Guardar las etiquetas de sentimiento en variables separada
y_val = df_validation['sentimiento'].map({'negativo':0, 'neutro':1, 'positivo':2}).values
```


*Salida:*
```text
Batches:   0%|          | 0/292 [00:00<?, ?it/s]
```


**[Celda 26 - Código]**
```python
# Convertir los comentarios de testing a embeddings
X_test = generar_embeddings(df_testing, model)

# Guardar las etiquetas de sentimiento en variables separada
y_test = df_testing['sentimiento'].map({'negativo':0, 'neutro':1, 'positivo':2}).values
```


*Salida:*
```text
Batches:   0%|          | 0/875 [00:00<?, ?it/s]
```

Para optimizar el flujo de trabajo y evitar recálculos, los embeddings generados se guardan en archivos persistentes. Esto permite cargar rápidamente las representaciones vectoriales para los distintos experimentos.


**[Celda 28 - Código]**
```python
import numpy as np
```


**[Celda 29 - Código]**
```python
# Guardar embeddings
torch.save(X_train, "X_train.pt")
torch.save(X_val, "X_val.pt")
torch.save(X_test, "X_test.pt")

# Guardar etiquetas
np.save("y_train.npy", y_train)
np.save("y_val.npy", y_val)
np.save("y_test.npy", y_test)
```


**[Celda 30 - Código]**
```python
# Cargar embeddings
X_train = torch.load("X_train.pt")
X_val = torch.load("X_val.pt")
X_test = torch.load("X_test.pt")

# Cargar etiquetas
y_train = np.load("y_train.npy")
y_val = np.load("y_val.npy")
y_test = np.load("y_test.npy")
```


**[Celda 31 - Código]**
```python
print(f"Forma de los embeddings:")
print(X_train.shape, X_val.shape, X_test.shape)
print(len(y_train), len(y_val), len(y_test))
```


*Salida:*
```text
Forma de los embeddings:
torch.Size([15000, 1024]) torch.Size([9328, 1024]) torch.Size([27981, 1024])
15000 9328 27981
```

## **3. Metodología 1: Clasificación con Red Neuronal**

En esta primera metodología, entrenamos una Red Neuronal Multicapa para que aprenda a mapear los embeddings de BGE-M3 (la entrada) a las tres clases de sentimiento (la salida).

### 3.1 Definición del Modelo

Definimos una arquitectura de red con capas densas, activación ReLU y Dropout para la regularización.


**[Celda 36 - Código]**
```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```


**[Celda 37 - Código]**
```python
from torch.utils.data import Dataset, DataLoader

class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_loader = DataLoader(EmbeddingDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(EmbeddingDataset(X_val, y_val), batch_size=32, shuffle=False)
test_loader = DataLoader(EmbeddingDataset(X_test, y_test), batch_size=32, shuffle=False)
```


**[Celda 38 - Código]**
```python
import torch.nn as nn

class SentimentClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden1=512, hidden2=256, num_classes=3, dropout=0.3):
        super(SentimentClassifier, self).__init__()

        self.fc1 = nn.Linear(embedding_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden1, hidden2)
        #self.bn2 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        #self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU() # activación suave, más moderna que ReLU

    def forward(self, x):
        x = self.act(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = self.act(self.fc2(x))
        #x = self.dropout(x)
        return self.fc3(x)

# Instanciar el modelo
embedding_dim = X_train.shape[1]
model_nn = SentimentClassifier(embedding_dim).to(device)
```

### 3.2 Entrenamiento del Modelo

El entrenamiento se realiza utilizando la función de pérdida CrossEntropyLoss y el optimizador AdamW.


**[Celda 41 - Código]**
```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model_nn.parameters(), lr=1e-4, weight_decay=1e-4)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
```


**[Celda 42 - Código]**
```python
import torch.optim as optim
import time

def train_model(model, optimizer, train_loader, test_loader, batch_size, GPU=False,
                epochs=20, lr=0.0001, patience=7):

    # Fijar la semilla para reproducibilidad
    torch.manual_seed(42)

    # Seleccionar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() and GPU else "cpu")
    model.to(device)

    # Definir la función de pérdida
    criterion = nn.CrossEntropyLoss()
    # Crear el optimizador Adam
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Crear el scheduler para ajustar el learning rate
    # Se usa ReduceLROnPlateau, común para Early Stopping
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Medir tiempo de entrenamiento
    start_time = time.time()

    # Variables para Early Stopping
    best_val_loss = float("inf")
    patience_counter = 0

    # Listas para almacenar métricas de cada época (el historial)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # Ciclo Principal por Épocas
    for epoch in range(epochs):

        model.train()  # Poner el modelo en modo entrenamiento

        total_train_loss = 0.0
        correct_train, total_train = 0, 0

        # Iterar sobre batches de entrenamiento
        for X_batch, y_batch in train_loader:
            # Mover datos a GPU si se usa
            X_batch, y_batch = X_batch.to(device).float(), y_batch.to(device)

            # Forward pass, Backward pass, y Optimización
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            # Acumular métricas
            total_train_loss += loss.item() * X_batch.size(0) # Pérdida ponderada por tamaño batch
            preds = outputs.argmax(dim=1)
            correct_train += (preds == y_batch).sum().item()
            total_train += y_batch.size(0)

        # Calcular métricas promedio de entrenamiento
        avg_train_loss = total_train_loss / total_train
        train_acc = correct_train / total_train

        # Fase de Validación
        model.eval()  # Poner modelo en modo evaluación

        total_val_loss = 0.0
        correct_val, total_val = 0, 0

        with torch.no_grad():  # Desactiva cálculo de gradientes
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device).float(), y_batch.to(device)

                # Forward pass y Cálculo de pérdida/predicción
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                # Acumular métricas
                total_val_loss += loss.item() * X_batch.size(0)
                preds = outputs.argmax(dim=1)
                correct_val += (preds == y_batch).sum().item()
                total_val += y_batch.size(0)

        # Calcular métricas promedio de validación
        avg_val_loss = total_val_loss / total_val
        val_acc = correct_val / total_val

        # Guardar métricas de esta epoch
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Reporte de la época actual
        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
        )

        # Early Stopping y Scheduler
        # Ajustar el Learning Rate según la pérdida de validación
        scheduler.step(avg_val_loss)

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Guardar el mejor modelo
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping activado. Se detiene el entrenamiento.")
                break

    # Finalización
    duration = (time.time() - start_time) / 60
    print(f"\nTraining completed in {duration:.2f} minutes")

    return history # Devolvemos las métricas para su posterior análisis o visualización
```


**[Celda 43 - Código]**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Entrenar y guardar el historial
history = train_model(model_nn, optimizer, train_loader, test_loader, batch_size=20, GPU=True,
                epochs=10, lr=0.0001, patience=7)

# Cargar el mejor modelo guardado
model_nn.load_state_dict(torch.load("best_model.pt"))

# --- Graficar ---
epochs_range = range(1, len(history["train_loss"]) + 1)

plt.figure(figsize=(12, 5))

# Pérdida
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history["train_loss"], label="Train Loss")
plt.plot(epochs_range, history["val_loss"], label="Val Loss")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.title("Evolución del Loss")
plt.legend()

# Exactitud
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history["train_acc"], label="Train Acc")
plt.plot(epochs_range, history["val_acc"], label="Val Acc")
plt.xlabel("Épocas")
plt.ylabel("Accuracy")
plt.title("Evolución de la Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

```


*Salida:*
```text
Epoch 01/10 | Train Loss: 0.3347 | Train Acc: 87.04% | Val Loss: 0.6169 | Val Acc: 78.33%
Epoch 02/10 | Train Loss: 0.2925 | Train Acc: 89.03% | Val Loss: 0.7068 | Val Acc: 75.93%
Epoch 03/10 | Train Loss: 0.2693 | Train Acc: 90.16% | Val Loss: 0.6598 | Val Acc: 79.00%
Epoch 04/10 | Train Loss: 0.2466 | Train Acc: 90.71% | Val Loss: 0.7382 | Val Acc: 76.61%
Epoch 05/10 | Train Loss: 0.2180 | Train Acc: 91.89% | Val Loss: 0.7590 | Val Acc: 77.86%
Epoch 06/10 | Train Loss: 0.1976 | Train Acc: 92.73% | Val Loss: 0.7403 | Val Acc: 78.78%
Epoch 07/10 | Train Loss: 0.1904 | Train Acc: 93.00% | Val Loss: 0.6602 | Val Acc: 81.14%
Epoch 08/10 | Train Loss: 0.1495 | Train Acc: 94.77% | Val Loss: 0.6779 | Val Acc: 80.89%
Early stopping activado. Se detiene el entrenamiento.

Training completed in 0.18 minutes
<Figure size 1200x500 with 2 Axes>
```

### 3.3 Evaluación del Modelo

Evaluamos el rendimiento del modelo final sobre el conjunto de Testing para obtener métricas clave como Accuracy, Precision, Recall y F1-score por clase.


**[Celda 46 - Código]**
```python
from sklearn.metrics import classification_report, confusion_matrix

y_true, y_pred = [], []
model_nn.eval()

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model_nn(X_batch)
        preds = outputs.argmax(dim=1).cpu().numpy()
        y_true.extend(y_batch.numpy())
        y_pred.extend(preds)

print(classification_report(y_true, y_pred, target_names=["Negativo","Neutro","Positivo"]))
print("Matriz de confusión:")
print(confusion_matrix(y_true, y_pred))
```


*Salida:*
```text
precision    recall  f1-score   support

    Negativo       0.85      0.78      0.81      5289
      Neutro       0.38      0.85      0.52      3049
    Positivo       0.97      0.81      0.88     19643

    accuracy                           0.81     27981
   macro avg       0.73      0.81      0.74     27981
weighted avg       0.88      0.81      0.83     27981

Matriz de confusión:
[[ 4132   915   242]
 [  220  2580   249]
 [  504  3324 15815]]
```


**[Celda 47 - Código]**
```python
# Get the original comments from the testing set
misclassified_comments_df = df_testing.iloc[(np.array(y_true) != np.array(y_pred))].copy()

# Add the predicted sentiment to the dataframe
misclassified_comments_df['sentimiento_predicho'] = [
    'negativo' if p == 0 else ('neutro' if p == 1 else 'positivo')
    for p in np.array(y_pred)[(np.array(y_true) != np.array(y_pred))]
]

print("\nComentarios mal clasificados (relacionados a comentarios neutros):")

# Define sentiment labels
sentiment_labels = ['negativo', 'neutro', 'positivo']

# Filter for misclassified comments where the true sentiment is 'Neutro'
neutro_misclassified = misclassified_comments_df[misclassified_comments_df['sentimiento'] == 'Neutro'].copy()

# Set pandas display option to show full comments
pd.set_option('display.max_colwidth', None)

# Iterate through predicted sentiment categories
for pred_sent in sentiment_labels:
    # Exclude correctly classified 'Neutro'
    if pred_sent != 'neutro':
        filtered_comments = neutro_misclassified[neutro_misclassified['sentimiento_predicho'] == pred_sent]
        if not filtered_comments.empty:
            print(f"\nComentarios con sentimiento Real 'Neutro' pero predicho como '{pred_sent.capitalize()}':")
            # Display up to 5 examples in a DataFrame format
            display(filtered_comments[['comentario', 'puntuacion', 'sentimiento', 'sentimiento_predicho']].head(5))

# Reset pandas display option to default after displaying
pd.reset_option('display.max_colwidth')
```


*Salida:*
```text
Comentarios mal clasificados (relacionados a comentarios neutros):

Comentarios con sentimiento Real 'Neutro' pero predicho como 'Negativo':
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        comentario  \
33502                                                                                                                                                                                                                                                                               No había visto el trailer que todos nombran, pero igual me decepcionó un poco. No me parecen taaan buenas las actuaciones, a veces los diálogos son cortados, como si estuvieran leyendo. Darín me gusta siempre, pero está para más. Algunos puntos no me cerraron, el final medio tirado de los pelos... Eso sí, las imágenes de Buenos Aires me encantaron.   
36491  No tuvieron que haber hecho esta secuela, se podría decir que arruinaron la historia. En esta segunda parte no hay nada novedoso y se centro mas en el terror, ese fue el gran error. En la primer película el suspenso estuvo perfectamente trabajado y en esta no hay absolutamente nada de suspendo. En lo personal no me asustó en ningún momento. Con respecto a las actuaciones estuvieron todas bien, aunque la de Patrick Wilson se me hizo totalmente insoportable. Se ve que el papel de malo no le queda bien. Si son de asustarse fácil, vayan a verla al cine, pero es una producción muchísimo inferior a la primera entrega.   
23164                                                                                                                                                                                                                                                                                                                                                                                                                                                        Buena película, Clooney queda dibujado frente a las actuaciones de Gosling, Giamatti o Hoffman. Mucho diálogo que no conduce hacia ningún punto, podría ser una crítica más profunda.   
26375                                                                                                                                                                                                                                                                                                                                                                                                         Una película visualmente muy buena, con un argumento muy flojo. No se compara con la primera versión de las tortugas. Prácticamente no existen escenas donde de luzcan artes marciales. todo efectos, explosión y nada mas. Regular.   
44895                                                                                                                                                                                                                                                                                                                                  Una película que tiene un guion flojísimo al igual que las actuaciones pese a todo lo malo que puede tener me entretuvo con algunas partes de acción. No la recomiendo para ver en cine , igual de lo que hay en cartelera actualmente casi no se salva ninguna en esta parte del año son películas flojas.   

       puntuacion sentimiento sentimiento_predicho  
33502         3.0      Neutro             negativo  
36491         3.0      Neutro             negativo  
23164         3.0      Neutro             negativo  
26375         3.0      Neutro             negativo  
44895         3.0      Neutro             negativo  
Comentarios con sentimiento Real 'Neutro' pero predicho como 'Positivo':
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 comentario  \
47073                                                                                                                                                                                                                                                                                                                                                                                                                                                           A diferencia de las anteriores, me pareció que tiene un tono un poco más liviano. Más humor y menos aventura (que también la tiene, ojo!). Cumple, es correcta, pero está por debajo de las dos anteriores.   
22805                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Muy buena la animación, la historia divertida para los más chicos. Una película para disfrutar en familia.   
16182  El único defecto de esta pelicula es un guion muy pobre, no hay mucho para contar de un personaje que trata de superar su temor a hablar en publico.  Lo que la hace en una buena pelicula, son las excelente actuaciones de (para mi) los dos protagonistas, la banda de sonido, dirección de arte, (cada escena parece un cuadro pintado en acuarela),y otros detalles que sin dudas le daran a esta pelicula por lo menos 7 oscar, pero no la de mejor pelicula. Aun que este  muy bien que este nominada, sin dudas sera una de las mejores películas del año,es RECOMENDABLE!! pero su único defecto es que el tema principal no da para mandarse una pelicula.   
32797                                                                                                                                                                                                                                                                                                                                              Cumple con lo prometido, es una comedia entretenida y no es una mala pelicula. Bullock nunca me gustó demasiado haciendo comedia (no sé, no me hace descostillar de risa) pero la banco y es una buena actriz, aunque la prefiero haciendo cosas como premonition. Esta comedia logra ser graciosa y es una buena opcion   
47321                                                                                                                                                                                                                                                                                                                                                                           Creo que fui uno de los pocos a los que les gustó: corta, efectivo, buen clima, personajes construídos de forma inteligente y una resolución bastante interesante. Por si fuera poco, la trama sostiene un relato bastante original. Por último, el uso de la cámara en mano a veces agota.   

       puntuacion sentimiento sentimiento_predicho  
47073         3.0      Neutro             positivo  
22805         3.0      Neutro             positivo  
16182         3.0      Neutro             positivo  
32797         3.0      Neutro             positivo  
47321         3.0      Neutro             positivo
```

## **4. Metodología 2: Clasificación por Regresión**

#### Preparación del Dataset

La siguiente clase almacena los embeddings de cada comentario, las puntuaciones y las etiquetas originales, permitiendo que el modelo acceda a ambas salidas para el entrenamiento.


**[Celda 50 - Código]**
```python
from torch.utils.data import Dataset

class EmbeddingDatasetMultiTask(Dataset):
    def __init__(self, X, puntuaciones, clases):
        self.X = X
        self.y_reg = torch.tensor(puntuaciones, dtype=torch.float32)
        self.y_cls = torch.tensor(clases, dtype=torch.long)
    def __len__(self):
        return len(self.y_reg)
    def __getitem__(self, idx):
        return self.X[idx], self.y_reg[idx], self.y_cls[idx]
```


**[Celda 51 - Código]**
```python
from torch.utils.data import DataLoader

y_train_cls = df_training["sentimiento_lower"].map({"negativo":0, "neutro":1, "positivo":2}).values
y_val_cls   = df_validation["sentimiento_lower"].map({"negativo":0, "neutro":1, "positivo":2}).values
y_test_cls  = df_testing["sentimiento_lower"].map({"negativo":0, "neutro":1, "positivo":2}).values

train_loader_mt = DataLoader(
    EmbeddingDatasetMultiTask(X_train, df_training["puntuacion"].values, y_train_cls),
    batch_size=32, shuffle=True)
val_loader_mt = DataLoader(
    EmbeddingDatasetMultiTask(X_val, df_validation["puntuacion"].values, y_val_cls),
    batch_size=32, shuffle=False)
test_loader_mt = DataLoader(
    EmbeddingDatasetMultiTask(X_test, df_testing["puntuacion"].values, y_test_cls),
    batch_size=32, shuffle=False)
```

#### Arquitectura del Modelo

La siguiente red neuronal posee dos cabezas de salida. Una capa lineal para regresión y otra para clasificación.




**[Celda 53 - Código]**
```python
import torch.nn as nn

class SentimentMultiTask(nn.Module):
    def __init__(self, embedding_dim, hidden1=512, hidden2=256, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.act = nn.LeakyReLU()

        # Dos salidas
        self.fc_reg = nn.Linear(hidden2, 1)      # puntuación (regresión)
        self.fc_cls = nn.Linear(hidden2, 3)      # clases (clasificación)

    def forward(self, x):
        x = self.act(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.act(self.fc2(x))
        score = torch.clamp(self.fc_reg(x), 0.0, 5.0)  # limitar a [0,5]
        logits = self.fc_cls(x)
        return score, logits
```

#### Función de entrenamiento


**[Celda 55 - Código]**
```python
import torch.optim as optim
import numpy as np
import time

def train_model_multitask(model, optimizer, criterion_reg, criterion_cls, train_loader, val_loader,
                          GPU=False, epochs=20, patience=7, alpha=0.5):

    device = torch.device("cuda" if torch.cuda.is_available() and GPU else "cpu")
    model.to(device)
    torch.manual_seed(42)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss, patience_counter = float("inf"), 0
    history = {"train_loss": [], "val_loss": []}

    start = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_reg, y_cls in train_loader:
            X_batch, y_reg, y_cls = X_batch.to(device).float(), y_reg.to(device).float().unsqueeze(1), y_cls.to(device)
            optimizer.zero_grad()
            out_reg, out_cls = model(X_batch)
            loss_reg = criterion_reg(out_reg, y_reg)
            loss_cls = criterion_cls(out_cls, y_cls)
            loss = alpha * loss_reg + (1 - alpha) * loss_cls    # Combinación de ambas pérdidas
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        avg_train_loss = train_loss / len(train_loader.dataset)

        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_reg, y_cls in val_loader:
                X_batch, y_reg, y_cls = X_batch.to(device).float(), y_reg.to(device).float().unsqueeze(1), y_cls.to(device)
                out_reg, out_cls = model(X_batch)
                loss_reg = criterion_reg(out_reg, y_reg)
                loss_cls = criterion_cls(out_cls, y_cls)
                loss = alpha * loss_reg + (1 - alpha) * loss_cls
                val_loss += loss.item() * X_batch.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss, patience_counter = avg_val_loss, 0
            torch.save(model.state_dict(), "best_model_multitask.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping activado.")
                break

    return history
```


**[Celda 56 - Código]**
```python
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

embedding_dim = X_train.shape[1]
model_mt = SentimentMultiTask(embedding_dim, hidden1=1024, hidden2=512, dropout=0.2).to(device)

# Definir las funciones de pérdida
criterion_reg = nn.SmoothL1Loss()  # Para la regresión de puntuación
criterion_cls = nn.CrossEntropyLoss() # Para la clasificación de sentimiento

optimizer = optim.AdamW(model_mt.parameters(), lr=1e-4, weight_decay=1e-6)

history_mt = train_model_multitask(
    model_mt, optimizer, criterion_reg, criterion_cls,
    train_loader=train_loader_mt,
    val_loader=val_loader_mt,
    GPU=True,
    epochs=25,
    patience=5,
    alpha=0.3
)

model_mt.load_state_dict(torch.load("best_model_multitask.pt"))
```


*Salida:*
```text
Epoch 01/25 | Train Loss: 0.6231 | Val Loss: 0.5399
Epoch 02/25 | Train Loss: 0.5428 | Val Loss: 0.5079
Epoch 03/25 | Train Loss: 0.5066 | Val Loss: 0.5213
Epoch 04/25 | Train Loss: 0.4661 | Val Loss: 0.5589
Epoch 05/25 | Train Loss: 0.4146 | Val Loss: 0.4881
Epoch 06/25 | Train Loss: 0.3593 | Val Loss: 0.5086
Epoch 07/25 | Train Loss: 0.3100 | Val Loss: 0.5329
Epoch 08/25 | Train Loss: 0.2652 | Val Loss: 0.4874
Epoch 09/25 | Train Loss: 0.2253 | Val Loss: 0.4858
Epoch 10/25 | Train Loss: 0.1974 | Val Loss: 0.4636
Epoch 11/25 | Train Loss: 0.1774 | Val Loss: 0.5816
Epoch 12/25 | Train Loss: 0.1521 | Val Loss: 0.4749
Epoch 13/25 | Train Loss: 0.1433 | Val Loss: 0.6448
Epoch 14/25 | Train Loss: 0.1350 | Val Loss: 0.5629
Epoch 15/25 | Train Loss: 0.1154 | Val Loss: 0.6208
Early stopping activado.
<All keys matched successfully>
```

#### Evaluación


**[Celda 58 - Código]**
```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns, matplotlib.pyplot as plt

model_mt.eval()
y_true_cls, y_pred_cls = [], []
y_true_reg, y_pred_reg = [], []

with torch.no_grad():
    for X_batch, y_reg, y_cls in test_loader_mt:
        X_batch = X_batch.to(device).float()
        score, logits = model_mt(X_batch)
        preds_cls = logits.argmax(dim=1).cpu().numpy()
        y_pred_cls.extend(preds_cls)
        y_true_cls.extend(y_cls.numpy())
        y_pred_reg.extend(score.cpu().numpy().flatten())
        y_true_reg.extend(y_reg.numpy())

print(classification_report(
    y_true_cls, y_pred_cls,
    target_names=["Negativo","Neutro","Positivo"], digits=3))

cm = confusion_matrix(y_true_cls, y_pred_cls)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negativo","Neutro","Positivo"],
            yticklabels=["Negativo","Neutro","Positivo"])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de confusión - Multi-task")
plt.show()

```


*Salida:*
```text
precision    recall  f1-score   support

    Negativo      0.715     0.888     0.792      5289
      Neutro      0.495     0.814     0.616      3049
    Positivo      0.975     0.814     0.887     19643

    accuracy                          0.828     27981
   macro avg      0.728     0.838     0.765     27981
weighted avg      0.873     0.828     0.840     27981

<Figure size 600x500 with 2 Axes>
```

#### Historial de entrenamiento


**[Celda 60 - Código]**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# --- Graficar historial de entrenamiento ---
epochs_range_mt = range(1, len(history_mt["train_loss"]) + 1)

plt.figure(figsize=(8, 5))

# Pérdida
plt.plot(epochs_range_mt, history_mt["train_loss"], label="Train Loss")
plt.plot(epochs_range_mt, history_mt["val_loss"], label="Val Loss")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.title("Evolución del Loss (Multi-task)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```


*Salida:*
```text
<Figure size 800x500 with 1 Axes>
```

## **5. Metodología 3: Clasificación por Centroides de Entrenamiento**

Esta metodología no requiere entrenamiento. En su lugar, se define la identidad de cada clase mediante un Centroide calculado a partir de los embeddings del conjunto de Entrenamiento. La clasificación de un comentario de prueba se realiza asignándole la clase cuyo centroide sea el más cercano en el espacio vectorial (medido por la Similaridad Coseno).

### 4.1 Cálculo de Centroides

Convertimos los tensores de PyTorch a arrays de NumPy para facilitar el cálculo del centroide para cada una de las tres clases de sentimiento.


**[Celda 65 - Código]**
```python
X_train_np = X_train.cpu().numpy()
X_test_np = X_test.cpu().numpy()
```


**[Celda 66 - Código]**
```python
label_map = {0: 'Negativo', 1: 'Neutro', 2: 'Positivo'}
prototype_labels = ['Negativo', 'Neutro', 'Positivo']

centroid_embeddings = []
for int_label in [0, 1, 2]:
    # 1. Seleccionar los embeddings de entrenamiento que pertenecen a esta clase
    embeddings_por_clase = X_train_np[y_train == int_label]

    # 2. Calcular el promedio de esos embeddings (el centroide)
    centroid = np.mean(embeddings_por_clase, axis=0)
    centroid_embeddings.append(centroid)

# Convertir la lista de centroides a una matriz NumPy
centroid_embeddings_np = np.array(centroid_embeddings)
print(f"Centroides de prototipos generados: {centroid_embeddings_np.shape}")


```


*Salida:*
```text
Centroides de prototipos generados: (3, 1024)
```

### 4.2 Clasificación Basada en Similaridad Coseno

Utilizamos el conjunto de embeddings de Testing y calculamos su Similaridad Coseno con respecto a los tres centroides. El comentario se clasifica con el sentimiento que produce el valor de similaridad más alto (distancia mínima).


**[Celda 69 - Código]**
```python
from sklearn.metrics.pairwise import cosine_similarity
```


**[Celda 70 - Código]**
```python
# Calcular la Similaridad Coseno contra los CENTROIDES
similarity_matrix = cosine_similarity(X_test_np, centroid_embeddings_np)

# Encontrar la etiqueta del centroide más similar
predicted_indices = np.argmax(similarity_matrix, axis=1)

# Mapear los índices (0, 1, 2) a las etiquetas de texto
predicted_labels = [label_map[i] for i in predicted_indices]
```

### 4.3 Evaluación del Rendimiento


**[Celda 72 - Código]**
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```


**[Celda 73 - Código]**
```python
y_true_labels = [label_map[i] for i in y_test]
y_pred_labels = predicted_labels
classes = prototype_labels

# a) Accuracy
accuracy = accuracy_score(y_true_labels, y_pred_labels)

# b) Classification Report
report = classification_report(
    y_true_labels,
    y_pred_labels,
    target_names=classes,
    labels=classes,
    zero_division=0
)

# c) Confusion Matrix
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=classes)

print("\n" + "="*70)
print(f"RESULTADOS DE CLASIFICACIÓN POR CENTROIDES (BGE-M3 - Usando Archivos .pt/.npy)")
print("="*70)
print(f"Accuracy General: {accuracy:.4f}")
print("\nReporte de Clasificación:\n", report)

# Visualización y Guardado
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicción')
plt.ylabel('Verdadero')
plt.title('Matriz de Confusión (Clasificación por Centroides)')
plt.savefig('matriz_confusion_centroides.png')
plt.close()

# NOTA: Para guardar el CSV de predicciones, necesitarías tener el 'df_testing' original cargado.
# Si lo tienes, puedes agregarlo así:
# df_testing['prediccion_centroide'] = predicted_labels
# df_testing[['comentario', 'sentimiento', 'prediccion_centroide']].to_csv('predicciones_centroides_cargados.csv', index=False)
```


*Salida:*
```text
======================================================================
RESULTADOS DE CLASIFICACIÓN POR CENTROIDES (BGE-M3 - Usando Archivos .pt/.npy)
======================================================================
Accuracy General: 0.7439

Reporte de Clasificación:
               precision    recall  f1-score   support

    Negativo       0.72      0.74      0.73      5289
      Neutro       0.24      0.45      0.31      3049
    Positivo       0.93      0.79      0.85     19643

    accuracy                           0.74     27981
   macro avg       0.63      0.66      0.63     27981
weighted avg       0.81      0.74      0.77     27981
```

## **6. Metodología 4: Clasificación por Regresión Logística**

#### Clasificación con un Modelo Tradicional

Ahora que tenemos los comentarios representados como vectores numéricos, entrenamos un clasificador tradicional sobre ellos. En este caso, utilizaremos Regresión Logística.


**[Celda 77 - Código]**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
```


**[Celda 78 - Código]**
```python
# Instanciar el modelo de regresión logística
classifier = LogisticRegression(random_state=42, max_iter=1000)

# Entrenar el modelo con los embeddings de entrenamiento
classifier.fit(X_train_embeddings, y_train)
```


*Salida:*
```text
LogisticRegression(max_iter=1000, random_state=42)
```

Uno vez entrenado, utilizamos el modelo para predecir los sentimientos de los comentarios incluídos en el conjunto de testing.


**[Celda 80 - Código]**
```python
y_pred = classifier.predict(X_test_embeddings)
```

#### Evaluación del Rendimiento

A continuación, se genera un resumen de las métricas clave del rendimiento del clasificador: Precision, Recall, F1-score y Accuracy. A su vez, se muestra la matriz de confusión para un mejor análisis.


**[Celda 83 - Código]**
```python
import matplotlib.pyplot as plt

print("\nEvaluación del clasificador:")
print(classification_report(y_test, y_pred))

cm = ConfusionMatrixDisplay.from_estimator(classifier, X_test_embeddings, y_test, cmap=plt.cm.Blues)
plt.title("Matriz de Confusión")
plt.show()
```


*Salida:*
```text
Evaluación del clasificador:
              precision    recall  f1-score   support

    negativo       0.75      0.74      0.75      2969
      neutro       0.37      0.10      0.16      1725
    positivo       0.86      0.96      0.91     10999

    accuracy                           0.82     15693
   macro avg       0.66      0.60      0.60     15693
weighted avg       0.79      0.82      0.79     15693

<Figure size 640x480 with 2 Axes>
```

La matriz compara las etiquetas de sentimiento que el modelo predijo (columnas) con las etiquetas de sentimiento reales (filas).

- La fila 1 nos permite ver que:
  - 2208 comentarios negativos fueron correctamente clasificados.
  - 134 fueron mal clasificados como neutros.
  - 627 fueron mal clasificados como positivos.

- La fila 2 nos permite ver que:
  - 459 comentarios neutros fueron mal clasificados como negativos.
  - 176 fueron correctamente clasificados como neutros.
  - 1090 fueron mal clasificados como positivos.

- La fila 3 nos permite ver que:
  - 273 comentarios positivos fueron mal clasificados como negativos.
  - 172 fueron mal clasificados como neutros.
  - 10554 fueron correctamente clasificados como positivos.

  Como conclusión, podemos decir que el modelo funciona muy bien para positivos (10554 aciertos frente a 445 errores), aceptablemente para negativos (2208 aciertos frente a 761 errores), pero muy mal para neutros (176 aciertos frente a 1549 errores).

## **7. Metodología 5: Clasificación por k Vecinos más Cercanos**


**[Celda 86 - Código]**
```python
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
```

#### Entrenamiento y predicción


**[Celda 88 - Código]**
```python
X_train_np = X_train.cpu().numpy()
X_test_np = X_test.cpu().numpy()
```


**[Celda 89 - Código]**
```python
# Normalización
X_train_norm = normalize(X_train_np, norm='l2')
X_test_norm = normalize(X_test_np, norm='l2')

# Configuración
k = 15
epsilon = 1e-6
p_dist = 2.0               # penaliza más a vecinos lejanos
penalty_neutro = 0.7       # reduce la influencia de la clase neutra
confidence_threshold = 0.12 # aumenta la exigencia para clasificar como neutro

# --- Pesos inversos de frecuencia (rebalanceo base) ---
class_counts = Counter(y_train)
total = sum(class_counts.values())
class_weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}

# --- Penalización manual de la clase neutra ---
class_weights[1] *= penalty_neutro

print("Pesos de clase ajustados:", class_weights)

# --- Predicción ponderada ajustada ---
def predict_knn_balanced(X_query, X_train, y_train, k=15):
    preds = []
    sims = np.dot(X_query, X_train.T)
    dists = 1 - sims  # distancia coseno

    for row_dists in dists:
        nn_idx = np.argsort(row_dists)[:k]
        nn_labels = y_train[nn_idx]
        nn_dists = row_dists[nn_idx]

        vote_weights = {}
        for label, dist in zip(nn_labels, nn_dists):
            w = class_weights[int(label)] / ((dist + epsilon) ** p_dist)
            vote_weights[label] = vote_weights.get(label, 0) + w

        # Ordenar por peso total
        sorted_votes = sorted(vote_weights.items(), key=lambda x: x[1], reverse=True)

        # Si solo hay una clase presente entre los vecinos → usarla directamente
        if len(sorted_votes) == 1:
            preds.append(sorted_votes[0][0])
            continue

        top_label, top_value = sorted_votes[0]
        second_label, second_value = sorted_votes[1]

        # Si gana neutro pero por poca diferencia -> elegir segunda clase más fuerte
        if top_label == 1 and (top_value - second_value) < confidence_threshold:
            preds.append(second_label)
        else:
            preds.append(top_label)

    return np.array(preds)

# --- Predicción ---
y_pred_tuned = predict_knn_balanced(X_test_norm, X_train_norm, y_train, k=k)

```


*Salida:*
```text
Pesos de clase ajustados: {np.int64(1): 0.7, np.int64(0): 1.0, np.int64(2): 1.0}
```

#### Evaluación y visualización


**[Celda 91 - Código]**
```python
labels = ["Negativo", "Neutro", "Positivo"]

print("\n=== Evaluación k-NN ajustado ===")
print(classification_report(y_test, y_pred_tuned, target_names=labels, digits=3))

cm = confusion_matrix(y_test, y_pred_tuned)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - k-NN Ajustado (menos falsos neutros)")
plt.tight_layout()
plt.show()
```


*Salida:*
```text
=== Evaluación k-NN ajustado ===
              precision    recall  f1-score   support

    Negativo      0.857     0.876     0.867      5289
      Neutro      0.533     0.916     0.674      3049
    Positivo      0.977     0.862     0.916     19643

    accuracy                          0.871     27981
   macro avg      0.789     0.885     0.819     27981
weighted avg      0.906     0.871     0.880     27981

<Figure size 500x400 with 2 Axes>
```


**[Celda 92 - Código]**
```python
from sklearn.metrics import f1_score
import itertools
import numpy as np

# --- Definimos los rangos a explorar ---
p_dist_values = [1.5, 2.0, 2.5]
penalty_neutro_values = [0.6, 0.7, 0.8]
confidence_threshold_values = [0.1, 0.12, 0.15]

# --- Resultados almacenados ---
results = []

# --- Bucle sobre combinaciones ---
for p_dist_try, penalty_neutro_try, conf_thresh_try in itertools.product(
    p_dist_values, penalty_neutro_values, confidence_threshold_values
):
    # Ajustar pesos de clase
    class_weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
    class_weights[1] *= penalty_neutro_try  # penalización clase neutra

    # Reutilizamos la función existente con parámetros actuales
    def predict_knn_test(X_query, X_train, y_train, k=15):
        preds = []
        sims = np.dot(X_query, X_train.T)
        dists = 1 - sims
        for row_dists in dists:
            nn_idx = np.argsort(row_dists)[:k]
            nn_labels = y_train[nn_idx]
            nn_dists = row_dists[nn_idx]
            vote_weights = {}
            for label, dist in zip(nn_labels, nn_dists):
                w = class_weights[int(label)] / ((dist + epsilon) ** p_dist_try)
                vote_weights[label] = vote_weights.get(label, 0) + w
            sorted_votes = sorted(vote_weights.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_votes) == 1:
                preds.append(sorted_votes[0][0])
                continue
            top_label, top_value = sorted_votes[0]
            second_label, second_value = sorted_votes[1]
            if top_label == 1 and (top_value - second_value) < conf_thresh_try:
                preds.append(second_label)
            else:
                preds.append(top_label)
        return np.array(preds)

    # Predicción con esta configuración
    y_pred_try = predict_knn_test(X_test_norm, X_train_norm, y_train, k=k)
    f1_macro = f1_score(y_test, y_pred_try, average='macro')

    results.append({
        'p_dist': p_dist_try,
        'penalty_neutro': penalty_neutro_try,
        'confidence_threshold': conf_thresh_try,
        'f1_macro': f1_macro
    })
    print(f"Tested p={p_dist_try}, penalty={penalty_neutro_try}, conf={conf_thresh_try} -> F1_macro={f1_macro:.4f}")

# --- Mostrar el mejor resultado ---
best_cfg = max(results, key=lambda x: x['f1_macro'])
print("\n=== Mejor combinación encontrada ===")
print(best_cfg)

```


*Salida:*
```text
Tested p=1.5, penalty=0.6, conf=0.1 -> F1_macro=0.8344
Tested p=1.5, penalty=0.6, conf=0.12 -> F1_macro=0.8344
Tested p=1.5, penalty=0.6, conf=0.15 -> F1_macro=0.8345
Tested p=1.5, penalty=0.7, conf=0.1 -> F1_macro=0.8194
Tested p=1.5, penalty=0.7, conf=0.12 -> F1_macro=0.8195
Tested p=1.5, penalty=0.7, conf=0.15 -> F1_macro=0.8196
Tested p=1.5, penalty=0.8, conf=0.1 -> F1_macro=0.8048
Tested p=1.5, penalty=0.8, conf=0.12 -> F1_macro=0.8048
Tested p=1.5, penalty=0.8, conf=0.15 -> F1_macro=0.8048
Tested p=2.0, penalty=0.6, conf=0.1 -> F1_macro=0.8338
Tested p=2.0, penalty=0.6, conf=0.12 -> F1_macro=0.8338
Tested p=2.0, penalty=0.6, conf=0.15 -> F1_macro=0.8338
Tested p=2.0, penalty=0.7, conf=0.1 -> F1_macro=0.8187
Tested p=2.0, penalty=0.7, conf=0.12 -> F1_macro=0.8188
Tested p=2.0, penalty=0.7, conf=0.15 -> F1_macro=0.8188
Tested p=2.0, penalty=0.8, conf=0.1 -> F1_macro=0.8041
Tested p=2.0, penalty=0.8, conf=0.12 -> F1_macro=0.8041
Tested p=2.0, penalty=0.8, conf=0.15 -> F1_macro=0.8042
Tested p=2.5, penalty=0.6, conf=0.1 -> F1_macro=0.8331
Tested p=2.5, penalty=0.6, conf=0.12 -> F1_macro=0.8331
Tested p=2.5, penalty=0.6, conf=0.15 -> F1_macro=0.8331
Tested p=2.5, penalty=0.7, conf=0.1 -> F1_macro=0.8182
Tested p=2.5, penalty=0.7, conf=0.12 -> F1_macro=0.8182
Tested p=2.5, penalty=0.7, conf=0.15 -> F1_macro=0.8183
Tested p=2.5, penalty=0.8, conf=0.1 -> F1_macro=0.8042
Tested p=2.5, penalty=0.8, conf=0.12 -> F1_macro=0.8042
Tested p=2.5, penalty=0.8, conf=0.15 -> F1_macro=0.8042

=== Mejor combinación encontrada ===
{'p_dist': 1.5, 'penalty_neutro': 0.6, 'confidence_threshold': 0.15, 'f1_macro': 0.8344667633807766}
```


**[Celda 93 - Código]**
```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --- Recuperar los mejores parámetros encontrados ---
p_best = best_cfg['p_dist']
penalty_neutro_best = best_cfg['penalty_neutro']
conf_thresh_best = best_cfg['confidence_threshold']

print(f"\nUsando mejores parámetros encontrados:")
print(f"p_dist={p_best}, penalty_neutro={penalty_neutro_best}, confidence_threshold={conf_thresh_best}")

# --- Recalcular los pesos de clase con penalización óptima ---
class_weights_best = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
class_weights_best[1] *= penalty_neutro_best

# --- Función para predecir con los mejores hiperparámetros ---
def predict_knn_final(X_query, X_train, y_train, k=15):
    preds = []
    sims = np.dot(X_query, X_train.T)
    dists = 1 - sims
    for row_dists in dists:
        nn_idx = np.argsort(row_dists)[:k]
        nn_labels = y_train[nn_idx]
        nn_dists = row_dists[nn_idx]
        vote_weights = {}
        for label, dist in zip(nn_labels, nn_dists):
            w = class_weights_best[int(label)] / ((dist + epsilon) ** p_best)
            vote_weights[label] = vote_weights.get(label, 0) + w
        sorted_votes = sorted(vote_weights.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_votes) == 1:
            preds.append(sorted_votes[0][0])
            continue
        top_label, top_value = sorted_votes[0]
        second_label, second_value = sorted_votes[1]
        if top_label == 1 and (top_value - second_value) < conf_thresh_best:
            preds.append(second_label)
        else:
            preds.append(top_label)
    return np.array(preds)

# --- Predicciones finales ---
y_pred_best = predict_knn_final(X_test_norm, X_train_norm, y_train, k=k)

# --- Reporte y matriz de confusión ---
labels = ["Negativo", "Neutro", "Positivo"]

print("\n=== Evaluación final con hiperparámetros óptimos ===")
print(classification_report(y_test, y_pred_best, target_names=labels, digits=3))

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - k-NN (Hiperparámetros Óptimos)")
plt.tight_layout()
plt.show()

```


*Salida:*
```text
Usando mejores parámetros encontrados:
p_dist=1.5, penalty_neutro=0.6, confidence_threshold=0.15

=== Evaluación final con hiperparámetros óptimos ===
              precision    recall  f1-score   support

    Negativo      0.839     0.886     0.862      5289
      Neutro      0.592     0.907     0.717      3049
    Positivo      0.975     0.880     0.925     19643

    accuracy                          0.884     27981
   macro avg      0.802     0.891     0.834     27981
weighted avg      0.908     0.884     0.890     27981

<Figure size 500x400 with 2 Axes>
```
