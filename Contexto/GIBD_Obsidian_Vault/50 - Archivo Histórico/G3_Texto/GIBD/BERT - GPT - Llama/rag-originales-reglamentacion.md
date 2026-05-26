---
aliases: [rag-originales-reglamentacion]
tags:
  - grupo/g3_texto
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2024-09-13
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/BERT - GPT - Llama/Notebooks/rag-originales-reglamentacion.ipynb"
tamanio_bytes: 160713
---

# Notebook: rag-originales-reglamentacion.ipynb

Ruta interna: `GIBD/BERT - GPT - Llama/Notebooks/rag-originales-reglamentacion.ipynb`

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
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m50.4/50.4 kB[0m [31m598.3 kB/s[0m eta [36m0:00:00[0m
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.0/1.0 MB[0m [31m5.9 MB/s[0m eta [36m0:00:00[0m
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m396.6/396.6 kB[0m [31m9.8 MB/s[0m eta [36m0:00:00[0m
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m289.8/289.8 kB[0m [31m8.3 MB/s[0m eta [36m0:00:00[0m
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m76.4/76.4 kB[0m [31m3.4 MB/s[0m eta [36m0:00:00[0m
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m77.9/77.9 kB[0m [31m2.4 MB/s[0m eta [36m0:00:00[0m
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m141.9/141.9 kB[0m [31m4.6 MB/s[0m eta [36m0:00:00[0m
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m58.3/58.3 kB[0m [31m2.4 MB/s[0m eta [36m0:00:00[0m
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.3/2.3 MB[0m [31m12.2 MB/s[0m eta [36m0:00:00[0m
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m49.3/49.3 kB[0m [31m2.8 MB/s[0m eta [36m0:00:00[0m
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m474.3/474.3 kB[0m [31m3.5 MB/s[0m eta [36m0:00:00[0m
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m116.3/116.3 kB[0m [31m8.4 MB/s[0m eta [36m0:00:00[0m
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m39.9/39.9 MB[0m [31m21.5 MB/s[0m eta [36m0:00:00[0m
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m134.8/134.8 kB[0m [31m6.9 MB/s[0m eta [36m0:00:00[0m
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m194.1/194.1 kB[0m [31m12.7 MB/s[0m eta [36m0:00:00[0m
[?25h[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
cudf-cu12 24.4.1 requires pyarrow<15.0.0a0,>=14.0.1, but you have pyarrow 17.0.0 which is incompatible.
ibis-framework 8.0.0 requires pyarrow<16,>=2, but you have pyarrow 17.0.0 which is incompatible.[0m[31m
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m232.6/232.6 kB[0m [31m1.8 MB/s[0m eta [36m0:00:00[0m
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m249.1/249.1 kB[0m [31m2.2 MB/s[0m eta [36m0:00:00[0m
[?25h
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
model = SentenceTransformer('BAAI/bge-m3')
```


*Salida:*
```text
modules.json:   0%|          | 0.00/349 [00:00<?, ?B/s]config_sentence_transformers.json:   0%|          | 0.00/123 [00:00<?, ?B/s]README.md:   0%|          | 0.00/15.8k [00:00<?, ?B/s]sentence_bert_config.json:   0%|          | 0.00/54.0 [00:00<?, ?B/s]config.json:   0%|          | 0.00/687 [00:00<?, ?B/s]pytorch_model.bin:   0%|          | 0.00/2.27G [00:00<?, ?B/s]tokenizer_config.json:   0%|          | 0.00/444 [00:00<?, ?B/s]sentencepiece.bpe.model:   0%|          | 0.00/5.07M [00:00<?, ?B/s]tokenizer.json:   0%|          | 0.00/17.1M [00:00<?, ?B/s]special_tokens_map.json:   0%|          | 0.00/964 [00:00<?, ?B/s]1_Pooling/config.json:   0%|          | 0.00/191 [00:00<?, ?B/s]
```


**[Celda 6 - Código]**
```python
import torch

# Mover el modelo a la GPU en caso de que no lo haga por defecto
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

# Creación de embeddings del reglamento


**[Celda 8 - Código]**
```python
#Cambiar por el path del archivo de reglamentación orignal según la computadora
with open("/content/_Reglamento.txt", "r", encoding="utf-8") as file:
    texto  = file.read()

articulos = texto.split("Documento:")
articulos = ["Documento:" + articulo.strip() for articulo in articulos if articulo.strip()]
print(f"Existen {len(articulos)} artículos.")
```


*Salida:*
```text
Existen 674 artículos.
```


**[Celda 9 - Código]**
```python
articles_embedding = []
for i,article in enumerate(articulos):
    article_embedding = model.encode(article)
    articles_embedding.append((i, article_embedding))
```

# Retrieval: recuperación de articulos más similares


**[Celda 11 - Código]**
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


**[Celda 12 - Código]**
```python
def funcionArticuloEncontrado(lista_articulos_proximos, articulo_correcto):
    for i, (article, _) in enumerate(lista_articulos_proximos):
        if articulo_correcto.strip() == article.strip():
            return i + 1
    return None
```

# Sección de pruebas y evaluación

## Importanción de DataSet de preguntas etiquetadas



**[Celda 15 - Código]**
```python
dataset = load_dataset("HuggMaxi/preguntasEtiquetadas")
df_preguntasEtiquetadas = pd.DataFrame(dataset['train'])
df_preguntasEtiquetadas
```


*Salida:*
```text
train-00000-of-00001.parquet:   0%|          | 0.00/98.2k [00:00<?, ?B/s]Generating train split:   0%|          | 0/300 [00:00<?, ? examples/s]     idConsulta                                           Consulta Articulo  \
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


**[Celda 17 - Código]**
```python
idConsulta = 18
texto_objetivo = df_preguntasEtiquetadas[df_preguntasEtiquetadas['idConsulta'] == idConsulta]['Texto'].values[0]
pregunta_consulta = df_preguntasEtiquetadas[df_preguntasEtiquetadas['idConsulta'] == idConsulta]['Consulta'].values[0]

resultadoArticulosEncontrados = devolver_N_articulos_similares(pregunta_consulta, model, articles_embedding, articulos,10)
resultadoPosicion = funcionArticuloEncontrado(resultadoArticulosEncontrados, texto_objetivo)

print(f"CONSULTA: {pregunta_consulta}")
print(f"POSICION: {resultadoPosicion}")
print("*********************************************")
print(texto_objetivo)
print("*********************************************")


```


**[Celda 18 - Código]**
```python
for i in range(len(resultadoArticulosEncontrados)):
    print(f"-------------------------------------- {i + 1}")
    print(resultadoArticulosEncontrados[i][0])
    print()
```

## Creación de Dataset de resultados


**[Celda 20 - Código]**
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
        resultadoPrueba1 = devolver_N_articulos_similares(pregunta_consulta, model, articles_embedding, articulos, 10)

        # Encontrar la posición del artículo correcto
        resultadoPosicion = funcionArticuloEncontrado(resultadoPrueba1, texto_consulta)

        # Agregar el resultado a la lista de resultados
        resultados.append({
            'idConsulta': idCon,
            'Consulta': pregunta_consulta,
            'Posicion': resultadoPosicion
        })

# Convertir la lista de resultados en un DataFrame
resultados_df = pd.DataFrame(resultados)

# Mostrar el DataFrame
print(resultados_df)

```


**[Celda 21 - Código]**
```python
#Convertir 'Posicion' a enteros, manejando los valores NaN como -1
resultados_df['Posicion'] = resultados_df['Posicion'].fillna(-1).astype(int)
```


**[Celda 22 - Código]**
```python
from google.colab import drive
drive.mount('/content/drive')
```


**[Celda 23 - Código]**
```python
#Opcional guardar Dataset
#Cambiar el path según preferencia
csv_path = r'/content/resultados_nomic_solo-consultas-originales.csv'
#resultados_df.to_csv(csv_path, index=False)
resultados_df = pd.read_csv(csv_path)

```

## Vizualización de resultados


**[Celda 25 - Código]**
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

plt.title('Distribución de Posiciones para bge-m3 ', fontsize=16)
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


**[Celda 26 - Código]**
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


**[Celda 27 - Código]**
```python
#Opcional guardar Dataset
#Cambiar el path según preferencia
csv_path1 = r'/content/resultados_nomic_solo-consultas-originales.csv'
csv_path2 = r'/content/resultados_bge-m3_solo-consultas-originales.csv'
csv_path3 = r'/content/resultados_beto_solo-consultas-originales.csv'

#resultados_df.to_csv(csv_path, index=False)
resultados_nomic = pd.read_csv(csv_path1)
resultados_bge = pd.read_csv(csv_path2)
resultados_beto = pd.read_csv(csv_path3)


```


**[Celda 28 - Código]**
```python
# Paso 1: Filtrar cada DataFrame para obtener solo las filas con posición -1
nomic_sin_respuesta = resultados_nomic[resultados_nomic['Posicion'] == -1]
beto_sin_respuesta = resultados_beto[resultados_beto['Posicion'] == -1]
#bge_sin_respuesta = resultados_bge[resultados_bge['Posicion'] == -1]

# Paso 2: Encontrar las consultas que están sin respuesta en los tres DataFrames
ids_sin_respuesta = set(nomic_sin_respuesta['idConsulta']) & \
                    set(beto_sin_respuesta['idConsulta']) #& \
 #                   set(bge_sin_respuesta['idConsulta'])

# Paso 3: Crear el nuevo DataFrame con las consultas sin respuesta en los tres modelos
consultas_sin_respuesta = resultados_nomic[resultados_nomic['idConsulta'].isin(ids_sin_respuesta)]

# Paso 4: Seleccionar solo las columnas 'idConsulta' y 'Consulta'
resultado_final = consultas_sin_respuesta[['idConsulta', 'Consulta']]

# Mostrar el resultado
print(resultado_final)
```


**[Celda 29 - Código]**
```python
import pandas as pd

# Asumiendo que ya tienes los DataFrames cargados: resultados_nomic, resultados_beto, resultados_bge

def comparar_modelos(df1, df2, nombre1, nombre2):
    # Filtrar cada DataFrame para obtener solo las filas con posición -1
    sin_respuesta1 = df1[df1['Posicion'] == -1]
    sin_respuesta2 = df2[df2['Posicion'] == -1]

    # Encontrar las consultas que están sin respuesta en ambos DataFrames
    ids_sin_respuesta = set(sin_respuesta1['idConsulta']) & set(sin_respuesta2['idConsulta'])

    # Crear el nuevo DataFrame con las consultas sin respuesta en ambos modelos
    consultas_sin_respuesta = df1[df1['idConsulta'].isin(ids_sin_respuesta)]

    # Seleccionar solo las columnas 'idConsulta' y 'Consulta'
    resultado = consultas_sin_respuesta[['idConsulta', 'Consulta']]

    print(f"Consultas sin respuesta en {nombre1} y {nombre2}:")
    print(resultado)
    print(f"Total de consultas sin respuesta: {len(resultado)}\n")

    # Opcionalmente, guardar el resultado en un nuevo archivo CSV
    # resultado.to_csv(f'consultas_sin_respuesta_{nombre1}_{nombre2}.csv', index=False)

    return resultado

# Realizar todas las comparaciones posibles
nomic_beto = comparar_modelos(resultados_nomic, resultados_beto, "nomic", "beto")
nomic_bge = comparar_modelos(resultados_nomic, resultados_bge, "nomic", "bge")
beto_bge = comparar_modelos(resultados_beto, resultados_bge, "beto", "bge")

# Bonus: Consultas sin respuesta en los tres modelos
ids_sin_respuesta_todos = set(nomic_beto['idConsulta']) & set(nomic_bge['idConsulta']) & set(beto_bge['idConsulta'])
consultas_sin_respuesta_todos = resultados_nomic[resultados_nomic['idConsulta'].isin(ids_sin_respuesta_todos)]
resultado_final_todos = consultas_sin_respuesta_todos[['idConsulta', 'Consulta']]

print("Consultas sin respuesta en los tres modelos (nomic, beto y bge):")
print(resultado_final_todos)
print(f"Total de consultas sin respuesta en los tres modelos: {len(resultado_final_todos)}")

# Opcionalmente, guardar el resultado final en un nuevo archivo CSV
# resultado_final_todos.to_csv('consultas_sin_respuesta_en_todos.csv', index=False)
```


**[Celda 30 - Código]**
```python

```
