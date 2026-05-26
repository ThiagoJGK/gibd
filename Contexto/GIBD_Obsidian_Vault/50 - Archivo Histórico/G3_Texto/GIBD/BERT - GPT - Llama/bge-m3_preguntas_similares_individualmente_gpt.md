---
aliases: [bge-m3_preguntas_similares_individualmente_gpt]
tags:
  - grupo/g3_texto
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2024-09-12
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/BERT - GPT - Llama/Notebooks/bge-m3_preguntas_similares_individualmente_gpt.ipynb"
tamanio_bytes: 112217
---

# Notebook: bge-m3_preguntas_similares_individualmente_gpt.ipynb

Ruta interna: `GIBD/BERT - GPT - Llama/Notebooks/bge-m3_preguntas_similares_individualmente_gpt.ipynb`

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
!pip install chromadb -q
```


**[Celda 3 - Código]**
```python
import pandas as pd
from datasets import Dataset
import PyPDF2
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import matplotlib.pyplot as plt
```


*Salida:*
```text
C:\Users\pablo\miniconda3\lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
```

### Seleccionar el modelo de embedding


**[Celda 5 - Código]**
```python
model = SentenceTransformer('BAAI/bge-m3')
```

# Creación de embeddings del reglamento


**[Celda 7 - Código]**
```python
with open("../Reglamento.txt", "r", encoding="utf-8") as file:
    texto  = file.read()

articulos = texto.split("Documento:")
articulos = ["Documento:" + articulo.strip() for articulo in articulos if articulo.strip()]
print(f"Existen {len(articulos)} artículos.")
```


*Salida:*
```text
Existen 674 artículos.
```

# Generación u Obtención de Embeddings desde ChromaDB


**[Celda 9 - Código]**
```python
import chromadb
from chromadb.config import Settings
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import os

# Define the path for data persistence
persist_directory = r"C:\Users\pablo\.chroma_storage"

# Create the persistence directory if it does not exist
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# Initialize ChromaDB client with the correct API implementation setting
client = chromadb.PersistentClient(
    path=persist_directory,
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)
# Attempt to get the "articles" collection, or create it if it doesn't exist
collection_name = "articles"
try:
    collection = client.get_collection(name=collection_name)
    # Check if the collection already has embeddings
    existing_embeddings_count = collection.count()  # Count the number of existing embeddings
    print(f"Collection '{collection_name}' exists with {existing_embeddings_count} embeddings.")
except Exception as e:
    # Collection does not exist or another issue occurred
    collection = client.create_collection(name=collection_name)
    existing_embeddings_count = 0
    print(f"Collection '{collection_name}' does not exist. Created a new one.")

# If the collection is empty, proceed to calculate and add embeddings
if existing_embeddings_count == 0:
    articles_embedding = []
    for i, article in enumerate(articulos):
        article_embedding = model.encode(article)
        articles_embedding.append((i, article_embedding))

    # Add embeddings to the collection
    for i, embedding in articles_embedding:
        collection.add(
            ids=[str(i)],  # IDs must be strings
            embeddings=[embedding.tolist()],  # Correct parameter is 'embeddings', not 'vectors'
            metadatas=[{"article": articulos[i]}]  # Store the original article as metadata
        )
    print(f"Added {len(articles_embedding)} embeddings to the collection '{collection_name}'.")
else:
    print(f"Embeddings already exist in the collection '{collection_name}', skipping embedding calculation.")

```


*Salida:*
```text
Collection 'articles' does not exist. Created a new one.
Added 674 embeddings to the collection 'articles'.
```

# Retrieval: recuperación de articulos más similares


**[Celda 11 - Código]**
```python
def devolver_N_articulos_similares(pregunta, model, client, N):
    # Obtener la colección "articles" del cliente
    collection_name = "articles"
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Error al obtener la colección '{collection_name}': {e}")
        return []

    # Codificar la pregunta usando el modelo y convertir el embedding a una lista
    question_embedding = model.encode(pregunta).tolist()

    # Consultar la colección de ChromaDB con el embedding de la pregunta
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=N
    )

    # Extraer y devolver los artículos relevantes y sus puntuaciones
    similar_articles = [(metadata["article"], score) for metadata, score in zip(results["metadatas"][0], results["distances"][0])]
    return similar_articles

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
Downloading readme: 100%|██████████████████████████████████████████████████████████████| 382/382 [00:00<00:00, 381kB/s]
Downloading data: 100%|████████████████████████████████████████████████████████████| 98.2k/98.2k [00:00<00:00, 157kB/s]
Generating train split: 100%|███████████████████████████████████████████████| 300/300 [00:00<00:00, 1374.29 examples/s]
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

## Evaluación por consulta individual


**[Celda 17 - Código]**
```python
idConsulta = 1
texto_objetivo = df_preguntasEtiquetadas[df_preguntasEtiquetadas['idConsulta'] == idConsulta]['Texto'].values[0]
pregunta_consulta = df_preguntasEtiquetadas[df_preguntasEtiquetadas['idConsulta'] == idConsulta]['Consulta'].values[0]

resultadoArticulosEncontrados = devolver_N_articulos_similares(pregunta_consulta, model, client,10)
resultadoPosicion = funcionArticuloEncontrado(resultadoArticulosEncontrados, texto_objetivo)

print(f"CONSULTA: {pregunta_consulta}")
print(f"POSICION: {resultadoPosicion}")
print("*********************************************")
print(texto_objetivo)
print("*********************************************")


```


*Salida:*
```text
CONSULTA:  Cuales son las condiciones para presentarse a un concurso?
POSICION: 1
*********************************************
Documento:8
Articulo: 9
Capitulo: III - INSCRIPCIÓN

Condiciones de los aspirantes: Para presentarse a Concurso el aspirante debe reunir los siguientes requisitos: a) Tener título universitario de grado o en su defecto acreditar antecedentes excepcionales que lo suplan, y b) no estar comprendido en las causales de inhabilitación para el desempeño de cargos públicos y de faltas a la ética universitaria que se mencionan en el artículo 18 del presente Reglamento.
*********************************************
```


**[Celda 18 - Código]**
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
Documento:8
Articulo: 18
Capitulo: III - INSCRIPCIÓN

Inscripciones múltiples: El aspirante que se presente a más de un Concurso deberá cumplir en cada uno de ellos con todos los requisitos establecidos en esta ordenanza, sin poder remitirse a los escritos o documentos presentados en los otros. Si los Concursos a los que se presenta integran el mismo llamado de una Unidad Académica, basta que el aspirante cumplimente en uno de ellos la exigencia del articulo 10°, refiriéndose en las solicitudes de inscripción de los otros Concursos, en cual agregó la documentación que menciona el citado articulo y siempre que no la haya retirado de la Unidad Académica.

-------------------------------------- 3
Documento:8
Articulo: 12
Capitulo: III - INSCRIPCIÓN

Solicitud de Inscripción: Las solicitudes de inscripción tendrán el carácter de Declaración Jurada y serán presentadas, bajo recibo en el que constará la fecha y hora de recepción, por los aspirantes o personas autorizadas, en la dependencia habilitada en la respectiva Unidad Académica en un (1) original y cinco (5) copias y, además, en soporte digital, con la información básica siguiente: 1) Apellido y nombres, nacionalidad, estado civil, fecha y lugar de nacimiento, clase y número de documento de identidad, domicilio real y constituir el especial dentro de la ciudad asiento de la Unidad Académica o en un radio de hasta 30 km de la misma. 2) Título universitario, con indicación de la Facultad y Universidad que lo otorgó. Los títulos universitarios deberán presentarse en original y fotocopia simple la que será autenticada por personal autorizado por la Facultad. En su defecto podrán presentarse fotocopias legalizadas ante escribano público o juez de paz. 3) Títulos no universitarios, si los tuviere, con indicación de la Entidad que los otorgó, los que deberán presentarse bajo las mismas formalidades indicadas en el inciso anterior. RESOLUCIÓN Nº289/08. 4) La totalidad de los antecedentes docentes y no docentes e índole de las tareas desarrolladas, indicando la institución, el período de ejercicio y la naturaleza de su designación. 5) Nómina de obras y publicaciones, consignando la editorial o revista y lugar y fecha de publicación. Los miembros del jurado podrán exigir que se presenten copias u originales de las publicaciones y trabajos realizados, las que serán devueltas una vez sustanciado el Concurso. 6) Otros antecedentes relacionados con la especialidad, tales como cursos realizados, conferencias dictadas, etc. 7) La totalidad de los antecedentes que hagan a su actuación en Universidades e Instituciones nacionales, provinciales y privadas, registradas en el país o en el extranjero; cargo que desempeñó o desempeña en la Administración Pública o en la actividad privada, en el país o en el extranjero. Los antecedentes de Instituciones extranjeras deberán presentarse acompañados de su traducción al español. 8) Participación en Congresos o acontecimientos similares, nacionales o internacionales. 9) Distinciones, premios y becas obtenidas. 10) Una síntesis de los aportes originales efectuados en el ejercicio de la especialidad respectiva. 11) Una síntesis de la actuación profesional. 12) Otros cargos y antecedentes que a juicio del aspirante pueden contribuir a una mejor ilustración sobre su competencia en la materia en Concurso. Indicación de la dedicación o dedicaciones a que aspira, o bien que el aspirante proponga una distinta a la consignada en el respectivo llamado.

-------------------------------------- 4
Documento:8
Articulo: 58
Capitulo: X - NORMAS GENERALES

La inscripción al Concurso importará para el aspirante su conformidad con las normas de este Reglamento y las específicas dictadas por cada Unidad Académica.

-------------------------------------- 5
Documento:8
Articulo: 61
Capitulo: X - NORMAS GENERALES

Los llamados a Concurso, se harán dentro de lo posible para una cátedra, área, núcleo, asignaturas que integran los núcleos, disciplina, laboratorio, pero no para los cursos en que estos estuvieran divididos, pudiendo cada Facultad adaptar lo aquí dispuesto a su estructura académica.

-------------------------------------- 6
Documento:8
Articulo: 50
Capitulo: VIII - RESOLUCIÓN DEL CONCURSO

Dentro de los treinta (30) días de haberse expedido el Jurado y sobre la base de su dictamen, de las observaciones formuladas por los participantes mencionados en el Articulo 44 y, de las impugnaciones que hubieran formulado los aspirantes, las cuales quedarán resueltas con asesoramiento legal si correspondiere, el Consejo Directivo, expresando los fundamentos podrá: A) Solicitar al Jurado la ampliación o aclaración del dictamen, en cuyo caso aquel deberá expedirse dentro de los ocho (8) días de tomar conocimiento de la solicitud. B) Proponer al Consejo Superior declarar desierto el Concurso. C) Proponer al Consejo Superior dejar sin efecto el Concurso y llamar uno nuevo. D) Aprobar el Concurso y elevarlo al Consejo Superior con las siguientes posibilidades: 1. Proponer al aspirante ubicado en primer término del dictamen único o alguno de los dictámenes del Jurado. 2. Proponer en forma alternativa para cubrir el/los cargo/s a dos (2) aspirantes ubicados en el primer y segundo término del orden de mérito del dictamen o algunos de los dictámenes del Jurado. Para los llamados a Concurso para cubrir más de un (1) cargo realizado de acuerdo al Artículo 58° de este Reglamento, podrá proponer a los aspirantes respetando el orden de mérito elaborado del Jurado conforme a las pautas del párrafo anterior.

-------------------------------------- 7
Documento:8
Articulo: 6
Capitulo: II - PUBLICIDAD

Difusión: La difusión del llamado a concurso estará a cargo del Rectorado quién publicará dentro de los diez (10) días de aprobado el llamado a concurso, haciendo como mínimo un aviso por un (1) día en un diario de circulación nacional y debiéndose publicar por tres (3) días, en un diario de la localidad sede y subsede de la Unidad Académica respectiva. En aquellas localidades en las que la tirada de los diarios sea de frecuencia semanal bastará con una sola publicación. En todos los casos, la publicidad deberá efectuarse entre los diarios de mayor tirada. El rectorado se hará cargo además, de la difusión del llamado a través de los medios de comunicación con que contara y las Unidades Académicas comunicaran asimismo a sus similares afines dentro del ámbito universitario nacional y a los medios de difusión que consideren convenientes. Las respectivas publicaciones se deberán agregar para constancia al expediente de Concurso.

-------------------------------------- 8
Documento:13
Articulo: 273
Capitulo: TÍTULO IV: DE LA EDUCACIÓN SUPERIOR UNIVERSITARIA. CAPÍTULO 4: De las instituciones universitarias nacionales. Sección I: Creación y bases organizativas

El ingreso a la carrera académica universitaria se hará mediante concurso publico y abierto de antecedentes y oposición, debiéndose asegurar la constitución de jurados integrados por profesores por concurso, o excepcionalmente por personas de idoneidad indiscutible aunque no reúnan esa condición, que garanticen la mayor imparcialidad y el máximo rigor académico. Con carácter excepcional, las universidades e institutos universitarios nacionales podrán contratar, al margen del régimen de concursos y solo por tiempo determinado, a personalidades de reconocido prestigio y méritos académicos sobresalientes para que desarrollen cursos, seminarios o actividades similares. Podrán igualmente prever la designación temporaria de docentes interinos, cuando ello sea imprescindible y mientras se sustancia el correspondiente concurso. Los docentes designados por concurso deberán representar un porcentaje no inferior al setenta por ciento (70%) de las respectivas plantas de cada institución universitaria.

-------------------------------------- 9
Documento:8
Articulo: 52
Capitulo: VIII - RESOLUCIÓN DEL CONCURSO

El Consejo Superior podrá solicitar aclaraciones sobre la o las propuestas del Consejo Directivo, en cuyo caso este deberá expedirse dentro de los quince (15) días de tomar conocimiento de la solicitud. En un plazo no mayor de veinticuatro (24) días de recibida la propuesta y/o las aclaraciones de dicho órgano, podrá aceptar la propuesta del Consejo Directivo, o rechazarla. Si la propuesta fuera rechazada el Concurso quedará sin efecto. Dicha resolución se notificará a los interesados y será difundida a través de las carteleras murales y de la página electrónica de la Unidad Académica correspondiente. En los Concursos en los que no se formulen propuestas para la totalidad de los cargos concursados, los que no se provean serán declarados desiertos.

-------------------------------------- 10
Documento:9
Articulo: 73
Capitulo: Reglamento de Concurso Docentes de Antecedentes

Las solicitudes de inscripción se formalizarán por nota dirigida al Decano, tendrán carácter de declaración jurada y serán presentadas por los aspirantes o personas autorizadas por éstos, en sede o subsede de la Facultad que corresponda, en tres (3) ejemplares, bajo recibo en el que constará la fecha de recepción, con la información básica siguiente: 4.1 Apellido y nombres, nacionalidad, estado civil, fecha y lugar de nacimiento, domicilio real, clase y número de documento. 4.2 Títulos con la indicación de la institución que los otorgó. 4.3 La totalidad de los antecedentes docentes (actuación en universidades y Facultades nacionales, provinciales y/o privados registrados en el país o en el extranjero) y los no docentes (cargos desempeñados en la administración pública o en la actividad privada) e índole de las tareas desarrolladas, indicando la institución, el período de ejercicio y la naturaleza de la designación. 4.4 Nómina de obras y publicaciones realizadas por el aspirante, consignando la editorial o revista y el lugar y fecha de publicación. Se podrá exigir que se presenten copias de las publicaciones y los trabajos mencionados las que serán devueltas una vez efectuada la evaluación de los antecedentes. 4.5 Otros antecedentes relacionados con la especialidad tales como: cursos realizados, cursos y conferencias dictados, etc. 4.6 Participación en Congresos y/o Jornadas nacionales y/o internacionales. 4.7 Distinciones, premios y becas obtenidos.
```


**[Celda 19 - Código]**
```python
import pandas as pd

# Cargar el CSV de preguntas similares
similaresCSV_df = pd.read_csv('../Experimentos/Datasets_Preguntas_Similares/GPT_4o_+_Supervised/DataSet_preguntasSimilares_survervisado.csv')
similaresCSV_df.columns = similaresCSV_df.columns.str.strip()

similaresCSV_df
print(similaresCSV_df.columns)
similaresCSV_df["3_preguntas_similares"][0].split("\n")

# Unir los DataFrames en base a la columna 'idConsulta'
df_similares = pd.merge(similaresCSV_df, df_preguntasEtiquetadas, left_on='idConsulta', right_on='idConsulta')
df_similares
```


*Salida:*
```text
Index(['idConsulta', 'Consulta', '10_preguntas_similares',
       '5_preguntas_similares', '3_preguntas_similares'],
      dtype='object')
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
2    Cómo se publicita un llamado a concurso?\n¿Cóm...   
3    Qué contenido tienen los anuncios?\n¿Qué infor...   
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

## Creación de Dataset de resultados

### Código Modificado para iterar por cada pregunta similar del dataset


**[Celda 22 - Código]**
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
        preguntas_similares = row['3_preguntas_similares'].split("\n")
        texto_consulta = row['Texto']

        for pregunta_consulta in preguntas_similares:
            # Obtener los N artículos más similares (N=10 en este caso)
            resultadoPrueba1 = devolver_N_articulos_similares(pregunta_consulta, model, client, 10)

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


*Salida:*
```text
idConsulta                                           Consulta  Posicion
0             1  Cuales son las condiciones para presentarse a ...       1.0
1             1  ¿Cuáles son los requisitos para participar en ...       1.0
2             1  ¿Qué condiciones se deben cumplir para present...       1.0
3             2  Quién debe regir la convocatoria para la desig...       1.0
4             2  ¿Quién tiene la autoridad para manejar la conv...       1.0
..          ...                                                ...       ...
895         299  ¿Qué poderes otorga la autonomía académica e i...       1.0
896         299  ¿Cuáles son las competencias de la autonomía a...       1.0
897         300  Cuales son los deberes de los docentes de las ...       1.0
898         300  ¿Cuáles son las obligaciones de los docentes e...       1.0
899         300  ¿Qué responsabilidades tienen los docentes de ...       1.0

[900 rows x 3 columns]
```


**[Celda 23 - Código]**
```python
resultados_df
```


*Salida:*
```text
idConsulta                                           Consulta  Posicion
0             1  Cuales son las condiciones para presentarse a ...       1.0
1             1  ¿Cuáles son los requisitos para participar en ...       1.0
2             1  ¿Qué condiciones se deben cumplir para present...       1.0
3             2  Quién debe regir la convocatoria para la desig...       1.0
4             2  ¿Quién tiene la autoridad para manejar la conv...       1.0
..          ...                                                ...       ...
895         299  ¿Qué poderes otorga la autonomía académica e i...       1.0
896         299  ¿Cuáles son las competencias de la autonomía a...       1.0
897         300  Cuales son los deberes de los docentes de las ...       1.0
898         300  ¿Cuáles son las obligaciones de los docentes e...       1.0
899         300  ¿Qué responsabilidades tienen los docentes de ...       1.0

[900 rows x 3 columns]
```


**[Celda 24 - Código]**
```python
#Convertir 'Posicion' a enteros, manejando los valores NaN como -1
resultados_df['Posicion'] = resultados_df['Posicion'].fillna(-1).astype(int)
```


**[Celda 25 - Código]**
```python
#Opcional guardar Dataset - Definir la ruta según la computadora a utilizar

csv_path = r'../Experimentos/Resultados/Single Models/Con Consultas Similares Generadas/GPT_4o_Supervised/resultados_bge-m3_gpt4-similares_TRES_preguntas-individuales.csv'
resultados_df.to_csv(csv_path, index=False)

```


**[Celda 26 - Código]**
```python
csv_path= r'../Experimentos/Resultados/Single Models/Con Consultas Similares Generadas/GPT_4o_Supervised/resultados_bge-m3_gpt4-similares_TRES_preguntas-individuales.csv'
file_name = os.path.basename(csv_path)
resultadosCSV_df = pd.read_csv(csv_path)
resultados_df = resultadosCSV_df
print(resultadosCSV_df .head())
```


*Salida:*
```text
idConsulta                                           Consulta  Posicion
0           1  Cuales son las condiciones para presentarse a ...         1
1           1  ¿Cuáles son los requisitos para participar en ...         1
2           1  ¿Qué condiciones se deben cumplir para present...         1
3           2  Quién debe regir la convocatoria para la desig...         1
4           2  ¿Quién tiene la autoridad para manejar la conv...         1
```

## Vizualización de resultados


**[Celda 28 - Código]**
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

plt.title('Distribución de Posiciones para bge-m3 3 Preguntas Similares Realizadas Invididualmente', fontsize=16)
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
1-3: 89.4% (805 preguntas)
4-6: 4.3% (39 preguntas)
7-10: 1.7% (15 preguntas)
No aparece: 4.6% (41 preguntas)
```


**[Celda 29 - Código]**
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
                '''

    # Preguntas no encontradas (Posicion == -1)
    no_encontradas = df[df['Posicion'] == -1]
    if not no_encontradas.empty:
        print("\nPreguntas no encontradas:")
        for _, row in no_encontradas.iterrows():
            print(f"  idConsulta: {row['idConsulta']}")

# Ejecutar la función
mostrar_preguntas_por_posicion(resultados_df)
```


*Salida:*
```text
Resumen de cantidad por posición:
Posición 1: 1094
Posición 2: 173
Posición 3: 72
Posición 4: 35
Posición 5: 22
Posición 6: 12
Posición 7: 10
Posición 8: 5
Posición 9: 7
Posición 10: 2
No encontradas: 68

Preguntas no encontradas:
  idConsulta: 11
  idConsulta: 18
  idConsulta: 18
  idConsulta: 18
  idConsulta: 18
  idConsulta: 18
  idConsulta: 27
  idConsulta: 47
  idConsulta: 47
  idConsulta: 53
  idConsulta: 53
  idConsulta: 53
  idConsulta: 53
  idConsulta: 89
  idConsulta: 97
  idConsulta: 99
  idConsulta: 99
  idConsulta: 99
  idConsulta: 99
  idConsulta: 99
  idConsulta: 134
  idConsulta: 134
  idConsulta: 134
  idConsulta: 151
  idConsulta: 151
  idConsulta: 151
  idConsulta: 151
  idConsulta: 155
  idConsulta: 161
  idConsulta: 163
  idConsulta: 194
  idConsulta: 194
  idConsulta: 194
  idConsulta: 194
  idConsulta: 194
  idConsulta: 198
  idConsulta: 198
  idConsulta: 198
  idConsulta: 198
  idConsulta: 198
  idConsulta: 205
  idConsulta: 206
  idConsulta: 212
  idConsulta: 213
  idConsulta: 221
  idConsulta: 225
  idConsulta: 226
  idConsulta: 230
  idConsulta: 236
  idConsulta: 242
  idConsulta: 242
  idConsulta: 242
  idConsulta: 242
  idConsulta: 247
  idConsulta: 247
  idConsulta: 249
  idConsulta: 249
  idConsulta: 259
  idConsulta: 262
  idConsulta: 267
  idConsulta: 272
  idConsulta: 284
  idConsulta: 284
  idConsulta: 288
  idConsulta: 290
  idConsulta: 290
  idConsulta: 290
  idConsulta: 290
```
