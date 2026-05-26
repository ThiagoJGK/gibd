---
aliases: [QA_reglamentoConRAG_ORIGINAL]
tags:
  - grupo/g3_texto
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2024-08-29
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/BERT - GPT - Llama/PC del BOX - Notebooks y Archivos/reglamento-utn-chatbot-main/RAG/NOTEBOOKS/QA_reglamentoConRAG_ORIGINAL.ipynb"
tamanio_bytes: 87864
---

# Notebook: QA_reglamentoConRAG_ORIGINAL.ipynb

Ruta interna: `GIBD/BERT - GPT - Llama/PC del BOX - Notebooks y Archivos/reglamento-utn-chatbot-main/RAG/NOTEBOOKS/QA_reglamentoConRAG_ORIGINAL.ipynb`

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
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
```


*Salida:*
```text
<All keys matched successfully>
```

# Creación de embeddings del reglamento


**[Celda 7 - Código]**
```python
with open("_Reglamento.txt", "r", encoding="utf-8") as file:
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
articles_embedding = []
for i,article in enumerate(articulos):
    article_embedding = model.encode(article)
    articles_embedding.append((i, article_embedding))
```

# Retrival: recuperación de articulos más similares


**[Celda 10 - Código]**
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


**[Celda 11 - Código]**
```python
def funcionArticuloEncontrado(lista_articulos_proximos, articulo_correcto):
    for i, (article, _) in enumerate(lista_articulos_proximos):
        if articulo_correcto.strip() == article.strip():
            return i + 1 
    return None  
```

# Sección de pruebas y evaluación 

## Importanción de DataSet de preguntas etiquetadas



**[Celda 14 - Código]**
```python
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

## Evaluación por consulta individual 


**[Celda 16 - Código]**
```python
idConsulta = 1
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


*Salida:*
```text
CONSULTA:  Cuales son las condiciones para presentarse a un concurso?
POSICION: None
*********************************************
Documento:8
Articulo: 9
Capitulo: III - INSCRIPCIÓN

Condiciones de los aspirantes: Para presentarse a Concurso el aspirante debe reunir los siguientes requisitos: a) Tener título universitario de grado o en su defecto acreditar antecedentes excepcionales que lo suplan, y b) no estar comprendido en las causales de inhabilitación para el desempeño de cargos públicos y de faltas a la ética universitaria que se mencionan en el artículo 18 del presente Reglamento.
*********************************************
```


**[Celda 17 - Código]**
```python
for i in range(len(resultadoArticulosEncontrados)):
    print(f"-------------------------------------- {i + 1}")
    print(resultadoArticulosEncontrados[i][0])
    print()  
```


*Salida:*
```text
-------------------------------------- 1
Documento:9
Articulo: 68
Capitulo: 

Apruébese el Reglamento de Concurso docente de Antecedentes con Presentación de Proyectos para la provisión de cargos docentes interinos, el que regirá para la cobertura de espacios curriculares en las distintas carreras que se cursan en esta Facultad, y cuyo texto pasa a formar parte de la presente como Anexo Único.

-------------------------------------- 2
Documento:10007
Articulo: 10120
Capitulo: DE LOS CONSEJOS DE CARRERA

Para ser electo como representantes docentes en el consejo de carrera y para ser electos de los mismos se requiere cumplir las condiciones fijadas en el articulo 17º del presente.

-------------------------------------- 3
Documento:15
Articulo: 442
Capitulo: TÍTULO IV: DE LOS CUERPOS UNIVERSITARIOS. SECCIÓN A: De los Docentes Universitarios

Los Consejos Directivos podrán, cuando ello sea imprescindible, designar temporariamente Docentes Interinos y mientras se sustancie el correspondiente concurso.

-------------------------------------- 4
Documento:10006
Articulo: 10083
Capitulo: Sobre Los Plazos.

La tesis debe ser presentada al jurado con una antelación no menor a 30 días de la fecha de presentación. Las eventuales correcciones o cambios en la presentación conducirán a un nuevo plazo de 30 días u deberán ser informadas al tesista, al Director de Tesis y a la Secretaría Académica en un plazo no mayor a los 20 días a partir de la fecha de recepción de la tesis.

-------------------------------------- 5
Documento:8
Articulo: 5
Capitulo: I - CONVOCATORIA

Las Unidades Académicas mediante resolución de sus Consejos Directivos podrán proponer al Consejo Superior la realización de concursos conjuntamente con otras Unidades Académicas de esta Universidad.

-------------------------------------- 6
Documento:8
Articulo: 4
Capitulo: I - CONVOCATORIA

Cada Unidad Académica mediante Resolución del Consejo Directivo propondrá al Consejo Superior la provisión de cargos de profesores por concurso especificando lo siguiente: a) las cátedras, áreas, núcleos, asignaturas que integran los núcleos, disciplinas, laboratorio a concursar. b) La categoría. c) La dedicación. d) Si la imputación presupuestaria correspondiente al cargo llamado está afectada a la partida Gastos de Personal En todos los casos, el Consejo Superior deberá resolver sobre la solicitud de llamado dentro de los treinta (30) días de recibida.

-------------------------------------- 7
Documento:8
Articulo: 52
Capitulo: VIII - RESOLUCIÓN DEL CONCURSO

El Consejo Superior podrá solicitar aclaraciones sobre la o las propuestas del Consejo Directivo, en cuyo caso este deberá expedirse dentro de los quince (15) días de tomar conocimiento de la solicitud. En un plazo no mayor de veinticuatro (24) días de recibida la propuesta y/o las aclaraciones de dicho órgano, podrá aceptar la propuesta del Consejo Directivo, o rechazarla. Si la propuesta fuera rechazada el Concurso quedará sin efecto. Dicha resolución se notificará a los interesados y será difundida a través de las carteleras murales y de la página electrónica de la Unidad Académica correspondiente. En los Concursos en los que no se formulen propuestas para la totalidad de los cargos concursados, los que no se provean serán declarados desiertos.

-------------------------------------- 8
Documento:8
Articulo: 27
Capitulo: V - DESIGNACIÓN DEL JURADO

Los integrantes del Jurado deberán reunir las siguientes condiciones: Los docentes: a) Ser o haber sido profesor por concurso de categoría no inferior a la del cargo en concurso, habiendo accedido a este por concurso publico y abierto de antecedentes y oposición. b) Por lo menos dos (2) titulares del Jurado deberán pertenecer o haber pertenecido a otras Universidades Nacionales y/o Provinciales. De sus tres miembros titulares, únicamente uno de ellos podrá pertenecer a la UADER. c) Poseer versación reconocida en el área del conocimiento específico o técnico, motivo del Concurso. Los estudiantes: a) Ser alumno regular de la Unidad Académica en cuestión. b) Haber aprobado la materia en Concurso. c) Tener aprobada como mínimo la mitad de la carrera.

-------------------------------------- 9
Documento:11
Articulo: 205
Capitulo: DE LOS CONSEJOS DE CARRERA

Las elecciones de representantes a Consejo de Carrera se realizarán conjuntamente con las de Director de Carrera y Jefes de Departamento.

-------------------------------------- 10
Documento:10006
Articulo: 10067
Capitulo: Del Codirector.

La propuesta sobre la participación de un codirector la deberá presentar por escrito el propio estudiante. En el caso de que se trate de una persona ajena a la Universidad Autónoma de Entre Ríos deberá anexar una copia de su curriculum.
```

## Creación de Dataset de resultados 


**[Celda 19 - Código]**
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


*Salida:*
```text
idConsulta                                           Consulta  Posicion
0             1   Cuales son las condiciones para presentarse a...       NaN
1             2   Quién debe regir la convocatoria para la desi...       1.0
2             3           Cómo se publicita un llamado a concurso?       1.0
3             4                 Qué contenido tienen los anuncios?       7.0
4             5   Cuales son los requisitos para los aspirantes...       1.0
..          ...                                                ...       ...
295         296   Cual es la responsabilidad principal del Esta...       9.0
296         297   Que sucede cuando una carrera no obtiene su a...       1.0
297         298   Quienes seran los organos de coordinacion y c...       7.0
298         299   Que atribuciones tiene la autonomia academica...       NaN
299         300   Cuales son los deberes de los docentes de las...       2.0

[300 rows x 3 columns]
```


**[Celda 20 - Código]**
```python
#Convertir 'Posicion' a enteros, manejando los valores NaN como -1
resultados_df['Posicion'] = resultados_df['Posicion'].fillna(-1).astype(int)
```


**[Celda 21 - Código]**
```python
#Opcional guardar Dataset


#csv_path = r'C:\Users\Usuario\Documents\GIBD\Archivos_CHATBOT_Reglamentacion\resultados_2.csv'
#resultados_df.to_csv(csv_path, index=False)

```

## Vizualización de resultados


**[Celda 23 - Código]**
```python
# RUTAS  EN LA PC DEL LABORATORIO

#csv_path = r'C:\Users\Usuario\Documents\GIBD\RAG\dataSets_Resultados\resultados_beto_llama31_individual.csv'
#csv_path = r'C:\Users\Usuario\Documents\GIBD\RAG\dataSets_Resultados\resultados_nomic_llama31_individual.csv'

#csv_path = r'C:\Users\Usuario\Documents\GIBD\RAG\dataSets_Resultados\resultados_nomic_llama31_similares_TRES.csv'
#csv_path = r'C:\Users\Usuario\Documents\GIBD\RAG\dataSets_Resultados\resultados_nomic_llama31_similares_DIEZ.csv'

```


**[Celda 24 - Código]**
```python
# Cargar el archivo CSV en un DataFrame, si es un DataSet importado 
csv_path = r'C:\Users\Usuario\Documents\GIBD\RAG\dataSets_Resultados\resultados_nomic_llama31_similares_TRES.csv'
resultadosCSV_df = pd.read_csv(csv_path)
resultados_df = resultadosCSV_df
print(resultadosCSV_df .head())
```


*Salida:*
```text
idConsulta                                preguntas_similares  Posicion
0           1  Cuales son las condiciones para presentarse a ...        -1
1           2  Quién debe regir la convocatoria para la desig...         1
2           3  Cómo se publicita un llamado a concurso?\nCómo...         1
3           4  Qué contenido tienen los anuncios?\n¿Cuál es l...         1
4           5  Cuales son los requisitos para los aspirantes ...         1
```


**[Celda 25 - Código]**
```python
# Mirar si es la columna Consulta preguntas_similares, dependiendo el dataSet que se cargue

idConsulta = 1
pregunta_consulta = resultados_df[df_preguntasEtiquetadas['idConsulta'] == idConsulta]['preguntas_similares'].values[0]
print(pregunta_consulta)
```


*Salida:*
```text
Cuales son las condiciones para presentarse a un concurso?
¿Qué requisitos hay que cumplir para participar en una competencia académica?
¿Cuáles son los criterios para acceder a un proceso de selección?
```


**[Celda 26 - Código]**
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

plt.title('Distribución de Posiciones para nomic-embed-text-v1 ', fontsize=16)
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
1-3: 64.3% (193 preguntas)
4-6: 11.3% (33 preguntas)
7-10: 4.7% (14 preguntas)
No aparece: 19.7% (58 preguntas)
```


**[Celda 27 - Código]**
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


*Salida:*
```text
Resumen de cantidad por posición:
Posición 1: 121
Posición 2: 42
Posición 3: 30
Posición 4: 12
Posición 5: 13
Posición 6: 9
Posición 7: 8
Posición 8: 3
Posición 9: 3
Posición 10: 0
No encontradas: 59

Preguntas detalladas por posición:

Posición 1:
```


**[Celda 28 - Código]**
```python

```
