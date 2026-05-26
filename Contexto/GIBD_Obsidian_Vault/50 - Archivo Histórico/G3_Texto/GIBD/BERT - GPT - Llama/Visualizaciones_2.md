---
aliases: [Visualizaciones]
tags:
  - grupo/g3_texto
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2024-09-12
origen_zip: GIBD-20260521T205218Z-3-004.zip
ruta_interna: "GIBD/BERT - GPT - Llama/Artículo CONAISI 2024/Notebooks/Visualizaciones.ipynb"
tamanio_bytes: 59205
---

# Notebook: Visualizaciones.ipynb

Ruta interna: `GIBD/BERT - GPT - Llama/Artículo CONAISI 2024/Notebooks/Visualizaciones.ipynb`

---


**[Celda 1 - Código]**
```python
# Importar las bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
import os
```


**[Celda 2 - Código]**
```python
# Cargar el archivo CSV en un DataFrame, si es un DataSet importado 

#Modelos Combinados
#csv_path = r'Combinados\resultados_bge-m3_nomic-embed-text-v1_llama3_1_similares_DIEZ.csv'
#csv_path = r'Combinados\resultados_bge-m3_nomic-embed-text-v1_llama3_1_similares_CINCO.csv'


#Modelos Simples
#csv_path = r'Single Models\resultados_bgem3_gpt4o_similares_DIEZ.csv'
#csv_path = r'Single Models\resultados_nomic_gpt4o_similares_DIEZ_dataset241Fin.csv'
csv_path = r'Single Models\resultados_bge-m3_llama3_1_similares_DIEZ.csv'

file_name = os.path.basename(csv_path)
resultadosCSV_df = pd.read_csv(csv_path)
resultados_df = resultadosCSV_df
print(resultadosCSV_df .head())
```


*Salida:*
```text
idConsulta                                preguntas_similares  Posicion
0           1  Cuales son las condiciones para presentarse a ...         1
1           2  Quién debe regir la convocatoria para la desig...         1
2           3  Cómo se publicita un llamado a concurso?\nCómo...         1
3           4  Qué contenido tienen los anuncios?\n¿Qué infor...         1
4           5  Cuales son los requisitos para los aspirantes ...         1
```


**[Celda 3 - Código]**
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

plt.title(f'Distribución de Posiciones para {file_name}', fontsize=16)
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
1-3: 84.0% (252 preguntas)
4-6: 6.3% (19 preguntas)
7-10: 1.7% (5 preguntas)
No aparece: 8.0% (24 preguntas)
```


**[Celda 4 - Código]**
```python

```
