---
aliases: [Test y cadena de aprendizaje basico]
tags:
  - grupo/general
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2024-09-05
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/2024/Test y cadena de aprendizaje basico.ipynb"
tamanio_bytes: 9679
---

# Notebook: Test y cadena de aprendizaje basico.ipynb

Ruta interna: `GIBD/2024/Test y cadena de aprendizaje basico.ipynb`

---


**[Celda 1 - Código]**
```python
# Palindromos
Word = input()
Word_Reversed = Word[::-1]
if Word == Word_Reversed:
    print("Es palindromo")
else:
    print("No es palindromo")
```


*Salida:*
```text
casa
No es palindromo
```


**[Celda 2 - Código]**
```python
# Entendimiento basico de listas
Lista_A = list("ABC")
print(Lista_A)
Lista_B = [1,2,3]
print(Lista_B)
Lista_C = Lista_A[:]
Lista_C.append("4")
print(Lista_C)
Lista_D = Lista_A
Lista_D.append("D")
print(Lista_D)
Lista_E = Lista_A.copy()
Lista_E[:] = []
print(Lista_E)
Lista_A.extend(Lista_B)
print(Lista_A)
```


*Salida:*
```text
['A', 'B', 'C']
[1, 2, 3]
['A', 'B', 'C', '4']
['A', 'B', 'C', 'D']
[]
['A', 'B', 'C', 'D', 1, 2, 3]
```


**[Celda 3 - Código]**
```python
# Entendimiento Basico de condicionales
i=5
while 1-i: #0 is false
    print(i)
    i -= 1
print(f"El ultimo valor es {i}"
f" y el siguiente es {i - 1}")
```


*Salida:*
```text
5
4
3
2
El ultimo valor es 1 y el siguiente es 0
```


**[Celda 4 - Código]**
```python
# Fizz buzz While's version
i=1
while i<=100: #0 is false
    if (i%3 == 0 and i%5 != 0):
      print("Fizz")
    elif (i%5 == 0 and i%3 != 0):
      print("Buzz")
    elif (i%15 == 0):
      print("Fizz Buzz")
    else:
      print(i)
    i += 1
print("Final")
```


*Salida:*
```text
1
2
Fizz
4
Buzz
Fizz
7
8
Fizz
Buzz
11
Fizz
13
14
Fizz Buzz
16
17
Fizz
19
Buzz
Fizz
22
23
Fizz
Buzz
26
Fizz
28
29
Fizz Buzz
31
32
Fizz
34
Buzz
Fizz
37
38
Fizz
Buzz
41
Fizz
43
44
Fizz Buzz
46
47
Fizz
49
Buzz
Fizz
52
53
Fizz
Buzz
56
Fizz
58
59
Fizz Buzz
61
62
Fizz
64
Buzz
Fizz
67
68
Fizz
Buzz
71
Fizz
73
74
Fizz Buzz
76
77
Fizz
79
Buzz
Fizz
82
83
Fizz
Buzz
86
Fizz
88
89
Fizz Buzz
91
92
Fizz
94
Buzz
Fizz
97
98
Fizz
Buzz
Final
```


**[Celda 5 - Código]**
```python
# Fizz buzz for's version
for i in range(1,101):
    if (i%3 == 0 and i%5 != 0):
      print("Fizz")
    elif (i%5 == 0 and i%3 != 0):
      print("Buzz")
    elif (i%15 == 0):
      print("Fizz Buzz")
    else:
      print(i)
    i += 1
print("Final")
```


**[Celda 6 - Código]**
```python
# Palindromos pero en función
def Palindromo (Word):
  Word = Word.lower()
  Word_Reversed = Word[::-1]
  if Word == Word_Reversed:
    return True
  else:
    return False
print("Ingrese una palabra: ")
Word = input()
if Palindromo(Word):
  print("Es palindromo")
else:
  print("No es palindromo")
```


**[Celda 7 - Código]**
```python
# Procedure
def Contar():
  numeros = list(range(1,11))
  for i in numeros:
    print(i)
Contar()
```


**[Celda 8 - Código]**
```python
# Fibbonacci y la recursividad
def Fibbonacci(Actual):
  if Actual <= 1:
    return Actual
  else:
    return Fibbonacci(Actual - 1) + Fibbonacci(Actual - 2)
def Fibbonacci_Hasta_X(x):
  if x <=0:
    x=0
  else:
    for i in range(1,x):
      Aux_Limitante = Fibbonacci(i)
      if(Aux_Limitante<=500):
        print(f"{i}", Aux_Limitante)
      else:
        break
Fibbonacci_Hasta_X(int(input()))
```


**[Celda 9 - Código]**
```python
import random
def Generar_Numero_Aleatorio(Minimo, Maximo):
  return random.randint(Minimo, Maximo)
Numero_Aleatorio = Generar_Numero_Aleatorio(1, 100)
print(f"El número aleatorio generado es: {Numero_Aleatorio}")
```


**[Celda 10 - Código]**
```python
# Imprime el abecedario intercalando minusculas y mayusculas
# range(65,91): Upper
# range(97,123): Lower
Abecedario_A = []
Abecedario_B = []
i=65
while i<=89:
  print(chr(i), chr(i+33), end=" ")
  Abecedario_A.append(chr(i))
  Abecedario_A.append(chr(i+33))
  i += 2
print()
i=97
while i<=121:
  print(chr(i), chr(i-31), end=" ")
  Abecedario_B.append(chr(i))
  Abecedario_B.append(chr(i-31))
  i += 2
print()
print("Abecedario A")
print(Abecedario_A)
print("Abecedario B")
print(Abecedario_B)
```


**[Celda 11 - Código]**
```python
# Continuación de listas
Lista_A = list("Empanada")
# print(Lista_A)
# print(enumerate(Lista_A,0))
# print(list(enumerate(Lista_A,0)))
Lista_B = list("Tarta")
# print(Lista_B)
# list(enumerate(Lista_B))
# print(Lista_B)
Listado_Doble = list([Lista_A,Lista_B])
print(Listado_Doble)
print(Listado_Doble[0])
print(Listado_Doble[1])
print(Listado_Doble[0][0])
print(Listado_Doble[1][0])
print()
Listado_Triple = list([Listado_Doble, [1,2,3]])
print(Listado_Triple)
print(Listado_Triple[0])
print(Listado_Triple[1])
# print(Listado_Triple[2]) => Error
print(Listado_Triple[0][0])
print(Listado_Triple[0][0][0])
print()
Listado_Triple_VersionB = list([Lista_A,Lista_B, [1,2,3]])
print(Listado_Triple_VersionB)
print(Listado_Triple_VersionB[0])
print(Listado_Triple_VersionB[1])
print(Listado_Triple_VersionB[2])
print(Listado_Triple_VersionB[0][0])
print(Listado_Triple_VersionB[0][0][0]) # La "E" es tratada como un array de un solo elemento al ser un string
print(str(Listado_Triple_VersionB[2][0])[0].isnumeric())
```


**[Celda 12 - Código]**
```python
# Más listas
def Listas(Lista_A = None,Lista_B= []):
  Lista_A = Lista_A or []
  Lista_B = Lista_B
  Lista_A.append(1)
  Lista_B.append(2)
  return (Lista_A,Lista_B)
Listado = Listas()
print(Listado)
Listado = Listas()
print(Listado)
# Primer momento: "No te entiendo python"
def Listas(Lista_A = None,Lista_B = []):
  Lista_A = Lista_A or []
  Lista_B = Lista_B
  Lista_A.append(1)
  Lista_B.append(2)
  return (Lista_A,Lista_B)
Lista_C = [1,2,3]
Lista_D = list("ABC")
Listado = Listas(Lista_C,Lista_D)
print(Listado)
Listado = Listas(Lista_C,Lista_D)
print(Listado)
# "Es raro, pero consistente"
```


**[Celda 13 - Código]**
```python
# Factorial de un numero y excepciones
def FactorialNumerico(Numero):
  if Numero == 0:
    return 1
  else:
    return Numero * FactorialNumerico(Numero-1)
Verdadero=True;
while Verdadero:
  try:
    Numero = int(input("Ingrese un número Natural: "))
    if Numero < 0:
      raise ValueError
    else:
      Factorial = FactorialNumerico(Numero)
      print(f"{Numero}! = {Factorial}")
      Verdadero = False
  except:
    print("Error")
```


*Salida:*
```text
Ingrese un número Natural: -7
Error
Ingrese un número Natural: 7
7! = 5040
```
