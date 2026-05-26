---
aliases: [Python 3 - De Principiante a avanzado ]
tags:
  - grupo/general
  - estado/archivo
  - tipo/documento
formato_original: docx
fecha_modificacion: 2025-03-31
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Capacitación/Python 3 - De Principiante a avanzado .docx"
tamanio_bytes: 614647
---

# Python 3 - De Principiante a avanzado 

Ruta interna: `GIBD/Capacitación/Python 3 - De Principiante a avanzado .docx`

---

Python
Lo mínimo Indispensable
Tipos de datos básicos:
Integers
Floats (reals)
Booleans (logicals)
Strings
Lists

my_list = [1, 2, "Buckle my shoe"] 
print(my_list) -> devuelve: [1, 2, 'Buckle my shoe']
new_list = [14,2,3,4,5,6,7,8,9,10] 
print(new_list[0]) -> devuelve 14


type() se puede utilizar para comprobar el tipo de datos de las variables.
round(56.78) returns 57
Operadores matemáticos
+ - * y / todo normal. Existe ** para potencia y // para división entera.
El módulo tiene el símbolo de ( %), y el resultado de “modificar” dos números enteros es el resto de la división.
Operadores de comparación
Operadores lógicos
Los dos valores booleanos (lógicos) en Python son Truey False. Utilizamos principalmente operadores lógicos para manipular expresiones creadas con estos valores booleanos.

Los operadores andy orson operadores de cortocircuito. Es decir, el segundo operando solo se evalúa si es necesario. Puede comprobarlo en los ejemplos a continuación.
En la expresión, 2 == 2 or f(x), la primera parte, ( 2 == 2), forma toda la expresión True(porque True o cualquier cosa es True), por lo que la segunda parte, f(x), nunca se evalúa.
De manera similar, en 2 == 5 and f(x), la primera parte, ( 2 == 5), hace que toda la expresión False sea , por lo que la segunda parte, f(x), nunca se evalúa.
Algunos puntos importantes a tener en cuenta al tratar con operadores lógicos:
En una expresión numérica, True tiene el valor 1 y False tiene el valor 0.
En una expresión booleana, todos los valores numéricos cero son False, y todos los valores distintos de cero son True.
El valor especial None indica “ningún valor” y se trata como False.
List
Una lista es una secuencia de valores entre corchetes. Es extremadamente versátil, ya que permite almacenar múltiples elementos de diferentes tipos de datos. Una lista también puede considerarse como un array.
En my_list, que mencionamos anteriormente, acceder a un elemento usando my_list[-1] accedería al último elemento de la lista denominada my_list.
De manera similar, my_list[-2] accedería al segundo último elemento de la lista, y así sucesivamente.
Y para acceder a la lista de un rango de valores: my_list[ 2 : 7] -> devuelve los elementos de la posición 3 a la 6.
Longitud de las listas
Además, se puede acceder a la longitud de una lista mediante una función incorporada llamada len().
String
Una cadena es una secuencia de caracteres entre comillas.
En Python, se pueden usar comillas simples o dobles para las cadenas. También se pueden insertar comillas simples dentro de cadenas entre comillas dobles, y viceversa.
Python generalmente utiliza comillas simples al imprimir resultados.
No existe un tipo de datos de “carácter” separado en Python.
Concatenación 
En Python, las cadenas se pueden concatenar (sumar) con el + operador.
También puede simplemente omitir el +signo entre cadenas literales (entre comillas) y separarlas utilizando espacios en blanco (espacios, tabulaciones o nuevas líneas).
Caracteres especiales en cadenas
En cadenas de Python, también se pueden incluir caracteres especiales mediante el método de " escapar ". Esto se puede hacer colocando una barra invertida ( \) delante del carácter que se desea añadir a la cadena.

Indexación 
En Python, las cadenas se pueden indexar como si fueran listas, lo que significa que se puede acceder a los caracteres individuales de las cadenas simplemente usando la notación de corchetes .
De manera similar a las listas, la indexación en cadenas también comienza desde el índice 0 y también admite la indexación negativa .
Además, si intentamos acceder a un carácter que está fuera del rango ( longitud de la cadena), aparecerá un error.
Y para acceder a la subcadena de un rango de valores: my_cadena[ 2 : 7] -> devuelve los elementos de la posición 3 a la 6.
Declaraciones y sangría
Reglas de sangría #
La sangría es importante en Python. Mientras que otros lenguajes usan llaves en las sentencias, Python usa sangría. Generalmente, existen algunas reglas que deben seguirse al usar la sangría en Python, a saber:
Nota: La sangría estándar es de cuatro espacios. No se pueden mezclar espacios con tabulaciones . Sin embargo, cualquier entorno de desarrollo integrado (IDE) o editor de texto de calidad puede configurarse para añadir cuatro espacios al pulsar la tecla Tab.
Las declaraciones del mismo nivel deben tener exactamente la misma sangría.
Las declaraciones anidadas, como las declaraciones dentro del cuerpo de un bucle, deben tener sangría relativa al inicio del bucle.
La primera línea de un programa no puede tener sangría.
Cada enunciado se escribe en una línea aparte. Sin embargo, si la línea contiene una instrucción sin cerrar (, [, or {, puede continuar en la siguiente.
Si desea colocar dos declaraciones en la misma línea, puede separarlas con un punto y coma ( ;).
Si estás acostumbrado a un lenguaje donde cada sentencia termina con punto y coma, esto funcionará en Python. Simplemente se ve extraño.
Diferentes tipos de declaraciones 
En cuanto a la sintaxis de las sentencias en Python, en lugar de analizarla en abstracto, es más fácil ver ejemplos de cada tipo de sentencia. Los tipos de sentencias que analizaremos son:
Declaraciones de asignación
Declaraciones de impresion
Declaraciones if
Bucles - While,For
Declaraciones de importación

Declaraciones de asignación
Las declaraciones de asignación constan de una variable , un signo igual y una expresión .
z = x + y
Atajos de asignación 
También puedes usar atajos al asignar valores. En Python, esto se hace usando un operador y un solo = signo.
x += y es un atajo para x = x + y
x -= y es un atajo para x = x - y
…y así sucesivamente, para cada uno de los demás operadores binarios.

Operador morsa
En Python, también existe el operador de morsa ( :=). Este es un tipo de operador de asignación que permite asignar un valor a una variable y devolverlo dentro de una expresión.
Puedes ver esto en el siguiente ejemplo, donde el valor actual de n se multiplica por 3, tras lo cual su resultado se asigna de nuevo a n mediante el operador de morsa. Luego, n se suma 1 a [número] y el resultado se transfiere a [número x]. Si, por ejemplo, [número] n es inicialmente 10, el resultado será [número] que n se establece en 30 y x[número] en 31.
Nota: La función del operador "walrus" solo está disponible en la versión >=3.8 de Python. No está disponible en versiones anteriores.
Declaraciones IF
Si se utilizan declaraciones , o declaraciones condicionales, en el proceso de toma de decisiones, donde se evalúa si una condición es verdadera o falsa, se ejecutan diferentes acciones según el resultado.
En Python, las sentencias condicionales utilizan las palabras clave if, elif y else, siendo ambas else y elif componentes opcionales. La sintaxis de las sentencias if en Python generalmente consta de la if palabra clave, una expresión booleana y dos puntos al final.
Cada línea if, elif, o elsedebe terminar con dos puntos ( :). Además, las sentencias bajo el control de cada caso también deben tener sangría.
Nota: Puedes utilizar tantos elifs como quieras, cada uno con su propia condición asociada, y también puedes tener un. final else.
Bucles While y For
Bucles 
Los bucles son un componente de programación que permite ejecutar una o más sentencias repetidamente hasta que se cumpla la condición final. Python tiene dos tipos de bucles: bucles while y for bucles.
Bucles While 
Los bucles While continúan ejecutándose mientras la condición final sea True.
n = 5
while n > 0: # Will keep on running till n is greater than 0
  print(n)
  n -= 1 # Without this, it will get stuck in infinite loop
  print("Blast off!")

El operador "walrus" también se puede usar en while bucles. Si lo usamos en el código anterior, el resultado es el siguiente:
n = 6
while (n := n - 1) > 0: # Decrementing and then checking if greater than 0 together
  print(n)
print("Blast off!")

Bucles for 
Los bucles for se utilizan para iterar sobre cada elemento de tipos de datos de secuencia: cadenas, listas, tuplas, etc.
Los tipos de datos de secuencia son aquellos que le permiten almacenar múltiples valores dentro de ellos en orden.

En el ejemplo que se muestra a continuación, utilizamos la range()función incorporada.

La rangefunción tiene tres formas:
range(b)dará los números 0, 1, 2, etc., hasta, pero sin incluir, b.
range(a, b)dará los números a, a+1, a+2, etc., hasta, pero sin incluir, b.
range(a, b, c)dará los números a, a+c, a+c+c, a+c+c+c, etc., hasta (o hasta), pero sin incluir, b.
Nota técnica : El valor devuelto por un rango no es una lista, sino un iterador (más información sobre esto más adelante). Si necesita una lista, use list(range(...)).
Declaración de impresión 
Las sentencias de impresión son en realidad llamadas a una función llamada print , pero todo el mundo las usa como sentencias. Puedes asignar a print cualquier número de argumentos, y todos se imprimirán en la misma línea con espacios entre ellos.
Argumentos de impresión 
Cada línea impresa termina con un carácter de nueva línea. Sin embargo, si desea que la siguiente sentencia de impresión continúe en la misma línea, puede cambiar el valor del argumento final de la sentencia de impresión, que, como su nombre indica, se imprime al final. El valor predeterminado es un carácter de escape de nueva línea ( \n).
Declaración de importación 
Python tiene algunas funciones predefinidas. Por ejemplo, la abs(n)función te dará el valor absoluto del número n. Si xes una lista, len(x)devolverá la longitud de la lista. O, si xes una cadena, len(x)devolverá el número de caracteres de la cadena.
Hay muchas más funciones disponibles, pero no están integradas. Para usarlas, debes importarlas desde un módulo. Un «módulo» es simplemente un archivo que contiene código.
Por ejemplo, la función raíz cuadrada, sqrt(), debe importarse desde el math módulo. 
from math import *
Esto importará todo desde math. Ahora, si quieres la raíz cuadrada de un número x, simplemente escribe sqrt(x).

Entrada del Usuario
Introducción a input
Python tiene una amplia gama de excelentes funciones integradas, como el len(my_list)comando, que devuelve la longitud de una lista .
De manera similar, para solicitarle información al usuario , puede utilizar la función incorporada input(_prompt_).
Nota : si omite el mensaje, el usuario se quedará mirando una pantalla en blanco y se preguntará por qué el programa no hace nada.
Un ejemplo de esto puede ser:

El resultado de una llamada a input siempre es una cadena . Si espera un entero o un valor de punto flotante, puede usar las funciones int``` float para convertir la cadena al tipo deseado. Por ejemplo:

Si el usuario escribe algo que no se puede convertir en un entero, se producirá un error. Más adelante en el curso, abordaremos el manejo de errores.
Ejemplo de función de entrada 
Para entender la función incorporada input, veamos un ejemplo.
A continuación, verá un botón azul " Ejecutar " en la pantalla. Haga clic en él para ejecutar el siguiente código (el ejemplo anterior):

Aquí aparecerá el mensaje " ¿Cuál es tu nombre? ". Ingresa tu name nombre. Después, verás un mensaje con tu name nombre y un mensaje de " Hola ".
Funciones
Las funciones son una serie de instrucciones que se utilizan para realizar una operación específica. En Python, existen dos tipos de funciones: funciones integradas y funciones personalizadas, estas últimas las analizaremos en esta lección.
Para definir una función en Python, usamos la defpalabra clave junto con el nombre de la función, una lista de parámetros entre paréntesis y dos puntos. A continuación, el cuerpo de la función, con sangría.

En funciones, la return palabra clave se utiliza para detener la ejecución de las declaraciones de la función y le permite devolver un valor de la función.
Este valor devuelto se puede almacenar en una variable y usarse como cualquier otra variable.
En el ejemplo anterior, podríamos haber incluido la instrucción de retorno final en una else parte, pero no es necesario. Se ejecutará si la if-elif instrucción no realiza ninguna acción.
En el ejemplo anterior, la segunda línea de la función es una cadena entre comillas triples. Este tipo de cadena, en esta ubicación, se utiliza para documentar el propósito de la función.
Las cadenas entre comillas triples pueden abarcar varias líneas. Documentar cada función de esta forma es opcional, pero muy recomendable. Cualquier función con una cadena de documentación help (function_name) imprimirá dicha cadena.
Cada función devuelve un valor. Si no se especifica un valor de retorno , la función devolverá None. (Este es un valor válido de tipo NoneType , por lo que se puede asignar a una variable o preguntar si una variable es igual a él).
El siguiente ejemplo demuestra esto.
Funciones locales 
Una función puede definirse dentro de otra función y se vuelve local a la función en la que ha sido definida.
Guía de estilo y convenciones de codificación de Python
Comentando 
Los comentarios son una de las partes más importantes de su código porque le dan una idea exacta de lo que se supone que debe hacer su código.
En Python, los comentarios de una sola línea comienzan con un signo numeral, # y continúan hasta el final de la línea.
Aunque no existe una sintaxis específica para comentarios de varias líneas en Python, puedes insertar un signo de numeral para cada línea de comentario o utilizar cadenas entre comillas triples ( '''), al principio y fin del comentario.
Orden de ejecución 
No todo en un programa Python tiene que estar dentro de una función. Puedes (y normalmente lo harás) tener algunas sentencias de " nivel superior ". Los programas muy cortos pueden consistir completamente en sentencias de nivel superior sin definiciones de funciones.
Los programas Python se evalúan desde el principio (primera línea) hasta el final (última línea). Las sentencias de nivel superior se evalúan conforme ocurren. Al evaluar la definición de una función ( def), esta se define, pero no se ejecuta hasta que otra sentencia la invoque.
Las funciones se definen dinámicamente, es decir, cuando def se evalúa a. Como en la mayoría de los lenguajes, no existen restricciones en el orden léxico de las funciones (cuáles aparecen primero en una lista), pero deben definirse antes de ser invocadas.
Es común que un programa conste de una colección de funciones, y que su última línea sea una única llamada de nivel superior a una función principal. Esta función principal suele llamarse main, pero no siempre.
Mejores Herramientas
Objetos
En esta lección, veremos varios tipos diferentes de objetos.
Notación de objeto 
Las listas y cadenas (que ya has visto), así como los conjuntos y diccionarios (que veremos próximamente), son objetos. Existe una terminología específica para referirse a ellos:
Un método es una función definida en un objeto.
No se "invocan" métodos. Se envían mensajes a un objeto. Esto es complejo, así que prefiero decir que se habla con objetos.
Junto con esto, hay una notación especial para el uso de objetos:
Nombra el objeto al que estás hablando.
Ponga un punto ( .).
Escriba el nombre del método, junto con todos los argumentos (separados por comas y entre paréntesis).
Por ejemplo, si s es la cadena Hello World, entonces s.lower()es la cadena hello world. Puedes pensar en esto como decir: "Oye s, dame una versión en minúsculas de ti mismo".
Los tipos de objetos que veremos son:
Listas
Tuplas
Conjuntos
Diccionarios
Strings

Lista
Definición 
Una lista de Python es una secuencia indexable de valores, que no necesariamente son del mismo tipo. El primer índice es 0. Puedes crear una lista escribiendo una secuencia de valores separados por comas y encerrados entre corchetes [].
s = ["Mary", 23, "A+"]
Listas de tamaño arbitrario 
En Python, se puede crear una lista de cualquier tamaño multiplicándola por un entero. El entero debe estar a la derecha del operador *.
none_list = [None] * 100 # result is 100 None’s
num_list = [2,4] * 3 # result is [2,4,2,4,2,4]

Efecto de copia superficial en listas 
Supongamos que tenemos una variable llamada x. Si x es una lista; y se indica y = x, entonces y se convierte en otro nombre para la misma lista, no en una copia de x. Es decir, cualquier cambio que se realice en y, también cambia x, y viceversa.
Funciones integradas 
Las listas tienen varias funciones integradas útiles, dos de las cuales se mencionan a continuación:
len(my_list)devuelve el número de elementos en la lista.
sorted(my_list)devuelve una copia de la lista con los elementos en orden ascendente.

Métodos incorporados 
Las listas también tienen varios métodos integrados extremadamente útiles, dos de los cuales se mencionan a continuación:
my_list.append(value)agrega valor al final de my_listy devuelve None.
my_list.pop()Elimina el último elemento de my_listy lo devuelve.
Nota : También puede utilizar el pop()método para eliminar un elemento de cualquier índice dado, ya que toma un índice como uno de sus argumentos my_list.pop(given_index):.
Rebanar 
Python permite tomar "segmentos" de una lista. Los segmentos generan una nueva lista que contiene algunos de los elementos de la lista original . Si my_listes una lista y iy json enteros, entonces
my_list[i:j]es una copia de los elementos de my_list[i] hasta, pero sin incluir, my_list[j] .
my_list[i:]es una copia de los elementos que comienzan en my_list[i]y continúan hasta el final .
my_list[:j]es una copia de los elementos comenzando desde my_list[0] hasta, pero sin incluir, my_list[j] .
my_list[:]es una copia de la lista completa .

my_list = [2,5,8,10,15,20]
first_slice = my_list[2:5]

También puede usar la notación de porción en el lado izquierdo de un operador de asignación. Si asigna una lista de valores a una porción, estos valores reemplazan los valores de la porción. La lista de valores no tiene que tener la misma longitud que la porción que reemplaza.
Por ejemplo, si x = [0, 1, 2, 3]y asigna x[1:3] = [11, 22, 33], x obtiene el valor [0, 11, 22, 33, 3] como resultado.
Consejo profesional: puedes tratar una cadena como una lista de letras, por ejemplo, abcdef[2:5]es cde.

Listas multidimensionales 
Como cualquier valor se puede incluir en una lista, se puede crear una lista de listas. 
grades = [["Mary", "A+"], ["Donald", "C-"]]
accessing_inner = grades[1][0]

Para obtener elementos individuales, debes indexar las listas externa e interna por separado: grades[1]es ["Donald", "C-"], entonces grades[1][0] es Donald.

Precaución: No intente crear una lista de listas con una construcción como [[None] * 3] * 2. Parecerá que funciona, pero cada valor de la lista externa será una referencia a la misma lista interna.

También puedes crear una lista bidimensional usando el método de tamaño arbitrario que explicamos anteriormente.
x = [0] * 3 
for i in range(0, 3): 
    x[i] = [0] * 5

En el ejemplo anterior, la asignación inicial crea una matriz de ceros , [0,0,0]. Luego, el bucle reemplaza cada cero con una matriz de ceros, [[0,...0], [0,...0], [0,...0]].
Precaución: No intente crear una lista de listas con una construcción como [[None] * 3] * 2. Parecerá que funciona, pero cada valor de la lista externa será una referencia a la misma lista interna.

Tuplas
Una tupla consta de dos o más valores separados por comas y encerrados entre paréntesis , por ejemplo, ("John", 23).
Las tuplas son útiles para mantener un número pequeño de valores juntos como una sola unidad. Por ejemplo, podrías escribir una función que devuelva las coordenadas xy de un objeto, o los valores máximo y mínimo de una lista.
def get_largest_smallest(my_list):
    largest = max(my_list) 
    smallest = min(my_list) 
    return (largest, smallest) 

my_list = [1,8,6,5,3,2]
print(get_largest_smallest(my_list)) # La salida será (8, 1)

Función incorporada de Divmod 
La función incorporada divmod toma dos números, numerador y denominador, y devuelve una tupla.
Por ejemplo, imaginemos que tenemos dos números, num1y num2, y queremos obtener el resultado de los operadores de división de enteros (//) y módulo (%). Una buena manera de obtenerlos es usar la función divmod, ya que el resultado que devuelve tiene el mismo formato que: (num1 // num2, num1 % num2).
divmod(20, 7) # Esto devuelve (2, 6)
Accediendo a tuplas 
Hay muchas cosas que se pueden hacer con las tuplas en Python. Una de ellas es acceder a los elementos de las tuplas, lo cual, al igual que en las listas, se puede hacer mediante la notación de [] corchetes.
x = (1, 2, 3) 
z = x[1:] # z será (2, 3)

De manera similar, también se puede acceder a los elementos de una tupla usando slicing , que devuelve una nueva tupla que tiene algunos de los elementos de la tupla original .
Empaquetado y desempaquetado de tuplas 
Cuando creamos y asignamos valores a una tupla, este proceso se denomina empaquetado de tuplas. Esto se puede observar en los ejemplos anteriores, donde creamos una tupla llamada x y le asignamos los valores 1, 2 y 3.
Además, las tuplas también pueden descomprimirse en sus elementos individuales mediante el operador de asignación, =. Esto se puede observar en los ejemplos a continuación:
x = (1, 2, 3)
(a, b, c) = x # unpacking x -> a is now 1, b is 2, and c is 3
(a, b, c) = (1, 2, 3) # unpacking x -> a is now 1, b is 2, and c is 3
(a, b, c) = "xyz" # same case with strings -> a is now "x", b is "y", and c is "z"

Nota: No se puede asignar a una tupla. Por ejemplo, si x es (1, 2, 3), entonces x[0] = 5 no es permitido.
Nota técnica: Se puede crear una tupla unitaria colocando paréntesis alrededor de un valor único y una coma después, como en este ejemplo: (3,). Sin embargo, esto probablemente no sea muy útil en general.
Conjuntos
Un conjunto es una colección de valores que no necesariamente son del mismo tipo. Para escribir un conjunto, use llaves alrededor de una lista separada por comas . Vea algunos ejemplos de conjuntos a continuación:	int_set = {1, 2, 3, 4, 5}		s = {10, "ten", "X"}

Conjunto vacío 
En Python, un conjunto vacío (sin elementos) no se puede escribir como. En su lugar, se utiliza {} la función set().		emp_dict = {} 			emp_set = set()

Establecer propiedades 
Los conjuntos tienen dos propiedades importantes:
Los conjuntos no tienen elementos duplicados. Un valor puede estar en el conjunto o no . Los conjuntos proporcionan los operadores `in` y `not in`, que permiten comprobar si hay duplicados. Dado s = {10, "ten", "X", "X"}, 10 in s será true y 5 en s será false.
El orden de los elementos de un conjunto no está especificado , por lo que no es posible indexarlos en un conjunto. Para s = {10, 'ten', "X", 25} no se puede hacer s[1].

Acceder a los elementos del conjunto 
Puedes usar un forbucle para recorrer cada elemento del conjunto por turno, siempre y cuando no te importe el orden en que se procesan los elementos.
Si el orden de los elementos es realmente importante, utilice una lista en lugar de un conjunto.
s = {10, 'ten', "X"} 
for elem in s: 
  print(elem)

Establecer conversiones 
En ocasiones, puede ser necesario realizar conversiones entre conjuntos y otros tipos de datos. Esto se puede realizar fácilmente con las funciones integradas de conversión de tipos de datos.
Por ejemplo, para convertir entre listas y conjuntos, simplemente puede utilizar las funciones list() y set().		set_to_list = list(my_set)			list_to_set = set(my_list)
Operaciones de conjunto 
Al igual que los conjuntos matemáticos, los conjuntos en Python también pueden realizar diversas operaciones matemáticas. Dado que los conjuntos son objetos, es necesario acceder a estas operaciones mediante la notación de objetos.
Las siguientes son algunas de las operaciones matemáticas que realizan los conjuntos:
set1.union(set2)devuelve el conjunto de elementos que están en set1 set2 , o en ambos .
set1.intersection(set2)devuelve el conjunto de elementos que están tanto en set1 como en set2.
set1.difference(set2)devuelve el conjunto de elementos que están en set1 pero no en set2 .
set1.symmetric_difference(set2)devuelve el conjunto de elementos que están exactamente en uno de los conjuntos.
set1.issubset(set2)devuelve Truesi cada elemento de set1también está en set2 .
set1.issuperset(set2)devuelve Truesi cada elemento de set2también está en set1 .

set1.union(set2)			set1.intersection(set2)		set1.difference(set2)
Uso de conjuntos con operadores de comparación 
En Python, también puedes usar los operadores de comparación ( <, <=, ==, !=, >=, >) para probar las relaciones de subconjunto/superconjunto y de igualdad/desigualdad entre dos conjuntos.
print(x <= y) # returns True as x is a subset of y - elements of x in y 
print(x == y) # returns False as x is not equal to y

Agregar o eliminar elementos de un conjunto 
También se pueden añadir o eliminar elementos de un conjunto mediante las funciones integradas de Python. A continuación, se ofrece una descripción general de estas funciones integradas:
set.add(element) Se agrega elementa set. Si element ya está en set, esto no hace nada.
set.discard(element) Se elimina element de set si está presente. Si element no está en set, no hace nada.
set.remove(element) elimina element de set, o genera un KeyError si element no está en set.
set.pop() Elimina y devuelve un elemento arbitrario set de , o lanza un KeyError si set está vacío. El código no debe depender de que se devuelva un elemento específico.
set.clear() elimina todos los elementos de set.

Diccionario
Un diccionario (de tipo dict) ofrece una forma rápida de buscar información. Los diccionarios se suelen usar cuando se desea crear una asignación de claves a valores sin tener que mantener su orden. Además, si se desea acceder a los elementos al instante sin tener que recorrer toda la colección, los diccionarios son la solución ideal.
Un diccionario consta de cero o más pares clave-valor entre llaves { }. Por ejemplo, podría definir un diccionario de "guía telefónica" de la siguiente manera:
telf  =  { "Alice" : 5551212 ,  "Jill" : 5556789 ,  "Bob" : 5559999 }
Encontrar elementos en diccionarios 
Una vez que tenga un diccionario, hay tres maneras de buscar algo en él. En los siguientes ejemplos, preste especial atención a dónde ( )se usan los paréntesis y los corchetes [ ]:
En el ejemplo que se da a continuación:
phones["Jill"] devolverá 5556789. Si no se encuentra la clave, por ejemplo si intenta , se phones["Xavier"] producirá  un KeyError .
phones.get(key)Funciona como phones[key], excepto que si no se encuentra la claveNone, se devuelve el valor .
phones.get(key, default_value)devuelve el valor asociado con la clave , o el valor predeterminado si no existe dicha clave.


Agregar, eliminar y actualizar valores en diccionarios 
Puedes agregar algo a un diccionario usando el operador de asignación .
phones = {"Alice":5551212, "Jill":5556789, "Bob":5559999}
phones["Xavier"] = 5556666 #esto agrega a "Xavier":5556666 al diccionario
Esto también funciona para cambiar el valor asociado a la clave. Por ejemplo, en el ejemplo anterior, podría actualizar el número de teléfono de alguien de esta manera: 
phones["Alice"] = 5554263
Para eliminar algo de un diccionario, utilizamos un comando especial llamado del.
El del comando generará un valor Key Error si key no está en el diccionario, así que compruébelo antes de usar del. Puede usar los operadores `in` y `not in`para comprobar si está en el diccionario.
Existe orden variable de los elementos en los diccionarios. Al igual que los conjuntos, los elementos de un diccionario se almacenan en el orden que Python considere más adecuado. Esto puede variar de una implementación a otra.
Propiedades clave del diccionario 
Las claves de un diccionario deben ser únicas y solo se puede asociar un valor a cada clave. Por otro lado, los valores no tienen por qué ser únicos. Muchas claves pueden tener el mismo valor.
No todo puede usarse como clave. Las claves deben ser inmutables. Esto es un tema largo, pero en resumen: cadenas, números y booleanos son buenas claves. Ningún objeto cuyos componentes puedan cambiar (como listas, conjuntos y diccionarios) puede usarse como clave.
Métodos de String
Introducción a los métodos de cadena 
Una cadena es otro tipo de objeto en Python. La mayoría de las operaciones con cadenas implican el uso de métodos, no de funciones. Tienen la forma:
string.method(arguments)
División de cadenas 
Para dividir cadenas en Python, usamos el split()método . Aquí, el string1.split(string2)comando devuelve una lista de las subcadenas de string1que están separadas por string2. El string2argumento es totalmente opcional. Si se omite, se usa un espacio como separador.     mystr_two.split(',')
Uniendo cadenas 
De forma similar a la división de cadenas, también puedes unir varias cadenas con el join()método . Aquí, string.join(list_of_strings)devuelve una sola cadena, con los elementos list_of_stringsseparados por string.
Cadenas formateadas 
Una cadena formateada, o f-string , lleva el prefijo f o F. En una f-string, cualquier expresión entre llaves {} se reemplaza por su valor.
pi = 3.141592653589793
print(f'pi is {round(pi, 4)}.')

Bucles para objetos
Recorriendo listas 
Si quieres hacer algo con cada elemento de una lista, puedes usar un forbucle. Un bucle simple forse ve así:
for e in my_list:
 print(e)
Si necesita trabajar no solo con los elementos de la lista, sino también con su posición en la lista, puede utilizar una versión más complicada del bucle for:
for i in range(0, len(my_list)):
 print(my_list[i], "is at", i)
Si tienes experiencia en C, C++ o Java, podrías sentirte tentado a usar la versión más compleja para todo. Sin embargo, eso no te ayuda. Cada versión tiene sus propios usos.
Recorriendo conjuntos 
Sólo la forma simple del forbucle funciona para los conjuntos (ya que los conjuntos no son indexables):
Recorriendo los diccionarios 
El bucle simple también funciona con diccionarios. Sin embargo, en este caso, lo que se asigna a la variable del bucle es solo la clave, no el par key:value completo:
for k in my_dict:
 print(k)
Se pordía utilizar el método key del diccionario también dentro del bucle para múltiples tareas.

Manejo de excepciones
Introducción al manejo de errores 
Los errores ocurren. Un programa puede intentar dividir un número entre cero, enviar un mensaje a Noneo leer un archivo que no existe. Cuando esto sucede, Python lanza una excepción .
Cada tipo de excepción tiene un nombre. En Python, casi no hay distinción entre un "error" y una "excepción". Cualquiera de los dos, si no se gestiona, provocará la finalización del programa.
Si sabe dónde es probable que ocurra un error, puede solucionarlo. Por ejemplo, podría usar la inputfunción para solicitar al usuario un entero. Lo que escriba el usuario se devolverá como una cadena, que puede convertir a un entero usando la intfunción. Esto funciona, a menos que el usuario escriba algo que no sean dígitos, en cuyo caso obtendrá un ValueError.
Para resolver este problema, se puede usar la instrucción try-except, o try-except-finally. Su forma general es la siguiente:
try:
 # code that could go wrong
except SomeErrorType:
 # what to do if it goes wrong
finally:
 # what to do afterwards
La parte finally del código anterior es opcional. Si está presente, se ejecutará independientemente de si se produce el error.
Funcionalidad de try-except 
A continuación, se presentan algunos puntos importantes para recordar:
Si se produce un error entre tryy except, el control pasa inmediatamente a la exceptpieza. El código restante en la trypieza no se ejecutará.
El código de una parte finally siempre se ejecutará. Si el código de la parte try o exceptejecuta una returninstrucción, finally se ejecutará antes de que la función regrese.
Si se produce un error en una función, pero este no se encuentra dentro de una sentencia try-except, la función regresa inmediatamente al punto donde se llamó. Si la llamada a la función se realizó dentro de una sentencia try-except, la excepción se gestiona allí. De lo contrario, la función también regresará inmediatamente.
De esta manera, cada función regresa a la función que la invocó con la excepción. Si no la detecta, se pasa a la siguiente llamada en la secuencia de llamadas, y se repite hasta que una de ellas la detecta. Si nunca se detecta la excepción, el programa termina con un error.
Cosas para recordar 
Python define una gran cantidad de tipos de excepción. A menos que desee clasificar diferentes tipos de excepciones y hacer algo distinto para cada uno, puede usar el tipo catch-all Exception. Normalmente, lo difícil de gestionar errores es saber qué hacer cuando ocurren.
Por último, try-except está destinado a detectar errores y no debe utilizarse para otros fines.
Conversiones de tipos
Funciones de conversión de tipos de datos 
Para convertir un valor de un tipo de dato a otro, a menudo es posible usar el nombre del tipo de dato deseado como si fuera una función. Por ejemplo:

Nota: Al trabajar con tipos de datos booleanos, lo siguiente se considera falso: False, None, 0, 0.0 y la cadena vacía.
Otros tipos de datos que pueden usarse como funciones de conversión son list, set, tupley dict(diccionario). Algunos puntos importantes a recordar son:
Cuando conviertes de a dicta cualquiera de los otros tipos, solo obtienes las claves, no los valores.
Puede convertir un list, set, o tupleen un dict solo si los elementos están agrupados de dos en dos, como una lista de 2-tuplas (tuplas con dos elementos).
Los conjuntos no tienen un orden intrínseco, por lo que la conversión hacia o desde un conjunto no necesariamente preserva el orden de los elementos.
Diferencia entre las funciones type y isinstance
La función type(x)devolverá un valor comparable al nombre de un tipo (no a una representación de cadena de ese nombre). Por ejemplo, la prueba type(5) == int devolverá True, pero type(5) == "int" devolverá False.
La función isinstance(value, type) comprueba si value es del tipo con nombre type. typePuede ser una tupla de tipos con nombre. Vea un ejemplo a continuación:
Alcance
Introducción al alcance 
El ámbito de un nombre es la parte del programa donde este tiene significado y puede usarse. Las reglas de ámbito de Python son inusuales porque siguen una "regla LEGB": Local, Cerrado, Global, Integrado.
Para entender esta regla, es útil recordar que:
Las variables se definen asignándoles un valor.
Las funciones/métodos pueden estar anidados dentro de otras funciones/métodos.
Al tratar el alcance de una variable en Python, tenga en cuenta el siguiente punto:
Las variables pueden declararse globales o no locales mediante una declaración con la forma global var1,...,varN o nonlocal var1,...,varN. La declaración debe preceder a cualquier uso de la variable.
Tipos de ámbitos 
A continuación, se muestran cuatro tipos diferentes de ámbito presentes en Python:
Local: una variable es local a una función si se le asigna un valor dentro de esa función y no se declara de otra manera.
Adjunto: una variable en una función puede hacer referencia a una variable declarada en una función envolvente, siempre que se cumpla alguna de las siguientes condiciones:
A la variable no se le asigna un valor en esta función.
Así lo declara en un nonlocal comunicado.
Global: Una variable declarada en el nivel superior (no en una función ni en un método) es global y puede referenciarse en todo el programa. Una variable en una función puede hacer referencia a una variable global, siempre que se cumpla alguna de las siguientes condiciones:
A la variable no se le asigna un valor en esta función ni en una función adjunta.
Así lo declara una declaración global.
Integrada: Una variable no declarada en el programa podría serlo. Algunos ejemplos son list, printy divmod.

Buenas prácticas 
Cuando una variable no es local (está encerrada) o es global, es una buena documentación declararla como tal, incluso cuando a la variable no se le asigna un valor dentro de la función.
Las variables globales generalmente se consideran indeseables y deben evitarse siempre que sea posible. Esto no es tan sencillo en Python como en otros lenguajes.
Algunos programadores usan la convención de comenzar el nombre de cada variable global con un guion bajo. Esta convención no afecta el alcance real de la variable.

E/S de archivos
Introducción al manejo de archivos 
Para leer o escribir un archivo, debes hacer tres cosas:
Abrir el archivo
Utilizar el archivo (lectura o escritura)
Cerrar el archivo
Abrir archivos en Python 
El file = open(file_name, mode)comando se abre y devuelve (en la variable file) una referencia al archivo indicado. En este caso, modesería uno de los siguientes:
rpara leer el archivo (este es el valor predeterminado si modese omite).
wpara borrar y escribir el archivo.
apara agregar al final de un archivo existente .
r+Tanto para leer como para escribir .
rb, wb, ab, y rb+para realizar entrada/salida binaria .
Un ejemplo de esto puede ser:
file = open(file_name) # mode omitted (r used by default) so file is being read     
file = open(file_name,'w') # w used here so writing to file
File methods in Python
The following are some of the methods that come along with the file object in Python:
file.read() will read in and return the entire file as a single string, including any newline characters.
file.readline() reads and returns a line of text, including the terminating newline (if any). If the empty string is returned, the end of the file has been reached.
file.readlines() reads the entire file and returns it as a list of strings, including terminating newlines.
file.write(string) writes the string to the file, and it returns the number of characters written.
An alternative of using the read, readline and readlines methods can be iterating over them using a loop. For example:
for line in file:
  statements
The above code will read each line in the file, assign it to the variable line, and execute the statements. Here, the statements are indented as usual.
Closing files in Python
Closing the file is a mandatory step as leaving the file open when you are done with it is likely to cause problems. The file.close() command is used for closing the file.
Un ejemplo de esto puede ser:
file = open(file_name) # file being read
file.close() # closing the file
Una excelente alternativa para cerrar archivos en Python es utilizar el withoperador .
with open(file_name) as file: 
  statements
El código anterior abrirá el archivo, lo ejecutará una statements vez y cerrará automáticamente el archivo sin usar el close método explícitamente.
Nota: Dado que statements solo se ejecutan una vez, suelen consistir en un bucle para leer y procesar cada línea por turno. La sangría es la habitual, es decir, statements se sangran debajo de la línea with.
Ejemplo de archivos 
El siguiente es un ejemplo de lectura de un archivo de texto, impresión de su contenido y cierre del archivo:
file = open('input.txt') # opening and reading file
data = file.read() # reading entire file as single string
print(data) # printing the contents of file
file.close() # closing the file

print('-----------------------------------------')

# using with operator to write to file
with open('input.txt', 'w') as file: # added the w mode in open
  file.write('Hey! New message got written right now') # using write method to write to file

# using with operator to read the text file contents after writing
with open('input.txt', 'r') as file:
  data = file.read()
  print(data)
Cosas a recordar al trabajar con archivos 
Por razones históricas, el carácter o caracteres utilizados para indicar un salto de línea ( \n) no son los mismos en archivos de Windows, archivos antiguos de Macintosh y archivos de Mac OS X/Linux. Al leer o escribir archivos de texto (archivos no binarios), los finales de línea específicos de la plataforma se convertirán automáticamente a la plataforma actual.
Leer o escribir archivos binarios como si fueran de texto los corromperá. En cambio, los archivos de texto se pueden leer y escribir como binarios sin sufrir daños.
Pickling
Introducción a la serialización 
La serialización (a veces llamada marshalling ) de un objeto lo convierte en un flujo lineal de bytes. Esto puede realizarse para guardar un objeto en un archivo o para transmitirlo a otro proceso. El flujo de bytes puede deserializarse ( desmarshalling ) para reconstruir el objeto original.
La forma más común de serializar objetos de Python se denomina «picking» . Python también puede usar JSON y XML para la serialización.

Los valores de Python y la mayoría de los objetos integrados se pueden procesar en picking, incluyendo las clases definidas por el usuario en el nivel superior de un módulo. Los objetos recursivos e interconectados también se pueden procesar en picking. Sin embargo, los generadores , las funciones lambda , las conexiones a bases de datos y los hilos son algunos de los elementos que no se pueden procesar en picking.
Métodos de decapado y desdecapado 
Para descompilar o descompilar objetos, primero debe importar su módulo. Esto se puede hacer con el import picklecomando . Ahora, puede usar los siguientes métodos:
pickle.dump(object, file)se guarda objecten el file, que debe abrirse en wbmodo (escritura binaria).
variable = pickle.load(file)reconstruye el objeto previamente escrito en file, el cual debe abrirse en rbmodo (lectura binaria).
str = pickle.dumps(object) se guarda objecten la strvariable como una cadena.
object = pickle.loads(str)reconstruye el objeto previamente escrito en la cadena str.
El pickling no es seguro . Puede contener objetos de código y datos que podrían atacar tu sistema. Asegúrate de que todo lo que desempaquetes provenga de una fuente confiable y no haya sido manipulado durante su transporte.
Clases
Clases y herencia
Una clase describe un nuevo tipo de objeto y agrupa los métodos necesarios para trabajar con objetos de ese tipo. He aquí una analogía: una clase es como una receta de pastel , mientras que los objetos son los pasteles que se pueden preparar siguiendo la receta.
Definiendo una clase 
La sintaxis para definir una clase es
class ClassName(superclass):
 variable and method definitions
Introducción a la herencia 
Cada clase (excepto la objectclase ) tiene una superclase y hereda variables y métodos de ella. Es decir, todas las variables y métodos definidos en (o heredados por) la superclase están disponibles para la nueva clase.
Por ejemplo, podría definir una clase Personcon las variables namey agey un método greet. Cada objeto de este tipo tendrá sus propias copias de esas variables y su propia referencia al greetmétodo. Si crea una clase Customercomo subclase de Person, cada objeto de tipo Customertendrá sus propias namevariables agey y un greetmétodo. Además, tendrá las variables y métodos adicionales que declare Customer(por ejemplo, una lista de compras).


Introducción a la clase de objeto 
El especial object classes la raíz de todas las clases. Si se omite superclassal definir una clase, esta tendrá por defecto la superclase object. Toda clase hereda de object, ya sea directa o indirectamente.
Terminología: Como sustantivo, «instancia» significa lo mismo que «objeto». Sin embargo, solemos usar «instancia» al referirnos a un objeto específico ( johnes una instancia de Person) o al usar la palabra como adjetivo ( namees una variable de instancia de la clase Person).
Terminología: “Campo” es otro nombre para “variable de instancia”.
He aquí un ejemplo de una definición de clase:
class Person(object):
  # defining variables of Person class
  name = 'Joe'
  age = 23

  # defining methods
  def say_hi(self):
    print("Hello, Joe.")
Hay numerosas cosas que podemos hacer con el ejemplo anterior:
Para crear una instancia (objeto) de una clase, podemos utilizar el nombre de la clase como si fuera un nombre de función, por ejemplo, p = Person().
Podemos acceder a los campos del objeto pusando la notación de puntos : p.namees “Joe” y p.agees 23 .
Podemos modificar los campos de p, por ejemplo, diciendo p.name = 'Jack'.
También podemos utilizar la notación de puntos para “hablar con” el objeto p o, más formalmente, “enviar un mensaje a” el objeto p: le p.say_hi()dice pa say_hi, y presponderá imprimiendo Hello, Joe..
Convención: Los nombres de clases siempre comienzan con una letra mayúscula y los nombres de variables y funciones siempre comienzan con una letra minúscula.
Se ha proporcionado el siguiente ejemplo basado en código para ayudarle a comprender mejor estos conceptos.
Desafortunadamente, cada instancia que creamos de esta clase será exactamente igual . Nos gustaría crear instancias de Person con diferentes valores para name y age. Para ello, debemos explorar una variable especial: self.
Constructores y self
Clases de Python vs otros lenguajes 
Si está familiarizado con las clases y los objetos en otro lenguaje, las clases y los objetos son muy similares en Python, a excepción de la molesta palabra self, que parece ser necesaria en todas partes.
Constructores en clases de Python 
Recuerda que cada objeto que creamos a partir de una clase tiene su propia copia de cada uno de los campos (variables de instancia) de esa clase. Si tenemos dos Personobjetos llamados joe y jane, cada uno tendrá un name campo y un age campo.
En pocas palabras, el nombre self utilizado dentro de una definición de clase es el objeto particular del que estamos hablando.
Cada método dentro de una clase debe tener self como primer parámetro.
Si recuerdas el Person ejemplo de la clase de la lección anterior, tenía la limitación de que todas las instancias creadas a partir de ella serían exactamente iguales. Podemos solucionar esto usando el self parámetro.
El siguiente es el mismo Person ejemplo de clase, pero también incluye el uso del self parámetro.
class Person(object):
  # defining variables of Person class
  def __init__(self, name, age):
    self.name = name
    self.age = age

  # defining methods
  def say_hi(self):
    print('Hello', self.name)

jill = Person('Jill', 30) # creating object of Person class
jill.say_hi() # calling say_hi method of Person class
Python tiene varios métodos especiales y __init__es uno de ellos. (Es init con dos guiones bajos antes y dos después). __init__es un constructor y se llama indirectamente cuando creas un nuevo objeto.
Para crear un objeto, no llame a ` __init__`. En su lugar, use el nombre de la clase como si fuera el nombre de una función y proporcione valores para todos los parámetros excepto self ``. Para crear variables de instancia, use el formato self.name = valueen el __init__método.
En el ejemplo anterior, creamos un objeto jillcon `` jill = Person('Jill', 30)`. Dentro de ``` __init__ constructor, se crea una namevariable de instancia asignando un valor a ` self.name`, y otra agevariable de instancia asignando un valor a ` self.age``.
Cuando nos encontramos fuera de la definición de clase, nos comunicamos con un objeto mediante la notación de punto . Esto se ve como jill.ageo jill.say_hi(). Dentro de la definición de clase, el objeto se comunica consigo mismo , por lo que decimos self.agey self.say_hi().
Introducción a self
Entonces, ¿qué es self? Es un primer parámetro explícito de cada método y un argumento implícito u oculto de cada mensaje que enviamos a un objeto.
En la definición de say_hi, selfdebe indicarse como primer parámetro. Sin embargo, cuando decimos jill.say_hi(), jill¿el objeto se pasa al método como valor de self?

Si un método hace referencia a una variable de instancia de la misma clase, self se debe usar la palabra. Si un método de una clase llama a otro método de la misma clase, self se debe usar la palabra. Cualquier variable (o método) que no esté etiquetado con el nombre self es local al método en el que se encuentra.
Nota técnica : selfes solo un nombre de variable, por lo que, en teoría, podría reemplazarse por cualquier otro nombre. Sin embargo, esto iría totalmente en contra de la convención.
Comparación con otros idiomas 
A diferencia de otros lenguajes orientados a objetos, en Python un objeto no tiene un conjunto fijo e invariable de variables de instancia. Se puede añadir una variable de instancia a un objeto simplemente mediante la asignación. Por ejemplo, se podría decir jill.occupation = 'scientist'. Esto afecta solo a jill, y los demás Personobjetos permanecen inalterados. De igual forma, se puede usar la deloperación para eliminar una variable de instancia de un objeto específico.
Nota importante de depuración: Al definir y usar clases y objetos, el error más común es proporcionar demasiados o insuficientes argumentos. Esto casi siempre se debe a que se olvida usar selfalgún objeto.
Comparación con Java 
El lenguaje de Java thises equivalente al de Python self, excepto por dos diferencias principales:
thises una palabra clave, pero selfes un nombre de variable
thisGeneralmente es implícito, pero selfdebe usarse explícitamente cada vez.
Subclases
Introducción a las subclases 
Una clase hereda todas las variables y métodos de instancia de su superclase. (Es una subclase de su superclase). Por ejemplo:
class Friend(Person):
  def smile(self):
    print('¯\_(^-^)_/¯')

meg = Friend('Margaret', 25)
En el ejemplo anterior, al crear meguna instancia de Friend, megse tendrán los métodos say_hi, get_oldery get_much_olderde la Personclase, junto con el smilemétodo y el __init__ constructor. Dado que el constructor se hereda, se pueden usar los mismos argumentos para crear un Friendque para crear un Person, como se puede ver en los ejemplos anteriores y posteriores:
meg = Friend('Margaret', 19)
Introducción a la anulación de 
En Python, también se puede anular (reemplazar) cualquier método heredado proporcionando otro método con el mismo nombre. Por ejemplo:


def say_hi(self, extra):
 print('Hi!', extra)
Observe esto en el siguiente ejemplo, donde llamar al say_hi método de la clase Persona sin parámetros da un error.
Introducción a la anulación de 
En Python, también se puede anular (reemplazar) cualquier método heredado proporcionando otro método con el mismo nombre. Por ejemplo:
def say_hi(self, extra):
 print('Hi!', extra)
Observe esto en el siguiente ejemplo, donde llamar al say_himétodo de la clase Persona sin parámetros da un error.
Si anula un método en una subclase, aún puede acceder a él usando super() como prefijo. Por ejemplo:
def say_hi(self, extra):
 super().say_hi()
 print(extra)
Puedes ver esto en el siguiente ejemplo. Al llamar al say_hi método de la clase Friend, primero se llama al método say_hi sobrescrito de la clase Person y luego al método say_hi de la clase Friend.
Normalmente, se escribe una subclase cuando se desea todo en su superclase, pero ¿qué ocurre si se desea que la subclase contenga información adicional ? Por ejemplo, un Friendobjeto es un Person, por lo que debería tener un namey un age, pero quizás también se desee que tenga un nickname. En este caso, se deberá escribir un nuevo __init__constructor, sobrescribiendo el heredado. Por ejemplo:
def __init__(self, name, age, nickname):
Nota : Si escribe una subclase porque solo desea algunas de las cosas de su superclase, esto indica que su clase se beneficiaría al ser reorganizada.
En el [nombre del constructor Friend constructor], se podría copiar y pegar todo el trabajo realizado en el [nombre Person constructordel constructor], pero es mucho mejor simplemente llamar a ese constructor. En este caso, se llama directamente __init__usando super()[nombre del constructor] como prefijo. Por ejemplo, el Friendconstructor podría verse así:
def __init__(self, name, age, nickname):
 super().__init__(name, age)
 self.nickname = nickname
Puedes comprobarlo en el ejemplo que se muestra a continuación:
class Person(object):
  def __init__(self, name, age):
    self.name = name
    self.age = age

  def say_hi(self):
    print('Hello', self.name)

class Friend(Person): # inheriting Person class
  def __init__(self, name, age, nickname): # overriding the inherited init constructor
    super().__init__(name, age) # calling the constructor of Person
    self.nickname = nickname # adding new portion
Método de impresión
El printmétodo acepta cualquier número de argumentos, convierte cada uno en una cadena llamando a su __str__método y los imprime. Al igual que con __init__, se usan dos guiones bajos antes y después del nombre.
Todas las clases tienen un __str__método heredado de object. Es recomendable sobrescribir este método. De lo contrario, los objetos mostrarán algo como esto:
<__main__.Friend object at 0x1064728d0>
En el caso de un Friendobjeto, su método podría verse así:
def __str__(self):
 return self.name + "'s age is " + str(self.age)
print(meg)daría como resultado Margaret's age is 25.
En el ejemplo anterior, la edad de Margret es un entero, y no podemos sumar un entero a una cadena. Por lo tanto, tuvimos que convertir explícitamente su edad a una cadena llamando a la función str(self.age). Una vez definido el __str__método en la clase, se pueden llamar los str functionobjetos on de esa clase, por str(meg)ejemplo.
Nota técnica : Sí, define el __str__método y luego úsalo llamando al str function.
Introducción a __repr__la función 
Similar a str(x), la repr(x) función devuelve una cadena que representa el objeto x, según lo define su __repr__(self) método. Siempre que sea posible, la cadena debe ser legible por Python para reconstruir el objeto.
De manera similar al __str__método, después de definir el __repr__método en su clase, puede llamar a la reprfunción en objetos de esa clase, por ejemplo, diciendo repr(meg).
Comparando objetos
Los operadores de igualdad ( ==) y desigualdad ( !=) funcionan bien con los tipos de objeto integrados de Python. Sin embargo, al definir clases propias, ==siempre retornarán Falseal comparar dos objetos.
Cuando hablamos de establecer comparaciones entre dos objetos, es importante tener presente la distinción entre igual e idéntico .
Dos objetos, ay b, son iguales si cada parte de aes igual a la parte correspondiente de b. Si cambias a, bpodría cambiar o no. Para comprobar la igualdad, es necesario comprobar todas las partes.
Sin embargo, si ay bson idénticos , entonces son simplemente dos nombres diferentes para el mismo objeto. Cambiar ese objeto al hacer algo en asignifica que btambién sufre el cambio. Comprobar la identidad simplemente implica comprobar si ay bambos se refieren a la misma ubicación en la memoria.
Al comparar objetos en Python, tenga en cuenta los siguientes puntos:
La asignación a = b nunca realiza una copia de un objeto b. Solo hace areferencia al mismo objeto que b.
Cuando se utiliza un objeto como argumento de una función, la función obtiene una referencia al objeto original , no una copia.
a is bcomprueba si ay bse refieren al mismo objeto. a is not bcomprueba si son objetos diferentes (pero posiblemente iguales).
Métodos de clase de igualdad incorporados 
Para su propia clase, puede definir fácilmente la igualdad y el orden definiendo algunos métodos especiales en la clase, que puede encontrar a continuación:
Nota : Todos estos nombres utilizan guiones bajos dobles.
__eq__(self, other)Debe devolverse Truesi se considera que " self objectsy" otheres igual , y Falseen caso contrario. Los objetos idénticos deben considerarse iguales (se puede comprobar la identidad con el isoperador). Este método se llamará al comparar los objetos con "<sub>" ==.
__ne__(self, other)Se llamará cuando se comparen objetos utilizando !=.
__lt__(self, other)Se llamará cuando se comparen objetos utilizando <.
__le__(self, other)Se llamará cuando se comparen objetos utilizando <=.
__ge__(self, other)Se llamará cuando se comparen objetos utilizando >=.
__gt__(self, other)Se llamará cuando se comparen objetos utilizando >.
Todos estos métodos son independientes. Definir algunos no implica definir automáticamente otros. Por ejemplo, si define un __eq__método, __ne__no se crea automáticamente. Si desea utilizar cualquiera de los operadores de comparación ( ==, ! =, etc.) para sus objetos, debe definir el método correspondiente. De lo contrario, obtendrá el comportamiento predeterminado de los operadores, que casi con seguridad será incorrecto.
Normalmente consideramos los operadores de comparación como comparaciones de  tamaños o magnitudes; puedes definir tus propios operadores de comparación para cualquier orden que desees. Esto ya se ha hecho para cadenas. Los operadores utilizan un orden lexicográfico (alfabético), donde todas las mayúsculas tienen un valor menor que todas las minúsculas.
Referencias
Valores simples y objetos complejos 
En sentido técnico, cada valor en Python es un objeto. En la práctica, se distingue entre valores simples y objetos más complejos. Los números y los booleanos pueden considerarse valores simples.
Supongamos que tenemos dos valores: ay b. Si el valor de la variable aes simple y se ejecuta la asignación b = a, el valor se copia de aa b. Posteriormente, se puede cambiar el valor de cualquiera de las dos ao bsin afectar el valor de la otra variable.
Sin embargo, consideremos el siguiente escenario:
a = [1, 2, 3]
b = a
b[1] = 99
print(a)  # prints [1, 99, 3]

En este ejemplo, aes una lista, y las listas son un tipo de objeto. Esta lista no se almacena en a . Se almacena en otra parte de la memoria, y lo que se almacena en aes un puntero (también conocido como enlace , referencia o dirección ) al objeto.
Puedes pensarlo así: una variable solo puede contener un valor simple, es decir, un valor que se pueda almacenar en ocho bytes o menos. Ejemplos de valores simples incluyen números, booleanos y punteros. Las listas, conjuntos, diccionarios, cadenas y otros objetos no son simples y requieren más de ocho bytes.
Esto significa que si acontiene un valor simple , como 5 , b = acopia el valor 5 en b. Sin embargo, si el valor de aes un objeto complejo , como una lista, acontiene un puntero a ese objeto y lob = a copia en . El resultado es que tanto como contienen un puntero a la misma lista .bba
Si se asigna otra lista a a, no hay problema. aObtiene un nuevo puntero, pero bno se modifica. Sin embargo, si algún elemento de la lista se modifica, por ejemplo, mediante b[3] = 17, dicho cambio es visible para todas las variables que contienen un puntero a la lista.
Considere la siguiente función:
def  f ( x ,  y ) :
 x [ 1 ]  =  99
 y  =  ordenado ( y )
Ahora, ejecute las siguientes declaraciones para ver qué sucede:
def f(x, y):
    x[1] = 99
    y = sorted(y)
    
a = [3, 1, 4, 1, 6]
b = [2, 7, 1, 8, 2]
f(a, b)

print(a, b) # prints [3, 99, 4, 1, 6] [2, 7, 1, 8, 2]
Esto es lo que pasó en la función f:
               
            

Este es uno de los aspectos más confusos de la programación orientada a objetos y da lugar a numerosos errores difíciles de detectar. El problema se agrava porque algunos métodos de Python modifican el objeto original, mientras que otros crean uno nuevo. Por ejemplo, sortedcrea una nueva lista, mientras que reversemodifica la lista original.
Nota : Al llamar a una función, los valores de los argumentos se copian en los parámetros de la función. Sin embargo, al retornar, la función no copia los valores de los parámetros de vuelta en los argumentos. En ocasiones, puede parecer que esto último ocurre porque se modifica el argumento (como la variable adel ejemplo anterior).
Nota : Los enteros en Python pueden ser demasiado largos para caber en ocho bytes. Esto debería hacerlos susceptibles a los problemas de puntero mencionados anteriormente, pero Python utiliza algunos trucos del compilador para que se comporten como valores simples. Esto significa que no hay que preocuparse de que los enteros se comporten de forma diferente al añadir uno o dos dígitos adicionales.
Nota : Las cadenas son inmutables. Nunca se puede cambiar el valor de una cadena, y solo se pueden crear nuevas. Esto es útil porque las hace actuar como valores simples.
Copias superficiales y profundas
Introducción a la copia superficial 
Los objetos pueden contener referencias a otros objetos. Si copiamos el objeto, normalmente obtenemos una copia superficial. Esto significa que se copian las referencias dentro del objeto, pero no los objetos referenciados.
Puedes ver esto en el siguiente ejemplo:
grades2 = grades[:] # setting up grades2 list
Introducción a la copia profunda 
Python también proporciona un método para realizar una copia profunda deepcopy completamente independiente de un método, hasta un cierto nivel.
grades3 = copy.deepcopy(grades)
En general, se trata el tema de la referencia, lo mejor es hacer una deepcopy, salvo que vayas a modificar el objeto de diferentes referencias.
Poniéndose Sofisticado
Declaraciones e identificadores
Descripción de algunas sentencias de Python 
Esta sección describe las sentencias de Python que no hemos cubierto hasta ahora o que tienen más opciones que las que ya se han tratado. Analicémoslas:
1. importDeclaración :
import moduleImportará todos los nombres del módulo especificado module, pero cada uso de un nombre debe ir precedido de moduleun punto ( .). Esto evita problemas al importar el mismo nombre desde más de un módulo.
from module import namesImportará los nombres dados, que no necesitan ir precedidos de moduleun punto ( .). Si namesse reemplaza por un asterisco ( *), significa que se importan todos los nombres.
import module as nameimportará todos los nombres de module, pero cada uso de un nombre debe tener como prefijo namey un punto ( .).
from module import name1 as name2importa name1pero le cambia el nombre a name2.
2.assert expression1, expression2
No hace nada si expression1es verdadero, pero lanza un AssertExceptionsi expression1es falso. El opcional expression2se usa como mensaje por el AssertException.
La assertdeclaración se considera mejor como "documentación ejecutable". Puede usarse al principio de una función para expresar lo que se requiere de los parámetros, o en la mitad o al final para afirmar algo sobre lo que se acaba de calcular.
3.var1, ..., varN = expr1, ..., exprN
La línea anterior se refiere a la llamada asignación simultánea. A las variables de la izquierda se les asignan los valores de las expresiones de la derecha. Es importante tener en cuenta que debe haber exactamente tantas variables como expresiones.
4.break
Si break se ejecuta dentro de un bucle while or for, hace que el bucle salga.
Si break se ejecuta dentro de bucles anidados, solo break se saldrá del bucle más interno que contiene el.
Si el bucle sale debido a un else, no se ejecutará break la siguiente cláusula. (Posiblemente exclusivo de Python, ya que los bucles pueden tener una cláusula else).
5.continue
Si continue se ejecuta dentro de un bucle while or for, cualquier instrucción restante dentro del bucle se omite y el control regresa al inicio del bucle (a la prueba en un whilebucle o al siguiente valor en un forbucle).
6. del variable
La variable del hace que la variable deje de existir (se vuelva indefinida).
7.else
Tanto un whilebucle como un forbucle pueden ir seguidos de una elsecláusula. El código de la elsecláusula se ejecuta cuando el bucle finaliza normalmente, pero no se ejecuta si el bucle termina como resultado de un [ breakerror]. Esta es una construcción inusual sin un propósito claro.
8.exec(arg)
exec (arg)Ejecuta una cadena, un archivo abierto o un objeto de código arg. Los objetos de código no se tratan en este curso.
execEs en realidad una función, pero retorna Nonesu valor. Por lo tanto, se utiliza como si fuera una sentencia.
No lo use execsi existe alguna posibilidad de que argcontenga código malicioso o dañino.
9.nonlocal variables
Las funciones pueden estar anidadas dentro de otras funciones. Por defecto, una función interna puede acceder, pero no modificar, las variables de la función que la contiene. Si la función interna declara las variables como nonlocal, puede acceder a ellas y modificarlas.
Las reglas son muy similares a las de los locales y globallas variables.
10.pass
La pass sentencia no hace nada. En ocasiones, resulta útil cuando la sintaxis requiere la presencia de una sentencia, pero no hay ninguna acción específica que realizar.
Una elipsis (tres puntos, ...) es lo mismo que pass.
11.print(expr1, ..., exprN, sep=sepString,end=endString, file=outputFile)
La printdeclaración evalúa e imprime las expresiones con sepStringentre ellas y endStringdespués de la última expresión en outputFile.
Los argumentos de palabras clave sep, end, y filepueden omitirse con los valores predeterminados ' '(un solo espacio), \n(una nueva línea) y stdout, respectivamente.
12.raise
raise ExceptionGenera la excepción nombrada Exception, que podría gestionarse posteriormente mediante una try-exceptinstrucción. Python ofrece una gran cantidad de tipos de excepción, o puedes definir los tuyos propios subclasificando Exception.
raise Exception(expression)genera la expresión nombrada Exceptiony utiliza el resultado de la expresión como un mensaje en la excepción.
Por sí solo, raisevuelve a generar la misma excepción en una cláusula except.

Introducción a los identificadores 
Los identificadores comienzan con una letra o un guion bajo y pueden contener letras, guiones bajos y dígitos. Se distinguen mayúsculas y minúsculas, y se pueden usar caracteres Unicode.
Por convención, los nombres de clases comienzan con una letra mayúscula, mientras que otros nombres no.
En una clase, a menudo se necesitan variables de instancia privadas, es decir, campos a los que solo se puede acceder o modificar desde la clase. Esto no es posible en Python. En su lugar, algunos programadores han adoptado la convención de colocar un guion bajo como primer carácter al nombrar la variable que desean que sea privada. A veces, se usan dos guiones bajos si se desea que sea aún más privada. Los nombres especiales definidos por el lenguaje suelen empezar y terminar con dos guiones bajos.
Números y cadenas
ntroducción a los números 
Existen varios tipos de números en Python. A continuación se describen todos ellos:
Un entero decimal consiste en el número 0 o una secuencia de dígitos que no comienza con 0 .
Un entero binario consta de dígitos binarios (0, 1), que comienzan con 0bo 0B.
Un entero octal consiste en una secuencia de dígitos octales (0 a 7), que comienzan con 0oo 0O.
Un entero hexadecimal consiste en una secuencia de dígitos hexadecimales (0 a 9 y a a f o A a F), que comienzan con 0xo 0X.
Un número de punto flotante («real») incluye un punto decimal, un sufijo de exponente o ambos. El exponente consta de un signo opcional, la letra eo E, y uno o más dígitos.
Un número imaginario consiste en un entero decimal o un número de punto flotante, y su sufijo es j( no i ) o J.
Un número complejo consiste en la suma o diferencia de un número entero o de punto flotante y un número imaginario.
Introducción a las cadenas 
Al igual que los números, una cadena se puede representar de varias maneras en Python. A continuación se describen todas ellas:
Una cadena es una secuencia de cero o más caracteres, entre comillas simples ( '...'), comillas dobles ( "..."), comillas simples triples ( '''...''') o comillas dobles triples ( """..."""). Puede considerarse una secuencia de caracteres.
Las cadenas entre comillas triples pueden extenderse a lo largo de varias líneas e incluir los saltos de línea como parte de la cadena, a menos que el salto de línea esté inmediatamente precedido por una barra invertida ( \).
Una cadena sin formato es una cadena con el prefijo ro R. En una cadena sin formato, la barra invertida no escapa caracteres; todos los caracteres se representan a sí mismos. Esto es especialmente útil al escribir expresiones regulares.
Las expresiones regulares en Python siguen los estándares POSIX y suelen escribirse como cadenas sin formato. No se abordarán aquí.
Se eval(string)evalúa stringcomo una expresión de Python y devuelve el resultado. Si lo usa, asegúrese de que stringno contenga código malicioso.
Formato de cadenas 
Hay tres maneras de formatear una cadena: f-strings , el formatmétodo y el formato tradicional %. Generalmente, se recomienda el método f-string.
Usando f-strings 
Una cadena formateada, o f-string , lleva el prefijo fo F. En una f-string, cualquier expresión entre llaves, {}, se reemplaza por su valor. A continuación, un ejemplo:
print(f'Area is {5*7/2}.') # Prints: Area is 17.5.
En Python 3.8 y versiones posteriores, si una expresión entre llaves va seguida de un signo igual, se imprime tanto la expresión como el resultado. Puedes ver este ejemplo a continuación:
print(f'Area is {5*7/2=}.') # Prints: Area is 5*7/2=17.5.
Si realmente desea imprimir llaves dentro de una cadena f, no funciona escaparlas con una barra invertida . En su lugar, duplíquelas ('{{' or '}}').
print(f'Area is {{5*7/2={5*7/2}}}.') # Prints: Area is {5*7/2=17.5}.
La sintaxis de los códigos de formato es bastante compleja, pero veamos algunos ejemplos simples.
from math import pi

def p():
    print(f"{{'abc':6}}  prints \'{'abc':6}\'")
    print(f"{{'abc':>6}} prints \'{'abc':>6}\'")
    print(f"{{123:6d}}   prints '{123:6d}'")
    print(f"{{123:<6d}}  prints '{123:<6d}'")
    print(f"{{pi:6.2f}}  prints '{pi:6.2f}'")
    print(f"{{pi:<6.2f}} prints '{pi:<6.2f}'")
p()
Usando formatel método 
La sintaxis del formatmétodo es string.format(values). Al igual que una cadena f, la cadena contiene llaves, pero estas solo contienen códigos de formato. Por ejemplo, el código siguiente devolverá la cadena pi is 3.1416, con tres espacios entre "la palabra es" y "3" .
from math import pi
print('pi is {:8.4f}'.format(pi))

Usando %el signo de porcentaje ( ) 
El formato antiguo %utiliza el %signo de porcentaje tanto dentro de la cadena para indicar las posiciones de sustitución como después de ella como operador. Por ejemplo, el código siguiente genera la cadena x = 7.
print('%s = %i' % ('x', 7))
Sin embargo, este estilo de formato ya no se recomienda y no se analizará más aquí.
Diferentes conversiones entre tipos de datos 
Para convertir una cadena en un número , utilice las funciones int(s), float(s)y complex(s). Si la cadena tiene un formato incorrecto, se obtendrá un resultado ValueError. En particular, una cadena que representa un número complejo no puede contener espacios. Por ejemplo, 3+5jestá bien , pero 3 + 5jno .
Para convertir un número en una cadena , utilice str(x)para una representación decimal o bin(i), oct(i), o hex(i)para una representación de cadena del entero icomo binario , octal o hexadecimal , respectivamente.
Para convertir un número a o desde un carácter Unicode , utilice las funciones chr(n)o ord(c)respectivamente.

Operaciones de bits
Descripción general de los operadores de bits 
Los siguientes operadores de bits se pueden aplicar a números enteros.

En caso de que no esté familiarizado con estos operadores, cada número se utiliza en su notación binaria, 0compuesta únicamente por bits cero y -1compuesta únicamente por bits uno.
Complemento bit a bit ( ~) 
El complemento bit a bit cambia cada 1 a 0 y cada 0 a 1. Numéricamente, el resultado es el mismo que cambiar el signo del número y luego restar 1.
Por ejemplo, en el siguiente ejemplo, cuando ~se aplica el operador al entero 50 , obtenemos -51 . Esto se debe a que el complemento bit a bit primero cambia el signo de 50 , haciéndolo negativo, y luego le resta 1 , sumando los dos enteros negativos para obtener -51 .
num1 = 50
result = ~num1 # Applying bitwise complement operator on the integer
print(result) # Displays -51

Bit a bit y ( &) 
Bit a bit y da un 1 cuando los bits correspondientes de ambos operandos son 1. Da un 0 en caso contrario.
He aquí un ejemplo de ello:
num1 = 50 # (00110010)
num2 = 10 # (00001010)
result = num1 & num2 # Applying and operator on the two integers
print(result) # Displays 2 # (00000010)
Puede obtener una comprensión más detallada de la funcionalidad de este operador en la siguiente figura:

Bit a bit o ( |) 
Bitwise or da un 1 cuando los bits correspondientes de uno o ambos operandos son 1. Da un 0 en caso contrario.
He aquí un ejemplo de ello:
num1 = 50 # (00110010)
num2 = 10 # (00001010)
result = num1 | num2 # Applying or operator on the two integers
print(result) # Displays 58 # (00111010)
Puede obtener una comprensión más detallada de la funcionalidad de este operador en la siguiente figura:

Exclusivo bit a bit o ( ^) 
Exclusivo bit a bit o devuelve un 1 cuando los bits correspondientes de los dos operandos son diferentes. Devuelve un 0 en caso contrario.
He aquí un ejemplo de ello:
num1 = 50 # (00110010)
num2 = 10 # (00001010)
result = num1 | num2 # Applying exclusive or operator on the two integers
print(result) # Displays 56 # (00111000)
Puede obtener una comprensión más detallada de la funcionalidad de este operador en la siguiente figura:

Shift izquierdo ( <<) 
El desplazamiento a la izquierda desplaza todos los bits a la izquierda y los completa con ceros al final. Aquí, el operando derecho representa la cantidad de desplazamiento del operando izquierdo.
He aquí un ejemplo de ello:
num1 = 50 # (00110010)
num2 = 2
result = num1 << num2 # Applying left shift operator on the num1 integer
print(result) # Displays 200 # (11001000)
Puede obtener una comprensión más detallada de la funcionalidad de este operador en la siguiente figura:

Desplazamiento a la derecha ( >>) 
El desplazamiento a la derecha mueve todos los bits a la derecha; algunos desaparecen del extremo derecho y aparecen ceros en el izquierdo. Aquí, el operando derecho representa la cantidad de desplazamiento del operando izquierdo.
He aquí un ejemplo de ello:
num1 = 50 # (00110010)
num2 = 2
result = num1 >> num2 # Applying left shift operator on the num1 integer
print(result) # Displays 12 # (00001100)
Puede obtener una comprensión más detallada de la funcionalidad de este operador en la siguiente figura:

Listas por comprensión
Una comprensión de listas es una forma de calcular una lista a partir de una secuencia o un iterador. La secuencia puede ser una lista , una tupla , un conjunto , las claves de un diccionario o una cadena .
Estructura general 
Las declaraciones de comprensión de listas siempre se escriben entre corchetes. []La estructura general que siguen estas declaraciones se divide en dos categorías:
[expresión para variable en secuencia] devuelve una lista de los valores de la expresión cuando cada variable en la secuencia se utiliza en la expresión.
Una lista de comprensión puede incluir una prueba opcional if.
[expresión para variable en secuencia si condición] devuelve una lista de los valores de expresión para cada variable en la secuencia que satisface la condición utilizada en la expresión.
La comprensión de listas utiliza 
Las listas por comprensión son eficaces y vale la pena conocerlas. Aquí hay algunos casos de uso sencillos:
Las listas por comprensión permiten aplicar una expresión a cada elemento de una lista. Esto se puede ver en el siguiente ejemplo:
print([2 * x for x in {1, 2, 3}]) # Prints [2, 4, 6]
Las listas por comprensión también pueden utilizarse para eliminar elementos no deseados de una lista. A continuación, se ofrece un ejemplo:
print([x for x in [1, 0, 2] if x > 0]) # Prints [1, 2]
Iteradores
Un objeto iterable es cualquier objeto que se pueda recorrer paso a paso. Entre los objetos iterables se incluyen listas, conjuntos, tuplas, cadenas, diccionarios, rangos y archivos. Un iterador es un objeto que implementa un método para recorrer paso a paso los elementos de un iterable.
Por ejemplo, puedes obtener un iterador de lista con `` it = iter([2, 3, 5]). Puedes recorrer los elementos de esta lista llamando repetidamente `` next(it). Esto devolverá ``2` , `` 3` y `` 5` . Otra llamada a `` next(it)generará una StopIterationexcepción.

Creación de objetos iterables personalizados e iteradores 
En Python, puedes implementar tus propios objetos iterables e iteradores. Un objeto iterable tiene al menos los métodos __init__y __iter__, mientras que un iterador tiene al menos los métodos __init__y __next__.
Para hacer que sus propios objetos sean iterables:
La clase iterable__init__ debe contener el método habitual y un __iter__método. El __iter__método crea y devuelve un nuevo objeto iterador.
La clase iterador__init__ contiene un método que toma el iterable como parámetro, lo guarda en una variable de instancia y realiza la configuración necesaria. Este __next__método se utiliza para encontrar o calcular el siguiente valor. En este caso, el __next__método debería generar una StopIterationexcepción si se invoca cuando no hay más valores que devolver.
A menudo, estas dos clases se combinan, de modo que el objeto iterable es su propio iterador. En este caso, el __iter__método simplemente devuelve self.
Se pueden usar iteradores en forbucles. En forlos bucles, la StopIterationexcepción no genera un error, sino que simplemente termina el forbucle. Con el ejemplo anterior, puede observar lo siguiente:
class Numbers():

    def __init__(self):
        self.n = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.n += 1
        if self.n > 10:
            raise StopIteration()
Es importante tener en cuenta que si tiene un iterador itque devuelve una cantidad finita de valores y necesita una lista de esos valores, simplemente diga list(it).
Generadores
Los generadores son funciones que devuelven un iterador. Un tipo de generador se parece a una lista por comprensión, excepto que los corchetes se sustituyen por paréntesis. Los generadores pueden usarse en cualquier lugar donde se utilice un iterador, for por ejemplo, en un bucle.
word = 'generator'
g = (c for c in word if c in 'aeiou') # similar to list comprehension

for i in g: 
 print(i, end=' ') # prints e e a o
Así es como funciona el código anterior:
En la línea 2 , dentro del paréntesis, el forbucle itera sobre la palabra generator. Luego, mediante una ifinstrucción, extrae todas las vocales de la palabra.
Dado que aquí utilizamos paréntesis, gcontiene un generador en lugar de un valor o una lista.
En la línea 4 , el forbucle itera sobre el generador y da acceso a cada uno de sus valores.
Esto nos proporciona el resultado e e a o, que son las vocales presentes en la palabra generator.
Precaución : Una vez que un generador devuelve todos sus valores, se agota. Si forse llamara al bucle anterior una segunda vez, no imprimiría nada, ya que la variable gcontendría un generador vacío.
Generadores de Python con rendimiento 
Puedes escribir funciones que actúen como generadores usando yielden lugar de return. Aquí tienes un ejemplo de generador para potencias de 2 :
def evens():
    n = 2
    for i in range(0, 5):
        yield n
        n *= 2
evensEn el ejemplo anterior, al llamar a la función , esta devuelve un generador , no un número. Puedes usar este generador en un forbucle de la misma manera que gantes (en el ejemplo de impresión de vocales) y obtendrás los valores 2, 4, 8, 16 y 32 .
Puedes verlo a continuación:
def evens():
    n = 2
    for i in range(0, 5):
        yield n
        n *= 2

e = evens()
for n in e:
    print(n)
O, al igual que con los iteradores , puedes usar next(e)para obtener el siguiente valor del generador. En este caso, tendrás que gestionar la excepción tú mismo.
Puedes ver esto en el ejemplo que se da a continuación:
def evens():
    n = 2
    for i in range(0, 5):
        yield n
        n *= 2

e = evens()
while True:
    try:
        print(next(e))
    except StopIteration:
        break
Así es como funciona el código anterior:
En la línea 7, la llamada e = evens() devuelve un generador y lo coloca en e.
La primera llamada next(e)evalúa la función hasta la yield instrucción (línea 4) y devuelve el valor obtenido, como una return instrucción normal. Sin embargo, la función también recuerda dónde se quedó.
Una llamada posterior a next(e)la línea 10 reanudará la ejecución inmediatamente después de la yield instrucción. Se conservarán todos los valores de las variables locales. En lo que respecta a la función, es como si yield nunca hubiera ocurrido. En este ejemplo, el forbucle continúa ejecutándose.
Puedes tener varias yield sentencias en un generador. La función recordará dónde yield se quedó y continuará desde allí.
Al igual que con los iteradores, un generador generará una StopIterationexcepción en la línea 11 cuando no haya más valores que devolver. Esto ocurre automáticamente al llegar al final de la función o al llegar a una return instrucción.

Parámetros y argumentos
Parámetros vs argumentos 
Sin sintaxis adicional, los argumentos (expresiones en una llamada a función) se corresponden con los parámetros (variables en una definición de función) por posición. El primer parámetro obtiene el valor de la primera expresión, y así sucesivamente.

Introducción a los parámetros 
Los parámetros en una definición de función pueden ser:
Una variable simple que se corresponde con un argumento por posición. Los parámetros posicionales deben preceder a cualquier otro parámetro.
variable=value:para dar un valor predeterminado para los argumentos faltantes.
*args- Para aceptar múltiples argumentos como una sola tupla. Por convención, argsse suele usar el nombre para este propósito.
**kwargs: para aceptar múltiples argumentos de palabras clave como un diccionario. Por convención, kwargsse suele usar el nombre (argumentos de palabras clave) para este propósito.
*argso **kwargssólo se pueden utilizar como últimos parámetros porque recogen todos los argumentos restantes.
Introducción a los argumentos 
Los argumentos en una llamada de función pueden ser:
Una expresión
name=value: Para asignar un valor al parámetro con ese nombre. Esto se denomina argumento con nombre o argumento de palabra clave . Si se utilizan argumentos posicionales y de palabra clave, todos los argumentos posicionales deben preceder a los argumentos de palabra clave.
*tuple:para pasar una tupla como argumentos separados.
**dictionary:para pasar valores a múltiples parámetros por nombre.
Argumentos posicionales vs. argumentos de palabras clave 
A partir de Python 3.8, se puede forzar que los argumentos sean solo posicionales. Una barra diagonal, /en lugar de un parámetro, significa que todos los argumentos anteriores deben ser posicionales. Un asterisco, *en lugar de un parámetro, significa que todos los argumentos siguientes deben ser palabras clave. Por ejemplo, dada la siguiente función:
def foo(a, b, /, c, d, *, e, f):
    print(a, b, c, d, e, f)
Los argumentos para a y b deben ser solo por posición, e y f deben ser palabras clave. Los argumentos para c y d pueden ser cualquiera de los dos (pero como los argumentos posicionales deben preceder a los argumentos de palabras clave, d no pueden ser posicionales si ces una palabra clave).
Muchas de las funciones descritas en la documentación oficial de Python parecen tener un solo asterisco como parámetro. Por ejemplo:
os.remove(path, *, dir_fd=None)
Esta es una convención de documentación que indica que los argumentos subsiguientes solo pueden proporcionarse como argumentos con nombre. Esto no significa que deba colocarse un asterisco en la llamada a la función.
Programación funcional
Definición 
Las definiciones de programación funcional difieren, pero en general, programación funcional significa
Las funciones son objetos y pueden tratarse como cualquier otro valor.
Las variables son de asignación única . Una vez que se les asigna un valor, no se modifican.
Todas las funciones son puras . El valor que devuelve una función depende únicamente de los argumentos que se le dan, y si se vuelve a llamar con los mismos argumentos, producirá el mismo valor. Esto excluye el uso de variables globales u otros factores externos, como el reloj del sistema.
La programación funcional se considera generalmente una técnica de torre de marfil, inadecuada para la programación diaria. Este no es el lugar para cuestionar este punto de vista, salvo para señalar que todos los lenguajes de programación modernos admiten la programación funcional en mayor o menor medida.
Las funciones son objetos desde el punto de vista 
En esta lección, consideraremos únicamente el primer punto: que las funciones son objetos . Consideraremos solo un ejemplo de dicho uso.
En el siguiente ejemplo, la función más grande encontrará el valor más grande en una lista de valores:
def biggest(values):
    big = values[0]
    for v in values:
        if v > big:
            big = v
    return big
En el ejemplo anterior, since >también se puede usar para comparar cadenas. Esta función permite encontrar la cadena lexicográficamente más grande en una lista de cadenas, pero para casi cualquier otro propósito (encontrar el número más pequeño, la cadena más larga, etc.), se requiere otra función prácticamente idéntica.
Por lo tanto, en lugar de escribir más y más funciones, podemos reemplazarlas >con una prueba genérica:
def most(values, more):
    best = values[0]
    for v in values:
        if more(v, best):
            best = v
    return best

def larger(a, b):
    return a > b
Y podemos llamar a la most función así:
most([1,·6,·1,·8,·0], larger)
O así:
def·longer(a,·b):
    return·len(a)·>·len(b)

most(["a", "generic", "list"], longer)
Funciones lambda 
Esto aún genera muchas funciones pequeñas como larger. Python también proporciona funciones literales, que a veces se denominan funciones anónimas porque no tienen nombre.
Las funciones literales están diseñadas para usarse únicamente en el lugar donde están escritas. Por razones históricas, una función literal se introduce con la palabra clave lambda. Por ejemplo,
lambda·a,·b:·len(a)·>·len(b)
Lambda can be used like in the example below:
def most(values, more):
    best = values[0]
    for v in values:
        if more(v, best):
            best = v
    return best

print(most(["a", "generic", "list"], lambda a, b: len(a) > len(b)))
Una expresión lambda consta de la palabra clave lambda, cualquier número de parámetros, dos puntos y una sola expresión. En el ejemplo anterior, el resultado de la expresión es un valor booleano, pero podría ser de cualquier tipo.
Funciones integradas de la programación funcional 
Python tiene varias funciones integradas que toman una función como parámetro. Estas son algunas de las más útiles:
map(function, sequence)devuelve un iterador cuya next función se aplicará function al siguiente elemento de sequence(que puede ser una lista, un conjunto, una cadena, una tupla o un diccionario) y devolverá el resultado.
filter(predicate, sequence)devuelve un iterador cuya next función devolverá el siguiente elemento de sequence que satisface el predicate.
functools.reduce(binaryFunction, sequence)Aplica binaryFunctiona los dos primeros elementos de sequence. Luego, se aplica repetidamente binaryFunctional resultado actual y al siguiente miembro de sequence, devolviendo un único valor como resultado. (Nota: Esta función debe importarse desde functools).

Testing
Filosofía
Ventajas del marco de pruebas 
El uso de un marco de pruebas adecuado tiene un gran número de ventajas, entre ellas:
Es mucho más probable que el código sea correcto.
Generalmente, el código tarda menos tiempo en escribirse porque se reduce el tiempo de depuración.
Las funciones escritas para ser probadas tienden a ser mucho más pequeñas y tener un solo propósito.
Las funciones escritas para ser probadas proporcionan una separación clara entre el cálculo y la E/S.
La existencia de un conjunto de pruebas hace que sea más fácil y seguro modificar el código en algún momento posterior.
Un buen marco de pruebas permite al programador ejecutar las pruebas rápidamente, generalmente con un solo clic, y detectar rápidamente si hay errores. Cualquier prueba que requiera más tiempo se ejecutará con frecuencia, o incluso nunca, lo que anula su utilidad.
Cómo escribir código comprobable 
Para escribir código comprobable, es necesario realizar las siguientes cosas:
Escriba muchas funciones pequeñas que hagan cada una una cosa, en lugar de unas pocas funciones grandes que hagan muchas cosas.
Escribe las pruebas mientras escribes el código, no después. Muchos expertos incluso recomiendan escribir las pruebas antes de escribir el código, ya que esto ayuda a aclarar qué debe hacer el código.
Puede resultar difícil o imposible escribir pruebas para código preexistente o heredado.
Minimiza el uso de variables globales y evítalas por completo si es posible. Las funciones cuyos valores dependen únicamente de sus argumentos son mucho más fáciles de probar de forma aislada.
Separe estrictamente las funciones que realizan cálculos de los cálculos que realizan entrada o salida, y pruebe solo las primeras.
Las funciones de E/S involucran al programador, por lo que impiden el uso de pruebas de un solo clic. Si bien existen técnicas para probar estas funciones, son técnicas avanzadas que no se abordan aquí.

Pruebas unitarias
Introducción y estructura de las pruebas unitarias 
Una prueba unitaria evalúa los métodos de una sola clase. Un caso de prueba evalúa la respuesta de un solo método a un conjunto específico de entradas.
Para realizar pruebas unitarias, debes hacer lo siguiente:
import unittest
import fileToBeTested o from fileToBeTested import *
Recordatorio : si usas , from file import *no es necesario que delante de cada llamada de función aparezca el nombre del archivo desde el que se importó.
Escribe un class SomeName(unittest.TestCase). Dentro de la clase:
Defina los métodos setUp(self)y tearDown(self), si lo desea. Ambos son opcionales.
Proporcione uno o más testSomething(self)métodos. Puede incluir otros métodos, pero los nombres de los métodos de prueba deben comenzar con test.
Al final del archivo de prueba, coloque unittest.main().
Papel que desempeña cada componente de las pruebas unitarias 
Esto es lo que unittest.main() hace. Para cada método cuyo nombre comience con test, el unittest.main método llama setUp() a si se ha proporcionado uno. Luego, llama al método de prueba y al tearDown() método si se ha proporcionado uno. Por lo tanto, cada prueba se intercala entre setUpy tearDown.
El propósito del setUpmétodo es asegurar que todo esté en un estado conocido y original antes de ejecutar el método de prueba. De esta manera, se puede garantizar que los resultados de una prueba no afecten los de una prueba posterior.
El propósito de tearDown es eliminar artefactos (como archivos) que puedan haberse creado. Se usa con mucha menos frecuencia que setUp.
Cada método de prueba suele probar solo una función, aunque puede llamarla varias veces. Se pueden escribir varias pruebas para la misma función.
He aquí un ejemplo trivial de un método de prueba:
def test_add(self):
    self.assertEqual(4, add(2,·2))
    self.assertEqual(0, add(2,·-2))
Afirmaciones 
Si alguna afirmación de un método de prueba falla, la prueba falla y las demás afirmaciones de ese método no se prueban. Por lo tanto, los métodos de prueba no deben ser demasiado largos.
Si el método a probar está en una clase C y se usó import Cen lugar de from C import *, también se debe usar el nombre de la clase, por ejemplo, self.assertEqual(4, C.add(2, 2)). La convención es colocar el resultado esperado (4) primero y la llamada a la función ( add) al final.
Estos son los métodos de afirmación más utilizados:
self.assertEqual(expected, actual)
self.assertAlmostEqual(expected, actual)para números de punto flotante
self.assertTrue(boolean)y assertFalse(boolean).
En un apéndice se ofrecen más métodos de afirmación.
La assertRaises función 
Puedes comprobar si una función genera una excepción cuando debería, pero este método tiene una forma especial. Esto es necesario porque los argumentos de una función se evalúan antes de llamarla. Por ejemplo, si dices lo siguiente:
self.assertRaises(ZeroDivisionError,·5/0)
El argumento 5/0 se evaluaría y generaría la excepción antes de assertRaises poder llamarlo.
La solución es pasar el nombre de la función separado de sus argumentos:
self.assertRaises(exception, function_name, arguments)
Esto permite que la assertRaisesfunción llame a la función que se va a probar dentro de un try-exceptbloque y la gestione adecuadamente.
Prácticas comunes 
Cuando no se realizan pruebas, una práctica común es colocar una llamada a la función principal como última línea del archivo de código, por ejemplo, [Nombre del archivo] main(). Esto hace que el mainmétodo se ejecute inmediatamente después de cargar el archivo. Al realizar pruebas unitarias, esto no es recomendable. En su lugar, reemplace esa línea con lo siguiente:
if __name__ == '__main__':
    main()
Y ponga el siguiente código al final del archivo de prueba:
unittest.main()
De esta manera, el programa se ejecutará si se carga desde el archivo de programa y las pruebas se ejecutarán si se cargan desde el archivo de prueba.
Ejemplo de prueba unitaria 
Si se proporciona un into un string, la siguiente sección de código devuelve una lista de dígitos del mismo:
def get_digits(number):
    """Given an int or string, return a list of digits in it."""
    string = str(number)
    return [x for x in string if x.isdigit()]

def main():
    s = input("Enter something: ")
    digits = get_digits(s)
    print("Digits found:", digits)
    if digits != []:
        main()

# Call main() if and only if started from this file
if __name__ == '__main__':
    main()
Así es como funciona el código anterior en detalle:
En la get_digits función, en primer lugar, number argument se convierte en una cadena en la línea 3.
En la línea 4, utilizamos una comprensión de lista dentro de la cual un for bucle itera sobre toda la cadena, extrae cada carácter de ella y utiliza el isdigit método de cadena para verificar si el carácter es un dígito o no.
De esta manera, la get_digits función devuelve una lista de dígitos si hay alguno presente en la cadena. Si no hay dígitos, devuelve una lista vacía.
En la main función, se espera que el usuario ingrese su entrada en la línea 7, que luego se pasa como argumento a la get_digits función en la línea 8. La salida de la get_digits función se almacena dentro de la digits variable.
A continuación, comprueba si la digits variable es una lista vacía o no en la línea 10.
Si no es así, vuelve a llamar a la main función en la línea 11, lo que reinicia todo el proceso.
Este archivo de prueba unitaria incluirá la misma estructura que mencionamos anteriormente. A continuación:
import unittest
from get_digits import *

class test_get_digits(unittest.TestCase):

    def test_get_digits(self):
        s = get_digits("<0.12-34 56abc789x")
        self.assertEqual(list("0123456789"),
                         get_digits(s))
        self.assertEqual(list("1230"),
                         get_digits(1230))

unittest.main()
Así es como funciona el código anterior:
Primero, unittest se importa la biblioteca en la línea 1. Luego, todos los datos del get_digits.py archivo se importan en la línea 2.
A continuación, creamos una clase llamada test_get_digits y la convertimos en una subclase de unittest.TestCase, que contiene todas las funciones necesarias para realizar pruebas en la línea 4 .
En la línea 6 , creamos una función, dentro de la cual, en la línea 7 , llamamos a la get_digits función definida en el get_digits.py archivo.
Usando la self.assertEqual función en la línea 8 , comparamos la salida esperada con la que proviene de la get_digits función.
En la línea 10 , usamos nuevamente el self.assertEqual método para verificar otro caso de prueba.
Finalmente, unittest.main se llama en la línea 13.

Interfaces de Usuario
Introducción a Tkinter
Obtenga información sobre los diálogos y la estructura de Tkinter, que viene incluida con la distribución estándar de Python.
Introducción a Tkinter 
Existen varios sistemas GUI (Interfaz Gráfica de Usuario, que se pronuncia "gooey") que pueden usarse con Python. Aquí nos centraremos principalmente en Tkinter , que viene incluido en la distribución estándar de Python.
Tkinter es la biblioteca GUI estándar preinstalada con Python y ofrece una interfaz muy sencilla y fácil de usar que permite crear aplicaciones GUI. Cuenta con una amplia colección de widgets, incluyendo botones, menús y campos de texto, que ayudan a los usuarios a crear y construir elementos GUI.
Diálogos en Tkinter 
Muchos programas no requieren una interfaz gráfica completa, solo uno o dos cuadros de diálogo. Tkinter ofrece varias. Para usarlas, importe messagebox, simpledialogy/o filedialogdesde Tkinter .
Cuadros de mensajes 
Los cuadros de mensaje se utilizan para mostrar mensajes con un título mediante messageboxel cuadro de diálogo de Tkinter. A continuación, se incluye una lista de messageboxtipos:
messagebox.showinfo(title, message)
messagebox.showwarning(title, message)
messagebox.showerror(title, message)
Todos los anteriores son básicamente iguales. La diferencia radica en el icono que se muestra con cada tipo. Todos ofrecen un title, un messagey un OK botón.
result = messagebox.askyesno(title, question)
Esto proporciona los botones No y Yes, que devuelven False o True, respectivamente.
El siguiente código crea cuatro cuadros de mensajes diferentes: uno con un mensaje informativo, uno con un mensaje de error, uno con un mensaje de advertencia y uno con una pregunta de sí o no:
from tkinter import messagebox

messagebox.showinfo("Important Information","A message displaying important information")
messagebox.showerror("Error Message", "A message displaying an Error message")
messagebox.showwarning("Warning Message","A message displaying a warning message")

answer = messagebox.askyesno("Question Dialog","Do you like Python?")
Cuadros de diálogo simples 
Los cuadros de diálogo simples solicitan una entrada al usuario, que puede ser un entero, un valor de punto flotante o una cadena. Aquí hay algunos simpledialogmétodos:
result = simpledialog.askfloat(title, message)
result = simpledialog.askinteger(title, message)
result = simpledialog.askstring(title, message)
Los tres anteriores permiten al usuario introducir un número de punto flotante, un entero o una cadena. Proporcionan los botones Cancely OK. Si se cancela el diálogo, el valor devuelto es None.
Diálogos de archivos 
Los cuadros de diálogo de archivos se utilizan para seleccionar y guardar carpetas y archivos. Existen dos filedialogmétodos:
input_file = filedialog.askopenfilename(initialdir=path, title=title)
solicita que se lea un archivo. Si initialdirse proporciona un argumento, la navegación comienza desde ese punto.
output_file = filedialog.asksaveasfilename(initialdir=path)
Solicita la ubicación donde guardar un archivo. Si initialdir se proporciona un argumento, la navegación comienza desde ese punto.
Una pequeña molestia es que, al invocar Tkinter, debe mostrar una "ventana raíz". Esta es una pequeña ventana que aparece en la pantalla y no tiene ninguna función. Para eliminarla, crea una ventana raíz tú mismo y ocúltala inmediatamente. Solo necesitas hacerlo una vez.
import tkinter
root = tkinter.Tk()
root.withdraw()
Estructura de Tkinter 
Los programas con interfaz gráfica de usuario (GUI) funcionan de forma diferente a los programas sin ella. En lugar de que todo el código esté bajo el control de un método principal, el programa crea una GUI y, a partir de entonces, todo lo que ocurre es resultado de alguna interacción con ella. Por ejemplo, el usuario pulsa un botón y ejecuta cierto código.
Los programas de Tkinter suelen comenzar con lo siguiente:
from tkinter import *
import tkinter.ttk
A continuación, su código debería crear una ventana “raíz”:
top = Tk()
Luego, llena la ventana con widgets de Tkinter. Aquí es donde suelen realizarse las tres tareas principales:
Crea algunos widgets (botones, áreas de texto, etc.).
Organizar los widgets en la ventana.
Asociar código con algunos de los widgets.
Finalmente, transfiera la ejecución a la GUI:
top.mainloop()
Puede ejecutar algún código de inicialización antes de llamar a mainloop, pero una vez que llama a mainloop, la GUI tiene el control.
En el ejemplo a continuación, puede añadir o editar el código para probar algo diferente. Pulse el botón Ejecutar después de modificar el código y, si no ve el resultado deseado en la interfaz gráfica, abra la pestaña Terminal para comprobar si se produjo un error de compilación o una excepción.
Para comprender mejor los conceptos de GUI analizados en esta lección, puede agregar código para crear diferentes tipos de diálogos y cuadros de mensajes para tener una idea de cómo funcionan en lo siguiente:
from tkinter import * # importing tkinter properties
import tkinter.ttk 

top = Tk() # creating root window
top.mainloop() # starting execution

Colocación de widgets en la GUI
Descripción general de un programa GUI 
La ventana que se abre al ejecutar un programa GUI es un contenedor, es decir, la sección que contiene los widgets (botones, áreas de texto, etc.). Un marco es un contenedor que se puede colocar dentro de una ventana o en otro marco.

Métodos para organizar widgets 
Hay tres métodos para organizar los widgets en la ventana principal y los marcos:
pack
grid
place
Sólo se debe utilizar uno de estos métodos en cada ventana o marco.
También se pueden crear diseños complejos colocando varios marcos en la ventana y/o en otros marcos y utilizando diferentes diseños para la ventana y cada marco.
Marco del paquete 
Simplemente añade el widget a la ventana o marco. Aquí, widget.pack(options)se muestran los siguientes:options
Utilice side=side donde side es uno de LEFT, RIGHT, TOP, o BOTTOM para agregar widgets comenzando desde ese lado.
Úselo expand=True para expandir el widget para llenar el espacio disponible.
Úselo fill=how para expandir el widget, donde how puede ser X(horizontalmente), Y(verticalmente), BOTH o NONE.

Marco de cuadrícula 
Añade el widget a una cuadrícula (tabla bidimensional). Aquí, widget.grid(options) incluye :options
row=n:Esto selecciona en qué fila colocar el widget. El valor predeterminado es la primera fila disponible.
column=n:Esto selecciona en qué columna colocar el widget. El valor predeterminado es 0.
rowspan=n:Este es el número de filas que debe ocupar el widget.
columnspan=n:Este es el número de columnas que debe ocupar el widget.
ipadx=n, ipady=n: Esta es la cantidad (en píxeles) que se debe rellenar el widget, horizontal y verticalmente.
sticky=d:Aquí es donde se coloca el widget si está en un espacio más grande. dpuede ser N, S, E, W, NE, SE, NW, o SW.
Coloque el marco 
Especifica widget.place(options)exactamente dónde colocar cada widget. optionsAquí están:
x=pixels, y=pixels: Esta es la xposición ydel punto de anclaje del widget, en relación con el padre
anchor=d: Esta es la parte del widget a la que se refieren los xelementos y . Puede ser , , , , , , o . El valor predeterminado es (esquina superior izquierda).ydNSEWNESENWSWNW
bordermode=OUTSIDE:Use esto para tener en cuenta el borde del padre al posicionar el widget; use de lo contrario INSIDE( INSIDEes el valor predeterminado).
height=pixels, width=pixels: Estos son la altura y el ancho del widget en píxeles.
relx=float, rely=float: Esta es la posición xy y, como una fracción entre 0,0 y 1,0, del ancho y la altura del padre.
relwidth=float, relheight=float: Este es el tamaño del widget, como una fracción entre 0.0 y 1.0, del ancho y alto del padre.

Creación de widgets
Widgets en Tkinter 
Hay quince tipos de widgets en Tkinter, y cada uno tiene numerosas opciones, indicadas con option=value. Esta lección cubrirá solo los tipos y opciones más comunes. Para más detalles, la documentación oficial de Python para Tkinter es una excelente referencia.
En el siguiente ejemplo de widget, asumiremos que la ventana se llama top.
fr = Frame(parent, option, ...)
Este es un contenedor para widgets. parentPuede ser la ventana de nivel superior ( top) u otro marco. Algunas opciones útiles son bg=color, el color de fondo (como nombre de color o número hexadecimal) y bd=n, el ancho del borde en píxeles.
but = Button(parent, text=string, command=function_name)
Esto crea un botón que contiene la cadena que llamará a la función nombrada al hacer clic. No se pueden proporcionar parámetros a la función.
lab = Label(parent, text=string)
Esto crea una etiqueta que se puede mostrar pero no editar.
Para cambiar el texto, utilice lab.configure(text=new_text).
ent = Entry(parent, width=n)
Esto crea un rectángulo lo suficientemente grande como para mostrar aproximadamente ncaracteres, en el que el usuario puede escribir una sola línea de texto. nSe pueden introducir más de caracteres, pero es posible que no todos sean visibles.
Para recuperar el texto, llame ent.get().
txt = Text(parent, width=num_characters, height=num_lines)
Esto crea un rectángulo num_charactersde ancho y num_linesalto, en el que el usuario puede escribir varias líneas de texto. Se puede introducir cualquier número de líneas, pero solo el número especificado será visible.
Para recuperar el texto, llame txt.get(1.0, END).
Para eliminar todo el texto, utilice txt.delete(1.0, END).
Para insertar texto, utilice txt.insert(END, text).
var = IntVar()
Esto se define varcomo un IntVar(ver Checkbutton más abajo).
chk = Checkbutton(parent, text=string, variable=var, command=function)
El vardebe definirse con IntVar().
Esto crea una casilla de verificación con el texto dado.
var.get()devolverá 1 si está marcado y 0 si no está marcado.
Ejemplo de Tkinter 
Aquí tienes un programa para lanzar un dado. Al ejecutarlo, la ventana debería aparecer en el centro de la pantalla. Puedes salir del programa cerrando la ventana.

Módulos
Módulos integrados
Introducción a los módulos 
La distribución estándar de Python incluye unos 300 módulos; hay muchos más disponibles, tanto comerciales como gratuitos. Antes de escribir código para resolver problemas en un dominio determinado, conviene comprobar primero qué hay disponible.
En este capítulo se explorarán los paquetes integrados math, statistics, y os(sistema operativo).
Módulo de matemáticas 
El módulo matemático ( import math) incluye muchas funciones que no están disponibles por defecto. Estas son algunas de ellas:
ceil(x): el entero más pequeño mayor o igual a x
floor(x): el entero más grande menor o igual a x
comb(n, k): el número de posibles subconjuntos de tamaño k de un conjunto de nelementos
perm(n, k): el número de posibles secuencias ordenadas de tamaño k, donde los elementos se extraen sin reemplazo de un conjunto de nelementos
factorial(n): el producto de los primeros nnúmeros enteros positivos
gcd(a, b):el máximo común divisor de los dos argumentos
log(x), log2(x), log10(x), log(x, b): devuelve el logaritmo de xen base e, 2, 10, y b, respectivamente
exp(x): regresamiincógnitamiincógnita
sqrt(x): devuelve la raíz cuadrada dex
Las funciones trigonométricas: sin(x), cos(x), tan(x), asin(x), acos(x)y atan(x), donde xestá en radianes.
Las funciones hiperbólicas: sinh(x), cosh(x), tanh(x), asinh(x), acosh(x)y atanh(x), donde xestá en radianes.
Las funciones de conversión: degrees(radians) y radians(degrees).

Comandos del sistema operativo
Introducción a los comandos del sistema operativo 
Python también proporciona un módulo con comandos del sistema operativo. Estas son algunas de las funciones disponibles cuando import os:
Úselo os.getcwd()para obtener el directorio de trabajo actual.
Úselo para cambiar el directorio de trabajo actual a .os.chdir(path)path
Se utiliza os.listdir(path=".")para devolver una lista que contiene los nombres de las entradas en el directorio ".".
Tenga en cuenta que path"(sin cursiva)" en el listdir método es una palabra clave, no una variable que pueda reemplazarse por otro nombre. En todas las demás funciones, path"(en cursiva)" representa una variable que contiene una cadena o la propia cadena.
Utilice os.mkdir(path, mode=0o777, dir_fd=None) para crear un directorio path con el nombre numérico mode 0o777. Aquí, dir_fd es un parámetro opcional que se utiliza como identificador único que hace referencia a un directorio con Nonecomo valor predeterminado.
Este modees un código octal de tres dígitos, de estilo Unix. Los tres bits de cada dígito especifican los permisos de lectura, escritura y ejecución para el propietario, el grupo y el mundo.
Utilice os.remove(path, dir_fd=None)para eliminar un solo archivo path. Esto no aplica a directorios.
Utilice os.rmdir(path, dir_fd=None)para eliminar (borrar) el directorio vacío path.
Úselo os.rename(src, dst)para cambiar el nombre del archivo o srcdirectorio a dst.
Se utiliza os.walk(top, topdown=True, onerror=None, followlinks=False)para generar los nombres de archivo en un árbol de directorios recorriéndolo de arriba a abajo o de abajo a arriba. Para cada directorio del árbol con raíz en el directorio superior (incluido topél mismo), se genera una 3-tupla ( dirpath, dirnames, filenames).

Estadística
Introducción al módulo de estadística 
Python también proporciona un módulo de estadísticas. Estas son algunas de las funciones que ofrece el import statistics comando:
mean(data): Este es el “promedio” matemático de los datos, calculado sumando los valores y dividiendo su suma por el número de valores.
median(data)Este es el número "medio" de los datos, lo que significa que la mitad de los valores son menores o iguales a él y la otra mitad son mayores o iguales a él. Si hay un número par de valores, el resultado es la media de los dos valores medios.
mode(data): Este es el número que aparece con mayor frecuencia en los datos (o si hay un empate, el primer número).
A continuación se muestran algunas funciones más utilizadas:
stdev(data)devuelve la desviación estándar de la muestra.
variance(data)devuelve la varianza de la muestra.
El statistics módulo también incluye funciones para trabajar con distribuciones normales.
Apéndice
Introducción a los métodos de cadena 
A continuación se muestran algunos de los métodos de cadena disponibles en el stringmódulo (en orden alfabético):
string.center(int)devuelve una copia de stringcentrada en una cadena de longitud int.
string.count(substring)devuelve el número de ocurrencias no superpuestas de substringen string.
string.endswith(suffix)devuelve Truesi stringtermina en suffix.
string.find(substring)devuelve el índice al comienzo de la primera aparición de substringin string, o devuelve -1 si no se encuentra.
string.isalnum()Comprueba si todos los caracteres son alfanuméricos (letras o dígitos). Devuelve Truesi todos los caracteres son alfanuméricos o Falseno (incluso si stringestá vacío).
string.isalpha()Comprueba si todos los caracteres son alfabéticos. Devuelve Truesi todos los caracteres son alfabéticos o Falseno (incluso si stringestá vacío).
string.isdigit()Comprueba si todos los caracteres son dígitos. Devuelve Truesi todos los caracteres son dígitos o Falseno.
string.isidentifier()Devuelve Truesi stringes una cadena no vacía que consta únicamente de letras, dígitos o guiones bajos, y no empieza con un dígito. Devuelve Falsesi la cadena está vacía, empieza con un dígito o contiene cualquier otro carácter.
string.islower()Comprueba si todas las letras stringestán en minúsculas. Devuelve Truesi todos los caracteres están en minúscula o Falseno (incluso si stringestá vacío).
string.isprintable()Comprueba si stringno contiene caracteres de control. Devuelve Truesi se pueden imprimir todos los caracteres o Falseno.
string.isspace()Comprueba si todos los caracteres son espacios en blanco (espacios, tabulaciones, saltos de línea y algunos caracteres Unicode). Devuelve Truesi todos los caracteres son espacios en blanco o Falseno (incluso si stringestá vacío).
string.isupper()Comprueba si todas las letras stringestán en mayúsculas. Devuelve Truesi todos los caracteres están en mayúsculas o Falseno (incluso si stringestá vacío).
string.ljust(int)devuelve una copia stringjustificada a la izquierda en un campo de longitud int.
string.lower()devuelve una copia de stringcon todas las letras mayúsculas reemplazadas por sus equivalentes en minúsculas.
string1.partition(string2)devuelve una tupla de 3: (la parte de string1before string2, string2ella misma y la parte after string2).
string1.replace(string2, string3)devuelve una copia de string1con todas las ocurrencias de string2reemplazadas con string3.
string.rjust(int)devuelve una copia de la cadena justificada a la derecha en un campo de longitud int.
string1.split(string2)Devuelve una lista de las subcadenas de string1que están separadas por string2. Si string2se omite, se utiliza un espacio como separador.
string.splitlines()devuelve una lista de las líneas en string, descartando las nuevas líneas.
string.startswith(prefix)devuelve Truesi stringcomienza con prefix.
string.strip()devuelve una copia stringcon todos los espacios iniciales y finales eliminados.
string.upper()devuelve una copia de stringcon todas las letras minúsculas reemplazadas por sus equivalentes en mayúsculas.
string.join()devuelve una cadena uniendo todos los elementos de una secuencia/iterable separados por un separador de cadena.
string.format()formatea los valores especificados y los inserta en los marcadores de posición de la cadena.
Además: una cadena puede tratarse como una lista de caracteres, por lo que todos los list métodos se pueden aplicar a cadenas.
Funciones numéricas
Números 
A continuación se muestran algunas funciones integradas en números:
abs(x)devuelve el valor absoluto de un número x(o la magnitud de un número complejo).
bin(int)devuelve una representación de cadena binaria de int.
chr(int)Devuelve el carácter cuya representación Unicode es int. La operación inversa es ord.
divmod(x, y)devuelve la tupla ( x // y,  x % y) para números enteros.
float(x)Convierte una cadena o un entero xen un número de punto flotante.
hex(int)devuelve una representación de cadena hexadecimal de int.
int(x)convierte una cadena xen un entero o trunca un flotante xen un entero.
oct(int)devuelve una representación de cadena octal de int.
pow(x, y)vuelve xal poder y.
round(float)devuelve el entero más cercano al floatvalor dado.
round(float, int)devuelve floatredondeado a intdígitos después del punto decimal.
Matemáticas 
Estas son algunas de las funciones que obtendrás si importas el mathmódulo:
math.ceil(x)devuelve el entero más pequeño mayor o igual a x.
math.floor(x)devuelve el entero más grande menor o igual a x.
math.trunc(x)devuelve el valor entero obtenido al eliminar todo lo que esté después del punto decimal.
math.sqrt(x)devuelve la raíz cuadrada de x.
math.log(x)y math.log10(x)devuelven el logaritmo natural y el logaritmo de base 10 de x, respectivamente.
Todas las funciones trigonométricas habituales: sin, cos, acos, radians, etc.
También en la biblioteca de matemáticas: las constantes math.pi, math.e, y math.tau.
Aleatorio 
Estas son algunas de las funciones que obtendrá si importa el randommódulo:
random.choice(seq)devuelve un elemento elegido aleatoriamente con un reemplazo de la secuencia seq.
random.shuffle(seq)baraja la secuencia seqen su lugar (es decir, se cambia la lista original).
random.random()devuelve un número de punto flotante aleatorio, r, en el rango 0.0 ≤ r < 1.0.
random.randint(a, b)devuelve un entero aleatorio, r, en el rango a ≤ r ≤ b.
Recordatorio: si utiliza la from module import * versión de la importdeclaración, puede omitir el module prefijo (en este caso, math.y random.).
Funciones en iterables
Introducción a las funciones en iterables 
Aquí hay algunas funciones que toman objetos iterables ( listas , conjuntos , tuplas , cadenas , diccionarios , rangos , archivos y posiblemente otros).
all(iterable)devuelve Truesi cada elemento de iterableevalúa un valor verdadero.
any(iterable)devuelve Truesi al menos un elemento de iterableevalúa un valor verdadero.
filter(test, iterable)devuelve un iterador para los elementos que iterablepasan el test.
len(iterable)devuelve el número de elementos en iterable.
list(iterable)devuelve una lista de los elementos en iterableel mismo orden.
map(function, iterable)Devuelve un iterador. Cada valor devuelto por el iterador será el resultado de aplicar functional valor correspondiente de iterable.
max(iterable)devuelve el valor más grande en iterable.
min(iterable)devuelve el valor más pequeño en iterable.
set(iterable)devuelve un conjunto de valores en iterable.
sorted(iterable)devuelve una lista de los elementos en iterableorden ordenado.
sum(iterable)devuelve la suma de los valores en iterable.
tuple(iterable)devuelve una tupla de los elementos en iterableel mismo orden.
zip(iterable1, ...,iterableN)Devuelve un iterador de N -tuplas, donde la primera tupla contiene el primer valor de cada iterable, la segunda tupla contiene el segundo valor de cada iterable, y así sucesivamente. La iteración se detiene cuando alguno de los iterables agota sus valores.
element in iterabledevuelve Truesi el elemento está en iterable.
element not in iterabledevuelve Truesi el elemento no está en iterable.
En general, las funciones que toman un objeto iterable también pueden tomar un iterador o un generador.
Algunas de las funciones anteriores deben examinar cada elemento del iterable ( maxpor ejemplo). Otras, como any, pueden o no examinar cada elemento. Tenga cuidado de no llamar a esta función con un iterador o generador que produzca un número infinito de valores.
Métodos de prueba unitaria
Introducción a los métodos de pruebas unitarias 
Los siguientes son algunos de los métodos disponibles en unittest, la mayoría de los cuales se explican por sí solos.
assertEqual(a, b)
assertNotEqual(a, b)
assertAlmostEqual(a, b)para números de punto flotante
assertAlmostEqual(a, b, places)
assertTrue(x)
assertFalse(x)
assertIs(a, b)pruebas de identidad
assertIsNot(a, b)
assertIsNone(x)
assertIsNotNone(x)
assertIn(a, b)
assertNotIn(a, b)
assertIsInstance(a, b)
assertNotInstance(a, b)
assertRaises(exception, function, arguments)
Todas las llamadas a estos métodos deben tener el prefijo self..
Cada uno de estos métodos tiene un parámetro final opcional, que puede ser cualquier expresión. Este parámetro final se utiliza como mensaje para AssertionError proporcionar cualquier información adicional necesaria.
