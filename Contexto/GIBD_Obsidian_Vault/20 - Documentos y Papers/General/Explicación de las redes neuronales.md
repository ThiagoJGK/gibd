---
aliases: [Explicación de las redes neuronales]
tags:
  - grupo/general
  - estado/util
  - tipo/documento
formato_original: docx
fecha_modificacion: 2022-09-27
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Tareas Iván/Explicación de las redes neuronales.docx"
tamanio_bytes: 1569343
---

# Explicación de las redes neuronales

Ruta interna: `GIBD/Tareas Iván/Explicación de las redes neuronales.docx`

---

Red neuronal entrenada para poder pasar grados C a F (Regresión)
En la programación regular lo que se hace es dar un conjunto de entradas, programamos un algoritmo y de eso se obtiene un resultado o una salida. 
El Aprendizaje automático es al revés tenemos un conjunto de entradas, un conjunto de resultados, pero no tenemos el algoritmo en sí para poder obtener el resultado, la red neuronal tiene que aprender por sí sola como obtener los resultados esperados, y averiguar el algoritmo. 

La fórmula en cuestión para poder pasar de Celsius a Fahrenheit, en programación regular programaríamos un método que haga el cálculo directo. 

En aprendizaje automático ya tenemos la entrada en C y la salida en Fahrenheit y esto tiene que aprender como convertir  
Las redes tienen capas y neuronas, las capas tienen una o varias neuronas, todas tienen una capa de entrada 

Además de una capa de salida donde van a estar los datos de salida, que estos van a ser los grados F. 
Además puede haber más capas en el medio que se llaman ocultas las capas quedan con conexiones y cada conexión lleva un peso y cada neurona tiene un sesgo. 

Los pesos son los números en las conexiones y cada sesgo es el número dentro de cada círculo y cada uno de esos círculos son una neurona. 

La entrada son los 15 celsius, luego se multiplica por el sesgo y cuando llega a la neurona de salida y entonces se obtienen los datos F. 
Inicialmente los valores de peso y sesgo son aleatorios por tanto son malos resultados, pero como le estamos diciendo que está bien y que mal la red irá modificandose hasta poder obtener valores de peso y sesgo preciso y así poder resultados de salida. 

Primer clasificador de imágenes
Hay que ver 6 temas importantes antes de desarrollarlas
1)
La red neuronal anterior fue de regresión, en este caso las salidas son siempre números, la red daba como salida los grados Fahrenheit, otro ejemplo de lo que puede hacerse con regresión sería un tasador de casas. Toma como entradas las características de las mismas y dando como salida el valor de la casa: 

El problema de esta red no es de regresión, es un problema de clasificación, entonces debe recibir como entrada la imagen y dar como salida la categoría a la que pertenece. 
Si tenemos 10 categorías posibles, por ejemplo: Remera, pantalón, Zapatillas, medias, etc. la red va a tener 10 neuronas de salida. 

2)
¿Cómo hacemos para poder dar como entrada una imagen en una red neuronal?
Si tomamos la imagen y la pasamos a blanco y negro, luego dividiendo en pixeles podemos asignar un número a cada pixel y de ahí dar como entrada esos números, asignando el 0 a totalmente blanco y 255 a totalmente negro. 
Ejemplo: Si tomamos una imagen con 100*100 pixeles tenemos 10.000 pixeles cada uno con un valor entre 0 y 255 pixeles. 

Podemos tomar cada uno de los 10mil pixeles y ponerlos en la capa de entrada de la red, como esto es un número muy grande entonces vamos a poder reducir las imágenes al mínimo que son 28*28 pixeles. 

Ahora ya sabemos que vamos a crear una red con 784 neuronas de entradas y 10 neuronas de salida 
3) ¿Qué tipo de red neuronal vamos a utilizar?
Las redes neuronales convolucionales son muy utilizadas para clasificar imágenes, pero vamos a ir más atrás suponiendo que estas todavía no han sido inventadas y entonces voy a ir creando con la red neuronal regular, como la de los grados Fahrenheit hasta poder llegar hasta las convolucionales. 
4)
La red neuronal que convierte grados solamente puede resolver problemas lineales, y no puede con problemas más complejos, entonces debemos darle más herramientas para que pueda con los problemas más complejos, estas herramientas son capas ocultas y funciones de activación 
5)

En el primer caso teníamos la entrada conectada a la salida hay poco espacio para transformaciones, para esto debemos agregar más capas ocultas: 

Cada capa resuelve problemas lineales que al aumentar el número de capas entonces resuelve problemas más complejos pero al fin y al cabo solo sabe resolver problemas lineales 
6)
Para que cada neurona haga su trabajo lineal y podamos resolver problemas de mayor complejidad entonces vamos a pasar cada resultado de cada neurona por una función llamada función de activación.
Esto funciona de la siguiente manera: 

la capa recibe como entrada 10, tiene como sesgo el 5 entonces daría como salida el 15, pero antes de mandar ese 15 como resultado al resto de neuronas lo pasamos por una función de activación, aquí hay varias opciones. Ejemplo: 

Función RELU: Si el número es negativo el resultado es 0, si es mayor a cero lo mantiene igual. 
Estos cambios permiten que la red aprenda a resolver problemas no lineales. 

Para entrenar a la red voy a usar un Dataset de la tienda de ropa Zalando: 
Es un conjunto de datos de 70.000 imágenes de diferente ropa, voy a tomar 60.000 para entrenar el modelo y las 10.000 restantes para poder probarlo de que ha sido entrenado 

Red neuronal de visión artificial
	