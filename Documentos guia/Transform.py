# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

a = np.random.randint(low=0, high=10, size=(5, 3))
df = pd.DataFrame(data=a, columns=['A', 'B', 'C'])
print(df)


# PIPE -> permite aplicar funciones de forma encadenada que reciban y retornen dataframes 


# Creamos las funciones
def funcion1(dataframe):
    return dataframe.abs()

def funcion2(dataframe, col):
    # col = (col^2 + col)^(1/2)
    dataframe[col] = np.sqrt(dataframe[col]**2 + dataframe[col])
    return dataframe


# Creamos el pipe con las funciones anteriores
print(df.pipe(funcion1).pipe(funcion2, 'A'))

# EJERCICIO 1
# Crear un ejemplo de pipe con tres funciones.
# La primera y la tercera función deben tener argumentos diferentes al dataframe


# APPLY -> permite aplicar funciones a dataframes, ya sean funciones existentes o definidas por el programador

# función ya existente
print(df.apply(np.mean))
print(df.apply(np.mean, axis=1))

# función definida
print(df.apply(lambda x: x.max() - x.min()))
print(df.apply(lambda x: x.max() - x.min(), axis=1))

# EJERCICIO 2
# Mediante lambda cree una fución que permita obtener el IMC a partir de un data frame que contenga el peso y la estatura de personas.



# AGG-> Crea un nuevo dataframe con una fila por cada función aplicada a cada una de las columnas de un dataframe de entrada

print(df.agg(['sum', 'mean']))

print(df.agg([np.sum, 'mean']))

# EJERCICIO 3
# Mediante agg cree un dataframe resumen que contenga 5 valores estadísticos de un dataframe de entrada



# TRANSFORM-> Crea un nuevo dataframe con los datos resultantes de aplicar una o varias funciones a cada uno de los elementos de un dataframe de entrada

print(df.transform([np.abs, lambda x: x+1]))
# EJERCICIO 4

# Genere un dataframe donde se almacene el precio (en pesos colombianos) de algunos productos a través de los meses de un año
# A partir del dataframe anterior genere un nuevo dataframe donde los precios de los productos se encuentren en dolares, euros y libras.

# APPLYMAP> Permite aplicar a un dataframe una función que recibe y retorna un único valor

f = lambda x: len(str(x/1000))
print(df.applymap(f))

# EJERCICIO 5
# Realice el ejercicio 4 pero únicamento convirtiendo de pesos a dolares



