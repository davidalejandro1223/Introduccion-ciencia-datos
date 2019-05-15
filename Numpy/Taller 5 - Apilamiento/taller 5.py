import numpy as np
from matplotlib.pyplot import imread, imshow

imagen = imread(fname="imagen.png")
print(imagen)
imshow(imagen)

def limites(id):
    x=id%4
    y=(id%16)//4

    limites = np.array([y,x])
    return limites

def desordenar_matriz(matriz):
    matriz_aleatoria = np.random.choice(np.arange(16), 16, replace=False).reshape((4,4))
    celdas_ver = np.zeros((55,275))
    for y in range(0,4,2):
        celdas_hor_1 = np.zeros((55,55))
        celdas_hor_2 = np.zeros((55,55))
        for x in range(0,4,2):
            #se toman los primeros 4 id aleatorios de arriba-abajo y izq-derecha 
            #de la matriz aleatoria para obtener los limites de cada bloque especifico
            id_celda_1 = limites(matriz_aleatoria[y,x])
            id_celda_2 = limites(matriz_aleatoria[y,x+1])
            id_celda_3 = limites(matriz_aleatoria[y+1,x])
            id_celda_4 = limites(matriz_aleatoria[y+1,x+1])
            
            #se extrae la informacion de cada bloque en especifico
            celda_temp_1 = np.array(matriz[id_celda_1[0]*55:(id_celda_1[0]*55)+55, id_celda_1[1]*55:(id_celda_1[1]*55)+55])
            celda_temp_2 = np.array(matriz[id_celda_2[0]*55:(id_celda_2[0]*55)+55, id_celda_2[1]*55:(id_celda_2[1]*55)+55])
            celda_temp_3 = np.array(matriz[id_celda_3[0]*55:(id_celda_3[0]*55)+55, id_celda_3[1]*55:(id_celda_3[1]*55)+55])
            celda_temp_4 = np.array(matriz[id_celda_4[0]*55:(id_celda_4[0]*55)+55, id_celda_4[1]*55:(id_celda_4[1]*55)+55])
            
            #se hace un apilamiento horizontal entre las celdas correspondientes a sus filas
            celdas_hor_1 = np.hstack((celdas_hor_1,celda_temp_1, celda_temp_2))
            celdas_hor_2 = np.hstack((celdas_hor_2,celda_temp_3, celda_temp_4))

        #se hace un apilamiento vertical de cadas filas completas
        celdas_ver = np.vstack((celdas_ver,celdas_hor_1, celdas_hor_2))
    return celdas_ver

#se elimina las primeras 55 filas y columnas puesto que fueron agregadas para poder
#usar la funcion de apilamiento vertical y horizontal
imagen_desordenada = desordenar_matriz(imagen)
imshow(imagen_desordenada[55:, 55:])