import numpy as np
from matplotlib.pyplot import imread, imshow

imagen = imread(fname="/home/david/Documentos/Introduccion a la ciencia de datos/Taller 5 - Apilamiento/imagen.png")
print(imagen)
imshow(imagen)

def limites(id):
    x=id%4
    y=(id%16)//4

    limites = np.array([y,x])
    return limites

def desordenar_matriz(matriz):
    a = np.arange(16)
    matrizAleatoria = np.random.choice(a, 16, replace=False).reshape((4,4))
    for y in range(0,4,2):
        imagenAleatoriaP = None
        imagenAleatoriaS = None
        for x in range(0,4,2):
            idP = limites(matrizAleatoria[y,x])
            idS = limites(matrizAleatoria[y,x+1])
            tempP = np.array(matriz[idP[0]*55:(idP[0]*55)+55, idP[1]*55:(idP[1]*55)+55])
            tempS = np.array(matriz[idS[0]*55:(idS[0]*55)+55, idS[1]*55:(idS[1]*55)+55])
            imagenAleatoriaP = np.array((tempP, tempS))
        for x in range(0,4,2):
            idP = limites(matrizAleatoria[y+1,x])
            idS = limites(matrizAleatoria[y+1,x+1])
            tempP = np.array(matriz[idP[0]*55:(idP[0]*55)+55, idP[1]*55:(idP[1]*55)+55])
            tempS = np.array(matriz[idS[0]*55:(idS[0]*55)+55, idS[1]*55:(idS[1]*55)+55])
            imagenAleatoriaS = np.hstack((tempP, tempS))
        imagenfinal = np.vstack((imagenAleatoriaP, imagenAleatoriaS))
    return imagenfinal

imagen_desordenada = desordenar_matriz(imagen)
imshow(imagen_desordenada)