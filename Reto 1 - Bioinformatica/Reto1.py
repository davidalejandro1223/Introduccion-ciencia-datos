import numpy as np

conjunto = np.array(['A', 'C', 'T', 'G'], dtype=str)
SecuenciaADN = np.random.choice(conjunto, 120, replace=True)
SecuenciaComparacion = np.random.choice(conjunto, 20, replace=True)
relleno = np.full(19,'*', dtype=str)

print(SecuenciaADN)
print(SecuenciaComparacion)

def comparacion(Secuencia):
    Secuencia = np.append(Secuencia, relleno)
    Secuencia = np.append(relleno, Secuencia)
    print(Secuencia.size)
    puntaje = np.zeros(1)

    
    for i in range(0, Secuencia.size-19):
        cont = 0;
        for j in range(19,-1,-1):
            if(SecuenciaComparacion[j]==Secuencia[(19-(19-j))+i]):
                cont+=1
        puntaje = np.append(puntaje, cont)
    return puntaje[1:]

matriz_puntaje = comparacion(SecuenciaADN)
print(matriz_puntaje.size)
indices_maximos = np.where(matriz_puntaje == np.amax(matriz_puntaje))

print(matriz_puntaje)
print(indices_maximos)