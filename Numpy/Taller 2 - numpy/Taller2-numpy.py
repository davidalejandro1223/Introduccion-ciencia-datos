import numpy as np
import copy

mat = np.arange(729).reshape(9,9,9)
print (mat)

idO = int(input('Ingrese el primer id \n'))
idD = int(input('Ingrese el segundo id \n'))

def limites(id):
    x=id%3
    y=(id%9)//3
    z=id//9

    limites = np.array([z,y,x])
    return limites

def mover(limitesA, limitesB):
    tempA = mat[limitesA[0]*3:(limitesA[0]*3)+3, limitesA[1]*3:(limitesA[1]*3)+3, limitesA[2]*3:(limitesA[2]*3)+3]
    tempB = copy.deepcopy(mat[limitesB[0]*3:(limitesB[0]*3)+3, limitesB[1]*3:(limitesB[1]*3)+3, limitesB[2]*3:(limitesB[2]*3)+3])
    mat[limitesB[0]*3:(limitesB[0]*3)+3, limitesB[1]*3:(limitesB[1]*3)+3, limitesB[2]*3:(limitesB[2]*3)+3] = tempA
    mat[limitesA[0]*3:(limitesA[0]*3)+3, limitesA[1]*3:(limitesA[1]*3)+3, limitesA[2]*3:(limitesA[2]*3)+3] = tempB     

limA=limites(idO)
limB=limites(idD)

mover(limA, limB)

print(mat)