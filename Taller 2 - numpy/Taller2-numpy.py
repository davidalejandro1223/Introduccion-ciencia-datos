import numpy as np
import copy

mat = np.arange(729).reshape(9,9,9)
print (mat)

# p = x[1,0:4,0:4]
# f = x[0,0:4,0:4]
# p = f
# print(p)

idO = int(input('Ingrese el primer id \n'))
idD = int(input('Ingrese el segundo id \n'))


def limites(id):
    x=id%3
    y=(id%9)//3
    z=id//9
    return np.array([z,y,x])

limitesA=limites(idO)
limitesB=limites(idD)

tempA = mat[limitesA[0]*3:(limitesA[0]*3)+3, limitesA[1]*3:(limitesA[1]*3)+3, limitesA[2]*3:(limitesA[2]*3)+3]
print(tempA)
tempB = copy.deepcopy(mat[limitesB[0]*3:(limitesB[0]*3)+3, limitesB[1]*3:(limitesB[1]*3)+3, limitesB[2]*3:(limitesB[2]*3)+3])
print(tempB)
# Solo hace la primera linea
mat[limitesB[0]*3:(limitesB[0]*3)+3, limitesB[1]*3:(limitesB[1]*3)+3, limitesB[2]*3:(limitesB[2]*3)+3] = tempA
mat[limitesA[0]*3:(limitesA[0]*3)+3, limitesA[1]*3:(limitesA[1]*3)+3, limitesA[2]*3:(limitesA[2]*3)+3] = tempB     

print(limitesA)
print(limitesB)

print(mat)