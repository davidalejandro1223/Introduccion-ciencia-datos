import numpy as np

mat = np.arange(729).reshape(9,9,9)
print (mat)

# p = x[1,0:4,0:4]
# f = x[0,0:4,0:4]
# p = f
# print(p)

idO = int(input('Ingrese el primer id'))
idD = int(input('Ingrese el primer id'))


def limites(id):
    x=id%9
    y=(id%9)//3
    z=id//9

    limites = np.array([z,y,x])
    return limites

def mover(limitesA, limitesB):
    temp = mat[limitesA[0]:limitesA[0]+3, limitesA[1]:limitesA[1]+3, limitesA[2]:limitesA[2]+3]
    mat[limitesA[0]:limitesA[0]+3, limitesA[1]:limitesA[1]+3, limitesA[2]:limitesA[2]+3] = mat[limitesB[0]:limitesB[0]+3, limitesB[1]:limitesB[1]+3, limitesB[2]:limitesB[2]+3]
    mat[limitesB[0]:limitesB[0]+3, limitesB[1]:limitesB[1]+3, limitesB[2]:limitesB[2]+3] = temp


limA=limites(idO)
limB=limites(idD)
mover(limA, limB)

print(mat)
