import numpy as np

x = np.arange(729).reshape(9,9,9)
print (x)

p = x[1,0:4,0:4]
f = x[0,0:4,0:4]
p = f
print(p)

idO = int(input('Ingrese el primer id'))
#idD = int(5input('Ingrese el primer id'))

x=0
y=0
z=0
cont=0

def limites(id, cont,x,y,z):
    cont=cont+1
    if(cont!=id):
        if(x<6):
            x=x+3
        else:
            x=0
            if(y<6):
                y=y+6
            else:
                y=0
                if(z<6):
                    z=z+3
                else:
                    z=0
        limites(id,cont,x,y,z)
    print(z,y,x)

limites(idO,0,0,0,0)
