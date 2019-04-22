import random

num = (random.randrange(0,99))
print (num)
intents = 0

while intents<10:
    inp = int(input('Ingrese tu numero\n'))
    if inp == num:
        print('Has adivinado el numero')
        print('Tu puntaje fue: ' + str((10-intents)))
        break
    elif inp < num:
        print('El numero a adivinar es mayor')
        intents+=1
    elif inp > num:
        print('El numero a adivinar es menor')
        intents+=1
else:
    print('no has adivinado, el numero es:' + str(num))
    



