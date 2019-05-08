from matplotlib.pyplot import imread, imsave, imshow

import numpy as np

imagen = imread(fname="imagen3.jpg")
imagen = np.insert(imagen, 0, imagen[0, 0:], 0)
imagen = np.insert(imagen, -1, imagen[-1, 0:], 0)
imagen = np.insert(imagen, 0, imagen[0:, 0], 1)
imagen = np.insert(imagen, -1, imagen[0:, -1], 1)
print('La matriz generada de la imagen es: \n', imagen)

imshow(imagen)

mascara = np.array([[0, -2/9, 0],[-2/9, 3/9, -2/9],[0, -2/9, 0]])

print('La mascara es \n', mascara)

sizeImg = imagen.shape

for i in range (0, sizeImg[0]-2):
    for j in range(0, sizeImg[1]-2):
        temp = imagen[i:i+3, j:j+3]*mascara
        result = temp.sum()
        imagen[i+1, j+1] = result

imshow(imagen)
imsave("modificada3.png", imagen)