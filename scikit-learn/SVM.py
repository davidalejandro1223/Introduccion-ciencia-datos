

#intento de entrega de taller de kernels

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np

data = np.array([-15, -14, -13, -12, 
                 -10, -9, -8, -7, 
                 -5, -4, -3, -2, 
                  0, 1, 2, 3,
                  5, 6, 7, 8,
                  10, 11, 12, 13])
data = np.reshape(data, (-1,1))

target = np.array([1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0])

x_train, x_test, y_train, y_test = train_test_split(data, target)

classifier = SVC(gamma=0.001, C=100)

classifier.fit(x_train, y_train)

classifier.score(x_test, y_test)

classifier.predict(data[0])
