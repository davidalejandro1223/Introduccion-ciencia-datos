import numpy as np
import sklearn.svm as svm
from sklearn.model_selection import train_test_split

def my_kernel(p1, p2):
    r = 1-np.dot(p1, p2.T)
    return np.exp(-r**2/4)

X1 = np.random.random((50,5))
#50x5
print('1', X1)
print('------')
X1 = X1/np.sum(X1)
print('2', X1)
print('----------')
X1 = X1.T
#5x50
print('3', X1)
print('---------------')
Y1 = [0, 0, 1, 1, 1]
print('4', Y1)

x_train, x_test, y_train, y_test = train_test_split(X1, Y1)

my_svm = svm.SVC(kernel=my_kernel)

my_svm.fit(x_train, y_train)

my_svm.score(x_test, y_test)

my_svm.predict(X1)

#my_svm.fit(X1,Y1)

#y_pred = my_svm.predict(X1)

#y_pred
