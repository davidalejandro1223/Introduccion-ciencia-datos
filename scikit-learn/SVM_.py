from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm, datasets
import numpy as np

'''
iris = datasets.load_iris()
X = iris.data[:, :1] 
Y = iris.target

print(iris)
print('----')
print(X)
print('----')
print(Y)
'''

data = np.array([[-15], [-14], [-13], [-12], 
                 [-10], [-9], [-8], [-7], 
                 [-5], [-4], [-3], [-2], 
                  [0], [1], [2], [3],
                  [5], [6], [7], [8],
                  [10], [11], [12], [13]])

target = np.array([1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0])

'''
print(data)
print('---')
print(target)
'''

def my_kernel(X, Y):
    r = np.dot(X, Y.T)
    return np.sin(r)

def my_kernel_1(X, Y):
    r = np.dot(X, Y.T)
    return np.power(r, 2)

x_train, x_test, y_train, y_test = train_test_split(data, target)

clf = svm.SVC(kernel=my_kernel)

clf.fit(data, target)

#clf.fit(x_train, y_train)

clf.score(data, target)

#clf.score(x_test, y_test)

clf.predict(data)
