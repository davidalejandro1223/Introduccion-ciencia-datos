# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import matplotlib.pyplot as plt

'''
Inicializacion de parametros

num_features: numero de filtros de salida en la convolucion
num_labels: numero de clases (7)
    0: Enojado
    1: Disgustado
    2: Temeroso
    3: Feliz
    4: Triste
    5: Sorpresa
    6: Neutral
batchs_size: cantidad imagenes por lote
epochs: cantidad de epocas
width, height: dimension de la imagen (pixeles)
'''
num_features = 64
num_labels = 7
batch_size = 64
epochs = 50
width, height = 48, 48

'''
Datos totales: 35887
Datos Entrenamiento: 29068
Datos Validacion: 3230
Datos Test:3589
Estructura: emotion, pixels
'''
data = pd.read_csv('fer2013.csv')

pixels = data['pixels'].tolist()

faces = []
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    face = np.asarray(face).reshape(width, height)
    faces.append(face.astype('float32'))

faces = np.asarray(faces)
faces = np.expand_dims(faces, -1)

emotions = pd.get_dummies(data['emotion']).as_matrix()

'''
test_size:  proporción del conjunto de datos que se incluirá en la división de prueba
random_state: es la semilla utilizada por el generador de números aleatorios
X_train, y_train: datos de entrenamiento
X_test, y_test: datos de prueba
X_val, y_val: datos validacion

'''
X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=41)

'''
Modelo CNN

4 Capas Convolucionales
Capa 2,3,4 incluye normalización por lotes, que normaliza las entradas sobre el lote
Funcion de activacion: relu
MaxP: 2x2
Dropout: 0.5
'''
model = Sequential()

model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(2*2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*num_features, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels, activation='softmax'))

#model.summary()

'''
Compilacion modelo

Funcion de perdida: categorical_crossentropy ya que se tienen mas de dos clases 
Optimizador: Adam
'''
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

'''
Caracteristicas

Ayuda a la función de pérdida a deshacerse de “mesetas” al reducir el parámetro 
de velocidad de aprendizaje de la función de optimización con un cierto valor (factor) 
si no hay una mejora en el valor de la función de pérdida para el conjunto 
de validación después de un cierta época (patience).

Registro de lo hecho durante el entrenamiento (TensorBoard)

Se detiene el entrenamiento del modelo si no hay ningún cambio en el valor
de la función de pérdida en el conjunto de validación para una época determinada (patience).
'''
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
tensorboard = TensorBoard(log_dir='logs')
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')

'''
Checkpoint de los entrenamientos
'''
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)

'''
Entrenamiento del modelo -- NO EJECUTAR: MODELO YA ENTRENADO
'''
model.fit(np.array(X_train), np.array(y_train),
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(np.array(X_val), np.array(y_val)),
          shuffle=True,
          callbacks=[lr_reducer, tensorboard, early_stopper, checkpointer])

'''
Evaluacion del modelo
'''
scores = model.evaluate(np.array(X_test), np.array(y_test), batch_size=batch_size)
print("Loss: " + str(scores[0]))
print("Accuracy: " + str(scores[1]))

'''
Guardar modelo
'''
model.save("model.h5")
model.save_weights("model_weights.h5")

'''
Carga del modelo y pesos
'''
model = load_model('model.h5')
model.load_weights("model_weights.h5")

'''
Proceso de prueba, prediccion, matriz de confusion y reporte de clasificacion
tras la carga del modelo
'''
scores = model.evaluate(np.array(X_test), np.array(y_test), batch_size=batch_size)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

prediction = model.predict(np.array(X_test))

Y_pred = prediction
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(y_test.argmax(axis=1), y_pred))
print('Classification Report')
target_names = ['Enojado', 'Disgustado', 'Temeroso', 'Feliz', 'Triste', 'Sorpresa', 'Neutral']
print(classification_report(y_test.argmax(axis=1), y_pred, target_names=target_names))

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(7,7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
# matriz de confusion
cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred)
np.set_printoptions(precision=2)

# plot matriz de confusion normalizada
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=target_names, title='Matriz de confusion normalizada')
plt.show()
