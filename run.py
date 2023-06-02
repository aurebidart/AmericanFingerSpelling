import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import model as m
import pandas as pd
import numpy as np
import tensorflow as tf

y_train = np.load('y_train.npy', allow_pickle=True)
x_train = np.load('tensor_dense_filled.npy', allow_pickle=True)
#x_train_r = np.load('tensor_ragged_filled.npy', allow_pickle=True)

y_test = np.load('y_test_dense.npy', allow_pickle=True)[:1000]
x_test = np.load('tensor_test_dense_filled.npy', allow_pickle=True)

x_test = tf.expand_dims(x_test, axis=0)

#genera salida rara
#x_test = np.expand_dims(x_test, axis=-1)


# Crear una instancia de la clase MiRedNeuronal
red_neuronal = m.NeuralNetwork()

# Obtener el resumen del modelo
red_neuronal.train(x_train, y_train)

prediccion = red_neuronal.predict(x_test[0])

#sumar los valores de prediccion
suma = 0
for i in range(len(prediccion)):
    suma += prediccion[i]

print(y_train[0])
print(prediccion)
print(suma)
