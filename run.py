import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import model as m
import pandas as pd
import numpy as np

y_train = np.load('y_train_dense.npy', allow_pickle=True)[:8996]

x_train = np.load('tensor_dense_filled.npy', allow_pickle=True)
print(y_train)

# Crear una instancia de la clase MiRedNeuronal
red_neuronal = m.MiRedNeuronal(num_frames=699, num_parametros=126, num_caracteres=59)

# Obtener el resumen del modelo
red_neuronal.resumen_modelo()

# Entrenar el modelo
red_neuronal.entrenar(x_train, y_train, epochs=10, batch_size=32)

# Evaluar el modelo
#red_neuronal.evaluar(x_test, y_test)

'''
# Parámetros de la red
vocab_size = 59  # Tamaño del vocabulario
embedding_dim = 50  # Dimensión del embedding
hidden_units = 256  # Unidades ocultas de la capa LSTM

# Construcción del modelo
model = m.build_model(vocab_size, embedding_dim, hidden_units)

# Compilación del modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Resumen del modelo
model.summary()

# Callback de TensorBoard
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', update_freq='epoch')
'''
