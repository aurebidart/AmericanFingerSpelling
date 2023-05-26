import tensorflow as tf
import model as m
import pandas as pd

# Cargar los datos
x_train = pd.read_csv("kk.csv")
y_train = pd.read_csv("train.csv", usecols=["sequence_id", "phrase"])


#phrase_string = x_train.loc[x_train['sequence_id']==1784552841]['phrase']
print(x_train)
print(y_train)


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

# Crear un callback para TensorBoard (visualización de métricas en el navegador puerto 6006)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)

# Entrenar el modelo con el callback de TensorBoard
num_epochs = 10
batch_size = 32
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=[tensorboard_callback])
