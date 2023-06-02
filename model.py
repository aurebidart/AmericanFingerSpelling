import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Attention, Dense, Conv1D, GlobalMaxPooling1D, BatchNormalization, Dropout, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Reshape
import numpy as np
import os
import datetime

import tensorflow as tf

class NeuralNetwork(tf.keras.Model):
    def __init__(self):

        super(NeuralNetwork, self).__init__()

        self.modelCNNDa = Sequential()
        self.modelCNNDa.add(Conv1D(32, 3, activation='selu', input_shape=(699, 126))) #relu
        self.modelCNNDa.add(MaxPooling1D(2))
        self.modelCNNDa.add(Dropout(0.2))
        self.modelCNNDa.add(Conv1D(64, 3, activation='selu')) #"selu"
        self.modelCNNDa.add(MaxPooling1D(2))
        self.modelCNNDa.add(GlobalMaxPooling1D())  # Cambio a GlobalMaxPooling1D
        self.modelCNNDa.add(Dense(126, activation='relu')) # , input_shape=(699, 126)
        self.modelCNNDa.add(Dense(59, activation='hard_sigmoid'))


        # self.modelCNNDa = Sequential()
        # self.modelCNNDa.add(Dense(64, input_shape=(699, 126), activation='selu')) #relu

        # # Aplanar los datos de entrada
        # self.modelCNNDa.add(tf.keras.layers.Flatten())

        # # Agregar la capa de salida
        # self.modelCNNDa.add(Dense(59, activation='hard_sigmoid')) #sigmoid

    def call(self, inputs, training=False):
        return self.modelCNNDa(inputs, training=training)
    
    def train(self, x_train, y_train, epochs=10, batch_size=128):

        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam()

        self.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

        logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

        self.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, use_multiprocessing=True, callbacks=[tensorboard_callback])
        

    def predict(self, x_test):
        # return tf.argmax(self(x_test), axis=1)

        #NUEVO
        predictions = self(x_test)
        probabilities_vector = predictions[0].numpy()
        return probabilities_vector
        
    def save(self, path):
        self.modelCNNDa.save(path)

    def load(self, path):
        self.modelCNNDa = tf.keras.models.load_model(path)


# predictions = self(x_test)
# probabilities = tf.nn.softmax(predictions)  # Aplica la función softmax a las predicciones

# # Convierte las probabilidades en un vector de longitud 59
# probabilities_vector = np.zeros(59)
# for i, prob in enumerate(probabilities[0]):
#     probabilities_vector[i] = prob

# return probabilities_vector


 
