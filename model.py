import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Attention, Dense, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback

# Definición de la arquitectura de la red
def build_model(vocab_size, embedding_dim, hidden_units):
    # Capa de entrada
    inputs = Input(shape=(None,))
    embedded = Embedding(vocab_size, embedding_dim)(inputs)
    
    # Capa recurrente
    lstm_output = LSTM(hidden_units, return_sequences=True)(embedded)
    
    # Capa de atención
    attention = Attention()([lstm_output, lstm_output])
    
    # Capa de decodificación
    decoded = Dense(vocab_size, activation='softmax')(attention)
    
    # Construcción del modelo
    model = tf.keras.Model(inputs=inputs, outputs=decoded)
    return model


class MiRedNeuronal:
    def __init__(self, num_frames, num_parametros, num_caracteres):
        self.num_frames = num_frames
        self.num_parametros = num_parametros
        self.num_caracteres = num_caracteres
        self.model = self._construir_modelo()

    def _construir_modelo(self):
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(self.num_frames, self.num_parametros)))
        model.add(LSTM(units=64, return_sequences=True))
        model.add(GlobalMaxPooling1D())  # Capa de pooling para reducir la secuencia a un solo valor
        model.add(Dense(31))  # Capa Dense con salida de tamaño 1
        #model.add(Dense(self.num_caracteres, activation='softmax'))
        #model.add(Dense(self.num_caracteres))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

    def resumen_modelo(self):
        self.model.summary()

    def entrenar(self, x_train, y_train, epochs, batch_size):
        # Definir el callback personalizado para seguir el progreso del entrenamiento
        class ProgresoEntrenamientoCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                # Imprimir métricas o información adicional después de cada época
                print(f"Época {epoch+1}/{epochs} - Pérdida: {logs['loss']}")

        # Crear instancia del callback personalizado
        progreso_callback = ProgresoEntrenamientoCallback()

        # Entrenar el modelo con el callback personalizado
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[progreso_callback])

    def evaluar(self, x_test, y_test):
        loss = self.model.evaluate(x_test, y_test)
        print(f"Pérdida en el conjunto de prueba: {loss}")
