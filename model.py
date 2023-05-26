import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Attention, Dense

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