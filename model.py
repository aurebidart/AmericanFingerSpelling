import tensorflow as tf

class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Input layer
        self.input_layer = tf.keras.layers.Input(shape=(126,), dtype=tf.float32)

        # Two convolutional layers
        self.conv1 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same')(self.input_layer)
        self.conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(self.conv1)

        # Dropout layer
        self.dropout = tf.keras.layers.Dropout(0.2)(self.conv2)

        # Output layer
        self.output_layer = tf.keras.layers.Dense(59, activation='softmax')(self.dropout)

    def call(self, inputs):
        return self.output_layer(inputs)

    def train(self, x_train, y_train, epochs=10, batch_size=128):
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam()

        for epoch in range(epochs):
            for batch_x, batch_y in tf.keras.utils.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size):
                loss = loss_fn(self(batch_x), batch_y)
                optimizer.minimize(loss)

    def predict(self, x_test):
        return tf.argmax(self(x_test), axis=1)

    def save(self, path):
        tf.keras.models.save_model(self, path)

    def load(self, path):
        self = tf.keras.models.load_model(path)
