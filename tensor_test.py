import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback

class TrackWeightsCallback(Callback):
    def __init__(self, model):
        super(TrackWeightsCallback, self).__init__()
        self.model = model
        self.weights_history = []

    def on_epoch_end(self, epoch, logs=None):
        # Track the weights of the model at the end of each epoch
        weights = self.model.get_weights()
        self.weights_history.append(weights)

# Create a simple sequential model
model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Load data (mnist dataset) and preprocess it
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape the input data to have shape (784,)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Create an instance of the custom callback
track_weights_callback = TrackWeightsCallback(model)

# Train the model with the custom callback
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1, callbacks=[track_weights_callback])
print(len(track_weights_callback.weights_history))