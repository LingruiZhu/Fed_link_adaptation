import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from Interference_prediction.lstm_model import build_lstm_predict_model
from auto_encoder import create_autoencoder
from Interference_prediction.data_preprocessing import prepare_data


def create_AE_LSTM_model(input_dim, output_dim, latent_dim):
    autoencoder = create_autoencoder(input_dim, latent_dim)
    lstm = build_lstm_predict_model(input_dim, 128, output_dim)
    autoencoder_input = Input(input_dim, )
    autoencoder_output = autoencoder(autoencoder_input)
    autoencoder_output = tf.expand_dims(autoencoder_output, axis=-1)
    lstm_prediction = lstm(autoencoder_output)
    autoencoder_lstm_model = Model(inputs=autoencoder_input, outputs=lstm_prediction)
    return autoencoder_lstm_model


def train_e2e_AE_LSTM_model():
    e2e_model = create_AE_LSTM_model(input_dim=40, output_dim=10, latent_dim=10)
    e2e_model.compile(optimizer="adam", loss="mean_squared_error")
    e2e_model.summary()
    x_train, y_train, x_test, y_test, _ = prepare_data(num_inputs=40, num_outputs=10)
    x_train = np.squeeze(x_train)
    e2e_model.fit(x_train, y_train, batch_size=128, epochs=500)
    e2e_model.save("models/feedback_models/e2e_models/e2e_ae_lstm.h5")


def test_e2e_AE_LSTM_model():
    pass


if __name__ == "__main__":
    train_e2e_AE_LSTM_model()
    
