
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

import sys 
sys.path.append("/home/zhu/Codes/Fed_Link_Adaptation")

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Reshape, Lambda
from tensorflow.keras.models import Model

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Interference_prediction import data_preprocessing


def create_dense_encoder(inputs_dims:int, latent_dims:int):
    inputs = Input(shape=(inputs_dims,))
    hidden1 = Dense(units=inputs_dims, activation="relu")(inputs)
    hidden2 = Dense(units=int(inputs_dims/2), activation="relu")(hidden1)
    encoded = Dense(latent_dims, activation="relu")(hidden2)
    dense_encoder_model = Model(inputs, encoded, name="dense_encoder")
    return dense_encoder_model


def create_dense_decoder(inputs_dim:int, latent_dims:int):
    inputs = Input(shape=(latent_dims,))
    hidden1 = Dense(units=latent_dims, activation="relu")(inputs)
    hidden2 = Dense(units=(inputs_dims/2), activation="relu")(hidden1)
    decoded = Dense(units=inputs_dim, activation="linear")(hidden2)
    dense_decoder_model = Model(inputs, decoded, name="dense_decoder")
    return dense_decoder_model
    
    
def create_dense_autoencoder(inputs_dims:int, latent_dims:int, optimizer:str="adam"):
    # Define the encoder and decoder 
    dense_encoder = create_dense_encoder(inputs_dims, latent_dims)
    dense_decoder = create_dense_decoder(inputs_dims, latent_dims)
    
    inputs = Input(shape=(inputs_dims,))
    encoded = dense_encoder(inputs)
    decoded = dense_decoder(encoded)

    # Create the autoencoder model
    autoencoder = Model(inputs, decoded, name="dense_autoencoder")
    autoencoder.compile(optimizer=optimizer, loss="mse")
    autoencoder.summary()
    return autoencoder


def create_lstm_encoder(inputs_dims:int, latent_dims:int, num_features:int=1):
    inputs = Input(shape=(inputs_dims, num_features))
    hidden1 = LSTM(inputs_dims, activation="relu", return_sequences=True)(inputs)
    hidden2 = LSTM(int(inputs_dims/2), activation="relu", return_sequences=True)(hidden1)
    encoded = LSTM(latent_dims, activation="relu", return_sequences=False)(hidden2)
    lstm_encoder_model = Model(inputs, encoded, name="lstm_encoder")
    return lstm_encoder_model


def create_lstm_decoder(inputs_dims:int, latent_dims:int, num_features:int=1):
    inputs = Input(shape=(latent_dims, num_features))
    hidden1 = LSTM(latent_dims, activation="relu", return_sequences=True)(inputs)
    hidden2 = LSTM(int(inputs_dims/2), activation="relu", return_sequences=False)(hidden1)
    decoded = Dense(inputs_dims, activation="linear")(hidden2)
    lstm_decoder_model = Model(inputs, decoded, name="lstm_decoder")
    return lstm_decoder_model


def create_lstm_autoencoder(inputs_dims:int, latent_dims, optimizer:str="adam", num_features:int=1):
    lstm_encoder = create_lstm_encoder(inputs_dims, latent_dims, num_features)
    lstm_decoder = create_lstm_decoder(inputs_dims, latent_dims, num_features)
    
    
    inputs = Input(shape=(inputs_dims, num_features))
    encoded = lstm_encoder(inputs)
    encoded_expand_dim = Lambda(lambda x: tf.expand_dims(x, axis=-1))(encoded)
    decoded = lstm_decoder(encoded_expand_dim)

    # Create the autoencoder model
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=optimizer, loss="mse")
    autoencoder.summary()
    return autoencoder
    

def train_autoencoder_input_compression(inputs_dims:int, latent_dims:int, optimizer:str="adam", type:str="dense", plot_figure:bool=True):
    if type == "dense":
        autoencoder = create_dense_autoencoder(inputs_dims, latent_dims, optimizer)
    elif type == "lstm":
        autoencoder = create_lstm_autoencoder(inputs_dims, latent_dims, optimizer)
    x_train, _, x_test, _, _ = data_preprocessing.prepare_data(num_inputs=40, num_outputs=10)
    
    if type == "dense":
        x_train = np.squeeze(x_train)
    
    x_test = np.squeeze(x_test)
        
    history = autoencoder.fit(x_train, x_train, epochs=300, batch_size=64)
    file_name = f"AE_{type}_input_{inputs_dims}_latent_{latent_dims}_optimizer_{optimizer}.h5"
    history_save_path = os.path.join("training_history", "ae", file_name)
    with h5py.File(history_save_path, "w") as hf:
        for key, value in history.history.items():
            hf.create_dataset(key, data=value)
    
    recover_test = autoencoder.predict(x_test)
    mse_rescale = mean_squared_error(recover_test, x_test)
    model_save_path = os.path.join("models", "ae_models", file_name)
    autoencoder.save(model_save_path)
    print(f"the MSE after rescale is {mse_rescale}")
    
    if plot_figure:
        x_test_recover_1d = recover_test[:10,:].flatten()
        x_test_true_1d = x_test[:10,:].flatten()
        
        plt.figure()
        plt.plot(x_test_recover_1d, "r-x", label="recoverd_signal")
        plt.plot(x_test_true_1d, "b-s", label="true signal")
        plt.grid()
        plt.legend()
        plt.show()
        plt.xlabel("time steps")
        plt.ylabel("SINR")


if __name__ == "__main__":
    inputs_dims = 40
    latent_dims = 20
    train_autoencoder_input_compression(inputs_dims=40, latent_dims=20, type="dense", optimizer="RMSprop")
    train_autoencoder_input_compression(inputs_dims=40, latent_dims=20, type="lstm", optimizer="RMSprop")