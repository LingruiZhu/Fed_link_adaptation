import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

import sys 
sys.path.append("/home/zhu/Codes/Fed_Link_Adaptation")

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Interference_prediction import data_preprocessing


def create_autoencoder(inputs_dims:int, latent_dims:int, optimizer:str="adam"):
    inputs = Input(shape=(inputs_dims,))
    hidden1 = Dense(units=int(inputs_dims/2), activation="relu")(inputs)
    encoded = Dense(latent_dims, activation="relu")(hidden1)

    # Define the decoder model
    hidden2 = Dense(int(inputs_dims/2), activation='relu')(encoded)
    decoded = Dense(inputs_dims, activation="linear")(hidden2)

    # Create the autoencoder model
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=optimizer, loss="mse")
    autoencoder.summary()
    return autoencoder
    

def train_autoencoder_input_compression(inputs_dims:int, latent_dims:int, optimizer:str="adam", plot_figure:bool=True):
    autoencoder = create_autoencoder(inputs_dims, latent_dims, optimizer)
    x_train, _, x_test, _, _ = data_preprocessing.prepare_data(num_inputs=40, num_outputs=10)
    
    x_train = np.squeeze(x_train)
    x_test = np.squeeze(x_test)
        
    history = autoencoder.fit(x_train, x_train, epochs=500, batch_size=64)
    file_name = f"vq_vae_ema_input_{inputs_dims}_latent_{latent_dims}_optimizer_{optimizer}.h5"
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
    train_autoencoder_input_compression(inputs_dims=40, latent_dims=20, optimizer="RMSprop")