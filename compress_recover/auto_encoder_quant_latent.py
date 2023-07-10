import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import h5py

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Activation, Dense, BatchNormalization
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import losses_utils

from sklearn.metrics import mean_squared_error

from Interference_prediction import data_preprocessing

import pdb



class QuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, num_bits, **kwargs):
        super(QuantizationLayer, self).__init__(**kwargs)
        self.num_bits = num_bits

    def build(self, input_shape):
        super(QuantizationLayer, self).build(input_shape)

    def call(self, inputs):
        # Calculate the range of the variable
        value_range = tf.reduce_max(inputs) - tf.reduce_min(inputs)

        # Calculate the step size between quantization levels
        step_size = value_range / (2 ** self.num_bits)

        # Quantize the inputs
        quantized_output = tf.round(inputs / step_size) * step_size

        return quantized_output


def create_encoder(input_dim, latent_dim):
    inputs = Input(shape=(input_dim,))
    hidden1 = Dense(units=int(input_dim/2), activation="relu")(inputs)
    encoder_output = Dense(units=latent_dim, activation="relu")(hidden1)
    encoder = Model(inputs, encoder_output, name="encoder")
    return encoder


def create_decoder(latent_dim, output_dim):
    decoder_inputs= Input(shape=(latent_dim,))
    hidden1 = Dense(units=(output_dim/2), activation="relu")(decoder_inputs)
    decoder_outputs = Dense(units=output_dim, activation="linear")(hidden1)
    decoder = Model(decoder_inputs, decoder_outputs, name="decoder")
    return decoder
    
    
def create_uniform_quantized_autoencoder(input_dim, latent_dim, output_dim, number_quant_bits):
    encoder = create_encoder(input_dim, latent_dim)
    decoder = create_decoder(latent_dim, output_dim)
    uniform_quantizer = QuantizationLayer(num_bits=number_quant_bits)
    
    encoder.summary()
    decoder.summary()
    
    inputs = Input(shape=(input_dim,))
    encoder_outputs = encoder(inputs)
    encoder_outputs_quantized = uniform_quantizer(encoder_outputs)
    
    decoder_output = decoder(encoder_outputs_quantized)
    
    vector_quant_autoencoder = Model(inputs=inputs, outputs=decoder_output, name="vector_quantized_autoencoder")
    return vector_quant_autoencoder


class VQVAETrainer(Model):
    def __init__(self, train_variance, input_dim, latent_dim=10, num_quant_bits=4, **kwargs):
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.number_quant_bits = num_quant_bits

        self.vqvae = create_uniform_quantized_autoencoder(self.input_dim, self.latent_dim, self.input_dim, self.number_quant_bits)
        self.vqvae.summary()

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            # total_loss = reconstruction_loss + sum(self.vqvae.losses)
            total_loss = reconstruction_loss
            
        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        
        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        }
    
    
    def call(self, x):
        return self.vqvae(x)
    
    
    def save_model_weights(self, file_path):
        self.vqvae.save_weights(file_path)


def train_vq_vae(inputs_dims:int, latent_dims:int, number_quant_bits:int, plot_figure:bool=True):
    x_train, _, x_test, _, _ = data_preprocessing.prepare_data(num_inputs=40, num_outputs=10)
    x_train = np.squeeze(x_train)
    x_test = np.squeeze(x_test)
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    
    variance = np.var(x_train)
    
    vq_vae_trainer = VQVAETrainer(variance, inputs_dims, latent_dims, number_quant_bits)
    vq_vae_trainer.compile(optimizer="adam")
    
    vq_vae_trainer.build((None, inputs_dims))
    
    x_train_hat = vq_vae_trainer.predict(x_train)
        
    history = vq_vae_trainer.fit(x=x_train, epochs=1000, batch_size=64)
    
    # save training history and weights
    file_name = f"vq_vae_input_{inputs_dims}_latent_{latent_dims}_num_quant_bits_{number_quant_bits}.h5"
    history_path = os.path.join("training_history", "vq_vae_uniform_quant", file_name)
    with h5py.File(history_path, "w") as hf:
        for key, value in history.history.items():
            hf.create_dataset(key, data=value)
    weights_path = os.path.join("models", "vq_vae_uniform_quant", file_name)
    vq_vae_trainer.save_model_weights(weights_path)
    x_test_pred = vq_vae_trainer.predict(x_test)
    mse = mean_squared_error(x_test, x_test_pred)
    
    if plot_figure:
        x_test_recover_1d = x_test_pred[:10,:].flatten()
        x_test_true_1d = x_test[:10,:].flatten()
        
        plt.figure()
        plt.plot(x_test_recover_1d, "r-x", label="recoverd_signal")
        plt.plot(x_test_true_1d, "b-s", label="true signal")
        plt.grid()
        plt.legend()
        plt.show()
        plt.xlabel("time steps")
        plt.ylabel("SINR")
    return mse
    

if __name__ == "__main__":
    train_vq_vae(inputs_dims=40, latent_dims=10, number_quant_bits=4, plot_figure=True)



    

    