from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def build_vae(input_dim, latent_dim):
    # Encoder
    encoder_inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(encoder_inputs)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    # Reparameterization trick
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = keras.backend.random_normal(shape=(keras.backend.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(64, activation="relu")(decoder_inputs)
    decoder_outputs = layers.Dense(input_dim, activation="sigmoid")(x)

    # Define the VAE model
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")

    # Connect the encoder and decoder models
    encoder_outputs = encoder(encoder_inputs)[2]
    vae_outputs = decoder(encoder_outputs)

    # Define the VAE model with the complete graph
    vae = keras.Model(encoder_inputs, vae_outputs, name="vae")

    # Define the VAE loss function
    reconstruction_loss = keras.losses.mean_squared_error(encoder_inputs, vae_outputs)
    reconstruction_loss *= input_dim  # Adjust the scaling based on the input shape
    kl_loss = 1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(z_log_var)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    # Compile the VAE model
    vae.compile(optimizer="adam")
    vae.summary()
    return vae


if __name__ == "__main__":
    input_dim = 40
    latent_dim = 10
    vae = build_vae(input_dim, latent_dim)
