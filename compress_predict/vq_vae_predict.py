import sys 
sys.path.append("/Users/lingrui/Codes/Fed_link_adaptation")
import os
import matplotlib.pyplot as plt

import h5py

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error


from compress_recover.vq_vae import create_quantized_autoencoder
from Interference_prediction.lstm_model import build_lstm_predict_model
from Interference_prediction import data_preprocessing


def create_VQ_VAE_predict(input_dim:int, latent_dim:int, output_dim:int, num_embedding:int):
    vq_vae_model = create_quantized_autoencoder(input_dim, latent_dim, input_dim, num_embedding)
    lstm_model = build_lstm_predict_model(num_inputs=input_dim, num_hidden=input_dim, num_outputs=output_dim)

    input_signal = Input(shape=(input_dim,))
    recovered_signal = vq_vae_model(input_signal)
    predicted_signal = lstm_model(recovered_signal)
    vq_vae_predict_model = Model(inputs=input_signal, outputs=predicted_signal, name="vq_vae_predict_model")
    return vq_vae_predict_model


class VQVAE_Pred_Trainer(Model):
    def __init__(self, train_variance, input_dim, output_dim, latent_dim=10, num_embeddings=128, with_bn_layer:bool=False, **kwargs):
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_embeddings = num_embeddings

        self.vqvae = create_VQ_VAE_predict(self.input_dim, self.latent_dim, self.output_dim, self.num_embeddings)
        self.vqvae.summary()

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        
        # self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")
        self.vq_codebook_loss_tracker = keras.metrics.Mean(name="vq_codebook_loss")
        self.vq_commitment_loss_tracker = keras.metrics.Mean(name="vq_commitment_loss")
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_codebook_loss_tracker,
            self.vq_commitment_loss_tracker
        ]

    def train_step(self, data):
        x,y = data
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((y - reconstructions) ** 2) / self.train_variance
            )
            # total_loss = reconstruction_loss + sum(self.vqvae.losses)
            codebook_loss = self.vqvae.losses[0]
            commitment_loss = self.vqvae.losses[1]
            total_loss = reconstruction_loss + codebook_loss + commitment_loss
            
        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        # self.vq_loss_tracker.update_state(sum(self.vqvae.losses))
        self.vq_codebook_loss_tracker.update_state(self.vqvae.losses[0])
        self.vq_commitment_loss_tracker.update_state(self.vqvae.losses[1])
        
        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "codebook_loss": self.vq_codebook_loss_tracker.result(),
            "commitment_loss": self.vq_codebook_loss_tracker.result()
        }
    
    
    def call(self, x):
        return self.vqvae(x)
    
    
    def save_model_weights(self, file_path):
        self.vqvae.save_weights(file_path)


def train_vq_vae_predict(input_dim:int, latent_dim:int, output_dim:int, num_embeddings:int, with_batch_norm:bool=False, plot_figure:bool=True):
    x_train, y_train, x_test, y_test, _ = data_preprocessing.prepare_data(num_inputs=40, num_outputs=10)
    x_train = np.squeeze(x_train)
    y_train = np.squeeze(y_train)
    x_test = np.squeeze(x_test)
    y_test = np.squeeze(y_test)
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    
    variance = np.var(x_train)
    
    vq_vae_trainer = VQVAE_Pred_Trainer(variance, input_dim, latent_dim, num_embeddings=num_embeddings, with_bn_layer=with_batch_norm)
    vq_vae_trainer.compile(optimizer="adam")
    
    vq_vae_trainer.build((None, input_dim))
            
    history = vq_vae_trainer.fit(x=x_train, y=y_train, epochs=200, batch_size=64)
    
    file_name = f"vq_vae_input_{input_dim}_latent_{latent_dim}_num_embeddings_{num_embeddings}_with_BN_{with_batch_norm}.h5"
    history_path = os.path.join("training_history", "vq_vae_predict", file_name)
    with h5py.File(history_path, "w") as hf:
        for key, value in history.history.items():
            hf.create_dataset(key, data=value)
    weights_path = os.path.join("models", "vq_vae_prediction", file_name)
    vq_vae_trainer.save_model_weights(weights_path)
    y_test_pred = vq_vae_trainer.predict(x_test)
    mse = mean_squared_error(y_test, y_test_pred)
    
    if plot_figure:
        y_test_recover_1d = y_test_pred[:10,:].flatten()
        y_test_true_1d = x_test[:10,:].flatten()
        
        plt.figure()
        plt.plot(y_test_recover_1d, "r-x", label="recoverd_signal")
        plt.plot(y_test_true_1d, "b-s", label="true signal")
        plt.grid()
        plt.legend()
        plt.show()
        plt.xlabel("time steps")
        plt.ylabel("SINR")
    return mse



if __name__ == "__main__":
    train_vq_vae_predict(input_dim=40, latent_dim=10, output_dim=10, num_embeddings=128)






