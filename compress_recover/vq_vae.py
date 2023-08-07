import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import h5py

import sys 
sys.path.append("/home/zhu/Codes/Fed_Link_Adaptation")

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Activation, Dense, BatchNormalization
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import mean_squared_error

from Interference_prediction import data_preprocessing

import pdb


class VectorQuantizer(Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )
        self.embedding_sample_count = tf.Variable(
            initial_value=tf.zeros(shape=(self.num_embeddings,), dtype="float32"),
            trainable=False,
            name="embedding_sampole_count")
        self.embedding_sample_accumulative_count = tf.Variable(
            initial_value=tf.zeros(shape=(self.num_embeddings,), dtype="float32"),
            trainable=False,
            name="embedding_sample_accumulative_count")


    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = self.beta * tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)   # need to tune with beta
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        # self.add_loss(self.beta * commitment_loss + codebook_loss)
        # separate the loss in to two parts and add them
        self.add_loss(codebook_loss)
        self.add_loss(commitment_loss)
        
        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized
    
    
    def track_embedding_space(self, x):
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        
        # Count how many samples will be assigned to which embeddings.
        count = tf.reduce_sum(encodings, 0)
        self.embedding_sample_count.assign(count)
        self.embedding_sample_accumulative_count.assign(self.embedding_sample_accumulative_count + count)
         

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs**2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings**2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices
    
        
    def get_config(self):
        config = super(VectorQuantizer, self).get_config()
        config.update({
            'num_embeddings': self.num_embeddings,
            'embedding_dim': self.embedding_dim,
            'embedding_sample_count': self.embedding_sample_count.numpy().tolist(),
            'embedding_sample_accumulative_count': self.embedding_sample_accumulative_count.numpy().tolist(),
        })
        return config 
    
    


# def calculate_vae_loss(encoder_output, quantized_latent_variable, variance, beta):
#     """Define the loss function of VAE based on the equation (3) from "Neural Discrete Representation Learning".

#     Args:
#         encoder_output (tf.Tensor): The output of the encoder.
#         quantized_latent_variable (tf.Tensor): The quantized latent variable.
#         variance (float): The variance of the encoder output.
#         beta (float): The weight of the commitment loss in the total loss.
        
#     Returns:
#         The VQ-VAE loss function.
#     """
#     def vq_vae_loss(x, x_hat):
#         reconstruction_loss = losses.mean_squared_error(x, x_hat) / variance
#         quantizer_loss = losses.mean_squared_error(tf.stop_gradient(encoder_output), quantized_latent_variable)
#         commitment_loss = losses.mean_squared_error(encoder_output, tf.stop_gradient((quantized_latent_variable)))
#         loss = reconstruction_loss + quantizer_loss + beta*commitment_loss
#         return loss
#     return vq_vae_loss


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
    
    
def create_quantized_autoencoder(input_dim, latent_dim, output_dim, num_embeddings:int=128, with_batch_normalization:bool=False):
    encoder = create_encoder(input_dim, latent_dim)
    decoder = create_decoder(latent_dim, output_dim)
    quantizer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=latent_dim)
    batch_norm_layer = BatchNormalization()
    
    encoder.summary()
    decoder.summary()
    
    inputs = Input(shape=(input_dim,))
    encoder_outputs = encoder(inputs)
    encoder_outputs_quantized = quantizer(encoder_outputs)
    if with_batch_normalization:
        encoder_outputs = batch_norm_layer(encoder_outputs)
    
    decoder_output = decoder(encoder_outputs_quantized)
    
    vector_quant_autoencoder = Model(inputs=inputs, outputs=decoder_output, name="vector_quantized_autoencoder")
    return vector_quant_autoencoder


class VQVAETrainer(Model):
    def __init__(self, train_variance, input_dim, latent_dim=10, num_embeddings=128, with_bn_layer:bool=False, **kwargs):
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.num_embeddings = num_embeddings

        self.vqvae = create_quantized_autoencoder(self.input_dim, self.latent_dim, self.input_dim, self.num_embeddings, with_bn_layer)
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


    def get_latent_vector(self, x):
        x1 = self.vqvae.layers[0](x)
        latent_vec = self.vqvae.layers[1](x1)
        return latent_vec
    

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            # total_loss = reconstruction_loss + sum(self.vqvae.losses)
            codebook_loss = self.vqvae.losses[0]
            commitment_loss = self.vqvae.losses[1]
            total_loss = reconstruction_loss + codebook_loss + commitment_loss
            
        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))
        
        # track latent variable and embedding space
        latent_var = self.get_latent_vector(x)
        self.vqvae.layers[2].track_embedding_space(latent_var)

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        # self.vq_loss_tracker.update_state(sum(self.vqvae.losses))
        self.vq_codebook_loss_tracker.update_state(codebook_loss)
        self.vq_commitment_loss_tracker.update_state(commitment_loss)
        
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
        self.vqvae.layers[2].embedding_sample_count.trainable = True
        self.vqvae.layers[2].embedding_sample_accumulative_count.trainable = True
        self.vqvae.save_weights(file_path)



class CountActivdeEmbeddings(keras.callbacks.Callback):
    def __init__(self, vqvae):
        super().__init__()
        self.vqvae = vqvae
        self.num_active_embeddings_list = []


    def on_epoch_end(self, batch, logs=None):
        num_active_embeddings = tf.math.count_nonzero(self.vqvae.layers[2].embedding_sample_accumulative_count)
        self.num_active_embeddings_list.append(tf.keras.backend.eval(num_active_embeddings))
        
        
    
def train_vq_vae(inputs_dims:int, latent_dims:int, num_embeddings, with_batch_norm:bool=False, plot_figure:bool=True, optimizer:str="adam"):
    x_train, _, x_test, _, _ = data_preprocessing.prepare_data(num_inputs=40, num_outputs=10)
    x_train = np.squeeze(x_train)
    x_test = np.squeeze(x_test)
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    
    variance = np.var(x_train)
    
    vq_vae_trainer = VQVAETrainer(variance, inputs_dims, latent_dims, num_embeddings=num_embeddings, with_bn_layer=with_batch_norm)
    vq_vae_trainer.compile(optimizer=optimizer)
    
    vq_vae_trainer.build((None, inputs_dims))
    
    # Custom callback to track the learning rate
    learning_rates = list()
    class LearningRateTracker(tf.keras.callbacks.Callback):
        def on_epoch_end(self, batch, logs=None):
            # Get the current learning rate from the optimizer
            current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
            # Append the learning rate to the list
            learning_rates.append(current_lr)
    
    # Define callback to track the embedding space
    active_embedding_tracker = CountActivdeEmbeddings(vq_vae_trainer.vqvae)
    
    early_stopping = EarlyStopping(monitor="val_total_loss", patience=20, mode="min")
    history = vq_vae_trainer.fit(x=x_train, validation_split=0.2, epochs=500, batch_size=64, callbacks=[LearningRateTracker(), active_embedding_tracker])
    num_active_embeddings_list = active_embedding_tracker.num_active_embeddings_list
    
    # save training history and weights
    file_name = f"vq_vae_input_{inputs_dims}_latent_{latent_dims}_num_embeddings_{num_embeddings}_with_BN_{with_batch_norm}_{optimizer}.h5"
    history_path = os.path.join("training_history", "vq_vae", file_name)
    with h5py.File(history_path, "w") as hf:
        for key, value in history.history.items():
            hf.create_dataset(key, data=value)
        hf.create_dataset("learning_rates", data=learning_rates)
        hf.create_dataset("num_active_embeddings", data=num_active_embeddings_list)
    weights_path = os.path.join("models", "vq_vae_models_num_embeddings_compare", file_name)
    vq_vae_trainer.save_model_weights(weights_path)
    print("Model weights have been saved to the path: " + weights_path)
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
    
    
def test_vq_vae(inputs_dims:int, latent_dims:int, num_embeddings, plot_figure:bool=True):
    vq_vae = create_quantized_autoencoder(inputs_dims, latent_dims, inputs_dims)
    vq_vae.load_weights("models/vq_vae_models/vq_vae_input_40_latent_10_num_embeddings_128.h5")
    
    _, _, x_test, _, _ = data_preprocessing.prepare_data(num_inputs=40, num_outputs=10)
    x_test = np.squeeze(x_test)
    x_test_recover = vq_vae.predict(x_test)
    mse = mean_squared_error(x_test, x_test_recover)
    
    if plot_figure:
        x_test_recover_1d = x_test_recover[:10,:].flatten()
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
    train_vq_vae(inputs_dims=40, latent_dims=20, num_embeddings=256, with_batch_norm=False, plot_figure=True, optimizer="RMSprop")


    