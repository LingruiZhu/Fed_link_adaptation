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
from sonnet.nets import VectorQuantizerEMA

from tensorflow.python.training import moving_averages


from sklearn.metrics import mean_squared_error

from Interference_prediction import data_preprocessing



class VectorQuantizer_EMA(Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=1, ema_decay=0.85, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta
        self.ema_decay = ema_decay
        
        self.is_training_ema = True

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(shape=(self.embedding_dim, self.num_embeddings), dtype="float32"),
            trainable=True,
            name="embeddings_vqvae")
        self.ema_count = tf.Variable(
            initial_value=tf.zeros(shape=(self.num_embeddings,), dtype="float32"),
            trainable=False,
            name="ema_count_vqvae")
        self.embeddings_sum = tf.Variable(
            initial_value=tf.zeros(shape=(self.embedding_dim, self.num_embeddings), dtype="float32"),
            trainable=False,
            name="embeddings_sum_vqvae")

    
    def enable_training_ema(self):
        self.is_training_ema = True
    
    
    def disable_training_ema(self):
        self.is_training_ema = False
        

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
        
        if self.is_training_ema:
            self.update_ema_embeddings(x)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss)      # here, codebook loss will be excluded since the 

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized


    def update_ema_embeddings(self, inputs):
        flattened_inputs = tf.reshape(inputs, [-1, self.embedding_dim])

        # Calculate the encoding indices based on the flattened inputs
        encoding_indices = self.get_code_indices(flattened_inputs)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        
        # Calculate the count of each codebook vector based on the encoding indices
        count = tf.reduce_sum(encodings, 0)

        # Update the EMA count using the decay factor
        self.ema_count.assign((1.0 - self.ema_decay) * self.ema_count + self.ema_decay * count)

        # Calculate the EMA of the codebook embeddings
        embeddings_sum = tf.matmul(flattened_inputs, encodings, transpose_a=True)
        updated_embeddings_sum = (1.0 - self.ema_decay) * self.embeddings_sum + self.ema_decay * embeddings_sum

        # Normalize the updated codebook embeddings using the count
        normalized_embeddings = updated_embeddings_sum / tf.maximum(self.ema_count, 1e-5)

        # Assign the normalized embeddings to the codebook
        self.embeddings.assign(normalized_embeddings)
        self.embeddings_sum.assign(updated_embeddings_sum)
        
        # print(inputs.shape)
        # print(flattened_inputs.shape)
        # print(encoding_indices.shape)
        # print(encodings.shape)
        # print(count.shape)
        # print(embeddings_sum.shape)
        
        # tf.print(count)
        # tf.print(self.ema_count)
        
        # pdb.set_trace()   # for debugging



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
        config = super(VectorQuantizer_EMA, self).get_config()
        config.update({
            'num_embeddings': self.num_embeddings,
            'embedding_dim': self.embedding_dim
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
    
    
def create_quantized_autoencoder_EMA(input_dim, latent_dim, output_dim, num_embeddings:int=128, ema_decay:float=0.99, commitment_factor:float=0.25):
    encoder = create_encoder(input_dim, latent_dim)
    decoder = create_decoder(latent_dim, output_dim)
    quantizer = VectorQuantizer_EMA(num_embeddings=num_embeddings, embedding_dim=latent_dim, ema_decay=ema_decay, beta=commitment_factor)
    bn_layer = BatchNormalization()
    
    quantizer.enable_training_ema()
    
    encoder.summary()
    decoder.summary()
    
    inputs = Input(shape=(input_dim,))
    encoder_outputs = encoder(inputs)
    
    encoder_outputs_quantized = quantizer(encoder_outputs)
    
    decoder_output = decoder(encoder_outputs_quantized)
    
    vector_quant_autoencoder = Model(inputs=inputs, outputs=decoder_output, name="vector_quantized_autoencoder")
    return vector_quant_autoencoder


class VQVAETrainer(Model):
    def __init__(self, train_variance, input_dim, latent_dim=10, num_embeddings=1, ema_decay:float=0.99, commitment_factor=0.25, **kwargs):
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.num_embeddings = num_embeddings

        self.vqvae = create_quantized_autoencoder_EMA(self.input_dim, self.latent_dim, self.input_dim, self.num_embeddings, ema_decay, commitment_factor=commitment_factor)
        self.vqvae.summary()

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }
    
    
    def call(self, x, is_ema_updating:bool=True):
        return self.vqvae(x, is_ema_updating)
    
    
    def save_model_weights(self, file_path):
        self.vqvae.save_weights(file_path)


def train_vq_vae(inputs_dims:int, latent_dims:int, num_embeddings:int, commitment_factor:float, ema_decay:float, plot_figure:bool=True):
    x_train, _, x_test, _, _ = data_preprocessing.prepare_data(num_inputs=40, num_outputs=10)
    x_train = np.squeeze(x_train)
    x_test = np.squeeze(x_test)
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    
    variance = np.var(x_train)
    
    vq_vae_trainer = VQVAETrainer(variance, inputs_dims, latent_dims, num_embeddings=num_embeddings, ema_decay=ema_decay, commitment_factor=commitment_factor)
    vq_vae_trainer.compile(optimizer="adam")
    
    vq_vae_trainer.build((None, inputs_dims))
    
    x_train_hat = vq_vae_trainer.predict(x_train)
        
    history = vq_vae_trainer.fit(x=x_train, epochs=1000, batch_size=64)
    file_name = f"vq_vae_ema_input_{inputs_dims}_latent_{latent_dims}_num_embeddings_{num_embeddings}_ema_decay_{ema_decay}_beta_{commitment_factor}.h5"
    weights_path = os.path.join("models", "vq_vae_ema_models", file_name)
    vq_vae_trainer.save_model_weights(weights_path)
    history_path = os.path.join("training_history", "vq_vae_ema", file_name)
    with h5py.File(history_path, "w") as hf:
        for key, value in history.history.items():
            hf.create_dataset(key, data=value)
    
    # disable ema updates
    vq_ema_layer = vq_vae_trainer.vqvae.get_layer("vector_quantizer_ema")
    vq_ema_layer.disable_training_ema()
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
    # load model from file
    vq_vae_ema = create_quantized_autoencoder_EMA(inputs_dims, latent_dims, inputs_dims, num_embeddings)
    vq_vae_ema.load_weights("models/vq_vae_ema_models/vq_vae_ema_input_40_latent_10_num_embeddings_128.h5")
    vq_ema_layer = vq_vae_ema.get_layer("vector_quantizer_ema")
    vq_ema_layer.disable_training_ema()
    
    # prepare dat
    _, _, x_test, _, _ = data_preprocessing.prepare_data(num_inputs=40, num_outputs=10)
    x_test_recover = vq_vae_ema.predict(x_test)
    
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


if __name__ == "__main__":

    # test_vq_vae(inputs_dims=40, latent_dims=10, num_embeddings=128)
    
    ema_decay = 0.99
    beta = 0.25
    train_vq_vae(inputs_dims=40, latent_dims=20, num_embeddings=256, commitment_factor=beta, plot_figure=True, ema_decay=ema_decay)

    