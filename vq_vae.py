import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Activation, Dense, Flatten, Dropout, Lambda, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, SpatialDropout2D
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import losses_utils
from sonnet.nets import VectorQuantizerEMA


from sklearn.metrics import mean_squared_error

from Interference_prediction import data_preprocessing


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
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized


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
    
    
def create_quantized_autoencoder(input_dim, latent_dim, output_dim, num_embeddings:int=128):
    encoder = create_encoder(input_dim, latent_dim)
    decoder = create_decoder(latent_dim, output_dim)
    quantizer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=latent_dim)
    
    encoder.summary()
    decoder.summary()
    
    inputs = Input(shape=(input_dim,))
    encoder_outputs = encoder(inputs)
    encoder_outputs_quantized = quantizer(encoder_outputs)
    
    decoder_output = decoder(encoder_outputs_quantized)
    
    vector_quant_autoencoder = Model(inputs=inputs, outputs=decoder_output, name="vector_quantized_autoencoder")
    return vector_quant_autoencoder


class VQVAETrainer(Model):
    def __init__(self, train_variance, input_dim, latent_dim=10, num_embeddings=1, **kwargs):
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.num_embeddings = num_embeddings

        self.vqvae = create_quantized_autoencoder(self.input_dim, self.latent_dim, self.input_dim, self.num_embeddings)
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
    
    
    def call(self, x):
        return self.vqvae(x)
    
    
    def save_model_weights(self, file_path):
        self.vqvae.save_weights(file_path)


def train_vq_vae(inputs_dims:int, latent_dims:int, num_embeddings, plot_figure:bool=True):
    x_train, _, x_test, _, _ = data_preprocessing.prepare_data(num_inputs=40, num_outputs=10)
    x_train = np.squeeze(x_train)
    x_test = np.squeeze(x_test)
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    
    variance = np.var(x_train)
    
    vq_vae_trainer = VQVAETrainer(variance, inputs_dims, latent_dims, num_embeddings=num_embeddings)
    vq_vae_trainer.compile(optimizer="adam")
    
    vq_vae_trainer.build((None, inputs_dims))
    
    x_train_hat = vq_vae_trainer.predict(x_train)
        
    vq_vae_trainer.fit(x=x_train, epochs=500, batch_size=64)
    vq_vae_trainer.save_model_weights(f"models/vq_vae_models/vq_vae_input_{inputs_dims}_latent_{latent_dims}_num_embeddings_{num_embeddings}.h5")
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
    # input_dim = 40
    # latent_dim = 10
    # vq_ae = create_quantized_autoencoder(input_dim=input_dim, latent_dim=latent_dim, output_dim=input_dim)
    # print(vq_ae.losses)
    # vq_ae.summary()
    
    train_vq_vae(inputs_dims=40, latent_dims=10, num_embeddings=64, plot_figure=True)
    # x_train, _, x_test, _, _ = data_preprocessing.prepare_data(num_inputs=40, num_outputs=10)
    # encoder = create_encoder(input_dim=40, latent_dim=10)
    # quant_layer = VectorQuantizer(num_embeddings=128, embedding_dim=10)
    # decoder = create_decoder(latent_dim=10, output_dim=40)
    
    # encoder_output = encoder(x_train)
    # encoder_output_quantized = quant_layer(encoder_output)
    # decoder_output = decoder(encoder_output_quantized)

    