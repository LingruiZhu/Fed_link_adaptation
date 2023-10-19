
import sys 
sys.path.append("/home/zhu/Codes/Fed_Link_Adaptation")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()

from tensorflow.keras import backend as K

import os
import h5py

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Dense, BatchNormalization


from sklearn.metrics import mean_squared_error

from Interference_prediction import data_preprocessing
from kmpp_initialization.kmpp import kmeans_plusplus_initialization
from compress_recover.auto_encoder import create_dense_encoder, create_dense_decoder, \
    create_lstm_encoder, create_lstm_decoder


class VectorQuantizer_EMA(Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=1, ema_decay=0.85, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta
        self.ema_decay = ema_decay
        
        self.is_training_ema = True

        # Initialize the embeddings.
        w_init = tf.random_uniform_initializer(-1,1)
        self.embeddings = tf.Variable(
            initial_value=w_init(shape=(self.embedding_dim, self.num_embeddings), dtype="float32"),
            trainable=True,
            name="embeddings_vqvae")
        self.ema_count = tf.Variable(
            initial_value=tf.zeros(shape=(self.num_embeddings,), dtype="float32"),
            trainable=False,
            name="ema_count_vqvae")
        self.count = tf.Variable(
            initial_value=tf.zeros(shape=(self.num_embeddings,), dtype="float32"),
            trainable=False,
            name="count_vqvae")
        self.embeddings_sum = tf.Variable(
            initial_value=tf.zeros(shape=(self.embedding_dim, self.num_embeddings), dtype="float32"),
            trainable=False,
            name="embeddings_sum_vqvae")
        self.embedding_sample_accumulative_count = tf.Variable(
            initial_value=tf.zeros(shape=(self.num_embeddings,), dtype="float32"),
            trainable=False,
            name="embedding_sample_accumulative_count")

    
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
        
        # if self.is_training_ema:
        #     self.update_ema_embeddings(x)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss)      # here, codebook loss will be excluded 

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized


    def update_ema_embeddings(self, inputs):
        print("now update ema")
        flattened_inputs = tf.reshape(inputs, [-1, self.embedding_dim])

        # Calculate the encoding indices based on the flattened inputs
        encoding_indices = self.get_code_indices(flattened_inputs)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        
        # Calculate the count of each codebook vector based on the encoding indices
        count = tf.reduce_sum(encodings, 0)

        # Update the EMA count using the decay factor
        self.ema_count.assign(self.ema_decay * self.ema_count + (1-self.ema_decay) * count)

        # Calculate the EMA of the codebook embeddings
        embeddings_sum = tf.matmul(flattened_inputs, encodings, transpose_a=True)
        updated_embeddings_sum = self.ema_decay * self.embeddings_sum + (1-self.ema_decay) * embeddings_sum

        # Normalize the updated codebook embeddings using the count
        normalized_embeddings = updated_embeddings_sum / tf.maximum(self.ema_count, 1e-5)

        # Assign the normalized embeddings to the codebook
        self.embeddings.assign(normalized_embeddings)
        self.embeddings_sum.assign(updated_embeddings_sum)
        self.count.assign(count)
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
        config = super(VectorQuantizer_EMA, self).get_config()
        config.update({
            'num_embeddings': self.num_embeddings,
            'embedding_dim': self.embedding_dim
        })
        return config 
    
    
def create_quantized_autoencoder_EMA(model_type, input_dim, latent_dim, output_dim, num_embeddings:int=128,\
    ema_decay:float=0.99, commitment_factor:float=0.25):
    if model_type == "dense":
        encoder = create_dense_encoder(input_dim, latent_dim)
        decoder = create_dense_decoder(output_dim, latent_dim)
    elif model_type == "lstm":
        encoder = create_lstm_encoder(input_dim, latent_dim)
        decoder = create_lstm_decoder(output_dim, latent_dim)
        
    quantizer = VectorQuantizer_EMA(num_embeddings=num_embeddings, embedding_dim=latent_dim, \
        ema_decay=ema_decay, beta=commitment_factor)
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


class VQVAETrainerEMA(Model):
    def __init__(self, model_type:str, train_variance:float, input_dim:int, latent_dim:int=10, \
        num_embeddings:int=1, ema_decay:float=0.99,\
        commitment_factor:float=0.25, **kwargs):
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.num_embeddings = num_embeddings
        self.commitment_factor = commitment_factor
        self.model_type = model_type
        
        self.vqvae = create_quantized_autoencoder_EMA(model_type, self.input_dim, self.latent_dim, self.input_dim,\
            self.num_embeddings, ema_decay, commitment_factor=commitment_factor)
        
        self.learning_rates_list = list()

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="codebook_loss")


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]
    
    
    def get_latent_vector(self, x):
        x1 = self.vqvae.layers[0](x)
        latent_vec = self.vqvae.layers[1](x1)
        return latent_vec
        

    def train_step(self, x):
        with tf.GradientTape(persistent=True) as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + self.commitment_factor * sum(self.vqvae.losses) # here vavae only include commitment loss

        # Backpropagation w.r.t. total loss
        # grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        # self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))
        
        # Backpropogation w.r.t. reconstruction loss
        reconstruction_loss_grads = tape.gradient(reconstruction_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(reconstruction_loss_grads, self.vqvae.trainable_variables))
        
        # Update embeding vectors
        latent_var = self.get_latent_vector(x)
        self.vqvae.layers[2].update_ema_embeddings(latent_var)

        
        # Backpropoagation w.r.t. commitment loss
        commitment_loss_grad = tape.gradient(self.vqvae.losses, self.vqvae.layers[1].trainable_variables) # need change the varible
        self.optimizer.apply_gradients(zip(commitment_loss_grad, self.vqvae.layers[1].trainable_variables))
        
        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "codebook_loss": self.vq_loss_tracker.result(),
        }
    
    
    def call(self, x, is_ema_updating:bool=True):
        return self.vqvae(x, is_ema_updating)
    
    
    def save_model_weights(self, file_path):
        self.vqvae.save_weights(file_path)


class LearningRateCallback(keras.callbacks.Callback):
    def __init__(self, vqvae):
        super().__init__()
        self.vqvae = vqvae
        self.learning_rates_list = []
        self.num_active_embeddings_list = []

    def on_epoch_end(self, batch, logs=None):
        # Calculate the learning rate
        learning_rate = (1 - self.vqvae.layers[2].ema_decay) * (self.vqvae.layers[2].count / (2 * self.vqvae.layers[2].ema_count + 1e-5))

        # Convert learning rate tensor to numpy array and append to the list
        self.learning_rates_list.append(tf.keras.backend.eval(learning_rate))
        num_active_embeddings = tf.math.count_nonzero(self.vqvae.layers[2].embedding_sample_accumulative_count)
        self.num_active_embeddings_list.append((tf.keras.backend.eval(num_active_embeddings)))


def train_vq_vae_ema(model_type:str, inputs_dims:int, latent_dims:int, num_embeddings:int, embedding_init:str="random", num_epochs:int=300, \
    commitment_factor:float=0.25, ema_decay:float=0.99, plot_figure:bool=True, optimizer:str="adam", init_epochs:int=100, re_init_interval:int=20, \
        simulation_index:int=None):
    
    x_train, _, x_test, _, _ = data_preprocessing.prepare_data(num_inputs=40, num_outputs=10)
    x_train = np.squeeze(x_train)
    x_test = np.squeeze(x_test)
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    variance = np.var(x_train)
    
    vq_vae_trainer = VQVAETrainerEMA(model_type, variance, inputs_dims, latent_dims, \
        num_embeddings=num_embeddings, ema_decay=ema_decay, commitment_factor=commitment_factor)
    vq_vae_trainer.compile(optimizer=optimizer)
    vq_vae_trainer.build((None, inputs_dims))
    
    learning_rate_callback = LearningRateCallback(vq_vae_trainer.vqvae)
    
    if embedding_init == "random":
        history = vq_vae_trainer.fit(x=x_train, validation_split=0.2, epochs=num_epochs, batch_size=64, \
            callbacks=[learning_rate_callback])
    elif embedding_init == "kmpp":
         # here define the encoder and assign new weights during the training process    
        input_tensor = vq_vae_trainer.vqvae.layers[1].input
        output_tensor = vq_vae_trainer.vqvae.layers[1].output
        encoder_model = Model(inputs=input_tensor, outputs=output_tensor)
                
        for epoch in range(num_epochs):
            print(f"Training epochs: {epoch}/{num_epochs}:")
            history_single_epoch = vq_vae_trainer.fit(x=x_train, validation_split=0.2, epochs=1, batch_size=64, \
                callbacks=[learning_rate_callback], verbose=1)
            if epoch == 0:                  # initialize history object
                history = history_single_epoch
            else:
                for key in history.history.keys():
                    history.history[key].append(history_single_epoch.history[key][0])
            if epoch > 0 and epoch<=init_epochs and epoch%re_init_interval==0:
                print(f"LOOK! HERE! Now embedding space is re-initialized at {epoch}-th epoch.")
                # assign updates weights to encoder
                encoder_model.set_weights(vq_vae_trainer.vqvae.layers[1].get_weights())
                # use new latent variable to renitialize centriods using kmpp
                latent_space = encoder_model.predict(x_train)
                kmpp_centroids = kmeans_plusplus_initialization(data=latent_space, k=num_embeddings)
                vq_vae_trainer.vqvae.layers[2].embeddings.assign(kmpp_centroids)
    
    learning_rate_list = learning_rate_callback.learning_rates_list
    num_active_embeddings_list = learning_rate_callback.num_active_embeddings_list
        
    file_name = f"{model_type}_vq_vae_ema_input_{inputs_dims}_latent_{latent_dims}_num_embeddings_{num_embeddings}_init_{embedding_init}_{optimizer}_ema_decay_{ema_decay}_beta_{commitment_factor}"
    if simulation_index != None:
        file_name = file_name + "_index_" + str(simulation_index)
    file_name = file_name + ".h5"
    
    if embedding_init == "random":  
        weights_path = os.path.join("models_new", "VQ_VAE_EMA_models", f"latent_dim_{latent_dims}", file_name)
        history_path = os.path.join("training_history_new", "VQ_VAE_EMA_history", f"latent_dim_{latent_dims}", file_name)
    elif embedding_init == "kmpp":        
        weights_path = os.path.join("models_new", "VQ_VAE_EMA_KMPP_models", f"latent_dim_{latent_dims}", file_name)
        history_path = os.path.join("training_history_new", "VQ_VAE_EMA_KMPP_history", f"latent_dim_{latent_dims}", file_name)
    
    # save weights and training history
    vq_vae_trainer.save_model_weights(weights_path)
    with h5py.File(history_path, "w") as hf:
        for key, value in history.history.items():
            hf.create_dataset(key, data=value)
        hf.create_dataset("learning_rates", data=learning_rate_list)
        hf.create_dataset("num_active_embeddings", data=num_active_embeddings_list)
    
    # disable ema updates
    vq_ema_layer = vq_vae_trainer.vqvae.layers[2]
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


def test_vq_vae_ema(inputs_dims:int, latent_dims:int, num_embeddings, plot_figure:bool=True):
    # load model from file
    vq_vae_ema = create_quantized_autoencoder_EMA(inputs_dims, latent_dims, inputs_dims, num_embeddings)
    vq_vae_ema.load_weights("models/vq_vae_ema_models/vq_vae_ema_input_40_latent_10_num_embeddings_128.h5")
    vq_ema_layer = vq_vae_ema.get_layer("vector_quantizer_ema")
    vq_ema_layer.disable_training_ema()
    
    # prepare data
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
        
        
def tensors_to_numpy_list(tensor_list):
    # Initialize an empty list to store NumPy arrays
    numpy_list = []

    # Iterate through the tensor_list and convert each tensor to a NumPy array
    for tensor in tensor_list:
        numpy_array = tf.make_ndarray(tensor)
        numpy_list.append(numpy_array)
    return numpy_list


if __name__ == "__main__":    
    ema_decay = 0.99
    beta = 0.25
    embeeding_init = "random"
    optimizer = "RMSprop"
    
    # train_vq_vae(inputs_dims=40, latent_dims=20, num_embeddings=16, commitment_factor=beta, plot_figure=False, ema_decay=ema_decay, embedding_init=embeeding_init)
    # train_vq_vae(inputs_dims=40, latent_dims=20, num_embeddings=32, commitment_factor=beta, plot_figure=False, ema_decay=ema_decay, embedding_init=embeeding_init)
    # train_vq_vae(inputs_dims=40, latent_dims=20, num_embeddings=64, commitment_factor=beta, plot_figure=False, ema_decay=ema_decay, embedding_init=embeeding_init)
    # train_vq_vae(inputs_dims=40, latent_dims=20, num_embeddings=128, commitment_factor=beta, plot_figure=False, ema_decay=ema_decay, embedding_init=embeeding_init)
    # train_vq_vae(inputs_dims=40, latent_dims=20, num_embeddings=256, commitment_factor=beta, plot_figure=False, ema_decay=ema_decay, embedding_init=embeeding_init)
    # train_vq_vae(inputs_dims=40, latent_dims=20, num_embeddings=512, commitment_factor=beta, plot_figure=False, ema_decay=ema_decay, embedding_init=embeeding_init)
    # train_vq_vae(inputs_dims=40, latent_dims=20, num_embeddings=1024, commitment_factor=beta, plot_figure=False, ema_decay=ema_decay, embedding_init=embeeding_init)

    train_vq_vae_ema(model_type="lstm", inputs_dims=40, latent_dims=20, num_embeddings=128, num_epochs=20, \
        commitment_factor=beta, plot_figure=False, ema_decay=ema_decay, embedding_init=embeeding_init, simulation_indedx=0)
