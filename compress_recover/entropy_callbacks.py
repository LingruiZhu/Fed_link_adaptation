import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import Callback

class LatentEntropyCallback(Callback):
    def __init__(self, model, validation_data, input_dim):
        super().__init__()
        self.model = model
        self.validation_data = validation_data
        self.entropy_values_list = []
        self.input_dim = input_dim
        
        
    def on_epoch_end(self, epoch, logs=None):
        # Create a model that includes only the encoder and vector quantizer
        encoder_output = self.model.get_latent_vector(self.validation_data)
        
        # Calculate the counts (number of embeddings assigned to each centroid)
        counts_tf = self.model.vqvae.layers[2].calculate_data_points_number_per_centorid(encoder_output)
        counts = counts_tf.numpy()

        # Normalize counts to obtain probabilities
        probabilities = counts / np.sum(counts)

        # Calculate the entropy for each centroid
        entropy_per_centroid = -np.sum(probabilities * np.log2(probabilities + 1e-10))

        # Log the entropy for this epoch
        print(f'Epoch {epoch + 1}, Entropy of Latent Variables: {entropy_per_centroid}')

        # Store the entropy values for later analysis
        self.entropy_values_list.append(entropy_per_centroid)



