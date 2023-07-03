import numpy as np
import matplotlib.pyplot as plt

from Interference_prediction import data_preprocessing
from vq_vae import create_quantized_autoencoder
from vq_vae_ema import create_quantized_autoencoder_EMA


def vq_vae_test(x_test, input_dims, latent_dims, num_embeddings, weights_path):
    vq_vae = create_quantized_autoencoder(input_dims, latent_dims, input_dims, num_embeddings)
    vq_vae.load_weights(weights_path)
    x_test_recover = vq_vae.predict(x_test)
    x_test_recover_1d = x_test_recover.flatten()
    x_test_1d = x_test.flatten()
    abs_deviation = np.abs(x_test_1d - x_test_recover_1d)     
    return x_test_recover_1d, abs_deviation


def vq_vae_ema_test(x_test, input_dims, latent_dims, num_embeddings, weights_path):
    vq_vae_ema = create_quantized_autoencoder_EMA(input_dims, latent_dims, input_dims, num_embeddings)
    vq_vae_ema.load_weights(weights_path)
    
    vq_ema_layer = vq_vae_ema.get_layer("vector_quantizer_ema")
    vq_ema_layer.disable_training_ema()
    x_test_recover = vq_vae_ema.predict(x_test)
    x_test_recover_1d = x_test_recover.flatten()
    x_test_1d = x_test.flatten()
    abs_deviation = np.abs(x_test_1d - x_test_recover_1d)     
    return x_test_recover_1d, abs_deviation


def compare_recover_performance():
    _, _, x_test, _, _ = data_preprocessing.prepare_data(num_inputs=40, num_outputs=10)
    x_test = np.squeeze(x_test)
    input_dims = 40
    latent_dims = 10
    num_embeddings = 128
    
    vq_vae_weights_file = "models/vq_vae_models/vq_vae_input_40_latent_10_num_embeddings_128.h5"
    vq_vae_recover, vq_vae_abs_devation = vq_vae_test(x_test, input_dims, latent_dims, num_embeddings, vq_vae_weights_file)
    
    vq_vae_ema_weights_file = "models/vq_vae_ema_models/vq_vae_ema_input_40_latent_10_num_embeddings_128.h5"
    vq_vae_ema_recover, vq_vae_ema_abs_devation = vq_vae_ema_test(x_test, input_dims, latent_dims, num_embeddings, vq_vae_ema_weights_file)
    
    x_test_1d = x_test.flatten()
    plt.figure()
    plt.plot(x_test_1d[100:500], "r", label="Ture SINR")
    plt.plot(vq_vae_recover[100:500], "b", label="VQ-VAE")
    plt.plot(vq_vae_ema_recover[100:500], "g", label="VQ-VAE-EMA")
    plt.ylabel("SINR (dB)")
    plt.xlabel("Time(ms)")
    plt.grid()
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    compare_recover_performance()
    
