import sys
sys.path.append("/home/zhu/Codes/Fed_Link_Adaptation")

from tensorflow.keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt

from Interference_prediction import data_preprocessing
from compress_recover.vq_vae import create_quantized_autoencoder
from compress_recover.vq_vae_ema import create_quantized_autoencoder_EMA
from compress_recover.auto_encoder_quant_latent import create_uniform_quantized_autoencoder

from tabulate import tabulate


def ae_uniform_quant_test(x_test, input_dims, latent_dims, num_quant_bits, weights_path):
    ae_uniform_quant = create_uniform_quantized_autoencoder(input_dims, latent_dims, input_dims, num_quant_bits)
    ae_uniform_quant.load_weights(weights_path)
    x_test_recover = ae_uniform_quant.predict(x_test)
    x_test_recover_1d = x_test_recover.flatten()
    x_test_1d = x_test.flatten()
    abs_deviation = np.abs(x_test_1d - x_test_recover_1d)     
    return x_test_recover, x_test_recover_1d, abs_deviation
    

def vq_vae_test(x_test, input_dims, latent_dims, num_embeddings, weights_path):
    vq_vae = create_quantized_autoencoder(input_dims, latent_dims, input_dims, num_embeddings)
    vq_vae.load_weights(weights_path)
    x_test_recover = vq_vae.predict(x_test)
    x_test_recover_1d = x_test_recover.flatten()
    x_test_1d = x_test.flatten()
    abs_deviation = np.abs(x_test_1d - x_test_recover_1d)     
    return x_test_recover, x_test_recover_1d, abs_deviation


def vq_vae_ema_test(x_test, input_dims, latent_dims, num_embeddings, weights_path):
    vq_vae_ema = create_quantized_autoencoder_EMA(input_dims, latent_dims, input_dims, num_embeddings)
    vq_vae_ema.load_weights(weights_path)
    
    vq_ema_layer = vq_vae_ema.get_layer("vector_quantizer_ema")
    vq_ema_layer.disable_training_ema()
    x_test_recover = vq_vae_ema.predict(x_test)
    x_test_recover_1d = x_test_recover.flatten()
    x_test_1d = x_test.flatten()
    abs_deviation = np.abs(x_test_1d - x_test_recover_1d)     
    return x_test_recover, x_test_recover_1d, abs_deviation


def compare_recover_performance():
    _, _, x_test, _, _ = data_preprocessing.prepare_data(num_inputs=40, num_outputs=10)
    x_test = np.squeeze(x_test)
    input_dims = 40
    latent_dims = 20
    num_embeddings = 256
    num_quant_bits = 4
    
    ae_quant_latent_weights_file = "models/vq_vae_uniform_quant/vq_vae_input_40_latent_20_num_quant_bits_4_optimizer_RMSprop.h5"
    _, ae_quant_recover, ae_quant_abs_devation = ae_uniform_quant_test(x_test, input_dims, latent_dims, num_quant_bits, ae_quant_latent_weights_file)
    
    vq_vae_weights_file = "models/vq_vae_larger_init/vq_vae_input_40_latent_20_num_embeddings_256_with_BN_False_RMSprop.h5"
    _, vq_vae_recover, vq_vae_abs_devation = vq_vae_test(x_test, input_dims, latent_dims, num_embeddings, vq_vae_weights_file)
    
    vq_vae_ema_weights_file = "models/vq_vae_ema_larger_init/vq_vae_ema_input_40_latent_20_num_embeddings_256_ema_decay_0.99_beta_0.25.h5"
    _, vq_vae_ema_recover, vq_vae_ema_abs_devation = vq_vae_ema_test(x_test, input_dims, latent_dims, num_embeddings, vq_vae_ema_weights_file)
    
    x_test_1d = x_test.flatten()
    
    print(tabulate([["VQ-VAE", np.mean(vq_vae_abs_devation**2)], ["VQ-VAE-EMA", np.mean(vq_vae_ema_abs_devation**2)], ["AE-Uniform-quant", np.mean(ae_quant_abs_devation**2)]], headers=["Method", "ABS"]))
    
    plt.figure()
    plt.plot(x_test_1d[100:500], "r", label="Ture SINR")
    plt.plot(vq_vae_recover[100:500], "b", label="VQ-VAE")
    plt.plot(vq_vae_ema_recover[100:500], "g", label="VQ-VAE-EMA")
    plt.plot(ae_quant_recover[100:500], "m", label="AE-Uniform-quant")
    plt.ylabel("SINR (dB)")
    plt.xlabel("Time (ms)")
    plt.grid()
    plt.legend()
    plt.savefig("figures/estimation/estimation.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    plt.show()
    

def compare_recover_prediction():
    _, _, x_test, y_test, _ = data_preprocessing.prepare_data(num_inputs=40, num_outputs=10)
    x_test = np.squeeze(x_test)
    input_dims = 40
    latent_dims = 20
    num_embeddings = 256
    num_quant_bits = 4
    
    ae_quant_latent_weights_file = "models/vq_vae_uniform_quant/vq_vae_input_40_latent_20_num_quant_bits_4_optimizer_RMSprop.h5"
    ae_quant_recover, ae_quant_recover_1d, ae_quant_abs_devation = ae_uniform_quant_test(x_test, input_dims, latent_dims, num_quant_bits, ae_quant_latent_weights_file)
    
    vq_vae_weights_file = "models/vq_vae_larger_init/vq_vae_input_40_latent_20_num_embeddings_256_with_BN_False_RMSprop.h5"
    vq_vae_recover, vq_vae_recover_1d, vq_vae_abs_devation = vq_vae_test(x_test, input_dims, latent_dims, num_embeddings, vq_vae_weights_file)
    
    vq_vae_ema_weights_file = "models/vq_vae_ema_larger_init/vq_vae_ema_input_40_latent_20_num_embeddings_256_ema_decay_0.99_beta_0.25.h5"
    vq_vae_ema_recover, vq_vae_ema_recover_1d, vq_vae_ema_abs_devation = vq_vae_ema_test(x_test, input_dims, latent_dims, num_embeddings, vq_vae_ema_weights_file)
    
    lstm_model = load_model("Interference_prediction/models/lstm.h5")
    ae_quant_prediction = lstm_model.predict(ae_quant_recover)
    vq_vae_prediction = lstm_model.predict(vq_vae_recover)
    
    ae_quant_prediction_1d = ae_quant_prediction.flatten()
    vq_vae_prediction_1d = vq_vae_prediction.flatten()
    true_sinr_1d = y_test.flatten()
    
    plt.figure()
    plt.plot(true_sinr_1d[100:500], "r", label="Ture SINR")
    plt.plot(vq_vae_prediction_1d[100:500], "b", label="VQ-VAE")
    plt.plot(ae_quant_prediction_1d[100:500], "m", label="AE-Uniform-quant")
    plt.ylabel("SINR (dB)")
    plt.xlabel("Time (ms)")
    plt.grid()
    plt.legend()
    plt.savefig("figures/prediction/estimation.pdf", format="pdf", bbox_inches="tight")
    plt.show()
        

if __name__ == "__main__":
    compare_recover_performance()
    # compare_recover_prediction()

