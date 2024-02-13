import sys
sys.path.append("/home/zhu/Codes/Fed_Link_Adaptation")

from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

import tensorflow as tf
from tensorflow.keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt

from Interference_prediction import data_preprocessing
from compress_recover.vq_vae import create_quantized_autoencoder
from compress_recover.vq_vae_ema import create_quantized_autoencoder_EMA
from compress_recover.auto_encoder_quant_latent import create_uniform_quantized_autoencoder

from tabulate import tabulate


class PerformancePlotter:
    def __init__(self, weights_path:str, model_type:str, initialization:str, input_dim:int, latent_dim:int, num_embeddings:int, num_quant_bits:int, \
        line_format:str, AE_plus_LM=False) -> None:
        self.weight_path = weights_path
        self.model_type = model_type
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.num_quant_bits = num_quant_bits
        self.initialization = initialization
        
        self.AE_Plus_LM = AE_plus_LM
        
        self.line_format = line_format
        if model_type == "AE_uniform_quant":
            self.label = f"{model_type}_{latent_dim*num_quant_bits}_quant_bits"
        elif model_type == "VQ_VAE" or model_type == "VQ_VAE_EMA":
            self.label = f"{model_type}_{initialization}_{num_embeddings}_embeddings"
        elif model_type.lower() == "lloyd_max":
            self.label = f"{model_type}_{initialization}_{num_embeddings}_embeedings"
            if AE_plus_LM:
                self.label = self.label + "_AE_recoverd_data"
        elif model_type.lower() == "ae_no_quant":
            self.label = "AE_no_quant"


    def recover_sequense(self, x_test, x_train=None):
        if self.model_type == "AE_uniform_quant":
            _, self.recovered_signal, self.abs_deviation, self.nmse = ae_uniform_quant_test(x_test, self.input_dim, self.latent_dim, num_quant_bits=self.num_quant_bits,\
                weights_path=self.weight_path)
        elif self.model_type == "VQ_VAE":
            _, self.recovered_signal, self.abs_deviation, self.nmse = vq_vae_test(x_test, self.input_dim, self.latent_dim, num_embeddings=self.num_embeddings,\
                weights_path=self.weight_path)
        elif self.model_type == "VQ_VAE_EMA":
            _, self.recovered_signal, self.abs_deviation, self.nmse = vq_vae_ema_test(x_test, self.input_dim, self.latent_dim, num_embeddings=self.num_embeddings, \
                weights_path=self.weight_path)
        elif self.model_type.lower() == "lloyd_max":
            if self.AE_Plus_LM:
                _, self.recovered_signal, self.abs_deviation, self.nmse = lloyd_max_test(x_train, x_test, num_embedings=self.num_embeddings, \
                    initialization=self.initialization, AE_test=True)
            else:
                _, self.recovered_signal, self.abs_deviation, self.nmse = lloyd_max_test(x_train, x_test, num_embedings=self.num_embeddings, \
                    initialization=self.initialization)
        elif self.model_type.lower() == "ae_no_quant":
            _, self.recovered_signal, self.abs_deviation, self.nmse = ae_no_quant_test(x_test, self.input_dim, self.latent_dim, model_path=self.weight_path)
        tf.keras.backend.clear_session()
    
    
    def predict_sequence(self, x_test, x_train=None):
        if self.model_type == "AE_uniform_quant":
            recoverd_signal_to_predict, _, _, _ = ae_uniform_quant_test(x_test, self.input_dim, self.latent_dim, num_quant_bits=self.num_quant_bits,\
            weights_path=self.weight_path)
        elif self.model_type == "VQ_VAE":
            recoverd_signal_to_predict, _, _, _ = vq_vae_test(x_test, self.input_dim, self.latent_dim, num_embeddings=self.num_embeddings,\
                weights_path=self.weight_path)
        elif self.model_type == "VQ_VAE_EMA":
            recoverd_signal_to_predict, _, _, _ = vq_vae_ema_test(x_test, self.input_dim, self.latent_dim, num_embeddings=self.num_embeddings, \
                weights_path=self.weight_path)
        elif self.model_type.lower() == "lloyd_max":
            recoverd_signal_to_predict, _, _, _ = lloyd_max_test(x_train, x_test, num_embedings=self.num_embeddings, \
                initialization=self.initialization)
        elif self.model_type.lower() == "ae_no_quant":
            recoverd_signal_to_predict, _, _, _ = ae_no_quant_test(x_test, self.input_dim, self.latent_dim, model_path=self.weight_path)
        lstm_model = load_model("Interference_prediction/models/lstm.h5")
        self.recover_sequense(x_test, x_train)
        self.precited_signal = lstm_model.predict(recoverd_signal_to_predict)
        tf.keras.backend.clear_session()


def ae_no_quant_test(x_test, input_dims, latent_dims, model_path):
    ae_model = load_model(model_path)
    x_test_recover = ae_model.predict(x_test)
    x_test_recover_1d = x_test_recover.flatten()
    x_test_1d = x_test.flatten()
    abs_deviation = np.abs(x_test_1d - x_test_recover_1d)
    nmse =  np.mean((np.abs(x_test_1d - x_test_recover_1d)**2) / (np.abs(x_test_1d)**2))
    tf.keras.backend.clear_session()
    return x_test_recover, x_test_recover_1d, abs_deviation, nmse


def ae_uniform_quant_test(x_test, input_dims, latent_dims, num_quant_bits, weights_path):
    ae_uniform_quant = create_uniform_quantized_autoencoder(input_dims, latent_dims, input_dims, num_quant_bits)
    ae_uniform_quant.load_weights(weights_path)
    x_test_recover = ae_uniform_quant.predict(x_test)
    x_test_recover_1d = x_test_recover.flatten()
    x_test_1d = x_test.flatten()
    abs_deviation = np.abs(x_test_1d - x_test_recover_1d)
    nmse =  np.mean((np.abs(x_test_1d - x_test_recover_1d)**2) / (np.abs(x_test_1d)**2))
    tf.keras.backend.clear_session()     
    return x_test_recover, x_test_recover_1d, abs_deviation, nmse
    

def vq_vae_test(x_test, input_dims, latent_dims, num_embeddings, weights_path, layer_type:str="dense"):
    vq_vae = create_quantized_autoencoder(layer_type, input_dims, latent_dims, input_dims, num_embeddings)
    vq_vae.load_weights(weights_path)
    x_test_recover = vq_vae.predict(x_test)
    x_test_recover_1d = x_test_recover.flatten()
    x_test_1d = x_test.flatten()
    abs_deviation = np.abs(x_test_1d - x_test_recover_1d)
    nmse =  np.mean((np.abs(x_test_1d - x_test_recover_1d)**2) / (np.abs(x_test_1d)**2))
    tf.keras.backend.clear_session()    
    return x_test_recover, x_test_recover_1d, abs_deviation, nmse


def vq_vae_ema_test(x_test, input_dims, latent_dims, num_embeddings, weights_path, layer_type:str="dense"):
    vq_vae_ema = create_quantized_autoencoder_EMA(layer_type, input_dims, latent_dims, input_dims, num_embeddings)
    vq_vae_ema.load_weights(weights_path)
    
    vq_ema_layer = vq_vae_ema.layers[2]
    vq_ema_layer.disable_training_ema()
    x_test_recover = vq_vae_ema.predict(x_test)
    x_test_recover_1d = x_test_recover.flatten()
    x_test_1d = x_test.flatten()
    abs_deviation = np.abs(x_test_1d - x_test_recover_1d)    
    nmse =  np.mean((np.abs(x_test_1d - x_test_recover_1d)**2) / (np.abs(x_test_1d)**2))    
    tf.keras.backend.clear_session() 
    return x_test_recover, x_test_recover_1d, abs_deviation, nmse


def lloyd_max_test(x_train, x_test, num_embedings, initialization, AE_test=False):
    # Prepare data for K-Means training
    data_train = np.squeeze(x_train)
    
    # Initialize and train K-Means model
    if initialization == "kmpp":
        init_method = "k-means++"
    elif initialization == "random":
        init_method = "random"
    kmeans = KMeans(n_clusters=num_embedings, init=init_method, random_state=0, max_iter=50)
    if AE_test:
        AE_model = load_model("models/ae_models/AE_input_40_latent_20_optimizer_RMSprop.h5")
        recovered_data = AE_model.predict(x_train)
        data_train_after_AE = np.squeeze(recovered_data)
        data_train_after_AE_64 = data_train_after_AE.astype(data_train.dtype)
        kmeans.fit(data_train_after_AE_64)
    else:
        kmeans.fit(data_train)
    
    # Prepare data and test K-Means model
    labels_test = kmeans.predict(x_test)
    centroids = kmeans.cluster_centers_
    quantized_sequence  = [centroids[label] for label in labels_test]
    
    # Change the output format
    x_test_recover = np.array(quantized_sequence)
    x_test_recover_1d = x_test_recover.flatten()
    x_test_1d = x_test.flatten()
    abs_deviation = np.abs(x_test_1d - x_test_recover_1d)
    nmse =  np.mean((np.abs(x_test_1d - x_test_recover_1d)**2) / (np.abs(x_test_1d)**2))    
    return x_test_recover, x_test_recover_1d, abs_deviation, nmse


def compare_recover_performance(plotter_list:list):
    x_train, y_train, x_test, y_test, _ = data_preprocessing.prepare_data(num_inputs=40, num_outputs=10)
    x_test = np.squeeze(x_test)
    # input_dims = 40
    # latent_dims = 20
    # num_embeddings = 256
    # num_quant_bits = 4
    
    # ae_quant_latent_weights_file = "models/vq_vae_uniform_quant/vq_vae_input_40_latent_20_num_quant_bits_4_optimizer_RMSprop.h5"
    # _, ae_quant_recover, ae_quant_abs_devation = ae_uniform_quant_test(x_test, input_dims, latent_dims, num_quant_bits, ae_quant_latent_weights_file)
    
    # vq_vae_weights_file = "models/vq_vae_larger_init/vq_vae_input_40_latent_20_num_embeddings_256_with_BN_False_RMSprop.h5"
    # _, vq_vae_recover, vq_vae_abs_devation = vq_vae_test(x_test, input_dims, latent_dims, num_embeddings, vq_vae_weights_file)
    
    # vq_vae_ema_weights_file = "models/vq_vae_ema_larger_init/vq_vae_ema_input_40_latent_20_num_embeddings_256_ema_decay_0.99_beta_0.25.h5"
    # _, vq_vae_ema_recover, vq_vae_ema_abs_devation = vq_vae_ema_test(x_test, input_dims, latent_dims, num_embeddings, vq_vae_ema_weights_file)
    
    x_test_1d = x_test.flatten()
    
    # print(tabulate([["VQ-VAE", np.mean(vq_vae_abs_devation**2)], ["VQ-VAE-EMA", np.mean(vq_vae_ema_abs_devation**2)], ["AE-Uniform-quant", np.mean(ae_quant_abs_devation**2)]], headers=["Method", "ABS"]))
    
    label_list = list()
    recovery_mse_list = list()
    prediction_mse_list = list()
    recovery_nmse_list = list()
    prediction_nmse_list = list()
    
    marker_interval = 10
    
    plt.figure()
    plt.plot(x_test_1d[:500], "k-o", label="Ture SINR", markevery=marker_interval)
    # plt.plot(vq_vae_recover[100:500], "b", label="VQ-VAE")
    # plt.plot(vq_vae_ema_recover[100:500], "g", label="VQ-VAE-EMA")
    # plt.plot(ae_quant_recover[100:500], "m", label="AE-Uniform-quant")
    for plotter in plotter_list:
        plotter.recover_sequense(x_test, x_train)
        current_mse = mean_squared_error(x_test_1d, plotter.recovered_signal)
        plotter.predict_sequence(x_test, x_train)
        current_pred_mse = mean_squared_error(y_test, plotter.precited_signal)
        label_list.append(plotter.label)
        recovery_mse_list.append(current_mse)
        prediction_mse_list.append(current_pred_mse)
        
        recovery_nmse_list.append(current_mse / np.mean(x_test_1d))
        prediction_nmse_list.append(current_pred_mse / np.mean(y_test))
        
        plt.plot(plotter.recovered_signal[:500], plotter.line_format, label=plotter.label, markevery=marker_interval)
    plt.ylabel("SINR (dB)")
    plt.xlabel("Time (ms)")
    plt.grid()
    plt.legend()
    plt.savefig("figures/estimation/estimation.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    plt.show()
    
    raw_data1 = [label_list, recovery_mse_list, prediction_mse_list]
    table_data1 = list(map(list, zip(*raw_data1)))
    headers = ["Method", "Recovery MSE", "prediction MSE"]
    table1 = tabulate(table_data1, headers, tablefmt="grid")
    print(table1)
    
    raw_data2 = [label_list, recovery_nmse_list, prediction_nmse_list]
    table_data2 = list(map(list, zip(*raw_data2)))
    headers_2 = ["Method", "Recovery NMSE", "prediction NMSE"]
    table1 = tabulate(table_data2, headers_2, tablefmt="grid")
    print(table1)
    

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
        

def algorithm_compare():
    input_dims = 40
    latent_dims = 20
    num_embeddings = 128
    
    plotter_list = list()
    
    lloyd_max_kmpp_plotter_AE_plus_LM = PerformancePlotter(weights_path=None, model_type="lloyd_max", initialization="kmpp", input_dim=input_dims, latent_dim=None, \
    num_embeddings=num_embeddings, num_quant_bits=0, line_format="y-^", AE_plus_LM=True)
    plotter_list.append(lloyd_max_kmpp_plotter_AE_plus_LM)
    
    ae_no_quant_path = "models/ae_models/AE_input_40_latent_20_optimizer_RMSprop.h5"
    ae_no_quant_plotter = PerformancePlotter(weights_path=ae_no_quant_path, model_type="ae_no_quant", initialization=None, \
        input_dim=input_dims, latent_dim=latent_dims, num_embeddings=0, num_quant_bits=None, line_format="g-d")
    plotter_list.append(ae_no_quant_plotter)
    
    ae_uniform_quant_path = "models/vq_vae_uniform_quant/vq_vae_input_40_latent_20_num_quant_bits_4_optimizer_RMSprop.h5"
    ae_uniform_quant_plotter = PerformancePlotter(weights_path =ae_uniform_quant_path, model_type="AE_uniform_quant", initialization=None,\
        input_dim=input_dims, latent_dim=latent_dims, num_embeddings=0, num_quant_bits=4, line_format="g-x")
    plotter_list.append(ae_uniform_quant_plotter)
    
    vq_vae_random_path = f"models/vq_vae/vq_vae_input_40_latent_20_num_embeddings_{num_embeddings}_init_random_with_BN_False_RMSprop.h5"
    vq_vae_random_plotter = PerformancePlotter(weights_path=vq_vae_random_path, model_type="VQ_VAE", initialization="Random",\
        input_dim=input_dims, latent_dim=latent_dims, num_embeddings=num_embeddings, num_quant_bits=0, line_format="b-x")
    plotter_list.append(vq_vae_random_plotter)
    
    vq_vae_kmpp_path = f"models/vq_vae_kmpp_init/vq_vae_input_40_latent_20_num_embeddings_{num_embeddings}_init_kmpp_with_BN_False_RMSprop.h5"
    vq_vae_kmpp_plotter = PerformancePlotter(weights_path=vq_vae_kmpp_path, model_type="VQ_VAE", initialization="KMPP", \
        input_dim=input_dims, latent_dim=latent_dims, num_embeddings=num_embeddings, num_quant_bits=0, line_format="m-+")
    plotter_list.append(vq_vae_kmpp_plotter)
    
    vq_vae_ema_path = f"models/vq_vae_ema/vq_vae_ema_input_40_latent_20_num_embeddings_{num_embeddings}_init_random_ema_decay_0.99_beta_0.25.h5"
    vq_vae_ema_plotter = PerformancePlotter(weights_path=vq_vae_ema_path, model_type="VQ_VAE_EMA", initialization="Random", \
        input_dim=input_dims, latent_dim=latent_dims, num_embeddings=num_embeddings, num_quant_bits=0, line_format="g-o")
    plotter_list.append(vq_vae_ema_plotter)
    
    vq_vae_ema_kmpp_path = f"models/vq_vae_ema_kmpp_init/vq_vae_ema_input_40_latent_20_num_embeddings_{num_embeddings}_init_kmpp_ema_decay_0.99_beta_0.25.h5"
    vq_vae_ema_kmpp_plotter = PerformancePlotter(weights_path=vq_vae_ema_kmpp_path, model_type="VQ_VAE_EMA", initialization="KMPP",\
        input_dim=input_dims, latent_dim=latent_dims, num_embeddings=num_embeddings, num_quant_bits=0, line_format="r-s")
    plotter_list.append(vq_vae_ema_kmpp_plotter)
    
    lloyd_max_random_path = None
    lloyd_max_random_plotter = PerformancePlotter(weights_path=lloyd_max_random_path, model_type="lloyd_max", initialization="random", input_dim=input_dims,\
        latent_dim=None, num_embeddings=num_embeddings, num_quant_bits=0, line_format="k-d")
    plotter_list.append(lloyd_max_random_plotter)
    
    lloyd_max_kmpp_plotter = PerformancePlotter(weights_path=None, model_type="lloyd_max", initialization="kmpp", input_dim=input_dims, latent_dim=None, \
        num_embeddings=num_embeddings, num_quant_bits=0, line_format="y-^")
    plotter_list.append(lloyd_max_kmpp_plotter)
    

    
    compare_recover_performance(plotter_list)
    # compare_recover_prediction()


def parameter_compare():
    num_bits_list = [4, 5, 6, 7, 8, 9, 10]
    num_embeddings_list = [2**i for i in num_bits_list]
    
    input_dim = 40
    latent_dim = 20
    _, _, x_test, y_test, _ = data_preprocessing.prepare_data(num_inputs=40, num_outputs=10)
    
    vq_vae_mse_list = list()
    vq_vae_kmpp_mse_list = list()
    vq_vae_ema_mse_list = list()
    vq_vae_ema_kmpp_mse_list = list()
    
    vq_vae_nmse_list = list()
    vq_vae_kmpp_nmse_list = list()
    vq_vae_ema_nmse_list = list()
    vq_vae_ema_kmpp_nmse_list = list()
    
    for num_ebd in num_embeddings_list:
        vq_vae_weights_path = f"models/vq_vae/vq_vae_input_40_latent_20_num_embeddings_{num_ebd}_init_random_with_BN_False_RMSprop.h5"
        _, _, vq_vae_abs_deviation, vq_vae_nmse = vq_vae_test(x_test, input_dim, latent_dim, num_ebd, weights_path=vq_vae_weights_path)
        vq_vae_mse = np.mean(vq_vae_abs_deviation**2)
        vq_vae_mse_list.append(vq_vae_mse)
        vq_vae_nmse_list.append(vq_vae_nmse)
        tf.keras.backend.clear_session()
        
        vq_vae_kmpp_weights_path = f"models/vq_vae_kmpp_init/vq_vae_input_40_latent_20_num_embeddings_{num_ebd}_init_kmpp_with_BN_False_RMSprop.h5"
        _, _, vq_vae_kmpp_abs_deviation, vq_vae_kmpp_nmse = vq_vae_test(x_test, input_dim, latent_dim, num_ebd, weights_path=vq_vae_kmpp_weights_path)        
        vq_vae_kmpp_mse = np.mean(vq_vae_kmpp_abs_deviation**2)
        vq_vae_kmpp_mse_list.append(vq_vae_kmpp_mse)
        vq_vae_kmpp_nmse_list.append(vq_vae_kmpp_nmse)
        tf.keras.backend.clear_session()
            
        vq_vae_ema_path = f"models/vq_vae_ema/vq_vae_ema_input_40_latent_20_num_embeddings_{num_ebd}_init_random_ema_decay_0.99_beta_0.25.h5"
        _, _, vq_vae_ema_abs_deviation, vq_vae_ema_nmse = vq_vae_ema_test(x_test, input_dim, latent_dim, num_ebd, weights_path=vq_vae_ema_path)
        vq_vae_ema_mse = np.mean(vq_vae_ema_abs_deviation**2)
        vq_vae_ema_mse_list.append(vq_vae_ema_mse)
        vq_vae_ema_nmse_list.append(vq_vae_ema_nmse)
        tf.keras.backend.clear_session()
        
        vq_vae_ema_kmpp_path = f"models/vq_vae_ema_kmpp_init/vq_vae_ema_input_40_latent_20_num_embeddings_{num_ebd}_init_kmpp_ema_decay_0.99_beta_0.25.h5"
        _, _, vq_vae_ema_kmpp_abs_deviation, vq_vae_ema_kmpp_nmse = vq_vae_ema_test(x_test, input_dim, latent_dim, num_ebd, weights_path=vq_vae_ema_kmpp_path)
        vq_vae_ema_kmpp_mse = np.mean(vq_vae_ema_kmpp_abs_deviation**2)
        vq_vae_ema_kmpp_mse_list.append(vq_vae_ema_kmpp_mse)
        vq_vae_ema_kmpp_nmse_list.append(vq_vae_ema_kmpp_nmse)
        tf.keras.backend.clear_session()
    
    
    plt.figure()
    plt.plot(num_bits_list, vq_vae_nmse_list, "b-x", label="VQ-VAE")
    plt.plot(num_bits_list, vq_vae_kmpp_nmse_list, "m-+", label="VQ-VAE-KMPP")
    plt.plot(num_bits_list, vq_vae_ema_nmse_list, "g-o", label="VQ-VAE-EMA")
    plt.plot(num_bits_list, vq_vae_ema_kmpp_nmse_list, "r-s", label="VQ-VAE-EMA-KMPP")
    plt.xlabel("Number of quantization bits")
    plt.ylabel("NMSE")
    plt.grid()
    plt.legend()
    plt.savefig("figures/estimation/NMSE_num_quant_bits.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    
        
if __name__ == "__main__":
    # parameter_compare()
    algorithm_compare()