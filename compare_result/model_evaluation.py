import sys 
sys.path.append("/home/zhu/Codes/Fed_Link_Adaptation")
import os

import numpy as np

from tensorflow.keras.models import load_model

from sklearn.metrics import mean_squared_error

from compress_recover.VQVAEParams import VQVAEParams
from Interference_prediction import data_preprocessing
from compress_recover.vq_vae import create_quantized_autoencoder
from compress_recover.vq_vae_ema import create_quantized_autoencoder_EMA


def get_file_path(params:VQVAEParams, file_index:int=None, return_list:bool=True):
    folder_name_1st = "VQ_VAE"
    if params.vq_update_type.lower() == "ema":
        folder_name_1st = folder_name_1st + "_EMA"
    if params.embedding_init_method.lower() == "kmpp":
        folder_name_1st = folder_name_1st + "_KMPP"
    if params.embedding_init_method.lower() == "pca":
        folder_name_1st = folder_name_1st + "_PCA"
    folder_name_1st = folder_name_1st + "_models"
    
    folder_name_2nd = f"latent_dim_{params.latent_dims}"
    folder_name_3rd = f"num_embeddings_{params.num_embeddings}"
    
    file_path_list = list()
    folder_path = os.path.join("models_new", folder_name_1st, folder_name_2nd, folder_name_3rd)
    for foldername, subfolders, file_names in os.walk(folder_path):
        for file_name in file_names:
            relative_path = os.path.relpath(os.path.join(folder_path, file_name))
            file_path_list.append(relative_path)
    if return_list:
        return file_path_list
    else:
        index_in_list = len(file_names)- 1 - file_index
        return file_path_list[index_in_list]


def evaluate_multiple_models(params:VQVAEParams):
    x_train, y_train, x_test, y_test, _ = data_preprocessing.prepare_data(num_inputs=40, num_outputs=10)
    file_path_list = get_file_path(params)
    mse_list = list()
    mse_prediction_list = list()
    for file_path in file_path_list:
        if params.vq_update_type.lower() == "embedding_loss":
            model = create_quantized_autoencoder(type=params.autoencoder_type,
                                                 input_dim=params.input_dims,
                                                 latent_dim=params.latent_dims,
                                                 output_dim=params.input_dims,
                                                 num_embeddings=params.num_embeddings)
        elif params.vq_update_type.lower() == "ema":
            model = create_quantized_autoencoder_EMA(model_type=params.autoencoder_type,
                                                     input_dim=params.input_dims,
                                                     latent_dim=params.latent_dims,
                                                     output_dim=params.input_dims,
                                                     num_embeddings=params.num_embeddings)
        model.load_weights(file_path)
        x_test_prediction = model.predict(x_test)
        x_test_squeezed = np.squeeze(x_test)
        mse = mean_squared_error(x_test_squeezed, x_test_prediction)
        mse_list.append(mse)
        
        # Prediction
        lstm_model = load_model("Interference_prediction/models/lstm.h5")
        y_test_prediction = lstm_model.predict(x_test_prediction)
        mse_prediction = mean_squared_error(y_test_prediction, y_test)
        mse_prediction_list.append(mse_prediction)
        
    nmse_mean = np.mean(mse_list) / np.mean(x_test)
    nmse_std = np.std(mse_list / np.mean(x_test))
    
    nmse_mean_pred = np.mean(mse_prediction_list) / np.mean(y_test)
    nmse_std_pred = np.std(mse_prediction_list / np.mean(y_test))
    return nmse_mean, nmse_std, nmse_mean_pred, nmse_std_pred


def main():
    simulation_parameters = VQVAEParams(
        autoencoder_type="dense",
        vq_update_type="ema",    # ema or embedding_loss
        input_dims=40,
        latent_dims=20,
        num_embeddings=64,
        optimizer="RMSprop",                # adam or RMSprop
        init_embedding_method="pca",       # random or kmpp or pca
        num_epochs=300,
        
        init_epochs=100,
        re_init_interval=20,
        beta=0.25,
        ema_decay=0.99,
        plot_figure=False,
        num_models=10)   

    nmse_mean, nmse_std, nmse_mean_pred, nmse_std_pred = evaluate_multiple_models(simulation_parameters)
    print(nmse_mean)
    print(nmse_std)
    print(nmse_mean_pred)
    print(nmse_std_pred)

if __name__ == "__main__":            
    main()
    