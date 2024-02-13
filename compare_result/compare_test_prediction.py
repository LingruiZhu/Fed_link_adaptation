import sys
sys.path.append("/home/zhu/Codes/Fed_Link_Adaptation")
import numpy as np
from tensorflow.keras.models import load_model
from enum import Enum, auto
import os
import h5py

from compress_recover.VQVAEParams import VQVAEParams
from Interference_prediction import data_preprocessing
from compare_and_plot import vq_vae_test, vq_vae_ema_test, ae_uniform_quant_test
from model_evaluation import get_file_path


class ModelType(Enum):
    ae_uniform_quant = auto()
    diff_quant = auto()
    vq_vae = auto()
    vq_vae_ema = auto()


def predict_sinr_sequence(model_params:VQVAEParams, model_type:ModelType):
    """Models predict SINR sequence and save prediction as well as true data into h5 file.

    Args:
        file_path (str): file path of neural network parameters
        model_type (str): the type of autoencoder models
    """
    file_index=0
    file_path = get_file_path(model_params, file_index, return_list=False)
    _, _, x_test, y_test, _ = data_preprocessing.prepare_data(num_inputs=40, num_outputs=10)
    
    num_quant_bits = 6
    input_dims = 40
    
    latent_dims = model_params.latent_dims
    num_embeddings = model_params.num_embeddings
    
    if model_type == ModelType.ae_uniform_quant:
        recovered_sequence, recover_sequence_1d, _, _ = ae_uniform_quant_test(x_test, input_dims, latent_dims, num_quant_bits, file_path)
    elif model_type == ModelType.vq_vae:
        recovered_sequence, recover_sequence_1d, _, _ = vq_vae_test(x_test, input_dims, latent_dims, num_embeddings, file_path)
    elif model_type == ModelType.vq_vae_ema:
        recovered_sequence, recover_sequence_1d, _, _ = vq_vae_ema_test(x_test, input_dims, latent_dims, num_embeddings, file_path)
    # debugging code, delete after finish
    print(np.shape(recovered_sequence))
    print(np.shape(recover_sequence_1d))
    print(np.shape(y_test))
    
    lstm_model = load_model("Interference_prediction/models/lstm.h5")
    y_test_prediction = lstm_model.predict(recovered_sequence)
    print(np.shape(y_test_prediction))
    # TODO: put y_test and y_test_prediction in h5 file.
    
    y_test_1d = y_test.flatten()
    y_test_prediction_1d = y_test_prediction.flatten()
    
    # create file path to save y_test and y_test_pred sequence
    model_name = model_params.vq_update_type + "_" + model_params.embedding_init_method
    file_name = f"{model_name}_input_{model_params.input_dims}_latent_{model_params.latent_dims}_" + \
                f"num_embeddings_{model_params.num_embeddings}_init_{model_params.embedding_init_method}_" + \
                f"{model_params.optimizer}_{file_index}.h5"
    folder_path = os.path.join("prediction_sequence",
        model_name, 
        f"latent_dim_{model_params.latent_dims}",
        f"num_embeddings_{model_params.num_embeddings}")
    file_path = os.path.join(folder_path, file_name)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    
    with h5py.File(file_path, 'w') as f:
        f.create_dataset("SINR_real", data=y_test_1d)
        f.create_dataset("SINR_prediction", data=y_test_prediction_1d)
    

def test():
    sim_paras = VQVAEParams(
        autoencoder_type="dense",
        vq_update_type="embedding_loss",    # ema or embedding_loss
        input_dims=40,
        latent_dims=20,
        num_embeddings=64,
        optimizer="RMSprop",                # adam or RMSprop
        init_embedding_method="random",       # random or kmpp or pca
        num_epochs=300,
        
        init_epochs=100,
        re_init_interval=20,
        beta=0.25,
        ema_decay=0.99,
        plot_figure=False,
        num_models=10) 
    predict_sinr_sequence(sim_paras, ModelType.vq_vae)


if __name__ == "__main__":
    test()