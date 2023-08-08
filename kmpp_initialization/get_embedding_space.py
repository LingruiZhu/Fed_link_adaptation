import sys 
sys.path.append("/home/zhu/Codes/Fed_Link_Adaptation")

import numpy as np
import tensorflow as tf
import h5py
import os 

from tensorflow.keras.models import load_model, Model

from Interference_prediction import data_preprocessing




def get_embedding_space():
    ae_model_path = "models/ae_models/vq_vae_ema_input_40_latent_20_optimizer_RMSprop.h5"
    autoencoder = load_model(ae_model_path)
    x_train, _, x_test, _, _ = data_preprocessing.prepare_data(num_inputs=40, num_outputs=10)
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[2].output)
    latent_variable = encoder.predict(x_train)
    
    latent_variable_path = os.path.join("kmpp_initialization", "latent_space.h5")
    with h5py.File(latent_variable_path, "w") as hf:
        hf.create_dataset("latent_variables", data=latent_variable)
    
    
    
    


if __name__ == "__main__":
    get_embedding_space()    