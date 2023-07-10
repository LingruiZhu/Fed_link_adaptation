'''
Filename: /home/zhu/Codes/Fed_Link_Adaptation/latent_PCA_analysis.py
Path: /home/zhu/Codes/Fed_Link_Adaptation
Created Date: May 31 2023
Author: zhu

Copyright (c) 2023 Lingrui Zhu @ Uni Bremen
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from compress_recover.vq_vae import create_quantized_autoencoder
from Interference_prediction import data_preprocessing


def vae_latent_pca_analyze():
    x_train, _, x_test, _, _ = data_preprocessing.prepare_data(num_inputs=40, num_outputs=10)
    
    vq_vae = create_quantized_autoencoder(input_dim=40, latent_dim=10, output_dim=40)
    vq_vae.load_weights("models/vq_vae_models/vq_vae_input_40_latent_10_num_embeddings_128.h5")
    vq_vae.summary()
    
    encoder_input = Input(shape=(40,))
    encoder_output = vq_vae.get_layer("encoder")(encoder_input)
    encoder = Model(inputs=encoder_input, outputs=encoder_output)

    # latent_variable_quant = encoder_quantizer.predict(x_train)
    latent_variables = encoder.predict(x_train)

    embeddings = vq_vae.get_layer("vector_quantizer").embeddings
    embeddings = tf.transpose(embeddings)
    nbrs = NearestNeighbors(n_neighbors=1).fit(embeddings)
    distances, nearest_indices = nbrs.kneighbors(latent_variables)
    
    pca = PCA(n_components=2)  # Specify the number of components as 3 for 3D plot
    latent_pca = pca.fit_transform(latent_variables)
    
    unique_indices = np.unique(nearest_indices)
    
    # plot preparation
    num_clusters_to_plot = 20
    random_plot_indices = random.sample(range(len(unique_indices)), num_clusters_to_plot)
    
    # Generate a color map with unique colors
    color_map = plt.cm.get_cmap('Set3', num_clusters_to_plot)

    # Generate a list of unique colors
    colors= [color_map(i) for i in range(num_clusters_to_plot)]
    
    
    plt.figure()
    # Create a scatter plot
    for i, index in enumerate(unique_indices[random_plot_indices]):
        # Select the latent variables corresponding to the current index
        indices = np.where(nearest_indices == index)[0]
        latent_vars_selected = latent_pca[indices]

        # Plot the selected latent variables with the assigned color
        plt.scatter(latent_vars_selected[:, 0], latent_vars_selected[:, 1], color=colors[i], label=f'Cluster {i}')

    # Add labels and title
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Latent Variables Visualization')

    # Add a legend
    plt.legend()
    
    pca_3d = PCA(n_components=3)
    latent_pca_3d = pca_3d.fit_transform(latent_variables)

    nbrs = NearestNeighbors(n_neighbors=1).fit(embeddings)
    distances, nearest_indices = nbrs.kneighbors(latent_variables)


    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the latent variables with the assigned color
    for i, index in enumerate(unique_indices[random_plot_indices]):
        # Select the latent variables corresponding to the current index
        indices = np.where(nearest_indices == index)[0]
        latent_vars_selected = latent_pca_3d[indices]

        # Plot the selected latent variables with the assigned color
        ax.scatter(latent_vars_selected[:, 0], latent_vars_selected[:, 1], latent_vars_selected[:, 2], color=colors[i], label=f'Cluster {i}')

    # Add labels and title
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('Latent Variables Visualization')

    # Add a legend
    ax.legend()
    
    # Show the plot
    plt.show()
    
    
if __name__ == "__main__":
    vae_latent_pca_analyze()