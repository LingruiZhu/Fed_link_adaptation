import numpy as np
import h5py


def kmeans_plusplus_initialization(data, k):    
    n_samples, n_features = data.shape
    centroids = np.empty((k, n_features))
    
    # Choose the first centroid randomly from the data
    first_centroid_index = np.random.choice(n_samples)
    centroids[0] = data[first_centroid_index]
    
    # Calculate the distances to the initial centroid
    distances = np.linalg.norm(data - centroids[0], axis=1)
    
    for i in range(1, k):
        # Choose the next centroid based on distances squared
        probabilities = (distances ** 2) / np.sum(distances ** 2)
        next_centroid_index = np.random.choice(n_samples, p=probabilities)
        centroids[i] = data[next_centroid_index]
        
        # Update distances with the new centroid
        new_distances = np.linalg.norm(data - centroids[i], axis=1)
        distances = np.minimum(distances, new_distances)
    centroids = np.transpose(centroids)
    return centroids


if __name__ == "__main__":
    latent_variable_file = "kmpp_initialization/latent_space.h5"
    with h5py.File(latent_variable_file, "r") as hf:
        latent_variables = hf["latent_variables"][:]
    centroids = kmeans_plusplus_initialization(latent_variables, 16)
    print(np.shape(centroids))