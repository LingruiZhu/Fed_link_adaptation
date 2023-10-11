import os
import sys
sys.path.append("/home/zhu/Codes/Fed_Link_Adaptation")

from Interference_prediction import data_preprocessing

import h5py
import numpy as np
from sklearn.cluster import KMeans
# from scipy.cluster.vq import kmeans2


def lloyd_max_vq_train(data, num_clusters, initialization:str="random"):
    # Create a K-Means model with the specified number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, init=initialization)
    # Fit the model to the data
    kmeans.fit(data)
    # Predict cluster labels for each data point
    labels = kmeans.predict(data)

    # Define file name
    file_name = f"lloyd_nax_num_embeedintg_{num_clusters}_init_{initialization}.h5"
    if initialization == "random":
        file_path = os.path.join("models", "lloyd_max", file_name)
    elif initialization == "kmpp":
        file_path = os.path.join("models", "lloyd_max_kmpp", file_name)
    
    # Get the cluster centroids
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    with h5py.File(file_path, "w") as file:
        file.create_dataset("centroids", data=centroids)
        file.create_dataset("labels", data=labels)


# Function to recover the original sequences from quantized data
def recover_sequences(quantized_sequences, centroids):
    recovered_sequences = [centroids[np.argmin(np.linalg.norm(centroids - seq, axis=1))] for seq in quantized_sequences]

    return recovered_sequences

# Example usage:
def lloyd_max_vq_test(test_data, model_path, num_clusters):
    # Read centroids from file
    with h5py.File(model_path, "r") as file:
        centroids = file["centroids"][:]
        labels = file["labels"][:]
    kmeans = KMeans(n_clusters=num_clusters, init=centroids, n_init=1)
    kmeans.labels_ = labels
    kmeans.cluster_centers_ = centroids
    predict_labels = kmeans.predict(test_data)
    recovered_sequences = [centroids[label] for label in predict_labels]
    return recovered_sequences
    
        

if __name__ == "__main__":
    x_train, y_train, x_test, y_test, _ = data_preprocessing.prepare_data(num_inputs=40, num_outputs=10)
    data = np.squeeze(x_train)
    
    # Number of clusters for vector quantization
    num_clusters = 128  # You can adjust this to change the quantization precision
    initialization = "random"
    # lloyd_max_vq_train(data, num_clusters, initialization)
    
    test_data = np.squeeze(x_test)
    file_path = "models/lloyd_max/lloyd_nax_num_embeedintg_128_init_random.h5"
    recovered_sequence = lloyd_max_vq_test(test_data, file_path, num_clusters)
    print(np.shape(recovered_sequence))
    print(np.shape(test_data))
