import numpy as np
from sklearn.decomposition import PCA


def pca_split_initialization(data, k):
    current_segments = [data]
    num_segments = k
    
    while len(current_segments) < num_segments:
        next_segments = []
        for segment in current_segments:
            # Step 1: Calculate the geometric center
            geometric_center = np.mean(segment, axis=0)

            # Step 2: Perform PCA to find the principal axis
            pca = PCA(n_components=segment.shape[1])
            pca.fit(segment)
            principal_axis = pca.components_[0]

            # Step 3: Split along the principal axis
            split_point = geometric_center 

            # Classify data points based on their position relative to the split point
            split_data_1 = segment[segment.dot(principal_axis) <= split_point.dot(principal_axis)]
            split_data_2 = segment[segment.dot(principal_axis) > split_point.dot(principal_axis)]

            next_segments.extend([split_data_1, split_data_2])

        current_segments = next_segments

    centroids_list = list()
    for segments in current_segments:
        centroids_list.append(np.mean(segments, axis=0))
    centroids = np.array(centroids_list)
    centroids = np.transpose(centroids)
    return centroids
