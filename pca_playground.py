import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def plot_segments(segments):
    plt.figure(figsize=(8, 8))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    for i, segment in enumerate(segments):
        plt.scatter(segment[:, 0], segment[:, 1], label=f'Segment {i}', color=colors[i % len(colors)], alpha=0.2)
        
        # Calculate and plot centroid
        centroid = np.mean(segment, axis=0)
        plt.scatter(centroid[0], centroid[1], marker='o', color='black', s=100, label=f'Centroid {i}')

    plt.title('Data Points and Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


def geometric_split(data, num_segments):
    current_segments = [data]

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
            split_point = geometric_center # + 0.5 * principal_axis

            # Classify data points based on their position relative to the split point
            split_data_1 = segment[segment.dot(principal_axis) <= split_point.dot(principal_axis)]
            split_data_2 = segment[segment.dot(principal_axis) > split_point.dot(principal_axis)]

            next_segments.extend([split_data_1, split_data_2])

        current_segments = next_segments

    return current_segments

# Example usage
# Assuming 'your_data' is your dataset (N x M matrix)
your_data = np.random.rand(100, 2)  # Replace with your actual data
num_segments = 8  # Replace with the desired number of segments (power of 2)

result_segments = geometric_split(your_data, num_segments)
plot_segments(result_segments)