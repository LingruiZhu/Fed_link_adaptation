import tensorflow as tf
import numpy as np


def get_code_indices(inputs, embeddings):
    # Calculate L2-normalized distance between the inputs and the codes.
    similarity = tf.matmul(inputs, embeddings)
    distances = (
        tf.reduce_sum(inputs**2, axis=1, keepdims=True)
        + tf.reduce_sum(embeddings**2, axis=0)
        - 2 * similarity
    )
    # Derive the indices for minimum distances.
    encoding_indices = tf.argmin(distances, axis=1)
    return encoding_indices


if __name__ == '__main__':
    embeddings_array = np.array(
        [[1, -1],
         [1, -1]], dtype=float
    )
    embeddings_tf = tf.Variable(initial_value=embeddings_array)
    inputs_array = np.array(
        [[2,2],
         [3,3],
         [-1,0]], dtype=float
    )
    inputs_tf = tf.Variable(initial_value=inputs_array)
    print(inputs_tf)
        
    indices = get_code_indices(inputs_tf, embeddings_tf)
    
    counts_array = np.array([2,3], dtype=float)
    counts_test = tf.Variable(initial_value=counts_array) 

    mat_result = tf.matmul(inputs_tf, embeddings_tf)
    print(mat_result)
    conditions = tf.Variable(initial_value=[True,False])
    tf_where_test = tf.where(conditions,
                             x = mat_result / counts_test,
                             y = inputs_tf)
    print(tf_where_test)