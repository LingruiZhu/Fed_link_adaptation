import sys 
sys.path.append("/home/zhu/Codes/Fed_Link_Adaptation")

from Interference_prediction import data_preprocessing
from Interference_prediction.data_preprocessing import preprocess_test

import numpy as np
from sklearn.metrics import mean_squared_error as calc_mse


# def differential_quantization_single_sequence(sequence, initial_state, quantization_step, num_bits):
#     # Calculate the quantization step size based on the number of bits

#     # Lists to store the quantized differences and the reconstructed sequence
#     quantized_differences = []
#     reconstructed_sequence = []

#     # Perform differential quantization
#     previous_predicted_value = initial_state
#     for sample in sequence:
#         difference = sample - previous_predicted_value
#         quantized_difference = round(difference / quantization_step)
        
#         if quantized_difference > num_bits:
#             quantized_difference = num_bits
#         elif quantized_difference < -num_bits:
#             quantized_difference = -num_bits
        
#         reconstructed_sample = previous_predicted_value + quantized_difference * quantization_step

#         quantized_differences.append(quantized_difference)
#         reconstructed_sequence.append(reconstructed_sample)

#         # Update the previous predicted value
#         previous_predicted_value = reconstructed_sample

#     return quantized_differences, reconstructed_sequence


def differential_quantization_single_sequence(sequence, initial_state, quantization_step, num_bits, sampling_interval):
    # Calculate the quantization step size based on the number of bits

    # Lists to store the quantized differences and the reconstructed sequence
    quantized_differences = []
    reconstructed_sequence = []

    # Perform differential quantization with the specified sampling interval
    previous_predicted_value = initial_state
    for i, sample in enumerate(sequence):
        # Check if the current index is a multiple of the sampling interval
        if i % sampling_interval == 0:
            difference = sample - previous_predicted_value
            quantized_difference = round(difference / quantization_step)

            if quantized_difference > num_bits:
                quantized_difference = num_bits
            elif quantized_difference < -num_bits:
                quantized_difference = -num_bits

            reconstructed_sample = previous_predicted_value + quantized_difference * quantization_step

            quantized_differences.append(quantized_difference)
            reconstructed_sequence.append(reconstructed_sample)

            # Update the previous predicted value
            previous_predicted_value = reconstructed_sample

    # Duplicate the reconstructed values to match the length of the original sequence
    reconstructed_sequence_extended = []
    for i in range(len(sequence)):
        reconstructed_sequence_extended.append(reconstructed_sequence[i // sampling_interval])

    return quantized_differences, reconstructed_sequence_extended



# Example usage
_, _, _, _, sequence = data_preprocessing.prepare_data(num_inputs=40, num_outputs=10)
initial_state = sequence[0]  # Use the first element of the sequence as the initial value
num_bits = 1 # Number of quantization bits
quantization_step = 1     # 
sampling_interval = 2
num_inputs = 40
num_outputs = 10

quantized_diff, reconstructed_seq = differential_quantization_single_sequence(sequence, initial_state, quantization_step, num_bits, sampling_interval)
reconstructed_seq_samples = preprocess_test(reconstructed_seq, num_inputs, num_outputs)
original_seq_samples = preprocess_test(sequence, num_inputs, num_outputs)

reconstructed_seq_input, reconstructed_seq_output = reconstructed_seq_samples[:, :num_inputs], reconstructed_seq_samples[:, -num_outputs:]
original_seq_input, original_seq_output = original_seq_samples[:, :num_inputs], original_seq_samples[:, -num_outputs:]

nmse = calc_mse(reconstructed_seq_input, original_seq_input) / np.mean(original_seq_input)

# print("Original Sequence:", sequence)
# print("Quantized Differences:", quantized_diff)
# print("Reconstructed Sequence:", reconstructed_seq)

print("normalized MSE: ", nmse)

