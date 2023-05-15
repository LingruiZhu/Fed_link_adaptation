import numpy as np
import h5py
from typing import Tuple

def read_file(file_path:str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    """read interference and SINR sequence from file

    Args:
        file_path (str): the path of file containing interfernce and sinr data

    Returns:
    """
    
    # start of the code
    data_file = h5py.File(file_path, "r")
    sinr_data = np.array(data_file.get("SINR"))
    sinr_dB_data = np.array(data_file.get("SINR_dB"))
    interference_data = np.array(data_file.get("Interference_power"))
    return sinr_data, sinr_dB_data, interference_data


# TODO: here need to distinguish the preprocessing for train and test
def preprocess_train(original_data:np.array, num_inputs:int, num_outputs:int, shuffle_samples:bool=False) -> np.ndarray:
    """this function help us convert 1-D sequense to 2-D data_samples

    Args:
        original_data (np.array): _description_
    """
    
    # start of the code
    sliding_window_length = num_inputs + num_outputs
    data_length = np.shape(original_data)[0]
    num_samples = data_length - sliding_window_length + 1
    data_sample_list = list()
    for i in range(num_samples):
        data_sample_list.append(original_data[i:i+sliding_window_length])
    data_sample = np.array(data_sample_list)
    if shuffle_samples:
        random_indices = np.random.permutation(data_sample.shape[0])
        data_sample = data_sample[random_indices]
    return data_sample


def preprocess_test(original_data:np.array, num_inputs:int, num_outputs:int, shuffle_samples:bool=False) -> np.ndarray:
    """this function help us convert 1-D sequense to 2-D data_samples

    Args:
        original_data (np.array): _description_
    """
    
    # start of the code
    sliding_window_length = num_inputs + num_outputs
    data_length = np.shape(original_data)[0]
    num_samples = int(data_length/sliding_window_length)
    data_sample_list = list()
    for i in range(num_samples):
        data_sample_list.append(original_data[i*sliding_window_length:(i+1) * sliding_window_length])
    data_sample = np.array(data_sample_list)
    if shuffle_samples:
        random_indices = np.random.permutation(data_sample.shape[0])
        data_sample = data_sample[random_indices]
    return data_sample
    
    
def prepare_data(num_inputs, num_outputs):
    data_file_path = "Interference_generation/interference_data/single_UE_data.h5"
    sinr_sequence, sinr_dB_sequence, interference_sequence = read_file(data_file_path)
    
    # Use sinr sequence to train model
    train_sinr_sequence, test_sinr_sequence = sinr_dB_sequence[:8000], sinr_dB_sequence[8000:]
    train_samples = preprocess_train(train_sinr_sequence, num_inputs=num_inputs, num_outputs=num_outputs, shuffle_samples=True)
    x_train, y_train = train_samples[:, :num_inputs], train_samples[:, num_inputs:]
    x_train = np.expand_dims(x_train, axis=-1)
    test_samples = preprocess_test(test_sinr_sequence, num_inputs=num_inputs, num_outputs=num_outputs, shuffle_samples=False)
    x_test, y_test = test_samples[:, :num_inputs], test_samples[:, num_inputs:]
    x_test = np.expand_dims(x_test, axis=-1)
    return x_train, y_train, x_test, y_test, test_sinr_sequence

if __name__ == "__main__":
    file_path = "Interference_generation/interference_data/single_UE_data.h5"
    sinr_data, sinr_dB_data, interference_data = read_file(file_path)
    print(np.shape(sinr_data)[0])
    print(np.size(sinr_data))
    
    