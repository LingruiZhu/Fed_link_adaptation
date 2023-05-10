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


def preprocess(original_data:np.array, num_inputs:int, num_outputs:int, shuffle_samples:bool=False) -> np.ndarray:
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
    

if __name__ == "__main__":
    file_path = "Interference_generation/interference_data/single_UE_data.h5"
    sinr_data, sinr_dB_data, interference_data = read_file(file_path)
    print(np.shape(sinr_data)[0])
    print(np.size(sinr_data))
    
    