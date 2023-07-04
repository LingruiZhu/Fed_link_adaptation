import numpy as np

from Interference_prediction import data_preprocessing


def generate_cqi_data():
    sinr_file_path = "Interference_generation/interference_data/single_UE_data.h5"
    _, sinr_data, _ = data_preprocessing.read_file(sinr_file_path)
    


if __name__ =="__main__":
    generate_cqi_data()