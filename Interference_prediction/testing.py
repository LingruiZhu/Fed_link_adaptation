import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from data_preprocessing import prepare_data


def test_model(model_path:str):
    model = load_model(model_path)
    
    _, _, x_test, y_test, test_sequence = prepare_data(num_inputs=40, num_outputs=10)
    
    y_prediction = model.predict(x_test)
    
    y_test_to_plot = y_test.flatten()
    y_prediction_to_plot = y_prediction.flatten()
    
    plt.figure()
    plt.plot(y_test_to_plot, "r", label="True SINR")
    plt.plot(y_prediction_to_plot, "b", label="Prediction")
    plt.grid()
    plt.xlabel("Time")
    plt.ylabel("SINR in dB")
    plt.legend()
    plt.show()
    
    
if __name__ == "__main__":
    test_model("Interference_prediction/models/lstm.h5")
    