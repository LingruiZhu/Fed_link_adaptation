import numpy as np
import matplotlib.pyplot as plt

from data_preprocessing import preprocess_train, preprocess_test, read_file
from lstm_model import build_lstm_predict_model


def prepare_data(num_inputs, num_outputs):
    data_file_path = "Interference_generation/interference_data/single_UE_data.h5"
    sinr_sequence, sinr_dB_sequence, interference_sequence = read_file(data_file_path)
    
    # Use sinr sequence to train model
    train_sinr_sequence, test_sinr_sequence = sinr_dB_sequence[:8000], sinr_dB_sequence[8000:]
    train_samples = preprocess_train(train_sinr_sequence, num_inputs=num_inputs, num_outputs=num_outputs, shuffle_samples=True)
    x_train, y_train = train_samples[:, :num_inputs], train_samples[:, num_inputs:]
    x_train = np.expand_dims(x_train, axis=-1)
    test_samples = preprocess_test(test_sinr_sequence, num_inputs=20, num_outputs=5, shuffle_samples=False)
    x_test, y_test = test_samples[:, :num_inputs], test_samples[:, num_inputs:]
    x_test = np.expand_dims(x_test, axis=-1)
    return x_train, y_train, x_test, y_test, test_sinr_sequence


def train_lstm_model():
    num_inputs = 40
    num_outputs = 10
    
    lstm_model = build_lstm_predict_model(num_inputs=num_inputs, num_hidden=128, num_outputs=num_outputs)
    
    x_train, y_train, x_test, y_test, test_sinr_sequence = prepare_data(num_inputs, num_outputs)
    
    lstm_model.fit(x_train, y_train, epochs=40, batch_size=64, verbose=2, )
    
    test_predicted = lstm_model.predict(x_test)
    prediction_sequence = test_predicted.flatten()
    
    plt.figure()
    plt.plot(test_sinr_sequence, "r", label="True SINR")
    plt.plot(prediction_sequence, "b", label="Prediction")
    plt.grid()
    plt.xlabel("Time")
    plt.ylabel("SINR in dB")
    plt.legend()
    plt.show()
    
    
if __name__ == "__main__":
    train_lstm_model() 