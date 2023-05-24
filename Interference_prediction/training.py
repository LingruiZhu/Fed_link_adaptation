import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

from Interference_prediction.data_preprocessing import preprocess_train, preprocess_test, read_file, prepare_data
from lstm_model import build_lstm_predict_model
from encode_decoder_model import build_encoder_decoder_model



def train_lstm_model():
    num_inputs = 40
    num_outputs = 10
    lstm_model = build_lstm_predict_model(num_inputs=num_inputs, num_hidden=128, num_outputs=num_outputs)
    x_train, y_train, x_test, y_test, test_sinr_sequence = prepare_data(num_inputs, num_outputs)
    
    lstm_model.fit(x_train, y_train, epochs=40, batch_size=64, verbose=2)
    lstm_model.save("Interference_prediction/models/lstm.h5")
    
    # test_predicted = lstm_model.predict(x_test)
    # prediction_sequence = test_predicted.flatten()
    
    # plt.figure()
    # plt.plot(test_sinr_sequence, "r", label="True SINR")
    # plt.plot(prediction_sequence, "b", label="Prediction")
    # plt.grid()
    # plt.xlabel("Time")
    # plt.ylabel("SINR in dB")
    # plt.legend()
    # plt.show()


def train_encoder_decoder_model():
    num_inputs = 40
    num_outputs = 10
    encoder_decoder_model = build_encoder_decoder_model(num_inputs, num_outputs, num_units=128)
    
    # data preparation
    x_train, y_train, x_test, y_test, test_sinr_sequence = prepare_data(num_inputs, num_outputs)
    encoder_input = x_train
    decoder_output = y_train
    decoder_input = np.zeros_like(decoder_output)
    decoder_input[:, 1:] = decoder_output[:, :-1]
    
    encoder_decoder_model.fit([encoder_input, decoder_input], decoder_output, batch_size=128, epochs=40)
    encoder_decoder_model.save("Interference_prediction/models/encoder_decoder.h5")
    
    
if __name__ == "__main__":
    # train_lstm_model() 
    train_encoder_decoder_model()