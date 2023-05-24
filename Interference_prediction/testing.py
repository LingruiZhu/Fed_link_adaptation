import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from Interference_prediction.data_preprocessing import prepare_data, prepare_data_for_encoder_decoder_test


def test_model(model_path:str, plot_result:bool=False):
    model = load_model(model_path)
    
    _, _, x_test, y_test, test_sequence = prepare_data(num_inputs=40, num_outputs=10)
    
    y_prediction = model.predict(x_test)
    
    y_test_to_plot = y_test.flatten()
    y_prediction_to_plot = y_prediction.flatten()
    
    if plot_result:
        plt.figure()
        plt.plot(y_test_to_plot, "r", label="True SINR")
        plt.plot(y_prediction_to_plot, "b", label="Prediction")
        plt.grid()
        plt.xlabel("Time")
        plt.ylabel("SINR in dB")
        plt.legend()
        plt.show()
    return y_prediction_to_plot, y_test_to_plot
    

def test_encoder_decoder_model(plot_result:bool=False):
    num_inputs = 40
    num_outputs = 10
    model = load_model("Interference_prediction/models/encoder_decoder.h5")
    extended_data_samples = prepare_data_for_encoder_decoder_test(num_inputs=num_inputs, num_outputs=num_outputs)
    y_test = extended_data_samples[:, -num_outputs:]
    
    num_test_samples = np.shape(extended_data_samples)[0]
    prediction_matrix = np.zeros((num_test_samples, num_outputs)) 
    for i in range(num_test_samples):
        data_sample_pool = extended_data_samples[i,:]
        decoder_input_non_zero_list = data_sample_pool[num_inputs - (num_outputs-1):num_inputs].tolist()   
        for j in range(num_outputs):  # todo: in the for loop try to finish the test phase.
            encoder_input = np.expand_dims(data_sample_pool[j:j+num_inputs], axis=-1)
            decoder_input_list = [0] + decoder_input_non_zero_list
            decoder_input = np.array(decoder_input_list)
            
            encoder_input = np.expand_dims(encoder_input, axis=0)
            decoder_input = np.expand_dims(decoder_input, axis=0)
            
            current_prediction = model.predict([encoder_input, decoder_input])
            prediction_matrix[i,j] = current_prediction[:,-1]     # save the last element of prediction to matrix
            decoder_input_non_zero_list.append(current_prediction[0,-1])
            decoder_input_non_zero_list.pop(0)
    prediction_sequence = prediction_matrix.flatten()
    y_test_sequence = y_test.flatten()
    
    if plot_result:
        plt.figure()
        plt.plot(y_test_sequence, "r", label="True SINR")
        plt.plot(prediction_sequence, "b", label="Prediction")
        plt.grid()
        plt.xlabel("Time")
        plt.ylabel("SINR in dB")
        plt.legend()
        plt.show()
    return prediction_sequence, y_test_sequence


def test_encoder_decoder_model_simple(plot_result:bool=False):
    num_inputs = 40
    num_outputs = 10
    model = load_model("Interference_prediction/models/encoder_decoder.h5")

    _, _, x_test, y_test, test_sequence = prepare_data(num_inputs=40, num_outputs=10)
    
    num_samples = np.shape(x_test)[0]
    x_test_decoder_input = np.zeros((num_samples, num_outputs))
    y_pred = model.predict([x_test, x_test_decoder_input])
    y_pred_sequence = y_pred.flatten()
    
    y_test_sequence = y_test.flatten()
    
    if plot_result:
        plt.figure()
        plt.plot(y_test_sequence, "r", label="True SINR")
        plt.plot(y_pred_sequence, "b", label="Prediction")
        plt.grid()
        plt.xlabel("Time")
        plt.ylabel("SINR in dB")
        plt.legend()
        plt.show()
    return y_pred_sequence, y_test_sequence
    
    
if __name__ == "__main__":
    lstm_pred, ture_sinr_lstm = test_model("Interference_prediction/models/lstm.h5")
    encoder_decoder_pred, ture_sinr_ende = test_encoder_decoder_model_simple()
    
    plt.figure()
    plt.plot(ture_sinr_lstm, "r-x", label="True SINR - LSTM")
    plt.plot(lstm_pred, "b-s", label="Prediction - LSTM")
    plt.plot(ture_sinr_ende, "m-+", label="True SINR - encoder decoder")
    plt.plot(encoder_decoder_pred, "g-v", label="Prediction - encoder decoder")
    plt.grid()
    plt.xlabel("Time")
    plt.ylabel("SINR in dB")
    plt.legend()
    plt.show()
    