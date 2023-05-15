from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam


def build_lstm_predict_model(num_inputs:int, num_hidden:int, num_outputs:int):
    """Generate a 3-layer LSTM model to predict channel quality or interfernce power

    Args:
        num_inputs (int): number of neurons in the input layer
        num_hidden (int): number of neurons in the hidden layer
        num_outputs (int): number of neourons in the output layer
    """
    lstm_model = Sequential()
    lstm_model.add(LSTM(num_hidden, input_shape=(num_inputs, 1)))
    lstm_model.add(Dense(num_outputs))
    adam = Adam()
    lstm_model.compile(loss="mean_squared_error", optimizer=adam)
    return lstm_model


   
