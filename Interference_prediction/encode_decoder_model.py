from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model


def build_encoder_decoder_model(input_dim, output_dim, num_units):
    encoder_inputs = Input(shape=(input_dim, 1))
    
    encoder = LSTM(num_units, return_state=True)
    _, state_h, state_c = encoder(encoder_inputs)
    
    encoder_states = [state_h, state_c]
    
    decoder_inputs = Input(shape=(output_dim,1))
    decoder = LSTM(num_units)
    decoder_outputs = decoder(decoder_inputs, initial_state=encoder_states)
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()
    return model



if __name__ == "__main__":
    en_de_model = encoder_decoder_model(input_dim=40, output_dim=10, num_units=128)
        