from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.losses import MeanSquaredError

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape)) # <- Ini adalah inti dari neural network, Layer Long Short-Term Memory dengan 50 unit — ini adalah neural network utama
    model.add(Dense(1))
    model.compile(optimizer='adam', loss=MeanSquaredError())
    return model