import json

import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import load_model


def vanilla_lstm_model(vocab_length, max_input_len, max_output_len=2):
    sequential = Sequential()

    sequential.add(Embedding(input_dim=vocab_length, output_dim=128, input_length=max_input_len))

    # sequential.add(Dropout(0.5))

    sequential.add(LSTM(units=200, activation='relu', return_sequences=True, dropout=0.5, recurrent_dropout=0.25))

    sequential.add(LSTM(units=200, activation='relu', return_sequences=True, dropout=0.5, recurrent_dropout=0.25))

    sequential.add(LSTM(units=200, activation='relu', return_sequences=True, dropout=0.5, recurrent_dropout=0.25))

    sequential.add(LSTM(units=200, activation='relu', return_sequences=True, dropout=0.5, recurrent_dropout=0.25))

    sequential.add(Flatten())

    # sequential.add(Dropout(0.5))

    sequential.add(Dense(units=max_output_len, activation='softmax'))

    sequential.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    sequential.summary()

    return sequential


if __name__ == '__main__':
    pass
