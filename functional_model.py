import json

import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import concatenate


def create_lstm_model(vocab_length, max_input_len, max_output_len=2, keyword_input_len=1):
    text_input = Input(shape=(max_input_len, ))

    keyword_input = Input(shape=(keyword_input_len, ))

    embedding_out = Embedding(input_dim=vocab_length, output_dim=128, input_length=max_input_len)(text_input)

    lstm_1 = LSTM(
        units=200,
        activation='relu',
        return_sequences=True,
        dropout=0.4,
        recurrent_dropout=0.25
    )(embedding_out)

    flatten = Flatten()(lstm_1)

    inputs = concatenate([flatten, keyword_input])

    output = Dense(units=max_output_len, activation='softmax')(inputs)

    model = Model(inputs=[text_input, keyword_input], outputs=[output])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    return model


if __name__ == '__main__':
    create_lstm_model(22000, max_input_len=33, max_output_len=2)
