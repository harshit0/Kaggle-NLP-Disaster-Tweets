import json

import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import LSTM, Bidirectional
from keras.layers import concatenate
from utils import get_sequence_of_tokens
from utils import generate_padded_sequences
from keras.models import load_model
from keras.optimizers import Adam


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


def create_lstm_model_1(vocab_length, max_input_len, max_output_len=2, keyword_input_len=1):
    text_input = Input(shape=(max_input_len, ))

    keyword_input = Input(shape=(keyword_input_len, ))

    embedding_out = Embedding(input_dim=vocab_length, output_dim=128, input_length=max_input_len)(text_input)

    lstm_1 = Bidirectional(LSTM(
        units=300,
        activation='relu',
        return_sequences=True,
        dropout=0.4,
        recurrent_dropout=0.25
    ))(embedding_out)

    lstm_2 = Bidirectional(LSTM(
        units=300,
        activation='relu',
        return_sequences=True,
        dropout=0.4,
        recurrent_dropout=0.25
    ))(lstm_1)

    flatten = Flatten()(lstm_2)

    dense_1 = Dense(units=512, activation='relu')(keyword_input)

    inputs = concatenate([flatten, dense_1])

    dense_2 = Dense(units=512, activation='relu')(inputs)

    dropout_1 = Dropout(0.25)(dense_2)

    output = Dense(units=max_output_len, activation='softmax')(dropout_1)

    model = Model(inputs=[text_input, keyword_input], outputs=[output])

    model.compile(
        # rmsprop
        optimizer=Adam(learning_rate=0.00005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    return model


def predict_functional(df_test, model_path, input_seq_len):
    df_test.drop(["location"], axis=1, inplace=True)

    df_test = df_test.astype("str")
    keyword = df_test.keyword.values
    keyword_list = [i.replace('%20', ' ').split() for i in keyword]

    test_text = df_test.text.values

    keywords = []
    for i, j in zip(test_text, keyword_list):
        output_vector = keyword_vector(i, j)
        keywords.append(output_vector)

    num_test_text, seq_test_text = get_sequence_of_tokens(test_text, refresh=False)

    input_seq_test_text, max_seq_len_test_text = generate_padded_sequences(seq_test_text, max_sequence_len=input_seq_len)
    input_keyword_vector, max_seq_len_keyword = generate_padded_sequences(keywords, max_sequence_len=input_seq_len)

    model = load_model(model_path)

    print(len(input_seq_test_text), len(input_keyword_vector))

    predictions = model.predict([input_seq_test_text, input_keyword_vector])

    predict_list = []
    for _1, _2 in predictions.tolist():
        predict_list.append(np.argmax([_1, _2]))

    df_test["target"] = predict_list

    return df_test


def keyword_vector(text, keyword):
    vector = []
    for i in text:
        if i in keyword:
            vector.append(1)
        else:
            vector.append(0)
    return vector


if __name__ == '__main__':
    create_lstm_model_1(22000, max_input_len=33, max_output_len=2, keyword_input_len=33)
