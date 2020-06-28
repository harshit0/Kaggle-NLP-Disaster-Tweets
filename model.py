import json

import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json


def generate_padded_sequences(input_sequences, max_sequence_len=0):
    """

    :param input_sequences:
    :return:
    """
    if not max_sequence_len:
        max_sequence_len = max([len(x) for x in input_sequences])
    print("max sequence length: {}".format(max_sequence_len))
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    return input_sequences, max_sequence_len


def get_sequence_of_tokens(corpus, refresh=True):
    """
    :param corpus:
    :param refresh:
    :return:
    """
    # tokenization
    if refresh:
        tokenizer = Tokenizer()
        # fit the tokenizer on the text
        tokenizer.fit_on_texts(corpus)
    else:
        with open("tokenizer.json", 'r') as tj:
            tokenizer = tokenizer_from_json(json.load(tj))

    tokenizer_json = tokenizer.to_json()

    with open('tokenizer.json', 'w') as fobj:
        json.dump(tokenizer_json, fobj)

    index_dict = tokenizer.word_index
    seq = tokenizer.texts_to_sequences(corpus)
    # calculate the vocab size
    total_words = len(tokenizer.word_index) + 1
    print(total_words)
    return total_words, seq


def lstm_model(vocab_length, max_input_len, max_output_len=2):
    sequential = Sequential()

    sequential.add(Embedding(input_dim=vocab_length, output_dim=256, input_length=max_input_len))

    sequential.add(Dropout(0.5))

    sequential.add(LSTM(units=200, activation='relu', return_sequences=True))

    sequential.add(Flatten())

    sequential.add(Dropout(0.5))

    sequential.add(Dense(units=max_output_len, activation='softmax'))

    sequential.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    sequential.summary()

    return sequential


def train(vocab_length, max_input_len, text, target):
    max_output_len = 2

    model = lstm_model(vocab_length, max_input_len, max_output_len)

    x = text
    y = target

    tb = TensorBoard(log_dir="./logs")

    model.fit(
        x, y,
        batch_size=64,
        epochs=10,
        verbose=1,
        validation_split=0,
        shuffle=True,
        callbacks=[tb]
    )

    model.save("disaster_tweets_model.h5")


def predict(df_test):
    df_test.drop(["keyword", "location"], axis=1, inplace=True)

    df_test = df_test.astype("str")

    test_text = df_test.text.values

    num_test_text, seq_test_text = get_sequence_of_tokens(test_text, refresh=False)

    input_seq_test_text, max_seq_len_test_text = generate_padded_sequences(seq_test_text, max_sequence_len=33)

    model = load_model("disaster_tweets_model.h5")

    predictions = model.predict(input_seq_test_text)

    predict_list = []
    for _1, _2 in predictions.tolist():
        predict_list.append(np.argmax([_1, _2]))

    df_test["target"] = predict_list

    return df_test
