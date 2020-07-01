import json

import numpy as np
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
import re


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


def train(model, inputs, target, model_path, epochs):

    x = inputs
    y = target

    tb = TensorBoard(log_dir="./logs")

    model.fit(
        x, y,
        batch_size=64,
        epochs=epochs,
        verbose=1,
        validation_split=0,
        shuffle=True,
        callbacks=[tb]
    )

    model.save(model_path)


def predict(df_test, model_path):
    df_test.drop(["keyword", "location"], axis=1, inplace=True)

    df_test = df_test.astype("str")

    test_text = df_test.text.values

    num_test_text, seq_test_text = get_sequence_of_tokens(test_text, refresh=False)

    input_seq_test_text, max_seq_len_test_text = generate_padded_sequences(seq_test_text, max_sequence_len=33)

    model = load_model(model_path)

    predictions = model.predict(input_seq_test_text)

    predict_list = []
    for _1, _2 in predictions.tolist():
        predict_list.append(np.argmax([_1, _2]))

    df_test["target"] = predict_list

    return df_test


def clean_text(texts):
    alpha_num_values = re.compile(r'([a-z]*[0-9]+)+')

    texts = [re.sub("@[\w]+ |http[:\/\w\.]+", "", i) for i in texts]

    texts = [re.sub(alpha_num_values, '', i) for i in texts]

    return texts


if __name__ == '__main__':
    pass
