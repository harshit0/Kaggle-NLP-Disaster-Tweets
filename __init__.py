import pandas as pd
from keras.utils import to_categorical
import os
from utils import get_sequence_of_tokens, generate_padded_sequences, train, predict
from model import vanilla_lstm_model
from functional_model import create_lstm_model, keyword_vector, predict_functional, create_lstm_model_1
import time
from utils import clean_text
from keras.preprocessing.text import text_to_word_sequence

BASE_DIR = os.path.dirname(__file__)

# define dataset dir and its path
dataset_dir = 'dataset'
models_dir = 'models'

dataset_dir_path = os.path.join(BASE_DIR, dataset_dir)
models_dir_path = os.path.join(BASE_DIR, models_dir)

training_file = 'train.csv'
testing_file = 'test.csv'

training_file_path = os.path.join(dataset_dir_path, training_file)
testing_file_path = os.path.join(dataset_dir_path, testing_file)


df = pd.read_csv(training_file_path)

df.drop(["id", "location"], axis=1, inplace=True)
df = df.astype("str")

keyword = df.keyword.values
unique_keywords = df['keyword'].unique()
# print(unique_keywords)
# exit(0)
df['text'] = clean_text(df['text'])
text = df.text.values

# convert target variable to vector representation
df.target = df.target.astype("int")
target = df.target.values

unique_target_values = df['target'].unique()
num_classes = len(unique_target_values)

target = to_categorical(target, num_classes=num_classes)
print(target)

# convert sequence of words to sequence of integers
keyword_list = [i.replace('%20', ' ').split() for i in keyword]
# print(keyword_list)

# tokenize
num_text, seq_text = get_sequence_of_tokens(text, refresh=False)

keywords = []
for i, j in zip(text, keyword_list):
    output_vector = keyword_vector(i, j)
    keywords.append(output_vector)
# pad all the arrays to same length
input_seq_text, max_seq_len_text = generate_padded_sequences(seq_text)
print(max_seq_len_text)
input_keyword_vector, max_seq_len_keyword = generate_padded_sequences(keywords, max_sequence_len=max_seq_len_text)


# create model
# vanilla_lstm_model = vanilla_lstm_model(vocab_length=num_text, max_input_len=max_seq_len_text, max_output_len=num_classes)

lstm_model = create_lstm_model_1(vocab_length=num_text, max_input_len=max_seq_len_text, max_output_len=num_classes, keyword_input_len=max_seq_len_text)


timestamp = int(time.time())
# vanilla_model_name = 'vanilla_model-{}.h5'.format(timestamp)
# vanilla_model_path = os.path.join(models_dir_path, vanilla_model_name)
# train(vanilla_lstm_model, inputs=input_seq_text, target=target, model_path=vanilla_model_path)

model_name = 'model-{}.h5'.format(timestamp)
model_path = os.path.join(models_dir_path, model_name)
train(lstm_model, [input_seq_text, input_keyword_vector], target, model_path=model_path, epochs=50)


# -----------------------------------------TEST-----------------------------------------------
df_test = predict_functional(pd.read_csv(testing_file_path), model_path, input_seq_len=max_seq_len_text)
# df_test = predict_functional(pd.read_csv(testing_file_path), 'models/model-1593452911.h5')

df_test.to_csv("test_output_128-{}.csv".format(timestamp), index=False)

df_test.drop(["text", 'keyword'], axis=1, inplace=True)

df_test.to_csv("submission_128-{}.csv".format(timestamp), index=False)


