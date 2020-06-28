import pandas as pd
from keras.utils import to_categorical
import os
from utils import get_sequence_of_tokens, generate_padded_sequences, train, predict
from model import vanilla_lstm_model
from functional_model import create_lstm_model
import time
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
text = df.text.values

# convert target variable to vector representation
df.target = df.target.astype("int")
target = df.target.values

unique_target_values = df['target'].unique()
num_classes = len(unique_target_values)

target = to_categorical(target, num_classes=num_classes)
print(target)

# convert sequence of words to sequence of integers
keyword_list = None
num_text, seq_text = get_sequence_of_tokens(text, refresh=True)

# pad all the arrays to same length
input_seq_text, max_seq_len_text = generate_padded_sequences(seq_text)

# create model
vanilla_lstm_model = vanilla_lstm_model(vocab_length=num_text, max_input_len=max_seq_len_text, max_output_len=num_classes)

# lstm_model = create_lstm_model(vocab_length=num_text, max_input_len=max_seq_len_text, max_output_len=num_classes)


timestamp = int(time.time())
vanilla_model_name = 'vanilla_model-{}.h5'.format(timestamp)
vanilla_model_path = os.path.join(models_dir_path, vanilla_model_name)
train(vanilla_lstm_model, inputs=input_seq_text, target=target, model_path=vanilla_model_path)

# model_name = 'model-{}.h5'.format(timestamp)
# model_path = os.path.join(models_dir_path, model_name)
# train(lstm_model, [input_seq_text, seq_keyword], target, model_path=model_path)


# -----------------------------------------TEST-----------------------------------------------

df_test = predict(pd.read_csv(testing_file_path), vanilla_model_path)

df_test.to_csv("test_output_128-{}.csv".format(timestamp), index=False)

df_test.drop(["text"], axis=1, inplace=True)

df_test.to_csv("submission_128-{}.csv".format(timestamp), index=False)
