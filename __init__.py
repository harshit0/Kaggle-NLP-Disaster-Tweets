import pandas as pd
from keras.utils import to_categorical

from model import get_sequence_of_tokens, generate_padded_sequences, train, predict

df = pd.read_csv(".\\train.csv")

df.drop(["id", "location"], axis=1, inplace=True)
df = df.astype("str")

keyword = df.keyword.values
text = df.text.values

df.target = df.target.astype("int")
target = df.target.values
target = to_categorical(target, num_classes=2)
print(target)

num_keyword, seq_keyword = get_sequence_of_tokens(keyword)
num_text, seq_text = get_sequence_of_tokens(text)

input_seq_kw, max_seq_len_kw = generate_padded_sequences(seq_keyword)
input_seq_text, max_seq_len_text = generate_padded_sequences(seq_text)

train(vocab_length=num_text, max_input_len=max_seq_len_text, text=input_seq_text, target=target)

# -----------------------------------------TEST-----------------------------------------------

df_test = predict(pd.read_csv("test.csv"))

df_test.to_csv("test_output_256.csv", index=False)

df_test.drop(["text"], axis=1, inplace=True)

df_test.to_csv("submission_256.csv", index=False)
