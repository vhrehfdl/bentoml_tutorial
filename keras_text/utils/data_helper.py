import pandas as pd
import numpy as np
import keras
from keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


class DataLoading:
    def __init__(self, train_dir, test_dir, test_size):
        train = pd.read_csv(train_dir)
        test = pd.read_csv(test_dir)

        train, val = train_test_split(train, test_size=test_size, random_state=42)

        self.train_x, self.train_y = train["text"], train["label"]
        self.val_x, self.val_y = val["text"], val["label"]
        self.test_x, self.test_y = test["text"], test["label"]

    def load_data(self, data_type, category_size=None):
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(self.train_y)
        val_y = encoder.fit_transform(self.val_y)
        test_y = encoder.fit_transform(self.test_y)

        target_names = encoder.classes_

        if data_type == "multi":
            train_y = keras.utils.to_categorical(train_y, category_size)
            val_y = keras.utils.to_categorical(val_y, category_size)

        return self.train_x, train_y, self.test_x, test_y, self.val_x, val_y, target_names


# convert Text data to vector.
def pre_processing(train_x, test_x, val_x):
    CHARS_TO_REMOVE = r'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'

    train_x = train_x.tolist()
    test_x = test_x.tolist()
    val_x = val_x.tolist()
    
    tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)
    tokenizer.fit_on_texts(train_x + test_x + val_x)  # Make dictionary

    # Text match to dictionary.
    train_x = tokenizer.texts_to_sequences(train_x)
    test_x = tokenizer.texts_to_sequences(test_x)
    val_x = tokenizer.texts_to_sequences(val_x)

    # total_list = list(train_x) + list(test_x) + list(val_x)
    # max_len = max([len(total_list[i]) for i in range(0, len(total_list))])
    max_len = 300

    train_x = sequence.pad_sequences(train_x, maxlen=max_len, padding='post')
    test_x = sequence.pad_sequences(test_x, maxlen=max_len, padding='post')
    val_x = sequence.pad_sequences(val_x, maxlen=max_len, padding='post')

    return train_x, test_x, val_x, tokenizer


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path, encoding="utf-8") as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


def text_to_vector(word_index, path, word_dimension):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, word_dimension))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass

    return embedding_matrix