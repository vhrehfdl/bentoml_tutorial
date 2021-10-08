import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import keras
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from models.text_cnn import TextCNN
from utils import set_env, create_callbacks
from utils.data_helper import pre_processing
from utils.evaluation import Evaluation

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM


# A dictionary mapping words to an integer index
def load_data(train_dir, test_dir, category_size):
    train = pd.read_csv(train_dir)
    test = pd.read_csv(test_dir)

    train, val = train_test_split(train, test_size=0.1, random_state=42)
    temp, train = train_test_split(train, test_size=0.1, random_state=42)

    train_x, train_y = train["turn3"], train["label"]
    test_x, test_y = test["turn3"], test["label"]
    val_x, val_y = val["turn3"], val["label"]

    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    test_y = encoder.fit_transform(test_y)
    val_y = encoder.fit_transform(val_y)

    train_y = keras.utils.to_categorical(train_y, category_size)
    val_y = keras.utils.to_categorical(val_y, category_size)

    return train_x, train_y, test_x, test_y, val_x, val_y, encoder


# Directory Setting
train_dir = "../data/multi_train.csv"
test_dir = "../data/multi_test.csv"
model_dir = "../model_save"



# HyperParameter
epoch = 1
batch = 256
max_len = 300
target_names = ['0', '1', '2', '3']

# Flow
# print("0. Setting Environment")
# set_env()

print("1. load data")
train_x, train_y, test_x, test_y, val_x, val_y, encoder = load_data(train_dir, test_dir, len(target_names))

print("2. pre processing")
train_x, test_x, val_x, tokenizer = pre_processing(train_x, test_x, val_x, max_len)
print(train_x[0])

model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(4, activation='softmax'))

model.summary()

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_x, train_y,
          batch_size=batch,
          epochs=2, # for demo purpose :P
          validation_data=(val_x, val_y))

# score, acc = model.evaluate(test_x, test_y, batch_size=batch)

# print('Test score:', score)
# print('Test accuracy:', acc)



# 1) import the custom BentoService defined above
from service import KerasClassification

# 2) `pack` it with required artifacts
bento_svc = KerasClassification()
bento_svc.pack('model', model)
# bento_svc.pack('word_index', word_index)

# 3) save your BentoSerivce
saved_path = bento_svc.save()