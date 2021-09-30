import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from utils import set_env, create_callbacks
from utils.data_helper import pre_processing
from models.text_cnn import TextCNN
from utils.evaluation import Evaluation



def load_data(train_dir, test_dir):
    train = pd.read_csv(train_dir)
    test = pd.read_csv(test_dir)

    train, val = train_test_split(train, test_size=0.1, random_state=42)

    train_x, train_y = train["text"], train["label"]
    test_x, test_y = test["text"], test["label"]
    val_x, val_y = val["text"], val["label"]

    # string to float
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    test_y = encoder.fit_transform(test_y)
    val_y = encoder.fit_transform(val_y)

    return train_x, train_y, test_x, test_y, val_x, val_y


# Directory Setting
train_dir = "../data/binary_train.csv"
test_dir = "../data/binary_test.csv"
model_dir = "../model_save"

# HyperParameter
epoch = 2
batch = 256
max_len = 300

# Flow
print("0. Setting Environment")
set_env()

print("1. load data")
train_x, train_y, test_x, test_y, val_x, val_y = load_data(train_dir, test_dir)

print("2. pre processing")
train_x, test_x, val_x, tokenizer = pre_processing(train_x, test_x, val_x, max_len)

print("3. build model")
model = TextCNN(
    sequence_len=train_x.shape[1],
    embedding_matrix=len(tokenizer.word_index) + 1,
    embedding_dim=300,
    filter_sizes=[3, 4, 5],
    flag="self_training",
    data_type="binary"
)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

callbacks = create_callbacks(model_dir)
model.fit(x=train_x, y=train_y, epochs=epoch, batch_size=batch, validation_data=(val_x, val_y), callbacks=callbacks)


# BentoML Serve
from service import KerasClassification
bento_svc = KerasClassification()
bento_svc.pack('model', model)
bento_svc.pack('tokenizer', tokenizer)

saved_path = bento_svc.save()