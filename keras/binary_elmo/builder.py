import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from models.elmo import ELMo
from utils import set_env, create_callbacks
from utils.evaluation import Evaluation


def load_data(train_dir, test_dir):
    train = pd.read_csv(train_dir)
    test = pd.read_csv(test_dir)

    train, val = train_test_split(train, test_size=0.1, random_state=42)
    train, temp = train_test_split(train, test_size=0.9, random_state=42)

    train_x, train_y = train["text"], train["label"]
    test_x, test_y = test["text"], test["label"]
    val_x, val_y = val["text"], val["label"]

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
max_len = 50
epoch = 2
batch = 256
hidden_units = 256

# Flow
print("0. Setting Environment")
set_env()

print("1. load data")
train_x, train_y, test_x, test_y, val_x, val_y = load_data(train_dir, test_dir)

print("2. pre processing")
train_x, val_x, test_x = train_x.tolist(), val_x.tolist(), test_x.tolist()

train_x = [' '.join(t.split()[0:max_len]) for t in train_x]
train_x = np.array(train_x, dtype=object)[:, np.newaxis]

val_x = [' '.join(t.split()[0:max_len]) for t in val_x]
val_x = np.array(val_x, dtype=object)[:, np.newaxis]

test_x = [' '.join(t.split()[0:max_len]) for t in test_x]
test_x = np.array(test_x, dtype=object)[:, np.newaxis]

print("3. build model")
model = ELMo(
    hidden_units=hidden_units,
    data_type="binary"
)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

callbacks = create_callbacks(model_dir)
model.fit(x=train_x, y=train_y, epochs=epoch, batch_size=batch, validation_data=(val_x, val_y), callbacks=callbacks)


# BentoML Serve
from service import KerasClassification
bento_svc = KerasClassification()
bento_svc.pack('model', model)

saved_path = bento_svc.save()