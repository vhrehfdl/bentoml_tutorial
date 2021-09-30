
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from service import KerasTextClassificationService

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from bert.tokenization import FullTokenizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

from models.bert import BERT
from keras.models import load_model, save_model


sess = tf.Session()


def load_data(train_dir, test_dir):
    train = pd.read_csv(train_dir)
    test = pd.read_csv(test_dir)

    train, val = train_test_split(train, test_size=0.1, random_state=42)

    train_x, train_y = train["text"], train["label"]
    test_x, test_y = test["text"], test["label"]
    val_x, val_y = val["text"], val["label"]

    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    test_y = encoder.fit_transform(test_y)
    val_y = encoder.fit_transform(val_y)

    return train_x, train_y, test_x, test_y, val_x, val_y


def create_tokenizer_from_hub_module():
    bert_module = hub.Module("http://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1")
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run([
        tokenization_info["vocab_file"],
        tokenization_info["do_lower_case"],
    ])

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)


# Directory Setting
train_dir = "../data/binary_train.csv"
test_dir = "../data/binary_test.csv"
model_dir = "../model_save/"

# HyperParameter
max_len = 50
epoch = 2
batch = 512
hidden_units = 256

# Flow
tokenizer = create_tokenizer_from_hub_module()

model = BERT(max_len, data_type="binary")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
initialize_vars(sess)

model.load_model(model_dir, custom_objects=None, compile=True)


bento_svc = KerasClassification()
bento_svc.pack('model', model)
bento_svc.pack('tokenizer', tokenizer)

saved_path = bento_svc.save()