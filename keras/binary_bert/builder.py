
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from bert.tokenization import FullTokenizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

from models.bert import BERT
from utils import set_env
from utils.bert_helper import convert_examples_to_features, convert_text_to_examples
from utils.evaluation import Evaluation


sess = tf.Session()


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


def create_tokenizer_from_hub_module():
    bert_module = hub.Module("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1")
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
model_dir = "../model_save"

# HyperParameter
max_len = 50
epoch = 2
batch = 512

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

tokenizer = create_tokenizer_from_hub_module()

train_examples = convert_text_to_examples(train_x, train_y)
val_examples = convert_text_to_examples(val_x, val_y)
test_examples = convert_text_to_examples(test_x, test_y)

train_input_ids, train_input_masks, train_segment_ids, train_labels = convert_examples_to_features(tokenizer, train_examples, max_len)
val_input_ids, val_input_masks, val_segment_ids, val_labels = convert_examples_to_features(tokenizer, val_examples, max_len)
test_input_ids, test_input_masks, test_segment_ids, test_labels = convert_examples_to_features(tokenizer, test_examples, max_len)

print("3. build model")
model = BERT(max_len, data_type="binary")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
initialize_vars(sess)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_dir + "/model-weights.{epoch:02d}-{val_acc:.6f}.hdf5", monitor='val_acc', save_best_only=True, verbose=1)
model.fit(
    [train_input_ids, train_input_masks, train_segment_ids], train_labels,
    validation_data=([val_input_ids, val_input_masks, val_segment_ids], val_labels),
    epochs=epoch,
    batch_size=batch,
    callbacks=[cp_callback]
)

print("4. evaluation")
evaluation = Evaluation(model, [test_input_ids, test_input_masks, test_segment_ids], test_y)
accuracy, cf_matrix, report = evaluation.eval_classification_bert(data_type="binary")
print("## Classification Report \n", report)
print("## Confusion Matrix \n", cf_matrix)
print("## Accuracy \n", accuracy)



# BentoML Serve
from service import KerasClassification
bento_svc = KerasClassification()
bento_svc.pack('model', model)

saved_path = bento_svc.save()