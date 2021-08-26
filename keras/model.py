from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.preprocessing import text, sequence


max_features = 1000
maxlen = 80 # cut texts after this number of words (among top max_features most common words)
batch_size = 256

train = pd.read_csv("binary_train.csv")
test = pd.read_csv("binary_test.csv")

train, val = train_test_split(train, test_size=0.1, random_state=42)

train_x, train_y = train["text"], train["label"]
test_x, test_y = test["text"], test["label"]
val_x, val_y = val["text"], val["label"]

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)
val_y = encoder.fit_transform(val_y)

train_x = train_x.tolist()
test_x = test_x.tolist()
val_x = val_x.tolist()

CHARS_TO_REMOVE = r'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)
tokenizer.fit_on_texts(train_x + test_x + val_x)  # Make dictionary

# Text match to dictionary.
train_x = tokenizer.texts_to_sequences(train_x)
test_x = tokenizer.texts_to_sequences(test_x)
val_x = tokenizer.texts_to_sequences(val_x)

total_list = list(train_x) + list(test_x) + list(val_x)
max_len = max([len(total_list[i]) for i in range(0, len(total_list))])

train_x = sequence.pad_sequences(train_x, maxlen=max_len, padding='post')
test_x = sequence.pad_sequences(test_x, maxlen=max_len, padding='post')
val_x = sequence.pad_sequences(val_x, maxlen=max_len, padding='post')

model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=3,
          validation_data=(val_x, val_y))

score, acc = model.evaluate(test_x, test_y, batch_size=batch_size)
print('Test score :', score)
print('Test accuracy :', acc)