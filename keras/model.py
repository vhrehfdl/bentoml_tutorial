from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.datasets import imdb

max_features = 1000
maxlen = 80 # cut texts after this number of words (among top max_features most common words)
batch_size = 300
index_from=3 # word index offset

# A dictionary mapping words to an integer index
imdb.load_data(num_words=max_features)
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+index_from) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown

# Use decode_review to look at original review text in training/testing data
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_review(encoded_text):
    return ' '.join([reverse_word_index.get(i, '?') for i in encoded_text])

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features, index_from=index_from)

x_train = sequence.pad_sequences(x_train,
                                 value=word_index["<PAD>"],
                                 padding='post',
                                 maxlen=maxlen)

x_test = sequence.pad_sequences(x_test,
                                value=word_index["<PAD>"],
                                padding='post',
                                maxlen=maxlen)

model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=3, # for demo purpose :P
          validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)