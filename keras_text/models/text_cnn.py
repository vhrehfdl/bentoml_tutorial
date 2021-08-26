from keras.layers import Embedding, Dense, Flatten, Input, concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras.models import Model


def TextCNN(sequence_len, embedding_matrix, embedding_dim, filter_sizes, flag, data_type, category_num=None):
    # Input Layer
    input_layer = Input(shape=(sequence_len,))

    # Embedding Layer
    if flag == "self_training":
        embedding_layer = Embedding(embedding_matrix, embedding_dim)(input_layer)
    elif flag == "pre_training":
        embedding_layer = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(input_layer)

    # Hideen Layer
    pooled_outputs = []
    for filter_size in filter_sizes:
        x = Conv1D(embedding_dim, filter_size, activation='relu')(embedding_layer)
        x = MaxPool1D(pool_size=2)(x)
        pooled_outputs.append(x)

    merged = concatenate(pooled_outputs, axis=1)
    dense_layer = Flatten()(merged)

    # Output Layer
    if data_type == "binary":
        output_layer = Dense(1, activation='sigmoid')(dense_layer)
    elif data_type == "multi":
        output_layer = Dense(category_num, activation='softmax')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

