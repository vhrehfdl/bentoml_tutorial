import keras.backend as K
import keras.layers as layers
import tensorflow as tf
import tensorflow_hub as hub
from keras.engine import Layer
from keras.models import Model


class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/3', trainable=self.trainable, name="{}_module".format(self.name))
        self.trainable_weights += tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1), as_dict=True, signature='default')['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)


def ELMo(hidden_units, data_type, category_size=None):
    # Input Layer
    input_layer = layers.Input(shape=(1,), dtype="string")

    # Embedding Layer
    embedding_layer = ElmoEmbeddingLayer()(input_layer)
    dense_elmo = layers.Dense(hidden_units, activation='relu')(embedding_layer)

    # Output Layer
    if data_type == "binary":
        output_layer = layers.Dense(1, activation='sigmoid')(dense_elmo)
    elif data_type == "multi":
        output_layer = layers.Dense(category_size, activation='softmax')(dense_elmo)

    model = Model(inputs=[input_layer], outputs=output_layer)

    return model