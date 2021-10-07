import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras.backend as K


class ElmoEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module(
            'https://tfhub.dev/google/elmo/2', 
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )
        if self.trainable:
            self._trainable_weights.extend(
                tf.trainable_variables(scope="^{}_module/.*".format(self.name))
            )
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(
            K.squeeze(K.cast(x, tf.string), axis=1),
            as_dict=True,
            signature='default',
        )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)


def ELMo(hidden_units, data_type, category_size=None, train_elmo=False):
    if data_type == "binary":
        output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    elif data_type == "multi":
        output_layer = tf.keras.layers.Dense(category_size, activation='softmax')

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(dtype='string', input_shape=(1,)),
        ElmoEmbeddingLayer(trainable=train_elmo),
        tf.keras.layers.Dense(hidden_units),
        output_layer
    ])

    sess = K.get_session()
    init = tf.global_variables_initializer()
    sess.run(init)

    return model