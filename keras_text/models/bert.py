import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import backend as K


class BertLayer(tf.keras.layers.Layer):
    def __init__(self, n_fine_tune_layers=10, **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        super(BertLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        return config

    def build(self, input_shape):
        self.bert = hub.Module(
            "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )

        trainable_vars = self.bert.variables

        # Remove unused layers
        trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]

        # Select how many layers to fine tune
        trainable_vars = trainable_vars[-self.n_fine_tune_layers:]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
            "pooled_output"
        ]
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)


def BERT(sequence_len, data_type, category_size=None):
    in_id = tf.keras.layers.Input(shape=(sequence_len,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(sequence_len,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(sequence_len,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    bert_output = BertLayer(n_fine_tune_layers=3)(bert_inputs)
    dense = tf.keras.layers.Dense(256, activation='relu')(bert_output)

    if data_type == "binary":
        output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
    elif data_type == "multi":
        output_layer = tf.keras.layers.Dense(category_size, activation='softmax')(dense)

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=output_layer)

    return model
