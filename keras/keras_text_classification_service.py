from typing import List

import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence, text
from bentoml import api, env, BentoService, artifacts
from bentoml.frameworks.keras import KerasModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
from bentoml.adapters import JsonInput
from bentoml.types import JsonSerializable


max_features = 1000

@artifacts([
    KerasModelArtifact('model'),
    PickleArtifact('word_index')
])
@env(pip_packages=['tensorflow==1.14.0', 'numpy', 'pandas'])
class KerasTextClassificationService(BentoService):
    def word_to_index(self, word):
        if word in self.artifacts.word_index and self.artifacts.word_index[word] <= max_features:
            return self.artifacts.word_index[word]
        else:
            return self.artifacts.word_index["<UNK>"]
    
    def preprocessing(self, text_str):
        sequence = text.text_to_word_sequence(text_str)
        return list(map(self.word_to_index, sequence))
    
    @api(input=JsonInput(), batch=True)
    def predict(self, parsed_jsons: List[JsonSerializable]):
        input_datas = [self.preprocessing(parsed_json['text']) for parsed_json in parsed_jsons]
        input_datas = sequence.pad_sequences(input_datas,
                                             value=self.artifacts.word_index["<PAD>"],
                                             padding='post',
                                             maxlen=80)

        return self.artifacts.model.predict_classes(input_datas).T[0]