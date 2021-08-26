from typing import List

from bentoml import api, env, BentoService, artifacts
from bentoml.frameworks.keras import KerasModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
from bentoml.adapters import JsonInput
from bentoml.types import JsonSerializable
from keras.preprocessing import text, sequence


max_features = 1000

@artifacts([KerasModelArtifact('model'), PickleArtifact('tokenizer')])
@env(pip_packages=['tensorflow==1.15.0', 'numpy', 'pandas'])
class KerasTextClassificationService(BentoService):
    @api(input=JsonInput(), batch=True)
    def predict(self, parsed_jsons: List[JsonSerializable]):
        input_datas = [parsed_json['text'] for parsed_json in parsed_jsons]
        input_text = input_datas[0]
        input_text = self.artifacts.tokenizer.texts_to_sequences(input_text)
        input_text = sequence.pad_sequences(input_text, maxlen=80, padding='post')
        
        return self.artifacts.model.predict(input_text).T[0]