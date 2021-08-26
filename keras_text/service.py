from typing import List

from bentoml import api, env, BentoService, artifacts
from bentoml.frameworks.keras import KerasModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
from bentoml.adapters import JsonInput
from bentoml.types import JsonSerializable
from keras.preprocessing import sequence

@artifacts([KerasModelArtifact('model'), PickleArtifact('tokenizer')])
@env(pip_packages=['tensorflow==1.15.0', 'numpy', 'pandas'])
class KerasTextClassificationService(BentoService):
    @api(input=JsonInput(), batch=True)
    def predict(self, parsed_jsons: List[JsonSerializable]):
        input_texts = [parsed_json['text'] for parsed_json in parsed_jsons]
        print(input_texts)

        input_text = self.artifacts.tokenizer.texts_to_sequences(input_texts)
        input_text = sequence.pad_sequences(input_text, maxlen=300, padding='post')
        predictions = self.artifacts.model.predict(input_text)
        y_pred = (predictions > 0.5)
        
        return y_pred