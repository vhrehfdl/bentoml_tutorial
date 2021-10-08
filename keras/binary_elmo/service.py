from typing import List

from bentoml import api, env, BentoService, artifacts
from bentoml.frameworks.keras import KerasModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
from bentoml.adapters import JsonInput
from bentoml.types import JsonSerializable


@artifacts([KerasModelArtifact('model')])
@env(pip_packages=['tensorflow==1.15.0', 'numpy', 'pandas'])
class KerasClassification(BentoService):
    @api(input=JsonInput(), batch=True)
    def predict(self, parsed_jsons: List[JsonSerializable]):
        maxlen = 50

        input_texts = [parsed_json['text'] for parsed_json in parsed_jsons]

        # input_text = self.artifacts.tokenizer.texts_to_sequences(input_texts)
        # input_text = sequence.pad_sequences(input_text, maxlen=300, padding='post')

        input_text = ' '.join(input_texts.split()[0:maxlen])
        print(input_text)

        predictions = self.artifacts.model.predict(input_text)
        y_pred = (predictions > 0.5)
        
        return y_pred