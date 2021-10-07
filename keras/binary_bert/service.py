import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from typing import List

from bentoml import api, env, BentoService, artifacts
from bentoml.frameworks.keras import KerasModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
from bentoml.adapters import JsonInput
from bentoml.types import JsonSerializable

from models.bert import BERT
from utils.bert_helper import convert_examples_to_features, convert_text_to_examples

@artifacts([KerasModelArtifact('model'), PickleArtifact('tokenizer')])
@env(pip_packages=['tensorflow==1.15.0', 'numpy', 'pandas'])
class KerasClassification(BentoService):
    @api(input=JsonInput(), batch=True)
    def predict(self, parsed_jsons: List[JsonSerializable]):
        input_texts = [parsed_json['text'] for parsed_json in parsed_jsons]

        input_examples = convert_text_to_examples(input_texts, "null")
        input_ids, input_masks, segment_ids, labels = convert_examples_to_features(self.artifacts.tokenizer, input_examples, 50)

        predictions = self.artifacts.model.predict([input_ids, input_masks, segment_ids])
        y_pred = (predictions > 0.5)
        
        return y_pred