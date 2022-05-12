from typing import List

from bentoml import artifacts, api, BentoService
from bentoml.adapters import JsonInput
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
from bentoml.types import JsonSerializable


@artifacts([SklearnModelArtifact('model'), PickleArtifact('tokenizer'), PickleArtifact('pos_tagging')])
class ReviewClassifier(BentoService):
    @api(input=JsonInput(), batch=True)
    def predict(self, parsed_jsons: List[JsonSerializable]):

        response_list = []
        for parsed_json in parsed_jsons:
            input_texts = parsed_json['text']

            text = self.artifacts.pos_tagging(input_texts)
            text = self.artifacts.tokenizer.transform(text)
            pred_y = self.artifacts.model.predict(text)

            response_list.append(pred_y)

        return [pred_y]