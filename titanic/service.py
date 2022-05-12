import pandas as pd

from typing import List
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.adapters import JsonInput
from bentoml.types import JsonSerializable

@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('model')])
class Titanic(BentoService):
    @api(input=JsonInput(), batch=True)
    def predict(self, parsed_jsons: List[JsonSerializable]):
        response_list = []
        for parsed_json in parsed_jsons:
            input_data = parsed_json['data']
            pred_y = self.artifacts.model.predict(input_data)
            response_list.append(pred_y)
            
        return response_list