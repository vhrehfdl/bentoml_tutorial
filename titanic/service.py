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
        input_data = parsed_jsons[0]['data']
        print(input_data)

        pred_y = self.artifacts.model.predict(input_data)
        return [pred_y]