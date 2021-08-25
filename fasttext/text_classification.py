from bentoml import env, artifacts, BentoService, api
from bentoml.frameworks.fasttext import FasttextModelArtifact
from bentoml.adapters import JsonInput


@env(infer_pip_packages=True)
@artifacts([FasttextModelArtifact('model')])
class FasttextClassification(BentoService):
    
    @api(input=JsonInput(), batch=True)
    def predict(self, json_list):
        input = [i['text'] for i in json_list]
        result = self.artifacts.model.predict(input)
        # return top result
        prediction_result = [i[0].replace('__label__', '') for i in result[0]]
        return prediction_result