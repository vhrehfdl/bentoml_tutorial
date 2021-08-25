from typing import List

from bentoml import env, artifacts, api, BentoService
from bentoml.frameworks.pytorch import PytorchModelArtifact

from bentoml.adapters import JsonInput
from bentoml.types import JsonSerializable


@env(pip_packages=['torch', 'numpy', 'torchtext', 'scikit-learn'])
@artifacts([PytorchModelArtifact('net')])
class PytorchTextClassifier(BentoService):
    
    @api(input=JsonInput(), batch=True)
    def predict(self, parsed_jsons: List[JsonSerializable]):
        input_datas = parsed_jsons[0]["text"]

        logit = self.artifacts.net(input_datas)
        print(logit)
        
        return self.artifacts.net.predict(input_datas)

            
    # @api(input=JsonInput(), batch=True)
    # def predict(self, parsed_jsons: List[JsonSerializable]):
    #     input_datas = [self.preprocessing(parsed_json['text']) for parsed_json in parsed_jsons]
    #     input_datas = sequence.pad_sequences(input_datas,
    #                                          value=self.artifacts.word_index["<PAD>"],
    #                                          padding='post',
    #                                          maxlen=80)

    #     return self.artifacts.model.predict_classes(input_datas).T[0]