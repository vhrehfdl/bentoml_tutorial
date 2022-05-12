import torch

from typing import List
from bentoml import env, artifacts, api, BentoService
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
from bentoml.adapters import JsonInput
from bentoml.types import JsonSerializable
from BERT import BERTDataset, BERTClassifier


@artifacts([PytorchModelArtifact('model'), PickleArtifact('tokenizer'), PickleArtifact('max_len'), 
            PickleArtifact('BERTDataset'), PickleArtifact('label_enc')])
class ReviewClassifier(BentoService):
    @api(input=JsonInput(), batch=True)
    def predict(self, parsed_jsons: List[JsonSerializable]):
        max_len = self.artifacts.max_len
        device = torch.device("cuda")

        response_list = []
        for parsed_json in parsed_jsons:
            input_texts = parsed_json['text']
        
            test_dataset = self.artifacts.BERTDataset(input_texts, [0]*len(input_texts), self.artifacts.tokenizer, max_len, True, False)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

            result = []
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
                output = self.artifacts.model(token_ids.long().to(device), valid_length, segment_ids.long().to(device))
                _, output = torch.max(output, 1)
                label = self.artifacts.label_enc.inverse_transform(output.tolist())
                result.append(label)

            response_list.append(result)

        return response_list
